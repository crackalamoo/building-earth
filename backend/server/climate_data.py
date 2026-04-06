"""Load simulation output and sample individual fields at (lat, lon, month)."""

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

import xarray as xr

# ── Field metadata: description + units ──────────────────────────────────────

FIELD_INFO: dict[str, dict[str, str]] = {
    # ── Common ──
    "temperature_2m": {"label": "Air temperature", "desc": "2-meter air temperature", "unit": "°C"},
    "precipitation": {"label": "Precipitation", "desc": "Precipitation rate", "unit": "kg/m²/s"},
    "humidity": {"label": "Humidity", "desc": "Specific humidity", "unit": "kg/kg"},
    "wind_speed_10m": {"label": "Surface wind speed", "desc": "10-m wind speed", "unit": "m/s"},
    "cloud_fraction": {
        "label": "Cloud cover",
        "desc": "Total cloud cover",
        "unit": "fraction (0-1)",
    },
    "surface_pressure": {"label": "Air pressure", "desc": "Surface pressure", "unit": "Pa"},
    "elevation": {"label": "Elevation", "desc": "Elevation (negative = ocean depth)", "unit": "m"},
    # ── Temperature layers ──
    "surface": {"label": "Surface temperature", "desc": "Surface / SST temperature", "unit": "°C"},
    "boundary_layer": {
        "label": "Lower atmosphere temp",
        "desc": "Boundary-layer temperature",
        "unit": "°C",
    },
    "atmosphere": {
        "label": "Upper atmosphere temp",
        "desc": "Free-atmosphere temperature",
        "unit": "°C",
    },
    # ── Wind detail ──
    "wind_u_10m": {"label": "Surface wind E/W", "desc": "10-m eastward wind", "unit": "m/s"},
    "wind_v_10m": {"label": "Surface wind N/S", "desc": "10-m northward wind", "unit": "m/s"},
    "wind_u": {
        "label": "Low-level wind E/W",
        "desc": "Boundary-layer eastward wind",
        "unit": "m/s",
    },
    "wind_v": {
        "label": "Low-level wind N/S",
        "desc": "Boundary-layer northward wind",
        "unit": "m/s",
    },
    "wind_speed": {
        "label": "Low-level wind speed",
        "desc": "Boundary-layer wind speed",
        "unit": "m/s",
    },
    "wind_u_geostrophic": {
        "label": "Pressure-driven wind E/W",
        "desc": "Geostrophic eastward wind",
        "unit": "m/s",
    },
    "wind_v_geostrophic": {
        "label": "Pressure-driven wind N/S",
        "desc": "Geostrophic northward wind",
        "unit": "m/s",
    },
    "wind_speed_geostrophic": {
        "label": "Pressure-driven wind speed",
        "desc": "Geostrophic wind speed",
        "unit": "m/s",
    },
    # ── Derived fields ──
    "relative_humidity": {"label": "Relative humidity", "desc": "Relative humidity", "unit": "%"},
    "saturation_humidity": {
        "label": "Saturation humidity",
        "desc": "Saturation specific humidity",
        "unit": "kg/kg",
    },
    "wind_direction_10m": {
        "label": "Wind direction",
        "desc": "10-m wind direction (compass bearing, 0°=N)",
        "unit": "°",
    },
    "dew_point": {"label": "Dew point", "desc": "Dew point temperature", "unit": "°C"},
    "lapse_rate": {
        "label": "Lapse rate",
        "desc": "Temperature lapse rate (BL to free atmosphere)",
        "unit": "°C/km",
    },
    # ── Cloud breakdown ──
    "cloud_high": {"label": "High clouds", "desc": "High cloud cover", "unit": "fraction (0-1)"},
    "cloud_low": {"label": "Low clouds", "desc": "Low cloud cover", "unit": "fraction (0-1)"},
    "cloud_convective": {
        "label": "Storm clouds",
        "desc": "Convective cloud cover",
        "unit": "fraction (0-1)",
    },
    "stratiform_cloud_frac": {
        "label": "Layer clouds",
        "desc": "Stratiform cloud fraction",
        "unit": "fraction (0-1)",
    },
    "marine_sc_cloud_frac": {
        "label": "Ocean low clouds",
        "desc": "Marine stratocumulus cloud fraction",
        "unit": "fraction (0-1)",
    },
    # ── Ocean & vertical ──
    "ocean_u": {
        "label": "Ocean current E/W",
        "desc": "Ocean surface eastward current",
        "unit": "m/s",
    },
    "ocean_v": {
        "label": "Ocean current N/S",
        "desc": "Ocean surface northward current",
        "unit": "m/s",
    },
    "w_ekman_pumping": {
        "label": "Upwelling",
        "desc": "Ekman pumping velocity (positive = upwelling)",
        "unit": "m/s",
    },
    "vertical_velocity": {
        "label": "Rising/sinking air",
        "desc": "Vertical velocity (positive = rising)",
        "unit": "m/s",
    },
    # ── Ocean temperature ──
    "sst": {
        "label": "Water temperature",
        "desc": "Sea surface temperature",
        "unit": "°C",
    },
    # ── Surface properties ──
    "albedo": {"label": "Albedo", "desc": "Surface albedo", "unit": "fraction (0-1)"},
    "soil_moisture": {"label": "Soil moisture", "desc": "Soil moisture", "unit": "fraction (0-1)"},
    "vegetation_fraction": {
        "label": "Vegetation",
        "desc": "Vegetation fraction",
        "unit": "fraction (0-1)",
    },
}

# Aliases: npz stores these under different names than the frontend binary
_FIELD_ALIASES = {"high_cloud_frac": "cloud_high", "convective_cloud_frac": "cloud_convective"}


def _to_display_units(value: float, raw_unit: str, imperial: bool) -> tuple[float, str]:
    """Convert raw simulation units to human-friendly display units."""
    if raw_unit == "°C":
        if imperial:
            return value * 9 / 5 + 32, "°F"
        return value, "°C"

    if raw_unit == "kg/m²/s":
        # Precipitation: convert to mm/month or in/month (more intuitive)
        mm_month = value * 86400 * 30
        if imperial:
            return mm_month / 25.4, "in/month"
        return mm_month, "mm/month"

    if raw_unit == "Pa":
        if imperial:
            return value / 3386.39, "inHg"
        return value / 100, "mb"

    if raw_unit == "m/s":
        if imperial:
            return value * 2.23694, "mph"
        return value * 3.6, "km/h"

    if raw_unit == "m":
        if imperial:
            return value * 3.28084, "ft"
        return value, "m"

    if raw_unit == "kg/kg":
        return value * 1000, "g/kg"

    if raw_unit == "fraction (0-1)":
        return value * 100, "%"

    if raw_unit == "%":
        return value, "%"

    if raw_unit == "°":
        return value, "°"

    return value, raw_unit


def _decode_float16(buf: bytes, count: int) -> np.ndarray:
    """Decode raw bytes as float16 → float32."""
    return np.frombuffer(buf, dtype=np.float16, count=count).astype(np.float32)


def _decode_uint8_temperature(buf: bytes, count: int) -> np.ndarray:
    """Decode uint8-quantized temperature: val * (120/255) - 60."""
    raw = np.frombuffer(buf, dtype=np.uint8, count=count).astype(np.float32)
    return raw * (120.0 / 255.0) - 60.0


# Fields that should NOT be exposed to the LLM (masks, snow_temperature internal)
_SKIP_FIELDS = {"land_mask", "land_mask_native", "land_mask_1deg", "snow_temperature"}


class ClimateDataStore:
    """Loads frontend binary (interpolated, high-res) with npz fallback."""

    def __init__(
        self,
        npz_path: str | Path = "data/main.npz",
        bin_path: str | Path = "frontend/public/main.bin.gz",
        manifest_path: str | Path = "frontend/public/main.manifest.json",
    ) -> None:
        self._data: dict[str, np.ndarray] = {}
        self._load_binary(Path(bin_path), Path(manifest_path))
        self._load_npz_fallback(Path(npz_path))
        self._compute_derived()

    def _load_binary(self, bin_path: Path, manifest_path: Path) -> None:
        """Load interpolated fields from the frontend binary export."""
        if not bin_path.exists() or not manifest_path.exists():
            return
        with open(manifest_path) as f:
            manifest = json.load(f)
        if bin_path.suffix == '.gz':
            with gzip.open(bin_path, 'rb') as f:
                raw = f.read()
        else:
            raw = bin_path.read_bytes()
        for field in manifest["fields"]:
            name = field["name"]
            if name in _SKIP_FIELDS:
                continue
            shape = tuple(field["shape"])
            offset = field["offset"]
            count = int(np.prod(shape))
            buf = raw[offset : offset + field["bytes"]]
            if field["dtype"] == "uint8" and name == "temperature_2m":
                arr = _decode_uint8_temperature(buf, count).reshape(shape)
            elif field["dtype"] == "float16":
                arr = _decode_float16(buf, count).reshape(shape)
            else:
                continue  # skip other uint8 fields
            self._data[name] = arr

    def _load_npz_fallback(self, path: Path) -> None:
        """Load fields from main.npz that aren't already loaded from the binary."""
        if not path.exists():
            return
        raw = np.load(path)
        for key in raw.files:
            # Apply aliases (e.g. high_cloud_frac → cloud_high)
            canonical = _FIELD_ALIASES.get(key, key)
            if canonical not in self._data and canonical not in _SKIP_FIELDS:
                self._data[canonical] = raw[key]
        # Keep native-resolution humidity for derived field computation even if
        # the binary already loaded an interpolated version
        if "humidity" in raw and "humidity" in self._data:
            native = raw["humidity"]
            if native.shape != self._data["humidity"].shape:
                self._native_humidity = native

    def _compute_derived(self) -> None:
        """Compute derived fields on the native 36x72 grid."""
        # Use native-resolution humidity (36x72) if the store has an
        # interpolated version from the binary export
        q = getattr(self, "_native_humidity", None)
        if q is None:
            q = self._data.get("humidity")

        # Saturation humidity & RH & dew point — all need BL temp + pressure + q
        t_bl = self._data.get("boundary_layer")  # °C, 36x72
        p = self._data.get("surface_pressure")  # Pa, 36x72
        if t_bl is not None:
            t_c = np.clip(t_bl, -100.0, 80.0)
            e_sat_hPa = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
            p_hPa = p / 100.0 if p is not None else 1013.25
            denom = np.maximum(p_hPa - (1 - 0.622) * e_sat_hPa, 1.0)
            q_sat = (0.622 * e_sat_hPa) / denom
            self._data["saturation_humidity"] = q_sat

            if q is not None:
                rh = np.clip(q / np.maximum(q_sat, 1e-10) * 100, 0, 100)
                self._data["relative_humidity"] = rh

                # Dew point: inverse Magnus from actual vapor pressure
                e_hPa = q * p_hPa / (0.622 + 0.378 * q)
                e_hPa = np.maximum(e_hPa, 1e-6)
                ln_ratio = np.log(e_hPa / 6.112)
                td = 243.5 * ln_ratio / (17.67 - ln_ratio)
                self._data["dew_point"] = td

        # Wind direction (compass bearing) — 36x72
        u = self._data.get("wind_u_10m")
        v = self._data.get("wind_v_10m")
        if u is not None and v is not None:
            direction = (270 - np.degrees(np.arctan2(v, u))) % 360
            self._data["wind_direction_10m"] = direction

        # Lapse rate (BL to free atmosphere, ~7.5 km separation) — 36x72
        t_atm = self._data.get("atmosphere")
        if t_bl is not None and t_atm is not None:
            self._data["lapse_rate"] = (t_bl - t_atm) / 7.5

    @property
    def available_fields(self) -> list[str]:
        return [f for f in FIELD_INFO if f in self._data]

    def sample_raw(
        self,
        field: str,
        lat: float,
        lon: float,
        month: int,
    ) -> float | None:
        """Sample a single field and return the raw float value (no unit conversion).

        Returns None if field is unknown. Raw units: °C, kg/kg, Pa, m/s,
        fraction (0-1), %, °.
        """
        if field not in self._data:
            return None

        arr = self._data[field]

        if arr.ndim == 2:
            nlat, nlon = arr.shape
            has_month = False
        elif arr.ndim == 3:
            _, nlat, nlon = arr.shape
            has_month = True
        else:
            return None

        lat_idx = int(np.clip(round((lat + 90) / 180 * nlat - 0.5), 0, nlat - 1))
        lon_norm = ((lon % 360) + 360) % 360
        lon_idx = int(np.floor(lon_norm / 360 * nlon)) % nlon
        month_idx = int(month) % 12

        if has_month:
            return float(arr[month_idx, lat_idx, lon_idx])
        return float(arr[lat_idx, lon_idx])

    def sample(
        self,
        field: str,
        lat: float,
        lon: float,
        month: int,
        *,
        imperial: bool = False,
    ) -> dict[str, Any]:
        """Sample a single field value at (lat, lon, month).

        Returns dict with keys: field, value, unit, description.
        When imperial=True, converts to °F / mph / inches / ft etc.
        """
        # SST: route to surface field, but only for ocean cells
        display_field = field  # field name to use for label/desc in response
        if field == "sst":
            land_mask = self._data.get("land_mask_native") or self._data.get("land_mask")
            if land_mask is not None:
                lm_nlat, lm_nlon = land_mask.shape
                lm_lat_idx = int(np.clip(round((lat + 90) / 180 * lm_nlat - 0.5), 0, lm_nlat - 1))
                lm_lon_norm = ((lon % 360) + 360) % 360
                lm_lon_idx = int(np.floor(lm_lon_norm / 360 * lm_nlon)) % lm_nlon
                if land_mask[lm_lat_idx, lm_lon_idx] != 0:
                    return {"field": "sst", "error": "This location is on land — SST is only available over ocean."}
            field = "surface"

        if field not in self._data:
            return {"field": display_field, "error": f"Unknown field '{display_field}'"}

        arr = self._data[field]
        info = FIELD_INFO.get(display_field, {"desc": display_field, "unit": "unknown"})

        # Determine grid dimensions – arrays are either (nlat, nlon) or (12, nlat, nlon)
        if arr.ndim == 2:
            nlat, nlon = arr.shape
            has_month = False
        elif arr.ndim == 3:
            _, nlat, nlon = arr.shape
            has_month = True
        else:
            return {"field": field, "error": f"Unexpected shape {arr.shape}"}

        # Map lat/lon to indices
        lat_idx = int(np.clip(round((lat + 90) / 180 * nlat - 0.5), 0, nlat - 1))
        lon_norm = ((lon % 360) + 360) % 360
        lon_idx = int(np.floor(lon_norm / 360 * nlon)) % nlon

        month_idx = int(month) % 12

        if has_month:
            value = float(arr[month_idx, lat_idx, lon_idx])
        else:
            value = float(arr[lat_idx, lon_idx])

        raw_unit = info["unit"]
        value, unit = _to_display_units(value, raw_unit, imperial)

        return {
            "field": field,
            "value": round(value, 2),
            "unit": unit,
            "description": info["desc"],
        }

    def sample_many(
        self,
        fields: list[str],
        lat: float,
        lon: float,
        month: int,
        *,
        imperial: bool = False,
    ) -> list[dict[str, Any]]:
        """Sample multiple fields at the same location and month."""
        return [self.sample(f, lat, lon, month, imperial=imperial) for f in fields]


# ── Observational data (NOAA 1981-2010 climatology) ─────────────────────────

# Maps tool field name → (NetCDF file key, NetCDF variable, label, raw unit, source file)
_CLIM_FILE = "ref_climatology_1deg_1981-2010.nc"
_HP_FILE = "ref_humidity_precip_1deg_1981-2010.nc"

OBS_FIELD_INFO: dict[str, dict[str, str]] = {
    "land_temperature": {
        "var": "t_land_clim",
        "file": _CLIM_FILE,
        "label": "Observed land temp",
        "desc": "Observed 2-meter air temperature over land (GHCN_CAMS stations, land only)",
        "unit": "°C",
    },
    "sst": {
        "var": "t_sst_clim",
        "file": _CLIM_FILE,
        "label": "Observed SST",
        "desc": "Observed sea surface temperature (COBE2 satellite, ocean only)",
        "unit": "°C",
    },
    "humidity": {
        "var": "shum_clim",
        "file": _HP_FILE,
        "label": "Observed humidity",
        "desc": "Observed specific humidity (NCEP reanalysis)",
        "unit": "kg/kg",
    },
    "precipitation": {
        "var": "precip_clim",
        "file": _HP_FILE,
        "label": "Observed precip",
        "desc": "Observed precipitation (GPCP satellite+gauge)",
        "unit": "kg/m²/s",  # stored as mm/day, converted on load
    },
    "pressure": {
        "var": "slp_clim",
        "file": _HP_FILE,
        "label": "Observed pressure",
        "desc": "Observed sea-level pressure (NCEP reanalysis)",
        "unit": "Pa",
    },
    "wind_u": {
        "var": "uwnd_clim",
        "file": _HP_FILE,
        "label": "Observed wind E/W",
        "desc": "Observed eastward wind (NCEP reanalysis)",
        "unit": "m/s",
    },
    "wind_v": {
        "var": "vwnd_clim",
        "file": _HP_FILE,
        "label": "Observed wind N/S",
        "desc": "Observed northward wind (NCEP reanalysis)",
        "unit": "m/s",
    },
}


class ObsDataStore:
    """Loads NOAA 1981-2010 climatology at 1° resolution."""

    def __init__(self, data_dir: str | Path = "data/processed") -> None:
        self._data: dict[str, np.ndarray] = {}
        data_dir = Path(data_dir)
        self._load(data_dir)

    def _load(self, data_dir: Path) -> None:
        # Group fields by source file
        by_file: dict[str, list[str]] = {}
        for field, info in OBS_FIELD_INFO.items():
            by_file.setdefault(info["file"], []).append(field)

        for filename, fields in by_file.items():
            path = data_dir / filename
            if not path.exists():
                continue
            ds = xr.open_dataset(path)
            for field in fields:
                var = OBS_FIELD_INFO[field]["var"]
                if var not in ds:
                    continue
                arr = np.asarray(ds[var].values, dtype=np.float32)
                # Precipitation: stored as mm/day, convert to kg/m²/s
                if field == "precipitation":
                    arr = arr / 86400.0
                self._data[field] = arr
            ds.close()

    @property
    def available_fields(self) -> list[str]:
        return [f for f in OBS_FIELD_INFO if f in self._data]

    def sample(
        self,
        field: str,
        lat: float,
        lon: float,
        month: int,
        *,
        imperial: bool = False,
    ) -> dict[str, Any]:
        """Sample an observation field at (lat, lon, month)."""
        if field not in self._data:
            return {"field": field, "error": f"Unknown observation field '{field}'"}

        arr = self._data[field]
        info = OBS_FIELD_INFO[field]

        # Fixed 1° grid: (12, 180, 360)
        if arr.ndim == 3:
            _, nlat, nlon = arr.shape
        elif arr.ndim == 2:
            nlat, nlon = arr.shape
        else:
            return {"field": field, "error": f"Unexpected shape {arr.shape}"}

        lat_idx = int(np.clip(round((lat + 90) / 180 * nlat - 0.5), 0, nlat - 1))
        lon_norm = ((lon % 360) + 360) % 360
        lon_idx = int(np.floor(lon_norm / 360 * nlon)) % nlon
        month_idx = int(month) % 12

        if arr.ndim == 3:
            value = float(arr[month_idx, lat_idx, lon_idx])
        else:
            value = float(arr[lat_idx, lon_idx])

        # If NaN, search nearby cells (up to 2° away) for the closest valid value
        if np.isnan(value):
            value = self._nearest_valid(arr, lat_idx, lon_idx, month_idx)

        if np.isnan(value):
            return {
                "field": field,
                "value": None,
                "unit": info["unit"],
                "description": info["desc"],
                "note": "No observation at this location (land-only or ocean-only dataset)",
            }

        raw_unit = info["unit"]
        value, unit = _to_display_units(value, raw_unit, imperial)

        return {
            "field": field,
            "value": round(value, 2),
            "unit": unit,
            "description": info["desc"],
        }

    @staticmethod
    def _nearest_valid(
        arr: np.ndarray,
        lat_idx: int,
        lon_idx: int,
        month_idx: int,
        max_radius: int = 1,
    ) -> float:
        """Search surrounding cells for the closest non-NaN value."""
        if arr.ndim == 3:
            nlat, nlon = arr.shape[1], arr.shape[2]
        else:
            nlat, nlon = arr.shape
        for r in range(1, max_radius + 1):
            for dlat in range(-r, r + 1):
                for dlon in range(-r, r + 1):
                    if abs(dlat) != r and abs(dlon) != r:
                        continue  # only check the ring at distance r
                    li = lat_idx + dlat
                    if li < 0 or li >= nlat:
                        continue
                    lo = (lon_idx + dlon) % nlon
                    v = float(arr[month_idx, li, lo] if arr.ndim == 3 else arr[li, lo])
                    if not np.isnan(v):
                        return v
        return float("nan")

    def sample_many(
        self,
        fields: list[str],
        lat: float,
        lon: float,
        month: int,
        *,
        imperial: bool = False,
    ) -> list[dict[str, Any]]:
        """Sample multiple observation fields at the same location and month."""
        return [self.sample(f, lat, lon, month, imperial=imperial) for f in fields]
