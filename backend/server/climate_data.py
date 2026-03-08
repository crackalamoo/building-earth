"""Load simulation output and sample individual fields at (lat, lon, month)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ── Field metadata: description + units ──────────────────────────────────────

FIELD_INFO: dict[str, dict[str, str]] = {
    "temperature_2m": {"label": "Air temperature", "desc": "2-meter air temperature", "unit": "°C"},
    "surface": {"label": "Surface temperature", "desc": "Surface / SST temperature", "unit": "°C"},
    "boundary_layer": {"label": "Lower atmosphere temp", "desc": "Boundary-layer temperature", "unit": "°C"},
    "atmosphere": {"label": "Upper atmosphere temp", "desc": "Free-atmosphere temperature", "unit": "°C"},
    "albedo": {"label": "Albedo", "desc": "Surface albedo", "unit": "fraction (0-1)"},
    "humidity": {"label": "Humidity", "desc": "Specific humidity", "unit": "kg/kg"},
    "precipitation": {"label": "Precipitation", "desc": "Precipitation rate", "unit": "kg/m²/s"},
    "soil_moisture": {"label": "Soil moisture", "desc": "Soil moisture", "unit": "fraction (0-1)"},
    "wind_u_10m": {"label": "Surface wind E/W", "desc": "10-m eastward wind", "unit": "m/s"},
    "wind_v_10m": {"label": "Surface wind N/S", "desc": "10-m northward wind", "unit": "m/s"},
    "wind_speed_10m": {"label": "Surface wind speed", "desc": "10-m wind speed", "unit": "m/s"},
    "wind_u": {"label": "Low-level wind E/W", "desc": "Boundary-layer eastward wind", "unit": "m/s"},
    "wind_v": {"label": "Low-level wind N/S", "desc": "Boundary-layer northward wind", "unit": "m/s"},
    "wind_speed": {"label": "Low-level wind speed", "desc": "Boundary-layer wind speed", "unit": "m/s"},
    "wind_u_geostrophic": {"label": "Pressure-driven wind E/W", "desc": "Geostrophic eastward wind", "unit": "m/s"},
    "wind_v_geostrophic": {"label": "Pressure-driven wind N/S", "desc": "Geostrophic northward wind", "unit": "m/s"},
    "wind_speed_geostrophic": {"label": "Pressure-driven wind speed", "desc": "Geostrophic wind speed", "unit": "m/s"},
    "ocean_u": {"label": "Ocean current E/W", "desc": "Ocean surface eastward current", "unit": "m/s"},
    "ocean_v": {"label": "Ocean current N/S", "desc": "Ocean surface northward current", "unit": "m/s"},
    "w_ekman_pumping": {"label": "Upwelling", "desc": "Ekman pumping velocity (positive = upwelling)", "unit": "m/s"},
    "cloud_fraction": {"label": "Cloud cover", "desc": "Total cloud cover", "unit": "fraction (0-1)"},
    "cloud_high": {"label": "High clouds", "desc": "High cloud cover", "unit": "fraction (0-1)"},
    "cloud_low": {"label": "Low clouds", "desc": "Low cloud cover", "unit": "fraction (0-1)"},
    "cloud_convective": {"label": "Storm clouds", "desc": "Convective cloud cover", "unit": "fraction (0-1)"},
    "convective_cloud_frac": {"label": "Storm clouds", "desc": "Convective cloud fraction", "unit": "fraction (0-1)"},
    "stratiform_cloud_frac": {"label": "Layer clouds", "desc": "Stratiform cloud fraction", "unit": "fraction (0-1)"},
    "marine_sc_cloud_frac": {"label": "Ocean low clouds", "desc": "Marine stratocumulus cloud fraction", "unit": "fraction (0-1)"},
    "high_cloud_frac": {"label": "High clouds", "desc": "High cloud fraction", "unit": "fraction (0-1)"},
    "vertical_velocity": {"label": "Rising/sinking air", "desc": "Vertical velocity (positive = rising)", "unit": "m/s"},
    "surface_pressure": {"label": "Air pressure", "desc": "Surface pressure", "unit": "Pa"},
    "vegetation_fraction": {"label": "Vegetation", "desc": "Vegetation fraction", "unit": "fraction (0-1)"},
    "elevation": {"label": "Elevation", "desc": "Elevation (negative = ocean depth)", "unit": "m"},
}


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
        bin_path: str | Path = "frontend/public/main.bin",
        manifest_path: str | Path = "frontend/public/main.manifest.json",
    ) -> None:
        self._data: dict[str, np.ndarray] = {}
        self._load_binary(Path(bin_path), Path(manifest_path))
        self._load_npz_fallback(Path(npz_path))

    def _load_binary(self, bin_path: Path, manifest_path: Path) -> None:
        """Load interpolated fields from the frontend binary export."""
        if not bin_path.exists() or not manifest_path.exists():
            return
        with open(manifest_path) as f:
            manifest = json.load(f)
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
            if key not in self._data and key not in _SKIP_FIELDS:
                self._data[key] = raw[key]

    @property
    def available_fields(self) -> list[str]:
        return [f for f in FIELD_INFO if f in self._data]

    def sample(
        self, field: str, lat: float, lon: float, month: int, *, imperial: bool = False,
    ) -> dict[str, Any]:
        """Sample a single field value at (lat, lon, month).

        Returns dict with keys: field, value, unit, description.
        When imperial=True, converts to °F / mph / inches / ft etc.
        """
        if field not in self._data:
            return {"field": field, "error": f"Unknown field '{field}'"}

        arr = self._data[field]
        info = FIELD_INFO.get(field, {"desc": field, "unit": "unknown"})

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
