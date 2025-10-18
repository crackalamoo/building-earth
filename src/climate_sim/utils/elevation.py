from pathlib import Path
import os
from dotenv import load_dotenv
import urllib.request

import numpy as np
import rioxarray
from PIL import Image
from functools import lru_cache
import xarray as xr

VON_KARMAN_CONSTANT = 0.4
REFERENCE_HEIGHT_M = 10.0
WATER_NEUTRAL_DRAG_COEFFICIENT = 2.0e-4
WATER_ROUGHNESS_LENGTH_M = float(
    REFERENCE_HEIGHT_M
    / np.exp(VON_KARMAN_CONSTANT / np.sqrt(WATER_NEUTRAL_DRAG_COEFFICIENT))
)

from climate_sim.utils.constants import ATMOSPHERE_MASS, EARTH_SURFACE_AREA_M2, GAS_CONSTANT_J_KG_K

def download_etopo(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_bed_elev_gtif/ETOPO_2022_v1_60s_N90W180_bed.tif"
    url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif"
    dest = dest_dir / "etopo_60s.tif" # 60 arc-second resolution

    if dest.exists():
        print(f"File already exists at {dest}, skipping download.")
        return dest
    
    print(f"Downloading ETOPO data from {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")
    return dest

def _wrap_longitudes(lon_deg: np.ndarray) -> np.ndarray:
    """Wrap longitudes to the [-180, 180) range."""

    return ((lon_deg + 180.0) % 360.0) - 180.0

@lru_cache(maxsize=1)
def load_elevation_data(path: str | Path | None = None) -> xr.DataArray | None:
    """Return an elevation dataset registered to WGS84 coordinates."""

    if path is None:
        data_dir = os.getenv("DATA_DIR")
        if data_dir is None:
            raise ValueError("Please set the DATA_DIR environment variable.")
        data_dir = Path(data_dir)

        path = data_dir / "etopo_60s.tif"

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Elevation data file not found at {dataset_path}")

    data = rioxarray.open_rasterio(dataset_path).squeeze()
    if data.rio.crs is None:
        data = data.rio.write_crs("EPSG:4326", inplace=False)
    return data

def compute_cell_elevation(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    data: xr.DataArray | None = None,
    sample_method: str = "center",
    cache: bool = True,
) -> np.ndarray:
    """Return the elevation (m) for the provided grid cells."""
    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    if cache:
        # try to load from disk
        data_dir = os.getenv("DATA_DIR")
        if data_dir is not None:
            data_dir = Path(data_dir)
            cache_path = data_dir / "elevation_cache.npz"
            if cache_path.exists():
                try:
                    with np.load(cache_path) as cached:
                        if cached['elevation'].shape == lon2d.shape:
                            return cached['elevation']
                except Exception as e:
                    print(f"Failed to load cached elevation data: {e}, recomputing...")

    lon_array = np.asarray(lon2d, dtype=float)
    lat_array = np.asarray(lat2d, dtype=float)

    dataset = data if data is not None else load_elevation_data()
    if dataset is None:
        return np.zeros_like(lon_array)

    if sample_method != "center":
        raise ValueError("Only 'center' sampling is currently supported")

    if dataset.rio.crs is None:
        dataset = dataset.rio.write_crs("EPSG:4326", inplace=False)

    lon_wrapped = _wrap_longitudes(lon_array)
    lat_clamped = np.clip(lat_array, -90.0, 90.0)

    lon_flat = lon_wrapped.ravel()
    lat_flat = lat_clamped.ravel()

    # Sample the dataset at the cell centre using bilinear interpolation when possible.
    coords = {
        "x": ("points", lon_flat),
        "y": ("points", lat_flat),
    }

    try:
        sampled = dataset.interp(
            x=coords["x"], y=coords["y"], method="linear", kwargs={"fill_value": None}
        )
    except TypeError:
        sampled = dataset.interp(x=coords["x"], y=coords["y"], method="linear")

    values = np.asarray(sampled.values, dtype=float).reshape(lon_array.shape)

    if np.any(~np.isfinite(values)):
        sampled_nearest = dataset.interp(
            x=coords["x"], y=coords["y"], method="nearest"
        )
        nearest_values = np.asarray(sampled_nearest.values, dtype=float).reshape(
            lon_array.shape
        )
        mask = ~np.isfinite(values)
        values[mask] = nearest_values[mask]

    res = np.nan_to_num(values, nan=0.0)

    if cache:
        # save to disk
        data_dir = os.getenv("DATA_DIR")
        assert data_dir is not None, "Please set the DATA_DIR environment variable to enable elevation caching."
        data_dir = Path(data_dir)
        cache_path = data_dir / "elevation_cache.npz"
        np.savez_compressed(cache_path, elevation=res)

    return res


def compute_cell_roughness_length(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    data: xr.DataArray | None = None,
    *,
    land_mask: np.ndarray | None = None,
    cache: bool = True,
) -> np.ndarray:
    """Return the surface roughness length (m) for the provided grid cells."""

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    if land_mask is not None and land_mask.shape != lon2d.shape:
        raise ValueError("Provided land mask must match the longitude/latitude grid shape")

    use_cache = cache and land_mask is None

    if use_cache:
        data_dir = os.getenv("DATA_DIR")
        if data_dir is not None:
            data_dir = Path(data_dir)
            cache_path = data_dir / "roughness_length_cache.npz"
            legacy_cache_path = data_dir / "drag_coefficient_cache.npz"
            for candidate in (cache_path, legacy_cache_path):
                if not candidate.exists():
                    continue
                try:
                    with np.load(candidate) as cached:
                        roughness = cached.get("roughness_length")
                        if roughness is None:
                            drag_coeff = cached.get("drag_coefficient")
                            if drag_coeff is not None:
                                drag_array = np.asarray(drag_coeff, dtype=float)
                                drag_array = np.maximum(drag_array, 1.0e-12)
                                sqrt_drag = np.sqrt(drag_array)
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    exponent = VON_KARMAN_CONSTANT / sqrt_drag
                                roughness = REFERENCE_HEIGHT_M / np.exp(exponent)
                                if candidate is legacy_cache_path:
                                    np.savez_compressed(
                                        cache_path, roughness_length=roughness
                                    )
                        if roughness is not None and roughness.shape == lon2d.shape:
                            return np.asarray(roughness, dtype=float)
                except Exception as exc:  # pragma: no cover - logging path
                    print(
                        f"Failed to load cached roughness length data: {exc}, recomputing..."
                    )

    lon_array = np.asarray(lon2d, dtype=float)
    lat_array = np.asarray(lat2d, dtype=float)

    dataset = data if data is not None else load_elevation_data()
    if dataset is None:
        return np.full_like(lon_array, WATER_ROUGHNESS_LENGTH_M, dtype=float)

    elevation_m = compute_cell_elevation(
        lon_array,
        lat_array,
        data=dataset,
        sample_method="center",
        cache=cache,
    )

    earth_radius_m = 6.371e6
    lat_centers = lat_array[:, 0]
    lon_centers = lon_array[0, :]

    nlat, nlon = elevation_m.shape

    grad_x = np.zeros_like(elevation_m)
    grad_y = np.zeros_like(elevation_m)

    if nlon > 1:
        lon_diff = np.diff(lon_centers)
        if not np.allclose(lon_diff, lon_diff[0]):
            raise ValueError("Longitude grid must have constant spacing for roughness calculation")
        dlon_rad = np.deg2rad(lon_diff[0])
        delta_x = earth_radius_m * np.cos(np.deg2rad(lat_centers)) * dlon_rad
        inv_two_delta_x = np.zeros_like(delta_x)
        valid_dx = np.abs(delta_x) > 0.0
        inv_two_delta_x[valid_dx] = 1.0 / (2.0 * delta_x[valid_dx])
        padded = np.pad(elevation_m, ((0, 0), (1, 1)), mode="edge")
        grad_x = (padded[:, 2:] - padded[:, :-2]) * inv_two_delta_x[:, np.newaxis]

    if nlat > 1:
        lat_diff = np.diff(lat_centers)
        if not np.allclose(lat_diff, lat_diff[0]):
            raise ValueError("Latitude grid must have constant spacing for roughness calculation")
        dlat_rad = np.deg2rad(lat_diff[0])
        delta_y = earth_radius_m * dlat_rad
        inv_two_delta_y = 0.0
        if delta_y != 0.0:
            inv_two_delta_y = 1.0 / (2.0 * delta_y)
        padded = np.pad(elevation_m, ((1, 1), (0, 0)), mode="edge")
        grad_y = (padded[2:, :] - padded[:-2, :]) * inv_two_delta_y

    slope = np.hypot(grad_x, grad_y)

    gamma = 0.3
    length_scale_m = 1000.0
    z0_base_land = 0.02

    z0_orographic = gamma * (slope ** 2) * length_scale_m
    z0_orographic = np.minimum(z0_orographic, 0.8)

    z0_total = np.full_like(elevation_m, z0_base_land, dtype=float)
    terrain_land_mask = elevation_m >= 0.0
    if np.any(terrain_land_mask):
        combined = z0_base_land + z0_orographic
        combined = np.clip(combined, 0.02, 2.0)
        z0_total[terrain_land_mask] = combined[terrain_land_mask]

    roughness_result = np.array(z0_total, copy=True)

    water_mask: np.ndarray
    if land_mask is not None:
        water_mask = ~np.asarray(land_mask, dtype=bool)
    else:
        water_mask = ~terrain_land_mask

    if np.any(water_mask):
        roughness_result[water_mask] = WATER_ROUGHNESS_LENGTH_M

    if use_cache:
        data_dir = os.getenv("DATA_DIR")
        assert (
            data_dir is not None
        ), "Please set the DATA_DIR environment variable to enable roughness caching."
        data_dir = Path(data_dir)
        cache_path = data_dir / "roughness_length_cache.npz"
        np.savez_compressed(cache_path, roughness_length=roughness_result)

    return roughness_result


def neutral_drag_from_roughness_length(
    roughness_length_m: np.ndarray | float,
) -> np.ndarray:
    """Convert roughness length (m) to a neutral 10 m drag coefficient."""

    z0 = np.maximum(np.asarray(roughness_length_m, dtype=float), 1.0e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_argument = REFERENCE_HEIGHT_M / z0
        drag = (VON_KARMAN_CONSTANT / np.log(log_argument)) ** 2
    return drag


def compute_cell_neutral_drag_coefficient(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    data: xr.DataArray | None = None,
    *,
    land_mask: np.ndarray | None = None,
    cache: bool = True,
) -> np.ndarray:
    """Return the neutral 10 m drag coefficient for the provided grid cells."""

    roughness = compute_cell_roughness_length(
        lon2d,
        lat2d,
        data=data,
        land_mask=land_mask,
        cache=cache,
    )
    return neutral_drag_from_roughness_length(roughness)


def pressure_from_temperature_elevation(
    temperature_K: np.ndarray,
    elevation_m: np.ndarray | None = None,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    """Compute surface pressure (Pa) from temperature and elevation using a hydrostatic profile."""

    if elevation_m is not None and temperature_K.shape != elevation_m.shape:
        raise ValueError("Temperature and elevation fields must share the same shape")

    temp_safe = np.maximum(temperature_K, 1.0)
    elev = 0.0 if elevation_m is None else elevation_m

    # Base mean pressure from global mass balance
    mean_p = ATMOSPHERE_MASS * gravity_m_s2 / EARTH_SURFACE_AREA_M2

    # Warmer columns (larger T) produce slightly higher geopotential height for same top pressure
    # -> lower surface pressure (thermal low)
    # The factor exp(-g * H / (R T)) captures that effect
    p_surface = mean_p * np.exp(-gravity_m_s2 * elev / (GAS_CONSTANT_J_KG_K * temp_safe))
    p_surface *= np.exp(-gravity_m_s2 * 12500 / (GAS_CONSTANT_J_KG_K * temp_safe))

    # Normalize to conserve total mass
    p_surface *= mean_p / np.nanmean(p_surface)

    return p_surface

def _resample_elevation_grid(
    dataset: xr.DataArray,
    *,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the provided dataset onto a regular lon/lat grid."""

    da = dataset
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    if da.dims[-1] != "x" or da.dims[-2] != "y":
        da = da.rename({da.dims[-1]: "x", da.dims[-2]: "y"})

    xmin, ymin, xmax, ymax = da.rio.bounds()
    xmin = max(-180.0, xmin)
    xmax = min(180.0, xmax)
    ymin = max(-90.0, ymin)
    ymax = min(90.0, ymax)

    lons = np.arange(np.floor(xmin), np.ceil(xmax) + 1e-9, resolution)
    lats = np.arange(np.floor(ymin), np.ceil(ymax) + 1e-9, resolution)
    lon_centers = lons[:-1] + resolution / 2.0
    lat_centers = lats[:-1] + resolution / 2.0

    da_coarse = da.interp(x=("x", lon_centers), y=("y", lat_centers), method="linear")

    data = da_coarse.values.astype(np.float64)
    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)
    return data, lon2d, lat2d


def _resolve_output_dir(tif_path: Path, out_dir: Path | None) -> Path:
    if out_dir is None:
        out_dir = tif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _roughness_to_image(roughness: np.ndarray) -> np.ndarray:
    rough_norm = np.clip(roughness / 1.0, 0.0, 1.0)
    gray = (1.0 - rough_norm) * 255.0
    red = np.clip(gray + rough_norm * 255.0, 0.0, 255.0)

    img = np.zeros(roughness.shape + (3,), dtype=np.uint8)
    gray_uint = np.clip(np.round(gray), 0.0, 255.0).astype(np.uint8)
    img[..., 0] = np.clip(np.round(red), 0.0, 255.0).astype(np.uint8)
    img[..., 1] = gray_uint
    img[..., 2] = gray_uint
    return np.flipud(img)


def write_elevation_png_1deg(tif_path: Path, out_dir: Path | None = None, resolution: float = 0.1, contrast_water: bool = True) -> Path:
    out_dir = _resolve_output_dir(tif_path, out_dir)

    da = rioxarray.open_rasterio(tif_path).squeeze()
    data, _, _ = _resample_elevation_grid(da, resolution=resolution)
    # Compute min/max excluding NaNs
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("Elevation array contains no finite values after resampling.")
    
    vmin = float(data[finite].min())
    vmax = float(data[finite].max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = (data - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    img_arr = (norm * 255.0).astype(np.uint8)

    # make color if contrast water (green for land, blue for water)
    if contrast_water:
        img_arr_color = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype=np.uint8)
        # Land: greenish
        land = data >= 0.0
        water = data < 0.0
        img_arr_color[..., 0][land] = np.floor(img_arr[land] * 0.7)  # R
        img_arr_color[..., 1][land] = img_arr[land]       # G
        img_arr_color[..., 2][land] = np.floor(img_arr[land] * 0.7)  # B
        # Water: bluish
        img_arr_color[..., 0][water] = np.floor(img_arr[water] * 0.7)    # R
        img_arr_color[..., 1][water] = np.floor(img_arr[water] * 0.7)    # G
        img_arr_color[..., 2][water] = img_arr[water]  # B
        img_arr_color = np.flipud(img_arr_color)
        img = Image.fromarray(img_arr_color)
    else:
        # y increases upward in data, but image origin is top-left; flip vertically
        img_arr = np.flipud(img_arr)
        img = Image.fromarray(img_arr)

    out_path = out_dir / f"elevation_{resolution}deg.png"
    img.save(out_path)
    return out_path


def write_roughness_png(
    tif_path: Path,
    out_dir: Path | None = None,
    *,
    resolution: float = 0.1,
) -> Path:
    out_dir = _resolve_output_dir(tif_path, out_dir)

    da = rioxarray.open_rasterio(tif_path).squeeze()
    _, lon2d, lat2d = _resample_elevation_grid(da, resolution=resolution)
    roughness = compute_cell_roughness_length(lon2d, lat2d, data=da)

    out_path = out_dir / f"roughness_{resolution}deg.png"
    Image.fromarray(_roughness_to_image(roughness)).save(out_path)
    return out_path

if __name__ == "__main__":
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    if data_dir is None:
        raise ValueError("Please set the DATA_DIR environment variable.")
    data_dir = Path(data_dir)

    print(data_dir)
    etopo_path = download_etopo(data_dir)

    # Load the ETOPO data using rioxarray
    etopo_ds = rioxarray.open_rasterio(etopo_path)
    print(etopo_ds)

    # Example: Access elevation data
    elevation_data = etopo_ds[0]  # Assuming single band
    print(elevation_data)
    print(f"Elevation at (29 N, 86 E): {elevation_data.sel(y=29, x=86, method='nearest').values} meters")
    print(f"Elevation at (5 N, 86 E): {elevation_data.sel(y=5, x=86, method='nearest').values} meters")

    resolution = 0.1
    out = write_elevation_png_1deg(etopo_path, resolution=resolution)
    print(f"Wrote: {out}")

    surface_da = etopo_ds.squeeze()
    _, lon2d, lat2d = _resample_elevation_grid(surface_da, resolution=resolution)
    elevation_grid = compute_cell_elevation(lon2d, lat2d, data=surface_da)
    roughness = compute_cell_roughness_length(lon2d, lat2d, data=surface_da)

    global_min = float(np.min(roughness))
    global_mean = float(np.mean(roughness))
    global_max = float(np.max(roughness))
    print(
        f"Roughness stats (global): min={global_min:.4f} m, "
        f"mean={global_mean:.4f} m, max={global_max:.4f} m"
    )

    land_mask = elevation_grid >= 0.0
    if np.any(land_mask):
        land_min = float(np.min(roughness[land_mask]))
        land_mean = float(np.mean(roughness[land_mask]))
        land_max = float(np.max(roughness[land_mask]))
        print(
            f"Roughness stats (land): min={land_min:.4f} m, "
            f"mean={land_mean:.4f} m, max={land_max:.4f} m"
        )
    else:
        print("No cells with elevation >= 0 m found for land roughness statistics.")

    out_dir = _resolve_output_dir(etopo_path, None)
    rough_path = out_dir / f"roughness_{resolution}deg.png"
    Image.fromarray(_roughness_to_image(roughness)).save(rough_path)
    print(f"Wrote roughness map: {rough_path}")
