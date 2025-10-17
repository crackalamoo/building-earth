from pathlib import Path
import os
from dotenv import load_dotenv
import urllib.request

import numpy as np
import rioxarray
from PIL import Image
from functools import lru_cache
import xarray as xr

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


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert cell centre coordinates to bounding edges."""

    centres = np.asarray(centers, dtype=float)
    if centres.size == 0:
        raise ValueError("At least one centre coordinate is required")

    if centres.size == 1:
        half_width = 0.5
        return np.array([centres[0] - half_width, centres[0] + half_width], dtype=float)

    deltas = np.diff(centres)
    edges = np.empty(centres.size + 1, dtype=float)
    edges[1:-1] = centres[:-1] + deltas / 2.0
    edges[0] = centres[0] - deltas[0] / 2.0
    edges[-1] = centres[-1] + deltas[-1] / 2.0
    return edges


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
    cache: bool = True,
) -> np.ndarray:
    """Return the surface roughness length (m) for the provided grid cells."""

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    if cache:
        data_dir = os.getenv("DATA_DIR")
        if data_dir is not None:
            data_dir = Path(data_dir)
            cache_path = data_dir / "roughness_cache.npz"
            if cache_path.exists():
                try:
                    with np.load(cache_path) as cached:
                        roughness = cached["roughness"]
                        if roughness.shape == lon2d.shape:
                            return roughness
                except Exception as exc:  # pragma: no cover - logging path
                    print(f"Failed to load cached roughness data: {exc}, recomputing...")

    lon_array = np.asarray(lon2d, dtype=float)
    lat_array = np.asarray(lat2d, dtype=float)

    dataset = data if data is not None else load_elevation_data()
    if dataset is None:
        return np.full_like(lon_array, 0.02, dtype=float)

    if dataset.rio.crs is None:
        dataset = dataset.rio.write_crs("EPSG:4326", inplace=False)

    da = dataset
    if da.dims[-1] != "x" or da.dims[-2] != "y":
        da = da.rename({da.dims[-1]: "x", da.dims[-2]: "y"})

    lon_centers = lon_array[0, :]
    lat_centers = lat_array[:, 0]

    lon_edges = _centers_to_edges(lon_centers)

    lat_centers_for_edges = lat_centers
    reverse_lat_order = False
    if lat_centers.size > 1 and lat_centers[0] > lat_centers[-1]:
        lat_centers_for_edges = lat_centers[::-1]
        reverse_lat_order = True
    lat_edges = _centers_to_edges(lat_centers_for_edges)

    lon_hi = np.asarray(da["x"].values, dtype=float)
    lat_hi = np.asarray(da["y"].values, dtype=float)
    elevation_hi = np.asarray(da.values, dtype=float)

    lon_idx = np.digitize(lon_hi, lon_edges, right=False) - 1
    lon_idx = np.clip(lon_idx, 0, lon_centers.size - 1)

    lat_idx_sorted = np.digitize(lat_hi, lat_edges, right=False) - 1
    lat_idx_sorted = np.clip(lat_idx_sorted, 0, lat_centers.size - 1)
    if reverse_lat_order:
        lat_idx = (lat_centers.size - 1) - lat_idx_sorted
    else:
        lat_idx = lat_idx_sorted

    ny, nx = elevation_hi.shape
    lon_idx2d = np.broadcast_to(lon_idx, (ny, nx))
    lat_idx2d = np.broadcast_to(lat_idx[:, np.newaxis], (ny, nx))

    cell_index = (lat_idx2d * lon_centers.size + lon_idx2d).ravel()
    flat_values = elevation_hi.ravel()
    valid_mask = np.isfinite(flat_values)

    cell_index = cell_index[valid_mask]
    flat_values = flat_values[valid_mask]

    n_cells = lon_centers.size * lat_centers.size
    counts = np.bincount(cell_index, minlength=n_cells)
    sums = np.bincount(cell_index, weights=flat_values, minlength=n_cells)
    sums_sq = np.bincount(cell_index, weights=flat_values * flat_values, minlength=n_cells)

    counts_safe = np.maximum(counts, 1)
    mean_vals = sums / counts_safe
    variance = sums_sq / counts_safe - mean_vals ** 2
    variance = np.maximum(variance, 0.0)
    sigma = np.sqrt(variance)

    mean_vals[counts == 0] = 0.0
    sigma[counts == 0] = 0.0

    mean_vals = mean_vals.reshape(lat_centers.size, lon_centers.size)
    sigma = sigma.reshape(lat_centers.size, lon_centers.size)

    roughness = np.full_like(mean_vals, 0.02, dtype=float)
    land_mask = mean_vals >= 0.0
    if np.any(land_mask):
        rough_land = 0.02 * (1.0 + 10.0 * sigma[land_mask] / 100.0)
        roughness[land_mask] = np.minimum(rough_land, 1.0)

    if cache:
        data_dir = os.getenv("DATA_DIR")
        assert data_dir is not None, "Please set the DATA_DIR environment variable to enable roughness caching."
        data_dir = Path(data_dir)
        cache_path = data_dir / "roughness_cache.npz"
        np.savez_compressed(cache_path, roughness=roughness)

    return roughness


def pressure_from_temperature_elevation(
    temperature_K: np.ndarray,
    elevation_m: np.ndarray | None = None,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    """Compute surface pressure (Pa) from temperature and elevation using a hydrostatic profile."""

    if elevation_m is not None and temperature_K.shape != elevation_m.shape:
        raise ValueError("Temperature and elevation fields must share the same shape")

    # temp_safe = np.maximum(np.asarray(temperature_K, dtype=float), 1.0)
    # elev = elevation_m if elevation_m is not None else 0
    # exponent = -gravity_m_s2 * elev / (GAS_CONSTANT_J_KG_K * temp_safe)
    # sea_level_pressure_pa = 10 * np.exp(gravity_m_s2 * 12500 / (GAS_CONSTANT_J_KG_K * temp_safe))
    # sea_level_pressure_pa *= ATMOSPHERE_MASS / EARTH_SURFACE_AREA_M2 * gravity_m_s2 / np.mean(sea_level_pressure_pa)
    # pressure = sea_level_pressure_pa * np.exp(exponent)

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


def write_elevation_png_1deg(tif_path: Path, out_dir: Path | None = None, resolution: float = 0.1, contrast_water: bool = True) -> Path:
    if out_dir is None:
        out_dir = tif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

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
    roughness = compute_cell_roughness_length(lon2d, lat2d, data=surface_da)

    rough_norm = np.clip(roughness / 1.0, 0.0, 1.0)
    gray = (1.0 - rough_norm) * 255.0
    red = np.clip(gray + rough_norm * 255.0, 0.0, 255.0)

    img = np.zeros((roughness.shape[0], roughness.shape[1], 3), dtype=np.uint8)
    gray_uint = np.clip(np.round(gray), 0.0, 255.0).astype(np.uint8)
    img[..., 0] = np.clip(np.round(red), 0.0, 255.0).astype(np.uint8)
    img[..., 1] = gray_uint
    img[..., 2] = gray_uint
    img = np.flipud(img)

    rough_path = etopo_path.parent / f"roughness_{resolution}deg.png"
    Image.fromarray(img).save(rough_path)
    print(f"Wrote roughness map: {rough_path}")
