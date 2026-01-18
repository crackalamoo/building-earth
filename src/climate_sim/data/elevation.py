"""Elevation data utilities and roughness calculations for the climate model."""

from pathlib import Path
import os
from dotenv import load_dotenv
import urllib.request

import numpy as np
import rasterio
import rioxarray
from PIL import Image
from functools import lru_cache
import xarray as xr

from climate_sim.data.constants import R_EARTH_METERS
from climate_sim.core.math_core import regular_latitude_edges, regular_longitude_edges

VON_KARMAN_CONSTANT = 0.4
REFERENCE_HEIGHT_M = 10.0
WATER_ROUGHNESS_LENGTH_M = 2.0e-4


# Module-level cache for rasterio data to avoid repeated file reads
_rasterio_cache: dict[str, tuple[np.ndarray, rasterio.Affine, int, int]] = {}


def _get_rasterio_data(tif_path: Path) -> tuple[np.ndarray, rasterio.Affine, int, int]:
    """Load and cache GeoTIFF data using rasterio.

    Caches the full image read which is faster than xarray (~1s vs 3.5s).
    """
    global _rasterio_cache

    path_str = str(tif_path)
    if path_str in _rasterio_cache:
        return _rasterio_cache[path_str]

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        height, width = src.height, src.width

    result = (data, transform, height, width)
    _rasterio_cache[path_str] = result
    return result


def _sample_elevation_points_rasterio(
    tif_path: Path,
    lon_samples: np.ndarray,
    lat_samples: np.ndarray,
) -> np.ndarray:
    """Sample elevation at given lon/lat points using rasterio.

    Uses nearest-neighbor sampling since the source grid (1 arc-minute) is
    fine enough relative to typical model resolutions.
    """
    data, transform, height, width = _get_rasterio_data(tif_path)

    rows, cols = rasterio.transform.rowcol(transform, lon_samples, lat_samples)
    rows = np.clip(np.asarray(rows), 0, height - 1).astype(int)
    cols = np.clip(np.asarray(cols), 0, width - 1).astype(int)

    return data[rows, cols].astype(float)

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
        # Try to download the data
        data_dir = dataset_path.parent
        download_etopo(data_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Elevation data file not found at {dataset_path} even after download attempt")

    data = rioxarray.open_rasterio(dataset_path)
    assert isinstance(data, xr.DataArray)
    data = data.squeeze()
    if data.rio.crs is None:
        data = data.rio.write_crs("EPSG:4326", inplace=False)
    return data


# Module-level cache for extracted numpy arrays to avoid repeated slow .values calls
_elevation_arrays_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_elevation_arrays(dataset: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and cache numpy arrays from an xarray dataset.

    Returns (x_coords, y_coords, data_values) with y_coords sorted ascending
    for compatibility with RegularGridInterpolator.
    """
    global _elevation_arrays_cache

    dataset_id = id(dataset)
    if dataset_id in _elevation_arrays_cache:
        return _elevation_arrays_cache[dataset_id]

    x_coords = dataset.coords["x"].values
    y_coords = dataset.coords["y"].values
    data_values = dataset.values

    if y_coords[0] > y_coords[-1]:
        y_coords = y_coords[::-1]
        data_values = data_values[::-1, :]

    result = (x_coords, y_coords, data_values)
    _elevation_arrays_cache[dataset_id] = result
    return result

def compute_cell_elevation(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    data: xr.DataArray | None = None,
    cache: bool = True,
    n_samples_per_cell: int = 5,
) -> np.ndarray:
    """Return the average elevation (m) for the provided grid cells.
    
    Elevation is computed by sampling at multiple points within each cell
    and averaging the results.
    
    Parameters
    ----------
    lon2d : np.ndarray
        2D array of longitude centers (degrees)
    lat2d : np.ndarray
        2D array of latitude centers (degrees)
    data : xr.DataArray | None, optional
        Elevation dataset. If None, loads from default location.
    cache : bool, default True
        Whether to use disk caching.
    n_samples_per_cell : int, default 5
        Number of sample points per cell dimension.
        Total samples per cell = n_samples_per_cell^2.
    
    Returns
    -------
    np.ndarray
        Average elevation values (m) for each grid cell.
    """
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

    if dataset.rio.crs is None:
        dataset = dataset.rio.write_crs("EPSG:4326", inplace=False)

    # Compute cell edges
    lat_centers = lat_array[:, 0]
    lon_centers = lon_array[0, :]
    
    lat_edges = regular_latitude_edges(lat_centers)
    lon_edges = regular_longitude_edges(lon_centers)
    
    nlat, nlon = lon_array.shape
    res = np.zeros_like(lon_array, dtype=float)
    
    # Create sub-grid sampling points within each cell (vectorized)
    # Use n_samples_per_cell points along each dimension
    sample_weights = np.linspace(0.0, 1.0, n_samples_per_cell + 1)
    # Use midpoints of sub-intervals for better coverage
    sample_weights = (sample_weights[:-1] + sample_weights[1:]) / 2.0

    # Build all sample coordinates using broadcasting (no Python loops)
    # Create full 2D arrays for cell edges
    lat_min_2d = np.broadcast_to(lat_edges[:-1, np.newaxis], (nlat, nlon))  # (nlat, nlon)
    lat_max_2d = np.broadcast_to(lat_edges[1:, np.newaxis], (nlat, nlon))   # (nlat, nlon)
    lon_min_2d = np.broadcast_to(lon_edges[np.newaxis, :-1], (nlat, nlon))  # (nlat, nlon)
    lon_max_2d = np.broadcast_to(lon_edges[np.newaxis, 1:], (nlat, nlon))   # (nlat, nlon)

    # Handle longitude wrapping: where lon_max < lon_min, add 360
    lon_max_2d = np.where(lon_max_2d < lon_min_2d, lon_max_2d + 360.0, lon_max_2d)

    # Broadcast to (nlat, nlon, n_samples, n_samples)
    # sample_weights shape: (n_samples,)
    lat_weights = sample_weights[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, n, 1)
    lon_weights = sample_weights[np.newaxis, np.newaxis, np.newaxis, :]  # (1, 1, 1, n)

    lat_min_4d = lat_min_2d[:, :, np.newaxis, np.newaxis]  # (nlat, nlon, 1, 1)
    lat_max_4d = lat_max_2d[:, :, np.newaxis, np.newaxis]
    lon_min_4d = lon_min_2d[:, :, np.newaxis, np.newaxis]
    lon_max_4d = lon_max_2d[:, :, np.newaxis, np.newaxis]

    # Compute sample points - lat varies on axis 2, lon on axis 3
    # Shape will be (nlat, nlon, n, 1) and (nlat, nlon, 1, n) respectively
    lat_samples_partial = lat_min_4d + (lat_max_4d - lat_min_4d) * lat_weights
    lat_samples_partial = np.clip(lat_samples_partial, -90.0, 90.0)
    lon_samples_partial = lon_min_4d + (lon_max_4d - lon_min_4d) * lon_weights
    lon_samples_partial = _wrap_longitudes(lon_samples_partial)

    # Broadcast to full (nlat, nlon, n_samples, n_samples) shape
    target_shape = (nlat, nlon, n_samples_per_cell, n_samples_per_cell)
    lat_samples_4d = np.broadcast_to(lat_samples_partial, target_shape)
    lon_samples_4d = np.broadcast_to(lon_samples_partial, target_shape)

    # Flatten to 1D for interpolation
    lon_samples_flat = lon_samples_4d.ravel()
    lat_samples_flat = lat_samples_4d.ravel()

    # Get the GeoTIFF path for windowed reading
    data_dir = os.getenv("DATA_DIR")
    tif_path = Path(data_dir) / "etopo_60s.tif" if data_dir else None

    if tif_path is not None and tif_path.exists():
        # Use fast rasterio read (avoids slow xarray .values call)
        sampled_values = _sample_elevation_points_rasterio(
            tif_path, lon_samples_flat, lat_samples_flat
        )
    else:
        # Fallback to scipy interpolator if we have xarray dataset
        from scipy.interpolate import RegularGridInterpolator

        # Get cached numpy arrays (avoids slow .values call on each invocation)
        x_coords, y_coords, data_values = _get_elevation_arrays(dataset)

        interp_linear = RegularGridInterpolator(
            (y_coords, x_coords),
            data_values,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Stack points as (N, 2) array with (y, x) order
        points = np.stack([lat_samples_flat, lon_samples_flat], axis=-1)
        sampled_values = interp_linear(points)

        # Fill missing values with nearest neighbor
        if np.any(~np.isfinite(sampled_values)):
            interp_nearest = RegularGridInterpolator(
                (y_coords, x_coords),
                data_values,
                method="nearest",
                bounds_error=False,
                fill_value=0.0,
            )
            mask = ~np.isfinite(sampled_values)
            sampled_values[mask] = interp_nearest(points[mask])

    sampled_values = np.nan_to_num(sampled_values, nan=0.0)

    # Reshape back to (nlat, nlon, n_samples, n_samples) and average over samples
    sampled_reshaped = sampled_values.reshape(nlat, nlon, n_samples_per_cell, n_samples_per_cell)
    res = np.mean(sampled_reshaped, axis=(2, 3))
    
    res = np.nan_to_num(res, nan=0.0)

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
        cache=cache,
    )

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
        delta_x = R_EARTH_METERS * np.cos(np.deg2rad(lat_centers)) * dlon_rad
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
        delta_y = R_EARTH_METERS * dlat_rad
        inv_two_delta_y = 0.0
        if delta_y != 0.0:
            inv_two_delta_y = 1.0 / (2.0 * delta_y)
        padded = np.pad(elevation_m, ((1, 1), (0, 0)), mode="edge")
        grad_y = (padded[2:, :] - padded[:-2, :]) * inv_two_delta_y

    slope = np.hypot(grad_x, grad_y)

    gamma = 0.3
    length_scale_m = 1000.0
    z0_base_land = 5e-3

    z0_orographic = gamma * (slope ** 2) * length_scale_m
    z0_orographic = np.minimum(z0_orographic, 0.8)

    z0_total = np.full_like(elevation_m, z0_base_land, dtype=float)
    terrain_land_mask = elevation_m >= 0.0
    if np.any(terrain_land_mask):
        combined = z0_base_land + z0_orographic
        combined = np.clip(combined, 5.0e-3, 5.0e-2)
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

    da = rioxarray.open_rasterio(tif_path)
    assert isinstance(da, xr.DataArray)
    da = da.squeeze()
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

    da = rioxarray.open_rasterio(tif_path)
    assert isinstance(da, xr.DataArray)
    da = da.squeeze()
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

    assert isinstance(etopo_ds, xr.DataArray)
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
