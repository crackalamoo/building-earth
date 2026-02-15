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
from scipy.interpolate import RegularGridInterpolator

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


def compute_cell_elevation_statistics(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    data: xr.DataArray | None = None,
    cache: bool = True,
    n_samples_per_cell: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sub-grid elevation statistics for each grid cell.

    Parameters
    ----------
    lon2d, lat2d : np.ndarray
        2D arrays of cell centers (degrees).
    data : xr.DataArray | None
        Elevation dataset. If None, loads from default location.
    cache : bool
        Whether to use disk caching.
    n_samples_per_cell : int
        Samples per cell dimension (15×15=225 for robust statistics).

    Returns
    -------
    elevation_std : np.ndarray
        Standard deviation of fine-resolution elevation within each cell (m).
    elevation_max : np.ndarray
        Peak elevation within each cell (m).
    """
    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    if cache:
        data_dir = os.getenv("DATA_DIR")
        if data_dir is not None:
            cache_path = Path(data_dir) / "elevation_statistics_cache.npz"
            if cache_path.exists():
                try:
                    with np.load(cache_path) as cached:
                        if cached['elevation_std'].shape == lon2d.shape:
                            return cached['elevation_std'], cached['elevation_max']
                except Exception as e:
                    print(f"Failed to load cached elevation statistics: {e}, recomputing...")

    lon_array = np.asarray(lon2d, dtype=float)
    lat_array = np.asarray(lat2d, dtype=float)

    dataset = data if data is not None else load_elevation_data()
    if dataset is None:
        return np.zeros_like(lon_array), np.zeros_like(lon_array)

    if dataset.rio.crs is None:
        dataset = dataset.rio.write_crs("EPSG:4326", inplace=False)

    # Compute cell edges
    lat_centers = lat_array[:, 0]
    lon_centers = lon_array[0, :]
    lat_edges = regular_latitude_edges(lat_centers)
    lon_edges = regular_longitude_edges(lon_centers)
    nlat, nlon = lon_array.shape

    # Build sub-grid sampling points (same logic as compute_cell_elevation)
    sample_weights = np.linspace(0.0, 1.0, n_samples_per_cell + 1)
    sample_weights = (sample_weights[:-1] + sample_weights[1:]) / 2.0

    lat_min_2d = np.broadcast_to(lat_edges[:-1, np.newaxis], (nlat, nlon))
    lat_max_2d = np.broadcast_to(lat_edges[1:, np.newaxis], (nlat, nlon))
    lon_min_2d = np.broadcast_to(lon_edges[np.newaxis, :-1], (nlat, nlon))
    lon_max_2d = np.broadcast_to(lon_edges[np.newaxis, 1:], (nlat, nlon))
    lon_max_2d = np.where(lon_max_2d < lon_min_2d, lon_max_2d + 360.0, lon_max_2d)

    lat_weights = sample_weights[np.newaxis, np.newaxis, :, np.newaxis]
    lon_weights = sample_weights[np.newaxis, np.newaxis, np.newaxis, :]

    lat_min_4d = lat_min_2d[:, :, np.newaxis, np.newaxis]
    lat_max_4d = lat_max_2d[:, :, np.newaxis, np.newaxis]
    lon_min_4d = lon_min_2d[:, :, np.newaxis, np.newaxis]
    lon_max_4d = lon_max_2d[:, :, np.newaxis, np.newaxis]

    lat_samples_partial = np.clip(
        lat_min_4d + (lat_max_4d - lat_min_4d) * lat_weights, -90.0, 90.0
    )
    lon_samples_partial = _wrap_longitudes(
        lon_min_4d + (lon_max_4d - lon_min_4d) * lon_weights
    )

    target_shape = (nlat, nlon, n_samples_per_cell, n_samples_per_cell)
    lat_samples_4d = np.broadcast_to(lat_samples_partial, target_shape)
    lon_samples_4d = np.broadcast_to(lon_samples_partial, target_shape)

    lon_flat = lon_samples_4d.ravel()
    lat_flat = lat_samples_4d.ravel()

    # Sample elevation
    data_dir_env = os.getenv("DATA_DIR")
    tif_path = Path(data_dir_env) / "etopo_60s.tif" if data_dir_env else None

    if tif_path is not None and tif_path.exists():
        sampled = _sample_elevation_points_rasterio(tif_path, lon_flat, lat_flat)
    else:
        from scipy.interpolate import RegularGridInterpolator
        x_coords, y_coords, data_values = _get_elevation_arrays(dataset)
        interp = RegularGridInterpolator(
            (y_coords, x_coords), data_values,
            method="nearest", bounds_error=False, fill_value=0.0,
        )
        sampled = interp(np.stack([lat_flat, lon_flat], axis=-1))

    sampled = np.nan_to_num(sampled, nan=0.0)
    sampled_4d = sampled.reshape(nlat, nlon, n_samples_per_cell, n_samples_per_cell)

    elevation_std = np.std(sampled_4d, axis=(2, 3))
    elevation_max = np.max(sampled_4d, axis=(2, 3))

    if cache:
        data_dir_env = os.getenv("DATA_DIR")
        if data_dir_env is not None:
            cache_path = Path(data_dir_env) / "elevation_statistics_cache.npz"
            np.savez_compressed(
                cache_path, elevation_std=elevation_std, elevation_max=elevation_max
            )

    return elevation_std, elevation_max


def compute_face_elevation_statistics(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    cache: bool = True,
    n_samples_per_cell: int = 15,
) -> dict[str, np.ndarray]:
    """Compute directional cross-sectional blockage ratios at cell faces.

    For each face between two adjacent cells, we sample fine-resolution
    elevation in a strip centered on the face (half a cell on each side).
    The blockage ratio is the fraction of the BL cross-section that is open.

    At each cross-section position (latitude row for east faces, longitude
    column for north faces), we find the silhouette height — the max elevation
    across the flow direction.  The blocked fraction at that position is
    ``clip((silhouette - z_entry) / H_BL, 0, 1)`` where ``z_entry`` is the
    upwind cell-mean elevation and ``H_BL`` is the effective BL depth.

    This naturally distinguishes:
    - Continuous ridges (blocked at every position → low open fraction)
    - Isolated peaks (blocked at few positions → high open fraction)
    - High plateaus (silhouette ≈ z_entry → minimal blocking)

    Returns
    -------
    dict with keys:
      r_east_pos : (nlat, nlon) open fraction for eastward  (u>0) flow
      r_east_neg : (nlat, nlon) open fraction for westward  (u<0) flow
      r_north_pos: (nlat, nlon) open fraction for northward (v>0) flow
      r_north_neg: (nlat, nlon) open fraction for southward (v<0) flow
    Values range from 0 (fully blocked) to 1 (fully open).
    """
    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    nlat, nlon = lon2d.shape
    cache_name = "face_elevation_cache.npz"
    expected_keys = [
        "r_east_pos", "r_east_neg", "r_north_pos", "r_north_neg",
        "r_east_pos_eddy", "r_east_neg_eddy", "r_north_pos_eddy", "r_north_neg_eddy",
        "grad_x_pos", "grad_x_neg", "grad_y_pos", "grad_y_neg",
    ]

    if cache:
        data_dir = os.getenv("DATA_DIR")
        if data_dir is not None:
            cache_path = Path(data_dir) / cache_name
            if cache_path.exists():
                try:
                    with np.load(cache_path) as cached:
                        if all(k in cached and cached[k].shape == (nlat, nlon) for k in expected_keys):
                            return {k: cached[k] for k in expected_keys}
                except Exception as e:
                    print(f"Failed to load face elevation cache: {e}, recomputing...")

    lat_centers = lat2d[:, 0]
    lon_centers = lon2d[0, :]
    lat_edges = regular_latitude_edges(lat_centers)
    lon_edges = regular_longitude_edges(lon_centers)

    # Sample weights within a single cell dimension
    sw = np.linspace(0.0, 1.0, n_samples_per_cell + 1)
    sw = (sw[:-1] + sw[1:]) / 2.0  # cell-interior sample positions [0..1]

    # ---- helper: sample a rectangle of fine-res elevation ----
    data_dir_env = os.getenv("DATA_DIR")
    tif_path = Path(data_dir_env) / "etopo_60s.tif" if data_dir_env else None
    dataset = None
    interp = None
    if tif_path is None or not tif_path.exists():
        dataset = load_elevation_data()
        if dataset is None:
            return {k: np.zeros((nlat, nlon)) for k in expected_keys}
        if dataset.rio.crs is None:
            dataset = dataset.rio.write_crs("EPSG:4326", inplace=False)
        x_coords, y_coords, data_values = _get_elevation_arrays(dataset)
        interp = RegularGridInterpolator(
            (y_coords, x_coords), data_values,
            method="nearest", bounds_error=False, fill_value=0.0,
        )

    def _sample_rect(lat_lo: np.ndarray, lat_hi: np.ndarray,
                     lon_lo: np.ndarray, lon_hi: np.ndarray,
                     n_lat: int, n_lon: int) -> np.ndarray:
        """Sample fine-res elevation in rectangles.

        lat_lo/hi, lon_lo/hi: (nlat, nlon) arrays of rectangle bounds.
        Returns: (nlat, nlon, n_lat, n_lon) sampled elevation.
        """
        sw_lat = np.linspace(0.0, 1.0, n_lat + 1)
        sw_lat = (sw_lat[:-1] + sw_lat[1:]) / 2.0
        sw_lon = np.linspace(0.0, 1.0, n_lon + 1)
        sw_lon = (sw_lon[:-1] + sw_lon[1:]) / 2.0

        # Wrap longitudes where needed
        lon_hi_w = np.where(lon_hi < lon_lo, lon_hi + 360.0, lon_hi)

        lat_s = np.clip(
            lat_lo[:, :, np.newaxis, np.newaxis]
            + (lat_hi - lat_lo)[:, :, np.newaxis, np.newaxis]
            * sw_lat[np.newaxis, np.newaxis, :, np.newaxis],
            -90.0, 90.0,
        )
        lon_s = _wrap_longitudes(
            lon_lo[:, :, np.newaxis, np.newaxis]
            + (lon_hi_w - lon_lo)[:, :, np.newaxis, np.newaxis]
            * sw_lon[np.newaxis, np.newaxis, np.newaxis, :],
        )

        target_shape = (nlat, nlon, n_lat, n_lon)
        lat_s = np.broadcast_to(lat_s, target_shape)
        lon_s = np.broadcast_to(lon_s, target_shape)
        lon_flat = lon_s.ravel()
        lat_flat = lat_s.ravel()

        if tif_path is not None and tif_path.exists():
            vals = _sample_elevation_points_rasterio(tif_path, lon_flat, lat_flat)
        else:
            assert interp is not None
            vals = interp(np.stack([lat_flat, lon_flat], axis=-1))

        # Clamp to >= 0: ocean bathymetry shouldn't create terrain barriers.
        return np.maximum(0.0, np.nan_to_num(vals, nan=0.0)).reshape(target_shape)

    # ---- East faces: strip centered on face, half-cell on each side ----
    # Lat bounds: same as each cell's lat range
    lat_lo_e = np.broadcast_to(lat_edges[:-1, np.newaxis], (nlat, nlon))
    lat_hi_e = np.broadcast_to(lat_edges[1:, np.newaxis], (nlat, nlon))
    # Lon bounds: cell center j to cell center j+1 (half-cell each side of face)
    lon_lo_e = np.broadcast_to(lon_centers[np.newaxis, :], (nlat, nlon))
    lon_hi_e = np.broadcast_to(np.roll(lon_centers, -1)[np.newaxis, :], (nlat, nlon))

    east_elev = _sample_rect(lat_lo_e, lat_hi_e, lon_lo_e, lon_hi_e,
                             n_samples_per_cell, n_samples_per_cell)
    # shape: (nlat, nlon, n_lat_samples, n_lon_samples)

    # Silhouette: at each latitude row, find the max elevation across longitudes.
    # This is the terrain wall the flow must cross at that latitude.
    east_silhouette = np.max(east_elev, axis=3)  # (nlat, nlon, n_lat_samples)

    # Entry elevation: cell-mean elevation on each side (our model's "surface").
    # Clamp to >= 0 so ocean cells have z_entry = 0 (sea level).
    cell_elev = np.maximum(0.0, compute_cell_elevation(lon2d, lat2d))
    z_entry_left = cell_elev[:, :, np.newaxis]                          # cell j
    z_entry_right = np.roll(cell_elev, -1, axis=1)[:, :, np.newaxis]   # cell j+1

    # Blockage ratios for two different physics:
    # 1. Wind (advection): H_wind = BL depth (~1000m). Mountains taller than the BL
    #    completely block surface wind — flow goes around, not over.
    # 2. Eddies (diffusion): H_eddy = moisture-weighted tropospheric depth (~5000m).
    #    Baroclinic eddies extend through the full troposphere, but most moisture
    #    lives below 3-4 km (scale height ~2 km), so a 3 km range blocks most of
    #    the moisture-carrying capacity even though storms pass over it.
    H_WIND = 1500.0   # BL depth (m) — for advection/wind blocking
    H_EDDY = 5000.0   # moisture-weighted tropospheric depth (m) — for diffusion blocking

    def _compute_r(silhouette: np.ndarray, z_entry: np.ndarray, H: float) -> np.ndarray:
        blocked = np.clip((silhouette - z_entry) / H, 0.0, 1.0)
        return 1.0 - np.mean(blocked, axis=2)

    r_east_pos_wind = _compute_r(east_silhouette, z_entry_left, H_WIND)
    r_east_neg_wind = _compute_r(east_silhouette, z_entry_right, H_WIND)
    r_east_pos_eddy = _compute_r(east_silhouette, z_entry_left, H_EDDY)
    r_east_neg_eddy = _compute_r(east_silhouette, z_entry_right, H_EDDY)

    # ---- North faces: strip centered on face, half-cell on each side ----
    lat_lo_n = np.broadcast_to(lat_centers[:, np.newaxis], (nlat, nlon))
    lat_hi_n = np.broadcast_to(np.roll(lat_centers, -1)[:, np.newaxis], (nlat, nlon))
    lon_lo_n = np.broadcast_to(lon_edges[np.newaxis, :-1], (nlat, nlon))
    lon_hi_n = np.broadcast_to(lon_edges[np.newaxis, 1:], (nlat, nlon))

    north_elev = _sample_rect(lat_lo_n, lat_hi_n, lon_lo_n, lon_hi_n,
                              n_samples_per_cell, n_samples_per_cell)
    # shape: (nlat, nlon, n_lat_samples, n_lon_samples)

    # Silhouette: at each longitude column, find max elevation across latitudes.
    north_silhouette = np.max(north_elev, axis=2)  # (nlat, nlon, n_lon_samples)

    z_entry_south = cell_elev[:, :, np.newaxis]                         # cell i
    z_entry_north = np.roll(cell_elev, -1, axis=0)[:, :, np.newaxis]   # cell i+1

    r_north_pos_wind = _compute_r(north_silhouette, z_entry_south, H_WIND)
    r_north_neg_wind = _compute_r(north_silhouette, z_entry_north, H_WIND)
    r_north_pos_eddy = _compute_r(north_silhouette, z_entry_south, H_EDDY)
    r_north_neg_eddy = _compute_r(north_silhouette, z_entry_north, H_EDDY)

    # ---- Directional orographic gradients from fine-res data ----
    # Sample fine-res elevation within each cell to compute effective terrain
    # gradients.  At coarse resolution, cell-mean gradients average out windward
    # and leeward slopes.  Instead we separate positive (upslope) and negative
    # (downslope) contributions so orographic precipitation sees the actual
    # mountain slope, not the smoothed cell-mean.
    lat_lo_c = np.broadcast_to(lat_edges[:-1, np.newaxis], (nlat, nlon))
    lat_hi_c = np.broadcast_to(lat_edges[1:, np.newaxis], (nlat, nlon))
    lon_lo_c = np.broadcast_to(lon_edges[np.newaxis, :-1], (nlat, nlon))
    lon_hi_c = np.broadcast_to(lon_edges[np.newaxis, 1:], (nlat, nlon))

    cell_elev_fine = _sample_rect(lat_lo_c, lat_hi_c, lon_lo_c, lon_hi_c,
                                  n_samples_per_cell, n_samples_per_cell)
    # shape: (nlat, nlon, n_lat, n_lon)

    # Compute ∂h/∂x from finite differences of fine-res samples within each cell.
    # dx between adjacent samples = cell_width / n_samples
    earth_radius = 6.371e6
    dlat = np.deg2rad(lat_edges[1:] - lat_edges[:-1])  # (nlat,)
    dlon = np.deg2rad(lon_edges[1:] - lon_edges[:-1])   # (nlon,)
    cos_lat = np.cos(np.deg2rad(lat_centers))

    # Distance between adjacent fine-res samples (m)
    dx_sample = (earth_radius * cos_lat[:, np.newaxis] * dlon[np.newaxis, :]
                 / n_samples_per_cell)  # (nlat, nlon)
    dy_sample = (earth_radius * dlat[:, np.newaxis]
                 / n_samples_per_cell)  # (nlat, 1) broadcast to (nlat, nlon)
    dy_sample = np.broadcast_to(dy_sample, (nlat, nlon))

    # Central differences for interior points, forward/backward at edges
    # ∂h/∂x: gradient in the longitude (east-west) direction
    dhdx = np.zeros_like(cell_elev_fine)
    dhdx[:, :, :, 1:-1] = ((cell_elev_fine[:, :, :, 2:] - cell_elev_fine[:, :, :, :-2])
                            / (2.0 * dx_sample[:, :, np.newaxis, np.newaxis]))
    dhdx[:, :, :, 0] = ((cell_elev_fine[:, :, :, 1] - cell_elev_fine[:, :, :, 0])
                         / dx_sample[:, :, np.newaxis])
    dhdx[:, :, :, -1] = ((cell_elev_fine[:, :, :, -1] - cell_elev_fine[:, :, :, -2])
                          / dx_sample[:, :, np.newaxis])

    # ∂h/∂y: gradient in the latitude (north-south) direction
    dhdy = np.zeros_like(cell_elev_fine)
    dhdy[:, :, 1:-1, :] = ((cell_elev_fine[:, :, 2:, :] - cell_elev_fine[:, :, :-2, :])
                            / (2.0 * dy_sample[:, :, np.newaxis, np.newaxis]))
    dhdy[:, :, 0, :] = ((cell_elev_fine[:, :, 1, :] - cell_elev_fine[:, :, 0, :])
                         / dy_sample[:, :, np.newaxis])
    dhdy[:, :, -1, :] = ((cell_elev_fine[:, :, -1, :] - cell_elev_fine[:, :, -2, :])
                          / dy_sample[:, :, np.newaxis])

    # Separate into positive (upslope) and negative (downslope) gradients,
    # then average over all sample points in each cell.
    # grad_x_pos: mean positive ∂h/∂x (upslope for eastward flow)
    # grad_x_neg: mean positive -∂h/∂x (upslope for westward flow)
    grad_x_pos = np.mean(np.maximum(dhdx, 0.0), axis=(2, 3))  # (nlat, nlon)
    grad_x_neg = np.mean(np.maximum(-dhdx, 0.0), axis=(2, 3))
    grad_y_pos = np.mean(np.maximum(dhdy, 0.0), axis=(2, 3))
    grad_y_neg = np.mean(np.maximum(-dhdy, 0.0), axis=(2, 3))

    result = {
        # Wind blocking (advection) — H = 1500m
        "r_east_pos": r_east_pos_wind,
        "r_east_neg": r_east_neg_wind,
        "r_north_pos": r_north_pos_wind,
        "r_north_neg": r_north_neg_wind,
        # Eddy blocking (diffusion) — H = 5000m
        "r_east_pos_eddy": r_east_pos_eddy,
        "r_east_neg_eddy": r_east_neg_eddy,
        "r_north_pos_eddy": r_north_pos_eddy,
        "r_north_neg_eddy": r_north_neg_eddy,
        # Directional orographic gradients (m/m)
        "grad_x_pos": grad_x_pos,
        "grad_x_neg": grad_x_neg,
        "grad_y_pos": grad_y_pos,
        "grad_y_neg": grad_y_neg,
    }

    if cache:
        data_dir_env = os.getenv("DATA_DIR")
        if data_dir_env is not None:
            cache_path = Path(data_dir_env) / cache_name
            np.savez_compressed(cache_path, **result)

    return result


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
