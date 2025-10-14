"""Land/sea masking utilities backed by Natural Earth geometry."""

from __future__ import annotations

import functools
from typing import Tuple

import numpy as np
from cartopy.io import shapereader
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
from typing import Optional

OCEAN_HEAT_CAPACITY_M2 = 4.0e8  # J m-2 K-1, ~40 m mixed-layer ocean
LAND_HEAT_CAPACITY_M2 = 6.0e6  # J m-2 K-1, ~3 m soil skin depth
OCEAN_ALBEDO = 0.3
LAND_ALBEDO = 0.3

_MASK_CACHE: dict[Tuple[int, int, float, float, float, float], np.ndarray] = {}


@functools.lru_cache(maxsize=1)
def _prepared_land_geometry():
    """Load and cache Natural Earth land polygons as a prepared geometry."""
    shapefile = shapereader.natural_earth(resolution="110m", category="physical", name="land")
    reader = shapereader.Reader(shapefile)
    geometry = unary_union(list(reader.geometries()))
    return prep(geometry)


def _grid_signature(lon2d: np.ndarray, lat2d: np.ndarray) -> Tuple[int, int, float, float, float, float]:
    """Return a tuple uniquely identifying a regular lon/lat grid."""
    nlat, nlon = lon2d.shape
    lon0 = float(lon2d[0, 0])
    lat0 = float(lat2d[0, 0])
    lon_step = float(np.round(lon2d[0, 1] - lon2d[0, 0], 6)) if nlon > 1 else 0.0
    lat_step = float(np.round(lat2d[1, 0] - lat2d[0, 0], 6)) if nlat > 1 else 0.0
    return (nlat, nlon, lon0, lat0, lon_step, lat_step)


def _compute_land_mask(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """Determine the land/sea classification for each grid centre."""
    prepared_land = _prepared_land_geometry()
    flat_lon = lon2d.ravel()
    flat_lat = lat2d.ravel()

    mask_flat = np.empty(flat_lon.size, dtype=bool)
    for idx, (lon, lat) in enumerate(zip(flat_lon, flat_lat, strict=True)):
        lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
        mask_flat[idx] = prepared_land.covers(Point(lon_wrapped, lat))

    return mask_flat.reshape(lon2d.shape)


def compute_land_mask(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """Return True where the grid cell centre lies on land, with caching."""
    key = _grid_signature(lon2d, lat2d)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _compute_land_mask(lon2d, lat2d)
    return _MASK_CACHE[key]

def compute_heat_capacity_field(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    ocean_heat_capacity: Optional[float]=None,
    land_heat_capacity: Optional[float]=None,
) -> np.ndarray:
    """Assign per-cell heat capacity based on land/sea classification."""
    land_mask = compute_land_mask(lon2d, lat2d)
    ocean_heat_capacity = ocean_heat_capacity if ocean_heat_capacity is not None else OCEAN_HEAT_CAPACITY_M2
    land_heat_capacity = land_heat_capacity if land_heat_capacity is not None else LAND_HEAT_CAPACITY_M2
    return np.where(land_mask, land_heat_capacity, ocean_heat_capacity)


def compute_albedo_field(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    land_albedo: Optional[float] = None,
    ocean_albedo: Optional[float] = None,
) -> np.ndarray:
    """Assign per-cell albedo based on the land/sea classification."""
    land_mask = compute_land_mask(lon2d, lat2d)
    land_albedo = land_albedo if land_albedo is not None else LAND_ALBEDO
    ocean_albedo = ocean_albedo if ocean_albedo is not None else OCEAN_ALBEDO
    return np.where(land_mask, land_albedo, ocean_albedo)

def _default_grid(resolution_deg: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a lon/lat grid matching the modeling module defaults."""
    lats = np.arange(-90.0 + resolution_deg / 2, 90.0, resolution_deg)
    lons = np.arange(resolution_deg / 2, 360.0, resolution_deg)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d


def _plot_mask(mask: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray) -> None:
    """Render the boolean mask for quick inspection."""
    lon_centers = lon2d[0, :]
    lon_wrapped = ((lon_centers + 180.0) % 360.0) - 180.0
    sort_idx = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[sort_idx]
    mask_sorted = mask[:, sort_idx]
    lon2d_sorted = np.repeat(lon_sorted[np.newaxis, :], mask_sorted.shape[0], axis=0)
    lat2d_sorted = lat2d[:, sort_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(lon2d_sorted, lat2d_sorted, mask_sorted.astype(float), shading="auto", cmap="Greens")
    ax.set_title("Land Mask (1° resolution)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Land=1, Ocean=0")
    plt.show()


def main() -> None:
    """Entry point to visualise the cached land mask."""
    lon2d, lat2d = _default_grid()
    mask = compute_land_mask(lon2d, lat2d)
    _plot_mask(mask, lon2d, lat2d)


if __name__ == "__main__":
    main()
