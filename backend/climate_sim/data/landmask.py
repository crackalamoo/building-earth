"""Land/sea masking utilities backed by Natural Earth geometry."""

from __future__ import annotations

import functools

import numpy as np
from cartopy.io import shapereader
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

OCEAN_HEAT_CAPACITY_M2 = 2.94e8  # J m-2 K-1, 70 m mixed-layer ocean
LAND_HEAT_CAPACITY_M2 = 3.0e6  # J m-2 K-1, 1.5 m soil skin depth
OCEAN_ALBEDO = 0.06
LAND_ALBEDO = 0.18

_MASK_CACHE: dict[
    tuple[int, int, float, float, float, float], tuple[np.ndarray, np.ndarray]
] = {}


@functools.lru_cache(maxsize=1)
def _prepared_land_geometry():
    """Load and cache Natural Earth land polygons as a prepared geometry."""
    shapefile = shapereader.natural_earth(resolution="110m", category="physical", name="land")
    reader = shapereader.Reader(shapefile)
    geometry = unary_union(list(reader.geometries()))
    return prep(geometry)


@functools.lru_cache(maxsize=1)
def _prepared_lake_geometry():
    """Load and cache Natural Earth lake polygons as a prepared geometry."""
    shapefile = shapereader.natural_earth(resolution="110m", category="physical", name="lakes")
    reader = shapereader.Reader(shapefile)
    geometry = unary_union(list(reader.geometries()))
    return prep(geometry)


def _grid_signature(lon2d: np.ndarray, lat2d: np.ndarray) -> tuple[int, int, float, float, float, float]:
    """Return a tuple uniquely identifying a regular lon/lat grid."""
    nlat, nlon = lon2d.shape
    lon0 = float(lon2d[0, 0])
    lat0 = float(lat2d[0, 0])
    lon_step = float(np.round(lon2d[0, 1] - lon2d[0, 0], 6)) if nlon > 1 else 0.0
    lat_step = float(np.round(lat2d[1, 0] - lat2d[0, 0], 6)) if nlat > 1 else 0.0
    return (nlat, nlon, lon0, lat0, lon_step, lat_step)


def _compute_land_and_lake_masks(
    lon2d: np.ndarray, lat2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Determine the land/sea classification and highlight lakes for each grid center."""
    prepared_land = _prepared_land_geometry()
    prepared_lakes = _prepared_lake_geometry()
    flat_lon = lon2d.ravel()
    flat_lat = lat2d.ravel()

    land_flat = np.empty(flat_lon.size, dtype=bool)
    lake_flat = np.empty(flat_lon.size, dtype=bool)
    for idx, (lon, lat) in enumerate(zip(flat_lon, flat_lat, strict=True)):
        lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
        point = Point(lon_wrapped, lat)
        is_lake = prepared_lakes.covers(point)
        lake_flat[idx] = is_lake
        land_flat[idx] = prepared_land.covers(point) and not is_lake

    shape = lon2d.shape
    return land_flat.reshape(shape), lake_flat.reshape(shape)


def compute_land_mask(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """Return True where the grid cell center lies on land, with caching."""
    key = _grid_signature(lon2d, lat2d)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _compute_land_and_lake_masks(lon2d, lat2d)
    land_mask, _ = _MASK_CACHE[key]
    return land_mask


def compute_lake_mask(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """Return True where the grid cell center lies on a lake surface, with caching."""
    key = _grid_signature(lon2d, lat2d)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _compute_land_and_lake_masks(lon2d, lat2d)
    _, lake_mask = _MASK_CACHE[key]
    return lake_mask

def compute_heat_capacity_field(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    ocean_heat_capacity: float | None = None,
    land_heat_capacity: float | None = None,
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
    land_albedo: float | None = None,
    ocean_albedo: float | None = None,
) -> np.ndarray:
    """Assign per-cell albedo based on the land/sea classification."""
    land_mask = compute_land_mask(lon2d, lat2d)
    land_albedo = land_albedo if land_albedo is not None else LAND_ALBEDO
    ocean_albedo = ocean_albedo if ocean_albedo is not None else OCEAN_ALBEDO
    return np.where(land_mask, land_albedo, ocean_albedo)

def _default_grid(resolution_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a lon/lat grid matching the modeling module defaults."""
    lats = np.arange(-90.0 + resolution_deg / 2, 90.0, resolution_deg)
    lons = np.arange(resolution_deg / 2, 360.0, resolution_deg)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d


def _plot_mask(land_mask: np.ndarray, lake_mask: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray) -> None:
    """Render the land and lake masks for quick inspection."""
    lon_centers = lon2d[0, :]
    lon_wrapped = ((lon_centers + 180.0) % 360.0) - 180.0
    sort_idx = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[sort_idx]
    lon2d_sorted = np.repeat(lon_sorted[np.newaxis, :], land_mask.shape[0], axis=0)
    lat2d_sorted = lat2d[:, sort_idx]

    classification = np.zeros_like(land_mask, dtype=int)
    classification[land_mask] = 2
    classification[lake_mask] = 1
    classification_sorted = classification[:, sort_idx]

    cmap = ListedColormap(["#4C72B0", "#A6CEE3", "#55A868"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(
        lon2d_sorted,
        lat2d_sorted,
        classification_sorted,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title("Surface Classification (1° resolution)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Ocean", "Lake", "Land"])
    plt.show()


def main() -> None:
    """Entry point to visualise the cached land mask."""
    lon2d, lat2d = _default_grid()
    land_mask = compute_land_mask(lon2d, lat2d)
    lake_mask = compute_lake_mask(lon2d, lat2d)
    _plot_mask(land_mask, lake_mask, lon2d, lat2d)


if __name__ == "__main__":
    main()
