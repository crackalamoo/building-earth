"""Land/sea masking utilities with optional Natural Earth support."""

from __future__ import annotations

import functools
import os
from typing import Tuple

import numpy as np
from cartopy.io import shapereader
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
from typing import Optional

OCEAN_HEAT_CAPACITY_M2 = 4.0e8  # J m-2 K-1, ~40 m mixed-layer ocean
LAND_HEAT_CAPACITY_M2 = 6.0e6  # J m-2 K-1, ~3 m soil skin depth
OCEAN_ALBEDO = 0.3
LAND_ALBEDO = 0.3

_MASK_CACHE: dict[
    Tuple[int, int, float, float, float, float], tuple[np.ndarray, np.ndarray]
] = {}

USE_NATURAL_EARTH = (
    os.environ.get("CLIMATE_SIM_USE_NATURAL_EARTH", "").strip().lower()
    in {"1", "true", "yes"}
)


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


def _grid_signature(lon2d: np.ndarray, lat2d: np.ndarray) -> Tuple[int, int, float, float, float, float]:
    """Return a tuple uniquely identifying a regular lon/lat grid."""
    nlat, nlon = lon2d.shape
    lon0 = float(lon2d[0, 0])
    lat0 = float(lat2d[0, 0])
    lon_step = float(np.round(lon2d[0, 1] - lon2d[0, 0], 6)) if nlon > 1 else 0.0
    lat_step = float(np.round(lat2d[1, 0] - lat2d[0, 0], 6)) if nlat > 1 else 0.0
    return (nlat, nlon, lon0, lat0, lon_step, lat_step)


def _fallback_land_lake_masks(
    lon2d: np.ndarray, lat2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return coarse analytic land and lake masks when Natural Earth data are unavailable."""
    lon_wrapped = ((lon2d + 180.0) % 360.0) - 180.0
    land_mask = np.zeros_like(lon2d, dtype=bool)

    def add_box(lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
        lon_cond = (lon_wrapped >= lon_min) & (lon_wrapped <= lon_max)
        lat_cond = (lat2d >= lat_min) & (lat2d <= lat_max)
        land_mask[:] |= lon_cond & lat_cond

    # Rough continental approximations to preserve large-scale contrasts for diagnostics.
    add_box(-170.0, -50.0, 15.0, 75.0)   # North America
    add_box(-85.0, -35.0, -55.0, 15.0)   # South America
    add_box(-20.0, 60.0, -35.0, 75.0)    # Africa and Europe
    add_box(40.0, 180.0, 5.0, 80.0)      # Asia
    add_box(110.0, 155.0, -45.0, -10.0)  # Australia
    add_box(-75.0, -15.0, 60.0, 85.0)    # Greenland
    add_box(165.0, 180.0, -50.0, -30.0)  # New Zealand (west lon)
    add_box(-180.0, -165.0, -50.0, -30.0)  # New Zealand (east lon)

    # Antarctica
    land_mask |= lat2d <= -60.0

    lake_mask = np.zeros_like(land_mask, dtype=bool)
    return land_mask, lake_mask


def _compute_land_and_lake_masks(
    lon2d: np.ndarray, lat2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Determine the land/sea classification and highlight lakes for each grid centre."""
    if USE_NATURAL_EARTH:
        try:
            prepared_land = _prepared_land_geometry()
            prepared_lakes = _prepared_lake_geometry()
        except Exception:
            return _fallback_land_lake_masks(lon2d, lat2d)
    else:
        return _fallback_land_lake_masks(lon2d, lat2d)
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
    """Return True where the grid cell centre lies on land, with caching."""
    key = _grid_signature(lon2d, lat2d)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _compute_land_and_lake_masks(lon2d, lat2d)
    land_mask, _ = _MASK_CACHE[key]
    return land_mask


def compute_lake_mask(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """Return True where the grid cell centre lies on a lake surface, with caching."""
    key = _grid_signature(lon2d, lat2d)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _compute_land_and_lake_masks(lon2d, lat2d)
    _, lake_mask = _MASK_CACHE[key]
    return lake_mask

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
