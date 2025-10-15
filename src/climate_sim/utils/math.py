"""Lightweight numerical helpers shared across model components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the element-wise harmonic mean for positive arrays."""

    denom = np.zeros_like(a)
    valid = (a > 0.0) & (b > 0.0)
    denom[valid] = (1.0 / a[valid]) + (1.0 / b[valid])

    result = np.zeros_like(a)
    valid_denom = valid & (denom > 0.0)
    result[valid_denom] = 2.0 / denom[valid_denom]
    return result


def _ensure_strictly_increasing(values: np.ndarray, name: str) -> None:
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing for a regular grid")


def regular_latitude_edges(lat_centers_deg: np.ndarray) -> np.ndarray:
    """Infer latitude edges for a regularly spaced grid of cell centres."""

    if lat_centers_deg.ndim != 1:
        raise ValueError("Latitude centres must be a one-dimensional array")

    nlat = lat_centers_deg.size
    if nlat == 0:
        raise ValueError("Latitude centres array must be non-empty")

    if nlat > 1:
        _ensure_strictly_increasing(lat_centers_deg, "Latitude centres")
        spacing = np.diff(lat_centers_deg)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("Latitude grid must have constant spacing")
        delta = float(spacing[0])
    else:
        delta = 180.0

    edges = np.empty(nlat + 1, dtype=float)
    edges[1:-1] = 0.5 * (lat_centers_deg[:-1] + lat_centers_deg[1:])
    edges[0] = lat_centers_deg[0] - 0.5 * delta
    edges[-1] = lat_centers_deg[-1] + 0.5 * delta

    edges[0] = max(edges[0], -90.0)
    edges[-1] = min(edges[-1], 90.0)
    return edges


def regular_longitude_edges(lon_centers_deg: np.ndarray) -> np.ndarray:
    """Infer longitude edges for a regularly spaced, wrapped grid."""

    if lon_centers_deg.ndim != 1:
        raise ValueError("Longitude centres must be a one-dimensional array")

    nlon = lon_centers_deg.size
    if nlon == 0:
        raise ValueError("Longitude centres array must be non-empty")

    if nlon > 1:
        spacing = np.diff(lon_centers_deg)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("Longitude grid must have constant spacing")
        delta = float(spacing[0])
    else:
        delta = 360.0

    edges = np.empty(nlon + 1, dtype=float)
    edges[1:-1] = 0.5 * (lon_centers_deg[:-1] + lon_centers_deg[1:])
    edges[0] = lon_centers_deg[0] - 0.5 * delta
    edges[-1] = lon_centers_deg[-1] + 0.5 * delta
    return edges


@dataclass(slots=True)
class _GridGeometryCacheEntry:
    """Reusable geometry terms for a regularly spaced spherical grid."""

    delta_sin: np.ndarray
    delta_lon: np.ndarray
    lat_edges_rad: np.ndarray
    delta_cell_lat_rad: np.ndarray
    lat_centers_rad: np.ndarray
    delta_centers_rad: np.ndarray
    cos_lat_centers: np.ndarray
    cos_interface_lat: np.ndarray
    base_area: np.ndarray
    cached_radius: float | None = field(default=None)
    cached_area: np.ndarray | None = field(default=None)


_GRID_GEOMETRY_CACHE: dict[Tuple[Tuple[int, float, float, str], Tuple[int, float, float, str]], _GridGeometryCacheEntry] = {}


def _regular_grid_signature(centers: np.ndarray) -> Tuple[int, float, float, str]:
    """Return a lightweight signature for a 1-D regularly spaced grid."""

    size = centers.size
    if size == 0:
        raise ValueError("Grid centre arrays must be non-empty")

    start = float(np.round(float(centers[0]), 12))
    if size > 1:
        step = float(np.round(float(centers[1] - centers[0]), 12))
    else:
        step = 0.0
    return size, start, step, centers.dtype.str


def _grid_geometry_from_cache(
    lat_centers: np.ndarray, lon_centers: np.ndarray
) -> _GridGeometryCacheEntry:
    """Fetch or create cached geometry terms for the given regular grid."""

    key = (_regular_grid_signature(lat_centers), _regular_grid_signature(lon_centers))
    cached = _GRID_GEOMETRY_CACHE.get(key)
    if cached is None:
        lat_edges = regular_latitude_edges(lat_centers)
        lon_edges = regular_longitude_edges(lon_centers)

        lat_edges_rad = np.deg2rad(lat_edges)
        lon_edges_rad = np.deg2rad(lon_edges)

        delta_sin = np.sin(lat_edges_rad[1:]) - np.sin(lat_edges_rad[:-1])
        delta_lon = lon_edges_rad[1:] - lon_edges_rad[:-1]
        delta_cell_lat_rad = lat_edges_rad[1:] - lat_edges_rad[:-1]

        lat_centers_rad = np.deg2rad(lat_centers)
        delta_centers_rad = np.diff(lat_centers_rad)
        cos_lat_centers = np.cos(lat_centers_rad)
        cos_interface_lat = np.cos(lat_edges_rad[1:-1])

        base_area = delta_sin[:, np.newaxis] * delta_lon[np.newaxis, :]

        for array in (
            delta_sin,
            delta_lon,
            lat_edges_rad,
            delta_cell_lat_rad,
            lat_centers_rad,
            delta_centers_rad,
            cos_lat_centers,
            cos_interface_lat,
            base_area,
        ):
            array.setflags(write=False)

        cached = _GridGeometryCacheEntry(
            delta_sin=delta_sin,
            delta_lon=delta_lon,
            lat_edges_rad=lat_edges_rad,
            delta_cell_lat_rad=delta_cell_lat_rad,
            lat_centers_rad=lat_centers_rad,
            delta_centers_rad=delta_centers_rad,
            cos_lat_centers=cos_lat_centers,
            cos_interface_lat=cos_interface_lat,
            base_area=base_area,
        )
        _GRID_GEOMETRY_CACHE[key] = cached

    return cached


def spherical_cell_area(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    earth_radius_m: float,
) -> np.ndarray:
    """Return the physical surface area of each lon/lat grid cell.

    The returned array is cached for reuse on subsequent calls with the same
    grid and Earth radius. It is marked read-only so callers should treat it as
    immutable.
    """

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    nlat, nlon = lon2d.shape
    if nlat < 1 or nlon < 1:
        raise ValueError("Longitude/latitude grids must be non-empty")

    lat_centers = lat2d[:, 0]
    lon_centers = lon2d[0, :]

    geometry = _grid_geometry_from_cache(lat_centers, lon_centers)

    radius = float(earth_radius_m)
    if geometry.cached_area is not None and geometry.cached_radius == radius:
        return geometry.cached_area

    area = (radius**2) * geometry.base_area
    area.setflags(write=False)
    geometry.cached_radius = radius
    geometry.cached_area = area

    return area


def spherical_meridional_metrics(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    earth_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary lengths and centre spacing for meridional diffusion."""

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    nlat, _ = lon2d.shape
    if nlat < 2:
        return np.zeros((0, lon2d.shape[1]), dtype=float), np.zeros(0, dtype=float)

    geometry = _grid_geometry_from_cache(lat2d[:, 0], lon2d[0, :])

    boundary_length_north = (
        earth_radius_m
        * geometry.cos_interface_lat[:, np.newaxis]
        * geometry.delta_lon[np.newaxis, :]
    )
    delta_y = earth_radius_m * geometry.delta_centers_rad

    return boundary_length_north, delta_y


def spherical_zonal_metrics(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    earth_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary lengths and centre spacing for zonal diffusion."""

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    _, nlon = lon2d.shape
    if nlon < 2:
        zero = np.zeros_like(lon2d, dtype=float)
        return zero, zero

    geometry = _grid_geometry_from_cache(lat2d[:, 0], lon2d[0, :])

    boundary_length_east = (
        earth_radius_m * geometry.delta_cell_lat_rad[:, np.newaxis]
    )
    delta_x = (
        earth_radius_m
        * geometry.cos_lat_centers[:, np.newaxis]
        * geometry.delta_lon[np.newaxis, :]
    )

    return boundary_length_east, delta_x

