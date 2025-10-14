"""Lightweight numerical helpers shared across model components."""

from __future__ import annotations

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


def spherical_cell_area(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    earth_radius_m: float,
) -> np.ndarray:
    """Return the physical surface area of each lon/lat grid cell."""

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    nlat, nlon = lon2d.shape
    if nlat < 1 or nlon < 1:
        raise ValueError("Longitude/latitude grids must be non-empty")

    lat_centers = lat2d[:, 0]
    lon_centers = lon2d[0, :]

    lat_edges = regular_latitude_edges(lat_centers)
    lon_edges = regular_longitude_edges(lon_centers)

    lat_edges_rad = np.deg2rad(lat_edges)
    lon_edges_rad = np.deg2rad(lon_edges)

    delta_sin = np.sin(lat_edges_rad[1:]) - np.sin(lat_edges_rad[:-1])
    delta_lon = lon_edges_rad[1:] - lon_edges_rad[:-1]

    area = (earth_radius_m**2) * delta_sin[:, np.newaxis] * delta_lon[np.newaxis, :]
    return area

