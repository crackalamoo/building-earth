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

    delta_lat_deg = float(abs(lat2d[1, 0] - lat2d[0, 0])) if nlat > 1 else 180.0
    delta_lon_deg = float(abs(lon2d[0, 1] - lon2d[0, 0])) if nlon > 1 else 360.0

    delta_lat_rad = np.deg2rad(delta_lat_deg)
    delta_lon_rad = np.deg2rad(delta_lon_deg)

    lat_rad = np.deg2rad(lat2d)
    cos_lat = np.clip(np.cos(lat_rad), 0.0, None)

    area = (earth_radius_m**2) * delta_lat_rad * delta_lon_rad * cos_lat
    return area


def meridional_boundary_length(
    lat_center_rad: np.ndarray,
    *,
    earth_radius_m: float,
    delta_lon_rad: float,
) -> np.ndarray:
    """Compute east-west boundary length for north/south interfaces."""

    cos_lat = np.clip(np.cos(lat_center_rad), 0.0, None)
    return earth_radius_m * delta_lon_rad * cos_lat


def zonal_boundary_length(
    *, earth_radius_m: float, delta_lat_rad: float, shape: tuple[int, int]
) -> np.ndarray:
    """Compute north-south boundary length for east/west interfaces."""

    return np.full(shape, earth_radius_m * delta_lat_rad, dtype=float)

