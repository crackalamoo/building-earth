"""Diffusion utilities for lateral energy transport on the climate grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6.371e6
MERIDIONAL_DIFFUSIVITY_M2_S = 5.0e7


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise harmonic mean guarding against zero coefficients."""
    denom = np.zeros_like(a)
    valid = (a > 0.0) & (b > 0.0)
    denom[valid] = (1.0 / a[valid]) + (1.0 / b[valid])
    result = np.zeros_like(a)
    valid_denom = valid & (denom > 0.0)
    result[valid_denom] = 2.0 / denom[valid_denom]
    return result


@dataclass
class DiffusionOperator:
    """Precomputed discrete diffusion operator for the solver grid."""

    north_coeff: np.ndarray
    south_coeff: np.ndarray
    east_coeff: np.ndarray
    west_coeff: np.ndarray
    diagonal: np.ndarray

    def tendency(self, temperature: np.ndarray) -> np.ndarray:
        """Return the diffusion tendency for the provided temperature field."""
        north_term = self.north_coeff * (np.roll(temperature, -1, axis=0) - temperature)
        south_term = self.south_coeff * (np.roll(temperature, 1, axis=0) - temperature)
        east_term = self.east_coeff * (np.roll(temperature, -1, axis=1) - temperature)
        west_term = self.west_coeff * (np.roll(temperature, 1, axis=1) - temperature)

        if temperature.shape[0] > 1:
            north_term[-1, :] = 0.0
            south_term[0, :] = 0.0

        return north_term + south_term + east_term + west_term


def create_diffusion_operator(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    heat_capacity_field: np.ndarray,
    *,
    diffusivity_m2_s: float = MERIDIONAL_DIFFUSIVITY_M2_S,
) -> DiffusionOperator:
    """Create a discrete diffusion operator for the supplied grid and materials."""
    if lon2d.shape != lat2d.shape or lon2d.shape != heat_capacity_field.shape:
        raise ValueError("Grid and heat capacity field must share the same shape")

    nlat, nlon = lon2d.shape

    delta_lat_deg = float(abs(lat2d[1, 0] - lat2d[0, 0])) if nlat > 1 else 0.0
    delta_lon_deg = float(abs(lon2d[0, 1] - lon2d[0, 0])) if nlon > 1 else 0.0

    delta_lat_rad = np.deg2rad(delta_lat_deg)
    delta_lon_rad = np.deg2rad(delta_lon_deg)

    delta_lat_m = EARTH_RADIUS_M * delta_lat_rad if delta_lat_rad > 0.0 else np.inf
    delta_lon_m = EARTH_RADIUS_M * delta_lon_rad if delta_lon_rad > 0.0 else np.inf

    lat_diffusivity = (
        diffusivity_m2_s / (heat_capacity_field * delta_lat_m**2)
        if np.isfinite(delta_lat_m)
        else np.zeros_like(heat_capacity_field)
    )
    lon_diffusivity = (
        diffusivity_m2_s / (heat_capacity_field * delta_lon_m**2)
        if np.isfinite(delta_lon_m)
        else np.zeros_like(heat_capacity_field)
    )

    if nlat > 1 and np.isfinite(delta_lat_m):
        north_coeff = _harmonic_mean(lat_diffusivity, np.roll(lat_diffusivity, -1, axis=0))
        north_coeff[-1, :] = 0.0
        south_coeff = _harmonic_mean(lat_diffusivity, np.roll(lat_diffusivity, 1, axis=0))
        south_coeff[0, :] = 0.0
    else:
        north_coeff = np.zeros_like(heat_capacity_field)
        south_coeff = np.zeros_like(heat_capacity_field)

    if nlon > 1 and np.isfinite(delta_lon_m):
        east_coeff = _harmonic_mean(lon_diffusivity, np.roll(lon_diffusivity, -1, axis=1))
        west_coeff = _harmonic_mean(lon_diffusivity, np.roll(lon_diffusivity, 1, axis=1))
    else:
        east_coeff = np.zeros_like(heat_capacity_field)
        west_coeff = np.zeros_like(heat_capacity_field)

    diagonal = -(north_coeff + south_coeff + east_coeff + west_coeff)

    return DiffusionOperator(
        north_coeff=north_coeff,
        south_coeff=south_coeff,
        east_coeff=east_coeff,
        west_coeff=west_coeff,
        diagonal=diagonal,
    )
