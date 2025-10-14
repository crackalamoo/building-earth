"""Diffusion utilities for lateral energy transport on the climate grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.utils.math import harmonic_mean


@dataclass(frozen=True)
class DiffusionConfig:
    """Container for lateral diffusion parameters."""

    earth_radius_m: float = 6.371e6
    meridional_diffusivity_m2_s: float = 5.0e7
    enabled: bool = True


@dataclass
class DiffusionOperator:
    """Precomputed discrete diffusion operator for the solver grid."""

    north_coeff: np.ndarray
    south_coeff: np.ndarray
    east_coeff: np.ndarray
    west_coeff: np.ndarray
    diagonal: np.ndarray

    enabled: bool = True

    @classmethod
    def disabled(cls, shape: tuple[int, int]) -> DiffusionOperator:
        zeros = np.zeros(shape, dtype=float)
        return cls(
            north_coeff=zeros,
            south_coeff=zeros,
            east_coeff=zeros,
            west_coeff=zeros,
            diagonal=zeros,
            enabled=False,
        )

    def tendency(self, temperature: np.ndarray) -> np.ndarray:
        """Return the diffusion tendency for the provided temperature field."""
        if not self.enabled:
            return np.zeros_like(temperature)
        north_term = self.north_coeff * (np.roll(temperature, -1, axis=0) - temperature)
        south_term = self.south_coeff * (np.roll(temperature, 1, axis=0) - temperature)
        east_term = self.east_coeff * (np.roll(temperature, -1, axis=1) - temperature)
        west_term = self.west_coeff * (np.roll(temperature, 1, axis=1) - temperature)

        if temperature.shape[0] > 1:
            north_term[-1, :] = 0.0
            south_term[0, :] = 0.0

        return north_term + south_term + east_term + west_term


@dataclass
class LayeredDiffusionOperator:
    """Diffusion operators for the surface ocean and atmosphere layers."""

    surface: DiffusionOperator
    atmosphere: DiffusionOperator

    @classmethod
    def disabled(cls, shape: tuple[int, int]) -> LayeredDiffusionOperator:
        disabled = DiffusionOperator.disabled(shape)
        return cls(surface=disabled, atmosphere=disabled)


def _build_single_layer_operator(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    heat_capacity_field: np.ndarray,
    *,
    config: DiffusionConfig,
) -> DiffusionOperator:
    if lon2d.shape != lat2d.shape or lon2d.shape != heat_capacity_field.shape:
        raise ValueError("Grid and heat capacity field must share the same shape")

    if not config.enabled:
        return DiffusionOperator.disabled(heat_capacity_field.shape)

    nlat, nlon = lon2d.shape

    delta_lat_deg = float(abs(lat2d[1, 0] - lat2d[0, 0])) if nlat > 1 else 0.0
    delta_lon_deg = float(abs(lon2d[0, 1] - lon2d[0, 0])) if nlon > 1 else 0.0

    delta_lat_rad = np.deg2rad(delta_lat_deg)
    delta_lon_rad = np.deg2rad(delta_lon_deg)

    earth_radius = config.earth_radius_m

    delta_lat_m = earth_radius * delta_lat_rad if delta_lat_rad > 0.0 else np.inf
    if delta_lon_rad > 0.0:
        cos_lat = np.cos(np.deg2rad(lat2d))
        delta_lon_m = earth_radius * cos_lat * delta_lon_rad
    else:
        delta_lon_m = np.full_like(heat_capacity_field, np.inf, dtype=float)

    lat_diffusivity = (
        config.meridional_diffusivity_m2_s / (heat_capacity_field * delta_lat_m**2)
        if np.isfinite(delta_lat_m)
        else np.zeros_like(heat_capacity_field)
    )
    lon_diffusivity = np.zeros_like(heat_capacity_field)
    if delta_lon_rad > 0.0:
        valid = np.abs(delta_lon_m) > 0.0
        lon_diffusivity[valid] = (
            config.meridional_diffusivity_m2_s
            / (heat_capacity_field[valid] * delta_lon_m[valid] ** 2)
        )

    if nlat > 1 and np.isfinite(delta_lat_m):
        north_coeff = harmonic_mean(lat_diffusivity, np.roll(lat_diffusivity, -1, axis=0))
        north_coeff[-1, :] = 0.0
        south_coeff = harmonic_mean(lat_diffusivity, np.roll(lat_diffusivity, 1, axis=0))
        south_coeff[0, :] = 0.0
    else:
        north_coeff = np.zeros_like(heat_capacity_field)
        south_coeff = np.zeros_like(heat_capacity_field)

    if nlon > 1 and delta_lon_rad > 0.0:
        east_coeff = harmonic_mean(lon_diffusivity, np.roll(lon_diffusivity, -1, axis=1))
        west_coeff = harmonic_mean(lon_diffusivity, np.roll(lon_diffusivity, 1, axis=1))
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


def create_diffusion_operator(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    heat_capacity_field: np.ndarray,
    *,
    land_mask: np.ndarray | None = None,
    atmosphere_heat_capacity: float | None = None,
    config: DiffusionConfig | None = None,
) -> LayeredDiffusionOperator:
    """Create diffusion operators for the surface ocean and atmosphere layers."""

    if lon2d.shape != lat2d.shape or lon2d.shape != heat_capacity_field.shape:
        raise ValueError("Grid and heat capacity field must share the same shape")

    config = config or DiffusionConfig()
    if not config.enabled:
        return LayeredDiffusionOperator.disabled(heat_capacity_field.shape)

    if land_mask is None:
        land_mask = np.zeros_like(heat_capacity_field, dtype=bool)

    if land_mask.shape != heat_capacity_field.shape:
        raise ValueError("Land mask must share the same shape as the heat capacity field")

    surface_heat_capacity = np.array(heat_capacity_field, copy=True)
    surface_heat_capacity[land_mask] = np.inf

    if atmosphere_heat_capacity is None:
        atmosphere_heat_capacity = 1.0

    atmosphere_heat_capacity_field = np.full_like(
        heat_capacity_field, atmosphere_heat_capacity, dtype=float
    )

    surface_operator = _build_single_layer_operator(
        lon2d, lat2d, surface_heat_capacity, config=config
    )
    atmosphere_operator = _build_single_layer_operator(
        lon2d, lat2d, atmosphere_heat_capacity_field, config=config
    )

    return LayeredDiffusionOperator(
        surface=surface_operator,
        atmosphere=atmosphere_operator,
    )
