"""Diffusion utilities for lateral energy transport on the climate grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.utils.math import (
    harmonic_mean,
    meridional_boundary_length,
    spherical_cell_area,
    zonal_boundary_length,
)


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
    cell_area_field: np.ndarray,
    *,
    config: DiffusionConfig,
    active_mask: np.ndarray | None = None,
) -> DiffusionOperator:
    if lon2d.shape != lat2d.shape or lon2d.shape != heat_capacity_field.shape:
        raise ValueError("Grid and heat capacity field must share the same shape")

    if cell_area_field.shape != heat_capacity_field.shape:
        raise ValueError("Cell area field must match the grid shape")

    if not config.enabled:
        return DiffusionOperator.disabled(heat_capacity_field.shape)

    nlat, nlon = lon2d.shape

    if active_mask is None:
        active_mask = np.ones_like(heat_capacity_field, dtype=bool)
    elif active_mask.shape != heat_capacity_field.shape:
        raise ValueError("Active mask must match the grid shape")

    total_heat_capacity = heat_capacity_field * cell_area_field
    safe_capacity = np.where(active_mask, total_heat_capacity, 1.0)
    diffusivity_field = np.where(
        active_mask,
        config.meridional_diffusivity_m2_s * heat_capacity_field,
        0.0,
    )

    delta_lat_deg = float(abs(lat2d[1, 0] - lat2d[0, 0])) if nlat > 1 else 0.0
    delta_lon_deg = float(abs(lon2d[0, 1] - lon2d[0, 0])) if nlon > 1 else 0.0

    delta_lat_rad = np.deg2rad(delta_lat_deg)
    delta_lon_rad = np.deg2rad(delta_lon_deg)

    earth_radius = config.earth_radius_m

    delta_lat_m = earth_radius * delta_lat_rad if delta_lat_rad > 0.0 else np.inf

    north_coeff = np.zeros_like(heat_capacity_field)
    south_coeff = np.zeros_like(heat_capacity_field)
    east_coeff = np.zeros_like(heat_capacity_field)
    west_coeff = np.zeros_like(heat_capacity_field)

    if nlat > 1 and np.isfinite(delta_lat_m):
        inv_delta_lat = 1.0 / delta_lat_m
        lat_rad = np.deg2rad(lat2d)
        north_lat_interface = 0.5 * (lat_rad + np.roll(lat_rad, -1, axis=0))
        boundary_length_north = meridional_boundary_length(
            north_lat_interface,
            earth_radius_m=earth_radius,
            delta_lon_rad=delta_lon_rad,
        )
        neighbor_mask_north = active_mask & np.roll(active_mask, -1, axis=0)
        north_diffusivity = harmonic_mean(
            diffusivity_field, np.roll(diffusivity_field, -1, axis=0)
        )
        north_conductance = north_diffusivity * boundary_length_north * inv_delta_lat
        north_coeff = np.where(
            neighbor_mask_north,
            north_conductance / safe_capacity,
            0.0,
        )
        north_coeff[-1, :] = 0.0

        south_lat_interface = 0.5 * (lat_rad + np.roll(lat_rad, 1, axis=0))
        boundary_length_south = meridional_boundary_length(
            south_lat_interface,
            earth_radius_m=earth_radius,
            delta_lon_rad=delta_lon_rad,
        )
        neighbor_mask_south = active_mask & np.roll(active_mask, 1, axis=0)
        south_diffusivity = harmonic_mean(
            diffusivity_field, np.roll(diffusivity_field, 1, axis=0)
        )
        south_conductance = south_diffusivity * boundary_length_south * inv_delta_lat
        south_coeff = np.where(
            neighbor_mask_south,
            south_conductance / safe_capacity,
            0.0,
        )
        south_coeff[0, :] = 0.0

    if nlon > 1 and delta_lon_rad > 0.0:
        lat_rad = np.deg2rad(lat2d)
        delta_lon_m = earth_radius * np.cos(lat_rad) * delta_lon_rad
        valid_lon = np.abs(delta_lon_m) > 1e-12
        inv_delta_lon = np.zeros_like(delta_lon_m)
        inv_delta_lon[valid_lon] = 1.0 / delta_lon_m[valid_lon]
        boundary_length_east = zonal_boundary_length(
            earth_radius_m=earth_radius,
            delta_lat_rad=delta_lat_rad,
            shape=heat_capacity_field.shape,
        )

        neighbor_mask_east = active_mask & np.roll(active_mask, -1, axis=1)
        east_diffusivity = harmonic_mean(
            diffusivity_field, np.roll(diffusivity_field, -1, axis=1)
        )
        east_conductance = east_diffusivity * boundary_length_east * inv_delta_lon
        east_coeff = np.where(
            neighbor_mask_east & valid_lon,
            east_conductance / safe_capacity,
            0.0,
        )

        neighbor_mask_west = active_mask & np.roll(active_mask, 1, axis=1)
        west_diffusivity = harmonic_mean(
            diffusivity_field, np.roll(diffusivity_field, 1, axis=1)
        )
        west_conductance = west_diffusivity * boundary_length_east * inv_delta_lon
        west_coeff = np.where(
            neighbor_mask_west & valid_lon,
            west_conductance / safe_capacity,
            0.0,
        )

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

    if atmosphere_heat_capacity is None:
        atmosphere_heat_capacity = 1.0

    atmosphere_heat_capacity_field = np.full_like(
        heat_capacity_field, atmosphere_heat_capacity, dtype=float
    )

    cell_area_field = spherical_cell_area(
        lon2d,
        lat2d,
        earth_radius_m=config.earth_radius_m,
    )

    surface_operator = _build_single_layer_operator(
        lon2d,
        lat2d,
        heat_capacity_field,
        cell_area_field,
        config=config,
        active_mask=~land_mask,
    )
    atmosphere_operator = _build_single_layer_operator(
        lon2d,
        lat2d,
        atmosphere_heat_capacity_field,
        cell_area_field,
        config=config,
    )

    return LayeredDiffusionOperator(
        surface=surface_operator,
        atmosphere=atmosphere_operator,
    )
