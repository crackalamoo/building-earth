"""Meridional diffusion utilities for lateral energy transport on the climate grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from climate_sim.utils.math import (
    harmonic_mean,
    regular_latitude_edges,
    regular_longitude_edges,
    spherical_cell_area,
)


def _assemble_sparse_matrix(
    north_coeff: np.ndarray,
    south_coeff: np.ndarray,
    east_coeff: np.ndarray,
    west_coeff: np.ndarray,
    diagonal: np.ndarray,
) -> sparse.csr_matrix | None:
    """Construct the sparse linear operator matching the diffusion tendency."""

    nlat, nlon = diagonal.shape
    size = nlat * nlon
    if size == 0:
        return sparse.csr_matrix((0, 0))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_entry(row: int, col: int, value: float) -> None:
        if value == 0.0:
            return
        rows.append(row)
        cols.append(col)
        data.append(value)

    for lat_idx in range(nlat):
        for lon_idx in range(nlon):
            row_index = lat_idx * nlon + lon_idx
            add_entry(row_index, row_index, diagonal[lat_idx, lon_idx])

            if lat_idx + 1 < nlat:
                coeff = north_coeff[lat_idx, lon_idx]
                if coeff != 0.0:
                    neighbor_index = (lat_idx + 1) * nlon + lon_idx
                    add_entry(row_index, neighbor_index, coeff)

            if lat_idx > 0:
                coeff = south_coeff[lat_idx, lon_idx]
                if coeff != 0.0:
                    neighbor_index = (lat_idx - 1) * nlon + lon_idx
                    add_entry(row_index, neighbor_index, coeff)

            coeff = east_coeff[lat_idx, lon_idx]
            if coeff != 0.0:
                neighbor_index = lat_idx * nlon + ((lon_idx + 1) % nlon)
                add_entry(row_index, neighbor_index, coeff)

            coeff = west_coeff[lat_idx, lon_idx]
            if coeff != 0.0:
                neighbor_index = lat_idx * nlon + ((lon_idx - 1) % nlon)
                add_entry(row_index, neighbor_index, coeff)

    if not rows:
        return sparse.csr_matrix((size, size))

    return sparse.csr_matrix((data, (rows, cols)), shape=(size, size))


@dataclass(frozen=True)
class DiffusionConfig:
    """Container for lateral diffusion parameters."""

    earth_radius_m: float = 6.371e6
    meridional_diffusivity_m2_s: float = 5.0e7
    enabled: bool = True
    use_spherical_geometry: bool = True


@dataclass
class DiffusionOperator:
    """Precomputed discrete diffusion operator for the solver grid."""

    north_coeff: np.ndarray
    south_coeff: np.ndarray
    east_coeff: np.ndarray
    west_coeff: np.ndarray
    diagonal: np.ndarray
    matrix: sparse.csr_matrix | None

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
            matrix=None,
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

    earth_radius = config.earth_radius_m
    use_geometry = config.use_spherical_geometry

    total_capacity = np.zeros_like(heat_capacity_field)
    total_capacity[active_mask] = (
        heat_capacity_field[active_mask] * cell_area_field[active_mask]
    )
    inv_capacity = np.zeros_like(total_capacity)
    inv_capacity[active_mask] = 1.0 / total_capacity[active_mask]

    diffusivity_field = np.zeros_like(heat_capacity_field)
    diffusivity_field[active_mask] = (
        config.meridional_diffusivity_m2_s * heat_capacity_field[active_mask]
    )

    lat_centers = lat2d[:, 0]
    lon_centers = lon2d[0, :]
    lat_edges = regular_latitude_edges(lat_centers)
    lon_edges = regular_longitude_edges(lon_centers)

    lat_edges_rad = np.deg2rad(lat_edges)
    lon_edges_rad = np.deg2rad(lon_edges)

    delta_lon_rad = lon_edges_rad[1:] - lon_edges_rad[:-1]
    if np.any(delta_lon_rad <= 0.0):
        raise ValueError("Longitude edges must be strictly increasing")

    north_coeff = np.zeros_like(heat_capacity_field)
    south_coeff = np.zeros_like(heat_capacity_field)
    east_coeff = np.zeros_like(heat_capacity_field)
    west_coeff = np.zeros_like(heat_capacity_field)

    if nlat > 1:
        delta_lat_centers = np.diff(lat_centers)
        if np.any(delta_lat_centers <= 0.0):
            raise ValueError("Latitude centres must be strictly increasing")

        delta_lat_centers_rad = np.deg2rad(delta_lat_centers)
        delta_y = earth_radius * delta_lat_centers_rad

        interface_lat_rad = np.deg2rad(0.5 * (lat_centers[:-1] + lat_centers[1:]))
        if use_geometry:
            boundary_length_north = (
                earth_radius
                * np.cos(interface_lat_rad)[:, np.newaxis]
                * delta_lon_rad[np.newaxis, :]
            )
        else:
            if lon_centers.size > 1:
                delta_lon_uniform = float(np.deg2rad(lon_centers[1] - lon_centers[0]))
            else:
                delta_lon_uniform = 2.0 * np.pi
            boundary_length_north = np.full(
                (nlat - 1, nlon), earth_radius * delta_lon_uniform, dtype=float
            )

        north_mask = active_mask[:-1] & active_mask[1:]
        north_diffusivity = harmonic_mean(
            diffusivity_field[:-1], diffusivity_field[1:]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            north_conductance = (
                north_diffusivity * boundary_length_north / delta_y[:, np.newaxis]
            )
        north_conductance = np.where(north_mask, north_conductance, 0.0)

        north_coeff[:-1] = north_conductance * inv_capacity[:-1]
        south_coeff[1:] = north_conductance * inv_capacity[1:]

    # Meridional-only diffusion: retain zero zonal coefficients so polar rows have three neighbors.

    diagonal = -(north_coeff + south_coeff)
    diagonal = np.where(active_mask, diagonal, 0.0)

    matrix = _assemble_sparse_matrix(
        north_coeff,
        south_coeff,
        east_coeff,
        west_coeff,
        diagonal,
    )

    return DiffusionOperator(
        north_coeff=north_coeff,
        south_coeff=south_coeff,
        east_coeff=east_coeff,
        west_coeff=west_coeff,
        diagonal=diagonal,
        matrix=matrix,
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

    if config.use_spherical_geometry:
        cell_area_field = spherical_cell_area(
            lon2d,
            lat2d,
            earth_radius_m=config.earth_radius_m,
        )
    else:
        lat_centers = lat2d[:, 0]
        lon_centers = lon2d[0, :]
        if lat_centers.size > 1:
            delta_lat_deg = float(lat_centers[1] - lat_centers[0])
        else:
            delta_lat_deg = 180.0
        if lon_centers.size > 1:
            delta_lon_deg = float(lon_centers[1] - lon_centers[0])
        else:
            delta_lon_deg = 360.0
        delta_lat_rad = np.deg2rad(delta_lat_deg)
        delta_lon_rad = np.deg2rad(delta_lon_deg)
        area = (config.earth_radius_m**2) * delta_lat_rad * delta_lon_rad
        cell_area_field = np.full_like(heat_capacity_field, area, dtype=float)

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
