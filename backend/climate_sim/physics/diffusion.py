"""Lateral (meridional + zonal) diffusion utilities for energy transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from climate_sim.core.math_core import (
    harmonic_mean,
    regular_latitude_edges,
    regular_longitude_edges,
    spherical_cell_area,
)
from climate_sim.data.constants import R_EARTH_METERS


def _infer_spacing_deg(coordinates: np.ndarray, fallback: float) -> float:
    if coordinates.size <= 1:
        return fallback
    deltas = np.diff(coordinates)
    if np.any(deltas <= 0.0):
        raise ValueError("Grid coordinates must be strictly increasing")
    return float(np.mean(deltas))


def _infer_grid_resolution_deg(lon2d: np.ndarray, lat2d: np.ndarray) -> float:
    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")
    lat_centers = lat2d[:, 0]
    lon_centers = lon2d[0, :]
    delta_lat_deg = _infer_spacing_deg(lat_centers, fallback=180.0)
    delta_lon_deg = _infer_spacing_deg(lon_centers, fallback=360.0)
    if delta_lat_deg <= 0.0 or delta_lon_deg <= 0.0:
        raise ValueError("Grid resolution must be positive in both dimensions")
    return float(np.sqrt(delta_lat_deg * delta_lon_deg))


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

    rows_arr = np.asarray(rows, dtype=np.int64)
    cols_arr = np.asarray(cols, dtype=np.int64)
    data_arr = np.asarray(data, dtype=np.float64)

    coo = sparse.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(size, size))
    return coo.tocsr()


@dataclass(frozen=True)
class DiffusionConfig:
    # Ocean mesoscale eddy diffusivity: ~500-2000 m²/s typical
    # Reduced from 2000 since we now have explicit ocean currents for advective transport
    surface_kappa_ref_m2_s: float = 1.0e3

    # Ocean latitude-dependent scaling to represent thermohaline circulation
    # MOC transports ~2 PW poleward, requiring strong effective diffusivity
    use_latitude_dependent_surface: bool = False
    surface_meridional_tropical_scale: float = 0.5    # 0-30°: Weak cross-equatorial
    surface_meridional_midlat_scale: float = 1.5      # 30-60°: Western boundary currents
    surface_meridional_polar_scale: float = 1.5       # 60-90°: Strong MOC/deep convection

    # Atmospheric eddy diffusivity: represents baroclinic eddy transport.
    # Transient eddies carry 60-80% of midlat heat transport (~2-3e6 m²/s).
    # Tropics: eddies are ~5-10% of transport (Hadley dominates).
    atmosphere_kappa_ref_m2_s: float = 1.0e6

    enabled: bool = True

    # Latitude-dependent atmospheric eddy diffusivity scaling
    # Eddies are roughly isotropic — same scaling for meridional and zonal.
    # Mean wind transport (trade winds, westerlies) is handled by advection, not diffusion.
    use_latitude_dependent_atmosphere: bool = True

    # Eady-based scaling: κ ∝ f × |dT/dy| (baroclinic instability).
    # f ∝ |sin(φ)|, |dT/dy| ∝ |sin(2φ)| for sinusoidal T profile.
    # Naturally gives low κ in tropics (Hadley, not eddies) and subtropics
    # (weak gradient, high stability), peak in midlatitudes (storm track).
    eady_kappa_max_m2_s: float = 2.0e6  # Cap to prevent excessive polar transport

    # Scaling reference values used by Eady formula
    atmosphere_meridional_tropical_scale: float = 0.3   # Floor: minimum eddy mixing
    atmosphere_meridional_midlat_scale: float = 2.5     # Normalization: Eady scaling at 45°

    # Boundary layer diffusivity scaling relative to free atmosphere.
    # Same κ as atmosphere; the smaller heat capacity (C_bl/C_atm ≈ 1/7.5)
    # already ensures BL carries only ~12% of total diffusive transport.
    boundary_layer_diffusivity_scale: float = 1.0

    def surface_diffusivity(self, grid_resolution_deg: float) -> float:
        return self.surface_kappa_ref_m2_s

    def atmosphere_diffusivity(self, grid_resolution_deg: float) -> float:
        return self.atmosphere_kappa_ref_m2_s

    def compute_latitude_scaling(
        self,
        lat_deg: np.ndarray,
        *,
        meridional: bool,
        layer: str = "atmosphere",
    ) -> np.ndarray:
        """Compute latitude-dependent scaling factor for diffusivity.

        Args:
            lat_deg: Latitude values in degrees
            meridional: Whether to use meridional (N-S) or zonal (E-W) scaling
            layer: Which layer scaling to use ("atmosphere" or "surface")
        """
        if layer == "surface":
            if not self.use_latitude_dependent_surface:
                return np.ones_like(lat_deg)
            # Surface only has meridional scaling (ocean gyres don't enhance zonal)
            if meridional:
                tropical_scale = self.surface_meridional_tropical_scale
                midlat_scale = self.surface_meridional_midlat_scale
                polar_scale = self.surface_meridional_polar_scale
            else:
                return np.ones_like(lat_deg)
        else:  # atmosphere
            if not self.use_latitude_dependent_atmosphere:
                return np.ones_like(lat_deg)

            lat_rad = np.deg2rad(lat_deg)
            # Eady-based: κ ∝ f × |dT/dy|
            # f ∝ |sin(φ)|, |dT/dy| ∝ |sin(2φ)| for sinusoidal T profile
            eady = np.abs(np.sin(lat_rad)) * np.abs(np.sin(2 * lat_rad))
            # Normalize so 45° gives midlat_scale
            eady_at_45 = np.sin(np.deg2rad(45.0)) * np.sin(np.deg2rad(90.0))
            scaling = eady / eady_at_45 * self.atmosphere_meridional_midlat_scale
            # Floor: tropics still need some eddy transport
            scaling = np.maximum(scaling, self.atmosphere_meridional_tropical_scale)
            # Cap to prevent excessive polar transport
            if self.eady_kappa_max_m2_s > 0:
                max_scaling = self.eady_kappa_max_m2_s / self.atmosphere_kappa_ref_m2_s
                scaling = np.minimum(scaling, max_scaling)
            return scaling


@dataclass
class DiffusionOperator:
    """Precomputed discrete diffusion operator for the solver grid."""

    north_coeff: np.ndarray
    south_coeff: np.ndarray
    east_coeff: np.ndarray
    west_coeff: np.ndarray
    diagonal: np.ndarray
    matrix: sparse.csr_matrix | None
    off_diagonal_matrix: sparse.csr_matrix | None = None

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
            off_diagonal_matrix=None,
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

    def linearised_tendency(self) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        """Return diagonal and off-diagonal pieces of the diffusion Jacobian."""

        diag = np.zeros_like(self.diagonal)
        if not self.enabled:
            return diag, None

        diag = self.diagonal.copy()

        if self.matrix is None:
            self.matrix = _assemble_sparse_matrix(
                self.north_coeff,
                self.south_coeff,
                self.east_coeff,
                self.west_coeff,
                self.diagonal,
            )
            assert isinstance(self.matrix, sparse.csr_matrix)

        if self.off_diagonal_matrix is None:
            base_matrix = self.matrix
            if not sparse.isspmatrix_csr(base_matrix):
                base_matrix = base_matrix.tocsr()

            diag_matrix = sparse.diags(diag.ravel(), format="csr")
            off_diag = base_matrix - diag_matrix
            off_diag.eliminate_zeros()

            if off_diag.nnz == 0:
                self.off_diagonal_matrix = sparse.csr_matrix(base_matrix.shape)
            else:
                self.off_diagonal_matrix = off_diag

        return diag, self.off_diagonal_matrix


@dataclass
class LayeredDiffusionOperator:
    """Diffusion operators for the surface ocean and atmosphere layers."""

    surface: DiffusionOperator
    atmosphere: DiffusionOperator
    boundary_layer: DiffusionOperator | None = None

    @classmethod
    def disabled(cls, shape: tuple[int, int]) -> LayeredDiffusionOperator:
        disabled = DiffusionOperator.disabled(shape)
        return cls(surface=disabled, atmosphere=disabled, boundary_layer=disabled)


def _build_single_layer_operator(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    heat_capacity_field: np.ndarray,
    cell_area_field: np.ndarray,
    *,
    config: DiffusionConfig,
    active_mask: np.ndarray | None = None,
    diffusivity_m2_s: float,
    use_latitude_scaling: bool = False,
    layer: str = "atmosphere",
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

    earth_radius = R_EARTH_METERS

    total_capacity = np.zeros_like(heat_capacity_field)
    total_capacity[active_mask] = (
        heat_capacity_field[active_mask] * cell_area_field[active_mask]
    )
    inv_capacity = np.zeros_like(total_capacity)
    inv_capacity[active_mask] = 1.0 / total_capacity[active_mask]

    # Base diffusivity field (will be scaled by latitude if requested)
    diffusivity_field = np.zeros_like(heat_capacity_field)
    diffusivity_field[active_mask] = (
        diffusivity_m2_s * heat_capacity_field[active_mask]
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
            raise ValueError("Latitude centers must be strictly increasing")

        delta_lat_centers_rad = np.deg2rad(delta_lat_centers)
        delta_y = earth_radius * delta_lat_centers_rad

        interface_lat_rad = np.deg2rad(0.5 * (lat_centers[:-1] + lat_centers[1:]))
        interface_lat_deg = 0.5 * (lat_centers[:-1] + lat_centers[1:])
        boundary_length_north = (
            earth_radius
            * np.cos(interface_lat_rad)[:, np.newaxis]
            * delta_lon_rad[np.newaxis, :]
        )

        north_mask = active_mask[:-1] & active_mask[1:]
        north_diffusivity = harmonic_mean(
            diffusivity_field[:-1], diffusivity_field[1:]
        )

        # Apply latitude-dependent scaling for meridional diffusion
        if use_latitude_scaling:
            meridional_scaling = config.compute_latitude_scaling(
                interface_lat_deg, meridional=True, layer=layer
            )
            north_diffusivity = north_diffusivity * meridional_scaling[:, np.newaxis]

        with np.errstate(divide="ignore", invalid="ignore"):
            north_conductance = (
                north_diffusivity * boundary_length_north / delta_y[:, np.newaxis]
            )
        north_conductance = np.where(north_mask, north_conductance, 0.0)

        north_coeff[:-1] = north_conductance * inv_capacity[:-1]
        south_coeff[1:] = north_conductance * inv_capacity[1:]

    # Zonal diffusion (periodic in longitude)
    if nlon > 1:
        # Horizontal distance between centers: R cos(phi) * d(lambda)
        delta_x = (
            earth_radius
            * np.cos(np.deg2rad(lat_centers))[:, np.newaxis]
            * (lon_edges_rad[1:] - lon_edges_rad[:-1])[np.newaxis, :]
        )
        # Length of north-south faces: R * d(phi)
        boundary_length_east = (
            earth_radius
            * (lat_edges_rad[1:] - lat_edges_rad[:-1])[:, np.newaxis]
        )

        east_mask = active_mask & np.roll(active_mask, -1, axis=1)
        east_diffusivity = harmonic_mean(
            diffusivity_field, np.roll(diffusivity_field, -1, axis=1)
        )

        # Apply latitude-dependent scaling for zonal diffusion
        if use_latitude_scaling:
            zonal_scaling = config.compute_latitude_scaling(
                lat_centers, meridional=False, layer=layer
            )
            east_diffusivity = east_diffusivity * zonal_scaling[:, np.newaxis]

        with np.errstate(divide="ignore", invalid="ignore"):
            east_conductance = east_diffusivity * (boundary_length_east / delta_x)
        east_conductance = np.where(east_mask, east_conductance, 0.0)

        east_coeff = east_conductance * inv_capacity
        west_coeff = np.roll(east_conductance, 1, axis=1) * inv_capacity

    diagonal = -(north_coeff + south_coeff + east_coeff + west_coeff)
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
    boundary_layer_heat_capacity: float | None = None,
    config: DiffusionConfig | None = None,
) -> LayeredDiffusionOperator:
    """Create diffusion operators for the surface ocean and atmosphere layers."""

    if lon2d.shape != lat2d.shape or lon2d.shape != heat_capacity_field.shape:
        raise ValueError("Grid and heat capacity field must share the same shape")

    config = config or DiffusionConfig()
    if not config.enabled:
        return LayeredDiffusionOperator.disabled(heat_capacity_field.shape)

    grid_resolution_deg = _infer_grid_resolution_deg(lon2d, lat2d)
    surface_diffusivity_m2_s = config.surface_diffusivity(grid_resolution_deg)
    atmosphere_diffusivity_m2_s = config.atmosphere_diffusivity(grid_resolution_deg)

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
        earth_radius_m=R_EARTH_METERS,
    )

    surface_operator = _build_single_layer_operator(
        lon2d,
        lat2d,
        heat_capacity_field,
        cell_area_field,
        config=config,
        active_mask=~land_mask,
        diffusivity_m2_s=surface_diffusivity_m2_s,
        use_latitude_scaling=True,  # Surface ocean uses latitude-dependent diffusivity
        layer="surface",
    )
    atmosphere_operator = _build_single_layer_operator(
        lon2d,
        lat2d,
        atmosphere_heat_capacity_field,
        cell_area_field,
        config=config,
        diffusivity_m2_s=atmosphere_diffusivity_m2_s,
        use_latitude_scaling=True,  # Atmosphere uses latitude-dependent diffusivity
    )

    # Optionally build boundary layer operator with its own heat capacity
    boundary_layer_operator = None
    if boundary_layer_heat_capacity is not None:
        boundary_layer_heat_capacity_field = np.full_like(
            heat_capacity_field, boundary_layer_heat_capacity, dtype=float
        )
        # BL diffusivity is scaled down - BL transport is mostly vertical,
        # not the large-scale eddies that transport across latitudes
        boundary_layer_diffusivity = atmosphere_diffusivity_m2_s * config.boundary_layer_diffusivity_scale
        boundary_layer_operator = _build_single_layer_operator(
            lon2d,
            lat2d,
            boundary_layer_heat_capacity_field,
            cell_area_field,
            config=config,
            diffusivity_m2_s=boundary_layer_diffusivity,
            use_latitude_scaling=True,  # BL uses same latitude pattern as atmosphere
        )

    return LayeredDiffusionOperator(
        surface=surface_operator,
        atmosphere=atmosphere_operator,
        boundary_layer=boundary_layer_operator,
    )
