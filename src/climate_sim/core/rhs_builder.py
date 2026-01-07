"""Right-hand-side factory for energy-balance model physics assembly."""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from dataclasses import dataclass

import climate_sim.physics.radiation as radiation
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeModel, SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeModel, LatentHeatExchangeConfig
from climate_sim.physics.atmosphere.boundary_layer import BoundaryLayerConfig
from climate_sim.physics.atmosphere.wind import WindModel
from climate_sim.physics.atmosphere.advection import AdvectionOperator
from climate_sim.physics.diffusion import LayeredDiffusionOperator
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.core.state import ModelState
from climate_sim.core.math_core import spherical_cell_area
from climate_sim.data.constants import R_EARTH_METERS
from typing import Callable

@dataclass
class Linearization:
    diag: np.ndarray
    cross: np.ndarray | None = None
    surface_diffusion_matrix: sparse.csc_matrix | None = None
    atmosphere_diffusion_matrix: sparse.csc_matrix | None = None
    atmosphere_advection_matrix: sparse.csc_matrix | None = None
    boundary_layer_diffusion_matrix: sparse.csc_matrix | None = None
    boundary_layer_advection_matrix: sparse.csc_matrix | None = None
    solver_fingerprint: str | None = None

type FloatArray = NDArray[np.floating]

type RhsFn = Callable[[ModelState, FloatArray, float | FloatArray], FloatArray]
type RhsDerivativeFn = Callable[[ModelState, FloatArray], Linearization]

@dataclass(frozen=True, slots=True)
class RhsBuildInputs:
    """Structured inputs for RHS factory functions."""
    heat_capacity_field: FloatArray
    diffusion_operator: LayeredDiffusionOperator
    radiation_config: RadiationConfig
    land_mask: FloatArray
    roughness_length: FloatArray
    topographic_elevation: FloatArray
    sensible_heat_cfg: SensibleHeatExchangeConfig
    latent_heat_cfg: LatentHeatExchangeConfig
    boundary_layer_cfg: BoundaryLayerConfig
    wind_model: WindModel | None
    advection_operator: AdvectionOperator | None
    lon2d: FloatArray
    lat2d: FloatArray

def create_rhs_functions(inputs: RhsBuildInputs) -> tuple[RhsFn, RhsDerivativeFn]:
    """Build RHS and Jacobian functions from physics configuration.

    This factory assembles the right-hand-side tendency function and its
    temperature derivative (Jacobian) from the provided physics operators.

    Parameters
    ----------
    inputs : RhsBuildInputs
        Structured inputs containing all physics operators and configuration.

    Returns
    -------
    tuple[RhsFn, RhsDerivativeFn]
        The RHS function and its derivative for use in the solver.
    """
    # Extract and linearize diffusion operators
    surface_diffusion_diag: np.ndarray | None = None
    surface_matrix = None
    if inputs.diffusion_operator.surface.enabled:
        surface_diffusion_diag, surface_offdiag = (
            inputs.diffusion_operator.surface.linearised_tendency()
        )
        if surface_offdiag is not None and surface_offdiag.nnz > 0:
            surface_matrix = (
                surface_offdiag
                if sparse.isspmatrix_csc(surface_offdiag)
                else surface_offdiag.tocsc()
            )

    atmosphere_diffusion_diag: np.ndarray | None = None
    atmosphere_matrix = None
    if inputs.radiation_config.include_atmosphere and inputs.diffusion_operator.atmosphere.enabled:
        atmosphere_diffusion_diag, atmosphere_offdiag = (
            inputs.diffusion_operator.atmosphere.linearised_tendency()
        )
        if atmosphere_offdiag is not None and atmosphere_offdiag.nnz > 0:
            atmosphere_matrix = (
                atmosphere_offdiag
                if sparse.isspmatrix_csc(atmosphere_offdiag)
                else atmosphere_offdiag.tocsc()
            )

    boundary_layer_diffusion_diag: np.ndarray | None = None
    boundary_layer_matrix = None
    if inputs.diffusion_operator.boundary_layer is not None and inputs.diffusion_operator.boundary_layer.enabled:
        boundary_layer_diffusion_diag, boundary_layer_offdiag = (
            inputs.diffusion_operator.boundary_layer.linearised_tendency()
        )
        if boundary_layer_offdiag is not None and boundary_layer_offdiag.nnz > 0:
            boundary_layer_matrix = (
                boundary_layer_offdiag
                if sparse.isspmatrix_csc(boundary_layer_offdiag)
                else boundary_layer_offdiag.tocsc()
            )

    # Build heat exchange models
    sensible_heat_model: SensibleHeatExchangeModel | None = None
    if inputs.radiation_config.include_atmosphere and inputs.sensible_heat_cfg.enabled:
        sensible_heat_model = SensibleHeatExchangeModel(
            land_mask=inputs.land_mask,
            surface_heat_capacity_J_m2_K=inputs.heat_capacity_field,
            atmosphere_heat_capacity_J_m2_K=inputs.radiation_config.atmosphere_heat_capacity,
            wind_model=inputs.wind_model,
            config=inputs.sensible_heat_cfg,
            boundary_layer_heat_capacity_J_m2_K=(
                inputs.radiation_config.boundary_layer_heat_capacity if inputs.boundary_layer_cfg.enabled else None
            ),
        )

    latent_heat_model: LatentHeatExchangeModel | None = None
    if inputs.radiation_config.include_atmosphere and inputs.latent_heat_cfg.enabled:
        latent_heat_model = LatentHeatExchangeModel(
            land_mask=inputs.land_mask,
            surface_heat_capacity_J_m2_K=inputs.heat_capacity_field,
            atmosphere_heat_capacity_J_m2_K=inputs.radiation_config.atmosphere_heat_capacity,
            wind_model=inputs.wind_model,
            config=inputs.latent_heat_cfg,
            boundary_layer_heat_capacity_J_m2_K=(
                inputs.radiation_config.boundary_layer_heat_capacity if inputs.boundary_layer_cfg.enabled else None
            ),
        )

    def rhs(state: ModelState, insolation: np.ndarray, declination_rad: float | np.ndarray) -> np.ndarray:
        """Compute temperature tendencies from all physics."""
        log_radiation = os.environ.get("LOG_RADIATION", "0") == "1"

        # Compute cell areas for area-weighted diagnostics
        cell_areas = None
        if log_radiation:
            cell_areas = spherical_cell_area(
                inputs.lon2d,
                inputs.lat2d,
                earth_radius_m=R_EARTH_METERS,
            )

        radiative = radiation.radiative_balance_rhs(
            state.temperature,
            insolation,
            heat_capacity_field=inputs.heat_capacity_field,
            albedo_field=state.albedo_field,
            config=inputs.radiation_config,
            land_mask=inputs.land_mask,
            humidity_q=state.humidity_field,
            log_diagnostics=log_radiation,
            cell_area_m2=cell_areas,
        )
        if inputs.radiation_config.include_atmosphere:
            radiative = radiative.copy()
            nlayers = state.temperature.shape[0]
            surface_temperature = state.temperature[0]
            wind_speed_ref = state.wind_field[2] if state.wind_field is not None else None
            humidity_field = state.humidity_field

            if inputs.diffusion_operator.surface.enabled:
                radiative[0] += inputs.diffusion_operator.surface.tendency(surface_temperature)

            if nlayers == 2:
                # Two-layer system
                atmosphere_temperature = state.temperature[1]

                if inputs.diffusion_operator.atmosphere.enabled:
                    radiative[1] += inputs.diffusion_operator.atmosphere.tendency(atmosphere_temperature)

                if inputs.advection_operator is not None and state.wind_field is not None:
                    wind_u, wind_v, _ = state.wind_field
                    advection_tendency = inputs.advection_operator.tendency(
                        atmosphere_temperature, wind_u, wind_v
                    )
                    radiative[1] += advection_tendency

                if sensible_heat_model is not None:
                    tendencies = sensible_heat_model.compute_tendencies(
                        surface_temperature_K=surface_temperature,
                        atmosphere_temperature_K=atmosphere_temperature,
                        wind_speed_reference_m_s=wind_speed_ref,
                        declination_rad=declination_rad,
                    )
                    assert isinstance(tendencies, tuple)
                    assert len(tendencies) == 2
                    surface_tendency, atmosphere_tendency = tendencies
                    radiative[0] += surface_tendency
                    radiative[1] += atmosphere_tendency

                if humidity_field is not None and latent_heat_model is not None:
                    tendencies = latent_heat_model.compute_tendencies(
                        surface_temperature_K=surface_temperature,
                        atmosphere_temperature_K=atmosphere_temperature,
                        humidity_q=humidity_field,
                        wind_speed_reference_m_s=wind_speed_ref,
                        declination_rad=declination_rad,
                    )
                    assert isinstance(tendencies, tuple)
                    assert len(tendencies) == 2
                    surface_tendency, atmosphere_tendency = tendencies
                    radiative[0] += surface_tendency
                    radiative[1] += atmosphere_tendency

            elif nlayers == 3:
                # Three-layer system
                boundary_temperature = state.temperature[1]
                atmosphere_temperature = state.temperature[2]

                if inputs.diffusion_operator.boundary_layer is not None and inputs.diffusion_operator.boundary_layer.enabled:
                    radiative[1] += inputs.diffusion_operator.boundary_layer.tendency(boundary_temperature)
                if inputs.diffusion_operator.atmosphere.enabled:
                    radiative[2] += inputs.diffusion_operator.atmosphere.tendency(atmosphere_temperature)

                # Apply advection to both layers
                if inputs.advection_operator is not None and state.wind_field is not None:
                    wind_u, wind_v, _ = state.wind_field
                    advection_boundary = inputs.advection_operator.tendency(
                        boundary_temperature, wind_u, wind_v
                    )
                    advection_atmosphere = inputs.advection_operator.tendency(
                        atmosphere_temperature, wind_u, wind_v
                    )
                    radiative[1] += advection_boundary
                    radiative[2] += advection_atmosphere

                if sensible_heat_model is not None:
                    tendencies = sensible_heat_model.compute_tendencies(
                        surface_temperature_K=surface_temperature,
                        atmosphere_temperature_K=atmosphere_temperature,
                        wind_speed_reference_m_s=wind_speed_ref,
                        declination_rad=declination_rad,
                        boundary_layer_temperature_K=boundary_temperature,
                        log_diagnostics=log_radiation,
                        cell_area_m2=cell_areas,
                    )
                    assert isinstance(tendencies, tuple)
                    assert len(tendencies) == 3
                    surface_tendency, boundary_tendency, atmosphere_tendency = tendencies
                    radiative[0] += surface_tendency
                    radiative[1] += boundary_tendency
                    radiative[2] += atmosphere_tendency

                if humidity_field is not None and latent_heat_model is not None:
                    tendencies = latent_heat_model.compute_tendencies(
                        surface_temperature_K=surface_temperature,
                        atmosphere_temperature_K=atmosphere_temperature,
                        humidity_q=humidity_field,
                        wind_speed_reference_m_s=wind_speed_ref,
                        declination_rad=declination_rad,
                        boundary_layer_temperature_K=boundary_temperature,
                    )
                    assert isinstance(tendencies, tuple)
                    assert len(tendencies) == 3
                    surface_tendency, boundary_tendency, atmosphere_tendency = tendencies
                    radiative[0] += surface_tendency
                    radiative[1] += boundary_tendency
                    radiative[2] += atmosphere_tendency

            return radiative
        if inputs.diffusion_operator.surface.enabled:
            radiative = radiative + inputs.diffusion_operator.surface.tendency(state.temperature)
        return radiative

    def rhs_derivative(state: ModelState, insolation: np.ndarray) -> Linearization:
        """Compute Jacobian of RHS with respect to temperature."""
        del insolation
        if inputs.radiation_config.include_atmosphere:
            radiative_derivative = radiation.radiative_balance_rhs_temperature_derivative(
                state.temperature,
                heat_capacity_field=inputs.heat_capacity_field,
                config=inputs.radiation_config,
                land_mask=inputs.land_mask,
                humidity_q=state.humidity_field,
            )
            assert isinstance(radiative_derivative, tuple)
            radiative_diag, cross = radiative_derivative
            diag = radiative_diag.copy()
            nlayers = state.temperature.shape[0]

            if surface_diffusion_diag is not None:
                diag[0] = diag[0] + surface_diffusion_diag

            if nlayers == 2:
                # 2-layer: atmosphere is layer 1
                if atmosphere_diffusion_diag is not None:
                    diag[1] = diag[1] + atmosphere_diffusion_diag
            elif nlayers == 3:
                # 3-layer: boundary is layer 1, atmosphere is layer 2
                if boundary_layer_diffusion_diag is not None:
                    diag[1] = diag[1] + boundary_layer_diffusion_diag
                if atmosphere_diffusion_diag is not None:
                    diag[2] = diag[2] + atmosphere_diffusion_diag

            # Compute advection Jacobians if enabled
            atmosphere_advection_matrix = None
            boundary_layer_advection_matrix = None
            if inputs.advection_operator is not None and state.wind_field is not None:
                wind_u, wind_v, _ = state.wind_field
                _, advection_matrix_csr = inputs.advection_operator.linearised_tendency(wind_u, wind_v)
                if advection_matrix_csr is not None:
                    advection_matrix_converted = (
                        advection_matrix_csr
                        if sparse.isspmatrix_csc(advection_matrix_csr)
                        else advection_matrix_csr.tocsc()
                    )
                    if nlayers == 3:
                        # Both boundary and atmosphere get advection
                        boundary_layer_advection_matrix = advection_matrix_converted
                        atmosphere_advection_matrix = advection_matrix_converted
                    else:
                        # 2-layer: only atmosphere gets advection
                        atmosphere_advection_matrix = advection_matrix_converted

            # Compute sensible heat exchange Jacobian if enabled
            if sensible_heat_model is not None:
                if nlayers == 3:
                    sh_diag, sh_cross = sensible_heat_model.compute_jacobian(
                        state.temperature[0],
                        state.temperature[2],
                        wind_speed_reference_m_s=state.wind_field[2] if state.wind_field is not None else None,
                        declination_rad=None,
                        boundary_layer_temperature_K=state.temperature[1],
                    )
                elif nlayers == 2:
                    sh_diag, sh_cross = sensible_heat_model.compute_jacobian(
                        state.temperature[0],
                        state.temperature[1],
                        wind_speed_reference_m_s=state.wind_field[2] if state.wind_field is not None else None,
                        declination_rad=None,
                    )
                else:
                    assert False, "Must have atmosphere for sensible heat exchange"
                # Add sensible heat directly to diagonal and cross terms
                diag = diag + sh_diag
                if cross is not None:
                    cross = cross + sh_cross
                else:
                    cross = sh_cross

            # Compute latent heat exchange Jacobian if enabled
            if latent_heat_model is not None and state.humidity_field is not None:
                if nlayers == 3:
                    lh_diag, lh_cross = latent_heat_model.compute_jacobian(
                        state.temperature[0],
                        state.temperature[2],
                        state.humidity_field,
                        wind_speed_reference_m_s=state.wind_field[2] if state.wind_field is not None else None,
                        declination_rad=None,
                        boundary_layer_temperature_K=state.temperature[1],
                    )
                elif nlayers == 2:
                    lh_diag, lh_cross = latent_heat_model.compute_jacobian(
                        state.temperature[0],
                        state.temperature[1],
                        state.humidity_field,
                        wind_speed_reference_m_s=state.wind_field[2] if state.wind_field is not None else None,
                        declination_rad=None,
                    )
                else:
                    assert False, "Must have atmosphere for latent heat exchange"
                # Add latent heat directly to diagonal and cross terms
                diag = diag + lh_diag
                if cross is not None:
                    cross = cross + lh_cross
                else:
                    cross = lh_cross

            return Linearization(
                diag=diag,
                cross=cross,
                surface_diffusion_matrix=surface_matrix,
                atmosphere_diffusion_matrix=atmosphere_matrix,
                atmosphere_advection_matrix=atmosphere_advection_matrix,
                boundary_layer_diffusion_matrix=boundary_layer_matrix,
                boundary_layer_advection_matrix=boundary_layer_advection_matrix,
            )
        radiative_derivative = radiation.radiative_balance_rhs_temperature_derivative(
            state.temperature,
            heat_capacity_field=inputs.heat_capacity_field,
            config=inputs.radiation_config,
            land_mask=inputs.land_mask,
            humidity_q=state.humidity_field,
        )
        assert isinstance(radiative_derivative, np.ndarray)
        diag = radiative_derivative.copy()
        if surface_diffusion_diag is not None:
            diag = diag + surface_diffusion_diag
        return Linearization(diag=diag, surface_diffusion_matrix=surface_matrix)

    return rhs, rhs_derivative
