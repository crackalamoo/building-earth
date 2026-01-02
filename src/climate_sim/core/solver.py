"""General-purpose periodic solver utilities for energy-balance models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

import climate_sim.physics.radiation as radiation
from climate_sim.physics.atmosphere.wind import (
    WindConfig,
    WindModel,
)
from climate_sim.physics.atmosphere.advection import (
    AdvectionConfig,
    AdvectionOperator,
)
from climate_sim.physics.diffusion import (
    DiffusionConfig,
    LayeredDiffusionOperator,
    create_diffusion_operator,
)
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig, AlbedoModel
from climate_sim.physics.sensible_heat_exchange import (
    SensibleHeatExchangeConfig,
    SensibleHeatExchangeModel,
)
from climate_sim.physics.latent_heat_exchange import (
    LatentHeatExchangeConfig,
    LatentHeatExchangeModel,
)
from climate_sim.physics.atmosphere.boundary_layer import BoundaryLayerConfig
from climate_sim.data.calendar import DAYS_PER_MONTH, SECONDS_PER_DAY
from climate_sim.core.grid import create_lat_lon_grid, expand_latitude_field
from climate_sim.physics.solar import (
    compute_monthly_insolation_field,
    compute_monthly_declinations,
)
from climate_sim.physics.humidity import compute_humidity_q
from climate_sim.data.landmask import (
    compute_albedo_field,
    compute_heat_capacity_field,
    compute_land_mask,
)
from climate_sim.data.elevation import (
    compute_cell_elevation,
    compute_cell_roughness_length,
)
from climate_sim.core.timing import time_block, get_profiler, reset_profiler
from climate_sim.core.math_core import (
    LinearSolveCache,
    DEFAULT_LINEAR_SOLVE_CACHE,
    get_identity_matrix,
    fingerprint_csc_matrix,
    get_factorized_solver,
    spherical_cell_area
)
from climate_sim.core.state import (
    ModelState,
    select_wind_temperature,
)
from climate_sim.core.postprocess import postprocess_periodic_cycle_results
from climate_sim.runtime.config import ModelConfig
from climate_sim.data.constants import ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K, R_EARTH_METERS
import os

NEWTON_STEP_TOLERANCE_K = 1.0
PERIODIC_FIXED_POINT_TOLERANCE_K = 0.5
PERIODIC_FIXED_POINT_TOLERANCE_K_99P = 1.0
NEWTON_MAX_ITERS = 16
NEWTON_BACKTRACK_REDUCTION = 0.5
NEWTON_BACKTRACK_CUTOFF = 1e-3
FIXED_POINT_MAX_ITERS = 100


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


@dataclass(frozen=True)
class SurfaceHeatCapacityContext:
    """Container for surface-layer heat-capacity metadata."""

    lon2d: np.ndarray
    lat2d: np.ndarray
    albedo_model: AlbedoModel
    wind_model: WindModel | None
    advection_operator: AdvectionOperator | None
    base_albedo: np.ndarray
    land_mask: np.ndarray
    base_C_land: float
    base_C_ocean: float
    baseline_capacity: np.ndarray


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

type RhsFactory = Callable[[RhsBuildInputs], tuple[RhsFn, RhsDerivativeFn]]
type InitialGuessFn = Callable[[FloatArray, FloatArray, RadiationConfig, FloatArray], FloatArray]
type MonthlyInsolationLatFn = Callable[[FloatArray], FloatArray]
type HeatCapacityFieldFn = Callable[[FloatArray, FloatArray], FloatArray]


def _build_surface_jacobian_block(
    ceff: np.ndarray,
    diag: np.ndarray,
    base_capacity: np.ndarray,
    diffusion_matrix: sparse.csc_matrix | None,
    dt_seconds: float,
) -> sparse.csc_matrix:
    """Build the surface layer jacobian block (shared between single and multi-layer)."""
    block = sparse.diags(ceff.ravel(), format="csc")
    block -= dt_seconds * sparse.diags((base_capacity * diag).ravel(), format="csc")
    if diffusion_matrix is not None and diffusion_matrix.nnz > 0:
        block -= dt_seconds * sparse.diags(base_capacity.ravel(), format="csc") @ diffusion_matrix
    return block


def monthly_step(
    state: ModelState,
    insolation_W_m2: np.ndarray,
    declination: np.ndarray,
    dt_seconds: float,
    *,
    rhs_fn: RhsFn,
    rhs_temperature_derivative_fn: RhsDerivativeFn,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
    surface_context: SurfaceHeatCapacityContext,
) -> ModelState:
    """Advance the column temperature one implicit backward-Euler step."""
    with time_block("monthly_step"):
        start_temp = state.temperature
        temp_next = np.maximum(start_temp, temperature_floor)
        cache = solver_cache or DEFAULT_LINEAR_SOLVE_CACHE

        base_capacity = surface_context.baseline_capacity

        def _effective_surface_capacity(temp_surface: np.ndarray) -> np.ndarray:
            return surface_context.albedo_model.effective_heat_capacity_surface(
                temp_surface,
                land_mask=surface_context.land_mask,
                base_C_land=surface_context.base_C_land,
                base_C_ocean=surface_context.base_C_ocean,
            )
        
        def _init_state(temp: np.ndarray) -> ModelState:
            with time_block("_init_state"):
                lat2d = surface_context.lat2d
                albedo_field = surface_context.albedo_model.apply_snow_albedo(base_albedo_field, temp[0])
                # Wind field uses free atmosphere temperature (layer 2 in 3-layer, layer 1 in 2-layer, layer 0 in 1-layer)
                if temp.shape[0] == 3:
                    wind_temp = temp[2]  # Free atmosphere in 3-layer mode
                elif temp.shape[0] == 2:
                    wind_temp = temp[1]  # Atmosphere in 2-layer mode
                else:
                    wind_temp = temp[0]  # Surface in 1-layer mode
                wind_field = surface_context.wind_model.wind_field(wind_temp, declination_rad=declination) if surface_context.wind_model else np.zeros_like(temp[0])
                humidity_field = compute_humidity_q(lat2d, temp[0], declination)
                return ModelState(
                    temperature=temp,
                    albedo_field=albedo_field,
                    wind_field=wind_field,
                    humidity_field=humidity_field,
                )

        base_albedo_field = surface_context.base_albedo

        # implicit solver loop
        solve_linear = None
        linearization = None
        for newton_iter in range(NEWTON_MAX_ITERS):
            with time_block("newton_iteration"):
                temp_capped = np.maximum(temp_next, temperature_floor)
                state_capped = _init_state(temp_capped)
                with time_block("rhs_evaluation"):
                    rhs_value = rhs_fn(state_capped, insolation_W_m2, declination)
                if linearization is None:
                    with time_block("rhs_derivative"):
                        linearization = rhs_temperature_derivative_fn(state_capped, insolation_W_m2)

                nlayers = temp_capped.shape[0]
                surface_diag = linearization.diag[0]
                ceff_surface = _effective_surface_capacity(temp_capped[0])
                flux_surface = base_capacity * rhs_value[0]
                residual_surface = ceff_surface * (temp_capped[0] - start_temp[0]) - dt_seconds * flux_surface
                
                nlat, nlon = surface_diag.shape
                size = nlat * nlon
                
                if nlayers == 1:
                    # Single-layer: only surface residual and jacobian
                    residual = residual_surface[np.newaxis, :, :]
                    residual_flat = residual_surface.ravel()
                    
                    with time_block("jacobian_assembly"):
                        jacobian = _build_surface_jacobian_block(
                            ceff_surface, surface_diag, base_capacity, linearization.surface_diffusion_matrix, dt_seconds
                        )
                    
                    fingerprint = linearization.solver_fingerprint or fingerprint_csc_matrix(jacobian)
                    linearization.solver_fingerprint = fingerprint
                    with time_block("linear_solve"):
                        if solve_linear is None:
                            with time_block("factorize_solver"):
                                solve_linear = get_factorized_solver(jacobian, cache=cache, fingerprint=fingerprint)
                        correction_flat = solve_linear(residual_flat)
                    correction = correction_flat.reshape((nlat, nlon))[np.newaxis, :, :]
                elif nlayers == 2:
                    # Two-layer: surface + atmosphere with coupling
                    atmosphere_diag = linearization.diag[1]
                    residual_atmosphere = temp_capped[1] - start_temp[1] - dt_seconds * rhs_value[1]
                    residual = np.stack([residual_surface, residual_atmosphere])

                    identity = get_identity_matrix(size, cache=cache)

                    with time_block("jacobian_assembly"):
                        surface_block = _build_surface_jacobian_block(
                            ceff_surface, surface_diag, base_capacity, linearization.surface_diffusion_matrix, dt_seconds
                        )
                        atmosphere_block = identity.copy()
                        atmosphere_block -= dt_seconds * sparse.diags(atmosphere_diag.ravel(), format="csc")
                        if linearization.atmosphere_diffusion_matrix is not None and linearization.atmosphere_diffusion_matrix.nnz > 0:
                            atmosphere_block -= dt_seconds * linearization.atmosphere_diffusion_matrix
                        if linearization.atmosphere_advection_matrix is not None and linearization.atmosphere_advection_matrix.nnz > 0:
                            atmosphere_block -= dt_seconds * linearization.atmosphere_advection_matrix

                        if linearization.cross is not None:
                            coupling_surface_atm = -dt_seconds * sparse.diags((base_capacity * linearization.cross[0, 1]).ravel(), format="csc")
                            coupling_atm_surface = -dt_seconds * sparse.diags(linearization.cross[1, 0].ravel(), format="csc")
                        else:
                            coupling_surface_atm = coupling_atm_surface = sparse.csc_matrix((size, size))

                        jacobian = sparse.bmat([[surface_block, coupling_surface_atm], [coupling_atm_surface, atmosphere_block]], format="csc")
                        assert isinstance(jacobian, sparse.csc_matrix)

                    residual_flat = np.concatenate([residual_surface.ravel(), residual_atmosphere.ravel()], axis=0)
                    fingerprint = linearization.solver_fingerprint or fingerprint_csc_matrix(jacobian)
                    linearization.solver_fingerprint = fingerprint
                    with time_block("linear_solve"):
                        if solve_linear is None:
                            with time_block("factorize_solver"):
                                solve_linear = get_factorized_solver(jacobian, cache=cache, fingerprint=fingerprint)
                        correction_flat = solve_linear(residual_flat)
                    correction_surface = correction_flat[:size].reshape(surface_diag.shape)
                    correction_atmosphere = correction_flat[size:].reshape(atmosphere_diag.shape)
                    correction = np.stack([correction_surface, correction_atmosphere])
                elif nlayers == 3:
                    # Three-layer: surface + boundary layer + atmosphere with adjacent coupling only
                    boundary_diag = linearization.diag[1]
                    atmosphere_diag = linearization.diag[2]
                    residual_boundary = temp_capped[1] - start_temp[1] - dt_seconds * rhs_value[1]
                    residual_atmosphere = temp_capped[2] - start_temp[2] - dt_seconds * rhs_value[2]
                    residual = np.stack([residual_surface, residual_boundary, residual_atmosphere])

                    identity = get_identity_matrix(size, cache=cache)
                    zero_matrix = sparse.csc_matrix((size, size))

                    with time_block("jacobian_assembly"):
                        surface_block = _build_surface_jacobian_block(
                            ceff_surface, surface_diag, base_capacity, linearization.surface_diffusion_matrix, dt_seconds
                        )

                        # Boundary layer block (with diffusion and advection if available)
                        boundary_block = identity.copy()
                        boundary_block -= dt_seconds * sparse.diags(boundary_diag.ravel(), format="csc")
                        if linearization.boundary_layer_diffusion_matrix is not None and linearization.boundary_layer_diffusion_matrix.nnz > 0:
                            boundary_block -= dt_seconds * linearization.boundary_layer_diffusion_matrix
                        if linearization.boundary_layer_advection_matrix is not None and linearization.boundary_layer_advection_matrix.nnz > 0:
                            boundary_block -= dt_seconds * linearization.boundary_layer_advection_matrix

                        # Atmosphere block (with diffusion and advection)
                        atmosphere_block = identity.copy()
                        atmosphere_block -= dt_seconds * sparse.diags(atmosphere_diag.ravel(), format="csc")
                        if linearization.atmosphere_diffusion_matrix is not None and linearization.atmosphere_diffusion_matrix.nnz > 0:
                            atmosphere_block -= dt_seconds * linearization.atmosphere_diffusion_matrix
                        if linearization.atmosphere_advection_matrix is not None and linearization.atmosphere_advection_matrix.nnz > 0:
                            atmosphere_block -= dt_seconds * linearization.atmosphere_advection_matrix

                        # Cross-coupling: includes surface-atmosphere coupling via transmission
                        if linearization.cross is not None:
                            coupling_01 = -dt_seconds * sparse.diags((base_capacity * linearization.cross[0, 1]).ravel(), format="csc")
                            coupling_02 = -dt_seconds * sparse.diags((base_capacity * linearization.cross[0, 2]).ravel(), format="csc")
                            coupling_10 = -dt_seconds * sparse.diags(linearization.cross[1, 0].ravel(), format="csc")
                            coupling_12 = -dt_seconds * sparse.diags(linearization.cross[1, 2].ravel(), format="csc")
                            coupling_20 = -dt_seconds * sparse.diags(linearization.cross[2, 0].ravel(), format="csc")
                            coupling_21 = -dt_seconds * sparse.diags(linearization.cross[2, 1].ravel(), format="csc")
                        else:
                            coupling_01 = coupling_02 = coupling_10 = coupling_12 = coupling_20 = coupling_21 = zero_matrix

                        jacobian = sparse.bmat([
                            [surface_block, coupling_01, coupling_02],
                            [coupling_10, boundary_block, coupling_12],
                            [coupling_20, coupling_21, atmosphere_block]
                        ], format="csc")
                        assert isinstance(jacobian, sparse.csc_matrix)

                    residual_flat = np.concatenate([
                        residual_surface.ravel(),
                        residual_boundary.ravel(),
                        residual_atmosphere.ravel()
                    ], axis=0)
                    fingerprint = linearization.solver_fingerprint or fingerprint_csc_matrix(jacobian)
                    linearization.solver_fingerprint = fingerprint
                    with time_block("linear_solve"):
                        if solve_linear is None:
                            with time_block("factorize_solver"):
                                solve_linear = get_factorized_solver(jacobian, cache=cache, fingerprint=fingerprint)
                        correction_flat = solve_linear(residual_flat)
                    correction_surface = correction_flat[:size].reshape(surface_diag.shape)
                    correction_boundary = correction_flat[size:2*size].reshape(boundary_diag.shape)
                    correction_atmosphere = correction_flat[2*size:].reshape(atmosphere_diag.shape)
                    correction = np.stack([correction_surface, correction_boundary, correction_atmosphere])
                else:
                    raise ValueError(f"Unsupported number of layers: {nlayers}")

                damping = 1.0
                max_residual = float(np.max(np.abs(residual)))
                accepted = False
                prev_temp = temp_next
                temp_candidate = temp_next
                residual_candidate = residual

                while damping >= NEWTON_BACKTRACK_CUTOFF:
                    temp_candidate = np.maximum(prev_temp - damping * correction, temperature_floor)
                    state_candidate = _init_state(temp_candidate)
                    with time_block("backtrack_rhs"):
                        rhs_candidate = rhs_fn(state_candidate, insolation_W_m2, declination)
                    
                    ceff_candidate = _effective_surface_capacity(temp_candidate[0])
                    nlayers = temp_candidate.shape[0]
                    residual_surface_candidate = ceff_candidate * (temp_candidate[0] - start_temp[0]) - dt_seconds * (base_capacity * rhs_candidate[0])

                    if nlayers == 1:
                        residual_candidate = residual_surface_candidate[np.newaxis, :, :]
                    elif nlayers == 2:
                        residual_atmosphere_candidate = temp_candidate[1] - start_temp[1] - dt_seconds * rhs_candidate[1]
                        residual_candidate = np.stack([residual_surface_candidate, residual_atmosphere_candidate])
                    elif nlayers == 3:
                        residual_boundary_candidate = temp_candidate[1] - start_temp[1] - dt_seconds * rhs_candidate[1]
                        residual_atmosphere_candidate = temp_candidate[2] - start_temp[2] - dt_seconds * rhs_candidate[2]
                        residual_candidate = np.stack([residual_surface_candidate, residual_boundary_candidate, residual_atmosphere_candidate])
                    else:
                        raise ValueError(f"Unsupported number of layers: {nlayers}")
                    candidate_norm = float(np.max(np.abs(residual_candidate)))

                    if candidate_norm <= (1.0 - 1e-4 * damping) * max_residual:
                        temp_next = temp_candidate
                        residual = residual_candidate
                        accepted = True
                        break

                    damping *= NEWTON_BACKTRACK_REDUCTION

                if not accepted:
                    temp_next = temp_candidate
                    residual = residual_candidate

                step = prev_temp - temp_next
                if np.max(np.abs(step)) < NEWTON_STEP_TOLERANCE_K:
                    break


        return _init_state(temp_next)

def evolve_year(
    state: ModelState,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    rhs_fn: RhsFn,
    rhs_temperature_derivative_fn: RhsDerivativeFn,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
    surface_context: SurfaceHeatCapacityContext,
) -> list[ModelState]:
    """Propagate the state through 12 implicit steps"""
    states: list[ModelState] = []
    monthly_declinations = compute_monthly_declinations()
    with time_block("evolve_year"):
        for month_n in range(12):
            month = (month_n + 2) % 12 # start from March so initial guess is better
            state = monthly_step(
                state,
                monthly_insolation[month],
                monthly_declinations[month],
                month_durations[month],
                rhs_fn=rhs_fn,
                rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
                temperature_floor=temperature_floor,
                solver_cache=solver_cache,
                surface_context=surface_context,
            )
            states.append(state)
        return states


def _solve_anderson_coefficients(residuals: list[np.ndarray]) -> np.ndarray | None:
    """Return coefficients that minimize the combined residual subject to sum(alpha)=1."""

    m = len(residuals)
    if m == 0:
        return None

    residual_matrix = np.column_stack(residuals)
    gram = residual_matrix.T @ residual_matrix
    scale = np.linalg.norm(gram, ord=np.inf)
    if not np.isfinite(scale):
        scale = 1.0
    regularisation = 1e-12 * scale + 1e-14
    gram = gram + regularisation * np.eye(m)

    ones = np.ones(m)
    augmented = np.empty((m + 1, m + 1), dtype=float)
    augmented[:m, :m] = gram
    augmented[:m, m] = ones
    augmented[m, :m] = ones
    augmented[m, m] = 0.0

    rhs = np.zeros(m + 1, dtype=float)
    rhs[-1] = 1.0

    try:
        solution = np.linalg.solve(augmented, rhs)
    except np.linalg.LinAlgError:
        return None

    alpha = solution[:m]
    if not np.all(np.isfinite(alpha)):
        return None
    return alpha


def find_periodic_temperature(
    initial_state: ModelState,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    rhs_fn: RhsFn,
    rhs_temperature_derivative_fn: RhsDerivativeFn,
    surface_context: SurfaceHeatCapacityContext,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
) -> list[ModelState]:
    """Solve P(T) = T for the annual map using Anderson acceleration."""

    with time_block("find_periodic_temperature"):
        state = initial_state
        temp = initial_state.temperature
        state.temperature = np.maximum(temp, temperature_floor)
        states = [initial_state] * 12
        residual_history: list[np.ndarray] = []
        advanced_history: list[np.ndarray] = []
        history_limit = 5
        residual_max = 0

        for iter_idx in range(FIXED_POINT_MAX_ITERS):
            with time_block("periodic_iteration"):
                advanced_states = evolve_year(
                    state,
                    monthly_insolation,
                    month_durations,
                    rhs_fn=rhs_fn,
                    rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
                    temperature_floor=temperature_floor,
                    solver_cache=solver_cache,
                    surface_context=surface_context,
                )

                advanced = advanced_states[-1]
                residual = np.array([
                    advanced_states[i].temperature[0] - states[i].temperature[0] for i in range(12)
                ])
                residual_rms = np.sqrt(np.mean(np.square(residual)))
                residual_99p = np.percentile(np.abs(residual), 99)
                residual_max = np.max(np.abs(residual))
                print(residual_rms, residual_99p, residual_max)

                if residual_rms < PERIODIC_FIXED_POINT_TOLERANCE_K and residual_99p < PERIODIC_FIXED_POINT_TOLERANCE_K_99P:
                    return [advanced_states[(i - 2) % 12] for i in range(12)] # convert from March start to January start

                states = advanced_states
                state = advanced_states[-1]

                residual_flat = residual.ravel()
                advanced_flat = advanced.temperature.ravel()

                if len(residual_history) == history_limit:
                    residual_history.pop(0)
                    advanced_history.pop(0)

                residual_history.append(residual_flat)
                advanced_history.append(advanced_flat)

                with time_block("anderson_acceleration"):
                    coefficients = None
                    if len(residual_history) > 1:
                        coefficients = _solve_anderson_coefficients(residual_history)
                        if coefficients is not None:
                            coeff_sum = float(np.sum(coefficients))
                            if not np.isfinite(coeff_sum) or abs(coeff_sum) < 1e-12:
                                coefficients = None
                            else:
                                coefficients = coefficients / coeff_sum

                    if coefficients is None:
                        T_next = advanced.temperature
                        residual_history = residual_history[-1:]
                        advanced_history = advanced_history[-1:]
                    else:
                        combined = np.zeros_like(advanced_flat)
                        for weight, advanced_state in zip(coefficients, advanced_history, strict=True):
                            combined += weight * advanced_state
                        T_next = combined.reshape(state.temperature.shape)

                        if not np.all(np.isfinite(T_next)):
                            residual_history = residual_history[-1:]
                            advanced_history = advanced_history[-1:]

        raise RuntimeError(
            "Failed to converge to a periodic solution after "
            f"{FIXED_POINT_MAX_ITERS} iterations (last residual {residual_max:.3e} K)"
        )

def solve_periodic_cycle_for_albedo(
    *,
    initial_state: ModelState | None,
    surface_context: SurfaceHeatCapacityContext,
    heat_capacity_field: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    diffusion_operator: LayeredDiffusionOperator,
    radiation_config: RadiationConfig,
    wind_model: WindModel | None,
    rhs_factory: RhsFactory,
    initial_guess_fn: InitialGuessFn,
    temperature_floor: float,
    solver_cache: LinearSolveCache,
    land_mask: np.ndarray,
    roughness_length: np.ndarray,
    topographic_elevation: np.ndarray,
    sensible_heat_cfg: SensibleHeatExchangeConfig,
    latent_heat_cfg: LatentHeatExchangeConfig,
    boundary_layer_cfg: BoundaryLayerConfig,
) -> list[ModelState]:
    """Compute the periodic solution and monthly mean temperatures for a given albedo field."""

    with time_block("solve_periodic_cycle_for_albedo"):
        base_albedo_field = surface_context.base_albedo
        wind_model = surface_context.wind_model
        advection_operator = surface_context.advection_operator
        
        if initial_state is None:
            with time_block("initial_guess"):
                guess = initial_guess_fn(
                    monthly_insolation,
                    base_albedo_field,
                    radiation_config,
                    land_mask,
                )

            state_init = ModelState(
                temperature=guess,
                albedo_field=base_albedo_field,
                wind_field=None,
                humidity_field=None,
            )
        else:
            state_init = initial_state

        with time_block("rhs_factory"):
            inputs = RhsBuildInputs(
                heat_capacity_field=heat_capacity_field,
                diffusion_operator=diffusion_operator,
                radiation_config=radiation_config,
                land_mask=land_mask,
                roughness_length=roughness_length,
                topographic_elevation=topographic_elevation,
                sensible_heat_cfg=sensible_heat_cfg,
                latent_heat_cfg=latent_heat_cfg,
                boundary_layer_cfg=boundary_layer_cfg,
                wind_model=wind_model,
                advection_operator=advection_operator,
                lon2d=surface_context.lon2d,
                lat2d=surface_context.lat2d,
            )
            rhs_fn, rhs_temperature_derivative_fn = rhs_factory(inputs)
            
    

    annual_periodic_states = find_periodic_temperature(
        state_init,
        monthly_insolation,
        month_durations,
        rhs_fn=rhs_fn,
        rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
        surface_context=surface_context,
        temperature_floor=temperature_floor,
        solver_cache=solver_cache,
    )

    return annual_periodic_states

def compute_periodic_cycle_kelvin(
    *,
    resolution_deg: float,
    monthly_insolation_lat_fn: MonthlyInsolationLatFn,
    heat_capacity_field_fn: HeatCapacityFieldFn,
    rhs_factory: RhsFactory,
    initial_guess_fn: InitialGuessFn,
    radiation_config: RadiationConfig,
    diffusion_config: DiffusionConfig,
    wind_config: WindConfig,
    advection_config: AdvectionConfig,
    snow_config: SnowAlbedoConfig,
    sensible_heat_config: SensibleHeatExchangeConfig,
    latent_heat_config: LatentHeatExchangeConfig,
    boundary_layer_config: BoundaryLayerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ModelState]]:
    """Solve for the periodic temperature cycle and return diagnostics.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[ModelState]]
        Longitude grid, latitude grid, topographic elevation (m), and the
        monthly model states expressed in Kelvin.
    """
    with time_block("compute_periodic_cycle_kelvin"):
        with time_block("setup_grids_and_fields"):
            lon2d, lat2d = create_lat_lon_grid(resolution_deg)

            monthly_insolation_lat = monthly_insolation_lat_fn(lat2d)
            monthly_insolation = expand_latitude_field(monthly_insolation_lat, lon2d.shape[1])

            heat_capacity_field = heat_capacity_field_fn(lon2d, lat2d)
            snow_cfg = snow_config
            sensible_heat_cfg = sensible_heat_config
            latent_heat_cfg = latent_heat_config
            boundary_layer_cfg = boundary_layer_config
            albedo_kwargs: dict[str, float] = {}
            land_mask = compute_land_mask(lon2d, lat2d)

            albedo_model = AlbedoModel(
                lat2d,
                lon2d,
                config=snow_cfg,
                land_mask=land_mask,
            )
            if snow_cfg.enabled:
                albedo_kwargs = {"land_albedo": 0.18, "ocean_albedo": 0.06}

            base_albedo_field = compute_albedo_field(lon2d, lat2d, **albedo_kwargs)

            roughness_length = compute_cell_roughness_length(
                lon2d,
                lat2d,
                land_mask=land_mask,
            )

            if sensible_heat_cfg.include_lapse_rate_elevation:
                topographic_elevation = compute_cell_elevation(lon2d, lat2d)
                topographic_elevation = np.maximum(topographic_elevation, 0.0)
            else:
                topographic_elevation = np.zeros_like(lon2d, dtype=float)

            # Pass boundary layer heat capacity if boundary layer is enabled
            bl_heat_cap = None
            if boundary_layer_config.enabled:
                bl_heat_cap = radiation_config.boundary_layer_heat_capacity

            diffusion_operator = create_diffusion_operator(
                lon2d,
                lat2d,
                heat_capacity_field,
                land_mask=land_mask,
                atmosphere_heat_capacity=radiation_config.atmosphere_heat_capacity,
                boundary_layer_heat_capacity=bl_heat_cap,
                config=diffusion_config,
            )
            wind_model: WindModel = WindModel(
                lon2d,
                lat2d,
                config=wind_config,
            )

            advection_operator: AdvectionOperator = AdvectionOperator(
                lon2d,
                lat2d,
                config=advection_config,
            )
            month_durations = DAYS_PER_MONTH * SECONDS_PER_DAY
            solver_cache = LinearSolveCache()

        previous_periodic: ModelState | None = None
        monthly: list[ModelState] | None = None

        # Initialize humidity fields from radiative equilibrium temperature guess
        with time_block("initialize_humidity_fields"):
            monthly_declinations = compute_monthly_declinations()
            initial_guess_temp = initial_guess_fn(
                monthly_insolation,
                base_albedo_field,
                radiation_config,
                land_mask,
            )

            current_humidity_fields: list[np.ndarray] = []
            for month_idx in range(12):
                # Extract surface temperature for this month (always use layer 0)
                temp_for_month = initial_guess_temp[0]
                
                # Compute humidity for this month
                humidity = compute_humidity_q(
                    lat2d,
                    temp_for_month,
                    monthly_declinations[month_idx],
                    land_mask=land_mask,
                )
                current_humidity_fields.append(humidity)

        land_values = heat_capacity_field[land_mask]
        if land_values.size == 0:
            base_C_land = float(np.mean(heat_capacity_field))
        else:
            base_C_land = float(np.mean(land_values))
        ocean_mask = ~land_mask
        ocean_values = heat_capacity_field[ocean_mask]
        if ocean_values.size == 0:
            base_C_ocean = base_C_land
        else:
            base_C_ocean = float(np.mean(ocean_values))

        baseline_capacity = np.where(land_mask, base_C_land, base_C_ocean).astype(float)

        surface_context = SurfaceHeatCapacityContext(
            lat2d=lat2d,
            lon2d=lon2d,
            albedo_model=albedo_model,
            wind_model=wind_model,
            advection_operator=advection_operator,
            base_albedo=base_albedo_field,
            land_mask=land_mask,
            base_C_land=base_C_land,
            base_C_ocean=base_C_ocean,
            baseline_capacity=baseline_capacity,
        )
        
        monthly = solve_periodic_cycle_for_albedo(
            initial_state=previous_periodic,
            surface_context=surface_context,
            heat_capacity_field=heat_capacity_field,
            monthly_insolation=monthly_insolation,
            month_durations=month_durations,
            diffusion_operator=diffusion_operator,
            radiation_config=radiation_config,
            wind_model=wind_model,
            rhs_factory=rhs_factory,
            initial_guess_fn=initial_guess_fn,
            temperature_floor=radiation_config.temperature_floor,
            solver_cache=solver_cache,
            land_mask=land_mask,
            roughness_length=roughness_length,
            topographic_elevation=topographic_elevation,
            sensible_heat_cfg=sensible_heat_cfg,
            latent_heat_cfg=latent_heat_cfg,
            boundary_layer_cfg=boundary_layer_cfg,
        )

        if wind_model is not None and wind_model.enabled:
            with time_block("update_wind_fields"):
                new_wind_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                new_geostrophic_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                for idx, month_state in enumerate(monthly):
                    wind_temperature = select_wind_temperature(month_state.temperature)
                    wind_field = month_state.wind_field or wind_model.wind_field(wind_temperature, declination_rad=monthly_declinations[idx])
                    geostrophic_field = wind_model.wind_field(wind_temperature, declination_rad=monthly_declinations[idx], ekman_drag=False)

                    new_wind_fields.append(wind_field)
                    new_geostrophic_fields.append(geostrophic_field)
                    monthly[idx] = ModelState(
                        temperature=month_state.temperature,
                        albedo_field=base_albedo_field,
                        wind_field=wind_field,
                        humidity_field=month_state.humidity_field,
                    )

        return lon2d, lat2d, topographic_elevation, monthly


def compute_periodic_cycle_results(
    resolution_deg: float = 1.0,
    *,
    model_config: ModelConfig,
    return_layer_map: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Produce the periodic cycle fields in Celsius along with the converged albedo layer.

    The solver consumes a fully-specified :class:`~climate_sim.runtime.config.ModelConfig`
    so that each feature toggle lives inside its own dataclass instead of being encoded
    as ``None`` versus instantiated configs.

    Returns the lon/lat grids and either the surface temperature field (default) or a
    dictionary of layer outputs when ``return_layer_map`` is True.
    """

    def monthly_insolation_lat_fn(lat2d: np.ndarray) -> np.ndarray:
        return compute_monthly_insolation_field(
            lat2d,
            solar_constant=model_config.solar_constant,
            use_elliptical_orbit=model_config.use_elliptical_orbit,
        )

    def heat_capacity_field_fn(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
        return compute_heat_capacity_field(
            lon2d,
            lat2d,
            ocean_heat_capacity=model_config.ocean_heat_capacity,
            land_heat_capacity=model_config.land_heat_capacity,
        )

    resolved_radiation = model_config.radiation
    resolved_diffusion = model_config.diffusion
    resolved_wind = model_config.wind
    resolved_advection = model_config.advection
    resolved_snow = model_config.snow
    sensible_heat_cfg = model_config.sensible_heat
    latent_heat_cfg = model_config.latent_heat
    boundary_layer_cfg = model_config.boundary_layer

    # Update radiation config with boundary layer heat capacities if enabled
    if boundary_layer_cfg.enabled:
        if not resolved_radiation.include_atmosphere:
            raise ValueError("Boundary layer requires include_atmosphere=True in radiation config")
        resolved_radiation = replace(
            resolved_radiation,
            atmosphere_heat_capacity=ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,  # Use 6km atmosphere
            boundary_layer_heat_capacity=boundary_layer_cfg.heat_capacity,
            boundary_layer_emissivity=boundary_layer_cfg.emissivity,
        )
    # else: keep default values (use legacy atmosphere heat capacity for 2-layer mode)

    def rhs_factory(inputs: RhsBuildInputs) -> tuple[RhsFn, RhsDerivativeFn]:
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

    def initial_guess_fn(
        monthly_insolation: np.ndarray,
        albedo_field: np.ndarray,
        config: RadiationConfig,
        land_mask: np.ndarray,
    ) -> np.ndarray:
        return radiation.radiative_equilibrium_initial_guess(
            monthly_insolation,
            albedo_field=albedo_field,
            config=config,
            land_mask=land_mask,
        )

    reset_profiler()
    
    lon2d, lat2d, topographic_elevation, monthly_states = compute_periodic_cycle_kelvin(
        resolution_deg=resolution_deg,
        monthly_insolation_lat_fn=monthly_insolation_lat_fn,
        heat_capacity_field_fn=heat_capacity_field_fn,
        rhs_factory=rhs_factory,
        initial_guess_fn=initial_guess_fn,
        radiation_config=resolved_radiation,
        diffusion_config=resolved_diffusion,
        wind_config=resolved_wind,
        advection_config=resolved_advection,
        snow_config=resolved_snow,
        sensible_heat_config=sensible_heat_cfg,
        latent_heat_config=latent_heat_cfg,
        boundary_layer_config=boundary_layer_cfg,
    )
    
    get_profiler().print_summary()
    
    return postprocess_periodic_cycle_results(
        lon2d,
        lat2d,
        monthly_states,
        resolved_wind=resolved_wind,
        resolved_radiation=resolved_radiation,
        return_layer_map=return_layer_map,
    )
