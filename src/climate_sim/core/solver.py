"""General-purpose periodic solver utilities for energy-balance models."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as splinalg

import climate_sim.physics.radiation as radiation
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.solar import (
    compute_monthly_declinations,
)
from climate_sim.physics.humidity import compute_humidity_q
from climate_sim.core.timing import time_block, get_profiler, reset_profiler
from climate_sim.core.math_core import (
    LinearSolveCache,
    DEFAULT_LINEAR_SOLVE_CACHE,
    get_identity_matrix,
)
from climate_sim.core.state import (
    ModelState,
    select_wind_temperature,
)
from climate_sim.core.postprocess import postprocess_periodic_cycle_results
from climate_sim.core.rhs_builder import create_rhs_functions, RhsFn, RhsDerivativeFn, RhsBuildInputs
from climate_sim.core.operators import SurfaceHeatCapacityContext
from climate_sim.runtime.config import ModelConfig

NEWTON_STEP_TOLERANCE_K = 1.0
PERIODIC_FIXED_POINT_TOLERANCE_K = 0.5
PERIODIC_FIXED_POINT_TOLERANCE_K_99P = 1.0
NEWTON_MAX_ITERS = 16
NEWTON_BACKTRACK_REDUCTION = 0.5
NEWTON_BACKTRACK_CUTOFF = 1e-3
FIXED_POINT_MAX_ITERS = 100

# Refresh the LU preconditioner every N Newton iterations (or earlier on failure).
INEXACT_NEWTON_REFACTORIZE_EVERY = 4
# GMRES tolerance for inexact Newton linear solves.
INEXACT_NEWTON_GMRES_RTOL = 1e-4
INEXACT_NEWTON_GMRES_ATOL = 0.0
INEXACT_NEWTON_GMRES_RESTART = 50
INEXACT_NEWTON_GMRES_MAXITER = 50


type FloatArray = NDArray[np.floating]

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
        base_albedo_field = surface_context.base_albedo

        # Lag diagnostic fields during the Newton solve.
        #
        # The Newton Jacobian does not include derivatives of diagnostic closures
        # (snow albedo, humidity, winds) with respect to temperature. If we update
        # these fields on every residual evaluation, the solve becomes inconsistent
        # and can stall with a persistent residual. We therefore compute them once
        # from the previous (start-of-step) state and keep them fixed across
        # Newton/backtracking iterations.
        start_temp_capped = np.maximum(start_temp, temperature_floor)
        lagged_albedo_field = surface_context.albedo_model.apply_snow_albedo(
            base_albedo_field,
            start_temp_capped[0],
        )
        if start_temp_capped.shape[0] == 3:
            wind_temp_start = start_temp_capped[2]
        elif start_temp_capped.shape[0] == 2:
            wind_temp_start = start_temp_capped[1]
        else:
            wind_temp_start = start_temp_capped[0]
        lagged_wind_field = (
            surface_context.wind_model.wind_field(wind_temp_start, declination_rad=declination)
            if surface_context.wind_model
            else None
        )
        lagged_humidity_field = compute_humidity_q(
            surface_context.lat2d,
            start_temp_capped[0],
            declination,
            land_mask=surface_context.land_mask,
        )

        def _effective_surface_capacity(temp_surface: np.ndarray) -> np.ndarray:
            return surface_context.albedo_model.effective_heat_capacity_surface(
                temp_surface,
                land_mask=surface_context.land_mask,
                base_C_land=surface_context.base_C_land,
                base_C_ocean=surface_context.base_C_ocean,
            )
        
        def _init_state(temp: np.ndarray) -> ModelState:
            with time_block("_init_state"):
                albedo_field = lagged_albedo_field
                wind_field = lagged_wind_field
                humidity_field = lagged_humidity_field
                return ModelState(
                    temperature=temp,
                    albedo_field=albedo_field,
                    wind_field=wind_field,
                    humidity_field=humidity_field,
                )

        # implicit solver loop
        preconditioner_solve: Callable[[np.ndarray], np.ndarray] | None = None
        preconditioner_age = 10**9

        def _solve_linear_system(
            jacobian: sparse.csc_matrix,
            rhs: np.ndarray,
            *,
            preconditioner_matrix: sparse.csc_matrix | None = None,
            newton_iter: int,
        ) -> np.ndarray:
            """Inexact Newton linear solve: GMRES with a reused LU preconditioner.

            Notes
            -----
            - We refresh the LU preconditioner occasionally (and on GMRES failure).
            - We do NOT cache factorisations across iterations in the global cache,
              because Jacobian values change each iteration.
            """
            nonlocal preconditioner_solve, preconditioner_age

            if (preconditioner_solve is None) or (preconditioner_age >= INEXACT_NEWTON_REFACTORIZE_EVERY):
                with time_block("factorize_solver"):
                    matrix = preconditioner_matrix if preconditioner_matrix is not None else jacobian
                    preconditioner_solve = splinalg.factorized(matrix)
                preconditioner_age = 0

            assert preconditioner_solve is not None

            preconditioner = splinalg.LinearOperator(
                shape=jacobian.shape,
                matvec=preconditioner_solve,
                dtype=float,
            )
            with time_block("gmres_solve"):
                sol, info = splinalg.gmres(
                    jacobian,
                    rhs,
                    M=preconditioner,
                    rtol=INEXACT_NEWTON_GMRES_RTOL,
                    atol=INEXACT_NEWTON_GMRES_ATOL,
                    restart=INEXACT_NEWTON_GMRES_RESTART,
                    maxiter=INEXACT_NEWTON_GMRES_MAXITER,
                )

            if info != 0:
                # GMRES did not converge: refresh preconditioner and fall back to direct solve.
                with time_block("factorize_solver"):
                    preconditioner_solve = splinalg.factorized(jacobian)
                preconditioner_age = 0
                return preconditioner_solve(rhs)

            return sol

        for newton_iter in range(NEWTON_MAX_ITERS):
            with time_block("newton_iteration"):
                temp_capped = np.maximum(temp_next, temperature_floor)
                state_capped = _init_state(temp_capped)
                with time_block("rhs_evaluation"):
                    rhs_value = rhs_fn(state_capped, insolation_W_m2, declination)
                # Recompute Jacobian every iteration for true Newton solve
                with time_block("rhs_derivative"):
                    linearization = rhs_temperature_derivative_fn(state_capped, insolation_W_m2)
                preconditioner_age += 1

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
                        surface_block = _build_surface_jacobian_block(
                            ceff_surface, surface_diag, base_capacity, linearization.surface_diffusion_matrix, dt_seconds
                        )
                        jacobian = surface_block
                    with time_block("linear_solve"):
                        correction_flat = _solve_linear_system(
                            jacobian,
                            residual_flat,
                            preconditioner_matrix=surface_block,
                            newton_iter=newton_iter,
                        )
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
                    preconditioner_matrix = sparse.block_diag(
                        [surface_block, atmosphere_block],
                        format="csc",
                    )
                    with time_block("linear_solve"):
                        correction_flat = _solve_linear_system(
                            jacobian,
                            residual_flat,
                            preconditioner_matrix=preconditioner_matrix,
                            newton_iter=newton_iter,
                        )
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
                    preconditioner_matrix = sparse.block_diag(
                        [surface_block, boundary_block, atmosphere_block],
                        format="csc",
                    )
                    with time_block("linear_solve"):
                        correction_flat = _solve_linear_system(
                            jacobian,
                            residual_flat,
                            preconditioner_matrix=preconditioner_matrix,
                            newton_iter=newton_iter,
                        )
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


        # Return a state that is internally consistent with the converged temperature.
        # (Still lagged within the step, but updated for the returned diagnostic state.)
        final_temp = np.maximum(temp_next, temperature_floor)
        final_state = _init_state(final_temp)
        final_state.albedo_field = surface_context.albedo_model.apply_snow_albedo(base_albedo_field, final_temp[0])
        # Update diagnostics for output.
        if final_temp.shape[0] == 3:
            wind_temp_final = final_temp[2]
        elif final_temp.shape[0] == 2:
            wind_temp_final = final_temp[1]
        else:
            wind_temp_final = final_temp[0]
        if surface_context.wind_model:
            final_state.wind_field = surface_context.wind_model.wind_field(
                wind_temp_final, declination_rad=declination
            )
        final_state.humidity_field = compute_humidity_q(
            surface_context.lat2d,
            final_temp[0],
            declination,
            land_mask=surface_context.land_mask,
        )
        return final_state

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


def find_periodic_climate_cycle(
    initial_state: ModelState,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    rhs_fn: RhsFn,
    rhs_derivative_fn: RhsDerivativeFn,
    surface_context: SurfaceHeatCapacityContext,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
) -> list[ModelState]:
    """Solve for the periodic annual climate cycle using Anderson acceleration.

    Finds the fixed point P(state) = state where P is the year-forward
    evolution operator, using Anderson acceleration for faster convergence.
    """

    with time_block("find_periodic_climate_cycle"):
        state = initial_state
        temp = initial_state.temperature
        state.temperature = np.maximum(temp, temperature_floor)
        states = [initial_state] * 12
        residual_history: list[np.ndarray] = []
        advanced_history: list[np.ndarray] = []
        history_limit = 5
        residual_max = 0.0

        for iter_idx in range(FIXED_POINT_MAX_ITERS):
            with time_block("periodic_iteration"):
                advanced_states = evolve_year(
                    state,
                    monthly_insolation,
                    month_durations,
                    rhs_fn=rhs_fn,
                    rhs_temperature_derivative_fn=rhs_derivative_fn,
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
                    return [advanced_states[(i - 2) % 12] for i in range(12)]

                states = advanced_states

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
                            T_next = advanced.temperature
                            residual_history = residual_history[-1:]
                            advanced_history = advanced_history[-1:]

                # Actually apply Anderson-accelerated state (fixing old bug where this was never used)
                state = ModelState(
                    temperature=T_next,
                    albedo_field=initial_state.albedo_field,
                    wind_field=None,
                    humidity_field=None,
                )

        raise RuntimeError(
            "Failed to converge to a periodic solution after "
            f"{FIXED_POINT_MAX_ITERS} iterations (last residual {residual_max:.3e} K)"
        )

def solve_periodic_climate(
    resolution_deg: float = 1.0,
    *,
    model_config: ModelConfig,
    return_layer_map: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Solve for the annual periodic climate cycle.

    This is the main entry point for running the climate model. It handles
    the complete workflow: building operators from configuration, solving
    for the periodic state, and post-processing results.

    Parameters
    ----------
    resolution_deg : float, default=1.0
        Grid resolution in degrees.
    model_config : ModelConfig
        Complete model configuration.
    return_layer_map : bool, default=False
        If True, return temperature fields for all layers. If False, return
        only surface temperature.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray] or tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]
        Longitude grid, latitude grid, and either:
        - Surface temperature field in Celsius (if return_layer_map=False)
        - Dictionary mapping layer names to temperature fields (if return_layer_map=True)
    """
    from climate_sim.core.operators import build_model_operators

    reset_profiler()

    # Build all operators and grids from configuration
    with time_block("build_operators"):
        operators = build_model_operators(resolution_deg, model_config)

    # Build RHS functions from operators
    with time_block("build_rhs"):
        rhs_inputs = RhsBuildInputs(
            heat_capacity_field=operators.heat_capacity_field,
            diffusion_operator=operators.diffusion_operator,
            radiation_config=operators.radiation_config,
            land_mask=operators.land_mask,
            roughness_length=operators.roughness_length,
            topographic_elevation=operators.topographic_elevation,
            sensible_heat_cfg=operators.sensible_heat_cfg,
            latent_heat_cfg=operators.latent_heat_cfg,
            boundary_layer_cfg=operators.boundary_layer_cfg,
            wind_model=operators.wind_model,
            advection_operator=operators.advection_operator,
            lon2d=operators.lon2d,
            lat2d=operators.lat2d,
        )
        rhs_fn, rhs_derivative_fn = create_rhs_functions(rhs_inputs)

    # Create initial guess using radiative equilibrium
    with time_block("initial_guess"):
        initial_temp_guess = radiation.radiative_equilibrium_initial_guess(
            operators.monthly_insolation,
            albedo_field=operators.base_albedo_field,
            config=operators.radiation_config,
            land_mask=operators.land_mask,
        )
        initial_state = ModelState(
            temperature=initial_temp_guess,
            albedo_field=operators.base_albedo_field,
            wind_field=None,
            humidity_field=None,
        )

    # Solve for periodic cycle
    monthly_states = find_periodic_climate_cycle(
        initial_state=initial_state,
        monthly_insolation=operators.monthly_insolation,
        month_durations=operators.month_durations,
        rhs_fn=rhs_fn,
        rhs_derivative_fn=rhs_derivative_fn,
        surface_context=operators.surface_context,
        temperature_floor=operators.radiation_config.temperature_floor,
        solver_cache=operators.solver_cache,
    )

    # Update wind fields if wind model is enabled
    if operators.wind_model is not None and operators.wind_model.enabled:
        with time_block("update_wind_fields"):
            monthly_declinations = compute_monthly_declinations()
            for idx, month_state in enumerate(monthly_states):
                wind_temperature = select_wind_temperature(month_state.temperature)
                wind_field = month_state.wind_field or operators.wind_model.wind_field(
                    wind_temperature, declination_rad=monthly_declinations[idx]
                )
                monthly_states[idx] = ModelState(
                    temperature=month_state.temperature,
                    albedo_field=month_state.albedo_field,
                    wind_field=wind_field,
                    humidity_field=month_state.humidity_field,
                )

    # Post-process results (convert to Celsius, reshape layers)
    get_profiler().print_summary()

    return postprocess_periodic_cycle_results(
        operators.lon2d,
        operators.lat2d,
        monthly_states,
        resolved_wind=model_config.wind,
        resolved_radiation=operators.radiation_config,
        return_layer_map=return_layer_map,
    )
