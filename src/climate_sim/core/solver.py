"""General-purpose periodic solver utilities for energy-balance models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import os
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
# Default high so we typically factorize once per monthly step (and rely on GMRES failure
# to trigger an earlier refresh if needed). Allow env override for tuning.
INEXACT_NEWTON_REFACTORIZE_EVERY = int(os.environ.get("INEXACT_NEWTON_REFACTORIZE_EVERY", "1000"))
# GMRES tolerance for inexact Newton linear solves.
INEXACT_NEWTON_GMRES_RTOL = float(os.environ.get("INEXACT_NEWTON_GMRES_RTOL", "5e-4"))
INEXACT_NEWTON_GMRES_ATOL = float(os.environ.get("INEXACT_NEWTON_GMRES_ATOL", "0.0"))
INEXACT_NEWTON_GMRES_RESTART = int(os.environ.get("INEXACT_NEWTON_GMRES_RESTART", "50"))
INEXACT_NEWTON_GMRES_MAXITER = int(os.environ.get("INEXACT_NEWTON_GMRES_MAXITER", "50"))

# Modified Newton option: reuse Jacobian/linearization for a few iterations within a monthly step.
# Default 1 => exact Newton (rebuild each iteration).
MODIFIED_NEWTON_JACOBIAN_EVERY = int(os.environ.get("MODIFIED_NEWTON_JACOBIAN_EVERY", "1"))

# Linear solve strategy for the inexact Newton step.
# - "gmres" (default): GMRES on the current Jacobian with LU-preconditioner reuse.
# - "lu": use the (possibly stale) LU solve directly as an approximate Newton step.
INEXACT_NEWTON_LINEAR_SOLVER = os.environ.get("INEXACT_NEWTON_LINEAR_SOLVER", "gmres").strip().lower()

# Krylov method for the Newton linear solve (when INEXACT_NEWTON_LINEAR_SOLVER="gmres").
# - "lgmres" (default; often faster for sequences of similar systems)
# - "lgmres" (can be faster for sequences of similar systems)
INEXACT_NEWTON_KRYLOV = os.environ.get("INEXACT_NEWTON_KRYLOV", "lgmres").strip().lower()

# Preconditioner for GMRES:
# - "lu" (default): full LU factorization via scipy's SuperLU (`factorized`)
# - "ilu": incomplete LU (`spilu`) with configurable drop tolerance/fill
INEXACT_NEWTON_PRECONDITIONER = os.environ.get("INEXACT_NEWTON_PRECONDITIONER", "lu").strip().lower()
INEXACT_NEWTON_ILU_DROP_TOL = float(os.environ.get("INEXACT_NEWTON_ILU_DROP_TOL", "1e-4"))
INEXACT_NEWTON_ILU_FILL_FACTOR = float(os.environ.get("INEXACT_NEWTON_ILU_FILL_FACTOR", "10"))

# Periodic solver method:
# - "anderson" (default): Anderson acceleration on year-forward map (current behavior).
# - "broyden" : quasi-Newton on fixed point F(T)=P(T)-T with L-BFGS inverse-Jacobian approximation.
PERIODIC_SOLVER_METHOD = os.environ.get("PERIODIC_SOLVER_METHOD", "anderson").strip().lower()

# Quasi-Newton tuning (only used when PERIODIC_SOLVER_METHOD="broyden")
PERIODIC_BROYDEN_HISTORY = int(os.environ.get("PERIODIC_BROYDEN_HISTORY", "6"))
PERIODIC_BROYDEN_DAMPING = float(os.environ.get("PERIODIC_BROYDEN_DAMPING", "1.0"))
PERIODIC_BROYDEN_MAX_STEP_K = float(os.environ.get("PERIODIC_BROYDEN_MAX_STEP_K", "5.0"))
PERIODIC_BROYDEN_TEMP_CEILING_K = float(os.environ.get("PERIODIC_BROYDEN_TEMP_CEILING_K", "400.0"))
PERIODIC_BROYDEN_FALLBACK_TO_ANDERSON = os.environ.get(
    "PERIODIC_BROYDEN_FALLBACK_TO_ANDERSON", "1"
).strip() in {"1", "true", "True"}

# Periodicity-aware acceleration for Anderson mode (opt-in):
# propose an accelerated iterate but only accept it if an *extra* year-advance
# demonstrates that it reduces the cycle-to-cycle residual (same metric you
# already use for convergence). This avoids basin-jumping without guessing.
PERIODIC_ACCEPT_REJECT_ACCELERATION = os.environ.get(
    "PERIODIC_ACCEPT_REJECT_ACCELERATION", "0"
).strip() in {"1", "true", "True"}
PERIODIC_ACCEL_MAX_STEP_FROM_BASE_K = float(os.environ.get("PERIODIC_ACCEL_MAX_STEP_FROM_BASE_K", "1.0"))
PERIODIC_ACCEL_MIN_RESIDUAL_99P_K = float(os.environ.get("PERIODIC_ACCEL_MIN_RESIDUAL_99P_K", "2.0"))
PERIODIC_ACCEL_ATTEMPT_EVERY = int(os.environ.get("PERIODIC_ACCEL_ATTEMPT_EVERY", "5"))

# Cheap safeguarded mixing (no extra year-advance): apply a small step toward the
# Anderson proposal, and disable it temporarily if residual grows.
PERIODIC_SAFE_MIXING = os.environ.get("PERIODIC_SAFE_MIXING", "0").strip() in {"1", "true", "True"}
PERIODIC_SAFE_MIXING_MAX_STEP_K = float(os.environ.get("PERIODIC_SAFE_MIXING_MAX_STEP_K", "0.5"))
PERIODIC_SAFE_MIXING_START_99P_K = float(os.environ.get("PERIODIC_SAFE_MIXING_START_99P_K", "3.0"))
PERIODIC_SAFE_MIXING_REJECT_GROWTH = float(os.environ.get("PERIODIC_SAFE_MIXING_REJECT_GROWTH", "0.10"))
PERIODIC_SAFE_MIXING_COOLDOWN = int(os.environ.get("PERIODIC_SAFE_MIXING_COOLDOWN", "2"))

# Monolithic 12-month periodic solve (Newton on the whole cycle).
#
# Unknowns are the temperature fields at each month boundary (12 states). Residuals enforce:
#   T_{m+1} - M_m(T_m) = 0  (with wrap-around)
# where M_m is the implicit monthly step operator.
#
# This is opt-in since it changes the outer algorithm.
PERIODIC_CYCLE_NEWTON_MAX_ITERS = int(os.environ.get("PERIODIC_CYCLE_NEWTON_MAX_ITERS", "6"))
PERIODIC_CYCLE_NEWTON_MAX_STEP_K = float(os.environ.get("PERIODIC_CYCLE_NEWTON_MAX_STEP_K", "1.0"))
PERIODIC_CYCLE_NEWTON_LINEAR_RTOL = float(os.environ.get("PERIODIC_CYCLE_NEWTON_LINEAR_RTOL", "1e-2"))
PERIODIC_CYCLE_NEWTON_LINEAR_MAXITER = int(os.environ.get("PERIODIC_CYCLE_NEWTON_LINEAR_MAXITER", "10"))
PERIODIC_CYCLE_NEWTON_FALLBACK_TO_ANDERSON = os.environ.get(
    "PERIODIC_CYCLE_NEWTON_FALLBACK_TO_ANDERSON", "1"
).strip() in {"1", "true", "True"}

# Coarse-grid warm start for the periodic solve (reduced-order acceleration).
#
# If enabled and resolution_deg > coarse_deg, we solve the periodic cycle on a coarser
# grid once, then bilinearly interpolate a single-month temperature field onto the
# target grid as the initial guess.
PERIODIC_WARM_START_COARSE_DEG = float(os.environ.get("PERIODIC_WARM_START_COARSE_DEG", "0"))

# Continuation / homotopy on selected physics strength (opt-in):
# runs a sequence of periodic solves with scaled tendencies, using the previous
# stage's solution as the initial guess for the next.
PERIODIC_CONTINUATION_SCALES = os.environ.get("PERIODIC_CONTINUATION_SCALES", "").strip()
PERIODIC_CONTINUATION_TARGETS = os.environ.get("PERIODIC_CONTINUATION_TARGETS", "sensible,latent").strip().lower()
PERIODIC_CONTINUATION_ADAPTIVE = os.environ.get("PERIODIC_CONTINUATION_ADAPTIVE", "0").strip() in {"1", "true", "True"}
PERIODIC_CONTINUATION_INITIAL_STEP = float(os.environ.get("PERIODIC_CONTINUATION_INITIAL_STEP", "0.5"))
PERIODIC_CONTINUATION_MIN_STEP = float(os.environ.get("PERIODIC_CONTINUATION_MIN_STEP", "0.05"))
PERIODIC_CONTINUATION_MAX_STEP = float(os.environ.get("PERIODIC_CONTINUATION_MAX_STEP", "0.5"))
PERIODIC_CONTINUATION_GROWTH = float(os.environ.get("PERIODIC_CONTINUATION_GROWTH", "1.5"))
PERIODIC_CONTINUATION_SHRINK = float(os.environ.get("PERIODIC_CONTINUATION_SHRINK", "0.5"))
PERIODIC_CONTINUATION_MAX_DELTA_99P_K = float(os.environ.get("PERIODIC_CONTINUATION_MAX_DELTA_99P_K", "3.0"))
PERIODIC_CONTINUATION_MAX_DELTA_MAX_K = float(os.environ.get("PERIODIC_CONTINUATION_MAX_DELTA_MAX_K", "10.0"))


type FloatArray = NDArray[np.floating]

type RhsFactory = Callable[[RhsBuildInputs], tuple[RhsFn, RhsDerivativeFn]]
type InitialGuessFn = Callable[[FloatArray, FloatArray, RadiationConfig, FloatArray], FloatArray]
type MonthlyInsolationLatFn = Callable[[FloatArray], FloatArray]
type HeatCapacityFieldFn = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class MonthSensitivity:
    """Linearised month map sensitivity: dT_next = S(dT_start)."""

    apply: Callable[[np.ndarray], np.ndarray]


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
    return_sensitivity: bool = False,
) -> ModelState | tuple[ModelState, MonthSensitivity]:
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
        # Reuse the last accepted damping factor as a hint for backtracking.
        # We still *always* try the full Newton step first to preserve convergence behavior.
        damping_hint = 1.0
        # Cached Jacobian/linearization for modified Newton (within this monthly step).
        cached_linearization = None
        cached_jacobian: sparse.csc_matrix | None = None
        cached_for_iter: int | None = None
        # LGMRES can reuse a small subspace across solves.
        krylov_outer_v: list[np.ndarray] = []
        # For periodic-cycle Newton methods, expose the local month map sensitivity.
        last_jacobian: sparse.csc_matrix | None = None
        last_direct_solve: Callable[[np.ndarray], np.ndarray] | None = None

        def _solve_linear_system(
            jacobian: sparse.csc_matrix,
            rhs: np.ndarray,
            *,
            newton_iter: int,
        ) -> np.ndarray:
            """Inexact Newton linear solve: GMRES with a reused LU preconditioner.

            Notes
            -----
            - We refresh the LU preconditioner occasionally (and on GMRES failure).
            - We do NOT cache factorisations across iterations in the global cache,
              because Jacobian values change each iteration.
            """
            nonlocal preconditioner_solve, preconditioner_age, last_direct_solve

            if (preconditioner_solve is None) or (preconditioner_age >= INEXACT_NEWTON_REFACTORIZE_EVERY):
                with time_block("factorize_solver"):
                    if INEXACT_NEWTON_PRECONDITIONER == "ilu":
                        try:
                            ilu = splinalg.spilu(
                                jacobian,
                                drop_tol=INEXACT_NEWTON_ILU_DROP_TOL,
                                fill_factor=INEXACT_NEWTON_ILU_FILL_FACTOR,
                            )
                            preconditioner_solve = ilu.solve
                        except RuntimeError:
                            # ILU can fail on (near-)singular Jacobians; fall back to full LU.
                            preconditioner_solve = splinalg.factorized(jacobian)
                    else:
                        preconditioner_solve = splinalg.factorized(jacobian)
                preconditioner_age = 0

            assert preconditioner_solve is not None
            if return_sensitivity and INEXACT_NEWTON_PRECONDITIONER != "ilu":
                # With LU preconditioning this is the exact solve for the current Jacobian.
                last_direct_solve = preconditioner_solve

            if INEXACT_NEWTON_LINEAR_SOLVER == "lu":
                # Modified Newton / quasi-Newton step: apply the cached LU solve directly.
                # This avoids GMRES iterations at the cost of using a potentially stale Jacobian.
                return preconditioner_solve(rhs)

            preconditioner = splinalg.LinearOperator(
                shape=jacobian.shape,
                matvec=preconditioner_solve,
                dtype=float,
            )
            with time_block("gmres_solve"):
                if INEXACT_NEWTON_KRYLOV == "lgmres":
                    sol, info = splinalg.lgmres(
                        jacobian,
                        rhs,
                        M=preconditioner,
                        rtol=INEXACT_NEWTON_GMRES_RTOL,
                        atol=INEXACT_NEWTON_GMRES_ATOL,
                        maxiter=INEXACT_NEWTON_GMRES_MAXITER,
                        inner_m=INEXACT_NEWTON_GMRES_RESTART,
                        outer_v=krylov_outer_v,
                        store_outer_Av=True,
                    )
                else:
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
                    direct_solve = splinalg.factorized(jacobian)
                preconditioner_solve = direct_solve
                if return_sensitivity:
                    last_direct_solve = direct_solve
                preconditioner_age = 0
                return direct_solve(rhs)

            return sol

        for newton_iter in range(NEWTON_MAX_ITERS):
            with time_block("newton_iteration"):
                temp_capped = np.maximum(temp_next, temperature_floor)
                state_capped = _init_state(temp_capped)
                with time_block("rhs_evaluation"):
                    rhs_value = rhs_fn(state_capped, insolation_W_m2, declination)
                # Jacobian/linearization (optionally reused for modified Newton)
                needs_relinearize = (
                    cached_linearization is None
                    or cached_for_iter is None
                    or MODIFIED_NEWTON_JACOBIAN_EVERY <= 1
                    or (newton_iter - cached_for_iter) >= MODIFIED_NEWTON_JACOBIAN_EVERY
                )
                if needs_relinearize:
                    with time_block("rhs_derivative"):
                        linearization = rhs_temperature_derivative_fn(state_capped, insolation_W_m2)
                    cached_linearization = linearization
                    cached_for_iter = newton_iter
                    cached_jacobian = None
                else:
                    linearization = cached_linearization
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
                    
                    if cached_jacobian is None:
                        with time_block("jacobian_assembly"):
                            jacobian = _build_surface_jacobian_block(
                                ceff_surface, surface_diag, base_capacity, linearization.surface_diffusion_matrix, dt_seconds
                            )
                        cached_jacobian = jacobian
                    else:
                        jacobian = cached_jacobian
                    last_jacobian = jacobian
                    with time_block("linear_solve"):
                        correction_flat = _solve_linear_system(jacobian, residual_flat, newton_iter=newton_iter)
                    correction = correction_flat.reshape((nlat, nlon))[np.newaxis, :, :]
                elif nlayers == 2:
                    # Two-layer: surface + atmosphere with coupling
                    atmosphere_diag = linearization.diag[1]
                    residual_atmosphere = temp_capped[1] - start_temp[1] - dt_seconds * rhs_value[1]
                    residual = np.stack([residual_surface, residual_atmosphere])

                    identity = get_identity_matrix(size, cache=cache)

                    if cached_jacobian is None:
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
                        cached_jacobian = jacobian
                    else:
                        jacobian = cached_jacobian

                    residual_flat = np.concatenate([residual_surface.ravel(), residual_atmosphere.ravel()], axis=0)
                    last_jacobian = jacobian
                    with time_block("linear_solve"):
                        correction_flat = _solve_linear_system(jacobian, residual_flat, newton_iter=newton_iter)
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

                    if cached_jacobian is None:
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
                        cached_jacobian = jacobian
                    else:
                        jacobian = cached_jacobian

                    residual_flat = np.concatenate([
                        residual_surface.ravel(),
                        residual_boundary.ravel(),
                        residual_atmosphere.ravel()
                    ], axis=0)
                    last_jacobian = jacobian
                    with time_block("linear_solve"):
                        correction_flat = _solve_linear_system(jacobian, residual_flat, newton_iter=newton_iter)
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

                # Always attempt the full Newton step first. If that fails and we have a
                # useful hint from a previous accepted step, jump directly to it to avoid
                # extra RHS evaluations at intermediate damping values.
                damping = 1.0
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
                        damping_hint = damping
                        break

                    if damping == 1.0 and damping_hint < 1.0:
                        damping = float(max(NEWTON_BACKTRACK_CUTOFF, damping_hint))
                    else:
                        damping *= NEWTON_BACKTRACK_REDUCTION

                if not accepted:
                    temp_next = temp_candidate
                    residual = residual_candidate
                    damping_hint = float(min(1.0, max(NEWTON_BACKTRACK_CUTOFF, damping)))

                step = prev_temp - temp_next
                if np.max(np.abs(step)) < NEWTON_STEP_TOLERANCE_K:
                    break

        # (No cross-month preconditioner caching: it proved too stale and changed results.)


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
        if not return_sensitivity:
            return final_state

        if last_jacobian is None:
            raise RuntimeError("Missing Jacobian for monthly_step sensitivity")

        ceff_final = _effective_surface_capacity(final_temp[0])
        nlayers_final = int(final_temp.shape[0])
        nlat, nlon = int(final_temp.shape[1]), int(final_temp.shape[2])
        size = nlat * nlon

        direct_solve = last_direct_solve
        if direct_solve is None:
            # Fallback: compute a direct solve from the last Jacobian.
            direct_solve = splinalg.factorized(last_jacobian)

        def _apply_sensitivity(dtemp: np.ndarray) -> np.ndarray:
            dtemp_arr = np.asarray(dtemp, dtype=float)
            if dtemp_arr.shape != final_temp.shape:
                raise ValueError("Sensitivity input must match temperature shape")
            rhs_blocks: list[np.ndarray] = []
            rhs_blocks.append((ceff_final * dtemp_arr[0]).ravel())
            if nlayers_final >= 2:
                rhs_blocks.append(dtemp_arr[1].ravel())
            if nlayers_final >= 3:
                rhs_blocks.append(dtemp_arr[2].ravel())
            rhs_flat = np.concatenate(rhs_blocks, axis=0)
            out_flat = direct_solve(rhs_flat)
            if nlayers_final == 1:
                out = out_flat.reshape((nlat, nlon))[np.newaxis, :, :]
            elif nlayers_final == 2:
                out0 = out_flat[:size].reshape((nlat, nlon))
                out1 = out_flat[size:].reshape((nlat, nlon))
                out = np.stack([out0, out1])
            else:
                out0 = out_flat[:size].reshape((nlat, nlon))
                out1 = out_flat[size:2 * size].reshape((nlat, nlon))
                out2 = out_flat[2 * size:].reshape((nlat, nlon))
                out = np.stack([out0, out1, out2])
            return out

        return final_state, MonthSensitivity(apply=_apply_sensitivity)

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


def _lbfgs_apply_inverse(
    vec: np.ndarray,
    *,
    s_list: list[np.ndarray],
    y_list: list[np.ndarray],
) -> np.ndarray:
    """Apply an L-BFGS inverse-Jacobian approximation to *vec*.

    This is a limited-memory quasi-Newton approximation suitable for very large
    state vectors. It stores only the last few (s, y) pairs where:
      s_k = x_{k+1} - x_k
      y_k = F_{k+1} - F_k

    Notes
    -----
    - Uses the standard two-loop recursion.
    - Uses a conservative scalar H0 based on the most recent (s, y) pair.
    """
    if not s_list or not y_list:
        return vec
    # Two-loop recursion
    q = vec.copy()
    alpha: list[float] = []
    rho: list[float] = []
    for s, y in zip(reversed(s_list), reversed(y_list), strict=True):
        ys = float(np.dot(y, s))
        if not np.isfinite(ys) or abs(ys) < 1e-20:
            alpha.append(0.0)
            rho.append(0.0)
            continue
        r = 1.0 / ys
        rho.append(r)
        a = r * float(np.dot(s, q))
        alpha.append(a)
        q = q - a * y

    # Initial scaling
    s_last = s_list[-1]
    y_last = y_list[-1]
    yy = float(np.dot(y_last, y_last))
    ys_last = float(np.dot(y_last, s_last))
    if np.isfinite(yy) and yy > 1e-20 and np.isfinite(ys_last):
        gamma = max(1e-6, ys_last / yy)
    else:
        gamma = 1.0
    r_vec = gamma * q

    for s, y, a, r in zip(s_list, y_list, reversed(alpha), reversed(rho), strict=True):
        if r == 0.0:
            continue
        b = r * float(np.dot(y, r_vec))
        r_vec = r_vec + s * (a - b)
    return r_vec


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
        def _solve_anderson(initial_state_for_anderson: ModelState) -> list[ModelState]:
            """Previous Anderson-based solver (kept as reference + fallback)."""
            state = initial_state_for_anderson
            temp = initial_state_for_anderson.temperature
            state.temperature = np.maximum(temp, temperature_floor)
            states = [initial_state_for_anderson] * 12
            residual_history: list[np.ndarray] = []
            advanced_history: list[np.ndarray] = []
            history_limit = 5
            residual_max = 0
            prev_residual_99p: float | None = None
            mixing_cooldown = 0

            for iter_idx in range(FIXED_POINT_MAX_ITERS):
                with time_block("periodic_iteration"):
                    start_temperature = state.temperature
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
                    residual = np.array(
                        [advanced_states[i].temperature[0] - states[i].temperature[0] for i in range(12)]
                    )
                    residual_rms = np.sqrt(np.mean(np.square(residual)))
                    residual_99p = np.percentile(np.abs(residual), 99)
                    residual_max = np.max(np.abs(residual))
                    print(residual_rms, residual_99p, residual_max)

                    if residual_rms < PERIODIC_FIXED_POINT_TOLERANCE_K and residual_99p < PERIODIC_FIXED_POINT_TOLERANCE_K_99P:
                        return [advanced_states[(i - 2) % 12] for i in range(12)]

                    if prev_residual_99p is not None and np.isfinite(prev_residual_99p):
                        if float(residual_99p) > (1.0 + PERIODIC_SAFE_MIXING_REJECT_GROWTH) * float(prev_residual_99p):
                            mixing_cooldown = PERIODIC_SAFE_MIXING_COOLDOWN
                    prev_residual_99p = float(residual_99p)

                    # For acceleration, use the *fixed-point* residual f_k = P(x_k) - x_k
                    # (end-of-year only). This matches standard Anderson acceleration.
                    fixed_point_residual_flat = (advanced.temperature - start_temperature).ravel()
                    advanced_flat = advanced.temperature.ravel()

                    if len(residual_history) == history_limit:
                        residual_history.pop(0)
                        advanced_history.pop(0)

                    residual_history.append(fixed_point_residual_flat)
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
                                T_next = advanced.temperature

                    # Default: advance using the unaccelerated map (stable fixed-point iteration).
                    accepted_states = advanced_states
                    accepted_state = advanced_states[-1]

                    # Cheap safeguarded mixing: small step toward the Anderson proposal (no extra evolve_year).
                    if (
                        PERIODIC_SAFE_MIXING
                        and coefficients is not None
                        and mixing_cooldown <= 0
                        and float(residual_99p) >= PERIODIC_SAFE_MIXING_START_99P_K
                    ):
                        delta = T_next - accepted_state.temperature
                        max_abs = float(np.max(np.abs(delta)))
                        if np.isfinite(max_abs) and max_abs > 0.0:
                            scale = min(1.0, PERIODIC_SAFE_MIXING_MAX_STEP_K / max_abs)
                        else:
                            scale = 0.0
                        proposal_limited = accepted_state.temperature + scale * delta
                        proposal_limited = np.maximum(proposal_limited, temperature_floor)
                        accepted_state = ModelState(
                            temperature=proposal_limited,
                            albedo_field=accepted_state.albedo_field,
                            wind_field=accepted_state.wind_field,
                            humidity_field=accepted_state.humidity_field,
                        )
                    else:
                        if mixing_cooldown > 0:
                            mixing_cooldown -= 1

                    # Accept/reject accelerated proposal (extra year-advance, so keep it rare).
                    if (
                        PERIODIC_ACCEPT_REJECT_ACCELERATION
                        and coefficients is not None
                        and residual_99p >= PERIODIC_ACCEL_MIN_RESIDUAL_99P_K
                        and (PERIODIC_ACCEL_ATTEMPT_EVERY > 0)
                        and (iter_idx % PERIODIC_ACCEL_ATTEMPT_EVERY == 0)
                    ):
                        base_next = advanced.temperature
                        delta = T_next - base_next
                        max_abs = float(np.max(np.abs(delta)))
                        if np.isfinite(max_abs) and max_abs > 0.0:
                            scale = min(1.0, PERIODIC_ACCEL_MAX_STEP_FROM_BASE_K / max_abs)
                        else:
                            scale = 0.0
                        proposal_limited = base_next + scale * delta
                        proposal_limited = np.maximum(proposal_limited, temperature_floor)

                        proposal_state = ModelState(
                            temperature=proposal_limited,
                            albedo_field=accepted_state.albedo_field,
                            wind_field=accepted_state.wind_field,
                            humidity_field=accepted_state.humidity_field,
                        )
                        proposal_states = evolve_year(
                            proposal_state,
                            monthly_insolation,
                            month_durations,
                            rhs_fn=rhs_fn,
                            rhs_temperature_derivative_fn=rhs_derivative_fn,
                            temperature_floor=temperature_floor,
                            solver_cache=solver_cache,
                            surface_context=surface_context,
                        )
                        proposal_residual = np.array(
                            [proposal_states[i].temperature[0] - states[i].temperature[0] for i in range(12)]
                        )
                        proposal_rms = float(np.sqrt(np.mean(np.square(proposal_residual))))
                        proposal_99p = float(np.percentile(np.abs(proposal_residual), 99))

                        if np.isfinite(proposal_rms) and np.isfinite(proposal_99p):
                            if (proposal_99p < float(residual_99p)) and (proposal_rms < float(residual_rms)):
                                accepted_states = proposal_states
                                accepted_state = proposal_states[-1]

                    states = accepted_states
                    state = accepted_state

            raise RuntimeError(
                "Failed to converge to a periodic solution after "
                f"{FIXED_POINT_MAX_ITERS} iterations (last residual {residual_max:.3e} K)"
            )

        if PERIODIC_SOLVER_METHOD == "cycle_newton":
            # Monolithic solve for the 12-month periodic cycle.
            #
            # Unknowns are the month-boundary temperatures for the March-start sequence used
            # by evolve_year: months = [Mar, Apr, ..., Feb]. Residuals enforce
            #   T_{k+1} - monthly_step_k(T_k) = 0 with wraparound.
            #
            # We use the implicit-step Jacobian at convergence to build a *local* sensitivity
            # operator S_k ≈ dT_{k+1}/dT_k, and solve the resulting cyclic block system.
            try:
                months_seq = [(m + 2) % 12 for m in range(12)]
                monthly_declinations = compute_monthly_declinations()
                base_albedo = surface_context.base_albedo

                # Start from one year-forward pass (same warm start as Anderson uses internally).
                warm_states = evolve_year(
                    initial_state,
                    monthly_insolation,
                    month_durations,
                    rhs_fn=rhs_fn,
                    rhs_temperature_derivative_fn=rhs_derivative_fn,
                    temperature_floor=temperature_floor,
                    solver_cache=solver_cache,
                    surface_context=surface_context,
                )
                temps = [s.temperature.copy() for s in warm_states]
                shape = temps[0].shape

                def _state_from_temp(temp: np.ndarray) -> ModelState:
                    return ModelState(
                        temperature=np.maximum(np.asarray(temp, dtype=float), temperature_floor),
                        albedo_field=base_albedo,
                        wind_field=None,
                        humidity_field=None,
                    )

                for iter_idx in range(PERIODIC_CYCLE_NEWTON_MAX_ITERS):
                    with time_block("periodic_iteration"):
                        best_99p = float("inf") if iter_idx == 0 else best_99p
                        preds: list[np.ndarray] = []
                        sens: list[MonthSensitivity] = []

                        for k in range(12):
                            month = months_seq[k]
                            start_state = _state_from_temp(temps[k])
                            step_out = monthly_step(
                                start_state,
                                monthly_insolation[month],
                                monthly_declinations[month],
                                month_durations[month],
                                rhs_fn=rhs_fn,
                                rhs_temperature_derivative_fn=rhs_derivative_fn,
                                temperature_floor=temperature_floor,
                                solver_cache=solver_cache,
                                surface_context=surface_context,
                                return_sensitivity=True,
                            )
                            next_state, sens_k = step_out
                            preds.append(next_state.temperature)
                            sens.append(sens_k)

                        residuals = [temps[(k + 1) % 12] - preds[k] for k in range(12)]
                        surface_residual = np.stack([r[0] for r in residuals], axis=0)
                        residual_rms = float(np.sqrt(np.mean(np.square(surface_residual))))
                        residual_99p = float(np.percentile(np.abs(surface_residual), 99))
                        residual_max = float(np.max(np.abs(surface_residual)))
                        print(residual_rms, residual_99p, residual_max)

                        if residual_rms < PERIODIC_FIXED_POINT_TOLERANCE_K and residual_99p < PERIODIC_FIXED_POINT_TOLERANCE_K_99P:
                            # Recompute a consistent diagnostic cycle for output.
                            out_states = evolve_year(
                                _state_from_temp(temps[0]),
                                monthly_insolation,
                                month_durations,
                                rhs_fn=rhs_fn,
                                rhs_temperature_derivative_fn=rhs_derivative_fn,
                                temperature_floor=temperature_floor,
                                solver_cache=solver_cache,
                                surface_context=surface_context,
                            )
                            return [out_states[(i - 2) % 12] for i in range(12)]

                        # If the monolithic iteration isn't improving quickly, stop early and fall back.
                        if residual_99p >= best_99p * 0.995:
                            # no meaningful improvement
                            if iter_idx >= 2:
                                raise RuntimeError("Cycle-Newton stalled")
                        best_99p = min(best_99p, residual_99p)

                        # Build the cyclic closure equation for delta_T0:
                        # (I - S11...S0) d0 = -[R11 + S11(R10 + S10(...(R1 + S1 R0))...)]
                        acc = residuals[0]
                        for k in range(1, 12):
                            acc = residuals[k] + sens[k].apply(acc)
                        rhs0 = (-acc).ravel()

                        n = rhs0.size

                        def _apply_A(vec_flat: np.ndarray) -> np.ndarray:
                            vec = vec_flat.reshape(shape)
                            out = vec
                            for k in range(12):
                                out = sens[k].apply(out)
                            return out.ravel()

                        def _matvec(vec_flat: np.ndarray) -> np.ndarray:
                            return vec_flat - _apply_A(vec_flat)

                        linop = splinalg.LinearOperator(
                            shape=(n, n),
                            matvec=_matvec,
                            dtype=float,
                        )

                        with time_block("anderson_acceleration"):
                            d0_flat, info = splinalg.lgmres(
                                linop,
                                rhs0,
                                rtol=PERIODIC_CYCLE_NEWTON_LINEAR_RTOL,
                                atol=0.0,
                                maxiter=PERIODIC_CYCLE_NEWTON_LINEAR_MAXITER,
                                inner_m=20,
                            )
                        if info != 0 and not np.all(np.isfinite(d0_flat)):
                            raise RuntimeError(f"Cycle-Newton linear solve failed (info={info})")

                        deltas: list[np.ndarray] = [np.zeros_like(temps[0]) for _ in range(12)]
                        deltas[0] = d0_flat.reshape(shape)
                        for k in range(0, 11):
                            deltas[k + 1] = sens[k].apply(deltas[k]) - residuals[k]

                        # Step limiting to stay in the same basin of attraction.
                        max_step = float(max(np.max(np.abs(d)) for d in deltas))
                        if not np.isfinite(max_step) or max_step <= 0.0:
                            step_scale = 0.0
                        else:
                            step_scale = min(1.0, PERIODIC_CYCLE_NEWTON_MAX_STEP_K / max_step)

                        for k in range(12):
                            temps[k] = temps[k] + step_scale * deltas[k]
                            temps[k] = np.maximum(temps[k], temperature_floor)

                raise RuntimeError(
                    f"Cycle-Newton did not converge after {PERIODIC_CYCLE_NEWTON_MAX_ITERS} iterations"
                )
            except Exception:
                if PERIODIC_CYCLE_NEWTON_FALLBACK_TO_ANDERSON:
                    return _solve_anderson(initial_state)
                raise

        if PERIODIC_SOLVER_METHOD == "broyden":
            # Quasi-Newton on the end-of-year fixed point F(T) = P(T) - T.
            #
            # We still monitor the same "cycle-to-cycle" residual statistics used by
            # the Anderson method to ensure we converge to the same accuracy target.
            state = initial_state
            temp0 = np.maximum(initial_state.temperature, temperature_floor)
            state.temperature = temp0

            x = temp0.ravel().copy()
            shape = temp0.shape

            # L-BFGS history for inverse-Jacobian approximation of F.
            s_list: list[np.ndarray] = []
            y_list: list[np.ndarray] = []

            cycle_reference_states: list[ModelState] = [initial_state] * 12
            best_cycle_states: list[ModelState] | None = None
            best_metric = float("inf")

            x_prev: np.ndarray | None = None
            f_prev: np.ndarray | None = None

            for iter_idx in range(FIXED_POINT_MAX_ITERS):
                with time_block("periodic_iteration"):
                    temp = x.reshape(shape)
                    temp = np.maximum(temp, temperature_floor)
                    state = ModelState(
                        temperature=temp,
                        albedo_field=initial_state.albedo_field,
                        wind_field=initial_state.wind_field,
                        humidity_field=initial_state.humidity_field,
                    )
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
                    if not np.all(np.isfinite(advanced.temperature)):
                        # Bail out early; quasi-Newton steps must preserve finite states.
                        raise FloatingPointError("Non-finite temperatures encountered in broyden periodic iteration")
                    p = advanced.temperature.ravel().copy()
                    f = p - x
                    if not np.all(np.isfinite(f)):
                        raise FloatingPointError("Non-finite fixed-point residual in broyden periodic iteration")

                    # Update L-BFGS secant history (s = Δx, y = ΔF)
                    if x_prev is not None and f_prev is not None:
                        s = x - x_prev
                        y = f - f_prev
                        ys = float(np.dot(y, s))
                        if np.isfinite(ys) and ys > 1e-12 * (np.linalg.norm(y) * np.linalg.norm(s) + 1e-20):
                            s_list.append(s)
                            y_list.append(y)
                            if len(s_list) > PERIODIC_BROYDEN_HISTORY:
                                s_list.pop(0)
                                y_list.pop(0)

                    # Convergence metric: same as Anderson (cycle-to-cycle change).
                    residual = np.array(
                        [
                            advanced_states[i].temperature[0] - cycle_reference_states[i].temperature[0]
                            for i in range(12)
                        ]
                    )

                    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
                    residual_99p = float(np.percentile(np.abs(residual), 99))
                    residual_max = float(np.max(np.abs(residual)))
                    print(residual_rms, residual_99p, residual_max)

                    if residual_rms < PERIODIC_FIXED_POINT_TOLERANCE_K and residual_99p < PERIODIC_FIXED_POINT_TOLERANCE_K_99P:
                        return [advanced_states[(i - 2) % 12] for i in range(12)]

                    # Track best state so far (for robust fallback)
                    metric = residual_99p
                    if np.isfinite(metric) and metric < best_metric:
                        best_metric = metric
                        best_cycle_states = advanced_states

                    # Quasi-Newton step: x_{k+1} = x_k - H_k f_k (with damping).
                    with time_block("anderson_acceleration"):
                        direction = _lbfgs_apply_inverse(f, s_list=s_list, y_list=y_list)
                        step = PERIODIC_BROYDEN_DAMPING * direction
                        # Trust-region / clipping: keep steps physically small to avoid
                        # driving the model into numerically unstable regimes (e.g., exp overflow).
                        max_abs = float(np.max(np.abs(step)))
                        if not np.isfinite(max_abs) or max_abs <= 0.0:
                            step_scale = 0.0
                        else:
                            step_scale = min(1.0, PERIODIC_BROYDEN_MAX_STEP_K / max_abs)
                        x_next = x - step_scale * step
                        # Project to a safe physical range (keeps humidity/latent numerics sane).
                        x_next = np.clip(x_next, temperature_floor, PERIODIC_BROYDEN_TEMP_CEILING_K)
                    x_prev = x
                    f_prev = f
                    x = x_next
                    cycle_reference_states = advanced_states

            # Failed: fall back to Anderson if enabled (using best cycle so far)
            if PERIODIC_BROYDEN_FALLBACK_TO_ANDERSON:
                # Use the best cycle found as a warm-start.
                if best_cycle_states is not None:
                    return _solve_anderson(best_cycle_states[-1])
                return _solve_anderson(initial_state)
            if best_cycle_states is not None:
                return [best_cycle_states[(i - 2) % 12] for i in range(12)]
            raise RuntimeError(
                "Failed to converge to a periodic solution after "
                f"{FIXED_POINT_MAX_ITERS} iterations (broyden mode, best metric {best_metric:.3e})"
            )

        return _solve_anderson(initial_state)

def solve_periodic_climate(
    resolution_deg: float = 1.0,
    *,
    model_config: ModelConfig,
    return_layer_map: bool = False,
    _enable_warm_start: bool = True,
    _reset_profiler: bool = True,
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

    if _reset_profiler:
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
        # Optional reduced-order warm start from a coarser periodic solve.
        if (
            _enable_warm_start
            and PERIODIC_WARM_START_COARSE_DEG
            and PERIODIC_WARM_START_COARSE_DEG > 0
            and resolution_deg > PERIODIC_WARM_START_COARSE_DEG
        ):
            try:
                coarse_deg = float(PERIODIC_WARM_START_COARSE_DEG)
                coarse_lon2d, coarse_lat2d, coarse_layers = solve_periodic_climate(
                    resolution_deg=coarse_deg,
                    model_config=model_config,
                    return_layer_map=True,
                    _enable_warm_start=False,
                    _reset_profiler=False,
                )
                assert isinstance(coarse_layers, dict)

                # Use March (index 2 in Jan-start) as a stable warm-start month.
                month_idx = 2

                fine_lon2d, fine_lat2d = operators.lon2d, operators.lat2d

                coarse_lat = np.asarray(coarse_lat2d[:, 0], dtype=float)
                coarse_lon = np.asarray(coarse_lon2d[0, :], dtype=float)
                fine_lat = np.asarray(fine_lat2d[:, 0], dtype=float)
                fine_lon = np.asarray(fine_lon2d[0, :], dtype=float) % 360.0

                def _interp_latlon(field2d: np.ndarray) -> np.ndarray:
                    # Periodic lon interpolation: extend by one wrap point.
                    lon_ext = np.concatenate([coarse_lon, [coarse_lon[0] + 360.0]])
                    field_ext = np.concatenate([field2d, field2d[:, :1]], axis=1)
                    # Interp along lon
                    tmp = np.empty((field2d.shape[0], fine_lon.size), dtype=float)
                    for i in range(field2d.shape[0]):
                        tmp[i, :] = np.interp(fine_lon, lon_ext, field_ext[i, :])
                    # Interp along lat
                    out = np.empty((fine_lat.size, fine_lon.size), dtype=float)
                    for j in range(fine_lon.size):
                        out[:, j] = np.interp(fine_lat, coarse_lat, tmp[:, j])
                    return out

                warm = initial_temp_guess.copy()
                # Coarse outputs are in Celsius; convert to Kelvin.
                if "surface" in coarse_layers:
                    warm[0] = _interp_latlon(coarse_layers["surface"][month_idx]) + 273.15
                if warm.shape[0] == 2:
                    if "atmosphere" in coarse_layers:
                        warm[1] = _interp_latlon(coarse_layers["atmosphere"][month_idx]) + 273.15
                elif warm.shape[0] == 3:
                    if "boundary_layer" in coarse_layers:
                        warm[1] = _interp_latlon(coarse_layers["boundary_layer"][month_idx]) + 273.15
                    if "atmosphere" in coarse_layers:
                        warm[2] = _interp_latlon(coarse_layers["atmosphere"][month_idx]) + 273.15
                initial_temp_guess = warm
            except Exception:
                # Warm start is optional; fall back to radiative equilibrium.
                pass
        initial_state = ModelState(
            temperature=initial_temp_guess,
            albedo_field=operators.base_albedo_field,
            wind_field=None,
            humidity_field=None,
        )

    # Solve for periodic cycle
    def _run_periodic(initial: ModelState) -> list[ModelState]:
        return find_periodic_climate_cycle(
            initial_state=initial,
            monthly_insolation=operators.monthly_insolation,
            month_durations=operators.month_durations,
            rhs_fn=rhs_fn,
            rhs_derivative_fn=rhs_derivative_fn,
            surface_context=operators.surface_context,
            temperature_floor=operators.radiation_config.temperature_floor,
            solver_cache=operators.solver_cache,
        )

    if _enable_warm_start and (PERIODIC_CONTINUATION_SCALES or PERIODIC_CONTINUATION_ADAPTIVE):
        # Parse scales like "0,0.5,1"
        targets = {t.strip() for t in PERIODIC_CONTINUATION_TARGETS.split(",") if t.strip()}
        env_keys: list[str] = []
        if "sensible" in targets:
            env_keys.append("SENSIBLE_EXCHANGE_SCALE")
        if "latent" in targets:
            env_keys.append("LATENT_EXCHANGE_SCALE")
        if "advection" in targets:
            env_keys.append("ADVECTION_SCALE")
        saved_env = {k: os.environ.get(k) for k in env_keys}
        try:
            stage_initial = initial_state
            stage_solution: list[ModelState] | None = None
            if PERIODIC_CONTINUATION_ADAPTIVE:
                s_cur = 0.0
                ds = float(np.clip(PERIODIC_CONTINUATION_INITIAL_STEP, PERIODIC_CONTINUATION_MIN_STEP, PERIODIC_CONTINUATION_MAX_STEP))
                # Start at scale 0
                for k in env_keys:
                    os.environ[k] = "0.0"
                stage_solution = _run_periodic(stage_initial)
                stage_initial = stage_solution[1]

                while s_cur < 1.0 - 1e-12:
                    s_try = float(min(1.0, s_cur + ds))
                    for k in env_keys:
                        os.environ[k] = str(s_try)
                    candidate_solution = _run_periodic(stage_initial)
                    candidate_start = candidate_solution[1].temperature
                    warm_start = stage_initial.temperature
                    delta = candidate_start - warm_start
                    delta_surface = delta[0]
                    d99 = float(np.percentile(np.abs(delta_surface), 99))
                    dmax = float(np.max(np.abs(delta_surface)))

                    if (d99 <= PERIODIC_CONTINUATION_MAX_DELTA_99P_K) and (dmax <= PERIODIC_CONTINUATION_MAX_DELTA_MAX_K):
                        # Accept
                        s_cur = s_try
                        stage_solution = candidate_solution
                        stage_initial = candidate_solution[1]
                        ds = float(min(PERIODIC_CONTINUATION_MAX_STEP, max(PERIODIC_CONTINUATION_MIN_STEP, ds * PERIODIC_CONTINUATION_GROWTH)))
                    else:
                        # Reject: shrink step and retry
                        ds = float(ds * PERIODIC_CONTINUATION_SHRINK)
                        if ds < PERIODIC_CONTINUATION_MIN_STEP:
                            raise RuntimeError(
                                "Adaptive continuation could not stay on the same branch: "
                                f"delta99={d99:.3f}K deltaMax={dmax:.3f}K at scale={s_try:.3f}"
                            )
                assert stage_solution is not None
            else:
                parts = [p.strip() for p in PERIODIC_CONTINUATION_SCALES.split(",") if p.strip()]
                stages = [float(p) for p in parts]
                if stages and abs(stages[-1] - 1.0) > 1e-12:
                    stages.append(1.0)
                for s in stages:
                    s_clamped = float(np.clip(s, 0.0, 1.0))
                    if "SENSIBLE_EXCHANGE_SCALE" in env_keys:
                        os.environ["SENSIBLE_EXCHANGE_SCALE"] = str(s_clamped)
                    if "LATENT_EXCHANGE_SCALE" in env_keys:
                        os.environ["LATENT_EXCHANGE_SCALE"] = str(s_clamped)
                    if "ADVECTION_SCALE" in env_keys:
                        os.environ["ADVECTION_SCALE"] = str(s_clamped)

                    stage_solution = _run_periodic(stage_initial)
                    # Warm start the next stage from February (Jan-start index 1), which is
                    # the start-of-March state for the March-start integration order.
                    stage_initial = stage_solution[1]
            assert stage_solution is not None
            monthly_states = stage_solution
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    else:
        monthly_states = _run_periodic(initial_state)

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
