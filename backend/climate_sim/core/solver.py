"""General-purpose periodic solver utilities for energy-balance models."""

from dataclasses import replace
from typing import Callable, Dict

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import climate_sim.physics.radiation as radiation
from climate_sim.physics.humidity import (
    compute_humidity_and_precipitation,
    compute_saturation_specific_humidity,
    specific_humidity_to_relative_humidity,
    COLUMN_MASS_KG_M2,
)
from climate_sim.physics.clouds import (
    compute_clouds_and_precipitation,
    compute_vertical_velocity_from_divergence,
    compute_vertical_velocity_from_pressure,
    compute_vertical_velocity_from_warm_advection,
)
from climate_sim.core.math_core import compute_divergence, compute_scalar_gradient_magnitude
from climate_sim.data.constants import (
    BOUNDARY_LAYER_HEIGHT_M,
    STANDARD_LAPSE_RATE_K_PER_M,
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
    LATENT_HEAT_VAPORIZATION_J_KG,
)
from climate_sim.physics.atmosphere.advection import AdvectionOperator
from climate_sim.physics.diffusion import DiffusionOperator
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeModel
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
from climate_sim.core.rhs_builder import (
    create_rhs_functions,
    RhsFn,
    RhsDerivativeFn,
    RhsBuildInputs,
)
from climate_sim.physics.precipitation import (
    compute_precipitation_recycling,
    compute_eddy_precipitation,
)
from climate_sim.physics.vertical_motion import (
    VerticalMotionConfig,
    compute_hadley_subsidence_velocity,
    compute_hadley_moisture_tendency,
)
from climate_sim.core.operators import SurfaceHeatCapacityContext, build_model_operators
from climate_sim.physics.atmosphere.hadley import compute_itcz_latitude
from climate_sim.physics.ocean_currents import (
    compute_ocean_currents,
    compute_deep_ocean_temperature,
)
from climate_sim.core.math_core import spherical_cell_area
from climate_sim.data.constants import (
    R_EARTH_METERS,
)
from climate_sim.runtime.config import ModelConfig

NEWTON_STEP_TOLERANCE_K = 1.0
PERIODIC_FIXED_POINT_TOLERANCE_K = 0.5
PERIODIC_FIXED_POINT_TOLERANCE_K_95P = 1.0
NEWTON_MAX_ITERS = 16
NEWTON_BACKTRACK_REDUCTION = 0.5
NEWTON_BACKTRACK_CUTOFF = 1e-3
FIXED_POINT_MAX_ITERS = 60

# Soft condensation: relax supersaturation toward q_sat with this timescale.
# Moisture removed becomes precipitation with latent heat release, closing
# the energy budget for moisture destroyed by the advection q_sat cap.
TAU_CONDENSATION_S = 30.0 * 86400.0  # 30 days

# Anderson acceleration parameters for periodic cycle solver
ANDERSON_HISTORY_LIMIT = 6
# Under-relaxation factor for outer (year-to-year) state updates.
# Blends Anderson output with previous state: x_new = α*x_anderson + (1-α)*x_old.
# Values < 1.0 stabilize strong pressure-wind-temperature coupling.
ANDERSON_RELAXATION = 0.4

# Refresh the LU preconditioner every N Newton iterations (or earlier on failure).
INEXACT_NEWTON_REFACTORIZE_EVERY = 6
# GMRES tolerance for inexact Newton linear solves.
INEXACT_NEWTON_GMRES_RTOL = 1e-4
INEXACT_NEWTON_GMRES_ATOL = 0.0
INEXACT_NEWTON_GMRES_RESTART = 50
INEXACT_NEWTON_GMRES_MAXITER = 50

# Soil moisture constants (shared between Newton solver and post-Newton fast drain)
SOIL_CAPACITY_KG_M2 = 300.0  # mm = kg/m², root zone depth
SOIL_THETA_FC = 0.35  # field capacity
SOIL_TAU_FAST_SECONDS = 3 * 86400.0  # fast gravitational drainage above field capacity
SOIL_TAU_SLOW_SECONDS = 365.0 * 86400.0  # slow baseflow below field capacity

# How often to update wind inside Newton loop.
WIND_UPDATE_EVERY = 4


def _build_surface_jacobian_block(
    ceff: np.ndarray,
    diag: np.ndarray,
    base_capacity: np.ndarray,
    diffusion_matrix: sparse.csc_matrix | None,
    dt_seconds: float,
    advection_matrix: sparse.csc_matrix | None = None,
) -> sparse.csc_matrix:
    """Build the surface layer jacobian block (shared between single and multi-layer)."""
    block = sparse.diags(ceff.ravel(), format="csc")
    block -= dt_seconds * sparse.diags((base_capacity * diag).ravel(), format="csc")
    if diffusion_matrix is not None and diffusion_matrix.nnz > 0:
        block -= dt_seconds * sparse.diags(base_capacity.ravel(), format="csc") @ diffusion_matrix
    # Ocean heat advection (surface layer only)
    if advection_matrix is not None and advection_matrix.nnz > 0:
        block -= dt_seconds * sparse.diags(base_capacity.ravel(), format="csc") @ advection_matrix
    return block


def monthly_step(
    state: ModelState,
    insolation_W_m2: np.ndarray,
    itcz_rad: np.ndarray,
    dt_seconds: float,
    *,
    rhs_fn: RhsFn,
    rhs_temperature_derivative_fn: RhsDerivativeFn,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
    surface_context: SurfaceHeatCapacityContext,
    ocean_advection_enabled: bool = True,
    latent_heat_model: "LatentHeatExchangeModel | None" = None,
    advection_operator: "AdvectionOperator | None" = None,
    humidity_diffusion_operator: "DiffusionOperator | None" = None,
    vertical_motion_cfg: VerticalMotionConfig | None = None,
    effective_mu: np.ndarray | None = None,
    ocean_albedo: np.ndarray | None = None,
) -> ModelState:
    """Advance the column temperature one implicit backward-Euler step."""
    with time_block("monthly_step"):
        start_temp = state.temperature
        temp_next = np.maximum(start_temp, temperature_floor)
        cache = solver_cache or DEFAULT_LINEAR_SOLVE_CACHE

        # Precompute eddy diffusivity field for eddy precipitation (Eady-scaled κ)
        lat_rad_kappa = np.deg2rad(surface_context.lat2d)
        eady_raw = np.abs(np.sin(lat_rad_kappa)) * np.abs(np.sin(2 * lat_rad_kappa))
        eady_at_45 = np.sin(np.deg2rad(45.0)) * np.sin(np.deg2rad(90.0))
        eddy_kappa = 1.0e6 * np.clip(eady_raw / eady_at_45 * 2.5, 0.3, 2.0)

        base_capacity = surface_context.baseline_capacity
        base_albedo_field = surface_context.base_albedo

        # Lag albedo and wind during Newton iterations for Jacobian consistency
        start_temp_capped = np.maximum(start_temp, temperature_floor)
        lagged_albedo_field = surface_context.albedo_model.apply_snow_albedo(
            base_albedo_field,
            start_temp_capped[0],
            soil_moisture=state.soil_moisture,
            vegetation_fraction=state.vegetation_fraction,
            effective_mu=effective_mu,
            ocean_albedo=ocean_albedo,
            ice_sheet_mask=surface_context.ice_sheet_mask,
        )

        def _compute_3layer_winds(t_atm: np.ndarray, t_bl: np.ndarray) -> tuple:
            """Compute geostrophic + Ekman wind fields for the 3-layer system.

            Returns (wind_field, bl_wind_field, bl_wind_unblocked) where
            bl_wind_field has orographic blocking applied when available, and
            bl_wind_unblocked is the pre-blocking (u, v) tuple (None if no
            orographic model).
            """
            wind = surface_context.wind_model.wind_field(
                t_atm,
                temperature_boundary_layer=t_bl,
                itcz_rad=itcz_rad,
                ekman_drag=False,
            )
            bl_wind = surface_context.wind_model.wind_field(
                t_atm,
                temperature_boundary_layer=t_bl,
                itcz_rad=itcz_rad,
                ekman_drag=True,
            )
            bl_unblocked = None
            if surface_context.orographic_model is not None:
                bl_u, bl_v = bl_wind[0], bl_wind[1]
                bl_unblocked = (bl_u, bl_v)
                bl_u_blocked, bl_v_blocked = surface_context.orographic_model.apply_flow_blocking(
                    bl_u, bl_v
                )
                bl_wind = (bl_u_blocked, bl_v_blocked, np.hypot(bl_u_blocked, bl_v_blocked))
            return wind, bl_wind, bl_unblocked

        # Compute lagged wind field(s)
        lagged_wind_field = None
        lagged_boundary_layer_wind_field = None
        lagged_bl_wind_unblocked = None
        if surface_context.wind_model:
            if start_temp_capped.shape[0] == 3:
                lagged_wind_field, lagged_boundary_layer_wind_field, lagged_bl_wind_unblocked = (
                    _compute_3layer_winds(start_temp_capped[2], start_temp_capped[1])
                )
            elif start_temp_capped.shape[0] == 2:
                lagged_wind_field = surface_context.wind_model.wind_field(
                    start_temp_capped[1], itcz_rad=itcz_rad, ekman_drag=True
                )
            else:
                lagged_wind_field = surface_context.wind_model.wind_field(
                    start_temp_capped[0], itcz_rad=itcz_rad, ekman_drag=True
                )

        # Compute lagged ocean currents from 10m wind (boundary layer or atmosphere wind)
        lagged_ocean_current_field = None
        lagged_ocean_ekman_current_field = None
        lagged_ocean_current_psi = None
        lagged_ekman_pumping = None
        # Deep ocean temperature is static (latitude-dependent only)
        deep_ocean_temp = compute_deep_ocean_temperature(surface_context.lat2d[:, 0])
        deep_ocean_temp_2d = np.broadcast_to(
            deep_ocean_temp[:, np.newaxis], surface_context.lat2d.shape
        ).copy()
        if ocean_advection_enabled:
            if lagged_boundary_layer_wind_field is not None:
                # Use boundary layer wind (10m equivalent)
                wind_u_10m, wind_v_10m = (
                    lagged_boundary_layer_wind_field[0],
                    lagged_boundary_layer_wind_field[1],
                )
            elif lagged_wind_field is not None:
                # Fallback: use atmosphere wind
                wind_u_10m, wind_v_10m = lagged_wind_field[0], lagged_wind_field[1]
            else:
                wind_u_10m, wind_v_10m = None, None

            if wind_u_10m is not None and wind_v_10m is not None:
                ocean_results = compute_ocean_currents(
                    wind_u_10m,
                    wind_v_10m,
                    surface_context.lon2d,
                    surface_context.lat2d,
                    surface_context.land_mask,
                    include_stommel=True,
                )
                lagged_ocean_current_field = (
                    ocean_results["u_velocity"],
                    ocean_results["v_velocity"],
                )
                lagged_ocean_ekman_current_field = (
                    ocean_results["u_ekman"],
                    ocean_results["v_ekman"],
                )
                lagged_ocean_current_psi = ocean_results["psi"]
                lagged_ekman_pumping = ocean_results["w_ekman_pumping"]

        # Compute cell areas once for ITCZ calculations inside Newton loop
        cell_areas = spherical_cell_area(
            surface_context.lon2d, surface_context.lat2d, earth_radius_m=R_EARTH_METERS
        )

        # Humidity is prognostic - carried from previous month, evolved after Newton converges
        lagged_humidity = state.humidity_field
        lagged_precipitation = state.precipitation_field
        soil_moisture_iter = state.soil_moisture
        start_soil_moisture = state.soil_moisture

        def _effective_surface_capacity(temp_surface: np.ndarray) -> np.ndarray:
            return surface_context.albedo_model.effective_heat_capacity_surface(
                temp_surface,
                land_mask=surface_context.land_mask,
                base_C_land=surface_context.base_C_land,
                base_C_ocean=surface_context.base_C_ocean,
                ice_sheet_mask=surface_context.ice_sheet_mask,
                ice_sheet_heat_capacity_multiplier=surface_context.ice_sheet_heat_capacity_multiplier,
            )

        def _compute_itcz_from_temp(temp: np.ndarray) -> np.ndarray:
            """Compute ITCZ from current temperature iterate."""
            nlayers = temp.shape[0]
            itcz_temp = temp[1] if nlayers >= 3 else temp[0]
            return compute_itcz_latitude(
                np.maximum(itcz_temp, temperature_floor),
                surface_context.lat2d,
                cell_areas,
            )

        # Helper to compute cloud fractions from current temperature and humidity.
        # Called both for the initial bootstrap and inside the Newton loop so that
        # cloud fractions track the evolving humidity (eliminates month-to-month
        # oscillation caused by frozen clouds lagging behind humidity changes).
        from climate_sim.physics.atmosphere.pressure import compute_pressure

        # Precompute divergence-based vertical velocity from lagged winds (constant
        # within a monthly step since winds are lagged).
        _cached_cloud_vertical_velocity: np.ndarray | None = None
        if lagged_boundary_layer_wind_field is not None:
            wind_u_cf, wind_v_cf = (
                lagged_boundary_layer_wind_field[0],
                lagged_boundary_layer_wind_field[1],
            )
            _cached_divergence = compute_divergence(
                wind_u_cf, wind_v_cf, surface_context.lat2d, surface_context.lon2d
            )
            _cached_cloud_vertical_velocity = compute_vertical_velocity_from_divergence(
                _cached_divergence
            )

        def _compute_cloud_fractions(
            temp: np.ndarray, humidity: np.ndarray, itcz: np.ndarray | None = None
        ):
            """Recompute cloud output from current temperature and humidity."""
            T_bl_cloud = temp[1]
            T_atm_cloud = temp[2]
            # Use provided ITCZ or compute from temperature as fallback
            itcz_cf = itcz if itcz is not None else _compute_itcz_from_temp(temp)
            # Use precomputed divergence-based w from lagged winds when available.
            if _cached_cloud_vertical_velocity is not None:
                vertical_velocity = _cached_cloud_vertical_velocity
            else:
                pressure = compute_pressure(
                    temp[0],
                    itcz_rad=itcz_cf,
                    lat2d=surface_context.lat2d,
                    lon2d=surface_context.lon2d,
                )
                nlat, nlon = temp[0].shape
                lat_spacing = 180.0 / nlat
                lat_centers = -90.0 + (np.arange(nlat) + 0.5) * lat_spacing
                cos_lat = np.clip(np.cos(np.deg2rad(lat_centers)), 1.0e-6, None)
                weights = np.broadcast_to(cos_lat[:, None], (nlat, nlon))
                mean_pressure = np.sum(pressure * weights) / np.sum(weights)
                dp = pressure - mean_pressure
                dp_norm = np.clip(dp / 1000.0, -1.0, 1.0)
                vertical_velocity = compute_vertical_velocity_from_pressure(dp_norm)
            rh = specific_humidity_to_relative_humidity(
                humidity,
                T_bl_cloud,
                itcz_rad=itcz_cf,
                lat2d=surface_context.lat2d,
                lon2d=surface_context.lon2d,
            )
            return compute_clouds_and_precipitation(
                T_bl_K=T_bl_cloud,
                T_atm_K=T_atm_cloud,
                q=humidity,
                rh=rh,
                vertical_velocity=vertical_velocity,
                T_surface_K=temp[0],
                ocean_mask=~surface_context.land_mask,
            )

        # Use the ITCZ passed in as the parameter (itcz_rad). When called from
        # evolve_year with precomputed monthly_itcz, this is lagged by one
        # periodic iteration. This prevents oscillation with sharp tau.
        lagged_itcz = itcz_rad

        # Bootstrap cloud output from start-of-month state
        lagged_cloud_output = None
        nlayers = start_temp_capped.shape[0]
        if lagged_humidity is not None and nlayers >= 2:
            lagged_cloud_output = _compute_cloud_fractions(
                start_temp_capped, lagged_humidity, itcz=lagged_itcz
            )

        # Precompute orographic vertical velocity from UNBLOCKED wind.
        # Orographic uplift represents air forced upward over terrain, which uses the
        # approaching (unblocked) wind speed. Wind blocking is a separate effect
        # (surface flow deflected around terrain).
        lagged_orographic_w = None
        if (
            surface_context.orographic_model is not None
            and lagged_boundary_layer_wind_field is not None
        ):
            lagged_orographic_w = (
                surface_context.orographic_model.compute_orographic_vertical_velocity(
                    lagged_bl_wind_unblocked[0],
                    lagged_bl_wind_unblocked[1],  # type: ignore[possibly-undefined]
                )
            )

        def _init_state(temp: np.ndarray) -> ModelState:
            """Create model state for RHS evaluation during Newton iterations.

            Uses lagged fields (albedo, wind, humidity, clouds, ITCZ) for
            Jacobian consistency.
            """
            with time_block("_init_state"):
                return ModelState(
                    temperature=temp,
                    albedo_field=lagged_albedo_field,
                    wind_field=lagged_wind_field,
                    humidity_field=lagged_humidity,
                    boundary_layer_wind_field=lagged_boundary_layer_wind_field,
                    ocean_current_field=lagged_ocean_current_field,
                    ocean_ekman_current_field=lagged_ocean_ekman_current_field,
                    ocean_current_psi=lagged_ocean_current_psi,
                    ocean_ekman_pumping=lagged_ekman_pumping,
                    deep_ocean_temperature=deep_ocean_temp_2d,
                    precipitation_field=lagged_precipitation,
                    cloud_output=lagged_cloud_output,  # Unified clouds (frozen for Jacobian consistency)
                    soil_moisture=soil_moisture_iter,
                    orographic_w=lagged_orographic_w,
                    itcz_rad=lagged_itcz,
                )

        # implicit solver loop
        preconditioner_solve: Callable[[np.ndarray], np.ndarray] | None = None
        preconditioner_age = 10**9
        # Jacobian lagging: reuse Jacobian when residual is decreasing
        cached_linearization = None
        jacobian_age: int = 0
        JACOBIAN_MAX_AGE = 3  # Recompute after this many reuses
        # Cache assembled Jacobian and preconditioner for reuse when linearization is lagged.
        # Only the surface ceff diagonal changes between iterations.
        cached_assembled_jacobian: sparse.csc_matrix | None = None
        cached_assembled_preconditioner: sparse.csc_matrix | None = None
        cached_surface_block: sparse.csc_matrix | None = None
        cached_ceff_surface: np.ndarray | None = None
        cached_include_implicit_humidity: bool = False

        def _solve_linear_system(
            jacobian: sparse.csc_matrix,
            rhs: np.ndarray,
            *,
            preconditioner_matrix: sparse.csc_matrix | None = None,
            newton_iter: int,
        ) -> np.ndarray:
            """Inexact Newton linear solve: GMRES with a reused LU preconditioner."""
            nonlocal preconditioner_solve, preconditioner_age

            if (preconditioner_solve is None) or (
                preconditioner_age >= INEXACT_NEWTON_REFACTORIZE_EVERY
            ):
                with time_block("factorize_solver"):
                    matrix = (
                        preconditioner_matrix if preconditioner_matrix is not None else jacobian
                    )
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
                    # Add tiny regularization to prevent exact singularity
                    # from cloud Jacobian chain-rule terms
                    reg = sparse.eye(jacobian.shape[0], format="csc") * 1e-14
                    preconditioner_solve = splinalg.factorized(jacobian + reg)
                preconditioner_age = 0
                return preconditioner_solve(rhs)

            return sol

        def _update_wind_from_temp(temp_arr: np.ndarray) -> None:
            """Recompute lagged wind fields from current temperature iterate."""
            nonlocal lagged_wind_field, lagged_boundary_layer_wind_field
            if surface_context.wind_model is None:
                return
            if temp_arr.shape[0] != 3:
                return
            tc = np.maximum(temp_arr, temperature_floor)
            with time_block("update_wind_fields"):
                lagged_wind_field, lagged_boundary_layer_wind_field, _ = _compute_3layer_winds(
                    tc[2], tc[1]
                )

        for newton_iter in range(NEWTON_MAX_ITERS):
            with time_block("newton_iteration"):
                temp_capped = np.maximum(temp_next, temperature_floor)

                # Periodically update wind from current temperature iterate
                # so advection of q uses winds consistent with current T.
                if newton_iter > 0 and newton_iter % WIND_UPDATE_EVERY == 0:
                    _update_wind_from_temp(temp_capped)

                # Update cloud fractions from current iterate so clouds track
                # the evolving humidity (prevents month-to-month oscillation).
                if lagged_humidity is not None and temp_capped.shape[0] >= 2:
                    lagged_cloud_output = _compute_cloud_fractions(
                        temp_capped, lagged_humidity, itcz=lagged_itcz
                    )

                state_capped = _init_state(temp_capped)
                # Update precipitation on state from current clouds so the RHS
                # latent heating uses current P (not lagged from previous month).
                # This tightens the P->LH->T coupling within Newton iterations.
                if lagged_cloud_output is not None:
                    state_capped = replace(
                        state_capped,
                        precipitation_field=lagged_cloud_output.total_precip,
                    )
                with time_block("rhs_evaluation"):
                    rhs_value = rhs_fn(state_capped, insolation_W_m2, state_capped.itcz_rad)

                # Jacobian lagging: skip rhs_derivative when residual is well-behaved
                need_jacobian = cached_linearization is None or jacobian_age >= JACOBIAN_MAX_AGE
                if need_jacobian:
                    with time_block("rhs_derivative"):
                        linearization = rhs_temperature_derivative_fn(
                            state_capped, insolation_W_m2, state_capped.itcz_rad
                        )
                    cached_linearization = linearization
                    jacobian_age = 0
                else:
                    linearization = cached_linearization
                    jacobian_age += 1
                preconditioner_age += 1

                nlayers = temp_capped.shape[0]
                surface_diag = linearization.diag[0]
                ceff_surface = _effective_surface_capacity(temp_capped[0])
                flux_surface = base_capacity * rhs_value[0]
                residual_surface = (
                    ceff_surface * (temp_capped[0] - start_temp[0]) - dt_seconds * flux_surface
                )

                nlat, nlon = surface_diag.shape
                size = nlat * nlon

                # Default: no implicit humidity/soil for non-3-layer cases
                include_implicit_humidity = False
                include_implicit_soil = False
                correction_humidity = None

                if nlayers == 1:
                    # Single-layer: only surface residual and jacobian
                    residual = residual_surface[np.newaxis, :, :]
                    residual_flat = residual_surface.ravel()

                    with time_block("jacobian_assembly"):
                        surface_block = _build_surface_jacobian_block(
                            ceff_surface,
                            surface_diag,
                            base_capacity,
                            linearization.surface_diffusion_matrix,
                            dt_seconds,
                            advection_matrix=linearization.surface_advection_matrix,
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
                elif nlayers == 3:
                    # Three-layer: surface + boundary layer + atmosphere with adjacent coupling only
                    # Optionally includes humidity as 4th prognostic field
                    boundary_diag = linearization.diag[1]
                    atmosphere_diag = linearization.diag[2]
                    residual_boundary = temp_capped[1] - start_temp[1] - dt_seconds * rhs_value[1]
                    residual_atmosphere = temp_capped[2] - start_temp[2] - dt_seconds * rhs_value[2]

                    identity = get_identity_matrix(size, cache=cache)
                    zero_matrix = sparse.csc_matrix((size, size))

                    # Include humidity in implicit solver when available
                    include_implicit_humidity = (
                        lagged_humidity is not None
                        and linearization.humidity_advection_matrix is not None
                    )

                    # Include soil moisture as 5th prognostic variable when available
                    include_implicit_soil = (
                        include_implicit_humidity
                        and soil_moisture_iter is not None
                        and linearization.soil_diag is not None
                    )

                    # When linearization is reused (lagged Jacobian), only the surface
                    # ceff diagonal changes. Update the cached Jacobian cheaply instead
                    # of rebuilding all blocks from scratch.
                    can_reuse_assembly = (
                        not need_jacobian
                        and cached_assembled_jacobian is not None
                        and cached_ceff_surface is not None
                        and cached_include_implicit_humidity == include_implicit_humidity
                    )

                    if can_reuse_assembly:
                        with time_block("jacobian_assembly"):
                            # Only the surface ceff diagonal changed — update it
                            delta_ceff = ceff_surface.ravel() - cached_ceff_surface.ravel()
                            # Build a sparse diagonal correction for the full Jacobian
                            # (delta_ceff goes in the first `size` diagonal entries)
                            n_total = cached_assembled_jacobian.shape[0]
                            full_delta = np.zeros(n_total)
                            full_delta[:size] = delta_ceff
                            jacobian = cached_assembled_jacobian + sparse.diags(
                                full_delta, format="csc"
                            )
                            cached_assembled_jacobian = jacobian

                            # Update surface block in preconditioner similarly
                            surface_block = cached_surface_block + sparse.diags(
                                delta_ceff, format="csc"
                            )
                            cached_surface_block = surface_block

                            preconditioner_delta = np.zeros(n_total)
                            preconditioner_delta[:size] = delta_ceff
                            preconditioner_matrix = cached_assembled_preconditioner + sparse.diags(
                                preconditioner_delta, format="csc"
                            )
                            cached_assembled_preconditioner = preconditioner_matrix
                            cached_ceff_surface = ceff_surface.copy()
                    else:
                        with time_block("jacobian_assembly"):
                            surface_block = _build_surface_jacobian_block(
                                ceff_surface,
                                surface_diag,
                                base_capacity,
                                linearization.surface_diffusion_matrix,
                                dt_seconds,
                                advection_matrix=linearization.surface_advection_matrix,
                            )

                            # Boundary layer block (with diffusion and advection if available)
                            boundary_block = identity.copy()
                            boundary_block -= dt_seconds * sparse.diags(
                                boundary_diag.ravel(), format="csc"
                            )
                            if (
                                linearization.boundary_layer_diffusion_matrix is not None
                                and linearization.boundary_layer_diffusion_matrix.nnz > 0
                            ):
                                boundary_block -= (
                                    dt_seconds * linearization.boundary_layer_diffusion_matrix
                                )
                            if (
                                linearization.boundary_layer_advection_matrix is not None
                                and linearization.boundary_layer_advection_matrix.nnz > 0
                            ):
                                boundary_block -= (
                                    dt_seconds * linearization.boundary_layer_advection_matrix
                                )

                            # Atmosphere block (with diffusion and advection)
                            atmosphere_block = identity.copy()
                            atmosphere_block -= dt_seconds * sparse.diags(
                                atmosphere_diag.ravel(), format="csc"
                            )
                            if (
                                linearization.atmosphere_diffusion_matrix is not None
                                and linearization.atmosphere_diffusion_matrix.nnz > 0
                            ):
                                atmosphere_block -= (
                                    dt_seconds * linearization.atmosphere_diffusion_matrix
                                )
                            if (
                                linearization.atmosphere_advection_matrix is not None
                                and linearization.atmosphere_advection_matrix.nnz > 0
                            ):
                                atmosphere_block -= (
                                    dt_seconds * linearization.atmosphere_advection_matrix
                                )

                            # Cross-coupling: includes surface-atmosphere coupling via transmission
                            if linearization.cross is not None:
                                coupling_01 = -dt_seconds * sparse.diags(
                                    (base_capacity * linearization.cross[0, 1]).ravel(),
                                    format="csc",
                                )
                                coupling_02 = -dt_seconds * sparse.diags(
                                    (base_capacity * linearization.cross[0, 2]).ravel(),
                                    format="csc",
                                )
                                coupling_10 = -dt_seconds * sparse.diags(
                                    linearization.cross[1, 0].ravel(), format="csc"
                                )
                                coupling_12 = -dt_seconds * sparse.diags(
                                    linearization.cross[1, 2].ravel(), format="csc"
                                )
                                coupling_20 = -dt_seconds * sparse.diags(
                                    linearization.cross[2, 0].ravel(), format="csc"
                                )
                                coupling_21 = -dt_seconds * sparse.diags(
                                    linearization.cross[2, 1].ravel(), format="csc"
                                )
                            else:
                                coupling_01 = coupling_02 = coupling_10 = coupling_12 = (
                                    coupling_20
                                ) = coupling_21 = zero_matrix

                            if include_implicit_humidity:
                                # Build 4×4 block Jacobian including humidity
                                humidity_block = identity.copy()
                                if linearization.humidity_diag is not None:
                                    humidity_block -= dt_seconds * sparse.diags(
                                        linearization.humidity_diag.ravel(), format="csc"
                                    )
                                # Soft condensation Jacobian: d(condensation_tendency)/dq
                                # = -1/tau where q > q_sat, 0 otherwise
                                q_sat_jac = compute_saturation_specific_humidity(temp_capped[1])
                                cond_jac_diag = np.where(
                                    lagged_humidity > q_sat_jac, -1.0 / TAU_CONDENSATION_S, 0.0
                                )
                                humidity_block -= dt_seconds * sparse.diags(
                                    cond_jac_diag.ravel(), format="csc"
                                )
                                if (
                                    linearization.humidity_advection_matrix is not None
                                    and linearization.humidity_advection_matrix.nnz > 0
                                ):
                                    humidity_block -= (
                                        dt_seconds * linearization.humidity_advection_matrix
                                    )
                                if (
                                    linearization.humidity_diffusion_matrix is not None
                                    and linearization.humidity_diffusion_matrix.nnz > 0
                                ):
                                    humidity_block -= (
                                        dt_seconds * linearization.humidity_diffusion_matrix
                                    )

                                # Temperature-humidity coupling (dR_T/dq)
                                if linearization.temp_humidity_coupling is not None:
                                    dR_Tsfc_dq, dR_Tbl_dq, dR_Tatm_dq = (
                                        linearization.temp_humidity_coupling
                                    )
                                    coupling_0q = -dt_seconds * sparse.diags(
                                        (base_capacity * dR_Tsfc_dq).ravel(), format="csc"
                                    )
                                    coupling_1q = (
                                        -dt_seconds * sparse.diags(dR_Tbl_dq.ravel(), format="csc")
                                        if dR_Tbl_dq is not None
                                        else zero_matrix
                                    )
                                    coupling_2q = -dt_seconds * sparse.diags(
                                        dR_Tatm_dq.ravel(), format="csc"
                                    )
                                else:
                                    coupling_0q = coupling_1q = coupling_2q = zero_matrix

                                cond_dTatm_dq = np.where(
                                    lagged_humidity > q_sat_jac,
                                    COLUMN_MASS_KG_M2
                                    / TAU_CONDENSATION_S
                                    * LATENT_HEAT_VAPORIZATION_J_KG
                                    / ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
                                    0.0,
                                )
                                coupling_2q = coupling_2q - dt_seconds * sparse.diags(
                                    cond_dTatm_dq.ravel(), format="csc"
                                )

                                # Humidity-temperature coupling (dR_q/dT)
                                if linearization.humidity_temp_coupling is not None:
                                    dR_q_dTsfc, dR_q_dTbl, dR_q_dTatm = (
                                        linearization.humidity_temp_coupling
                                    )
                                    coupling_q0 = -dt_seconds * sparse.diags(
                                        dR_q_dTsfc.ravel(), format="csc"
                                    )
                                    coupling_q1 = -dt_seconds * sparse.diags(
                                        dR_q_dTbl.ravel(), format="csc"
                                    )
                                    coupling_q2 = -dt_seconds * sparse.diags(
                                        dR_q_dTatm.ravel(), format="csc"
                                    )
                                else:
                                    coupling_q0 = coupling_q1 = coupling_q2 = zero_matrix

                                jacobian = sparse.bmat(
                                    [
                                        [surface_block, coupling_01, coupling_02, coupling_0q],
                                        [coupling_10, boundary_block, coupling_12, coupling_1q],
                                        [coupling_20, coupling_21, atmosphere_block, coupling_2q],
                                        [coupling_q0, coupling_q1, coupling_q2, humidity_block],
                                    ],
                                    format="csc",
                                )
                            else:
                                jacobian = sparse.bmat(
                                    [
                                        [surface_block, coupling_01, coupling_02],
                                        [coupling_10, boundary_block, coupling_12],
                                        [coupling_20, coupling_21, atmosphere_block],
                                    ],
                                    format="csc",
                                )
                            assert isinstance(jacobian, sparse.csc_matrix)

                            # Cache for reuse when linearization is lagged
                            cached_assembled_jacobian = jacobian.copy()
                            cached_surface_block = surface_block.copy()
                            cached_ceff_surface = ceff_surface.copy()
                            cached_include_implicit_humidity = include_implicit_humidity

                    # Build residual and preconditioner
                    residual = np.stack([residual_surface, residual_boundary, residual_atmosphere])
                    temp_residuals = [
                        residual_surface.ravel(),
                        residual_boundary.ravel(),
                        residual_atmosphere.ravel(),
                    ]
                    if can_reuse_assembly:
                        temp_blocks = [surface_block]  # Only surface block is needed for update
                    else:
                        temp_blocks = [surface_block, boundary_block, atmosphere_block]

                    if include_implicit_humidity:
                        if lagged_boundary_layer_wind_field is not None:
                            wind_u_q, wind_v_q = lagged_boundary_layer_wind_field[:2]
                        elif lagged_wind_field is not None:
                            wind_u_q, wind_v_q = lagged_wind_field[:2]
                        else:
                            wind_u_q = wind_v_q = np.zeros_like(temp_capped[0])

                        # Compute humidity tendency
                        t_bl, t_atm = temp_capped[1], temp_capped[2]

                        # Evaporation rate from latent heat model
                        if latent_heat_model is not None:
                            wind_speed_ref = (
                                lagged_boundary_layer_wind_field[2]
                                if lagged_boundary_layer_wind_field is not None
                                else lagged_wind_field[2]
                                if lagged_wind_field is not None
                                else None
                            )
                            tendencies = latent_heat_model.compute_tendencies(
                                surface_temperature_K=temp_capped[0],
                                atmosphere_temperature_K=t_atm,
                                humidity_q=lagged_humidity,
                                wind_speed_reference_m_s=wind_speed_ref,
                                itcz_rad=itcz_rad,
                                boundary_layer_temperature_K=t_bl,
                                precipitation_rate=lagged_precipitation,
                                soil_moisture=soil_moisture_iter,
                                vegetation_fraction=state.vegetation_fraction,
                            )
                            evap_rate = tendencies[-1]
                        else:
                            evap_rate = np.zeros_like(lagged_humidity)

                        # Use precipitation from current cloud output (updated inside Newton)
                        # rather than lagged start-of-month value, to avoid lag oscillation.
                        if lagged_cloud_output is not None:
                            precip_rate = lagged_cloud_output.total_precip
                        elif lagged_precipitation is not None:
                            precip_rate = lagged_precipitation
                        else:
                            precip_rate = np.zeros_like(lagged_humidity)
                        # Use flux-form advection for humidity conservation: -∇·(uq)
                        # Subcycled to keep CFL ≤ 1 (30-day timestep with ~6 m/s winds gives CFL≈28)
                        # Cap q at saturation each substep to prevent convergence
                        # zones from accumulating unphysical supersaturation.
                        q_sat_cap = compute_saturation_specific_humidity(t_bl)
                        advection_tendency = (
                            advection_operator.subcycled_flux_tendency(
                                lagged_humidity,
                                wind_u_q,
                                wind_v_q,
                                dt=dt_seconds,
                                field_max=q_sat_cap,
                            )
                            if advection_operator is not None
                            else np.zeros_like(lagged_humidity)
                        )
                        # Humidity diffusion for turbulent mixing (stabilizes implicit solver)
                        diffusion_tendency = (
                            humidity_diffusion_operator.tendency(lagged_humidity)
                            if humidity_diffusion_operator is not None
                            and humidity_diffusion_operator.enabled
                            else np.zeros_like(lagged_humidity)
                        )
                        # Hadley subsidence drying: large-scale descent mixes dry upper-tropospheric air into BL
                        if (
                            vertical_motion_cfg is not None
                            and vertical_motion_cfg.enabled
                            and vertical_motion_cfg.hadley_descent_velocity_m_s > 0
                        ):
                            lat_rad = np.deg2rad(surface_context.lat2d)
                            w_hadley = compute_hadley_subsidence_velocity(
                                lat_rad,
                                itcz_rad,
                                peak_velocity_m_s=vertical_motion_cfg.hadley_descent_velocity_m_s,
                            )
                            hadley_drying = compute_hadley_moisture_tendency(
                                w_hadley,
                                lagged_humidity,
                                lat_rad,
                                upper_troposphere_q_fraction=vertical_motion_cfg.upper_troposphere_q_fraction,
                            )
                        else:
                            hadley_drying = np.zeros_like(lagged_humidity)
                        # Compute RH for precipitation gates
                        q_sat_bl = compute_saturation_specific_humidity(temp_capped[1])
                        rh_bl = np.clip(lagged_humidity / np.maximum(q_sat_bl, 1e-10), 0.0, 1.0)
                        # Direct orographic precipitation: P_oro = rh_gate · η · max(w, 0) · q · ρ
                        if (
                            lagged_orographic_w is not None
                            and surface_context.orographic_model is not None
                        ):
                            oro_precip = (
                                surface_context.orographic_model.compute_orographic_precipitation(
                                    lagged_orographic_w,
                                    lagged_humidity,
                                    temp_capped[1],
                                    rh_bl,
                                )
                            )
                            precip_rate = precip_rate + oro_precip
                        # Precipitation recycling (Eltahir & Bras 1996)
                        wind_speed_recycle = (
                            lagged_boundary_layer_wind_field[2]
                            if lagged_boundary_layer_wind_field is not None
                            else lagged_wind_field[2]
                            if lagged_wind_field is not None
                            else np.full_like(lagged_humidity, 3.0)
                        )
                        grid_deg = abs(surface_context.lat2d[1, 0] - surface_context.lat2d[0, 0])
                        precip_rate = precip_rate + compute_precipitation_recycling(
                            evap_rate,
                            lagged_humidity,
                            wind_speed_recycle,
                            surface_context.land_mask,
                            resolution_deg=grid_deg,
                        )
                        # Eddy precipitation: moisture wrung out during baroclinic transport
                        grad_q = compute_scalar_gradient_magnitude(
                            lagged_humidity,
                            surface_context.lat2d,
                            surface_context.lon2d,
                        )
                        precip_rate = precip_rate + compute_eddy_precipitation(
                            lagged_humidity,
                            grad_q,
                            eddy_kappa,
                            rh_bl,
                        )
                        # Soft condensation: relax supersaturation back to q_sat
                        # with fast timescale.
                        q_excess = np.maximum(lagged_humidity - q_sat_bl, 0.0)
                        condensation_tendency = -q_excess / TAU_CONDENSATION_S  # kg/kg/s, ≤ 0
                        condensation_precip_rate = (
                            q_excess / TAU_CONDENSATION_S * COLUMN_MASS_KG_M2
                        )  # kg/m²/s

                        humidity_tendency = (
                            (evap_rate - precip_rate) / COLUMN_MASS_KG_M2
                            + advection_tendency
                            + diffusion_tendency
                            + hadley_drying
                            + condensation_tendency
                        )

                        # Humidity residual: q_new - q_old - dt * tendency
                        start_humidity = (
                            state.humidity_field
                            if state.humidity_field is not None
                            else lagged_humidity
                        )
                        residual_humidity = (
                            lagged_humidity - start_humidity - dt_seconds * humidity_tendency
                        )

                        # Latent heat from soft condensation → atmosphere
                        condensation_heating_K_s = (
                            condensation_precip_rate
                            * LATENT_HEAT_VAPORIZATION_J_KG
                            / ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K
                        )
                        # Subtract from atmosphere residual (extra heating =
                        # larger RHS = smaller residual)
                        temp_residuals[2] = (
                            temp_residuals[2] - (dt_seconds * condensation_heating_K_s).ravel()
                        )

                        residual_flat = np.concatenate(
                            temp_residuals + [residual_humidity.ravel()], axis=0
                        )
                        if can_reuse_assembly:
                            preconditioner_matrix = cached_assembled_preconditioner
                        else:
                            preconditioner_matrix = sparse.block_diag(
                                temp_blocks + [humidity_block], format="csc"
                            )
                            cached_assembled_preconditioner = preconditioner_matrix.copy()
                    else:
                        residual_flat = np.concatenate(temp_residuals, axis=0)
                        if can_reuse_assembly:
                            preconditioner_matrix = cached_assembled_preconditioner
                        else:
                            preconditioner_matrix = sparse.block_diag(temp_blocks, format="csc")
                            cached_assembled_preconditioner = preconditioner_matrix.copy()

                    with time_block("linear_solve"):
                        correction_flat = _solve_linear_system(
                            jacobian,
                            residual_flat,
                            preconditioner_matrix=preconditioner_matrix,
                            newton_iter=newton_iter,
                        )

                    correction_surface = correction_flat[:size].reshape(surface_diag.shape)
                    correction_boundary = correction_flat[size : 2 * size].reshape(
                        boundary_diag.shape
                    )
                    if include_implicit_humidity:
                        correction_atmosphere = correction_flat[2 * size : 3 * size].reshape(
                            atmosphere_diag.shape
                        )
                        correction_humidity = correction_flat[3 * size :].reshape(
                            surface_diag.shape
                        )
                    else:
                        correction_atmosphere = correction_flat[2 * size :].reshape(
                            atmosphere_diag.shape
                        )
                        correction_humidity = None
                    correction = np.stack(
                        [correction_surface, correction_boundary, correction_atmosphere]
                    )
                else:
                    raise ValueError(f"Unsupported number of layers: {nlayers}")

                damping = 1.0
                max_residual = float(np.max(np.abs(residual)))
                accepted = False
                prev_temp = temp_next
                temp_candidate = temp_next
                residual_candidate = residual

                # When using a lagged Jacobian, limit backtracking to save
                # RHS evaluations — the Newton direction is approximate anyway.
                backtrack_cutoff = NEWTON_BACKTRACK_CUTOFF if need_jacobian else 0.25

                while damping >= backtrack_cutoff:
                    temp_candidate = np.maximum(prev_temp - damping * correction, temperature_floor)
                    state_candidate = _init_state(temp_candidate)
                    with time_block("backtrack_rhs"):
                        rhs_candidate = rhs_fn(
                            state_candidate, insolation_W_m2, state_candidate.itcz_rad
                        )

                    ceff_candidate = _effective_surface_capacity(temp_candidate[0])
                    nlayers = temp_candidate.shape[0]
                    residual_surface_candidate = ceff_candidate * (
                        temp_candidate[0] - start_temp[0]
                    ) - dt_seconds * (base_capacity * rhs_candidate[0])

                    if nlayers == 1:
                        residual_candidate = residual_surface_candidate[np.newaxis, :, :]
                    elif nlayers == 3:
                        residual_boundary_candidate = (
                            temp_candidate[1] - start_temp[1] - dt_seconds * rhs_candidate[1]
                        )
                        residual_atmosphere_candidate = (
                            temp_candidate[2] - start_temp[2] - dt_seconds * rhs_candidate[2]
                        )
                        residual_candidate = np.stack(
                            [
                                residual_surface_candidate,
                                residual_boundary_candidate,
                                residual_atmosphere_candidate,
                            ]
                        )
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

                # Apply humidity correction from implicit solve
                if nlayers == 3 and include_implicit_humidity and correction_humidity is not None:
                    lagged_humidity = np.maximum(
                        lagged_humidity - damping * correction_humidity, 1e-3
                    )

                # Implicit SM update: solve SM equation separately using
                # current P and E from the T+q Newton iterate.
                # SM = SM_start + dt * ((P - E) / capacity - SM / tau_slow)
                # => SM * (1 + dt / tau_slow) = SM_start + dt * (P - E) / capacity
                # => SM = (SM_start + dt * (P - E) / capacity) / (1 + dt / tau_slow)
                if include_implicit_soil and soil_moisture_iter is not None:
                    # Use the evap_rate and precip_rate already computed
                    # inside the humidity tendency block above
                    if include_implicit_humidity:
                        p_minus_e = precip_rate - evap_rate
                        source_rate = p_minus_e / SOIL_CAPACITY_KG_M2
                        # Semi-implicit backward Euler:
                        # SM_new = (SM_start + dt * source_rate) / (1 + dt / tau_slow)
                        denom_sm = 1.0 + dt_seconds / SOIL_TAU_SLOW_SECONDS
                        sm_new = np.where(
                            surface_context.land_mask,
                            (start_soil_moisture + dt_seconds * source_rate) / denom_sm,
                            1.0,
                        )
                        sm_new = np.clip(sm_new, 0.0, 1.0)
                        # Light under-relaxation: take small steps toward equilibrium
                        # to prevent ITCZ-edge bistability from causing oscillation.
                        SM_RELAX = 0.15
                        soil_moisture_iter = (
                            SM_RELAX * sm_new + (1.0 - SM_RELAX) * soil_moisture_iter
                        )
                        soil_moisture_iter = np.clip(soil_moisture_iter, 0.0, 1.0)
                        soil_moisture_iter = np.where(
                            surface_context.land_mask, soil_moisture_iter, 1.0
                        )

                step = prev_temp - temp_next
                if np.max(np.abs(step)) < NEWTON_STEP_TOLERANCE_K:
                    break

        # Return a state that is internally consistent with the converged temperature.
        final_temp = np.maximum(temp_next, temperature_floor)
        nlayers_final = final_temp.shape[0]

        # Ensure final BL is also overridden (consistent with Newton loop)
        # Get wind for final precipitation/soil calculation
        if lagged_boundary_layer_wind_field is not None:
            wind_u, wind_v = (
                lagged_boundary_layer_wind_field[0],
                lagged_boundary_layer_wind_field[1],
            )
            wind_speed_ref = lagged_boundary_layer_wind_field[2]
        elif lagged_wind_field is not None:
            wind_u, wind_v = lagged_wind_field[0], lagged_wind_field[1]
            wind_speed_ref = lagged_wind_field[2]
        else:
            wind_u = wind_v = np.zeros_like(final_temp[0])
            wind_speed_ref = None

        # Initialize humidity if not yet available (first timestep bootstrap)
        if lagged_humidity is None:
            t_for_humidity = final_temp[1] if nlayers_final == 3 else final_temp[0]
            t_atm = final_temp[2] if nlayers_final == 3 else None
            current_itcz_init = _compute_itcz_from_temp(final_temp)
            lagged_humidity, lagged_precipitation = compute_humidity_and_precipitation(
                wind_u,
                wind_v,
                surface_context.land_mask,
                surface_context.lat2d,
                surface_context.lon2d,
                t_for_humidity,
                itcz_rad=current_itcz_init,
                atmosphere_temperature=t_atm,
            )

        # Humidity is evolved in the implicit Newton solver with diffusion for stability.
        # Apply saturation cap only AFTER Newton converges (not inside loop) to avoid
        # breaking Newton convergence. Allow supersaturation for numerical stability.
        condensation_precip = np.zeros_like(final_temp[0])
        if lagged_humidity is not None:
            t_for_cap = final_temp[1] if nlayers_final == 3 else final_temp[0]
            # Over ocean, our BL midpoint (~500m) is colder than the real
            # near-surface marine BL (~250m midpoint for a ~500m ocean BL).
            # Correct q_sat cap upward so moisture isn't artificially wrung out.
            ocean_bl_correction_K = (
                (BOUNDARY_LAYER_HEIGHT_M - 500.0) / 2.0 * STANDARD_LAPSE_RATE_K_PER_M
            )
            t_for_cap_corrected = t_for_cap + np.where(
                surface_context.land_mask, 0.0, ocean_bl_correction_K
            )
            q_sat = compute_saturation_specific_humidity(t_for_cap_corrected)
            q_max = q_sat  # Cap at saturation — excess condenses as convective precip
            q_excess = np.maximum(lagged_humidity - q_max, 0.0)
            lagged_humidity = np.clip(lagged_humidity, 1e-6, q_max)

            # Excess moisture condenses → precipitation + latent heat release
            if nlayers_final == 3 and np.any(q_excess > 0):
                # Convert excess specific humidity to precipitation (kg/m²/s)
                condensation_precip = q_excess * COLUMN_MASS_KG_M2 / dt_seconds
                # Latent heat release split between BL and atmosphere
                bl_frac = np.where(surface_context.land_mask, 0.05, 0.20)
                L_v = LATENT_HEAT_VAPORIZATION_J_KG
                final_temp[1] += (
                    condensation_precip
                    * L_v
                    * bl_frac
                    * dt_seconds
                    / BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K
                )
                final_temp[2] += (
                    condensation_precip
                    * L_v
                    * (1 - bl_frac)
                    * dt_seconds
                    / ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K
                )

        # Compute final precipitation from converged humidity using prognostic P only.
        # SM is now solved inside the Newton loop, eliminating phantom precipitation
        # from diagnostic humidity.
        t_for_humidity = final_temp[1] if nlayers_final == 3 else final_temp[0]
        t_atm = final_temp[2] if nlayers_final == 3 else None
        _, precip_prognostic = compute_humidity_and_precipitation(
            wind_u,
            wind_v,
            surface_context.land_mask,
            surface_context.lat2d,
            surface_context.lon2d,
            t_for_humidity,
            itcz_rad=itcz_rad,
            atmosphere_temperature=t_atm,
            humidity_q=lagged_humidity,
        )
        if precip_prognostic is None:
            precip_prognostic = np.zeros_like(final_temp[0])
        final_precipitation = precip_prognostic
        # Add condensation precipitation from saturation clamp
        if lagged_humidity is not None:
            final_precipitation = final_precipitation + condensation_precip

        # Compute evaporation for precipitation recycling/eddy additions
        if latent_heat_model is not None:
            boundary_temp = final_temp[1] if nlayers_final == 3 else None
            tendencies = latent_heat_model.compute_tendencies(
                surface_temperature_K=final_temp[0],
                atmosphere_temperature_K=final_temp[-1],
                humidity_q=lagged_humidity,
                wind_speed_reference_m_s=wind_speed_ref,
                itcz_rad=itcz_rad,
                boundary_layer_temperature_K=boundary_temp,
                precipitation_rate=final_precipitation,
                soil_moisture=soil_moisture_iter,
                vegetation_fraction=state.vegetation_fraction,
            )
            evaporation_rate = tendencies[-1]
        else:
            q_sat = compute_saturation_specific_humidity(t_for_humidity)
            tau_evap = 7 * 86400.0
            evaporation_rate = np.maximum(q_sat - lagged_humidity, 0) * COLUMN_MASS_KG_M2 / tau_evap

        # Add precipitation recycling and eddy precip to final precipitation
        if lagged_humidity is not None:
            grid_deg = abs(surface_context.lat2d[1, 0] - surface_context.lat2d[0, 0])
            final_precipitation = final_precipitation + compute_precipitation_recycling(
                evaporation_rate,
                lagged_humidity,
                wind_speed_ref
                if wind_speed_ref is not None
                else np.full_like(lagged_humidity, 3.0),
                surface_context.land_mask,
                resolution_deg=grid_deg,
            )
            grad_q_soil = compute_scalar_gradient_magnitude(
                lagged_humidity,
                surface_context.lat2d,
                surface_context.lon2d,
            )
            q_sat_soil = compute_saturation_specific_humidity(t_for_humidity)
            rh_soil = np.clip(lagged_humidity / np.maximum(q_sat_soil, 1e-10), 0.0, 1.0)
            final_precipitation = final_precipitation + compute_eddy_precipitation(
                lagged_humidity,
                grad_q_soil,
                eddy_kappa,
                rh_soil,
            )

        # Soil moisture: use Newton-converged SM with fast-drain correction only
        # (slow drainage and P-E balance are handled inside the Newton solve)
        if soil_moisture_iter is not None:
            excess = np.maximum(soil_moisture_iter - SOIL_THETA_FC, 0.0)
            decay_fast = np.exp(-dt_seconds / SOIL_TAU_FAST_SECONDS)
            new_soil = np.minimum(soil_moisture_iter, SOIL_THETA_FC) + excess * decay_fast
            new_soil = np.clip(new_soil, 0.0, 1.0)
            new_soil = np.where(surface_context.land_mask, new_soil, 1.0)
        elif state.soil_moisture is not None:
            new_soil = state.soil_moisture
        else:
            # Bootstrap: initialize SM from relative humidity
            q_sat_init = compute_saturation_specific_humidity(t_for_humidity)
            rh_init = (
                np.clip(lagged_humidity / np.maximum(q_sat_init, 1e-10), 0, 1)
                if lagged_humidity is not None
                else np.full_like(final_temp[0], 0.5)
            )
            new_soil = np.where(surface_context.land_mask, rh_init, 1.0)

        # Compute cloud fractions for diagnostics
        convective_frac = None
        stratiform_frac = None
        marine_sc_frac = None
        final_vertical_velocity = None
        if lagged_humidity is not None and nlayers_final >= 2:
            # Compute relative humidity
            # Over ocean, use warmer reference T for q_sat (ocean BL is shallower)
            ocean_bl_corr = (BOUNDARY_LAYER_HEIGHT_M - 500.0) / 2.0 * STANDARD_LAPSE_RATE_K_PER_M
            t_for_rh = np.where(
                surface_context.land_mask, t_for_humidity, t_for_humidity + ocean_bl_corr
            )
            rh = specific_humidity_to_relative_humidity(
                lagged_humidity,
                t_for_rh,
                itcz_rad=itcz_rad,
                lat2d=surface_context.lat2d,
                lon2d=surface_context.lon2d,
            )
            # Compute vertical velocity from wind divergence + frontal lifting
            divergence = compute_divergence(
                wind_u, wind_v, surface_context.lat2d, surface_context.lon2d
            )
            vertical_velocity = compute_vertical_velocity_from_divergence(divergence)
            # Add frontal (warm advection) component
            lat_rad = np.deg2rad(surface_context.lat2d)
            dy_m = (
                np.deg2rad(surface_context.lat2d[1, 0] - surface_context.lat2d[0, 0])
                * R_EARTH_METERS
            )
            dx_m = (
                np.deg2rad(surface_context.lon2d[0, 1] - surface_context.lon2d[0, 0])
                * R_EARTH_METERS
                * np.cos(lat_rad)
            )
            w_frontal = compute_vertical_velocity_from_warm_advection(
                t_for_humidity,
                wind_u,
                wind_v,
                dx_m,
                abs(dy_m),
            )
            vertical_velocity = vertical_velocity + w_frontal
            final_vertical_velocity = vertical_velocity

            # Compute cloud output
            cloud_output = compute_clouds_and_precipitation(
                T_bl_K=t_for_humidity,
                T_atm_K=t_atm if t_atm is not None else t_for_humidity,
                q=lagged_humidity,
                rh=rh,
                vertical_velocity=vertical_velocity,
                T_surface_K=final_temp[0],
                ocean_mask=~surface_context.land_mask,
            )
            convective_frac = cloud_output.convective_frac
            stratiform_frac = cloud_output.stratiform_frac
            marine_sc_frac = cloud_output.marine_sc_frac
            high_cloud_frac = cloud_output.high_cloud_frac
            final_precipitation = cloud_output.total_precip
            # Add orographic precipitation to total
            if lagged_orographic_w is not None and surface_context.orographic_model is not None:
                final_precipitation = (
                    final_precipitation
                    + surface_context.orographic_model.compute_orographic_precipitation(
                        lagged_orographic_w,
                        lagged_humidity,
                        final_temp[1],
                        rh,
                    )
                )
            # Add eddy precipitation (baroclinic moisture transport)
            grad_q_final = compute_scalar_gradient_magnitude(
                lagged_humidity,
                surface_context.lat2d,
                surface_context.lon2d,
            )
            final_precipitation = final_precipitation + compute_eddy_precipitation(
                lagged_humidity,
                grad_q_final,
                eddy_kappa,
                rh,
            )
            # Add precipitation recycling
            grid_deg_final = abs(surface_context.lat2d[1, 0] - surface_context.lat2d[0, 0])
            final_precipitation = final_precipitation + compute_precipitation_recycling(
                evaporation_rate,
                lagged_humidity,
                wind_speed_ref
                if wind_speed_ref is not None
                else np.full_like(lagged_humidity, 3.0),
                surface_context.land_mask,
                resolution_deg=grid_deg_final,
            )

        # Use the lagged ITCZ (passed in, dampened) for consistency with the solve.
        # This ensures the output winds/pressure match what was used during Newton.
        final_itcz = lagged_itcz
        final_state = ModelState(
            temperature=final_temp,
            albedo_field=lagged_albedo_field,
            wind_field=lagged_wind_field,
            humidity_field=lagged_humidity,
            boundary_layer_wind_field=lagged_boundary_layer_wind_field,
            ocean_current_field=lagged_ocean_current_field,
            ocean_ekman_current_field=lagged_ocean_ekman_current_field,
            ocean_current_psi=lagged_ocean_current_psi,
            ocean_ekman_pumping=lagged_ekman_pumping,
            deep_ocean_temperature=deep_ocean_temp_2d,
            precipitation_field=final_precipitation,
            soil_moisture=new_soil,
            vegetation_fraction=state.vegetation_fraction,  # Carry forward from input state
            convective_cloud_frac=convective_frac,
            stratiform_cloud_frac=stratiform_frac,
            marine_sc_cloud_frac=marine_sc_frac,
            high_cloud_frac=high_cloud_frac,
            vertical_velocity=final_vertical_velocity,
        )

        final_state.albedo_field = surface_context.albedo_model.apply_snow_albedo(
            base_albedo_field,
            final_temp[0],
            soil_moisture=new_soil,
            vegetation_fraction=state.vegetation_fraction,
            effective_mu=effective_mu,
            ocean_albedo=ocean_albedo,
            ice_sheet_mask=surface_context.ice_sheet_mask,
        )

        # Update wind diagnostics for output using the converged ITCZ.
        if surface_context.wind_model:
            if final_temp.shape[0] == 3:
                # Three-layer: compute separate winds for boundary layer and free atmosphere
                # Free atmosphere wind: geostrophic (no drag), with BL temp for column average
                final_state.wind_field = surface_context.wind_model.wind_field(
                    final_temp[2],
                    temperature_boundary_layer=final_temp[1],
                    itcz_rad=final_itcz,
                    ekman_drag=False,
                )
                # Boundary layer wind: T_atm for pressure gradient, T_BL for drag
                final_state.boundary_layer_wind_field = surface_context.wind_model.wind_field(
                    final_temp[2],
                    temperature_boundary_layer=final_temp[1],
                    itcz_rad=final_itcz,
                    ekman_drag=True,
                )
                if surface_context.orographic_model is not None:
                    bl_u, bl_v = (
                        final_state.boundary_layer_wind_field[0],
                        final_state.boundary_layer_wind_field[1],
                    )
                    bl_u_b, bl_v_b = surface_context.orographic_model.apply_flow_blocking(
                        bl_u, bl_v
                    )
                    final_state.boundary_layer_wind_field = (
                        bl_u_b,
                        bl_v_b,
                        np.hypot(bl_u_b, bl_v_b),
                    )
            else:
                # One-layer: single wind with drag
                final_state.wind_field = surface_context.wind_model.wind_field(
                    final_temp[0], itcz_rad=final_itcz, ekman_drag=True
                )

        # Compute surface pressure for diagnostic output
        final_state.surface_pressure = compute_pressure(
            final_temp[0],
            itcz_rad=final_itcz,
            lat2d=surface_context.lat2d,
            lon2d=surface_context.lon2d,
        )

        # Store ITCZ on state so next month can blend with it
        final_state.itcz_rad = final_itcz

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
    ocean_advection_enabled: bool = True,
    latent_heat_model: LatentHeatExchangeModel | None = None,
    advection_operator: AdvectionOperator | None = None,
    humidity_diffusion_operator: DiffusionOperator | None = None,
    vertical_motion_cfg: VerticalMotionConfig | None = None,
    monthly_effective_mu: np.ndarray | None = None,
    monthly_ocean_albedo: np.ndarray | None = None,
    monthly_itcz: list[np.ndarray] | None = None,
) -> list[ModelState]:
    """Propagate the state through 12 implicit steps."""
    states: list[ModelState] = []
    with time_block("evolve_year"):
        cell_areas = spherical_cell_area(
            surface_context.lon2d, surface_context.lat2d, earth_radius_m=R_EARTH_METERS
        )
        for month_n in range(12):
            month = (month_n + 2) % 12  # start from March so initial guess is better
            # Use precomputed ITCZ if available, otherwise compute from temperature
            if monthly_itcz is not None:
                itcz_rad = monthly_itcz[month]
            else:
                nlayers = state.temperature.shape[0]
                itcz_temp = state.temperature[1] if nlayers >= 3 else state.temperature[0]
                itcz_rad = compute_itcz_latitude(
                    np.maximum(itcz_temp, temperature_floor),
                    surface_context.lat2d,
                    cell_areas,
                )
            state = monthly_step(
                state,
                monthly_insolation[month],
                itcz_rad,
                month_durations[month],
                rhs_fn=rhs_fn,
                rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
                temperature_floor=temperature_floor,
                solver_cache=solver_cache,
                surface_context=surface_context,
                ocean_advection_enabled=ocean_advection_enabled,
                latent_heat_model=latent_heat_model,
                advection_operator=advection_operator,
                humidity_diffusion_operator=humidity_diffusion_operator,
                vertical_motion_cfg=vertical_motion_cfg,
                effective_mu=monthly_effective_mu[month]
                if monthly_effective_mu is not None
                else None,
                ocean_albedo=monthly_ocean_albedo[month]
                if monthly_ocean_albedo is not None
                else None,
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
    # Adaptive regularization: more history needs more regularization for stability
    # This prevents ill-conditioning when using many past iterates
    regularisation = (1e-10 * m * m) * scale + 1e-14
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
    ocean_advection_enabled: bool = True,
    latent_heat_model: LatentHeatExchangeModel | None = None,
    advection_operator: AdvectionOperator | None = None,
    humidity_diffusion_operator: DiffusionOperator | None = None,
    vertical_motion_cfg: VerticalMotionConfig | None = None,
    monthly_effective_mu: np.ndarray | None = None,
    monthly_ocean_albedo: np.ndarray | None = None,
) -> list[ModelState]:
    """Solve for the periodic annual climate cycle using Anderson acceleration.

    Finds the fixed point P(state) = state where P is the year-forward
    evolution operator, using Anderson acceleration for faster convergence.

    Anderson acceleration mixes all prognostic state variables (temperature,
    humidity, soil moisture) to ensure consistency. Diagnostic variables
    (precipitation, clouds, albedo, wind) are recomputed from the mixed state.
    """

    # Scale factors for Anderson state vector (for numerical stability)
    T_SCALE = 300.0  # ~order of magnitude for Kelvin
    Q_SCALE = 0.01  # ~order of magnitude for specific humidity (kg/kg)
    SOIL_SCALE = 1.0  # already 0-1
    cell_areas_periodic = spherical_cell_area(
        surface_context.lon2d, surface_context.lat2d, earth_radius_m=R_EARTH_METERS
    )

    def _itcz_from_temp(temp: np.ndarray, tau: float = 2.0) -> np.ndarray:
        nlayers = temp.shape[0]
        itcz_temp = temp[1] if nlayers >= 3 else temp[0]
        return compute_itcz_latitude(
            np.maximum(itcz_temp, temperature_floor),
            surface_context.lat2d,
            cell_areas_periodic,
            tau=tau,
        )

    # Determine which prognostic variables are enabled based on model configuration
    # (not based on initial state, since initial state may not have them yet)
    humidity_is_prognostic = latent_heat_model is not None
    soil_is_prognostic = latent_heat_model is not None  # Soil evolves with humidity

    with time_block("find_periodic_climate_cycle"):
        state = initial_state
        temp = initial_state.temperature
        state.temperature = np.maximum(temp, temperature_floor)
        states = [initial_state] * 12
        residual_history: list[np.ndarray] = []
        advanced_history: list[np.ndarray] = []
        residual_max = 0.0

        # Track vegetation fraction - computed from annual precipitation (lagged by 1 year)
        # Use 50% initial vegetation fraction for first year
        current_vegetation_fraction: np.ndarray | None = None

        for iter_idx in range(FIXED_POINT_MAX_ITERS):
            with time_block("periodic_iteration"):
                # Set vegetation fraction on input state for this year
                # (computed from previous year's precipitation)
                state.vegetation_fraction = current_vegetation_fraction

                # Precompute ITCZ for each month from previous iteration's states.
                # Use sharp tau (0.5) to tightly track the thermal equator.
                # Dampen updates: blend new ITCZ with previous to prevent the
                # ITCZ→pressure→T→ITCZ feedback loop from overshooting.
                # The gain of this loop is >1 at tau=0.5, so we need alpha < 1/gain.
                ITCZ_UPDATE_ALPHA = 0.15  # fraction of new ITCZ to use each iteration
                monthly_itcz: list[np.ndarray] = [None] * 12  # type: ignore[list-item]
                for month_n in range(12):
                    cal_month = (month_n + 2) % 12
                    s = states[month_n]
                    new_itcz = _itcz_from_temp(s.temperature, tau=0.5)
                    if s.itcz_rad is not None:
                        monthly_itcz[cal_month] = (
                            ITCZ_UPDATE_ALPHA * new_itcz + (1.0 - ITCZ_UPDATE_ALPHA) * s.itcz_rad
                        )
                    else:
                        monthly_itcz[cal_month] = new_itcz

                advanced_states = evolve_year(
                    state,
                    monthly_insolation,
                    month_durations,
                    rhs_fn=rhs_fn,
                    rhs_temperature_derivative_fn=rhs_derivative_fn,
                    temperature_floor=temperature_floor,
                    solver_cache=solver_cache,
                    surface_context=surface_context,
                    ocean_advection_enabled=ocean_advection_enabled,
                    latent_heat_model=latent_heat_model,
                    advection_operator=advection_operator,
                    humidity_diffusion_operator=humidity_diffusion_operator,
                    vertical_motion_cfg=vertical_motion_cfg,
                    monthly_effective_mu=monthly_effective_mu,
                    monthly_ocean_albedo=monthly_ocean_albedo,
                    monthly_itcz=monthly_itcz,
                )

                advanced = advanced_states[-1]
                prev_end = states[-1]

                # Check what prognostic variables are actually available
                # (may differ from config on first iteration when they're being bootstrapped)
                has_humidity = (
                    humidity_is_prognostic
                    and advanced.humidity_field is not None
                    and prev_end.humidity_field is not None
                )
                has_soil = (
                    soil_is_prognostic
                    and advanced.soil_moisture is not None
                    and prev_end.soil_moisture is not None
                )

                # Compute residuals for all prognostic state variables
                # Using end-of-year state (the state we're iterating on)
                # Temperature: use all layers for the residual
                temp_residual = (advanced.temperature - prev_end.temperature) / T_SCALE

                # Humidity residual (if prognostic humidity is enabled)
                if has_humidity:
                    humidity_residual = (
                        advanced.humidity_field - prev_end.humidity_field
                    ) / Q_SCALE
                else:
                    humidity_residual = np.array([])

                # Soil moisture residual (if prognostic)
                if has_soil:
                    soil_residual = (advanced.soil_moisture - prev_end.soil_moisture) / SOIL_SCALE
                else:
                    soil_residual = np.array([])

                # Combined residual for Anderson acceleration (scaled)
                # ITCZ is not included — it's fully lagged from previous iteration's
                # temperatures and updated implicitly through temperature changes.
                residual_combined = np.concatenate(
                    [
                        temp_residual.ravel(),
                        humidity_residual.ravel(),
                        soil_residual.ravel(),
                    ]
                )

                # Compute month-by-month surface temperature residual for convergence diagnostics
                # (This is separate from the Anderson residual - just for reporting)
                monthly_temp_diff = np.array(
                    [
                        advanced_states[i].temperature[0] - states[i].temperature[0]
                        for i in range(12)
                    ]
                )
                residual_rms = np.sqrt(np.mean(np.square(monthly_temp_diff)))
                residual_95p = np.percentile(np.abs(monthly_temp_diff), 95)
                residual_max = np.max(np.abs(monthly_temp_diff))
                print(
                    f"iter {iter_idx:2d}: RMS={residual_rms:.3f}K  95p={residual_95p:.3f}K  max={residual_max:.3f}K"
                )

                # Compute annual precipitation and growing season for vegetation
                # Sum monthly precipitation rates (kg/m²/s) weighted by month duration
                precip_fields = [s.precipitation_field for s in advanced_states]
                if all(p is not None for p in precip_fields):
                    annual_precip_kg_m2 = sum(
                        p * month_durations[i] for i, p in enumerate(precip_fields)
                    )
                    # Convert kg/m² to mm (1 kg/m² = 1 mm)
                    annual_precip_mm = annual_precip_kg_m2

                    # Get monthly temperatures for growing season calculation
                    # Use boundary layer temperature (more representative of vegetation conditions)
                    # For 3-layer: use BL temp; for 1-layer: use surface temp
                    monthly_temps_c = np.array(
                        [
                            (s.temperature[1] if s.temperature.shape[0] >= 3 else s.temperature[0])
                            - 273.15
                            for s in advanced_states
                        ]
                    )

                    # Compute vegetation fraction from precipitation and growing season
                    current_vegetation_fraction = (
                        surface_context.albedo_model.compute_vegetation_fraction(
                            annual_precip_mm,
                            monthly_temperatures_c=monthly_temps_c,
                        )
                    )

                if (
                    residual_rms < PERIODIC_FIXED_POINT_TOLERANCE_K
                    and residual_95p < PERIODIC_FIXED_POINT_TOLERANCE_K_95P
                ):
                    # Ensure vegetation fraction and consistent albedo on all returned states
                    final_states = [advanced_states[(i - 2) % 12] for i in range(12)]
                    if current_vegetation_fraction is not None:
                        base_albedo = surface_context.base_albedo
                        for month_idx, s in enumerate(final_states):
                            s.vegetation_fraction = current_vegetation_fraction
                            # Recompute albedo to be consistent with vegetation fraction
                            month_mu = (
                                monthly_effective_mu[month_idx]
                                if monthly_effective_mu is not None
                                else None
                            )
                            month_ocean_alb = (
                                monthly_ocean_albedo[month_idx]
                                if monthly_ocean_albedo is not None
                                else None
                            )
                            s.albedo_field = surface_context.albedo_model.apply_snow_albedo(
                                base_albedo,
                                s.temperature[0],
                                soil_moisture=s.soil_moisture,
                                vegetation_fraction=current_vegetation_fraction,
                                annual_precip_mm_year=annual_precip_mm,
                                effective_mu=month_mu,
                                ocean_albedo=month_ocean_alb,
                                ice_sheet_mask=surface_context.ice_sheet_mask,
                            )
                    print(
                        f"Converged at iter {iter_idx} (RMS={residual_rms:.3f}K  95p={residual_95p:.3f}K)"
                    )
                    return final_states

                states = advanced_states

                # Build augmented state vector for Anderson acceleration
                # Include all prognostic variables: temperature, humidity, soil moisture
                # (all scaled for numerical stability)
                advanced_temp_scaled = advanced.temperature.ravel() / T_SCALE

                if has_humidity:
                    advanced_humidity_scaled = advanced.humidity_field.ravel() / Q_SCALE
                else:
                    advanced_humidity_scaled = np.array([])

                if has_soil:
                    advanced_soil_scaled = advanced.soil_moisture.ravel() / SOIL_SCALE
                else:
                    advanced_soil_scaled = np.array([])

                advanced_flat = np.concatenate(
                    [
                        advanced_temp_scaled,
                        advanced_humidity_scaled,
                        advanced_soil_scaled,
                    ]
                )

                # Check for state vector shape change (e.g., humidity bootstrapped after iter 0)
                # If shape changed, clear history to avoid size mismatch
                if len(advanced_history) > 0 and advanced_flat.shape != advanced_history[-1].shape:
                    residual_history = []
                    advanced_history = []

                if len(residual_history) == ANDERSON_HISTORY_LIMIT:
                    residual_history.pop(0)
                    advanced_history.pop(0)

                residual_history.append(residual_combined)
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

                    # Sizes for unpacking the combined state vector
                    n_temp = state.temperature.size
                    n_humidity = advanced.humidity_field.size if has_humidity else 0
                    n_soil = advanced.soil_moisture.size if has_soil else 0

                    if coefficients is None:
                        # No Anderson mixing - use advanced state directly
                        T_next = advanced.temperature
                        q_next = advanced.humidity_field
                        soil_next = advanced.soil_moisture
                        residual_history = residual_history[-1:]
                        advanced_history = advanced_history[-1:]
                    else:
                        # Anderson mixing of all prognostic state variables
                        combined = np.zeros_like(advanced_flat)
                        for weight, advanced_state in zip(
                            coefficients, advanced_history, strict=True
                        ):
                            combined += weight * advanced_state

                        # Unpack and rescale the combined state
                        T_next = combined[:n_temp].reshape(state.temperature.shape) * T_SCALE

                        if has_humidity:
                            q_next = (
                                combined[n_temp : n_temp + n_humidity].reshape(
                                    advanced.humidity_field.shape
                                )
                                * Q_SCALE
                            )
                            # Ensure humidity stays positive
                            q_next = np.maximum(q_next, 1e-3)
                        else:
                            q_next = None

                        if has_soil:
                            soil_next = (
                                combined[
                                    n_temp + n_humidity : n_temp + n_humidity + n_soil
                                ].reshape(advanced.soil_moisture.shape)
                                * SOIL_SCALE
                            )
                            # Clip soil moisture to [0, 1]
                            soil_next = np.clip(soil_next, 0.0, 1.0)
                            # Ocean cells always saturated
                            soil_next = np.where(surface_context.land_mask, soil_next, 1.0)
                        else:
                            soil_next = None

                        # Validate the mixed state
                        if not np.all(np.isfinite(T_next)):
                            T_next = advanced.temperature
                            q_next = advanced.humidity_field
                            soil_next = advanced.soil_moisture
                            residual_history = residual_history[-1:]
                            advanced_history = advanced_history[-1:]

                # Under-relax: blend Anderson output with previous state
                if ANDERSON_RELAXATION < 1.0:
                    alpha = ANDERSON_RELAXATION
                    T_next = alpha * T_next + (1.0 - alpha) * state.temperature
                    if q_next is not None and state.humidity_field is not None:
                        q_next = alpha * q_next + (1.0 - alpha) * state.humidity_field
                        q_next = np.maximum(q_next, 1e-3)
                    if soil_next is not None and state.soil_moisture is not None:
                        soil_next = alpha * soil_next + (1.0 - alpha) * state.soil_moisture
                        soil_next = np.clip(soil_next, 0.0, 1.0)

                # Clamp soil moisture changes to prevent oscillation at ITCZ-edge
                # cells where P switches on/off between iterations.
                if soil_next is not None and state.soil_moisture is not None:
                    MAX_SOIL_STEP = 0.1  # max SM change per outer iteration
                    delta_sm = soil_next - state.soil_moisture
                    delta_sm = np.clip(delta_sm, -MAX_SOIL_STEP, MAX_SOIL_STEP)
                    soil_next = state.soil_moisture + delta_sm
                    soil_next = np.clip(soil_next, 0.0, 1.0)
                    soil_next = np.where(surface_context.land_mask, soil_next, 1.0)

                # Clamp per-cell changes to prevent oscillation at a few
                # tropical cells (ITCZ migration zone) from blocking
                # convergence while the rest of the globe has settled.
                MAX_TEMP_STEP_K = 5.0
                delta_T = T_next - state.temperature
                delta_T = np.clip(delta_T, -MAX_TEMP_STEP_K, MAX_TEMP_STEP_K)
                T_next = state.temperature + delta_T

                # Clamp humidity changes to prevent oscillation at the
                # ITCZ edge where small q differences cause large latent
                # heat swings within the monthly Newton solve.
                if q_next is not None and state.humidity_field is not None:
                    MAX_Q_STEP = 0.001  # kg/kg (~1 g/kg per iteration)
                    delta_q = q_next - state.humidity_field
                    delta_q = np.clip(delta_q, -MAX_Q_STEP, MAX_Q_STEP)
                    q_next = state.humidity_field + delta_q
                    q_next = np.maximum(q_next, 1e-3)

                # Apply Anderson-accelerated state with mixed prognostic variables
                # Diagnostics (albedo, wind, precipitation, clouds) will be recomputed
                # at the start of the next year's first monthly step
                state = ModelState(
                    temperature=T_next,
                    albedo_field=None,  # Recomputed from mixed T at start of next month
                    wind_field=None,  # Recomputed from mixed T at start of next month
                    humidity_field=q_next,  # Anderson-mixed humidity
                    soil_moisture=soil_next,  # Anderson-mixed soil moisture
                    precipitation_field=None,  # Recomputed from mixed q at start of next month
                    vegetation_fraction=current_vegetation_fraction,  # From this year's precipitation
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
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]
):
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
    reset_profiler()

    # Build all operators and grids from configuration
    with time_block("build_operators"):
        operators = build_model_operators(resolution_deg, model_config)

    # Humidity diffusion uses a reduced diffusivity (Lewis number < 1) because
    # moisture precipitates out during eddy transport, reducing effective flux.
    humidity_diffusion_op = (
        (operators.diffusion_operator.humidity) if operators.latent_heat_cfg.enabled else None
    )

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
            wind_model=operators.wind_model,
            advection_operator=operators.advection_operator,
            lon2d=operators.lon2d,
            lat2d=operators.lat2d,
            ocean_advection_cfg=operators.ocean_advection_cfg,
            vertical_motion_cfg=operators.vertical_motion_cfg,
            humidity_diffusion_operator=humidity_diffusion_op,
            orographic_model=operators.orographic_model,
            amoc_velocity=operators.amoc_velocity,
        )
        rhs_fn, rhs_derivative_fn = create_rhs_functions(rhs_inputs)

    # Create initial guess using radiative equilibrium
    with time_block("initial_guess"):
        initial_albedo = operators.albedo_model.guess_albedo_field()
        initial_temp_guess = radiation.radiative_equilibrium_initial_guess(
            operators.monthly_insolation,
            albedo_field=initial_albedo,
            config=operators.radiation_config,
            land_mask=operators.land_mask,
            lat2d=operators.lat2d,
            lon2d=operators.lon2d,
        )
        initial_state = ModelState(
            temperature=initial_temp_guess,
            albedo_field=initial_albedo,
            wind_field=None,
            humidity_field=None,
        )

    # Build latent heat model for prognostic humidity
    latent_heat_model = None
    if operators.latent_heat_cfg.enabled:
        latent_heat_model = LatentHeatExchangeModel(
            land_mask=operators.land_mask,
            surface_heat_capacity_J_m2_K=operators.heat_capacity_field,
            atmosphere_heat_capacity_J_m2_K=operators.radiation_config.atmosphere_heat_capacity,
            wind_model=operators.wind_model,
            config=operators.latent_heat_cfg,
            boundary_layer_heat_capacity_J_m2_K=operators.radiation_config.boundary_layer_heat_capacity,
        )

    # Solve for periodic cycle
    ocean_advection_enabled = operators.ocean_advection_cfg.enabled

    monthly_states = find_periodic_climate_cycle(
        initial_state=initial_state,
        monthly_insolation=operators.monthly_insolation,
        month_durations=operators.month_durations,
        rhs_fn=rhs_fn,
        rhs_derivative_fn=rhs_derivative_fn,
        surface_context=operators.surface_context,
        temperature_floor=operators.radiation_config.temperature_floor,
        solver_cache=operators.solver_cache,
        ocean_advection_enabled=ocean_advection_enabled,
        latent_heat_model=latent_heat_model,
        advection_operator=operators.advection_operator,
        humidity_diffusion_operator=humidity_diffusion_op,
        vertical_motion_cfg=operators.vertical_motion_cfg,
        monthly_effective_mu=operators.monthly_effective_mu,
        monthly_ocean_albedo=operators.monthly_ocean_albedo,
    )

    # Update wind fields if wind model is enabled
    if operators.wind_model is not None and operators.wind_model.enabled:
        with time_block("update_wind_fields"):
            cell_areas = spherical_cell_area(
                operators.lon2d, operators.lat2d, earth_radius_m=R_EARTH_METERS
            )

            for idx, month_state in enumerate(monthly_states):
                nlayers = month_state.temperature.shape[0]

                # Compute ITCZ from boundary layer temp (3-layer) or surface temp (1-layer)
                itcz_temp = (
                    month_state.temperature[1] if nlayers >= 3 else month_state.temperature[0]
                )
                itcz_temp = np.maximum(itcz_temp, operators.radiation_config.temperature_floor)
                itcz_rad = compute_itcz_latitude(itcz_temp, operators.lat2d, cell_areas)

                # Compute wind fields if not already present
                if nlayers == 3:
                    # Three-layer: compute both winds if not present
                    atm_wind = month_state.wind_field
                    bl_wind = month_state.boundary_layer_wind_field

                    if atm_wind is None:
                        atm_wind = operators.wind_model.wind_field(
                            month_state.temperature[2],
                            temperature_boundary_layer=month_state.temperature[1],
                            itcz_rad=itcz_rad,
                            ekman_drag=False,
                        )
                    if bl_wind is None:
                        bl_wind = operators.wind_model.wind_field(
                            month_state.temperature[2],
                            temperature_boundary_layer=month_state.temperature[1],
                            itcz_rad=itcz_rad,
                            ekman_drag=True,
                        )
                        if operators.orographic_model is not None:
                            bu, bv = bl_wind[0], bl_wind[1]
                            bu_b, bv_b = operators.orographic_model.apply_flow_blocking(bu, bv)
                            bl_wind = (bu_b, bv_b, np.hypot(bu_b, bv_b))

                    # Compute ocean currents from boundary layer wind
                    ocean_current_field = None
                    ocean_ekman_current_field = None
                    ocean_current_psi = None
                    ekman_pumping = None
                    if ocean_advection_enabled and bl_wind is not None:
                        ocean_results = compute_ocean_currents(
                            bl_wind[0],
                            bl_wind[1],
                            operators.lon2d,
                            operators.lat2d,
                            operators.land_mask,
                            include_stommel=True,
                        )
                        ocean_current_field = (
                            ocean_results["u_velocity"],
                            ocean_results["v_velocity"],
                        )
                        ocean_ekman_current_field = (
                            ocean_results["u_ekman"],
                            ocean_results["v_ekman"],
                        )
                        ocean_current_psi = ocean_results["psi"]
                        ekman_pumping = ocean_results["w_ekman_pumping"]

                    # Deep ocean temperature (static)
                    deep_temp = compute_deep_ocean_temperature(operators.lat2d[:, 0])
                    deep_temp_2d = np.broadcast_to(
                        deep_temp[:, np.newaxis], operators.lat2d.shape
                    ).copy()

                    monthly_states[idx] = ModelState(
                        temperature=month_state.temperature,
                        albedo_field=month_state.albedo_field,
                        wind_field=atm_wind,
                        humidity_field=month_state.humidity_field,
                        boundary_layer_wind_field=bl_wind,
                        ocean_current_field=ocean_current_field,
                        ocean_ekman_current_field=ocean_ekman_current_field,
                        ocean_current_psi=ocean_current_psi,
                        ocean_ekman_pumping=ekman_pumping,
                        deep_ocean_temperature=deep_temp_2d,
                        precipitation_field=month_state.precipitation_field,
                        soil_moisture=month_state.soil_moisture,
                        vegetation_fraction=month_state.vegetation_fraction,
                        convective_cloud_frac=month_state.convective_cloud_frac,
                        stratiform_cloud_frac=month_state.stratiform_cloud_frac,
                        marine_sc_cloud_frac=month_state.marine_sc_cloud_frac,
                        high_cloud_frac=month_state.high_cloud_frac,
                        vertical_velocity=month_state.vertical_velocity,
                        surface_pressure=month_state.surface_pressure,
                        itcz_rad=itcz_rad,
                    )
                else:
                    # One-layer: single wind field with drag
                    wind_temperature = select_wind_temperature(month_state.temperature)
                    wind_field = month_state.wind_field or operators.wind_model.wind_field(
                        wind_temperature, itcz_rad=itcz_rad, ekman_drag=True
                    )

                    # Compute ocean currents from atmosphere wind
                    ocean_current_field = None
                    ocean_ekman_current_field = None
                    ocean_current_psi = None
                    ekman_pumping = None
                    if ocean_advection_enabled and wind_field is not None:
                        ocean_results = compute_ocean_currents(
                            wind_field[0],
                            wind_field[1],
                            operators.lon2d,
                            operators.lat2d,
                            operators.land_mask,
                            include_stommel=True,
                        )
                        ocean_current_field = (
                            ocean_results["u_velocity"],
                            ocean_results["v_velocity"],
                        )
                        ocean_ekman_current_field = (
                            ocean_results["u_ekman"],
                            ocean_results["v_ekman"],
                        )
                        ocean_current_psi = ocean_results["psi"]
                        ekman_pumping = ocean_results["w_ekman_pumping"]

                    # Deep ocean temperature (static)
                    deep_temp = compute_deep_ocean_temperature(operators.lat2d[:, 0])
                    deep_temp_2d = np.broadcast_to(
                        deep_temp[:, np.newaxis], operators.lat2d.shape
                    ).copy()

                    monthly_states[idx] = ModelState(
                        temperature=month_state.temperature,
                        albedo_field=month_state.albedo_field,
                        wind_field=wind_field,
                        humidity_field=month_state.humidity_field,
                        boundary_layer_wind_field=month_state.boundary_layer_wind_field,
                        ocean_current_field=ocean_current_field,
                        ocean_ekman_current_field=ocean_ekman_current_field,
                        ocean_current_psi=ocean_current_psi,
                        ocean_ekman_pumping=ekman_pumping,
                        deep_ocean_temperature=deep_temp_2d,
                        precipitation_field=month_state.precipitation_field,
                        soil_moisture=month_state.soil_moisture,
                        vegetation_fraction=month_state.vegetation_fraction,
                        convective_cloud_frac=month_state.convective_cloud_frac,
                        stratiform_cloud_frac=month_state.stratiform_cloud_frac,
                        marine_sc_cloud_frac=month_state.marine_sc_cloud_frac,
                        high_cloud_frac=month_state.high_cloud_frac,
                        vertical_velocity=month_state.vertical_velocity,
                        surface_pressure=month_state.surface_pressure,
                        itcz_rad=itcz_rad,
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
        topographic_elevation=operators.topographic_elevation,
    )
