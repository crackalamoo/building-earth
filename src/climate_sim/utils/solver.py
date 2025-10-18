"""General-purpose periodic solver utilities for energy-balance models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import climate_sim.modeling.radiation as radiation
from climate_sim.modeling.advection import (
    AdvectionConfig,
    AdvectionModel,
)
from climate_sim.modeling.diffusion import (
    DiffusionConfig,
    LayeredDiffusionOperator,
    create_diffusion_operator,
)
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig, AlbedoModel
from climate_sim.utils.grid import create_lat_lon_grid, expand_latitude_field
from climate_sim.utils.solar import DAYS_PER_MONTH, SECONDS_PER_DAY, compute_monthly_insolation_field
from climate_sim.utils.landmask import (
    compute_albedo_field,
    compute_heat_capacity_field,
    compute_land_mask,
)

NEWTON_STEP_TOLERANCE_K = 1.0
PERIODIC_FIXED_POINT_TOLERANCE_K = 1.0
NEWTON_MAX_ITERS = 16
NEWTON_BACKTRACK_REDUCTION = 0.5
NEWTON_BACKTRACK_CUTOFF = 1e-3
FIXED_POINT_MAX_ITERS = 100


@dataclass
class Linearisation:
    diag: np.ndarray
    cross: np.ndarray | None = None
    surface_diffusion_matrix: sparse.csr_matrix | None = None
    atmosphere_diffusion_matrix: sparse.csr_matrix | None = None
    solver_fingerprint: str | None = None


@dataclass
class LinearSolveCache:
    identity_matrices: Dict[int, sparse.csc_matrix] = field(default_factory=dict)
    factorized_solvers: Dict[str, Callable[[np.ndarray], np.ndarray]] = field(default_factory=dict)


DEFAULT_LINEAR_SOLVE_CACHE = LinearSolveCache()


def _get_identity_matrix(size: int, *, cache: LinearSolveCache) -> sparse.csc_matrix:
    identity = cache.identity_matrices.get(size)
    if identity is None:
        identity = sparse.eye(size, format="csc")
        cache.identity_matrices[size] = identity
    return identity


def _fingerprint_csc_matrix(matrix: sparse.csc_matrix) -> str:
    if not sparse.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()
    hasher = hashlib.sha1()
    hasher.update(matrix.shape[0].to_bytes(4, byteorder="little", signed=False))
    hasher.update(matrix.shape[1].to_bytes(4, byteorder="little", signed=False))
    hasher.update(matrix.indptr.tobytes())
    hasher.update(matrix.indices.tobytes())
    hasher.update(matrix.data.tobytes())
    return hasher.hexdigest()


def _get_factorized_solver(
    matrix: sparse.csc_matrix, *, cache: LinearSolveCache, fingerprint: str | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    key = fingerprint or _fingerprint_csc_matrix(matrix)
    solver = cache.factorized_solvers.get(key)
    if solver is None:
        solver = splinalg.factorized(matrix)
        cache.factorized_solvers[key] = solver
    return solver


@dataclass
class DiagnosticParameters:
    """Diagnostic (non-solver) parameters for the model."""
    albedo_field: np.ndarray
    wind_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

@dataclass
class DiagnosticModels:
    snow_albedo_model: AlbedoModel
    advection_model: AdvectionModel | None = None

@dataclass
class ModelState:
    """State variables for the climate model."""
    temperature: np.ndarray
    diagnostics: DiagnosticParameters

RhsFunc = Callable[[ModelState, np.ndarray], np.ndarray]
RhsDerivative = Linearisation
RhsDerivativeFunc = Callable[[ModelState, np.ndarray], RhsDerivative]
RhsFactory = Callable[
    [
        np.ndarray,
        LayeredDiffusionOperator,
        RadiationConfig,
    ],
    Tuple[RhsFunc, RhsDerivativeFunc],
]
InitialGuessFunc = Callable[[np.ndarray, np.ndarray, RadiationConfig], np.ndarray]
MonthlyInsolationLatFunc = Callable[[np.ndarray], np.ndarray]
HeatCapacityFieldFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


def monthly_step(
    state: ModelState,
    diagnostic_models: DiagnosticModels,
    insolation_W_m2: np.ndarray,
    dt_seconds: float,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
) -> ModelState:
    """Advance the column temperature one implicit backward-Euler step."""
    start_temp = state.temperature
    temp_next = np.maximum(start_temp, temperature_floor)
    cache = solver_cache or DEFAULT_LINEAR_SOLVE_CACHE

    if start_temp.ndim == 2:
        identity_single_layer = _get_identity_matrix(start_temp.size, cache=cache)
    elif start_temp.ndim == 3 and start_temp.shape[0] == 2:
        identity_single_layer = _get_identity_matrix(
            start_temp.shape[1] * start_temp.shape[2], cache=cache
        )
    else:
        identity_single_layer = None

    # implicit solver loop
    for _ in range(NEWTON_MAX_ITERS):
        temp_capped = np.maximum(temp_next, temperature_floor)
        state_capped = ModelState(
            temperature=temp_capped,
            diagnostics=state.diagnostics,
        )
        rhs_value = rhs_fn(state_capped, insolation_W_m2)
        residual = temp_capped - start_temp - dt_seconds * rhs_value
        linearisation = rhs_temperature_derivative_fn(state_capped, insolation_W_m2)

        if temp_capped.ndim == 2:
            diag = linearisation.diag
            diffusion_matrix = linearisation.surface_diffusion_matrix

            size = temp_capped.size
            residual_flat = residual.ravel()

            if identity_single_layer is not None and identity_single_layer.shape[0] == size:
                jacobian = identity_single_layer.copy()
            else:
                jacobian = _get_identity_matrix(size, cache=cache).copy()
            jacobian -= dt_seconds * sparse.diags(diag.ravel(), format="csc")

            if diffusion_matrix is not None:
                jacobian = jacobian - dt_seconds * diffusion_matrix

            fingerprint = linearisation.solver_fingerprint or _fingerprint_csc_matrix(jacobian)
            linearisation.solver_fingerprint = fingerprint
            solve_linear = _get_factorized_solver(jacobian, cache=cache, fingerprint=fingerprint)
            correction_flat = solve_linear(residual_flat)
            correction = correction_flat.reshape(temp_capped.shape)
        else:
            if temp_capped.ndim < 1 or temp_capped.shape[0] != 2:
                raise ValueError("Layered derivative requires a two-layer temperature field")

            diag = linearisation.diag
            cross = linearisation.cross
            if cross is None:
                raise ValueError("Missing cross-layer coupling for layered system")

            surface_diag = diag[0]
            atmosphere_diag = diag[1]
            surface_matrix = linearisation.surface_diffusion_matrix
            atmosphere_matrix = linearisation.atmosphere_diffusion_matrix

            nlat, nlon = surface_diag.shape
            size = nlat * nlon

            identity = identity_single_layer
            if identity is None or identity.shape[0] != size or identity.shape[1] != size:
                identity = _get_identity_matrix(size, cache=cache)
            surface_block = identity.copy()
            surface_block -= dt_seconds * sparse.diags(surface_diag.ravel(), format="csc")
            atmosphere_block = identity.copy()
            atmosphere_block -= dt_seconds * sparse.diags(atmosphere_diag.ravel(), format="csc")

            if surface_matrix is not None:
                surface_block = surface_block - dt_seconds * surface_matrix
            if atmosphere_matrix is not None:
                atmosphere_block = atmosphere_block - dt_seconds * atmosphere_matrix

            coupling_surface_atm = -dt_seconds * sparse.diags(
                cross[0, 1].ravel(), format="csc"
            )
            coupling_atm_surface = -dt_seconds * sparse.diags(
                cross[1, 0].ravel(), format="csc"
            )

            jacobian = sparse.bmat(
                [[surface_block, coupling_surface_atm], [coupling_atm_surface, atmosphere_block]],
                format="csc",
            )

            residual_flat = np.concatenate(
                [residual[0].ravel(), residual[1].ravel()],
                axis=0,
            )

            assert isinstance(jacobian, sparse.csc_matrix)
            fingerprint = linearisation.solver_fingerprint or _fingerprint_csc_matrix(jacobian)
            linearisation.solver_fingerprint = fingerprint
            solve_linear = _get_factorized_solver(jacobian, cache=cache, fingerprint=fingerprint)
            correction_flat = solve_linear(residual_flat)
            correction_surface = correction_flat[:size].reshape(surface_diag.shape)
            correction_atmosphere = correction_flat[size:].reshape(atmosphere_diag.shape)
            correction = np.stack([correction_surface, correction_atmosphere])

        damping = 1.0
        max_residual = float(np.max(np.abs(residual)))
        accepted = False
        prev_temp = temp_next
        temp_candidate = temp_next
        residual_candidate = residual

        while damping >= NEWTON_BACKTRACK_CUTOFF:
            temp_candidate = np.maximum(prev_temp - damping * correction, temperature_floor)
            state_candidate = ModelState(
                temperature=temp_candidate,
                diagnostics=state.diagnostics,
            )
            rhs_candidate = rhs_fn(state_candidate, insolation_W_m2)
            residual_candidate = temp_candidate - start_temp - dt_seconds * rhs_candidate
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

    return ModelState(
        temperature=temp_next,
        diagnostics=state.diagnostics,
    )


def apply_annual_map(
    state: ModelState,
    diagnostic_models: DiagnosticModels,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
) -> ModelState:
    """Propagate the state through 12 implicit steps and return the end-of-December temperature."""
    for month in range(12):
        state = monthly_step(
            state,
            diagnostic_models,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
            solver_cache=solver_cache,
        )
    return state


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
    diagnostic_models: DiagnosticModels,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
) -> ModelState:
    """Solve P(T) = T for the annual map using Anderson acceleration."""

    state = initial_state
    state.temperature = np.maximum(initial_state.temperature, temperature_floor)
    residual_history: list[np.ndarray] = []
    advanced_history: list[np.ndarray] = []
    history_limit = 5
    max_residual = 0.0

    for _ in range(FIXED_POINT_MAX_ITERS):
        advanced = apply_annual_map(
            state,
            diagnostic_models,
            monthly_insolation,
            month_durations,
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
            solver_cache=solver_cache,
        )

        residual = advanced.temperature - state.temperature
        max_residual = float(np.max(np.abs(residual)))

        if max_residual < PERIODIC_FIXED_POINT_TOLERANCE_K:
            return advanced

        residual_flat = residual.ravel()
        advanced_flat = advanced.temperature.ravel()

        if len(residual_history) == history_limit:
            residual_history.pop(0)
            advanced_history.pop(0)

        residual_history.append(residual_flat)
        advanced_history.append(advanced_flat)

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

        state.temperature = np.maximum(T_next, temperature_floor)

    raise RuntimeError(
        "Failed to converge to a periodic solution after "
        f"{FIXED_POINT_MAX_ITERS} iterations (last residual {max_residual:.3e} K)"
    )


def integrate_periodic_cycle(
    state: ModelState,
    diagnostic_models: DiagnosticModels,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
) -> list[ModelState]:
    """Return the 12-month sequence of month-midpoint temperatures for the periodic solution."""
    states = []
    for month in range(12):
        next_state = monthly_step(
            state,
            diagnostic_models,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
            solver_cache=solver_cache,
        )
        state = next_state
        states.append(next_state)
    return states


def solve_periodic_cycle_for_albedo(
    *,
    initial_state: ModelState | None,
    current_albedo_field: np.ndarray,
    diagnostic_models: DiagnosticModels,
    heat_capacity_field: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    diffusion_operator: LayeredDiffusionOperator,
    radiation_config: RadiationConfig,
    advection_model: AdvectionModel | None,
    rhs_factory: RhsFactory,
    initial_guess_fn: InitialGuessFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache,
) -> tuple[ModelState, list[ModelState]]:
    """Compute the periodic solution and monthly mean temperatures for a given albedo field."""

    if initial_state is None:
        guess = initial_guess_fn(
            monthly_insolation,
            current_albedo_field,
            radiation_config,
        )

        initial_wind = None
        if advection_model is not None:
            initial_wind, _, _ = advection_model.wind_field(guess[0])

        diagnostics = DiagnosticParameters(
            albedo_field=current_albedo_field,
            wind_field=initial_wind,
        )

        initial_state = ModelState(
            temperature=guess,
            diagnostics=diagnostics,
        )
    else:
        initial_state = ModelState(
            temperature=initial_state.temperature,
            diagnostics=DiagnosticParameters(
                albedo_field=current_albedo_field,
                wind_field=initial_state.diagnostics.wind_field,
            ),
        )

    rhs_fn, rhs_temperature_derivative_fn = rhs_factory(
        heat_capacity_field,
        diffusion_operator,
        radiation_config,
    )

    periodic_state = find_periodic_temperature(
        initial_state,
        diagnostic_models,
        monthly_insolation,
        month_durations,
        rhs_fn=rhs_fn,
        rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
        temperature_floor=temperature_floor,
        solver_cache=solver_cache,
    )

    monthly_states = integrate_periodic_cycle(
        periodic_state,
        diagnostic_models,
        monthly_insolation,
        month_durations,
        rhs_fn=rhs_fn,
        rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
        temperature_floor=temperature_floor,
        solver_cache=solver_cache,
    )

    return periodic_state, monthly_states


def compute_periodic_cycle_kelvin(
    *,
    resolution_deg: float,
    monthly_insolation_lat_fn: MonthlyInsolationLatFunc,
    heat_capacity_field_fn: HeatCapacityFieldFunc,
    rhs_factory: RhsFactory,
    initial_guess_fn: InitialGuessFunc,
    radiation_config: RadiationConfig,
    diffusion_config: DiffusionConfig,
    advection_config: AdvectionConfig | None = None,
    snow_config: SnowAlbedoConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ModelState]]:
    """Solve for the periodic temperature cycle and return results in Kelvin with the converged albedo field."""
    lon2d, lat2d = create_lat_lon_grid(resolution_deg)

    monthly_insolation_lat = monthly_insolation_lat_fn(lat2d)
    monthly_insolation = expand_latitude_field(monthly_insolation_lat, lon2d.shape[1])

    heat_capacity_field = heat_capacity_field_fn(lon2d, lat2d)
    snow_cfg = snow_config or SnowAlbedoConfig()
    albedo_kwargs: dict[str, float] = {}
    land_mask = compute_land_mask(lon2d, lat2d)

    albedo_model = AlbedoModel(
        lat2d,
        lon2d,
        config=snow_cfg,
        land_mask=land_mask,
    )
    if snow_cfg.enabled:
        albedo_kwargs = {"land_albedo": 0.25, "ocean_albedo": 0.25}

    base_albedo_field = compute_albedo_field(lon2d, lat2d, **albedo_kwargs)
    current_albedo_field = base_albedo_field

    diffusion_operator = create_diffusion_operator(
        lon2d,
        lat2d,
        heat_capacity_field,
        land_mask=land_mask,
        atmosphere_heat_capacity=radiation_config.atmosphere_heat_capacity,
        config=diffusion_config,
    )
    advection_model: AdvectionModel | None = None
    if (
        advection_config is not None
        and advection_config.enabled
        and radiation_config.include_atmosphere
    ):
        advection_model = AdvectionModel(
            lon2d,
            lat2d,
            config=advection_config,
        )
    month_durations = DAYS_PER_MONTH * SECONDS_PER_DAY
    solver_cache = LinearSolveCache()

    diagnostic_models = DiagnosticModels(
        snow_albedo_model=albedo_model,
        advection_model=advection_model,
    )

    previous_periodic: ModelState | None = None
    monthly: list[ModelState] | None = None

    iterations = snow_cfg.picard_iterations if snow_cfg.enabled else 1

    for iteration in range(iterations):
        periodic_state, monthly = solve_periodic_cycle_for_albedo(
            initial_state=previous_periodic,
            current_albedo_field=current_albedo_field,
            diagnostic_models=diagnostic_models,
            heat_capacity_field=heat_capacity_field,
            monthly_insolation=monthly_insolation,
            month_durations=month_durations,
            diffusion_operator=diffusion_operator,
            radiation_config=radiation_config,
            advection_model=advection_model,
            rhs_factory=rhs_factory,
            initial_guess_fn=initial_guess_fn,
            temperature_floor=radiation_config.temperature_floor,
            solver_cache=solver_cache,
        )

        if not snow_cfg.enabled:
            previous_periodic = periodic_state
            break

        if iteration == iterations - 1:
            previous_periodic = periodic_state
            break

        monthly_temperatures = np.array([state.temperature for state in monthly])
        if monthly_temperatures.ndim == 4:
            surface_temperatures = monthly_temperatures[:, 0]
        else:
            surface_temperatures = monthly_temperatures

        updated_albedo = albedo_model.apply_snow_albedo(
            base_albedo_field,
            surface_temperatures,
        )

        if np.array_equal(updated_albedo, current_albedo_field):
            previous_periodic = periodic_state
            break

        current_albedo_field = updated_albedo
        periodic_state = ModelState(
            temperature=periodic_state.temperature,
            diagnostics=DiagnosticParameters(
                albedo_field=current_albedo_field,
                wind_field=periodic_state.diagnostics.wind_field,
            ),
        )
        previous_periodic = periodic_state

    assert monthly is not None

    return lon2d, lat2d, monthly


def compute_periodic_cycle_results(
    resolution_deg: float = 1.0,
    *,
    solar_constant: float | None = None,
    ocean_heat_capacity: float | None = None,
    land_heat_capacity: float | None = None,
    radiation_config: RadiationConfig | None = None,
    diffusion_config: DiffusionConfig | None = None,
    advection_config: AdvectionConfig | None = None,
    snow_config: SnowAlbedoConfig | None = None,
    return_layer_map: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Produce the periodic cycle fields in Celsius along with the converged albedo layer.

    Returns the lon/lat grids and either the surface temperature field (default) or a
    dictionary of layer outputs when ``return_layer_map`` is True.
    """

    def monthly_insolation_lat_fn(lat2d: np.ndarray) -> np.ndarray:
        return compute_monthly_insolation_field(lat2d, solar_constant=solar_constant)

    def heat_capacity_field_fn(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
        return compute_heat_capacity_field(
            lon2d,
            lat2d,
            ocean_heat_capacity=ocean_heat_capacity,
            land_heat_capacity=land_heat_capacity,
        )

    resolved_radiation = radiation_config or RadiationConfig()
    resolved_diffusion = diffusion_config or DiffusionConfig()
    resolved_advection = advection_config or AdvectionConfig()
    resolved_snow = snow_config or SnowAlbedoConfig()

    def rhs_factory(
        heat_capacity_field: np.ndarray,
        diffusion_operator: LayeredDiffusionOperator,
        config: RadiationConfig,
    ):
        surface_matrix = None
        if diffusion_operator.surface.enabled:
            surface_matrix = diffusion_operator.surface.matrix
            if not sparse.isspmatrix_csc(surface_matrix):
                surface_matrix = surface_matrix.tocsc()

        atmosphere_matrix = None
        if config.include_atmosphere and diffusion_operator.atmosphere.enabled:
            atmosphere_matrix = diffusion_operator.atmosphere.matrix
            if not sparse.isspmatrix_csc(atmosphere_matrix):
                atmosphere_matrix = atmosphere_matrix.tocsc()
        # The diffusion matrices are already expressed as d(rhs)/dT in 1/s because the
        # discrete operator divides by the local heat capacity when it is assembled.

        def rhs(state: ModelState, insolation: np.ndarray) -> np.ndarray:
            radiative = radiation.radiative_balance_rhs(
                state.temperature,
                insolation,
                heat_capacity_field=heat_capacity_field,
                albedo_field=state.diagnostics.albedo_field,
                config=config,
            )
            if config.include_atmosphere:
                radiative = radiative.copy()
                if diffusion_operator.surface.enabled:
                    radiative[0] += diffusion_operator.surface.tendency(state.temperature[0])
                if diffusion_operator.atmosphere.enabled:
                    radiative[1] += diffusion_operator.atmosphere.tendency(state.temperature[1])
                return radiative
            if diffusion_operator.surface.enabled:
                radiative = radiative + diffusion_operator.surface.tendency(state.temperature)
            return radiative

        def rhs_derivative(state: ModelState, insolation: np.ndarray) -> RhsDerivative:
            del insolation
            radiative_derivative = radiation.radiative_balance_rhs_temperature_derivative(
                state.temperature,
                heat_capacity_field=heat_capacity_field,
                config=config,
            )
            if config.include_atmosphere:
                diag, cross = radiative_derivative
                return Linearisation(
                    diag=diag,
                    cross=cross,
                    surface_diffusion_matrix=surface_matrix,
                    atmosphere_diffusion_matrix=atmosphere_matrix,
                )
            return Linearisation(
                diag=radiative_derivative,
                surface_diffusion_matrix=surface_matrix,
            )

        return rhs, rhs_derivative

    def initial_guess_fn(
        monthly_insolation: np.ndarray,
        albedo_field: np.ndarray,
        config: RadiationConfig,
    ) -> np.ndarray:
        return radiation.radiative_equilibrium_initial_guess(
            monthly_insolation,
            albedo_field=albedo_field,
            config=config,
        )

    lon2d, lat2d, monthly_states = compute_periodic_cycle_kelvin(
        resolution_deg=resolution_deg,
        monthly_insolation_lat_fn=monthly_insolation_lat_fn,
        heat_capacity_field_fn=heat_capacity_field_fn,
        rhs_factory=rhs_factory,
        initial_guess_fn=initial_guess_fn,
        radiation_config=resolved_radiation,
        diffusion_config=resolved_diffusion,
        advection_config=resolved_advection,
        snow_config=resolved_snow,
    )

    monthly_T = np.array([state.temperature for state in monthly_states])
    if monthly_T.ndim == 3:
        monthly_surface_K = monthly_T
        layers_map = {"surface": monthly_surface_K - 273.15}
    else:
        monthly_surface_K = monthly_T[:, 0]
        monthly_atmosphere_K = monthly_T[:, 1]
        layers_map = {
            "surface": monthly_surface_K - 273.15,
            "atmosphere": monthly_atmosphere_K - 273.15,
        }

    layers_map["albedo"] = np.array([state.diagnostics.albedo_field for state in monthly_states])

    if return_layer_map:
        return lon2d, lat2d, layers_map

    return lon2d, lat2d, layers_map["surface"]
