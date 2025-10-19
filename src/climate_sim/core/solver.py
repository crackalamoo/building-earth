"""General-purpose periodic solver utilities for energy-balance models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import climate_sim.physics.radiation as radiation
from climate_sim.physics.advection import (
    AdvectionConfig,
    AdvectionModel,
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
from climate_sim.physics.atmosphere import adjust_temperature_by_elevation
from climate_sim.data.calendar import DAYS_PER_MONTH, SECONDS_PER_DAY
from climate_sim.core.grid import create_lat_lon_grid, expand_latitude_field
from climate_sim.physics.solar import compute_monthly_insolation_field
from climate_sim.data.landmask import (
    compute_albedo_field,
    compute_heat_capacity_field,
    compute_land_mask,
)
from climate_sim.data.elevation import (
    compute_cell_elevation,
    compute_cell_roughness_length,
)

NEWTON_STEP_TOLERANCE_K = 1.0
PERIODIC_FIXED_POINT_TOLERANCE_K = 1.0
NEWTON_MAX_ITERS = 16
NEWTON_BACKTRACK_REDUCTION = 0.5
NEWTON_BACKTRACK_CUTOFF = 1e-3
FIXED_POINT_MAX_ITERS = 100
ATMOSPHERE_REFERENCE_HEIGHT_M = 5000.0


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


@dataclass(frozen=True)
class SurfaceHeatCapacityContext:
    """Container for surface-layer heat-capacity metadata."""

    albedo_model: AlbedoModel
    land_mask: np.ndarray
    base_C_land: float
    base_C_ocean: float
    baseline_capacity: np.ndarray


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
class ModelState:
    """State variables for the climate model."""

    temperature: np.ndarray
    albedo_field: np.ndarray
    atmosphere_albedo_field: np.ndarray | None = None
    wind_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _select_wind_temperature(temperature: np.ndarray) -> np.ndarray:
    """Return the temperature field to use when computing wind diagnostics."""
    if temperature.ndim == 2:
        return temperature
    if temperature.ndim == 3:
        if temperature.shape[0] == 1:
            return temperature[0]
        if temperature.shape[0] >= 2:
            return temperature[1]
    raise ValueError("Unsupported temperature field shape for wind calculation")


def _winds_equal(
    winds_a: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None,
    winds_b: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None,
) -> bool:
    if winds_a is winds_b:
        return True
    if winds_a is None or winds_b is None:
        return winds_a is winds_b
    if len(winds_a) != len(winds_b):
        return False
    for wa, wb in zip(winds_a, winds_b, strict=True):
        if wa is None and wb is None:
            continue
        if (wa is None) != (wb is None):
            return False
        assert wa is not None and wb is not None
        if not (
            np.array_equal(wa[0], wb[0])
            and np.array_equal(wa[1], wb[1])
            and np.array_equal(wa[2], wb[2])
        ):
            return False
    return True

RhsFunc = Callable[[ModelState, np.ndarray], np.ndarray]
RhsDerivative = Linearisation
RhsDerivativeFunc = Callable[[ModelState, np.ndarray], RhsDerivative]
RhsFactory = Callable[
    [
        np.ndarray,
        LayeredDiffusionOperator,
        RadiationConfig,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        SensibleHeatExchangeConfig,
    ],
    Tuple[RhsFunc, RhsDerivativeFunc],
]
InitialGuessFunc = Callable[
    [np.ndarray, np.ndarray, RadiationConfig, np.ndarray | None],
    np.ndarray,
]
MonthlyInsolationLatFunc = Callable[[np.ndarray], np.ndarray]
HeatCapacityFieldFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


def monthly_step(
    state: ModelState,
    wind_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    insolation_W_m2: np.ndarray,
    dt_seconds: float,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
    surface_context: SurfaceHeatCapacityContext | None = None,
) -> ModelState:
    """Advance the column temperature one implicit backward-Euler step."""
    start_temp = state.temperature
    temp_next = np.maximum(start_temp, temperature_floor)
    cache = solver_cache or DEFAULT_LINEAR_SOLVE_CACHE

    baseline_capacity_field = (
        surface_context.baseline_capacity if surface_context is not None else None
    )

    def _effective_surface_capacity(temp_surface: np.ndarray) -> np.ndarray:
        if surface_context is None:
            return np.ones_like(temp_surface, dtype=float)
        return surface_context.albedo_model.effective_heat_capacity_surface(
            temp_surface,
            land_mask=surface_context.land_mask,
            base_C_land=surface_context.base_C_land,
            base_C_ocean=surface_context.base_C_ocean,
        )

    def _baseline_surface_capacity(temp_surface: np.ndarray) -> np.ndarray:
        if surface_context is None:
            return np.ones_like(temp_surface, dtype=float)
        return baseline_capacity_field

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
            albedo_field=state.albedo_field,
            atmosphere_albedo_field=state.atmosphere_albedo_field,
            wind_field=wind_field,
        )
        rhs_value = rhs_fn(state_capped, insolation_W_m2)
        linearisation = rhs_temperature_derivative_fn(state_capped, insolation_W_m2)

        if temp_capped.ndim == 2:
            base_capacity = _baseline_surface_capacity(temp_capped)
            ceff = _effective_surface_capacity(temp_capped)
            diag = linearisation.diag
            diffusion_matrix = linearisation.surface_diffusion_matrix

            flux_term = base_capacity * rhs_value
            residual = ceff * (temp_capped - start_temp) - dt_seconds * flux_term
            size = temp_capped.size
            residual_flat = residual.ravel()

            jacobian = sparse.diags(ceff.ravel(), format="csc")
            jacobian -= dt_seconds * sparse.diags(
                (base_capacity * diag).ravel(), format="csc"
            )

            if diffusion_matrix is not None:
                if surface_context is None:
                    jacobian = jacobian - dt_seconds * diffusion_matrix
                else:
                    capacity_diag = sparse.diags(
                        base_capacity.ravel(), format="csc"
                    )
                    jacobian = jacobian - dt_seconds * capacity_diag @ diffusion_matrix

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

            base_capacity_surface = _baseline_surface_capacity(temp_capped[0])
            ceff_surface = _effective_surface_capacity(temp_capped[0])
            flux_surface = base_capacity_surface * rhs_value[0]
            residual_surface = ceff_surface * (temp_capped[0] - start_temp[0]) - dt_seconds * (
                flux_surface
            )
            residual_atmosphere = temp_capped[1] - start_temp[1] - dt_seconds * rhs_value[1]
            residual = np.stack([residual_surface, residual_atmosphere])

            nlat, nlon = surface_diag.shape
            size = nlat * nlon

            identity = identity_single_layer
            if identity is None or identity.shape[0] != size or identity.shape[1] != size:
                identity = _get_identity_matrix(size, cache=cache)
            surface_block = sparse.diags(ceff_surface.ravel(), format="csc")
            surface_block -= dt_seconds * sparse.diags(
                (base_capacity_surface * surface_diag).ravel(), format="csc"
            )
            atmosphere_block = identity.copy()
            atmosphere_block -= dt_seconds * sparse.diags(atmosphere_diag.ravel(), format="csc")

            if surface_matrix is not None:
                if surface_context is None:
                    surface_block = surface_block - dt_seconds * surface_matrix
                else:
                    capacity_diag = sparse.diags(
                        base_capacity_surface.ravel(), format="csc"
                    )
                    surface_block = surface_block - dt_seconds * capacity_diag @ surface_matrix
            if atmosphere_matrix is not None:
                atmosphere_block = atmosphere_block - dt_seconds * atmosphere_matrix

            coupling_surface_atm = -dt_seconds * sparse.diags(
                (base_capacity_surface * cross[0, 1]).ravel(), format="csc"
            )
            coupling_atm_surface = -dt_seconds * sparse.diags(
                cross[1, 0].ravel(), format="csc"
            )

            jacobian = sparse.bmat(
                [[surface_block, coupling_surface_atm], [coupling_atm_surface, atmosphere_block]],
                format="csc",
            )

            residual_flat = np.concatenate(
                [residual_surface.ravel(), residual_atmosphere.ravel()],
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
                albedo_field=state.albedo_field,
                atmosphere_albedo_field=state.atmosphere_albedo_field,
                wind_field=wind_field,
            )
            rhs_candidate = rhs_fn(state_candidate, insolation_W_m2)
            if temp_candidate.ndim == 2:
                base_capacity_candidate = _baseline_surface_capacity(temp_candidate)
                ceff_candidate = _effective_surface_capacity(temp_candidate)
                residual_candidate = ceff_candidate * (
                    temp_candidate - start_temp
                ) - dt_seconds * (base_capacity_candidate * rhs_candidate)
            else:
                if temp_candidate.ndim < 1 or temp_candidate.shape[0] != 2:
                    raise ValueError("Layered derivative requires a two-layer temperature field")
                base_capacity_candidate = _baseline_surface_capacity(temp_candidate[0])
                ceff_candidate = _effective_surface_capacity(temp_candidate[0])
                residual_surface_candidate = ceff_candidate * (
                    temp_candidate[0] - start_temp[0]
                ) - dt_seconds * (base_capacity_candidate * rhs_candidate[0])
                residual_atmosphere_candidate = (
                    temp_candidate[1] - start_temp[1] - dt_seconds * rhs_candidate[1]
                )
                residual_candidate = np.stack(
                    [residual_surface_candidate, residual_atmosphere_candidate]
                )
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
        albedo_field=state.albedo_field,
        atmosphere_albedo_field=state.atmosphere_albedo_field,
        wind_field=wind_field,
    )


def apply_annual_map(
    state: ModelState,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    wind_fields: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
    surface_context: SurfaceHeatCapacityContext | None = None,
) -> ModelState:
    """Propagate the state through 12 implicit steps and return the end-of-December temperature."""
    for month in range(12):
        state = monthly_step(
            state,
            wind_fields[month] if wind_fields is not None else None,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
            solver_cache=solver_cache,
            surface_context=surface_context,
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
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    wind_fields: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
    surface_context: SurfaceHeatCapacityContext | None = None,
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
            monthly_insolation,
            month_durations,
            wind_fields,
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
            solver_cache=solver_cache,
            surface_context=surface_context,
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
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    wind_fields: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache | None = None,
    surface_context: SurfaceHeatCapacityContext | None = None,
) -> list[ModelState]:
    """Return the 12-month sequence of month-midpoint temperatures for the periodic solution."""
    states = []
    for month in range(12):
        next_state = monthly_step(
            state,
            wind_fields[month] if wind_fields is not None else None,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
            solver_cache=solver_cache,
            surface_context=surface_context,
        )
        state = next_state
        states.append(next_state)
    return states


def solve_periodic_cycle_for_albedo(
    *,
    initial_state: ModelState | None,
    current_albedo_field: np.ndarray,
    atmosphere_albedo_field: np.ndarray | None,
    heat_capacity_field: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    wind_fields: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray] | None] | None,
    diffusion_operator: LayeredDiffusionOperator,
    radiation_config: RadiationConfig,
    advection_model: AdvectionModel | None,
    rhs_factory: RhsFactory,
    initial_guess_fn: InitialGuessFunc,
    temperature_floor: float,
    solver_cache: LinearSolveCache,
    land_mask: np.ndarray,
    roughness_length: np.ndarray,
    topographic_elevation: np.ndarray,
    sensible_heat_cfg: SensibleHeatExchangeConfig,
    albedo_model: AlbedoModel,
) -> tuple[ModelState, list[ModelState]]:
    """Compute the periodic solution and monthly mean temperatures for a given albedo field."""

    if initial_state is None:
        guess = initial_guess_fn(
            monthly_insolation,
            current_albedo_field,
            radiation_config,
            atmosphere_albedo_field,
        )

        state_init = ModelState(
            temperature=guess,
            albedo_field=current_albedo_field,
            atmosphere_albedo_field=atmosphere_albedo_field,
            wind_field=wind_fields[0] if wind_fields is not None else None,
        )
    else:
        state_init = ModelState(
            temperature=initial_state.temperature,
            albedo_field=current_albedo_field,
            atmosphere_albedo_field=atmosphere_albedo_field,
            wind_field=wind_fields[0] if wind_fields is not None else initial_state.wind_field,
        )

    rhs_fn, rhs_temperature_derivative_fn = rhs_factory(
        heat_capacity_field,
        diffusion_operator,
        radiation_config,
        land_mask,
        roughness_length,
        topographic_elevation,
        sensible_heat_cfg,
    )

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
        albedo_model=albedo_model,
        land_mask=land_mask,
        base_C_land=base_C_land,
        base_C_ocean=base_C_ocean,
        baseline_capacity=baseline_capacity,
    )

    periodic_state = find_periodic_temperature(
        state_init,
        monthly_insolation,
        month_durations,
        wind_fields,
        rhs_fn=rhs_fn,
        rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
        temperature_floor=temperature_floor,
        solver_cache=solver_cache,
        surface_context=surface_context,
    )

    monthly_states = integrate_periodic_cycle(
        periodic_state,
        monthly_insolation,
        month_durations,
        wind_fields,
        rhs_fn=rhs_fn,
        rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
        temperature_floor=temperature_floor,
        solver_cache=solver_cache,
        surface_context=surface_context,
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
    sensible_heat_config: SensibleHeatExchangeConfig | None = None,
    clouds: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ModelState]]:
    """Solve for the periodic temperature cycle and return diagnostics.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[ModelState]]
        Longitude grid, latitude grid, topographic elevation (m), and the
        monthly model states expressed in Kelvin.
    """
    lon2d, lat2d = create_lat_lon_grid(resolution_deg)

    monthly_insolation_lat = monthly_insolation_lat_fn(lat2d)
    monthly_insolation = expand_latitude_field(monthly_insolation_lat, lon2d.shape[1])

    heat_capacity_field = heat_capacity_field_fn(lon2d, lat2d)
    snow_cfg = snow_config or SnowAlbedoConfig()
    sensible_heat_cfg = sensible_heat_config or SensibleHeatExchangeConfig()
    land_mask = compute_land_mask(lon2d, lat2d)

    albedo_model = AlbedoModel(
        lat2d,
        lon2d,
        config=snow_cfg,
        land_mask=land_mask,
    )
    if clouds:
        albedo_kwargs: dict[str, float] = {"land_albedo": 0.18, "ocean_albedo": 0.06}
    elif snow_cfg.enabled:
        albedo_kwargs = {"land_albedo": 0.25, "ocean_albedo": 0.25}
    else:
        albedo_kwargs = {}

    base_albedo_field = compute_albedo_field(lon2d, lat2d, **albedo_kwargs)
    current_albedo_field = base_albedo_field

    atmosphere_albedo_field: np.ndarray | None
    if clouds:
        atmosphere_albedo_field = np.where(land_mask, 0.18, 0.22)
    else:
        atmosphere_albedo_field = None

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

    previous_periodic: ModelState | None = None
    monthly: list[ModelState] | None = None

    if advection_model is not None and advection_model.enabled:
        current_wind_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * 12
        wind_iterations = 2
    else:
        current_wind_fields = [None] * 12
        wind_iterations = 1

    snow_iterations = snow_cfg.picard_iterations if snow_cfg.enabled else 1
    iterations = max(snow_iterations, wind_iterations)

    for iteration in range(iterations):
        periodic_state, monthly = solve_periodic_cycle_for_albedo(
            initial_state=previous_periodic,
            current_albedo_field=current_albedo_field,
            atmosphere_albedo_field=atmosphere_albedo_field,
            heat_capacity_field=heat_capacity_field,
            monthly_insolation=monthly_insolation,
            month_durations=month_durations,
            wind_fields=current_wind_fields,
            diffusion_operator=diffusion_operator,
            radiation_config=radiation_config,
            advection_model=advection_model,
            rhs_factory=rhs_factory,
            initial_guess_fn=initial_guess_fn,
            temperature_floor=radiation_config.temperature_floor,
            solver_cache=solver_cache,
            land_mask=land_mask,
            roughness_length=roughness_length,
            topographic_elevation=topographic_elevation,
            sensible_heat_cfg=sensible_heat_cfg,
            albedo_model=albedo_model,
        )

        monthly_temperatures = np.array([state.temperature for state in monthly])
        if monthly_temperatures.ndim == 4:
            surface_temperatures = monthly_temperatures[:, 0]
        else:
            surface_temperatures = monthly_temperatures

        updated_albedo = current_albedo_field
        snow_converged = True
        if snow_cfg.enabled:
            updated_albedo = albedo_model.apply_snow_albedo(
                base_albedo_field,
                surface_temperatures,
            )
            snow_converged = np.array_equal(updated_albedo, current_albedo_field)
            current_albedo_field = updated_albedo

        if advection_model is not None and advection_model.enabled:
            new_wind_fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for idx, month_state in enumerate(monthly):
                wind_temperature = _select_wind_temperature(month_state.temperature)
                wind_components, _, _ = advection_model.wind_field(wind_temperature)
                new_wind_fields.append(wind_components)
                monthly[idx] = ModelState(
                    temperature=month_state.temperature,
                    albedo_field=current_albedo_field,
                    atmosphere_albedo_field=month_state.atmosphere_albedo_field,
                    wind_field=wind_components,
                )
            wind_converged = _winds_equal(new_wind_fields, current_wind_fields)
            current_wind_fields = new_wind_fields
            periodic_state = ModelState(
                temperature=periodic_state.temperature,
                albedo_field=current_albedo_field,
                atmosphere_albedo_field=periodic_state.atmosphere_albedo_field,
                wind_field=new_wind_fields[-1],
            )
        else:
            wind_converged = True
            current_wind_fields = [None] * 12
            for idx, month_state in enumerate(monthly):
                monthly[idx] = ModelState(
                    temperature=month_state.temperature,
                    albedo_field=current_albedo_field,
                    atmosphere_albedo_field=month_state.atmosphere_albedo_field,
                    wind_field=None,
                )
            periodic_state = ModelState(
                temperature=periodic_state.temperature,
                albedo_field=current_albedo_field,
                atmosphere_albedo_field=periodic_state.atmosphere_albedo_field,
                wind_field=None,
            )

        previous_periodic = periodic_state

        if snow_converged and wind_converged:
            break

    assert monthly is not None

    return lon2d, lat2d, topographic_elevation, monthly


def compute_periodic_cycle_results(
    resolution_deg: float = 1.0,
    *,
    solar_constant: float | None = None,
    use_elliptical_orbit: bool = True,
    clouds: bool = True,
    ocean_heat_capacity: float | None = None,
    land_heat_capacity: float | None = None,
    radiation_config: RadiationConfig | None = None,
    diffusion_config: DiffusionConfig | None = None,
    advection_config: AdvectionConfig | None = None,
    snow_config: SnowAlbedoConfig | None = None,
    sensible_heat_config: SensibleHeatExchangeConfig | None = None,
    return_layer_map: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Produce the periodic cycle fields in Celsius along with the converged albedo layer.

    Returns the lon/lat grids and either the surface temperature field (default) or a
    dictionary of layer outputs when ``return_layer_map`` is True.
    """

    def monthly_insolation_lat_fn(lat2d: np.ndarray) -> np.ndarray:
        return compute_monthly_insolation_field(
            lat2d,
            solar_constant=solar_constant,
            use_elliptical_orbit=use_elliptical_orbit,
        )

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
    sensible_heat_cfg = sensible_heat_config or SensibleHeatExchangeConfig()

    if not sensible_heat_cfg.enabled and resolved_advection.enabled:
        resolved_advection = replace(resolved_advection, enabled=False)

    def rhs_factory(
        heat_capacity_field: np.ndarray,
        diffusion_operator: LayeredDiffusionOperator,
        config: RadiationConfig,
        land_mask: np.ndarray,
        roughness_length: np.ndarray,
        topographic_elevation: np.ndarray,
        sensible_heat_cfg_local: SensibleHeatExchangeConfig,
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

        sensible_heat_model: SensibleHeatExchangeModel | None = None
        if config.include_atmosphere and sensible_heat_cfg_local.enabled:
            sensible_heat_model = SensibleHeatExchangeModel(
                land_mask=land_mask,
                roughness_length_m=roughness_length,
                surface_heat_capacity_J_m2_K=heat_capacity_field,
                atmosphere_heat_capacity_J_m2_K=config.atmosphere_heat_capacity,
                topographic_elevation_m=topographic_elevation,
                config=sensible_heat_cfg_local,
            )

        def rhs(state: ModelState, insolation: np.ndarray) -> np.ndarray:
            radiative = radiation.radiative_balance_rhs(
                state.temperature,
                insolation,
                heat_capacity_field=heat_capacity_field,
                albedo_field=state.albedo_field,
                atmosphere_albedo_field=state.atmosphere_albedo_field,
                config=config,
            )
            if config.include_atmosphere:
                radiative = radiative.copy()
                if diffusion_operator.surface.enabled:
                    radiative[0] += diffusion_operator.surface.tendency(state.temperature[0])
                if diffusion_operator.atmosphere.enabled:
                    radiative[1] += diffusion_operator.atmosphere.tendency(state.temperature[1])

                if sensible_heat_model is not None:
                    surface_temperature = state.temperature[0]
                    atmosphere_temperature = state.temperature[1]
                    wind_speed_ref = state.wind_field[2] if state.wind_field is not None else None
                    (
                        surface_tendency,
                        atmosphere_tendency,
                    ) = sensible_heat_model.compute_tendencies(
                        surface_temperature_K=surface_temperature,
                        atmosphere_temperature_K=atmosphere_temperature,
                        wind_speed_reference_m_s=wind_speed_ref,
                    )

                    radiative[0] += surface_tendency
                    radiative[1] += atmosphere_tendency

                return radiative
            if diffusion_operator.surface.enabled:
                radiative = radiative + diffusion_operator.surface.tendency(state.temperature)
            return radiative

        def rhs_derivative(state: ModelState, insolation: np.ndarray) -> RhsDerivative:
            del insolation
            radiative_derivative = radiation.radiative_balance_rhs_temperature_derivative(
                state.temperature,
                heat_capacity_field=heat_capacity_field,
                atmosphere_albedo_field=state.atmosphere_albedo_field,
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
        atmosphere_albedo_field: np.ndarray | None,
    ) -> np.ndarray:
        return radiation.radiative_equilibrium_initial_guess(
            monthly_insolation,
            albedo_field=albedo_field,
            atmosphere_albedo_field=atmosphere_albedo_field,
            config=config,
        )

    lon2d, lat2d, topographic_elevation, monthly_states = compute_periodic_cycle_kelvin(
        resolution_deg=resolution_deg,
        monthly_insolation_lat_fn=monthly_insolation_lat_fn,
        heat_capacity_field_fn=heat_capacity_field_fn,
        rhs_factory=rhs_factory,
        initial_guess_fn=initial_guess_fn,
        radiation_config=resolved_radiation,
        diffusion_config=resolved_diffusion,
        advection_config=resolved_advection,
        snow_config=resolved_snow,
        sensible_heat_config=sensible_heat_cfg,
        clouds=clouds,
    )

    monthly_T = np.array([state.temperature for state in monthly_states])
    temperature_2m_c: np.ndarray | None = None
    if monthly_T.ndim == 3:
        monthly_surface_K = monthly_T
        layers_map = {"surface": monthly_surface_K - 273.15}
    else:
        monthly_surface_K = monthly_T[:, 0]
        monthly_atmosphere_K = monthly_T[:, 1]
        atmosphere_c = monthly_atmosphere_K - 273.15
        delta_to_two_m = 2.0 - ATMOSPHERE_REFERENCE_HEIGHT_M
        if sensible_heat_cfg.include_lapse_rate_elevation:
            delta_to_two_m = delta_to_two_m + topographic_elevation
        temperature_2m_c = adjust_temperature_by_elevation(
            atmosphere_c,
            delta_to_two_m,
        )
        layers_map = {
            "surface": monthly_surface_K - 273.15,
            "atmosphere": atmosphere_c,
        }

    if temperature_2m_c is None:
        temperature_2m_c = layers_map["surface"].copy()

    layers_map["temperature_2m"] = temperature_2m_c

    layers_map["albedo"] = np.array([state.albedo_field for state in monthly_states])

    atmosphere_albedo_fields = [state.atmosphere_albedo_field for state in monthly_states]
    if all(field is not None for field in atmosphere_albedo_fields):
        layers_map["atmosphere_albedo"] = np.stack(atmosphere_albedo_fields, axis=0)

    wind_fields = [state.wind_field for state in monthly_states]
    if all(wind is not None for wind in wind_fields):
        wind_u = np.stack([wind[0] for wind in wind_fields if wind is not None], axis=0)
        wind_v = np.stack([wind[1] for wind in wind_fields if wind is not None], axis=0)
        wind_speed = np.stack([wind[2] for wind in wind_fields if wind is not None], axis=0)
        layers_map["wind_u"] = wind_u
        layers_map["wind_v"] = wind_v
        layers_map["wind_speed"] = wind_speed

    if return_layer_map:
        return lon2d, lat2d, layers_map

    return lon2d, lat2d, layers_map["surface"]
