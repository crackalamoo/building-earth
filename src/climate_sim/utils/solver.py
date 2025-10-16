"""General-purpose periodic solver utilities for energy-balance models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import climate_sim.modeling.radiation as radiation
from climate_sim.modeling.advection import (
    GeostrophicAdvectionConfig,
    GeostrophicAdvectionOperator,
)
from climate_sim.modeling.diffusion import (
    DiffusionConfig,
    LayeredDiffusionOperator,
    create_diffusion_operator,
)
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig, apply_snow_albedo
from climate_sim.utils.grid import create_lat_lon_grid, expand_latitude_field
from climate_sim.utils.solar import DAYS_PER_MONTH, SECONDS_PER_DAY, compute_monthly_insolation_field
from climate_sim.utils.landmask import (
    compute_albedo_field,
    compute_heat_capacity_field,
    compute_land_mask,
)

NEWTON_TOLERANCE = 1  # K
NEWTON_MAX_ITERS = 16
NEWTON_BACKTRACK_REDUCTION = 0.5
NEWTON_BACKTRACK_CUTOFF = 1e-3
FIXED_POINT_MAX_ITERS = 100
ARCTIC_CIRCLE_LATITUDE_DEG = 66.5


WindField = tuple[np.ndarray, np.ndarray, np.ndarray] | None

RhsFunc = Callable[[np.ndarray, np.ndarray, WindField], np.ndarray]
@dataclass
class Linearisation:
    diag: np.ndarray
    cross: np.ndarray | None = None
    surface_diffusion_matrix: sparse.csr_matrix | None = None
    atmosphere_diffusion_matrix: sparse.csr_matrix | None = None


RhsDerivative = Linearisation
RhsDerivativeFunc = Callable[[np.ndarray, np.ndarray], RhsDerivative]
RhsFactory = Callable[
    [
        np.ndarray,
        np.ndarray,
        LayeredDiffusionOperator,
        RadiationConfig,
        np.ndarray,
        GeostrophicAdvectionOperator | None,
    ],
    Tuple[RhsFunc, RhsDerivativeFunc],
]
InitialGuessFunc = Callable[[np.ndarray, np.ndarray, RadiationConfig], np.ndarray]
MonthlyInsolationLatFunc = Callable[[np.ndarray], np.ndarray]
HeatCapacityFieldFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


def implicit_monthly_step(
    temperature_K: np.ndarray,
    insolation_W_m2: np.ndarray,
    dt_seconds: float,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    wind_field: WindField,
    temperature_floor: float,
) -> np.ndarray:
    """Advance the column temperature one implicit backward-Euler step."""
    temp_next = np.maximum(temperature_K, temperature_floor)

    if temperature_K.ndim == 2:
        identity_single_layer = sparse.eye(temperature_K.size, format="csc")
    elif temperature_K.ndim == 3 and temperature_K.shape[0] == 2:
        identity_single_layer = sparse.eye(
            temperature_K.shape[1] * temperature_K.shape[2], format="csc"
        )
    else:
        identity_single_layer = None

    for _ in range(NEWTON_MAX_ITERS):
        temp_capped = np.maximum(temp_next, temperature_floor)
        rhs_value = rhs_fn(temp_capped, insolation_W_m2, wind_field)
        residual = temp_capped - temperature_K - dt_seconds * rhs_value
        linearisation = rhs_temperature_derivative_fn(temp_capped, insolation_W_m2)

        if temp_capped.ndim == 2:
            diag = linearisation.diag
            diffusion_matrix = linearisation.surface_diffusion_matrix

            size = temp_capped.size
            residual_flat = residual.ravel()

            if (
                identity_single_layer is None
                or identity_single_layer.shape[0] != size
                or identity_single_layer.shape[1] != size
            ):
                jacobian = sparse.eye(size, format="csc")
            else:
                jacobian = identity_single_layer.copy()
            jacobian -= dt_seconds * sparse.diags(diag.ravel(), format="csc")

            if diffusion_matrix is not None:
                jacobian = jacobian - dt_seconds * diffusion_matrix

            solve_linear = splinalg.factorized(jacobian)
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
            if (
                identity is None
                or identity.shape[0] != size
                or identity.shape[1] != size
            ):
                identity = sparse.eye(size, format="csc")
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

            solve_linear = splinalg.factorized(jacobian)
            correction_flat = solve_linear(residual_flat)
            correction_surface = correction_flat[:size].reshape(surface_diag.shape)
            correction_atmosphere = correction_flat[size:].reshape(atmosphere_diag.shape)
            correction = np.stack([correction_surface, correction_atmosphere])

        damping = 1.0
        max_residual = float(np.max(np.abs(residual)))
        accepted = False
        prev_temp = temp_next

        while damping >= NEWTON_BACKTRACK_CUTOFF:
            temp_candidate = np.maximum(prev_temp - damping * correction, temperature_floor)
            rhs_candidate = rhs_fn(temp_candidate, insolation_W_m2, wind_field)
            residual_candidate = temp_candidate - temperature_K - dt_seconds * rhs_candidate
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
        if np.max(np.abs(step)) < NEWTON_TOLERANCE:
            break

    return temp_next


def apply_annual_map(
    temperature_K: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    wind_factory: Callable[[], WindField] | None,
    wind_update_fn: Callable[[np.ndarray], WindField] | None,
) -> np.ndarray:
    """Propagate the state through 12 implicit steps and return the end-of-December temperature."""
    state = temperature_K
    winds = wind_factory() if wind_factory is not None else None
    for month in range(12):
        state = implicit_monthly_step(
            state,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            wind_field=winds,
            temperature_floor=temperature_floor,
        )
        if wind_update_fn is not None:
            winds = wind_update_fn(state)
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
    initial_temperature: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    initial_wind_factory: Callable[[], WindField] | None,
    wind_update_fn: Callable[[np.ndarray], WindField] | None,
) -> np.ndarray:
    """Solve P(T) = T for the annual map using Anderson acceleration."""

    state = np.maximum(initial_temperature, temperature_floor)
    residual_history: list[np.ndarray] = []
    advanced_history: list[np.ndarray] = []
    history_limit = 5

    for _ in range(FIXED_POINT_MAX_ITERS):
        advanced = apply_annual_map(
            state,
            monthly_insolation,
            month_durations,
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
            wind_factory=initial_wind_factory,
            wind_update_fn=wind_update_fn,
        )

        residual = advanced - state
        max_residual = float(np.max(np.abs(residual)))

        if max_residual < NEWTON_TOLERANCE:
            return advanced

        residual_flat = residual.ravel()
        advanced_flat = advanced.ravel()

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
            state_next = advanced
            residual_history = residual_history[-1:]
            advanced_history = advanced_history[-1:]
        else:
            combined = np.zeros_like(advanced_flat)
            for weight, advanced_state in zip(coefficients, advanced_history, strict=True):
                combined += weight * advanced_state
            state_next = combined.reshape(state.shape)

            if not np.all(np.isfinite(state_next)):
                state_next = advanced
                residual_history = residual_history[-1:]
                advanced_history = advanced_history[-1:]

        state = np.maximum(state_next, temperature_floor)

    raise RuntimeError(
        "Failed to converge to a periodic solution after "
        f"{FIXED_POINT_MAX_ITERS} iterations (last residual {max_residual:.3e} K)"
    )


def integrate_periodic_cycle(
    initial_temperature: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
    initial_wind_factory: Callable[[], WindField] | None,
    wind_update_fn: Callable[[np.ndarray], WindField] | None,
) -> np.ndarray:
    """Return the 12-month sequence of month-midpoint temperatures for the periodic solution."""
    temps = np.empty((12,) + initial_temperature.shape, dtype=float)
    state = initial_temperature
    winds = initial_wind_factory() if initial_wind_factory is not None else None
    for month in range(12):
        next_state = implicit_monthly_step(
            state,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            wind_field=winds,
            temperature_floor=temperature_floor,
        )
        temps[month] = 0.5 * (state + next_state)
        state = next_state
        if wind_update_fn is not None:
            winds = wind_update_fn(state)
    return temps


def compute_periodic_cycle_kelvin(
    *,
    resolution_deg: float,
    monthly_insolation_lat_fn: MonthlyInsolationLatFunc,
    heat_capacity_field_fn: HeatCapacityFieldFunc,
    rhs_factory: RhsFactory,
    initial_guess_fn: InitialGuessFunc,
    radiation_config: RadiationConfig,
    diffusion_config: DiffusionConfig,
    advection_config: GeostrophicAdvectionConfig | None = None,
    snow_config: SnowAlbedoConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve for the periodic temperature cycle and return results in Kelvin with the converged albedo field."""
    lon2d, lat2d = create_lat_lon_grid(resolution_deg)

    monthly_insolation_lat = monthly_insolation_lat_fn(lat2d)
    monthly_insolation = expand_latitude_field(monthly_insolation_lat, lon2d.shape[1])

    heat_capacity_field = heat_capacity_field_fn(lon2d, lat2d)
    snow_cfg = snow_config or SnowAlbedoConfig()
    albedo_kwargs: dict[str, float] = {}
    if snow_cfg.enabled:
        albedo_kwargs = {"land_albedo": 0.25, "ocean_albedo": 0.25}
    base_albedo_field = compute_albedo_field(lon2d, lat2d, **albedo_kwargs)
    land_mask = compute_land_mask(lon2d, lat2d)

    if snow_cfg.enabled:
        arctic_land = (lat2d >= ARCTIC_CIRCLE_LATITUDE_DEG) & land_mask
        if np.any(arctic_land):
            base_albedo_field = np.where(
                arctic_land, snow_cfg.snow_albedo, base_albedo_field
            )
    diffusion_operator = create_diffusion_operator(
        lon2d,
        lat2d,
        heat_capacity_field,
        land_mask=land_mask,
        atmosphere_heat_capacity=radiation_config.atmosphere_heat_capacity,
        config=diffusion_config,
    )
    ocean_mask = ~land_mask
    advection_operator: GeostrophicAdvectionOperator | None = None
    if (
        advection_config is not None
        and advection_config.enabled
        and radiation_config.include_atmosphere
    ):
        advection_operator = GeostrophicAdvectionOperator(
            lon2d,
            lat2d,
            config=advection_config,
        )
    month_durations = DAYS_PER_MONTH * SECONDS_PER_DAY

    def solve_with_albedo(
        albedo_field: np.ndarray,
        initial_temperature: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rhs_fn, rhs_temperature_derivative_fn = rhs_factory(
            heat_capacity_field,
            albedo_field,
            diffusion_operator,
            radiation_config,
            ocean_mask,
            advection_operator,
        )

        wind_factory: Callable[[], WindField] | None = None
        wind_update_fn: Callable[[np.ndarray], WindField] | None = None
        if advection_operator is not None and advection_operator.enabled:
            def make_zero_wind() -> WindField:
                return advection_operator.zero_wind_field()

            def update_wind(temperature: np.ndarray) -> WindField:
                if radiation_config.include_atmosphere:
                    return advection_operator.wind_field(temperature[1])
                return advection_operator.wind_field(temperature)

            wind_factory = make_zero_wind
            wind_update_fn = update_wind

        if initial_temperature is None:
            guess = initial_guess_fn(
                monthly_insolation, albedo_field, radiation_config
            )
        else:
            guess = initial_temperature

        periodic_temperature = find_periodic_temperature(
            guess,
            monthly_insolation,
            month_durations,
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=radiation_config.temperature_floor,
            initial_wind_factory=wind_factory,
            wind_update_fn=wind_update_fn,
        )

        monthly_temperatures = integrate_periodic_cycle(
            periodic_temperature,
            monthly_insolation,
            month_durations,
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=radiation_config.temperature_floor,
            initial_wind_factory=wind_factory,
            wind_update_fn=wind_update_fn,
        )

        return periodic_temperature, monthly_temperatures

    albedo_field = base_albedo_field
    previous_periodic: np.ndarray | None = None
    final_monthly: np.ndarray | None = None

    iterations = snow_cfg.picard_iterations if snow_cfg.enabled else 1

    for iteration in range(iterations):
        previous_periodic, final_monthly = solve_with_albedo(
            albedo_field, previous_periodic
        )

        if not snow_cfg.enabled:
            break

        if iteration == iterations - 1:
            break

        updated_albedo = apply_snow_albedo(
            base_albedo_field,
            land_mask,
            final_monthly,
            config=snow_cfg,
        )

        if np.array_equal(updated_albedo, albedo_field):
            break

        albedo_field = updated_albedo

    assert final_monthly is not None

    return lon2d, lat2d, final_monthly, albedo_field


def compute_periodic_cycle_celsius(
    resolution_deg: float = 1.0,
    *,
    solar_constant: float = None,
    ocean_heat_capacity: float = None,
    land_heat_capacity: float = None,
    radiation_config: RadiationConfig | None = None,
    diffusion_config: DiffusionConfig | None = None,
    advection_config: GeostrophicAdvectionConfig | None = None,
    snow_config: SnowAlbedoConfig | None = None,
    return_layer_map: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Convenience wrapper wiring the radiation component into the solver."""

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
    resolved_advection = advection_config or GeostrophicAdvectionConfig()
    resolved_snow = snow_config or SnowAlbedoConfig()

    def rhs_factory(
        heat_capacity_field: np.ndarray,
        albedo_field: np.ndarray,
        diffusion_operator: LayeredDiffusionOperator,
        config: RadiationConfig,
        ocean_mask: np.ndarray,
        advection_operator: GeostrophicAdvectionOperator | None,
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

        def rhs(
            temperature: np.ndarray, insolation: np.ndarray, wind_field: WindField
        ) -> np.ndarray:
            radiative = radiation.radiative_balance_rhs(
                temperature,
                insolation,
                heat_capacity_field=heat_capacity_field,
                albedo_field=albedo_field,
                config=config,
            )
            if config.include_atmosphere:
                radiative = radiative.copy()
                if diffusion_operator.surface.enabled:
                    radiative[0] += diffusion_operator.surface.tendency(temperature[0])
                if diffusion_operator.atmosphere.enabled:
                    radiative[1] += diffusion_operator.atmosphere.tendency(temperature[1])
                if advection_operator is not None and advection_operator.enabled:
                    radiative[1] += advection_operator.tendency_from_wind(
                        temperature[1], wind_field
                    )
                return radiative
            if diffusion_operator.surface.enabled:
                radiative = radiative + diffusion_operator.surface.tendency(temperature)
            if advection_operator is not None and advection_operator.enabled:
                radiative = radiative + advection_operator.tendency_from_wind(
                    temperature, wind_field
                )
            return radiative

        def rhs_derivative(temperature: np.ndarray, insolation: np.ndarray) -> RhsDerivative:
            del insolation
            radiative_derivative = radiation.radiative_balance_rhs_temperature_derivative(
                temperature,
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

    lon2d, lat2d, monthly_temperatures_K, final_albedo = compute_periodic_cycle_kelvin(
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

    if monthly_temperatures_K.ndim == 3:
        monthly_surface_K = monthly_temperatures_K
        layers_map = {"surface": monthly_surface_K - 273.15}
    else:
        monthly_surface_K = monthly_temperatures_K[:, 0]
        monthly_atmosphere_K = monthly_temperatures_K[:, 1]
        layers_map = {
            "surface": monthly_surface_K - 273.15,
            "atmosphere": monthly_atmosphere_K - 273.15,
        }

    layers_map["albedo"] = final_albedo

    if return_layer_map:
        return lon2d, lat2d, layers_map

    return lon2d, lat2d, layers_map["surface"]
