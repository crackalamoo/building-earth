"""General-purpose periodic solver utilities for energy-balance models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import climate_sim.modeling.radiation as radiation
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig, apply_snow_albedo
from climate_sim.utils.grid import create_lat_lon_grid, expand_latitude_field
from climate_sim.utils.solar import DAYS_PER_MONTH, SECONDS_PER_DAY, compute_monthly_insolation_field
from climate_sim.modeling.diffusion import (
    DiffusionConfig,
    LayeredDiffusionOperator,
    create_diffusion_operator,
)
from climate_sim.utils.landmask import (
    compute_albedo_field,
    compute_heat_capacity_field,
    compute_land_mask,
)

NEWTON_TOLERANCE = 1e-5  # K
NEWTON_MAX_ITERS = 16
NEWTON_DAMPING = 0.5
FIXED_POINT_MAX_ITERS = 300


RhsFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
@dataclass
class Linearisation:
    diag: np.ndarray
    cross: np.ndarray | None = None
    surface_diffusion_matrix: sparse.csr_matrix | None = None
    atmosphere_diffusion_matrix: sparse.csr_matrix | None = None


RhsDerivative = Linearisation
RhsDerivativeFunc = Callable[[np.ndarray, np.ndarray], RhsDerivative]
RhsFactory = Callable[[np.ndarray, np.ndarray, LayeredDiffusionOperator, RadiationConfig], Tuple[RhsFunc, RhsDerivativeFunc]]
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
    temperature_floor: float,
) -> np.ndarray:
    """Advance the column temperature one implicit backward-Euler step."""
    temp_next = np.maximum(temperature_K, temperature_floor)

    for _ in range(NEWTON_MAX_ITERS):
        temp_capped = np.maximum(temp_next, temperature_floor)
        rhs_value = rhs_fn(temp_capped, insolation_W_m2)
        residual = temp_capped - temperature_K - dt_seconds * rhs_value
        linearisation = rhs_temperature_derivative_fn(temp_capped, insolation_W_m2)

        if temp_capped.ndim == 2:
            diag = linearisation.diag
            diffusion_matrix = linearisation.surface_diffusion_matrix

            size = temp_capped.size
            residual_flat = residual.ravel()

            jacobian = sparse.identity(size, format="csr") - dt_seconds * sparse.diags(
                diag.ravel(), format="csr"
            )

            if diffusion_matrix is not None:
                jacobian = jacobian - dt_seconds * diffusion_matrix

            correction_flat = splinalg.spsolve(jacobian, residual_flat)
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

            identity = sparse.identity(size, format="csr")
            surface_block = identity - dt_seconds * sparse.diags(
                surface_diag.ravel(), format="csr"
            )
            atmosphere_block = identity - dt_seconds * sparse.diags(
                atmosphere_diag.ravel(), format="csr"
            )

            if surface_matrix is not None:
                surface_block = surface_block - dt_seconds * surface_matrix
            if atmosphere_matrix is not None:
                atmosphere_block = atmosphere_block - dt_seconds * atmosphere_matrix

            coupling_surface_atm = -dt_seconds * sparse.diags(
                cross[0, 1].ravel(), format="csr"
            )
            coupling_atm_surface = -dt_seconds * sparse.diags(
                cross[1, 0].ravel(), format="csr"
            )

            jacobian = sparse.bmat(
                [[surface_block, coupling_surface_atm], [coupling_atm_surface, atmosphere_block]],
                format="csr",
            )

            residual_flat = np.concatenate(
                [residual[0].ravel(), residual[1].ravel()],
                axis=0,
            )

            correction_flat = splinalg.spsolve(jacobian, residual_flat)
            correction_surface = correction_flat[:size].reshape(surface_diag.shape)
            correction_atmosphere = correction_flat[size:].reshape(atmosphere_diag.shape)
            correction = np.stack([correction_surface, correction_atmosphere])

        temp_candidate = temp_next - correction
        temp_next = np.maximum(temp_candidate, temperature_floor)

        if np.max(np.abs(correction)) < NEWTON_TOLERANCE:
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
) -> np.ndarray:
    """Propagate the state through 12 implicit steps and return the end-of-December temperature."""
    state = temperature_K
    for month in range(12):
        state = implicit_monthly_step(
            state,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
        )
    return state


def find_periodic_temperature(
    initial_temperature: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
) -> np.ndarray:
    """Solve P(T) = T for the annual map using under-relaxed substitution."""

    state = np.maximum(initial_temperature, temperature_floor)
    damping = NEWTON_DAMPING
    previous_residual = np.inf

    for _iter in range(FIXED_POINT_MAX_ITERS):
        advanced = apply_annual_map(
            state,
            monthly_insolation,
            month_durations,
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
        )

        residual = advanced - state
        max_residual = float(np.max(np.abs(residual)))

        if max_residual < NEWTON_TOLERANCE:
            return advanced

        if max_residual > previous_residual:
            damping = max(damping * 0.5, 0.1)
        else:
            damping = min(damping * 1.1, 0.8)

        state = np.maximum(state + damping * residual, temperature_floor)
        previous_residual = max_residual

    raise RuntimeError(
        "Failed to converge to a periodic solution after "
        f"{FIXED_POINT_MAX_ITERS} iterations (residual {previous_residual:.3e} K)"
    )


def integrate_periodic_cycle(
    initial_temperature: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    rhs_fn: RhsFunc,
    rhs_temperature_derivative_fn: RhsDerivativeFunc,
    temperature_floor: float,
) -> np.ndarray:
    """Return the 12-month sequence of start-of-month temperatures for the periodic solution."""
    temps = np.empty((12,) + initial_temperature.shape, dtype=float)
    state = initial_temperature
    for month in range(12):
        next_state = implicit_monthly_step(
            state,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
        )
        temps[month] = 0.5 * (state + next_state)
        state = next_state
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
    diffusion_operator = create_diffusion_operator(
        lon2d,
        lat2d,
        heat_capacity_field,
        land_mask=land_mask,
        atmosphere_heat_capacity=radiation_config.atmosphere_heat_capacity,
        config=diffusion_config,
    )
    month_durations = DAYS_PER_MONTH * SECONDS_PER_DAY

    def solve_with_albedo(
        albedo_field: np.ndarray,
        initial_temperature: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rhs_fn, rhs_temperature_derivative_fn = rhs_factory(
            heat_capacity_field, albedo_field, diffusion_operator, radiation_config
        )

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
        )

        monthly_temperatures = integrate_periodic_cycle(
            periodic_temperature,
            monthly_insolation,
            month_durations,
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=radiation_config.temperature_floor,
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
    resolved_snow = snow_config or SnowAlbedoConfig()

    def rhs_factory(
        heat_capacity_field: np.ndarray,
        albedo_field: np.ndarray,
        diffusion_operator: LayeredDiffusionOperator,
        config: RadiationConfig,
    ):
        surface_matrix = (
            diffusion_operator.surface.matrix
            if diffusion_operator.surface.enabled
            else None
        )
        atmosphere_matrix = (
            diffusion_operator.atmosphere.matrix
            if config.include_atmosphere and diffusion_operator.atmosphere.enabled
            else None
        )

        def rhs(temperature: np.ndarray, insolation: np.ndarray) -> np.ndarray:
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
                return radiative
            if diffusion_operator.surface.enabled:
                return radiative + diffusion_operator.surface.tendency(temperature)
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
