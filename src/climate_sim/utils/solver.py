"""General-purpose periodic solver utilities for energy-balance models."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import optimize

import climate_sim.modeling.radiation as radiation
from climate_sim.utils.grid import create_lat_lon_grid, expand_latitude_field
from climate_sim.utils.solar import DAYS_PER_MONTH, SECONDS_PER_DAY, compute_monthly_insolation_field
from climate_sim.utils.landmask import compute_heat_capacity_field

NEWTON_TOLERANCE = 1e-5  # K
NEWTON_MAX_ITERS = 16
NEWTON_DAMPING = 0.5
TEMPERATURE_FLOOR_K = 10.0  # K


RhsFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
RhsDerivativeFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
RhsFactory = Callable[[np.ndarray], Tuple[RhsFunc, RhsDerivativeFunc]]
InitialGuessFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
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
        rhs_derivative = rhs_temperature_derivative_fn(temp_capped, insolation_W_m2)
        derivative = 1.0 - dt_seconds * rhs_derivative

        correction = residual / derivative
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
    """Solve P(T) = T for the annual map using a fixed-point iteration."""

    def annual_map_flat(state_flat: np.ndarray) -> np.ndarray:
        state = state_flat.reshape(initial_temperature.shape)
        advanced = apply_annual_map(
            state,
            monthly_insolation,
            month_durations,
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
        )
        damped = NEWTON_DAMPING * advanced + (1.0 - NEWTON_DAMPING) * state
        return damped.ravel()

    solution_flat = optimize.fixed_point(
        annual_map_flat,
        initial_temperature.ravel(),
        xtol=NEWTON_TOLERANCE,
        maxiter=NEWTON_MAX_ITERS,
    )
    return solution_flat.reshape(initial_temperature.shape)


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
        temps[month] = state
        state = implicit_monthly_step(
            state,
            monthly_insolation[month],
            month_durations[month],
            rhs_fn=rhs_fn,
            rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
            temperature_floor=temperature_floor,
        )
    return temps


def compute_periodic_cycle_kelvin(
    *,
    resolution_deg: float,
    monthly_insolation_lat_fn: MonthlyInsolationLatFunc,
    heat_capacity_field_fn: HeatCapacityFieldFunc,
    rhs_factory: RhsFactory,
    initial_guess_fn: InitialGuessFunc,
    temperature_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve for the periodic temperature cycle and return results in Kelvin."""
    lon2d, lat2d = create_lat_lon_grid(resolution_deg)

    monthly_insolation_lat = monthly_insolation_lat_fn(lat2d)
    monthly_insolation = expand_latitude_field(monthly_insolation_lat, lon2d.shape[1])

    heat_capacity_field = heat_capacity_field_fn(lon2d, lat2d)
    rhs_fn, rhs_temperature_derivative_fn = rhs_factory(heat_capacity_field)

    initial_temperature = initial_guess_fn(monthly_insolation, heat_capacity_field)

    month_durations = DAYS_PER_MONTH * SECONDS_PER_DAY

    periodic_temperature = find_periodic_temperature(
        initial_temperature,
        monthly_insolation,
        month_durations,
        rhs_fn=rhs_fn,
        rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
        temperature_floor=temperature_floor,
    )

    monthly_temperatures = integrate_periodic_cycle(
        periodic_temperature,
        monthly_insolation,
        month_durations,
        rhs_fn=rhs_fn,
        rhs_temperature_derivative_fn=rhs_temperature_derivative_fn,
        temperature_floor=temperature_floor,
    )

    return lon2d, lat2d, monthly_temperatures


def compute_periodic_cycle_celsius(
    resolution_deg: float = 1.0,
    *,
    solar_constant: float = None,
    ocean_heat_capacity: float = None,
    land_heat_capacity: float = None,
    emissivity_sfc: float = None,
    emissivity_atm: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def rhs_factory(heat_capacity_field: np.ndarray):
        def rhs(temperature: np.ndarray, insolation: np.ndarray) -> np.ndarray:
            return radiation.radiative_balance_rhs(
                temperature,
                insolation,
                heat_capacity_field=heat_capacity_field,
                emissivity_sfc=emissivity_sfc,
                emissivity_atm=emissivity_atm,
            )

        def rhs_derivative(temperature: np.ndarray, insolation: np.ndarray) -> np.ndarray:
            del insolation
            return radiation.radiative_balance_rhs_temperature_derivative(
                temperature,
                heat_capacity_field=heat_capacity_field,
                emissivity_sfc=emissivity_sfc,
                emissivity_atm=emissivity_atm,
            )

        return rhs, rhs_derivative

    def initial_guess_fn(
        monthly_insolation: np.ndarray,
        heat_capacity_field: np.ndarray,
    ) -> np.ndarray:
        del heat_capacity_field
        return radiation.radiative_equilibrium_initial_guess(
            monthly_insolation,
            emissivity_sfc=emissivity_sfc,
            emissivity_atm=emissivity_atm,
        )

    lon2d, lat2d, monthly_temperatures_K = compute_periodic_cycle_kelvin(
        resolution_deg=resolution_deg,
        monthly_insolation_lat_fn=monthly_insolation_lat_fn,
        heat_capacity_field_fn=heat_capacity_field_fn,
        rhs_factory=rhs_factory,
        initial_guess_fn=initial_guess_fn,
        temperature_floor=TEMPERATURE_FLOOR_K,
    )

    monthly_temperatures_C = monthly_temperatures_K - 273.15
    return lon2d, lat2d, monthly_temperatures_C
