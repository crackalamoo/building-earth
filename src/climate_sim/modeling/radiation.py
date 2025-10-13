"""Monthly radiative balance solver with a one-year periodic boundary condition.

The model computes a 12-month temperature cycle on a regular latitude–longitude
grid by enforcing C dT/dt = S - εσT⁴ for each column, with the constraint that
the solution at the end of December matches the start of January.  Because the
columns decouple, the solver operates on the entire grid simultaneously using a
vectorised fixed-point iteration.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ..utils.landmask import compute_land_mask

SOLAR_CONSTANT = 1361.0  # W m-2
STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4
OBLIQUITY_DEGREES = 23.44  # Earth axial tilt
SECONDS_PER_DAY = 86400.0
DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
ANNUAL_DAYS = DAYS_PER_MONTH.sum()
OCEAN_HEAT_CAPACITY_M2 = 4.0e8  # J m-2 K-1, ~40 m mixed-layer ocean
LAND_HEAT_CAPACITY_M2 = 8.0e7  # J m-2 K-1, ~2 m soil
EMISSIVITY = 1.0
NEWTON_TOLERANCE = 1e-5  # K
NEWTON_MAX_ITERS = 16
MIN_TEMPERATURE_K = 50.0  # Prevent numerical underflow in radiative terms


def create_lat_lon_grid(resolution_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D lon/lat arrays on a regular grid centred on cell middles."""
    lats = np.arange(-90.0 + resolution_deg / 2, 90.0, resolution_deg)
    lons = np.arange(resolution_deg / 2, 360.0, resolution_deg)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d


def compute_heat_capacity_field(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    ocean_heat_capacity: float,
    land_heat_capacity: float,
) -> np.ndarray:
    """Assign per-cell heat capacity based on land/sea classification."""
    land_mask = compute_land_mask(lon2d, lat2d)
    heat_capacity = np.where(land_mask, land_heat_capacity, ocean_heat_capacity)
    return heat_capacity


def monthly_midpoint_days() -> np.ndarray:
    """Return the day-of-year for the midpoint of each calendar month."""
    starts = np.concatenate(([0.0], np.cumsum(DAYS_PER_MONTH)[:-1]))
    return starts + 0.5 * DAYS_PER_MONTH


def solar_declination(day_of_year: np.ndarray) -> np.ndarray:
    """Compute solar declination angle (radians) for given day of year."""
    mean_longitude = 2.0 * np.pi * (day_of_year - 80.0) / ANNUAL_DAYS
    obliquity_rad = np.deg2rad(OBLIQUITY_DEGREES)
    return np.arcsin(np.sin(obliquity_rad) * np.sin(mean_longitude))


def daily_mean_insolation(lat_rad: np.ndarray, declination_rad: np.ndarray, *, solar_constant: float) -> np.ndarray:
    """Compute daily-mean top-of-atmosphere insolation for a given declination."""
    tan_lat = np.tan(lat_rad)
    tan_dec = np.tan(declination_rad)
    cos_hour_angle = -tan_lat[:, None] * tan_dec[None, :]

    cos_hour_angle = np.clip(cos_hour_angle, -1.0, 1.0)
    hour_angle = np.arccos(cos_hour_angle)

    polar_night = cos_hour_angle >= 1.0
    polar_day = cos_hour_angle <= -1.0
    hour_angle[polar_night] = 0.0
    hour_angle[polar_day] = np.pi

    sin_lat = np.sin(lat_rad)[:, None]
    cos_lat = np.cos(lat_rad)[:, None]
    sin_dec = np.sin(declination_rad)[None, :]
    cos_dec = np.cos(declination_rad)[None, :]

    flux = (
        solar_constant
        / np.pi
        * (hour_angle * sin_lat * sin_dec + cos_lat * cos_dec * np.sin(hour_angle))
    )
    return np.maximum(flux, 0.0)


def compute_monthly_insolation_field(
    lat2d: np.ndarray,
    *,
    solar_constant: float = SOLAR_CONSTANT,
) -> np.ndarray:
    """Return monthly-averaged insolation (W m-2) at the top of the atmosphere."""
    lat_rad = np.deg2rad(lat2d[:, 0])
    midpoints = monthly_midpoint_days()
    declinations = solar_declination(midpoints)
    insolation_by_lat_month = daily_mean_insolation(lat_rad, declinations, solar_constant=solar_constant)
    return insolation_by_lat_month.T  # month, lat


def expand_lon(insolation_lat: np.ndarray, lon_size: int) -> np.ndarray:
    """Broadcast a latitude-only field across longitude."""
    return np.repeat(insolation_lat[..., None], lon_size, axis=2)


def radiative_balance_rhs(
    temperature_K: np.ndarray,
    insolation_W_m2: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    emissivity: float,
) -> np.ndarray:
    """Return dT/dt = (S - εσT⁴)/C for the column energy balance."""
    emitted = emissivity * STEFAN_BOLTZMANN * np.power(np.maximum(temperature_K, MIN_TEMPERATURE_K), 4)
    return (insolation_W_m2 - emitted) / heat_capacity_field


def implicit_monthly_step(
    temperature_K: np.ndarray,
    insolation_W_m2: np.ndarray,
    dt_seconds: float,
    *,
    heat_capacity_field: np.ndarray,
    emissivity: float,
) -> np.ndarray:
    """Advance the column temperature one implicit backward-Euler step."""
    temp_next = np.maximum(temperature_K, MIN_TEMPERATURE_K)

    for _ in range(NEWTON_MAX_ITERS):
        temp_capped = np.maximum(temp_next, MIN_TEMPERATURE_K)
        rhs_value = radiative_balance_rhs(
            temp_capped,
            insolation_W_m2,
            heat_capacity_field=heat_capacity_field,
            emissivity=emissivity,
        )
        residual = temp_capped - temperature_K - dt_seconds * rhs_value
        derivative = 1.0 + (
            4.0
            * emissivity
            * STEFAN_BOLTZMANN
            * dt_seconds
            * np.power(temp_capped, 3)
            / heat_capacity_field
        )

        correction = residual / derivative
        temp_candidate = temp_next - correction
        temp_next = np.maximum(temp_candidate, MIN_TEMPERATURE_K)

        if np.max(np.abs(correction)) < NEWTON_TOLERANCE:
            break

    return temp_next


def apply_annual_map(
    temperature_K: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    emissivity: float,
) -> np.ndarray:
    """Propagate the state through 12 implicit steps and return the end-of-year temperature."""
    state = temperature_K
    for month in range(12):
        state = implicit_monthly_step(
            state,
            monthly_insolation[month],
            month_durations[month],
            heat_capacity_field=heat_capacity_field,
            emissivity=emissivity,
        )
    return state


def find_periodic_temperature(
    initial_temperature: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    emissivity: float,
    damping: float = 0.5,
    tolerance: float = 1e-4,
    max_iterations: int = 50,
) -> np.ndarray:
    """Solve P(T) = T for the annual map using SciPy's fixed-point solver."""

    def annual_map_flat(state_flat: np.ndarray) -> np.ndarray:
        state = state_flat.reshape(initial_temperature.shape)
        advanced = apply_annual_map(
            state,
            monthly_insolation,
            month_durations,
            heat_capacity_field=heat_capacity_field,
            emissivity=emissivity,
        )
        damped = damping * advanced + (1.0 - damping) * state
        return damped.ravel()

    solution_flat = optimize.fixed_point(
        annual_map_flat,
        initial_temperature.ravel(),
        xtol=tolerance,
        maxiter=max_iterations,
    )
    return solution_flat.reshape(initial_temperature.shape)


def integrate_periodic_cycle(
    initial_temperature: np.ndarray,
    monthly_insolation: np.ndarray,
    month_durations: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    emissivity: float,
) -> np.ndarray:
    """Return the 12-month start-of-month temperatures for the periodic solution."""
    temps = np.empty((12,) + initial_temperature.shape, dtype=float)
    state = initial_temperature
    for month in range(12):
        temps[month] = state
        state = implicit_monthly_step(
            state,
            monthly_insolation[month],
            month_durations[month],
            heat_capacity_field=heat_capacity_field,
            emissivity=emissivity,
        )
    return temps


def compute_periodic_radiative_cycle_celsius(
    resolution_deg: float = 1.0,
    *,
    solar_constant: float = SOLAR_CONSTANT,
    ocean_heat_capacity: float = OCEAN_HEAT_CAPACITY_M2,
    land_heat_capacity: float = LAND_HEAT_CAPACITY_M2,
    emissivity: float = EMISSIVITY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 12-month °C temperature cycle with a strictly periodic solution."""
    lon2d, lat2d = create_lat_lon_grid(resolution_deg)

    monthly_insolation_lat = compute_monthly_insolation_field(lat2d, solar_constant=solar_constant)
    monthly_insolation = expand_lon(monthly_insolation_lat, lon2d.shape[1])

    month_durations = DAYS_PER_MONTH * SECONDS_PER_DAY

    heat_capacity_field = compute_heat_capacity_field(
        lon2d,
        lat2d,
        ocean_heat_capacity=ocean_heat_capacity,
        land_heat_capacity=land_heat_capacity,
    )

    annual_mean_insolation = monthly_insolation.mean(axis=0)
    equilibrium_guess = np.power(
        np.maximum(annual_mean_insolation, 1e-6) / (emissivity * STEFAN_BOLTZMANN), 0.25
    )
    periodic_temperature = find_periodic_temperature(
        equilibrium_guess,
        monthly_insolation,
        month_durations,
        heat_capacity_field=heat_capacity_field,
        emissivity=emissivity,
    )

    monthly_temperatures_K = integrate_periodic_cycle(
        periodic_temperature,
        monthly_insolation,
        month_durations,
        heat_capacity_field=heat_capacity_field,
        emissivity=emissivity,
    )
    monthly_temperatures_C = monthly_temperatures_K - 273.15

    return lon2d, lat2d, monthly_temperatures_C


def compute_temperature_celsius(
    resolution_deg: float = 1.0,
    *,
    solar_constant: float = SOLAR_CONSTANT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper returning January °C temperatures."""
    lon2d, lat2d, monthly_cycle = compute_periodic_radiative_cycle_celsius(
        resolution_deg=resolution_deg,
        solar_constant=solar_constant,
    )
    january = monthly_cycle[0]
    return lon2d, lat2d, january
