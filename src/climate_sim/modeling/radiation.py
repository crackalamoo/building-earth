"""Radiative column model components with optional atmospheric layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadiationConfig:
    """Container for radiative transfer parameters."""

    stefan_boltzmann: float = 5.670374419e-8  # W m-2 K-4
    emissivity_surface: float = 1.0
    include_atmosphere: bool = True
    atmosphere_heat_capacity: float = 1.0e7  # J m-2 K-1, ~2-3 km troposphere column
    temperature_floor: float = 10.0  # K
    relative_humidity: float = 0.7
    water_vapor_absorption_coefficient: float = 120.0
    reference_surface_pressure: float = 101325.0  # Pa


_WATER_VAPOR_TO_DRY_AIR_MASS_RATIO = 0.622
_LATENT_HEAT_VAPORIZATION = 2.5e6  # J kg-1
_VAPOR_GAS_CONSTANT = 461.5  # J kg-1 K-1
_TRIPLE_POINT_TEMPERATURE = 273.16  # K
_TRIPLE_POINT_SATURATION_PRESSURE = 611.2  # Pa


def _saturation_vapor_pressure(temperature_K: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure following the Clausius-Clapeyron relation."""

    temperature = np.asarray(temperature_K)
    exponent = (
        _LATENT_HEAT_VAPORIZATION
        / _VAPOR_GAS_CONSTANT
        * (1.0 / _TRIPLE_POINT_TEMPERATURE - 1.0 / temperature)
    )
    return _TRIPLE_POINT_SATURATION_PRESSURE * np.exp(exponent)


def _saturation_specific_humidity(
    temperature_K: np.ndarray, pressure_Pa: float
) -> np.ndarray:
    """Saturation specific humidity for moist air at the given pressure."""

    e_s = _saturation_vapor_pressure(temperature_K)
    epsilon = _WATER_VAPOR_TO_DRY_AIR_MASS_RATIO
    denominator = np.maximum(pressure_Pa - (1.0 - epsilon) * e_s, 1e-6)
    return epsilon * e_s / denominator


def _saturation_specific_humidity_derivative(
    temperature_K: np.ndarray, pressure_Pa: float
) -> np.ndarray:
    """Temperature derivative of the saturation specific humidity."""

    temperature = np.asarray(temperature_K)
    e_s = _saturation_vapor_pressure(temperature)
    de_s_dT = (
        e_s
        * _LATENT_HEAT_VAPORIZATION
        / _VAPOR_GAS_CONSTANT
        / np.power(temperature, 2)
    )

    epsilon = _WATER_VAPOR_TO_DRY_AIR_MASS_RATIO
    denominator = np.maximum(pressure_Pa - (1.0 - epsilon) * e_s, 1e-6)
    numerator = epsilon * e_s

    dnumerator_dT = epsilon * de_s_dT
    ddenominator_dT = -(1.0 - epsilon) * de_s_dT

    return (dnumerator_dT * denominator - numerator * ddenominator_dT) / np.power(
        denominator, 2
    )


def _clear_sky_emissivity(
    atmospheric_temperature_K: np.ndarray,
    moisture_temperature_K: np.ndarray,
    config: RadiationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Clear-sky emissivity and its derivative with respect to moisture temperature."""

    rh = config.relative_humidity
    kappa = config.water_vapor_absorption_coefficient
    pressure = config.reference_surface_pressure

    q_sat = _saturation_specific_humidity(moisture_temperature_K, pressure)
    dq_dT_moisture = _saturation_specific_humidity_derivative(
        moisture_temperature_K, pressure
    )

    optical_depth = kappa * rh * q_sat
    emissivity = 1.0 - np.exp(-optical_depth)
    d_emissivity_dT_moisture = np.exp(-optical_depth) * kappa * rh * dq_dT_moisture

    # The emissivity does not directly depend on the atmospheric layer temperature,
    # but the argument is kept so the caller can use a consistent interface.
    _ = atmospheric_temperature_K

    return emissivity, d_emissivity_dT_moisture


def _with_floor(values: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(values, floor)


def radiative_balance_rhs(
    temperature_K: np.ndarray,
    insolation_W_m2: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    albedo_field: np.ndarray,
    config: RadiationConfig,
) -> np.ndarray:
    """Column energy-balance tendency for the configured radiative model."""

    floor = config.temperature_floor

    if not config.include_atmosphere:
        temperature = _with_floor(temperature_K, floor)
        emitted = config.emissivity_surface * config.stefan_boltzmann * np.power(temperature, 4)
        absorbed = insolation_W_m2 * (1.0 - albedo_field)
        return (absorbed - emitted) / heat_capacity_field

    surface = _with_floor(temperature_K[0], floor)
    atmosphere = _with_floor(temperature_K[1], floor)

    sigma = config.stefan_boltzmann
    eps_sfc = config.emissivity_surface
    eps_atm, _ = _clear_sky_emissivity(atmosphere, surface, config)

    emitted_surface = eps_sfc * sigma * np.power(surface, 4)
    emitted_atmosphere = eps_atm * sigma * np.power(atmosphere, 4)
    absorbed_shortwave = insolation_W_m2 * (1.0 - albedo_field)

    downward_longwave = emitted_atmosphere
    absorbed_from_surface = eps_atm * emitted_surface

    surface_tendency = (
        absorbed_shortwave + downward_longwave - emitted_surface
    ) / heat_capacity_field

    atmosphere_tendency = (
        absorbed_from_surface - 2.0 * emitted_atmosphere
    ) / config.atmosphere_heat_capacity

    return np.stack([surface_tendency, atmosphere_tendency])


def radiative_balance_rhs_temperature_derivative(
    temperature_K: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    config: RadiationConfig,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Partial derivatives of the radiative tendency with respect to temperature."""

    floor = config.temperature_floor
    sigma = config.stefan_boltzmann

    if not config.include_atmosphere:
        temperature = _with_floor(temperature_K, floor)
        coeff = -4.0 * config.emissivity_surface * sigma * np.power(temperature, 3)
        return coeff / heat_capacity_field

    surface = _with_floor(temperature_K[0], floor)
    atmosphere = _with_floor(temperature_K[1], floor)

    eps_atm, deps_dT_surface = _clear_sky_emissivity(atmosphere, surface, config)

    emitted_surface = config.emissivity_surface * sigma * np.power(surface, 4)

    surface_diag = (
        deps_dT_surface * sigma * np.power(atmosphere, 4)
        - 4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
    ) / heat_capacity_field
    atmosphere_diag = (
        -8.0 * eps_atm * sigma * np.power(atmosphere, 3)
    ) / config.atmosphere_heat_capacity

    surface_coupling = (
        4.0 * eps_atm * sigma * np.power(atmosphere, 3)
    ) / heat_capacity_field
    atmosphere_coupling = (
        deps_dT_surface * (emitted_surface - 2.0 * sigma * np.power(atmosphere, 4))
        + 4.0
        * eps_atm
        * config.emissivity_surface
        * sigma
        * np.power(surface, 3)
    ) / config.atmosphere_heat_capacity

    diag = np.stack([surface_diag, atmosphere_diag])
    cross = np.zeros((2, 2) + surface.shape, dtype=float)
    cross[0, 1] = surface_coupling
    cross[1, 0] = atmosphere_coupling
    return diag, cross


def radiative_equilibrium_initial_guess(
    monthly_insolation: np.ndarray,
    *,
    albedo_field: np.ndarray,
    config: RadiationConfig,
) -> np.ndarray:
    """Initial temperature guess via local radiative equilibrium."""

    sigma = config.stefan_boltzmann
    absorbed = monthly_insolation.mean(axis=0) * (1.0 - albedo_field)
    absorbed = np.maximum(absorbed, 1e-6)

    if not config.include_atmosphere:
        surface = np.power(absorbed / (config.emissivity_surface * sigma), 0.25)
        return np.maximum(surface, config.temperature_floor)

    surface = np.power(absorbed / (config.emissivity_surface * sigma), 0.25)
    atmosphere = np.power(absorbed / sigma, 0.25)
    for _ in range(4):
        epsilon_atm, _ = _clear_sky_emissivity(atmosphere, surface, config)
        denom = np.maximum(1.0 - 0.5 * epsilon_atm, 1e-6)
        surface = np.power(absorbed / (denom * sigma), 0.25)
        atmosphere = np.power(0.5, 0.25) * surface

    stacked = np.stack([surface, atmosphere])
    return _with_floor(stacked, config.temperature_floor)
