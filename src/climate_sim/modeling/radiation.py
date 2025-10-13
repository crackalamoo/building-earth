"""Radiative column model components with optional atmospheric layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadiationConfig:
    """Container for radiative transfer parameters."""

    stefan_boltzmann: float = 5.670374419e-8  # W m-2 K-4
    emissivity_surface: float = 1.0
    emissivity_atmosphere: float = 0.77
    include_atmosphere: bool = False
    atmosphere_heat_capacity: float = 1.0e7  # J m-2 K-1, ~2-3 km troposphere column
    temperature_floor: float = 10.0  # K


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
    eps_atm = config.emissivity_atmosphere

    emitted_surface = eps_sfc * sigma * np.power(surface, 4)
    emitted_atmosphere = eps_atm * sigma * np.power(atmosphere, 4)
    absorbed_shortwave = insolation_W_m2 * (1.0 - albedo_field)

    surface_tendency = (
        absorbed_shortwave + eps_atm * emitted_atmosphere - emitted_surface
    ) / heat_capacity_field

    atmosphere_tendency = (
        emitted_surface - 2.0 * eps_atm * emitted_atmosphere
    ) / config.atmosphere_heat_capacity

    return np.stack([surface_tendency, atmosphere_tendency])


def radiative_balance_rhs_temperature_derivative(
    temperature_K: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    config: RadiationConfig,
) -> np.ndarray:
    """Partial derivative d(dT/dt)/dT for the configured model (diagonal terms)."""

    floor = config.temperature_floor
    sigma = config.stefan_boltzmann

    if not config.include_atmosphere:
        temperature = _with_floor(temperature_K, floor)
        coeff = -4.0 * config.emissivity_surface * sigma * np.power(temperature, 3)
        return coeff / heat_capacity_field

    surface = _with_floor(temperature_K[0], floor)
    atmosphere = _with_floor(temperature_K[1], floor)

    surface_coeff = (
        -4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
    ) / heat_capacity_field
    atmosphere_coeff = (
        -8.0 * config.emissivity_atmosphere * sigma * np.power(atmosphere, 3)
    ) / config.atmosphere_heat_capacity

    return np.stack([surface_coeff, atmosphere_coeff])


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

    surface = np.power(
        2.0 * absorbed / (config.emissivity_surface * sigma),
        0.25,
    )
    atmosphere = np.power(
        absorbed / (config.emissivity_atmosphere * sigma),
        0.25,
    )

    stacked = np.stack([surface, atmosphere])
    return _with_floor(stacked, config.temperature_floor)
