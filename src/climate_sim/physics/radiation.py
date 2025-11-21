"""Radiative column model components with optional atmospheric layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.humidity import (
    compute_cloud_cover,
    specific_humidity_to_relative_humidity,
)


@dataclass(frozen=True)
class RadiationConfig:
    """Container for radiative transfer parameters."""

    stefan_boltzmann: float = 5.670374419e-8  # W m-2 K-4
    emissivity_surface: float = 1.0
    emissivity_atmosphere: float = 0.88
    include_atmosphere: bool = True
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
    land_mask: np.ndarray | None = None,
    humidity_q: np.ndarray | None = None,
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

    # Compute cloud cover from humidity if available, else use latitude fallback
    if humidity_q is not None:
        rh = specific_humidity_to_relative_humidity(humidity_q, surface)
        cloud_cover = compute_cloud_cover(
            relative_humidity=rh,
            land_mask=land_mask,
        )
    else:
        # Fallback to latitude-based cloud cover (for compatibility with radiation module)
        cloud_cover = compute_cloud_cover(
            temperature=temperature_K,
            land_mask=land_mask,
        )
    atm_albedo_field = 0.05 + 0.35 * cloud_cover

    sigma = config.stefan_boltzmann
    eps_sfc = config.emissivity_surface
    eps_atm = 0.7 + 0.25 * cloud_cover
    eps_toa = 0.6

    emitted_surface = eps_sfc * sigma * np.power(surface, 4)
    emitted_atmosphere = eps_atm * sigma * np.power(atmosphere, 4)
    emitted_toa = eps_toa * sigma * np.power(atmosphere, 4)
    # Shortwave partitioning
    alpha_atm = atm_albedo_field
    beta_atm = getattr(config, "shortwave_absorptance_atmosphere", 0.0)

    # SW absorbed in atmosphere
    absorbed_shortwave_atm = beta_atm * insolation_W_m2

    # SW reaching surface, then partially absorbed
    sw_down_surface = (1.0 - alpha_atm - beta_atm) * insolation_W_m2
    absorbed_shortwave_sfc = sw_down_surface * (1.0 - albedo_field)

    # Longwave
    downward_longwave = emitted_atmosphere
    absorbed_from_surface = eps_atm * emitted_surface

    surface_tendency = (
        absorbed_shortwave_sfc + downward_longwave - emitted_surface
    ) / heat_capacity_field

    atmosphere_tendency = (
        absorbed_shortwave_atm + absorbed_from_surface - emitted_atmosphere - emitted_toa
    ) / config.atmosphere_heat_capacity

    return np.stack([surface_tendency, atmosphere_tendency])


def radiative_balance_rhs_temperature_derivative(
    temperature_K: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    config: RadiationConfig,
    land_mask: np.ndarray | None = None,
    humidity_q: np.ndarray | None = None,
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

    # Compute cloud cover from humidity if available, else use latitude fallback
    if humidity_q is not None:
        rh = specific_humidity_to_relative_humidity(humidity_q, surface)
        cloud_cover = compute_cloud_cover(
            relative_humidity=rh,
            land_mask=land_mask,
        )
    else:
        cloud_cover = compute_cloud_cover(
            temperature=temperature_K,
            land_mask=land_mask,
        )
    eps_atm = 0.7 + 0.3 * cloud_cover

    surface_diag = (
        -4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
    ) / heat_capacity_field
    atmosphere_diag = (
        -8.0 * eps_atm * sigma * np.power(atmosphere, 3)
    ) / config.atmosphere_heat_capacity

    surface_coupling = (
        4.0
        * eps_atm
        * sigma
        * np.power(atmosphere, 3)
        / heat_capacity_field
    )
    atmosphere_coupling = (
        4.0
        * eps_atm
        * config.emissivity_surface
        * sigma
        * np.power(surface, 3)
        / config.atmosphere_heat_capacity
    )

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
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Initial temperature guess via local radiative equilibrium."""

    sigma = config.stefan_boltzmann
    absorbed = monthly_insolation.mean(axis=0) * (1.0 - albedo_field)
    absorbed = np.maximum(absorbed, 1e-6)

    if not config.include_atmosphere:
        surface = np.power(absorbed / (config.emissivity_surface * sigma), 0.25)
        return np.maximum(surface, config.temperature_floor)

    # Use latitude-based fallback for initial guess since we don't have actual humidity yet
    dummy_temp = np.zeros((2,) + albedo_field.shape, dtype=float)
    cloud_cover = compute_cloud_cover(temperature=dummy_temp, land_mask=land_mask)
    epsilon_atm = 0.7 + 0.3 * cloud_cover

    denom = np.maximum(2.0 - epsilon_atm, 1e-6)
    atmosphere = np.power(absorbed / (denom * sigma), 0.25)
    surface = np.power(2.0 * absorbed / (denom * sigma), 0.25)

    stacked = np.stack([surface, atmosphere])
    return _with_floor(stacked, config.temperature_floor)
