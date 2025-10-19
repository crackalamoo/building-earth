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
    include_atmosphere: bool = True
    atmosphere_heat_capacity: float = 1.0e7  # J m-2 K-1, ~2-3 km troposphere column
    temperature_floor: float = 10.0  # K
    shortwave_absorptance_atmosphere: float = 0.12


def _with_floor(values: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(values, floor)


def _clip_unit_interval(field: np.ndarray) -> np.ndarray:
    """Clip albedo-like fields to the physically admissible range."""

    return np.clip(field, 0.0, 0.999)


def combine_surface_and_atmosphere_albedo(
    surface_albedo: np.ndarray,
    atmosphere_albedo: np.ndarray | None,
    *,
    shortwave_absorptance_atmosphere: float,
) -> np.ndarray:
    """Return the effective top-of-atmosphere albedo for the column."""

    surface = _clip_unit_interval(np.asarray(surface_albedo, dtype=float))
    if atmosphere_albedo is None:
        return surface

    atmosphere = _clip_unit_interval(np.asarray(atmosphere_albedo, dtype=float))
    absorptance = float(np.clip(shortwave_absorptance_atmosphere, 0.0, 1.0))

    transmitted_fraction = (1.0 - atmosphere) * (1.0 - absorptance)
    combined = atmosphere + transmitted_fraction * surface
    return _clip_unit_interval(combined)


def infer_surface_albedo_from_toa(
    top_of_atmosphere_albedo: np.ndarray,
    atmosphere_albedo: np.ndarray | None,
    *,
    shortwave_absorptance_atmosphere: float,
) -> np.ndarray:
    """Infer the surface albedo given the total column and atmospheric components."""

    toa = _clip_unit_interval(np.asarray(top_of_atmosphere_albedo, dtype=float))
    if atmosphere_albedo is None:
        return toa

    atmosphere = _clip_unit_interval(np.asarray(atmosphere_albedo, dtype=float))
    absorptance = float(np.clip(shortwave_absorptance_atmosphere, 0.0, 1.0))

    denom = (1.0 - atmosphere) * (1.0 - absorptance)
    denom = np.maximum(denom, 1.0e-6)
    numerator = toa - atmosphere
    inferred = numerator / denom
    surface = np.where(
        numerator <= 0.0,
        toa,
        inferred,
    )
    return _clip_unit_interval(surface)


def _partition_shortwave_flux(
    insolation_W_m2: np.ndarray,
    surface_albedo: np.ndarray,
    atmosphere_albedo: np.ndarray | None,
    *,
    shortwave_absorptance_atmosphere: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the shortwave absorption by the surface and atmosphere."""

    insolation = np.asarray(insolation_W_m2, dtype=float)
    surface = _clip_unit_interval(np.asarray(surface_albedo, dtype=float))
    if atmosphere_albedo is None:
        atmosphere = 0.0
    else:
        atmosphere = _clip_unit_interval(np.asarray(atmosphere_albedo, dtype=float))

    absorptance = float(np.clip(shortwave_absorptance_atmosphere, 0.0, 1.0))

    reflected_by_atmosphere = insolation * atmosphere
    transmitted_after_reflection = insolation - reflected_by_atmosphere
    absorbed_by_atmosphere = transmitted_after_reflection * absorptance
    transmitted_to_surface = transmitted_after_reflection - absorbed_by_atmosphere
    absorbed_by_surface = transmitted_to_surface * (1.0 - surface)

    return absorbed_by_surface, absorbed_by_atmosphere


def radiative_balance_rhs(
    temperature_K: np.ndarray,
    insolation_W_m2: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    albedo_field: np.ndarray,
    atmosphere_albedo_field: np.ndarray | None = None,
    config: RadiationConfig,
) -> np.ndarray:
    """Column energy-balance tendency for the configured radiative model."""

    floor = config.temperature_floor

    alpha_sw_atm = config.shortwave_absorptance_atmosphere
    if not config.include_atmosphere:
        alpha_sw_atm = 0.0
    albedo_atmosphere = atmosphere_albedo_field

    if not config.include_atmosphere:
        temperature = _with_floor(temperature_K, floor)
        emitted = config.emissivity_surface * config.stefan_boltzmann * np.power(temperature, 4)
        absorbed_surface, _ = _partition_shortwave_flux(
            insolation_W_m2,
            albedo_field,
            albedo_atmosphere,
            shortwave_absorptance_atmosphere=alpha_sw_atm,
        )
        return (absorbed_surface - emitted) / heat_capacity_field

    surface = _with_floor(temperature_K[0], floor)
    atmosphere = _with_floor(temperature_K[1], floor)

    sigma = config.stefan_boltzmann
    eps_sfc = config.emissivity_surface
    eps_atm = config.emissivity_atmosphere

    emitted_surface = eps_sfc * sigma * np.power(surface, 4)
    emitted_atmosphere = eps_atm * sigma * np.power(atmosphere, 4)

    absorbed_shortwave_sfc, absorbed_shortwave_atm = _partition_shortwave_flux(
        insolation_W_m2,
        albedo_field,
        albedo_atmosphere,
        shortwave_absorptance_atmosphere=alpha_sw_atm,
    )

    downward_longwave = emitted_atmosphere
    absorbed_from_surface = eps_atm * emitted_surface

    surface_tendency = (
        absorbed_shortwave_sfc + downward_longwave - emitted_surface
    ) / heat_capacity_field

    atmosphere_tendency = (
        absorbed_shortwave_atm + absorbed_from_surface - 2.0 * emitted_atmosphere
    ) / config.atmosphere_heat_capacity

    return np.stack([surface_tendency, atmosphere_tendency])


def radiative_balance_rhs_temperature_derivative(
    temperature_K: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    atmosphere_albedo_field: np.ndarray | None = None,
    config: RadiationConfig,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Partial derivatives of the radiative tendency with respect to temperature."""

    del atmosphere_albedo_field  # shortwave reflection is temperature-independent
    floor = config.temperature_floor
    sigma = config.stefan_boltzmann

    if not config.include_atmosphere:
        temperature = _with_floor(temperature_K, floor)
        coeff = -4.0 * config.emissivity_surface * sigma * np.power(temperature, 3)
        return coeff / heat_capacity_field

    surface = _with_floor(temperature_K[0], floor)
    atmosphere = _with_floor(temperature_K[1], floor)

    surface_diag = (
        -4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
    ) / heat_capacity_field
    atmosphere_diag = (
        -8.0 * config.emissivity_atmosphere * sigma * np.power(atmosphere, 3)
    ) / config.atmosphere_heat_capacity

    surface_coupling = (
        4.0
        * config.emissivity_atmosphere
        * sigma
        * np.power(atmosphere, 3)
        / heat_capacity_field
    )
    atmosphere_coupling = (
        4.0
        * config.emissivity_atmosphere
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
    atmosphere_albedo_field: np.ndarray | None = None,
    config: RadiationConfig,
) -> np.ndarray:
    """Initial temperature guess via local radiative equilibrium."""

    sigma = config.stefan_boltzmann
    alpha_sw_atm = config.shortwave_absorptance_atmosphere
    if not config.include_atmosphere:
        alpha_sw_atm = 0.0
    if atmosphere_albedo_field is None:
        albedo_atmosphere = 0.0
    else:
        albedo_atmosphere = atmosphere_albedo_field

    mean_insolation = monthly_insolation.mean(axis=0)
    absorbed_surface, absorbed_atmosphere = _partition_shortwave_flux(
        mean_insolation,
        albedo_field,
        albedo_atmosphere,
        shortwave_absorptance_atmosphere=alpha_sw_atm,
    )
    absorbed_surface = np.maximum(absorbed_surface, 1e-6)
    absorbed_atmosphere = np.maximum(absorbed_atmosphere, 0.0)

    if not config.include_atmosphere:
        surface = np.power(
            absorbed_surface / (config.emissivity_surface * sigma),
            0.25,
        )
        return np.maximum(surface, config.temperature_floor)

    epsilon_atm = max(config.emissivity_atmosphere, 1.0e-6)
    epsilon_surface = max(config.emissivity_surface, 1.0e-6)

    surface_numerator = absorbed_surface + 0.5 * absorbed_atmosphere
    surface_denom = epsilon_surface * sigma * (1.0 - 0.5 * epsilon_atm)
    surface = np.power(
        np.maximum(surface_numerator, 1.0e-6) / np.maximum(surface_denom, 1.0e-6),
        0.25,
    )

    atmosphere_term = absorbed_atmosphere / (2.0 * epsilon_atm * sigma)
    atmosphere_term = np.maximum(atmosphere_term, 0.0)
    atmosphere = np.power(
        atmosphere_term + 0.5 * epsilon_surface * np.power(surface, 4),
        0.25,
    )

    stacked = np.stack([surface, atmosphere])
    return _with_floor(stacked, config.temperature_floor)
