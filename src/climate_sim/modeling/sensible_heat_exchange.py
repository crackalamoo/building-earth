"""Neutral sensible heat exchange between the surface and atmosphere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.utils.atmosphere import (
    STANDARD_LAPSE_RATE_K_PER_M,
    adjust_temperature_by_elevation,
)
from climate_sim.utils.elevation import VON_KARMAN_CONSTANT
from climate_sim.utils.constants import GAS_CONSTANT_J_KG_K


@dataclass(frozen=True)
class SensibleHeatExchangeConfig:
    """Configuration for the neutral sensible heat exchange model."""

    enabled: bool = True
    von_karman: float = VON_KARMAN_CONSTANT
    heat_capacity_air_J_kg_K: float = 1005.0
    gas_constant_dry_air_J_kg_K: float = GAS_CONSTANT_J_KG_K
    lapse_rate_K_per_m: float = STANDARD_LAPSE_RATE_K_PER_M
    minimum_wind_speed_m_s: float = 0.1
    reference_height_surface_m: float = 10.0
    land_reference_height_m: float = 1000.0
    ocean_reference_height_m: float = 500.0


def sensible_heat_exchange_tendencies(
    surface_temperature_K: np.ndarray,
    atmosphere_temperature_K: np.ndarray,
    *,
    surface_pressure_Pa: np.ndarray,
    wind_speed_10m_m_s: np.ndarray,
    roughness_length_m: np.ndarray,
    surface_heat_capacity_J_m2_K: np.ndarray,
    atmosphere_heat_capacity_J_m2_K: np.ndarray,
    land_mask: np.ndarray,
    config: SensibleHeatExchangeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the surface and atmospheric tendencies from sensible heat exchange."""

    if not config.enabled:
        zeros = np.zeros_like(surface_temperature_K)
        return zeros, zeros

    surface_temperature = np.asarray(surface_temperature_K, dtype=float)
    atmosphere_temperature = np.asarray(atmosphere_temperature_K, dtype=float)
    pressure = np.asarray(surface_pressure_Pa, dtype=float)
    wind_speed = np.asarray(wind_speed_10m_m_s, dtype=float)
    roughness = np.asarray(roughness_length_m, dtype=float)
    heat_capacity_surface = np.asarray(surface_heat_capacity_J_m2_K, dtype=float)
    heat_capacity_atmosphere = np.asarray(atmosphere_heat_capacity_J_m2_K, dtype=float)
    land_mask_bool = np.asarray(land_mask, dtype=bool)

    if surface_temperature.shape != atmosphere_temperature.shape:
        raise ValueError("Surface and atmosphere temperatures must share the same shape")
    if surface_temperature.shape != pressure.shape:
        raise ValueError("Surface pressure must match the temperature field shape")
    if surface_temperature.shape != wind_speed.shape:
        raise ValueError("Wind speed must match the temperature field shape")
    if surface_temperature.shape != roughness.shape:
        raise ValueError("Roughness length must match the temperature field shape")
    if surface_temperature.shape != heat_capacity_surface.shape:
        raise ValueError("Surface heat capacity must match the temperature field shape")
    if surface_temperature.shape != heat_capacity_atmosphere.shape:
        raise ValueError("Atmosphere heat capacity must match the temperature field shape")
    if surface_temperature.shape != land_mask_bool.shape:
        raise ValueError("Land mask must match the temperature field shape")

    reference_height_surface = config.reference_height_surface_m
    land_height = config.land_reference_height_m
    ocean_height = config.ocean_reference_height_m

    z_ref_atm = np.where(land_mask_bool, land_height, ocean_height)
    delta_z = z_ref_atm - reference_height_surface

    atmosphere_temperature_c = atmosphere_temperature - 273.15
    near_surface_air_c = adjust_temperature_by_elevation(
        atmosphere_temperature_c,
        delta_z,
    )
    near_surface_air_K = near_surface_air_c + 273.15
    near_surface_air_K = np.maximum(near_surface_air_K, 1.0)

    ta10_safe = near_surface_air_K

    gas_constant = config.gas_constant_dry_air_J_kg_K
    rho = pressure / (gas_constant * ta10_safe)

    z0_safe = np.maximum(roughness, 1.0e-6)
    log_argument = reference_height_surface / z0_safe
    log_argument = np.maximum(log_argument, 1.0 + 1.0e-9)
    lm = np.log(log_argument)
    with np.errstate(divide="ignore", invalid="ignore"):
        ch = (config.von_karman**2) / (lm**2)

    wind_abs = np.maximum(np.abs(wind_speed), config.minimum_wind_speed_m_s)

    lapse_rate = config.lapse_rate_K_per_m
    delta_temperature = surface_temperature - atmosphere_temperature + lapse_rate * delta_z

    heat_flux = rho * config.heat_capacity_air_J_kg_K * ch * wind_abs * delta_temperature

    heat_capacity_surface_safe = np.maximum(heat_capacity_surface, 1.0e-9)
    heat_capacity_atmosphere_safe = np.maximum(heat_capacity_atmosphere, 1.0e-9)

    surface_tendency = -heat_flux / heat_capacity_surface_safe
    atmosphere_tendency = heat_flux / heat_capacity_atmosphere_safe

    return surface_tendency, atmosphere_tendency
