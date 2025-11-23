"""Neutral sensible heat exchange between the surface and atmosphere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.atmosphere import (
    compute_two_meter_temperature,
    STANDARD_LAPSE_RATE_K_PER_M
)
from climate_sim.data.constants import GAS_CONSTANT_J_KG_K, HEAT_CAPACITY_AIR_J_KG_K
from climate_sim.data.elevation import VON_KARMAN_CONSTANT

from climate_sim.physics.advection import AdvectionModel


@dataclass(frozen=True)
class SensibleHeatExchangeConfig:
    """Configuration for the neutral sensible heat exchange model."""

    enabled: bool = True
    von_karman: float = VON_KARMAN_CONSTANT
    gas_constant_dry_air_J_kg_K: float = GAS_CONSTANT_J_KG_K
    lapse_rate_K_per_m: float = STANDARD_LAPSE_RATE_K_PER_M
    minimum_wind_speed_m_s: float = 0.1
    reference_height_surface_m: float = 2.0
    include_lapse_rate_elevation: bool = False


class SensibleHeatExchangeModel:
    """Compute tendencies from neutral sensible heat exchange."""

    def __init__(
        self,
        *,
        land_mask: np.ndarray,
        surface_heat_capacity_J_m2_K: np.ndarray,
        atmosphere_heat_capacity_J_m2_K: np.ndarray | float,
        advection_model: AdvectionModel | None = None,
        config: SensibleHeatExchangeConfig | None = None,
    ) -> None:
        self._config = config or SensibleHeatExchangeConfig()

        land_mask_bool = np.asarray(land_mask, dtype=bool)
        heat_capacity_surface = np.asarray(surface_heat_capacity_J_m2_K, dtype=float)
        heat_capacity_atmosphere = np.asarray(atmosphere_heat_capacity_J_m2_K, dtype=float)

        if land_mask_bool.shape != heat_capacity_surface.shape:
            raise ValueError("Surface heat capacity must match the land mask shape")
        if heat_capacity_atmosphere.shape not in ((), land_mask_bool.shape):
            raise ValueError(
                "Atmospheric heat capacity must be scalar or match the land mask shape"
            )

        if heat_capacity_atmosphere.shape == ():
            heat_capacity_atmosphere = np.full(
                land_mask_bool.shape, float(heat_capacity_atmosphere)
            )

        self._land_mask = land_mask_bool
        self._surface_heat_capacity = np.maximum(heat_capacity_surface, 1.0e-9)
        self._atmosphere_heat_capacity = np.maximum(heat_capacity_atmosphere, 1.0e-9)
        self._advection_model = advection_model

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def compute_tendencies(
        self,
        surface_temperature_K: np.ndarray,
        atmosphere_temperature_K: np.ndarray,
        *,
        wind_speed_reference_m_s: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return surface and atmospheric tendencies from sensible heat exchange."""

        if not self.enabled:
            zeros = np.zeros_like(surface_temperature_K, dtype=float)
            return zeros, zeros

        surface_temperature = np.asarray(surface_temperature_K, dtype=float)
        atmosphere_temperature = np.asarray(atmosphere_temperature_K, dtype=float)

        if surface_temperature.shape != self._surface_heat_capacity.shape:
            raise ValueError(
                "Surface temperature must match the surface heat capacity field shape"
            )
        if atmosphere_temperature.shape != self._surface_heat_capacity.shape:
            raise ValueError(
                "Atmosphere temperature must match the surface heat capacity field shape"
            )

        # Use advection model for atmospheric properties if available
        if self._advection_model is not None:
            _pressure, rho, wind_speed_10m, ch = self._advection_model.compute_atmospheric_properties(
                surface_temperature,
                atmosphere_temperature,
                wind_speed_reference_m_s,
            )
        else:
            # Fallback: no wind, no exchange
            zeros = np.zeros_like(surface_temperature_K, dtype=float)
            return zeros, zeros

        # Compute near-surface air temperature
        atmosphere_temperature_c = atmosphere_temperature - 273.15
        near_surface_air_c = compute_two_meter_temperature(
            atmosphere_temperature_c,
            surface_temperature - 273.15,
        )
        near_surface_air_K = np.maximum(near_surface_air_c + 273.15, 10.0)

        wind_abs = np.maximum(np.abs(wind_speed_10m), self._config.minimum_wind_speed_m_s)

        # New: resistive throttling to the free-air node (Ta ~ your atmosphere_temperature)
        cp = HEAT_CAPACITY_AIR_J_KG_K

        # Surface conductance (W m-2 K-1)
        g_surf = rho * cp * ch * wind_abs
        r_surf = 1.0 / np.maximum(g_surf, 1e-9)

        # Mixing resistance (choose once; split land/ocean if you want)
        Cbl = 1.2e5  # J m-2 K-1
        tau = np.where(self._land_mask, 2*86400.0, 4*86400.0)  # s
        tau = (self._surface_heat_capacity * self._atmosphere_heat_capacity) / (    
            self._surface_heat_capacity + self._atmosphere_heat_capacity
        ) / (rho
            * HEAT_CAPACITY_AIR_J_KG_K
            * ch
            * wind_abs)
        r_mix = tau / Cbl

        # Effective flux to the free-air node (your atmosphere_temperature)
        delta_to_free = surface_temperature - near_surface_air_K
        heat_flux = delta_to_free / (r_surf + r_mix)

        # print(np.mean(ch_land), np.mean(ch_ocean), np.mean(ch), np.min(ch), np.max(ch))

        surface_tendency = -heat_flux / self._surface_heat_capacity
        atmosphere_tendency = heat_flux / self._atmosphere_heat_capacity
        # return np.zeros_like(surface_tendency), np.zeros_like(atmosphere_tendency)
        return surface_tendency, atmosphere_tendency
