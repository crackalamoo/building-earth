"""Latent heat exchange between the surface and atmosphere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.atmosphere import (
    compute_two_meter_temperature,
    log_law_map_wind_speed,
    STANDARD_LAPSE_RATE_K_PER_M
)
from climate_sim.data.constants import GAS_CONSTANT_J_KG_K, HEAT_CAPACITY_AIR_J_KG_K
from climate_sim.data.elevation import (
    VON_KARMAN_CONSTANT,
)
from climate_sim.physics.pressure import pressure_from_temperature_elevation


@dataclass(frozen=True)
class LatentHeatExchangeConfig:
    """Configuration for the latent heat exchange model."""

    enabled: bool = True
    minimum_wind_speed_m_s: float = 0.1


class LatentHeatExchangeModel:
    """Compute tendencies from latent heat exchange."""

    def __init__(
        self,
        *,
        land_mask: np.ndarray,
        roughness_length_m: np.ndarray,
        surface_heat_capacity_J_m2_K: np.ndarray,
        atmosphere_heat_capacity_J_m2_K: np.ndarray | float,
        topographic_elevation_m: np.ndarray | None = None,
        config: LatentHeatExchangeConfig | None = None,
    ) -> None:
        self._config = config or LatentHeatExchangeConfig()

        land_mask_bool = np.asarray(land_mask, dtype=bool)
        roughness = np.asarray(roughness_length_m, dtype=float)
        heat_capacity_surface = np.asarray(surface_heat_capacity_J_m2_K, dtype=float)
        heat_capacity_atmosphere = np.asarray(atmosphere_heat_capacity_J_m2_K, dtype=float)
        if topographic_elevation_m is None:
            topographic = np.zeros_like(land_mask_bool, dtype=float)
        else:
            topographic = np.asarray(topographic_elevation_m, dtype=float)

        if land_mask_bool.shape != roughness.shape:
            raise ValueError("Land mask and roughness fields must share the same shape")
        if land_mask_bool.shape != heat_capacity_surface.shape:
            raise ValueError("Surface heat capacity must match the land mask shape")
        if heat_capacity_atmosphere.shape not in ((), land_mask_bool.shape):
            raise ValueError(
                "Atmospheric heat capacity must be scalar or match the land mask shape"
            )
        if topographic.shape != land_mask_bool.shape:
            raise ValueError("Topographic elevation must match the land mask shape")

        if heat_capacity_atmosphere.shape == ():
            heat_capacity_atmosphere = np.full(
                land_mask_bool.shape, float(heat_capacity_atmosphere)
            )

        self._roughness = np.maximum(roughness, 1.0e-9)
        self._land_mask = land_mask_bool
        self._surface_heat_capacity = np.maximum(heat_capacity_surface, 1.0e-9)
        self._atmosphere_heat_capacity = np.maximum(heat_capacity_atmosphere, 1.0e-9)
        self._topographic_elevation = np.maximum(topographic, 0.0)

        # Atmosphere reference height no longer configurable here; two-meter
        # temperature is computed via compute_two_meter_temperature.
        self._elevation_delta_to_surface = np.zeros_like(land_mask_bool, dtype=float)


    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def _wind_speed_10m(self, wind_speed_reference_m_s: np.ndarray | None) -> np.ndarray:
        if wind_speed_reference_m_s is None:
            return np.zeros_like(self._surface_heat_capacity)

        wind_speed = np.asarray(wind_speed_reference_m_s, dtype=float)
        if wind_speed.shape != self._surface_heat_capacity.shape:
            raise ValueError("Wind speed field must match the surface heat capacity shape")

        return log_law_map_wind_speed(
            wind_speed,
            height_ref_m=100,
            height_target_m=10,
            roughness_length_m=self._roughness,
        )

    def compute_tendencies(
        self,
        surface_temperature_K: np.ndarray,
        atmosphere_temperature_K: np.ndarray,
        humidity_q: np.ndarray,
        *,
        wind_speed_reference_m_s: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return surface and atmospheric tendencies from latent heat exchange."""

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

        
        pressure = pressure_from_temperature_elevation(atmosphere_temperature)
        wind_speed_10m = self._wind_speed_10m(wind_speed_reference_m_s)
        wind_abs = np.maximum(np.abs(wind_speed_10m), self._config.minimum_wind_speed_m_s)

        atmosphere_temperature_c = atmosphere_temperature - 273.15
        near_surface_air_c = compute_two_meter_temperature(
                atmosphere_temperature_c,
                surface_temperature - 273.15,
            )
        near_surface_air_K = np.maximum(near_surface_air_c + 273.15, 10.0)

        gas_constant = GAS_CONSTANT_J_KG_K
        rho = pressure / (gas_constant * near_surface_air_K)

        log_height_surface = 10.0
        roughness_momentum = self._roughness
        roughness_heat = np.maximum(roughness_momentum / 10.0, 1.0e-9)

        lm = np.log(
            np.maximum(log_height_surface / roughness_momentum, 1.0 + 1.0e-9)
        )
        lh = np.log(
            np.maximum(log_height_surface / roughness_heat, 1.0 + 1.0e-9)
        )
        ch_raw = (VON_KARMAN_CONSTANT**2) / (lm * lh)

        ch_land = np.clip(ch_raw, 1e-4, 2.0e-3)
        ch_ocean = np.clip(ch_raw, 3e-4, 3.0e-3)
        ch = np.where(self._land_mask, ch_land, ch_ocean)

        # Magnus formula requires temperature in Celsius
        surface_temperature_C = surface_temperature_K - 273.15
        e_sat = 6.112 * np.exp(17.67 * surface_temperature_C / (surface_temperature_C + 243.5))
        q_sat = (0.622 * e_sat) / (pressure - (1 - 0.622) * e_sat)

        humidity_q = np.minimum(humidity_q, q_sat)
        heat_flux = rho * 2.5e6 * ch * wind_abs * (q_sat - humidity_q)
        heat_flux = np.where(self._land_mask, 0, heat_flux)

        surface_tendency = -heat_flux / self._surface_heat_capacity
        atmosphere_tendency = heat_flux / self._atmosphere_heat_capacity

        return surface_tendency, atmosphere_tendency

