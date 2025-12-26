"""Latent heat exchange between the surface and atmosphere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.wind.wind import WindModel


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
        surface_heat_capacity_J_m2_K: np.ndarray,
        atmosphere_heat_capacity_J_m2_K: np.ndarray | float,
        wind_model: WindModel | None = None,
        config: LatentHeatExchangeConfig | None = None,
    ) -> None:
        self._config = config or LatentHeatExchangeConfig()

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
        self._wind_model = wind_model


    @property
    def enabled(self) -> bool:
        return self._config.enabled

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

        # Use advection model for atmospheric properties if available
        if self._wind_model is not None:
            pressure, rho, wind_speed_10m, ch = self._wind_model.compute_atmospheric_properties(
                surface_temperature,
                atmosphere_temperature,
                wind_speed_reference_m_s,
            )
        else:
            # Fallback: no wind, no exchange
            zeros = np.zeros_like(surface_temperature_K, dtype=float)
            return zeros, zeros

        wind_abs = np.maximum(np.abs(wind_speed_10m), self._config.minimum_wind_speed_m_s)

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
