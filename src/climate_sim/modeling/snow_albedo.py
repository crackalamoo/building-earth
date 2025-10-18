"""Snow albedo parameterisation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARCTIC_CIRCLE_LATITUDE_DEG = 66.5

@dataclass(frozen=True)
class SnowAlbedoConfig:
    """Configuration for the diagnostic snow albedo scheme."""

    enabled: bool = True
    snow_albedo: float = 0.65
    freeze_temperature_c: float = -2.0
    melt_temperature_c: float = 1.0
    picard_iterations: int = 2


class AlbedoModel:
    def __init__(self, lat2d: np.ndarray, lon2d: np.ndarray, land_mask: np.ndarray, config: SnowAlbedoConfig | None = None) -> None:
        """Initialize the snow albedo model.

        Args:
            land_mask: Boolean array of shape (lat, lon) indicating land grid cells.
        """
        self.lat2d = lat2d
        self.lon2d = lon2d
        self.land_mask = land_mask
        if config is not None:
            self.config = config
        else:
            self.config = SnowAlbedoConfig()

    def guess_albedo_field(self) -> np.ndarray:
        if self.config.enabled:
            return 0.3 * np.ones_like(self.lat2d)
        else:
            albedo = 0.25 * np.ones_like(self.lat2d)
            in_arctic = np.abs(self.lat2d) >= ARCTIC_CIRCLE_LATITUDE_DEG
            albedo = np.where(in_arctic, self.config.snow_albedo, albedo)
            return albedo

    def apply_snow_albedo(
        self,
        base_albedo: np.ndarray,
        monthly_temperatures_K: np.ndarray,
    ) -> np.ndarray:
        """Return an albedo field with snow adjustments applied."""

        if not self.config.enabled:
            return base_albedo

        if self.config.freeze_temperature_c >= self.config.melt_temperature_c:
            raise ValueError("Snow melt temperature must exceed freeze temperature")

        if monthly_temperatures_K.ndim == 3:
            surface_temperatures = monthly_temperatures_K
        elif monthly_temperatures_K.ndim == 4:
            surface_temperatures = monthly_temperatures_K[:, 0]
        else:
            raise ValueError(
                "Expected 3-D (month, lat, lon) or 4-D (month, layer, lat, lon) "
                "temperature arrays for snow albedo calculation."
            )

        temperatures_c = surface_temperatures - 273.15

        denom = self.config.melt_temperature_c - self.config.freeze_temperature_c
        u = (self.config.melt_temperature_c - temperatures_c) / denom
        u_clamped = np.clip(u, 0.0, 1.0)
        snow_fraction = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)

        mean_snow_fraction = snow_fraction.mean(axis=0)
        mean_snow_fraction = np.clip(mean_snow_fraction, 0.0, 1.0)

        land_snow_fraction = np.where(self.land_mask, mean_snow_fraction, 0.0)

        adjusted = base_albedo + (self.config.snow_albedo - base_albedo) * land_snow_fraction
        return adjusted
