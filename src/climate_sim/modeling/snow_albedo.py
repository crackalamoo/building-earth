"""Snow albedo parameterisation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SnowAlbedoConfig:
    """Configuration for the diagnostic snow albedo scheme."""

    enabled: bool = True
    snow_albedo: float = 0.65
    freeze_temperature_c: float = -2.0
    melt_temperature_c: float = 1.0
    picard_iterations: int = 2


def apply_snow_albedo(
    base_albedo: np.ndarray,
    land_mask: np.ndarray,
    monthly_temperatures_K: np.ndarray,
    *,
    config: SnowAlbedoConfig,
) -> np.ndarray:
    """Return an albedo field with snow adjustments applied."""

    if not config.enabled:
        return base_albedo

    if config.freeze_temperature_c >= config.melt_temperature_c:
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

    denom = config.melt_temperature_c - config.freeze_temperature_c
    u = (config.melt_temperature_c - temperatures_c) / denom
    u_clamped = np.clip(u, 0.0, 1.0)
    snow_fraction = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)

    mean_snow_fraction = snow_fraction.mean(axis=0)
    mean_snow_fraction = np.clip(mean_snow_fraction, 0.0, 1.0)

    land_snow_fraction = np.where(land_mask, mean_snow_fraction, 0.0)

    adjusted = base_albedo + (config.snow_albedo - base_albedo) * land_snow_fraction
    return adjusted
