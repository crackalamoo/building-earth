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


def apply_snow_albedo(
    base_albedo: np.ndarray,
    land_mask: np.ndarray,
    temperatures_K: np.ndarray,
    *,
    config: SnowAlbedoConfig,
) -> np.ndarray:
    """Return an albedo field with snow adjustments applied."""

    if not config.enabled:
        return base_albedo

    if config.freeze_temperature_c >= config.melt_temperature_c:
        raise ValueError("Snow melt temperature must exceed freeze temperature")

    if temperatures_K.ndim == 2:
        surface_temperatures_c = temperatures_K - 273.15
        mean_snow_fraction = _compute_snow_fraction(
            surface_temperatures_c, config=config
        )
    elif temperatures_K.ndim == 3:
        if temperatures_K.shape[0] == 2:
            surface_temperatures_c = temperatures_K[0] - 273.15
            mean_snow_fraction = _compute_snow_fraction(
                surface_temperatures_c, config=config
            )
        else:
            surface_temperatures_c = temperatures_K - 273.15
            snow_fraction = _compute_snow_fraction(
                surface_temperatures_c, config=config
            )
            mean_snow_fraction = snow_fraction.mean(axis=0)
    elif temperatures_K.ndim == 4 and temperatures_K.shape[1] >= 1:
        surface_temperatures_c = temperatures_K[:, 0] - 273.15
        snow_fraction = _compute_snow_fraction(
            surface_temperatures_c, config=config
        )
        mean_snow_fraction = snow_fraction.mean(axis=0)
    else:
        raise ValueError(
            "Expected temperature arrays of shape (lat, lon), (layer, lat, lon), "
            "(month, lat, lon), or (month, layer, lat, lon) for snow albedo calculation."
        )

    mean_snow_fraction = np.clip(mean_snow_fraction, 0.0, 1.0)

    land_snow_fraction = np.where(land_mask, mean_snow_fraction, 0.0)

    adjusted = base_albedo + (config.snow_albedo - base_albedo) * land_snow_fraction
    return adjusted


def _compute_snow_fraction(
    temperatures_c: np.ndarray, *, config: SnowAlbedoConfig
) -> np.ndarray:
    """Return the smooth step snow fraction for the provided temperatures in Celsius."""

    denom = config.melt_temperature_c - config.freeze_temperature_c
    u = (config.melt_temperature_c - temperatures_c) / denom
    u_clamped = np.clip(u, 0.0, 1.0)
    return u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)
