"""Snow albedo parameterisation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SnowAlbedoConfig:
    """Configuration for the diagnostic snow albedo scheme."""

    enabled: bool = False
    snow_albedo: float = 0.65
    snow_temperature_threshold_c: float = 0.0
    picard_iterations: int = 2

    def threshold_kelvin(self) -> float:
        """Return the snow formation temperature threshold in Kelvin."""

        return self.snow_temperature_threshold_c + 273.15


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

    if monthly_temperatures_K.ndim == 3:
        surface_temperatures = monthly_temperatures_K
    elif monthly_temperatures_K.ndim == 4:
        surface_temperatures = monthly_temperatures_K[:, 0]
    else:
        raise ValueError(
            "Expected 3-D (month, lat, lon) or 4-D (month, layer, lat, lon) "
            "temperature arrays for snow albedo calculation."
        )

    below_freezing = surface_temperatures < config.threshold_kelvin()
    snow_cells = np.any(below_freezing, axis=0)

    adjusted = np.where(
        land_mask & snow_cells,
        config.snow_albedo,
        base_albedo,
    )
    return adjusted
