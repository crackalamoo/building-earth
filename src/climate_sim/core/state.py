"""Core state types and utilities for the climate model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelState:
    """State variables for the climate model."""
    temperature: np.ndarray
    albedo_field: np.ndarray
    wind_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    humidity_field: np.ndarray | None = None


def select_wind_temperature(temperature: np.ndarray) -> np.ndarray:
    """Return the temperature field to use when computing wind diagnostics."""
    if temperature.ndim == 2:
        return temperature
    if temperature.ndim == 3:
        if temperature.shape[0] == 1:
            return temperature[0]
        if temperature.shape[0] >= 2:
            return temperature[1]
    raise ValueError("Unsupported temperature field shape for wind calculation")


def select_humidity_temperature(temperature: np.ndarray) -> np.ndarray:
    """Return the temperature field to use when computing humidity diagnostics."""
    if temperature.ndim == 2:
        return temperature
    if temperature.ndim == 3:
        # Always use surface temperature (layer 0) for humidity
        return temperature[0]
    raise ValueError("Unsupported temperature field shape for humidity calculation")
