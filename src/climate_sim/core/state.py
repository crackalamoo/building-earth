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
    boundary_layer_wind_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def select_wind_temperature(temperature: np.ndarray) -> np.ndarray:
    """Return the temperature field to use when computing wind diagnostics."""
    nlayers = temperature.shape[0]
    if nlayers == 1:
        return temperature[0]
    if nlayers == 2:
        return temperature[1]  # Atmosphere layer
    if nlayers == 3:
        return temperature[2]  # Free atmosphere layer (not boundary layer)
    raise ValueError(f"Unsupported number of layers: {nlayers}")


def select_humidity_temperature(temperature: np.ndarray) -> np.ndarray:
    """Return the temperature field to use when computing humidity diagnostics."""
    # Always use surface temperature (layer 0) for humidity
    return temperature[0]
