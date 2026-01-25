"""Core state types and utilities for the climate model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.clouds import CloudPrecipOutput


@dataclass
class ModelState:
    """State variables for the climate model."""
    temperature: np.ndarray
    albedo_field: np.ndarray
    wind_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    humidity_field: np.ndarray | None = None
    boundary_layer_wind_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ocean_current_field: tuple[np.ndarray, np.ndarray] | None = None  # (u, v) in m/s
    ocean_current_psi: np.ndarray | None = None  # Streamfunction in Sv
    precipitation_field: np.ndarray | None = None  # Precipitation rate in kg/m²/s
    soil_moisture: np.ndarray | None = None  # Soil moisture fraction (0-1), only for land
    convective_cloud_frac: np.ndarray | None = None  # Convective cloud fraction (0-1)
    stratiform_cloud_frac: np.ndarray | None = None  # Stratiform cloud fraction (0-1)
    marine_sc_cloud_frac: np.ndarray | None = None  # Marine stratocumulus cloud fraction (0-1)
    high_cloud_frac: np.ndarray | None = None  # High cloud (cirrus/anvil) fraction (0-1)
    # Frozen cloud output for Newton solver Jacobian consistency
    # When set, RHS uses this instead of recomputing clouds from temperature
    cloud_output: CloudPrecipOutput | None = None


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
