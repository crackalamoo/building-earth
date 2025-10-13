"""Radiative column model components."""

from __future__ import annotations

import numpy as np

STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4
OCEAN_HEAT_CAPACITY_M2 = 4.0e8  # J m-2 K-1, ~40 m mixed-layer ocean
LAND_HEAT_CAPACITY_M2 = 8.0e7  # J m-2 K-1, ~2 m soil
EMISSIVITY = 1.0
TEMPERATURE_FLOOR_K = 10.0

def radiative_balance_rhs(
    temperature_K: np.ndarray,
    insolation_W_m2: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    emissivity: float,
) -> np.ndarray:
    """Column energy-balance tendency C dT/dt = S - εσT⁴."""
    temperature = np.maximum(temperature_K, TEMPERATURE_FLOOR_K)
    emitted = emissivity * STEFAN_BOLTZMANN * np.power(temperature, 4)
    return (insolation_W_m2 - emitted) / heat_capacity_field


def radiative_balance_rhs_temperature_derivative(
    temperature_K: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    emissivity: float,
) -> np.ndarray:
    """Partial derivative d(dT/dt)/dT."""
    temperature = np.maximum(temperature_K, TEMPERATURE_FLOOR_K)
    return (-4.0 * emissivity * STEFAN_BOLTZMANN * np.power(temperature, 3)) / heat_capacity_field


def radiative_equilibrium_initial_guess(
    monthly_insolation: np.ndarray,
    *,
    emissivity: float,
) -> np.ndarray:
    """Initial temperature guess via local radiative equilibrium."""
    annual_mean_insolation = monthly_insolation.mean(axis=0)
    return np.power(
        np.maximum(annual_mean_insolation, 1e-6) / (emissivity * STEFAN_BOLTZMANN),
        0.25,
    )
