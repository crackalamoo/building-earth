"""Radiative column model components."""

from __future__ import annotations

import numpy as np
from typing import Optional

STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4
EMISSIVITY_SFC = 1.0
EMISSIVITY_ATM = 0.77
DEFAULT_ALBEDO = 0.3  # Typical Earth value
TEMPERATURE_FLOOR_K = 10.0

def radiative_balance_rhs(
    temperature_K: np.ndarray,
    insolation_W_m2: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    emissivity_sfc: Optional[float] = EMISSIVITY_SFC,
    emissivity_atm: Optional[float] = EMISSIVITY_ATM,
    albedo_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Column energy-balance tendency C dT/dt = S - εσT⁴."""
    emissivity_sfc = emissivity_sfc if emissivity_sfc is not None else EMISSIVITY_SFC
    emissivity_atm = emissivity_atm if emissivity_atm is not None else EMISSIVITY_ATM
    temperature = np.maximum(temperature_K, TEMPERATURE_FLOOR_K)
    emissivity_eff = emissivity_sfc * (1 - 0.5 * emissivity_atm)
    emitted = emissivity_eff * STEFAN_BOLTZMANN * np.power(temperature, 4)
    albedo = albedo_field if albedo_field is not None else DEFAULT_ALBEDO
    return (insolation_W_m2 * (1 - albedo) - emitted) / heat_capacity_field


def radiative_balance_rhs_temperature_derivative(
    temperature_K: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    emissivity_sfc: float,
    emissivity_atm: float,
) -> np.ndarray:
    """Partial derivative d(dT/dt)/dT."""
    emissivity_sfc = emissivity_sfc if emissivity_sfc is not None else EMISSIVITY_SFC
    emissivity_atm = emissivity_atm if emissivity_atm is not None else EMISSIVITY_ATM
    temperature = np.maximum(temperature_K, TEMPERATURE_FLOOR_K)
    emissivity_eff = emissivity_sfc * (1 - 0.5 * emissivity_atm)
    return (-4.0 * emissivity_eff * STEFAN_BOLTZMANN * np.power(temperature, 3)) / heat_capacity_field


def radiative_equilibrium_initial_guess(
    monthly_insolation: np.ndarray,
    *,
    emissivity_sfc: Optional[float] = EMISSIVITY_SFC,
    emissivity_atm: Optional[float] = EMISSIVITY_ATM,
) -> np.ndarray:
    """Initial temperature guess via local radiative equilibrium."""
    emissivity_sfc = emissivity_sfc if emissivity_sfc is not None else EMISSIVITY_SFC
    emissivity_atm = emissivity_atm if emissivity_atm is not None else EMISSIVITY_ATM
    annual_mean_insolation = monthly_insolation.mean(axis=0)
    emissivity_eff = emissivity_sfc * (1 - 0.5 * emissivity_atm)
    return np.power(
        np.maximum(annual_mean_insolation, 1e-6) / (emissivity_eff * STEFAN_BOLTZMANN),
        0.25,
    )
