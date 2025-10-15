"""Surface-atmosphere bulk aerodynamic coupling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Specific heat of dry air at constant pressure (J kg-1 K-1)
SPECIFIC_HEAT_AIR_J_PER_KG_K = 1004.0


@dataclass(frozen=True)
class BulkCouplingConfig:
    """Configuration for the surface–atmosphere bulk aerodynamic exchange."""

    enabled: bool = True
    rho_air: float = 1.225  # kg m-3, sea-level standard atmosphere
    c_p: float = SPECIFIC_HEAT_AIR_J_PER_KG_K
    C_H: float = 1.3e-3  # bulk transfer coefficient (dimensionless)
    U_ocean: float = 6.0  # m s-1 characteristic marine wind speed
    U_land: float = 3.0  # m s-1 characteristic continental wind speed
    atmosphere_heat_capacity: float = 1.0e7  # J m-2 K-1, consistent with RadiationConfig


def compute_bulk_flux(
    surface_K: np.ndarray,
    atmosphere_K: np.ndarray,
    ocean_mask: np.ndarray,
    config: BulkCouplingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the conductance and heat flux implied by the bulk aerodynamic formula."""

    if surface_K.shape != atmosphere_K.shape:
        raise ValueError("Surface and atmosphere fields must share the same shape")
    if ocean_mask.shape != surface_K.shape:
        raise ValueError("Ocean mask must match the temperature field shape")

    if not config.enabled:
        zeros = np.zeros_like(surface_K, dtype=float)
        return zeros, zeros

    U = np.where(ocean_mask, config.U_ocean, config.U_land)
    G = config.rho_air * config.c_p * config.C_H * U
    H = G * (surface_K - atmosphere_K)
    return G, H


def bulk_coupling_tendencies(
    surface_K: np.ndarray,
    atmosphere_K: np.ndarray,
    heat_capacity_field: np.ndarray,
    ocean_mask: np.ndarray,
    config: BulkCouplingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the surface and atmosphere temperature tendencies from bulk coupling."""

    if heat_capacity_field.shape != surface_K.shape:
        raise ValueError("Heat capacity field must match the temperature field shape")

    G, H = compute_bulk_flux(surface_K, atmosphere_K, ocean_mask, config)

    if not config.enabled:
        zeros = np.zeros_like(surface_K, dtype=float)
        return zeros, zeros

    if np.any(~np.isfinite(heat_capacity_field)):
        raise ValueError("Surface heat capacity field must contain finite values")

    surface_tendency = (-H) / heat_capacity_field

    C_a = config.atmosphere_heat_capacity
    if C_a <= 0.0 or not np.isfinite(C_a):
        raise ValueError("Atmosphere heat capacity must be a positive finite value")
    atmosphere_tendency = (+H) / C_a
    return surface_tendency, atmosphere_tendency


def bulk_coupling_jacobian(
    heat_capacity_field: np.ndarray,
    ocean_mask: np.ndarray,
    config: BulkCouplingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return linearisation terms for the bulk coupling tendencies."""

    if heat_capacity_field.shape != ocean_mask.shape:
        raise ValueError("Heat capacity field and ocean mask must share the same shape")

    if not config.enabled:
        zeros = np.zeros_like(heat_capacity_field, dtype=float)
        cross = np.zeros((2, 2) + heat_capacity_field.shape, dtype=float)
        return zeros, zeros, cross

    if np.any(~np.isfinite(heat_capacity_field)):
        raise ValueError("Surface heat capacity field must contain finite values")

    U = np.where(ocean_mask, config.U_ocean, config.U_land)
    G = config.rho_air * config.c_p * config.C_H * U
    C_s = heat_capacity_field
    C_a = config.atmosphere_heat_capacity
    if C_a <= 0.0 or not np.isfinite(C_a):
        raise ValueError("Atmosphere heat capacity must be a positive finite value")

    surface_diag = (-G) / C_s
    atmosphere_diag = (-G) / C_a

    cross = np.zeros((2, 2) + C_s.shape, dtype=float)
    cross[0, 1] = (+G) / C_s
    cross[1, 0] = (+G) / C_a
    return surface_diag, atmosphere_diag, cross
