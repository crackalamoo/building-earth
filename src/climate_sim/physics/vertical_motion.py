"""Vertical motion physics: adiabatic heating/cooling and humidity transport.

Physics:
- Subsidence warming: Air descending in the subtropics warms adiabatically
  at the dry lapse rate (Γ_dry = 9.8 K/km) since descending air is unsaturated.

- Ascent cooling: Rising air at the ITCZ cools at the moist lapse rate
  (Γ_moist ≈ 6.5 K/km) because condensation releases latent heat, partially
  offsetting the adiabatic cooling. This implicitly accounts for latent heat
  release - no separate precipitation heating term needed.

- Humidity transport:
  - Subsidence: Brings dry upper-tropospheric air down, reducing surface humidity.
    Upper troposphere has ~20% of boundary layer humidity due to precipitation
    removing moisture as air ascends.
  - Ascent: Moisture is removed via precipitation (handled separately).

Vertical velocity is inferred from horizontal divergence via mass continuity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.data.constants import (
    ATMOSPHERE_LAYER_HEIGHT_M,
    ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
    BOUNDARY_LAYER_HEIGHT_M,
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    STANDARD_LAPSE_RATE_K_PER_M,
    HEAT_CAPACITY_AIR_J_KG_K,
)


# Physical constants
GAMMA_DRY = 0.0098  # K/m, dry adiabatic lapse rate (for subsidence)
GAMMA_MOIST = STANDARD_LAPSE_RATE_K_PER_M  # ~0.0065 K/m, moist lapse rate (for ascent)

# Upper troposphere humidity fraction relative to boundary layer
# Air rising through the troposphere loses most moisture via precipitation
# By the time it reaches the upper troposphere (~300 hPa), q is ~20% of surface value
UPPER_TROPOSPHERE_Q_FRACTION = 0.20


@dataclass(frozen=True)
class VerticalMotionConfig:
    """Configuration for vertical motion physics."""

    enabled: bool = True


def compute_vertical_motion_heating(
    divergence: np.ndarray,
    h_atm: float = ATMOSPHERE_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute atmosphere heating/cooling from vertical motion.

    When surface divergence > 0, air descends and warms at the dry lapse rate.
    When surface convergence < 0, air ascends and cools at the moist lapse rate
    (latent heat release partially offsets the cooling).
    """
    # Surface divergence implies vertical motion through mass continuity.
    # We scale divergence to a realistic vertical velocity using an
    # effective depth scale.
    # Empirically: div ~ 1e-5 /s in subtropics
    # Target: ~1 K/day = 1e-5 K/s warming
    effective_depth = 100.0  # m

    # Vertical velocity: positive = descent, negative = ascent
    w = divergence * effective_depth  # m/s

    # Use dry lapse rate for descent (unsaturated), moist for ascent (saturated)
    # dT/dt = w * Γ (positive w and positive Γ = warming)
    heating = np.where(
        w > 0,
        w * GAMMA_DRY,   # Descent: warm at dry rate
        w * GAMMA_MOIST  # Ascent: cool at moist rate (less cooling due to latent heat)
    )

    return heating


def compute_vertical_motion_tendency(
    divergence: np.ndarray,
    h_atm: float = ATMOSPHERE_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute atmosphere heating from vertical motion.

    Uses dry lapse rate for descent (subsidence warming in subtropics)
    and moist lapse rate for ascent (reduced cooling at ITCZ due to
    latent heat release from condensation).
    """
    return compute_vertical_motion_heating(divergence, h_atm)


def compute_vertical_motion_tendencies(
    divergence: np.ndarray,
    T_bl: np.ndarray,
    T_atm: np.ndarray,
    C_bl: float = BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    C_atm: float = ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BL and atmosphere heating from vertical motion (energy-conserving).

    Vertical motion exchanges air between boundary layer and free atmosphere.
    This function computes the heat flux between layers and returns tendencies
    for both layers that conserve energy.

    Physics:
    - Subsidence (div > 0): Air descends from atmosphere into BL.
      Atmosphere loses enthalpy, BL gains enthalpy.
    - Ascent (div < 0): Air rises from BL into atmosphere.
      BL loses enthalpy, atmosphere gains enthalpy (minus latent heat released).

    The heat flux at the interface is:
        Q = rho * cp * w * delta_T
    where delta_T is the temperature difference accounting for adiabatic change.
    """
    # Vertical velocity from divergence (positive = downward)
    # Scale divergence to get realistic w values
    # div ~ 1e-5 /s in subtropics → target w ~ 0.001 m/s
    effective_depth = 100.0  # m
    w = divergence * effective_depth  # m/s

    # Air density at interface (approximate)
    rho = 1.0  # kg/m³ (lower than surface due to altitude)
    cp = HEAT_CAPACITY_AIR_J_KG_K  # J/kg/K

    # Energy-conserving heat flux approach:
    # Compute heat flux Q (W/m²) between layers based on mass flux and
    # temperature difference, accounting for adiabatic temperature changes.
    #
    # Subsidence (w > 0): Air descends from atmosphere into BL, warming at
    # the dry lapse rate. Heat flux Q_sub = rho * cp * w * (T_atm - T_bl + Γ_dry * h_bl)
    #
    # Ascent (w < 0): Air rises from BL into atmosphere, cooling at the moist
    # lapse rate. Heat flux Q_asc = rho * cp * |w| * (T_bl - T_atm - Γ_moist * h_bl)

    h_bl = BOUNDARY_LAYER_HEIGHT_M

    # Subsidence case (w > 0): heat flux from atmosphere to BL
    w_down = np.maximum(w, 0)
    Q_subsidence = rho * cp * w_down * (T_atm - T_bl + GAMMA_DRY * h_bl)
    # The Γ*h term accounts for adiabatic warming during descent

    # Ascent case (w < 0, so -w > 0 is upward velocity)
    w_up = np.maximum(-w, 0)
    # Heat flux from BL to atm during ascent:
    # Air leaves BL at T_bl, arrives at atm after cooling
    Q_ascent = rho * cp * w_up * (T_bl - T_atm - GAMMA_MOIST * h_bl)
    # Note: GAMMA_MOIST < GAMMA_DRY because latent heat release partially offsets cooling

    # Total heat flux from atmosphere to BL (positive = downward)
    Q_total = Q_subsidence - Q_ascent  # Q_ascent is BL→atm, so subtract

    # Temperature tendencies (K/s)
    dT_bl = Q_total / C_bl
    dT_atm = -Q_total / C_atm  # Opposite sign to conserve energy

    return dT_bl, dT_atm


def compute_subsidence_drying(
    divergence: np.ndarray,
    humidity_field: np.ndarray,
) -> np.ndarray:
    """Compute humidity reduction from subsidence bringing dry air down.

    In regions of divergence (subtropics), air descends from the upper
    troposphere where it is much drier (moisture precipitated out during
    ascent at ITCZ). This mixes dry air into the boundary layer, reducing
    humidity.

    The tendency is proportional to:
    - Divergence (descent rate)
    - Humidity difference between BL and upper troposphere

    Parameters
    ----------
    divergence : np.ndarray
        Horizontal divergence (1/s). Positive = subsidence.
    humidity_field : np.ndarray
        Current specific humidity (kg/kg) in the boundary layer.

    Returns
    -------
    np.ndarray
        Humidity tendency (kg/kg/s). Negative where subsidence occurs.
    """
    # Upper troposphere humidity (dry air coming down)
    q_upper = humidity_field * UPPER_TROPOSPHERE_Q_FRACTION

    # Humidity difference that gets mixed in during subsidence
    delta_q = q_upper - humidity_field  # Always negative (upper is drier)

    # Same effective depth scaling as temperature
    effective_depth = 100.0  # m

    # Descent rate (m/s), only where divergence > 0
    w = np.maximum(divergence, 0) * effective_depth

    # Mixing timescale: how fast subsidence replaces BL air
    # w / h_BL gives the fraction of BL replaced per second
    h_bl = 1000.0  # boundary layer height (m)
    mixing_rate = w / h_bl  # 1/s

    # Humidity tendency: rate of mixing * humidity difference
    dq_dt = mixing_rate * delta_q

    return dq_dt
