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
from climate_sim.physics.atmosphere.pressure import (
    LAT_SUBTROPICS_BASE,
    SUBTROPICS_ITCZ_COUPLING,
    SIGMA_SUBTROPICS,
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

    # Hadley subsidence: large-scale descent at subtropical highs
    hadley_descent_velocity_m_s: float = 0.0003

    # How dry the descending air is relative to boundary layer.
    upper_troposphere_q_fraction: float = 0.20

    # Background BL-atmosphere mixing timescale (seconds).
    # Represents subsidence, entrainment, and turbulent exchange that
    # returns latent heat from the free atmosphere back to the BL.
    tau_bl_atm_mixing_s: float = 20.0 * 86400.0  # 20 days (Cronin 2013)


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
    """Compute BL and atmosphere heating from vertical motion.

    Uses potential temperature conservation for adiabatic vertical motion.

    T_bl is sea-level equivalent temperature (≈ potential temperature at P₀).
    T_atm is actual temperature at ~500 hPa.

    Physics:
    - Subsidence (div > 0): Air descends from above into the BL. Descending air
      conserves θ; if θ_exchange > θ_bl, subsidence warms the BL. The atmosphere
      cools by the same amount (energy conserving). This is a true inter-layer
      heat exchange.

    - Ascent (div < 0): Air rises from BL into atmosphere. The BL is replenished
      by horizontal convergence (advection) at similar θ, so the BL temperature
      is unchanged. However, the atmosphere DOES receive air at θ_bl, which
      affects its temperature. The latent heat from condensation is handled
      separately by the precipitation module.

    The exchange happens at the BL top (~850 hPa), not at 500 hPa where T_atm
    is defined. We interpolate θ in log-pressure space between θ_bl (surface)
    and θ_atm (500 hPa) to get θ at the exchange level.
    """
    h_bl = BOUNDARY_LAYER_HEIGHT_M
    w = divergence * h_bl  # m/s, positive = downward (subsidence)

    rho = 1.0  # kg/m³
    cp = HEAT_CAPACITY_AIR_J_KG_K  # J/kg/K

    # Pressure levels
    P0 = 1013.25  # hPa, sea-level reference
    P_ATM = 500.0  # hPa, atmosphere layer
    P_EXCHANGE = 850.0  # hPa, exchange level just above BL
    KAPPA = 0.286  # R/cp

    # Potential temperatures at known levels
    theta_bl = T_bl  # T_bl is sea-level equivalent ≈ θ
    theta_atm = T_atm * (P0 / P_ATM) ** KAPPA

    # Interpolate θ to exchange level (850 hPa) in log-pressure space
    # f = fraction of the way from surface to 500 hPa
    ln_P0 = np.log(P0)
    ln_P_atm = np.log(P_ATM)
    ln_P_exchange = np.log(P_EXCHANGE)
    f = (ln_P0 - ln_P_exchange) / (ln_P0 - ln_P_atm)  # ≈ 0.25

    theta_exchange = theta_bl + f * (theta_atm - theta_bl)

    # Split into subsidence and ascent
    w_subsidence = np.maximum(w, 0)  # positive where sinking
    w_ascent = np.minimum(w, 0)      # negative where rising

    # SUBSIDENCE (w > 0): Energy-conserving exchange between atm and BL
    # Air at θ_exchange descends into BL, warming it
    Q_subsidence = rho * cp * w_subsidence * (theta_exchange - theta_bl)

    # ASCENT (w < 0): BL unchanged (replacement from advection), but
    # atmosphere receives air at θ_bl. Since θ_bl < θ_exchange typically,
    # this "cools" the atmosphere (adds relatively cool air).
    # Q_ascent is the heat flux INTO atmosphere from rising BL air
    # w_ascent is negative, and we want positive flux when θ_bl < θ_exchange
    Q_ascent = rho * cp * w_ascent * (theta_exchange - theta_bl)
    # When w < 0 and θ_exchange > θ_bl: Q_ascent < 0 (atmosphere cools)

    # BL tendency: only from subsidence (ascent replacement via advection)
    dT_bl = Q_subsidence / C_bl

    # Atmosphere tendency: from both subsidence (loses heat) and ascent (gains cool air)
    dT_atm = -Q_subsidence / C_atm + Q_ascent / C_atm

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


def compute_hadley_subsidence_velocity(
    lat_rad: np.ndarray,
    itcz_rad: np.ndarray,
    peak_velocity_m_s: float = 0.003,
) -> np.ndarray:
    """Compute downward vertical velocity from Hadley cell overturning.

    Returns vertical velocity (m/s), positive = descent.
    """
    lat_subtrop_north = LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad
    lat_subtrop_south = -LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad

    w = peak_velocity_m_s * (
        np.exp(-((lat_rad - lat_subtrop_south) / SIGMA_SUBTROPICS) ** 2)
        + np.exp(-((lat_rad - lat_subtrop_north) / SIGMA_SUBTROPICS) ** 2)
    )

    return w


def compute_hadley_subsidence_drying(
    w_descent: np.ndarray,
    humidity_field: np.ndarray,
    upper_troposphere_q_fraction: float = UPPER_TROPOSPHERE_Q_FRACTION,
    boundary_layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute humidity tendency from Hadley subsidence mixing dry air into BL.

    dq/dt = (w / h_BL) * (q_upper - q_BL)
    """
    q_upper = humidity_field * upper_troposphere_q_fraction
    delta_q = q_upper - humidity_field  # Always negative
    mixing_rate = w_descent / boundary_layer_height_m
    return mixing_rate * delta_q


def hadley_subsidence_drying_jacobian(
    w_descent: np.ndarray,
    upper_troposphere_q_fraction: float = UPPER_TROPOSPHERE_Q_FRACTION,
    boundary_layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Diagonal of the humidity Jacobian from Hadley subsidence drying.

    d(dq/dt)/dq = (w / h_BL) * (f_upper - 1) = -(1 - f_upper) * w / h_BL
    Always negative (stabilizing).
    """
    return w_descent / boundary_layer_height_m * (upper_troposphere_q_fraction - 1.0)

# Potential temperature factor: θ_atm = T_atm × (P0/P_ATM)^κ
_P0 = 1013.25  # hPa
_P_ATM = 500.0  # hPa
_KAPPA = 0.286  # R/cp
_ALPHA = (_P0 / _P_ATM) ** _KAPPA  # ≈ 1.219


def compute_bl_atm_mixing_tendencies(
    T_bl: np.ndarray,
    T_atm: np.ndarray,
    tau_s: float,
    C_bl: float = BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    C_atm: float = ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Background BL-atmosphere heat exchange via subsidence and mixing.

    The free atmosphere has high potential temperature from latent heating.
    Air descending adiabatically arrives warmer than the BL.  This term
    relaxes the BL toward the potential temperature of the free atmosphere,
    closing the energy loop: surface → evaporation → condensation aloft →
    subsidence warming back to BL.

    Energy-conserving: C_bl dT_bl + C_atm dT_atm = 0.
    """
    theta_atm = T_atm * _ALPHA  # Potential temperature of free atm at surface
    heat_flux = C_bl * (theta_atm - T_bl) / tau_s  # W/m²

    dT_bl = heat_flux / C_bl      # = (θ_atm - T_bl) / τ
    dT_atm = -heat_flux / C_atm   # Energy conservation

    return dT_bl, dT_atm
