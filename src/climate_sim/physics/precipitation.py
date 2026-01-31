"""Precipitation physics.

This module provides all precipitation computations:
- Marine stratocumulus drizzle (cloud-coupled)
- Convective precipitation (cloud-coupled)
- Stratiform precipitation (cloud-coupled)
- Jacobian calculations for the implicit solver
"""

from __future__ import annotations

import numpy as np

from climate_sim.data.constants import (
    GRAVITY_M_S2,
    HEAT_CAPACITY_AIR_J_KG_K,
    GAS_CONSTANT_WATER_VAPOR_J_KG_K,
    LATENT_HEAT_VAPORIZATION_J_KG,
    STANDARD_LAPSE_RATE_K_PER_M,
)
from climate_sim.physics.humidity import compute_saturation_specific_humidity

# Aliases for readability
GRAVITY = GRAVITY_M_S2
SPECIFIC_HEAT_AIR = HEAT_CAPACITY_AIR_J_KG_K

# Air density at surface (kg/m³)
RHO_AIR = 1.2

# Precipitation efficiency - fraction of moisture flux converted to precipitation
# Stratiform: ~20% (steady, widespread)
# Convective: ~30% (intense, localized)
STRATIFORM_PRECIP_EFFICIENCY = 0.20
CONVECTIVE_PRECIP_EFFICIENCY = 0.30

# Grid-mean convective vertical velocity contribution (m/s)
# This represents the effective grid-mean vertical motion from convection.
# In-cloud updrafts are 1-5 m/s, but convection covers only ~5-10% of area,
# so grid-mean contribution is ~0.05-0.5 m/s.
# With 20% cloud fraction and 0.3 efficiency, w_eff=0.1 gives ~5 mm/day tropical max.
CONVECTIVE_UPDRAFT_VELOCITY = 0.10  # m/s (grid-mean effective)

# Column mass for converting kg/kg to kg/m² (used in marine Sc drizzle)
COLUMN_MASS_KG_M2 = 5000.0

# Marine Sc drizzle timescale (slow autoconversion in stable layer)
MARINE_SC_DRIZZLE_TIMESCALE = 14 * 86400  # 2 weeks - very slow

# Cloud top heights for precipitation calculations (must match clouds.py)
MARINE_SC_CLOUD_TOP_HEIGHT_M = 1000.0  # Marine Sc tops at ~1 km (below inversion)


def compute_moist_adiabatic_lapse_rate(T_K: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute moist adiabatic lapse rate (K/m).

    The moist adiabatic lapse rate is less than the dry rate (9.8 K/km)
    because latent heat is released during condensation:

    Γ_m = Γ_d × (1 + L_v × q / (R_d × T)) / (1 + L_v² × q / (c_p × R_v × T²))

    Typical values:
    - Dry: 9.8 K/km
    - Moist tropical (T=300K, q=20g/kg): ~4-5 K/km
    - Moist midlatitude (T=280K, q=10g/kg): ~6 K/km
    """
    # Constants
    g = GRAVITY  # 9.81 m/s²
    cp = SPECIFIC_HEAT_AIR  # 1004 J/kg/K
    Lv = LATENT_HEAT_VAPORIZATION_J_KG  # 2.5e6 J/kg
    Rd = 287.0  # J/kg/K - gas constant for dry air
    Rv = GAS_CONSTANT_WATER_VAPOR_J_KG_K  # 461.0 J/(kg·K)

    # Dry adiabatic lapse rate
    gamma_d = g / cp  # ~0.00976 K/m = 9.76 K/km

    # Moist adiabatic lapse rate (simplified form)
    # Using mixing ratio r ≈ q for small q
    numerator = 1 + (Lv * q) / (Rd * T_K)
    denominator = 1 + (Lv**2 * q) / (cp * Rv * T_K**2)

    gamma_m = gamma_d * numerator / denominator

    return gamma_m


def compute_marine_sc_precipitation(
    marine_sc_frac: np.ndarray,
    q: np.ndarray,
    T_bl_K: np.ndarray,
    drizzle_timescale: float = MARINE_SC_DRIZZLE_TIMESCALE,
) -> np.ndarray:
    """Compute marine stratocumulus precipitation (drizzle).

    Marine Sc produce light drizzle, not heavy rain. Unlike stratiform and
    convective precipitation which use moisture flux, marine Sc drizzle uses
    slow autoconversion because these clouds form in SUBSIDING air (no w > 0).

    The drizzle rate is the excess moisture (above saturation at cloud level)
    slowly converting to precipitation via collision-coalescence.

    Parameters
    ----------
    marine_sc_frac : np.ndarray
        Marine Sc cloud fraction (0-1).
    q : np.ndarray
        Specific humidity (kg/kg).
    T_bl_K : np.ndarray
        Boundary layer temperature (K).
    drizzle_timescale : float
        Autoconversion timescale (s). Default 2 weeks (very slow).

    Returns
    -------
    np.ndarray
        Drizzle rate (kg/m²/s). Much lighter than stratiform precip.

    Notes
    -----
    Marine Sc form in subsidence zones where w < 0, so the moisture flux
    approach doesn't work. Instead, drizzle is modeled as slow autoconversion
    of cloud water. Typical drizzle rates are 0.1-0.5 mm/day.
    """
    # Temperature at cloud level (~1 km)
    T_cloud = T_bl_K - STANDARD_LAPSE_RATE_K_PER_M * MARINE_SC_CLOUD_TOP_HEIGHT_M

    # Saturation specific humidity at cloud level
    q_sat_cloud = compute_saturation_specific_humidity(T_cloud)

    # Excess moisture above saturation (cloud water content proxy)
    excess_q = np.maximum(q - q_sat_cloud, 0)

    # Drizzle: slow autoconversion of excess moisture
    P_drizzle = (
        marine_sc_frac
        * excess_q
        * COLUMN_MASS_KG_M2
        / drizzle_timescale
    )

    return P_drizzle


def compute_convective_precipitation(
    convective_frac: np.ndarray,
    q: np.ndarray,
    w_updraft: float = CONVECTIVE_UPDRAFT_VELOCITY,
    efficiency: float = CONVECTIVE_PRECIP_EFFICIENCY,
    rho: float = RHO_AIR,
) -> np.ndarray:
    """Compute convective precipitation from moisture flux in convective updrafts.

    Convective precipitation = cloud_fraction × efficiency × w_updraft × q × ρ

    The cloud fraction already encodes where convection is active (via LTS,
    vertical velocity, and humidity in the cloud scheme). Once convective
    clouds exist, precipitation scales with moisture flux through the updrafts.

    Parameters
    ----------
    convective_frac : np.ndarray
        Convective cloud fraction (0-1). Already encodes instability/rising.
    q : np.ndarray
        Specific humidity (kg/kg).
    w_updraft : float
        Typical convective updraft velocity (m/s). Default 2 m/s.
    efficiency : float
        Precipitation efficiency (0-1). Default 0.30.
    rho : float
        Air density (kg/m³). Default 1.2.

    Returns
    -------
    np.ndarray
        Convective precipitation rate (kg/m²/s).

    Notes
    -----
    Physical interpretation:
    - Moisture flux = ρ × w × q (kg/m²/s of water vapor moving upward)
    - Precipitation = efficiency × moisture_flux × cloud_fraction
    - cloud_fraction accounts for fractional area of convection in grid cell

    Example: conv_frac=0.1, q=0.015, w=2 m/s, eff=0.3, ρ=1.2
    → P = 0.1 × 0.3 × 2 × 0.015 × 1.2 = 1.1e-3 kg/m²/s ≈ 1 mm/day
    """
    # Moisture flux through convective updrafts
    P_convective = convective_frac * efficiency * w_updraft * q * rho

    return P_convective


def compute_stratiform_precipitation(
    stratiform_frac: np.ndarray,
    q: np.ndarray,
    w_largescale: np.ndarray,
    efficiency: float = STRATIFORM_PRECIP_EFFICIENCY,
    rho: float = RHO_AIR,
) -> np.ndarray:
    """Compute stratiform precipitation from large-scale moisture flux.

    Stratiform precipitation = cloud_fraction × efficiency × max(w, 0) × q × ρ

    Precipitation occurs where there are stratiform clouds AND large-scale
    rising motion (w > 0). The rate scales with the moisture flux through
    the cloud layer.

    Parameters
    ----------
    stratiform_frac : np.ndarray
        Stratiform cloud fraction (0-1).
    q : np.ndarray
        Specific humidity (kg/kg).
    w_largescale : np.ndarray
        Large-scale vertical velocity (m/s). Positive = rising.
    efficiency : float
        Precipitation efficiency (0-1). Default 0.20.
    rho : float
        Air density (kg/m³). Default 1.2.

    Returns
    -------
    np.ndarray
        Stratiform precipitation rate (kg/m²/s).

    Notes
    -----
    Physical interpretation:
    - Only rising air (w > 0) produces precipitation
    - Moisture flux = ρ × w × q (kg/m²/s of water vapor lifted)
    - Precipitation = efficiency × moisture_flux × cloud_fraction

    The large-scale w includes both:
    - Pressure-driven ascent (ITCZ convergence, frontal lifting)
    - Warm advection at fronts (from temperature gradients)

    Example: strat_frac=0.3, q=0.010, w=0.02 m/s, eff=0.2, ρ=1.2
    → P = 0.3 × 0.2 × 0.02 × 0.010 × 1.2 = 1.4e-5 kg/m²/s ≈ 1.2 mm/day
    """
    # Only precipitate where air is rising
    w_rising = np.maximum(w_largescale, 0.0)

    # Moisture flux through stratiform clouds
    P_stratiform = stratiform_frac * efficiency * w_rising * q * rho

    return P_stratiform


def compute_static_stability(
    T_bl_K: np.ndarray,
    T_atm_K: np.ndarray,
    delta_z: float = 5000.0,
) -> np.ndarray:
    """Compute Brunt-Vaisala frequency N from vertical temperature profile.

    N^2 = (g/T) * (dT/dz + g/cp). Returns N (1/s), clipped to positive values.
    """
    g = GRAVITY
    cp = SPECIFIC_HEAT_AIR

    # Mean temperature for the layer
    T_mean = 0.5 * (T_bl_K + T_atm_K)

    # Actual lapse rate (K/m) - positive if T decreases with height
    dT_dz = (T_atm_K - T_bl_K) / delta_z

    # N² = (g/T) × (dT/dz + g/cp)
    # The g/cp term (~0.0098 K/m) is the dry adiabatic lapse rate
    # If dT/dz > -g/cp (less steep than dry adiabatic), N² > 0 (stable)
    # If dT/dz < -g/cp (steeper than dry adiabatic), N² < 0 (unstable)
    N_squared = (g / T_mean) * (dT_dz + g / cp)

    # Clip to small positive value when unstable (convection dominates there anyway)
    N_squared = np.maximum(N_squared, 1e-6)
    N = np.sqrt(N_squared)

    return N


def compute_precipitation_jacobian(
    convective_frac: np.ndarray,
    stratiform_frac: np.ndarray,
    marine_sc_frac: np.ndarray,
    q: np.ndarray,
    w_largescale: np.ndarray,
    T_bl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobian of precipitation rate w.r.t. temperatures and humidity.

    With the moisture flux formulation:
    - P_conv = conv_frac × eff × w_updraft × q × ρ  (linear in q)
    - P_strat = strat_frac × eff × max(w, 0) × q × ρ  (linear in q)
    - P_drizzle = marine_frac × excess_q × column_mass / tau  (depends on q, T_bl)

    The main derivative is ∂P/∂q since precipitation scales linearly with moisture.
    Temperature dependence is weak (only through marine Sc drizzle saturation).

    Parameters
    ----------
    convective_frac : np.ndarray
        Convective cloud fraction (0-1).
    stratiform_frac : np.ndarray
        Stratiform cloud fraction (0-1).
    marine_sc_frac : np.ndarray
        Marine Sc cloud fraction (0-1).
    q : np.ndarray
        Specific humidity (kg/kg).
    w_largescale : np.ndarray
        Large-scale vertical velocity (m/s).
    T_bl : np.ndarray
        Boundary layer temperature (K).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (dP_dT_bl, dP_dT_atm, dP_dq) - derivatives of precipitation rate (kg/m²/s)
        w.r.t. T_bl (K), T_atm (K), and q (kg/kg)
    """
    # Initialize Jacobians (ensure float dtype)
    dP_dT_bl = np.zeros_like(T_bl, dtype=np.float64)
    dP_dT_atm = np.zeros_like(T_bl, dtype=np.float64)
    dP_dq = np.zeros_like(q, dtype=np.float64)

    # =========================================================================
    # 1. Convective precipitation Jacobian
    # P_conv = conv_frac × eff × w_updraft × q × ρ
    # ∂P_conv/∂q = conv_frac × eff × w_updraft × ρ
    # =========================================================================
    dP_conv_dq = (
        convective_frac
        * CONVECTIVE_PRECIP_EFFICIENCY
        * CONVECTIVE_UPDRAFT_VELOCITY
        * RHO_AIR
    )
    dP_dq += dP_conv_dq

    # =========================================================================
    # 2. Stratiform precipitation Jacobian
    # P_strat = strat_frac × eff × max(w, 0) × q × ρ
    # ∂P_strat/∂q = strat_frac × eff × max(w, 0) × ρ  (only where w > 0)
    # =========================================================================
    w_rising = np.maximum(w_largescale, 0.0)
    dP_strat_dq = (
        stratiform_frac
        * STRATIFORM_PRECIP_EFFICIENCY
        * w_rising
        * RHO_AIR
    )
    dP_dq += dP_strat_dq

    # =========================================================================
    # 3. Marine Sc drizzle Jacobian
    # P_drizzle = marine_frac × max(q - q_sat(T_cloud), 0) × column_mass / tau
    # ∂P_drizzle/∂q = marine_frac × column_mass / tau  (where supersaturated)
    # ∂P_drizzle/∂T_bl = marine_frac × (-∂q_sat/∂T) × column_mass / tau
    # =========================================================================
    # Temperature at cloud level
    T_cloud = T_bl - STANDARD_LAPSE_RATE_K_PER_M * MARINE_SC_CLOUD_TOP_HEIGHT_M
    q_sat_cloud = compute_saturation_specific_humidity(T_cloud)

    is_supersaturated = q > q_sat_cloud

    # ∂q_sat/∂T using Clausius-Clapeyron
    T_cloud_C = T_cloud - 273.15
    e_sat = 6.112 * np.exp(17.67 * T_cloud_C / (T_cloud_C + 243.5))
    de_sat_dT = e_sat * 17.67 * 243.5 / np.power(T_cloud_C + 243.5, 2)
    p_hPa = 1013.25
    denom_q = p_hPa - 0.378 * e_sat
    dq_sat_dT = 0.622 * p_hPa / (denom_q * denom_q) * de_sat_dT

    tau = MARINE_SC_DRIZZLE_TIMESCALE

    dP_drizzle_dq = np.where(
        is_supersaturated,
        marine_sc_frac * COLUMN_MASS_KG_M2 / tau,
        0.0
    )
    dP_drizzle_dT_bl = np.where(
        is_supersaturated,
        -marine_sc_frac * dq_sat_dT * COLUMN_MASS_KG_M2 / tau,
        0.0
    )

    dP_dq += dP_drizzle_dq
    dP_dT_bl += dP_drizzle_dT_bl

    return dP_dT_bl, dP_dT_atm, dP_dq
