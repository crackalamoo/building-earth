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
    LATENT_HEAT_VAPORIZATION_J_KG,
    GAS_CONSTANT_WATER_VAPOR_J_KG_K,
    ATMOSPHERE_LAYER_MIDPOINT_M,
    STANDARD_LAPSE_RATE_K_PER_M,
)
from climate_sim.physics.humidity import compute_saturation_specific_humidity

# Aliases for readability
GRAVITY = GRAVITY_M_S2
SPECIFIC_HEAT_AIR = HEAT_CAPACITY_AIR_J_KG_K

# Convection parameters (used by Jacobian)
UPPER_TROPOSPHERE_Q_FRACTION = 0.20  # q_upper / q_surface (dry air aloft)
MSE_INSTABILITY_THRESHOLD = 5000.0  # J/kg - minimum instability for convection
MSE_SATURATION_SCALE = 20000.0  # J/kg - instability scale for saturation
MAX_CONVECTIVE_PRECIP_RATE = 15.0 / 86400.0  # kg/m²/s (15 mm/day)

# Column mass for converting kg/kg to kg/m²
COLUMN_MASS_KG_M2 = 5000.0

# Cloud base height for large-scale condensation (LCL approximation)
CLOUD_BASE_HEIGHT_M = 1500.0  # ~1.5 km typical LCL

# Cloud top heights for precipitation calculations (must match clouds.py)
MARINE_SC_CLOUD_TOP_HEIGHT_M = 1000.0  # Marine Sc tops at ~1 km (below inversion)
STRATIFORM_CLOUD_TOP_HEIGHT_M = 1500.0  # Shallow stratiform decks at ~1.5 km


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
    q_surface: np.ndarray,
    T_bl_K: np.ndarray,
    marine_sc_cloud_top_height_m: float,
    stratiform_autoconversion_time: float,
) -> np.ndarray:
    """Compute marine stratocumulus precipitation (drizzle).

    Marine Sc produce light drizzle, not heavy rain. The precipitation
    is a slow autoconversion process in the cloud deck.

    Parameters
    ----------
    marine_sc_frac : np.ndarray
        Marine Sc cloud fraction (0-1).
    q_surface : np.ndarray
        Surface specific humidity (kg/kg).
    T_bl_K : np.ndarray
        Boundary layer temperature (K).
    marine_sc_cloud_top_height_m : float
        Marine Sc cloud top height (m).
    stratiform_autoconversion_time : float
        Autoconversion timescale (s).

    Returns
    -------
    np.ndarray
        Drizzle rate (kg/m²/s). Much lighter than stratiform precip.
    """
    # Temperature at cloud level
    T_cloud = T_bl_K - STANDARD_LAPSE_RATE_K_PER_M * marine_sc_cloud_top_height_m

    # Saturation specific humidity at cloud level
    q_sat_cloud = compute_saturation_specific_humidity(T_cloud)

    # Excess moisture (drizzle is weak - only slight supersaturation)
    excess_q = np.maximum(q_surface - q_sat_cloud, 0)

    # Drizzle: very slow autoconversion (2x longer than stratiform)
    # Marine Sc are stable and don't precipitate heavily
    drizzle_timescale = 2.0 * stratiform_autoconversion_time
    P_drizzle = (
        marine_sc_frac
        * excess_q
        * COLUMN_MASS_KG_M2
        / drizzle_timescale
    )

    return P_drizzle


def compute_convective_precipitation(
    convective_frac: np.ndarray,
    mse_instability: np.ndarray,
    q_surface: np.ndarray,
    mse_instability_threshold: float,
    mse_saturation_scale: float,
    max_convective_precip_rate: float,
) -> np.ndarray:
    """Compute convective precipitation from cloud fraction and instability.

    KEY CONSTRAINT: No convective clouds = no convective precipitation.
    This naturally handles:
    - Hot dry deserts: high instability BUT low clouds (subsidence) -> low precip
    - Moist tropics: high instability AND high clouds (rising) -> high precip

    Parameters
    ----------
    convective_frac : np.ndarray
        Convective cloud fraction (0-1).
    mse_instability : np.ndarray
        MSE instability (J/kg).
    q_surface : np.ndarray
        Surface specific humidity (kg/kg).
    mse_instability_threshold : float
        Minimum instability for convection (J/kg).
    mse_saturation_scale : float
        Instability scale for saturation (J/kg).
    max_convective_precip_rate : float
        Maximum convective precipitation rate (kg/m²/s).

    Returns
    -------
    np.ndarray
        Convective precipitation rate (kg/m²/s).
    """
    # Instability factor (same as for cloud formation)
    instability_excess = np.maximum(mse_instability - mse_instability_threshold, 0)
    instability_factor = instability_excess / (instability_excess + mse_saturation_scale)

    # Moisture factor: more moisture = more precipitation
    # Reference q ~ 0.015 kg/kg for tropical BL
    moisture_factor = np.clip(q_surface / 0.015, 0.2, 1.5)

    # Convective precipitation: SCALES WITH CLOUD FRACTION
    # No clouds = no precipitation (enforces physical consistency)
    P_convective = (
        convective_frac
        * instability_factor
        * moisture_factor
        * max_convective_precip_rate
    )

    return P_convective


def compute_stratiform_precipitation(
    stratiform_frac: np.ndarray,
    q_surface: np.ndarray,
    T_bl_K: np.ndarray,
    stratiform_cloud_top_height_m: float,
    stratiform_autoconversion_time: float,
) -> np.ndarray:
    """Compute stratiform precipitation from large-scale condensation.

    Stratiform precipitation occurs when:
    1. Stratiform clouds are present (requires stable, moist, rising air)
    2. Moisture exceeds saturation at cloud level

    The precipitation rate is the excess moisture relaxing toward saturation
    over the autoconversion timescale (~1 week).

    Parameters
    ----------
    stratiform_frac : np.ndarray
        Stratiform cloud fraction (0-1).
    q_surface : np.ndarray
        Surface specific humidity (kg/kg).
    T_bl_K : np.ndarray
        Boundary layer temperature (K).
    stratiform_cloud_top_height_m : float
        Stratiform cloud top height (m).
    stratiform_autoconversion_time : float
        Autoconversion timescale (s).

    Returns
    -------
    np.ndarray
        Stratiform precipitation rate (kg/m²/s).
    """
    # Temperature at cloud base (~1.5 km above surface)
    # Use moist adiabatic lapse rate for rising air
    T_cloud = T_bl_K - STANDARD_LAPSE_RATE_K_PER_M * stratiform_cloud_top_height_m

    # Saturation specific humidity at cloud level (colder = lower q_sat)
    q_sat_cloud = compute_saturation_specific_humidity(T_cloud)

    # Excess moisture above saturation
    excess_q = np.maximum(q_surface - q_sat_cloud, 0)

    # Stratiform precipitation: SCALES WITH CLOUD FRACTION
    # No clouds = no precipitation
    # Relaxation: P = cloud_frac * excess_q * COLUMN_MASS / tau
    P_stratiform = (
        stratiform_frac
        * excess_q
        * COLUMN_MASS_KG_M2
        / stratiform_autoconversion_time
    )

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
    T_bl: np.ndarray,
    T_atm: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobian of precipitation rate w.r.t. temperatures and humidity.

    Returns derivatives for the atmosphere heating tendency from precipitation:
        dT_atm/dt = P × L_v / C_atm

    Parameters
    ----------
    T_bl : np.ndarray
        Boundary layer temperature (K)
    T_atm : np.ndarray
        Free atmosphere temperature (K)
    q : np.ndarray
        Specific humidity (kg/kg)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (dP_dT_bl, dP_dT_atm, dP_dq) - derivatives of precipitation rate (kg/m²/s)
        w.r.t. T_bl (K), T_atm (K), and q (kg/kg)
    """
    cp = SPECIFIC_HEAT_AIR
    Lv = LATENT_HEAT_VAPORIZATION_J_KG

    # Initialize Jacobians
    dP_dT_bl = np.zeros_like(T_bl)
    dP_dT_atm = np.zeros_like(T_atm)
    dP_dq = np.zeros_like(q)

    # =========================================================================
    # 1. Convective precipitation Jacobian
    # P_conv = MAX_RATE × f(instability)  where f(x) = x / (x + scale)
    # instability = MSE_bl - MSE_atm = cp×T_bl + Lv×q - (cp×T_atm + Lv×q_upper + g×z)
    # =========================================================================
    # Lazy import to avoid circular dependency (clouds imports precipitation)
    from climate_sim.physics.clouds import compute_mse

    MSE_surface = compute_mse(T_bl, q, height_m=0.0)
    q_upper = q * UPPER_TROPOSPHERE_Q_FRACTION
    MSE_upper = compute_mse(T_atm, q_upper, height_m=ATMOSPHERE_LAYER_MIDPOINT_M)
    instability = MSE_surface - MSE_upper

    instability_excess = np.maximum(instability - MSE_INSTABILITY_THRESHOLD, 0)
    is_unstable = instability > MSE_INSTABILITY_THRESHOLD

    # Derivative of f(x) = x / (x + scale): f'(x) = scale / (x + scale)²
    denom = instability_excess + MSE_SATURATION_SCALE
    df_dx = MSE_SATURATION_SCALE / (denom * denom)

    # ∂instability/∂T_bl = cp, ∂instability/∂T_atm = -cp
    # ∂instability/∂q = Lv × (1 - UPPER_TROP_Q_FRACTION)
    dP_conv_dT_bl = np.where(is_unstable, MAX_CONVECTIVE_PRECIP_RATE * df_dx * cp, 0.0)
    dP_conv_dT_atm = np.where(is_unstable, -MAX_CONVECTIVE_PRECIP_RATE * df_dx * cp, 0.0)
    dP_conv_dq = np.where(
        is_unstable,
        MAX_CONVECTIVE_PRECIP_RATE * df_dx * Lv * (1 - UPPER_TROPOSPHERE_Q_FRACTION),
        0.0
    )

    dP_dT_bl += dP_conv_dT_bl
    dP_dT_atm += dP_conv_dT_atm
    dP_dq += dP_conv_dq

    # =========================================================================
    # 2. Supersaturation precipitation Jacobian
    # P_ss = max(q - q_sat(T_cloud), 0) × COLUMN_MASS_KG_M2 / tau
    # T_cloud = T_bl - γ_m × h_cloud  (γ_m depends on T_bl and q, but weakly)
    # Approximate: ∂T_cloud/∂T_bl ≈ 1, ∂γ_m/∂T_bl ≈ 0
    # =========================================================================
    # Lazy import to avoid circular dependency (humidity imports precipitation)
    from climate_sim.physics.humidity import compute_saturation_specific_humidity

    tau_seconds = 7 * 86400  # 1 week

    gamma_m = compute_moist_adiabatic_lapse_rate(T_bl, q)
    T_cloud = T_bl - gamma_m * CLOUD_BASE_HEIGHT_M
    q_sat_cloud = compute_saturation_specific_humidity(T_cloud)

    is_supersaturated = q > q_sat_cloud

    # ∂q_sat/∂T using Clausius-Clapeyron (Magnus formula derivative)
    T_cloud_C = T_cloud - 273.15
    e_sat = 6.112 * np.exp(17.67 * T_cloud_C / (T_cloud_C + 243.5))
    de_sat_dT = e_sat * 17.67 * 243.5 / np.power(T_cloud_C + 243.5, 2)
    p_hPa = 1013.25
    denom_q = p_hPa - 0.378 * e_sat
    dq_sat_dT = 0.622 * p_hPa / (denom_q * denom_q) * de_sat_dT

    # ∂P_ss/∂T_bl = -∂q_sat/∂T_cloud × ∂T_cloud/∂T_bl × COLUMN_MASS_KG_M2/tau
    #             ≈ -dq_sat_dT × COLUMN_MASS_KG_M2/tau  (since ∂T_cloud/∂T_bl ≈ 1)
    dP_ss_dT_bl = np.where(is_supersaturated, -dq_sat_dT * COLUMN_MASS_KG_M2 / tau_seconds, 0.0)

    # ∂P_ss/∂q = COLUMN_MASS_KG_M2 / tau (direct q dependence)
    dP_ss_dq = np.where(is_supersaturated, COLUMN_MASS_KG_M2 / tau_seconds, 0.0)

    dP_dT_bl += dP_ss_dT_bl
    dP_dq += dP_ss_dq

    return dP_dT_bl, dP_dT_atm, dP_dq
