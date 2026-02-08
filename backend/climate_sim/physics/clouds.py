"""Cloud formation physics.

This module handles cloud formation and cloud properties (fraction, albedo, top temperature).
Precipitation is computed in precipitation.py and called from here for the unified interface.

Cloud types:
- Convective clouds (cumulonimbus): Require MSE instability AND rising motion.
  Deep towers (~10-12 km), moderate per-cloud albedo (~0.25), very cold cloud tops.

- Stratiform clouds (stratus, stratocumulus): Require RH > threshold AND large-scale ascent.
  Shallow decks (~1-2 km), higher per-cloud albedo (~0.40), warm cloud tops.

- Marine stratocumulus: Form over ocean in SUBSIDENCE zones (subtropical highs).
  The subsidence creates a temperature inversion that traps moisture in the boundary
  layer. Very high coverage (70-90%), very high albedo (~0.55), warm tops.
  These are Earth's brightest clouds but have weak LW effect (warm tops).

- High clouds (cirrus, cirrostratus): Thin ice clouds at 8-12 km.
  Low albedo (~0.15), very cold tops. Weak SW reflection but strong LW trapping
  (greenhouse effect exceeds albedo cooling).

The physics flow is: Lift -> Clouds -> Precipitation (precipitation in precipitation.py)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.core.math_core import sigmoid
from climate_sim.data.constants import (
    ATMOSPHERE_LAYER_HEIGHT_M,
    ATMOSPHERE_LAYER_MIDPOINT_M,
    BOUNDARY_LAYER_HEIGHT_M,
    GRAVITY_M_S2,
    HEAT_CAPACITY_AIR_J_KG_K,
    LATENT_HEAT_VAPORIZATION_J_KG,
    STANDARD_LAPSE_RATE_K_PER_M,
)
from climate_sim.physics.humidity import compute_saturation_specific_humidity
from climate_sim.physics.precipitation import (
    compute_convective_precipitation,
    compute_marine_sc_precipitation,
    compute_stratiform_precipitation,
)


# Aliases for readability
SPECIFIC_HEAT_AIR = HEAT_CAPACITY_AIR_J_KG_K

# Dry adiabatic lapse rate (K/m) for LTS calculation
GAMMA_DRY = 9.8 / 1000.0  # ~9.8 K/km

# LTS thresholds for cloud type discrimination
LTS_CRIT = 18.0  # K - critical LTS for marine Sc vs convective transition

# Vertical velocity thresholds
W_CRIT = 0.005  # m/s - threshold for rising/sinking discrimination

# RH exponents for cloud types
# Lower exponents = less sensitivity to RH (clouds form at lower RH)
RH_EXPONENT_MARINE_SC = 0.5    # Marine Sc: weak RH sensitivity (inversion traps moisture)
RH_EXPONENT_STRATIFORM = 1.0   # Stratiform: linear RH sensitivity

# Convective precipitation onset threshold (Bretherton et al. 2004, Rushley et al. 2018)
# Precipitation picks up sharply at column RH ~ 0.7-0.8 over tropical oceans.
# Below this threshold, the atmosphere is too dry for deep moist convection to
# produce surface precipitation (sub-cloud evaporation, insufficient moisture depth).
RH_CRIT_CONVECTIVE = 0.65      # Critical RH for convective onset
RH_EXPONENT_CONVECTIVE = 1.5   # Super-linear above threshold (smoother than quadratic)

# High cloud parameters
# Increase factors to get ~20-30% high cloud coverage
HIGH_CLOUD_CONVECTIVE_FACTOR = 0.8   # Fraction of convective clouds that produce anvils
HIGH_CLOUD_FRONTAL_FACTOR = 0.5      # Fraction of frontal precip that produces cirrus
HIGH_CLOUD_BACKGROUND = 0.10         # Background cirrus coverage (large-scale ascent)

# Layer heights for cloud geometry
CONVECTIVE_CLOUD_TOP_HEIGHT_M = 10000.0  # Deep convective clouds reach ~10 km
CONVECTIVE_CLOUD_BASE_HEIGHT_M = 1500.0  # LCL for convective clouds ~1.5 km
STRATIFORM_CLOUD_TOP_HEIGHT_M = 1500.0   # Shallow stratiform decks at ~1.5 km
STRATIFORM_CLOUD_BASE_HEIGHT_M = 500.0   # Stratiform bases ~0.5 km
MARINE_SC_CLOUD_TOP_HEIGHT_M = 1000.0    # Marine Sc tops at ~1 km (below inversion)
MARINE_SC_CLOUD_BASE_HEIGHT_M = 300.0    # Marine Sc bases ~300 m

# High cloud (cirrus/anvil) parameters
# High clouds form from detrainment of convective anvils and large-scale ascent
HIGH_CLOUD_TOP_HEIGHT_M = 12000.0    # Cirrus/anvils at ~12 km (tropopause)
HIGH_CLOUD_BASE_HEIGHT_M = 8000.0    # High cloud bases ~8 km

# ============================================================
# OPTICAL DEPTH CONSTANTS
# ============================================================
# For water clouds: τ_LW ≈ τ_SW (same value)
# For ice clouds: τ_LW > τ_SW (ice more absorbing in IR)

# Cloud type          τ_SW    τ_LW    (reason)
CONVECTIVE_CLOUD_TAU_SW = 40.0
CONVECTIVE_CLOUD_TAU_LW = 40.0   # Water cloud: τ_LW = τ_SW

STRATIFORM_CLOUD_TAU_SW = 7.0
STRATIFORM_CLOUD_TAU_LW = 7.0    # Water cloud: τ_LW = τ_SW

MARINE_SC_CLOUD_TAU_SW = 10.0
MARINE_SC_CLOUD_TAU_LW = 10.0    # Water cloud: τ_LW = τ_SW

HIGH_CLOUD_TAU_SW = 0.8          # Ice cloud: thin in SW
HIGH_CLOUD_TAU_LW = 1.5          # Ice cloud: thicker in LW (more absorbing)

# Two-stream G parameter for albedo: α = τ/(τ+G)
TWO_STREAM_G_WATER = 7.0         # Water clouds
TWO_STREAM_G_ICE = 12.0          # Ice clouds (forward scattering)

# Single-scattering albedo ω₀
WATER_CLOUD_OMEGA0 = 0.99
ICE_CLOUD_OMEGA0 = 0.97

# Clear-sky SW absorptance (split between atmosphere and boundary layer)
# Total ~20%, but water vapor is concentrated in lower troposphere
CLEAR_SKY_SW_ABSORPTANCE_ATM = 0.12   # Free atmosphere (O3, some H2O)
CLEAR_SKY_SW_ABSORPTANCE_BL = 0.08    # Boundary layer (H2O-rich)
# TODO: BL absorptance should depend on humidity (range ~5-12%)

# Legacy aliases for backwards compatibility
HIGH_CLOUD_TAU = HIGH_CLOUD_TAU_SW
HIGH_CLOUD_G = TWO_STREAM_G_ICE
CONVECTIVE_CLOUD_TAU = CONVECTIVE_CLOUD_TAU_SW
STRATIFORM_CLOUD_TAU = STRATIFORM_CLOUD_TAU_SW
MARINE_SC_CLOUD_TAU = MARINE_SC_CLOUD_TAU_SW

# MSE instability thresholds
MSE_INSTABILITY_THRESHOLD = 5000.0  # J/kg - minimum instability for convection
MSE_SATURATION_SCALE = 20000.0  # J/kg - instability scale for saturation
UPPER_TROPOSPHERE_Q_FRACTION = 0.20  # q_upper / q_BL (dry air aloft)

# Precipitation parameters
MAX_CONVECTIVE_PRECIP_RATE = 15.0 / 86400.0  # kg/m²/s (~15 mm/day max)
STRATIFORM_AUTOCONVERSION_TIME = 7 * 86400  # seconds (~1 week)

# Two-stream cloud albedo parameter
# From solving the two-stream radiative transfer equations for a non-absorbing
# scattering layer:
#   α_cloud = τ_SW / (τ_SW + D)
#
# where τ_SW is shortwave optical depth and D is a diffusivity factor
# that accounts for multiple scattering and non-normal incidence.
# Larger D → more transparent clouds (lower albedo for same optical depth).
# Typical values: D ≈ 7-10 for climatological mean conditions.
TWO_STREAM_G = 7.0  # Two-stream diffusivity factor


def compute_cloud_emissivity(tau_lw: float | np.ndarray) -> float | np.ndarray:
    """Compute cloud LW emissivity from optical depth using Beer-Lambert.

    ε = 1 - exp(-τ_LW)

    This is used for LONGWAVE radiation (thermal infrared).

    Parameters
    ----------
    tau_lw : float or np.ndarray
        Longwave optical depth.

    Returns
    -------
    float or np.ndarray
        Cloud emissivity (0-1).
    """
    return 1.0 - np.exp(-tau_lw)


def compute_cloud_sw_absorptance(
    tau_sw: float | np.ndarray,
    omega0: float = WATER_CLOUD_OMEGA0,
) -> float | np.ndarray:
    """Compute cloud SW absorptance from optical depth and single-scattering albedo.

    Uses simplified two-stream approximation:
    absorptance ≈ (1 - ω₀) × τ / (1 + τ)

    This is used for SHORTWAVE radiation (visible/solar).
    Water clouds are highly scattering (ω₀≈0.99), so absorptance is low.

    Parameters
    ----------
    tau_sw : float or np.ndarray
        Shortwave optical depth.
    omega0 : float
        Single-scattering albedo. Water clouds: ~0.99, ice clouds: ~0.97.

    Returns
    -------
    float or np.ndarray
        Cloud SW absorptance (fraction of incident light absorbed).
    """
    return (1.0 - omega0) * tau_sw / (1.0 + tau_sw)


@dataclass(frozen=True)
class CloudConfig:
    """Configuration for cloud-precipitation physics."""

    enabled: bool = True

    # Convective cloud parameters
    convective_cloud_albedo_base: float = 0.20  # Base albedo for convective clouds
    convective_cloud_albedo_rh_factor: float = 0.10  # Additional albedo from RH
    convective_cloud_top_height_m: float = CONVECTIVE_CLOUD_TOP_HEIGHT_M

    # Stratiform cloud parameters
    stratiform_cloud_albedo_base: float = 0.35  # Base albedo for stratiform clouds
    stratiform_cloud_albedo_rh_factor: float = 0.15  # Additional albedo from RH
    stratiform_cloud_top_height_m: float = STRATIFORM_CLOUD_TOP_HEIGHT_M

    # Marine stratocumulus parameters
    marine_sc_cloud_albedo: float = 0.55  # High albedo - brightest clouds on Earth
    marine_sc_cloud_top_height_m: float = MARINE_SC_CLOUD_TOP_HEIGHT_M
    marine_sc_base_coverage: float = 0.75  # Base coverage in marine Sc regions


@dataclass
class CloudPrecipOutput:
    """Output from unified cloud-precipitation computation.

    Contains computed properties for all cloud types. Heights are module constants,
    not stored here.
    """

    # Convective cloud properties (deep towers, anvils)
    convective_frac: np.ndarray      # Cloud fraction (0-1)
    convective_albedo: np.ndarray    # Per-cloud albedo
    convective_top_K: np.ndarray     # Cloud top temperature (K)
    convective_precip: np.ndarray    # Precipitation rate (kg/m²/s)

    # Stratiform cloud properties (shallow decks, frontal clouds)
    stratiform_frac: np.ndarray      # Cloud fraction (0-1)
    stratiform_albedo: np.ndarray    # Per-cloud albedo
    stratiform_top_K: np.ndarray     # Cloud top temperature (K)
    stratiform_precip: np.ndarray    # Precipitation rate (kg/m²/s)

    # Marine stratocumulus properties (ocean + subsidence zones)
    marine_sc_frac: np.ndarray       # Cloud fraction (0-1)
    marine_sc_albedo: np.ndarray     # Per-cloud albedo
    marine_sc_top_K: np.ndarray      # Cloud top temperature (K)
    marine_sc_precip: np.ndarray     # Precipitation rate (kg/m²/s)

    # High cloud properties (cirrus, anvils at tropopause)
    high_cloud_frac: np.ndarray      # Cloud fraction (0-1)
    high_cloud_albedo: np.ndarray    # Per-cloud albedo
    high_cloud_top_K: np.ndarray     # Cloud top temperature (K)

    # ============================================================
    # LONGWAVE OPTICAL PROPERTIES (for thermal IR emission/absorption)
    # ============================================================
    # Per-cloud-type emissivities derived from τ_LW via Beer-Lambert
    eps_convective: float | np.ndarray | None = None
    eps_stratiform: float | np.ndarray | None = None
    eps_marine_sc: float | np.ndarray | None = None
    eps_high: float | np.ndarray | None = None

    # ============================================================
    # SHORTWAVE OPTICAL PROPERTIES (for solar absorption)
    # ============================================================
    # Per-cloud-type SW absorptances derived from τ_SW and ω₀
    abs_sw_convective: float | np.ndarray | None = None
    abs_sw_stratiform: float | np.ndarray | None = None
    abs_sw_marine_sc: float | np.ndarray | None = None
    abs_sw_high: float | np.ndarray | None = None

    @property
    def total_frac(self) -> np.ndarray:
        """Total cloud fraction (for diagnostics only, not physics)."""
        # Combine low clouds using probabilistic overlap (same level)
        low_frac = self.convective_frac
        low_frac = low_frac + self.stratiform_frac * (1.0 - low_frac)
        low_frac = low_frac + self.marine_sc_frac * (1.0 - low_frac)
        # High clouds can overlap with low clouds (different vertical levels)
        total = low_frac + self.high_cloud_frac * (1.0 - low_frac)
        return total

    @property
    def total_precip(self) -> np.ndarray:
        """Total precipitation rate (kg/m²/s)."""
        return self.convective_precip + self.stratiform_precip + self.marine_sc_precip


def compute_mse(
    temperature_K: np.ndarray,
    specific_humidity: np.ndarray,
    height_m: float = 0.0,
) -> np.ndarray:
    """Compute moist static energy.

    MSE = c_p * T + L_v * q + g * z

    MSE is approximately conserved during moist adiabatic processes.
    """
    return (
        SPECIFIC_HEAT_AIR * temperature_K
        + LATENT_HEAT_VAPORIZATION_J_KG * specific_humidity
        + GRAVITY_M_S2 * height_m
    )


def compute_mse_instability(
    T_bl_K: np.ndarray,
    T_atm_K: np.ndarray,
    q_bl: np.ndarray,
) -> np.ndarray:
    """Compute MSE instability for convection.

    Convection occurs when boundary layer MSE exceeds upper-troposphere MSE.
    The instability drives overturning that produces clouds and precipitation.

    Parameters
    ----------
    T_bl_K : np.ndarray
        Boundary layer temperature (K).
    T_atm_K : np.ndarray
        Free atmosphere temperature (K).
    q_bl : np.ndarray
        Boundary layer specific humidity (kg/kg).

    Returns
    -------
    np.ndarray
        MSE instability (J/kg). Positive = unstable.
    """
    # MSE at boundary layer
    MSE_bl = compute_mse(T_bl_K, q_bl, height_m=0.0)

    # MSE at atmosphere layer (midpoint of free troposphere)
    # Upper troposphere is much drier (moisture precipitates out during ascent)
    q_upper = q_bl * UPPER_TROPOSPHERE_Q_FRACTION
    MSE_upper = compute_mse(T_atm_K, q_upper, height_m=ATMOSPHERE_LAYER_MIDPOINT_M)

    # Instability: positive when BL MSE exceeds upper MSE
    return MSE_bl - MSE_upper


def compute_lts(
    T_bl_K: np.ndarray,
    T_atm_K: np.ndarray,
) -> np.ndarray:
    """Compute Lower Tropospheric Stability (LTS).

    LTS measures the strength of the temperature inversion between the boundary
    layer and free atmosphere. High LTS = strong inversion = stable = favors
    stratocumulus. Low LTS = weak inversion = unstable = favors convection.

    The formula is a proxy for potential temperature difference:
        LTS = T_atm - T_bl + gamma_dry * delta_z

    where delta_z is the height difference between layer midpoints.

    Parameters
    ----------
    T_bl_K : np.ndarray
        Boundary layer temperature (K).
    T_atm_K : np.ndarray
        Free atmosphere temperature (K).

    Returns
    -------
    np.ndarray
        Lower tropospheric stability (K). Higher = more stable.

    Notes
    -----
    The dry adiabatic correction (gamma_dry * delta_z) accounts for the
    expected temperature decrease with height in a neutrally stable atmosphere.
    If the actual temperature decrease is less than adiabatic, the atmosphere
    is stable (positive LTS).
    """
    # Height difference between layer midpoints
    # BL midpoint: BOUNDARY_LAYER_HEIGHT_M / 2
    # Atm midpoint: BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M / 2
    delta_z = (ATMOSPHERE_LAYER_HEIGHT_M - BOUNDARY_LAYER_HEIGHT_M) / 2.0 + BOUNDARY_LAYER_HEIGHT_M

    # LTS = theta_atm - theta_bl (approximately)
    # Using the proxy: T_atm - T_bl + gamma_dry * delta_z
    lts = T_atm_K - T_bl_K + GAMMA_DRY * delta_z

    return lts




def compute_convective_clouds(
    lts: np.ndarray,
    vertical_velocity: np.ndarray,
    rh: np.ndarray,
    config: CloudConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute convective cloud fraction and albedo using LTS-based scheme.

    Convective clouds require:
    1. Low LTS (unstable atmosphere) - LTS < LTS_crit
    2. Rising motion (w > w_crit)
    3. Sufficient moisture (RH^b factor)

    Formula: C_conv = C_max × RH^b × sigma(w - w_crit) × sigma(LTS_crit - LTS)

    Parameters
    ----------
    lts : np.ndarray
        Lower tropospheric stability (K). Lower = more unstable.
    vertical_velocity : np.ndarray
        Large-scale vertical velocity (m/s). Positive = rising.
    rh : np.ndarray
        Relative humidity (0-1).
    config : CloudConfig
        Cloud configuration parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (convective_frac, convective_albedo)
    """
    # RH factor: need moist boundary layer for deep moist convection.
    # Bretherton et al. (2004): precipitation onset at CRH ~ 0.7.
    # Below RH_CRIT, air is too dry — any condensate re-evaporates (virga).
    # Use sigmoid for smooth transition (solver convergence).
    rh_clipped = np.clip(rh, 0.0, 1.0)
    rh_factor = sigmoid(rh_clipped - RH_CRIT_CONVECTIVE, scale=0.12) ** RH_EXPONENT_CONVECTIVE

    # Rising motion factor: sigma(w - w_crit)
    # w > w_crit → factor approaches 1
    # w < w_crit → factor approaches 0
    w_factor = sigmoid(vertical_velocity - W_CRIT, scale=W_CRIT)

    # Instability factor: sigma(LTS_crit - LTS)
    # LTS < LTS_crit → factor approaches 1 (unstable, convection favored)
    # LTS > LTS_crit → factor approaches 0 (stable, convection suppressed)
    lts_factor = sigmoid(LTS_CRIT - lts, scale=5.0)  # 5 K transition width

    # Convective cloud fraction
    # Max ~0.60 in active convection zones (ITCZ)
    convective_frac = 0.60 * rh_factor * w_factor * lts_factor
    convective_frac = np.clip(convective_frac, 0.0, 0.60)

    # Cloud albedo from config
    convective_albedo = np.full_like(rh, config.convective_cloud_albedo_base)

    return convective_frac, convective_albedo


def compute_stratiform_clouds(
    lts: np.ndarray,
    vertical_velocity: np.ndarray,
    rh: np.ndarray,
    config: CloudConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute stratiform cloud fraction and albedo using LTS-based scheme.

    Stratiform clouds (frontal, mid-latitude) require:
    1. Rising motion (w > -w_crit, i.e., not strong subsidence)
    2. Sufficient moisture (RH factor)
    3. Moderate stability (LTS not too low, else convection dominates)

    Formula: C_strat = C_max * RH * sigma(w + w_crit) * sigma(LTS - threshold)
    """
    # RH factor: moderate moisture dependence
    rh_factor = np.power(np.clip(rh, 0.0, 1.0), RH_EXPONENT_STRATIFORM)

    # Rising motion factor: sigma(w + w_crit)
    # w > -w_crit → factor approaches 1 (allow weak subsidence)
    # w << -w_crit → factor approaches 0 (strong subsidence suppresses)
    w_factor = sigmoid(vertical_velocity + W_CRIT, scale=W_CRIT)

    # LTS factor: stratiform clouds suppressed at low LTS (unstable air)
    # where convection dominates. Use a lower threshold than marine Sc.
    # LTS > LTS_crit - 5 K → factor approaches 1 (stable enough for stratiform)
    # LTS < LTS_crit - 10 K → factor approaches 0 (too unstable, convection dominates)
    lts_threshold_strat = LTS_CRIT - 5.0  # 13 K with LTS_CRIT=18
    lts_factor = sigmoid(lts - lts_threshold_strat, scale=3.0)

    # Stratiform cloud fraction
    # Max ~0.70 in frontal zones
    stratiform_frac = 0.70 * rh_factor * w_factor * lts_factor
    stratiform_frac = np.clip(stratiform_frac, 0.0, 0.70)

    # Cloud albedo from config
    stratiform_albedo = np.full_like(rh, config.stratiform_cloud_albedo_base)

    return stratiform_frac, stratiform_albedo


def compute_marine_stratocumulus(
    lts: np.ndarray,
    vertical_velocity: np.ndarray,
    rh: np.ndarray,
    ocean_mask: np.ndarray,
    config: CloudConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute marine stratocumulus cloud fraction and albedo using LTS-based scheme.

    Marine stratocumulus form over ocean with:
    1. High LTS (strong inversion) - LTS > LTS_crit
    2. Subsidence (w < 0, creates the inversion)
    3. Moisture (RH^a factor)

    Formula: C_msc = C_max × RH^a × sigma(-w) × sigma(LTS - LTS_crit) × ocean_mask

    These are Earth's brightest clouds, covering subtropical ocean eastern
    boundaries (California, Peru, Namibia, Canary coasts).

    Parameters
    ----------
    lts : np.ndarray
        Lower tropospheric stability (K). Higher = more stable = favors Sc.
    vertical_velocity : np.ndarray
        Large-scale vertical velocity (m/s). Negative = subsidence.
    rh : np.ndarray
        Relative humidity (0-1).
    ocean_mask : np.ndarray
        Boolean mask, True over ocean.
    config : CloudConfig
        Cloud configuration parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (marine_sc_frac, marine_sc_albedo)
    """
    # Marine Sc only form over ocean
    over_ocean = ocean_mask.astype(float)

    # RH factor: moderate moisture dependence
    rh_factor = np.power(np.clip(rh, 0.0, 1.0), RH_EXPONENT_MARINE_SC)

    # Subsidence factor: sigma(-w)
    # w < 0 (subsidence) → factor approaches 1
    # w > 0 (rising) → factor approaches 0 (convection breaks up deck)
    w_factor = sigmoid(-vertical_velocity, scale=W_CRIT)

    # Stability factor: sigma(LTS - LTS_crit)
    # LTS > LTS_crit → factor approaches 1 (stable, inversion present)
    # LTS < LTS_crit → factor approaches 0 (unstable, no inversion)
    lts_factor = sigmoid(lts - LTS_CRIT, scale=5.0)  # 5 K transition width

    # Marine Sc fraction: high coverage where conditions are right
    marine_sc_frac = 0.90 * over_ocean * rh_factor * w_factor * lts_factor
    marine_sc_frac = np.clip(marine_sc_frac, 0.0, 0.90)

    # Albedo from config
    marine_sc_albedo = np.full_like(rh, config.marine_sc_cloud_albedo)

    return marine_sc_frac, marine_sc_albedo


def compute_cloud_top_temperatures(
    T_surface_K: np.ndarray,
    config: CloudConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute cloud top temperatures for radiation.

    Cloud top temperature determines longwave emission:
    - Convective clouds: very cold tops (~220 K), reduce OLR significantly
    - Stratiform clouds: warm tops (~280 K), near surface temperature
    - Marine Sc: very warm tops (~285 K), nearly at surface temperature
    - High clouds: very cold tops (~210 K), at tropopause

    Parameters
    ----------
    T_surface_K : np.ndarray
        Surface temperature (K).
    config : CloudConfig
        Cloud configuration parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (convective_top_K, stratiform_top_K, marine_sc_top_K, high_cloud_top_K)
    """
    # Convective cloud tops: high altitude, very cold
    convective_top_K = T_surface_K - STANDARD_LAPSE_RATE_K_PER_M * config.convective_cloud_top_height_m
    convective_top_K = np.maximum(convective_top_K, 180.0)  # Floor at tropopause

    # Stratiform cloud tops: low altitude, warm
    stratiform_top_K = T_surface_K - STANDARD_LAPSE_RATE_K_PER_M * config.stratiform_cloud_top_height_m
    stratiform_top_K = np.maximum(stratiform_top_K, 240.0)  # Reasonable floor

    # Marine Sc cloud tops: very low altitude, very warm (near surface)
    marine_sc_top_K = T_surface_K - STANDARD_LAPSE_RATE_K_PER_M * config.marine_sc_cloud_top_height_m
    marine_sc_top_K = np.maximum(marine_sc_top_K, 250.0)  # Reasonable floor

    # High cloud tops: very high altitude, at tropopause
    high_cloud_top_K = T_surface_K - STANDARD_LAPSE_RATE_K_PER_M * HIGH_CLOUD_TOP_HEIGHT_M
    high_cloud_top_K = np.maximum(high_cloud_top_K, 200.0)  # Floor at tropopause

    return convective_top_K, stratiform_top_K, marine_sc_top_K, high_cloud_top_K


def compute_high_cloud_fraction(
    convective_frac: np.ndarray,
    stratiform_precip: np.ndarray,
) -> np.ndarray:
    """Compute high cloud (cirrus/anvil) fraction.

    High clouds form from:
    1. Detrainment of convective anvils (α × convective_frac)
    2. Frontal precipitation lifting moisture to high altitudes (β × stratiform_precip)

    Formula: C_high = α × C_conv + β × P_strat_normalized

    Parameters
    ----------
    convective_frac : np.ndarray
        Convective cloud fraction (0-1). Anvils detrain from convective towers.
    stratiform_precip : np.ndarray
        Stratiform precipitation rate (kg/m²/s). Frontal systems lift moisture.

    Returns
    -------
    np.ndarray
        High cloud fraction (0-1).

    Notes
    -----
    - Convective contribution: 50% of convective clouds produce anvils
    - Frontal contribution: scaled by precipitation intensity
    - Background minimum ~5% (thin cirrus from general subsidence)
    """
    # Convective anvil contribution: fraction of deep convection that detrains
    anvil_contribution = HIGH_CLOUD_CONVECTIVE_FACTOR * convective_frac

    # Frontal contribution: scales with stratiform precipitation intensity
    # Normalize by typical frontal precip rate (~2 mm/day = 2.3e-5 kg/m²/s)
    frontal_precip_scale = 2.0 / 86400.0  # kg/m²/s
    frontal_normalized = np.clip(stratiform_precip / frontal_precip_scale, 0.0, 1.0)
    frontal_contribution = HIGH_CLOUD_FRONTAL_FACTOR * frontal_normalized

    # Total high cloud fraction with background cirrus
    high_frac = HIGH_CLOUD_BACKGROUND + anvil_contribution + frontal_contribution
    high_frac = np.clip(high_frac, 0.0, 0.70)

    return high_frac


def compute_high_cloud_albedo(shape: tuple) -> np.ndarray:
    """Compute high cloud albedo using two-stream approximation.

    High clouds (cirrus/anvils) have fixed optical depth τ = 2.

    Parameters
    ----------
    shape : tuple
        Shape of output array.

    Returns
    -------
    np.ndarray
        High cloud albedo (uniform value).
    """
    tau = HIGH_CLOUD_TAU
    albedo = tau / (tau + HIGH_CLOUD_G)
    return np.full(shape, albedo)


def compute_vertical_velocity_from_divergence(
    divergence: np.ndarray,
) -> np.ndarray:
    """Compute vertical velocity from horizontal divergence.

    Mass continuity: w = -∫ div(V) dz ≈ -div(V) × H

    Parameters
    ----------
    divergence : np.ndarray
        Horizontal divergence (1/s). Positive = divergence (subsidence).

    Returns
    -------
    np.ndarray
        Vertical velocity (m/s). Positive = rising, negative = sinking.
    """
    # Vertical velocity at BL top from mass continuity
    # Divergence > 0 (subtropical highs) -> subsidence (w < 0)
    # Convergence < 0 (ITCZ) -> rising motion (w > 0)
    w = -divergence * BOUNDARY_LAYER_HEIGHT_M
    return w


def compute_vertical_velocity_from_pressure(
    dp_norm: np.ndarray,
    w_scale: float = 0.01,
) -> np.ndarray:
    """Compute vertical velocity from normalized pressure anomaly.

    This provides a diagnostic vertical velocity that depends on temperature
    (via pressure) rather than wind divergence, avoiding lag issues in the solver.

    Low pressure (dp_norm < 0) → rising motion (convection, ITCZ)
    High pressure (dp_norm > 0) → sinking motion (subsidence, subtropical highs)

    Parameters
    ----------
    dp_norm : np.ndarray
        Normalized pressure anomaly, clipped to [-1, 1].
        Computed from temperature via compute_pressure() in radiation.py.
    w_scale : float
        Vertical velocity scale (m/s). Default 0.01 m/s ≈ 1 cm/s.

    Returns
    -------
    np.ndarray
        Vertical velocity (m/s). Positive = rising, negative = sinking.
    """
    # Linear relationship: dp_norm = -1 (ITCZ low) → w = +w_scale (rising)
    #                      dp_norm = +1 (high pressure) → w = -w_scale (sinking)
    w = -dp_norm * w_scale
    return w


def compute_vertical_velocity_from_warm_advection(
    T: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    dx: np.ndarray,
    dy: float,
    w_scale: float = 500.0,
) -> np.ndarray:
    """Compute vertical velocity from warm advection (frontal lifting).

    At fronts, horizontal temperature advection forces vertical motion.
    From quasi-geostrophic theory: w ∝ -V · ∇T (warm advection = rising).

    When wind blows from warm to cold regions (warm advection), air rises.
    When wind blows from cold to warm regions (cold advection), air sinks.

    Parameters
    ----------
    T : np.ndarray
        Temperature field (K), shape (nlat, nlon).
    wind_u : np.ndarray
        Zonal wind component (m/s), shape (nlat, nlon).
    wind_v : np.ndarray
        Meridional wind component (m/s), shape (nlat, nlon).
    dx : np.ndarray
        Zonal grid spacing (m), shape (nlat, 1) or (nlat, nlon).
    dy : float
        Meridional grid spacing (m).
    w_scale : float
        Conversion factor from K/s to m/s. Default 500 m/(K/s).
        Based on QG scaling: w ~ (f/N²) * T_advection * H / T

    Returns
    -------
    np.ndarray
        Vertical velocity (m/s). Positive = rising, negative = sinking.

    Notes
    -----
    The scaling relates horizontal temperature advection (K/s) to vertical
    velocity (m/s). A typical strong front has dT/dx ~ 10 K / 1000 km = 1e-5 K/m,
    and wind ~ 10 m/s, giving warm advection ~ 1e-4 K/s.
    With w_scale = 500, this gives w ~ 0.05 m/s = 5 cm/s, which is reasonable
    for frontal lifting.
    """
    nlat, nlon = T.shape

    # Compute temperature gradients using centered differences
    # Zonal gradient: periodic in longitude
    T_east = np.roll(T, -1, axis=1)
    T_west = np.roll(T, 1, axis=1)
    dT_dx = (T_east - T_west) / (2.0 * dx)

    # Meridional gradient: handle poles with one-sided differences
    dT_dy = np.zeros_like(T)
    if nlat > 2:
        # Interior: centered difference
        dT_dy[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dy)
        # Boundaries: one-sided
        dT_dy[0, :] = (T[1, :] - T[0, :]) / dy
        dT_dy[-1, :] = (T[-1, :] - T[-2, :]) / dy

    # Warm advection: -V · ∇T
    # Positive when wind blows from warm to cold (warm air advected in)
    warm_advection = -(wind_u * dT_dx + wind_v * dT_dy)  # K/s

    # Convert to vertical velocity
    # Warm advection (positive) → rising (positive w)
    w_frontal = warm_advection * w_scale

    # Limit to reasonable values (±0.1 m/s = ±10 cm/s)
    w_frontal = np.clip(w_frontal, -0.1, 0.1)

    return w_frontal


def compute_unified_vertical_velocity(
    dp_norm: np.ndarray,
    T: np.ndarray | None = None,
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    dx: np.ndarray | None = None,
    dy: float | None = None,
    w_pressure_scale: float = 0.01,
    w_frontal_scale: float = 500.0,
) -> np.ndarray:
    """Compute unified large-scale vertical velocity.

    Combines:
    1. Pressure-driven w (ITCZ convergence, subtropical subsidence)
    2. Frontal w from warm advection (midlatitude fronts)

    Parameters
    ----------
    dp_norm : np.ndarray
        Normalized pressure anomaly from compute_pressure().
    T : np.ndarray | None
        Temperature field for warm advection calculation.
    wind_u, wind_v : np.ndarray | None
        Wind components for warm advection calculation.
    dx : np.ndarray | None
        Zonal grid spacing (m).
    dy : float | None
        Meridional grid spacing (m).
    w_pressure_scale : float
        Scale for pressure-driven vertical velocity.
    w_frontal_scale : float
        Scale for frontal warm advection vertical velocity.

    Returns
    -------
    np.ndarray
        Total large-scale vertical velocity (m/s).
    """
    # Pressure-driven component (always computed)
    w_pressure = compute_vertical_velocity_from_pressure(dp_norm, w_scale=w_pressure_scale)

    # Frontal component (if wind and temperature available)
    if T is not None and wind_u is not None and wind_v is not None and dx is not None and dy is not None:
        w_frontal = compute_vertical_velocity_from_warm_advection(
            T, wind_u, wind_v, dx, dy, w_scale=w_frontal_scale
        )
        w_total = w_pressure + w_frontal
    else:
        w_total = w_pressure

    return w_total


def compute_clouds_and_precipitation(
    T_bl_K: np.ndarray,
    T_atm_K: np.ndarray,
    q: np.ndarray,
    rh: np.ndarray,
    vertical_velocity: np.ndarray,
    T_surface_K: np.ndarray | None = None,
    ocean_mask: np.ndarray | None = None,
    config: CloudConfig | None = None,
) -> CloudPrecipOutput:
    """Compute unified cloud and precipitation fields.

    This is the main entry point for the cloud-precipitation module.
    Uses LTS (Lower Tropospheric Stability) to distinguish cloud types:
    - Low LTS + rising → convective clouds
    - High LTS + subsidence + ocean → marine stratocumulus
    - Rising motion + RH → stratiform clouds
    - Convective + frontal precip → high clouds

    Parameters
    ----------
    T_bl_K : np.ndarray
        Boundary layer temperature (K).
    T_atm_K : np.ndarray
        Free atmosphere temperature (K).
    q : np.ndarray
        Specific humidity (kg/kg).
    rh : np.ndarray
        Relative humidity (0-1).
    vertical_velocity : np.ndarray
        Large-scale vertical velocity (m/s). Positive = rising.
    T_surface_K : np.ndarray | None
        Surface temperature (K). If None, uses T_bl_K for cloud top calculation.
    ocean_mask : np.ndarray | None
        Boolean mask, True over ocean. Required for marine Sc.
    config : CloudConfig | None
        Cloud configuration. Uses defaults if None.

    Returns
    -------
    CloudPrecipOutput
        Unified cloud and precipitation output.
    """
    if config is None:
        config = CloudConfig()

    if T_surface_K is None:
        T_surface_K = T_bl_K

    # 1. Compute LTS (Lower Tropospheric Stability)
    lts = compute_lts(T_bl_K, T_atm_K)

    # 2. Compute convective clouds (low LTS + rising + RH)
    convective_frac, _ = compute_convective_clouds(lts, vertical_velocity, rh, config)

    # 3. Compute stratiform clouds (rising + RH, no stability requirement)
    stratiform_frac, _ = compute_stratiform_clouds(lts, vertical_velocity, rh, config)

    # 4. Compute marine stratocumulus (high LTS + subsidence + ocean + RH)
    if ocean_mask is not None:
        marine_sc_frac, _ = compute_marine_stratocumulus(
            lts, vertical_velocity, rh, ocean_mask, config
        )
    else:
        marine_sc_frac = np.zeros_like(T_bl_K)

    # 5. Compute cloud top temperatures
    convective_top_K, stratiform_top_K, marine_sc_top_K, high_cloud_top_K = compute_cloud_top_temperatures(
        T_surface_K, config
    )

    # ============================================================
    # COMPUTE LONGWAVE OPTICAL PROPERTIES (emissivities from τ_LW)
    # ============================================================
    eps_conv = compute_cloud_emissivity(CONVECTIVE_CLOUD_TAU_LW)
    eps_strat = compute_cloud_emissivity(STRATIFORM_CLOUD_TAU_LW)
    eps_marine = compute_cloud_emissivity(MARINE_SC_CLOUD_TAU_LW)
    eps_high = compute_cloud_emissivity(HIGH_CLOUD_TAU_LW)

    # ============================================================
    # COMPUTE SHORTWAVE OPTICAL PROPERTIES (absorptances from τ_SW)
    # ============================================================
    abs_sw_conv = compute_cloud_sw_absorptance(CONVECTIVE_CLOUD_TAU_SW, WATER_CLOUD_OMEGA0)
    abs_sw_strat = compute_cloud_sw_absorptance(STRATIFORM_CLOUD_TAU_SW, WATER_CLOUD_OMEGA0)
    abs_sw_marine = compute_cloud_sw_absorptance(MARINE_SC_CLOUD_TAU_SW, WATER_CLOUD_OMEGA0)
    abs_sw_high = compute_cloud_sw_absorptance(HIGH_CLOUD_TAU_SW, ICE_CLOUD_OMEGA0)

    # ============================================================
    # COMPUTE SHORTWAVE CLOUD ALBEDOS (from τ_SW with appropriate G)
    # ============================================================
    convective_albedo = np.full_like(
        T_bl_K, CONVECTIVE_CLOUD_TAU_SW / (CONVECTIVE_CLOUD_TAU_SW + TWO_STREAM_G_WATER)
    )
    stratiform_albedo = np.full_like(
        T_bl_K, STRATIFORM_CLOUD_TAU_SW / (STRATIFORM_CLOUD_TAU_SW + TWO_STREAM_G_WATER)
    )
    marine_sc_albedo = np.full_like(
        T_bl_K, MARINE_SC_CLOUD_TAU_SW / (MARINE_SC_CLOUD_TAU_SW + TWO_STREAM_G_WATER)
    )

    # 7. Compute precipitation using moisture flux formulation
    # Convective: uses sub-grid updraft velocity (cloud fraction already encodes instability)
    convective_precip = compute_convective_precipitation(
        convective_frac,
        q,
    )

    # Stratiform: uses large-scale vertical velocity (w > 0 regions)
    stratiform_precip = compute_stratiform_precipitation(
        stratiform_frac,
        q,
        vertical_velocity,  # large-scale w
    )

    # Marine Sc: drizzle via slow autoconversion (these form in subsiding air)
    marine_sc_precip = compute_marine_sc_precipitation(
        marine_sc_frac,
        q,
        T_bl_K,
    )

    # 8. Compute high clouds (anvils from convection + frontal cirrus)
    high_cloud_frac = compute_high_cloud_fraction(convective_frac, stratiform_precip)
    high_cloud_albedo = compute_high_cloud_albedo(T_bl_K.shape)

    return CloudPrecipOutput(
        convective_frac=convective_frac,
        convective_albedo=convective_albedo,
        convective_top_K=convective_top_K,
        convective_precip=convective_precip,
        stratiform_frac=stratiform_frac,
        stratiform_albedo=stratiform_albedo,
        stratiform_top_K=stratiform_top_K,
        stratiform_precip=stratiform_precip,
        marine_sc_frac=marine_sc_frac,
        marine_sc_albedo=marine_sc_albedo,
        marine_sc_top_K=marine_sc_top_K,
        marine_sc_precip=marine_sc_precip,
        high_cloud_frac=high_cloud_frac,
        high_cloud_albedo=high_cloud_albedo,
        high_cloud_top_K=high_cloud_top_K,
        # LW emissivities (from τ_LW via Beer-Lambert)
        eps_convective=eps_conv,
        eps_stratiform=eps_strat,
        eps_marine_sc=eps_marine,
        eps_high=eps_high,
        # SW absorptances (from τ_SW and ω₀)
        abs_sw_convective=abs_sw_conv,
        abs_sw_stratiform=abs_sw_strat,
        abs_sw_marine_sc=abs_sw_marine,
        abs_sw_high=abs_sw_high,
    )
