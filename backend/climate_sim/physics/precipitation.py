"""Precipitation physics.

This module provides all precipitation computations:
- Convective precipitation with RH gate (unified convective + stratiform)
- Orographic precipitation (in orographic_effects.py)
- Precipitation recycling (land only)
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
    R_EARTH_METERS,
)
from climate_sim.physics.humidity import compute_saturation_specific_humidity

# Aliases for readability
GRAVITY = GRAVITY_M_S2
SPECIFIC_HEAT_AIR = HEAT_CAPACITY_AIR_J_KG_K

# Air density at surface (kg/m³)
RHO_AIR = 1.2

# Precipitation efficiency - fraction of moisture flux converted to precipitation.
# Previously 0.30 with cloud-fraction area gating (~0.2), giving effective ~0.06.
# Now that cloud fraction is removed (RH gate only), reduce to compensate.
CONVECTIVE_PRECIP_EFFICIENCY = 0.05

# Grid-mean convective vertical velocity contribution (m/s)
# This represents the effective grid-mean vertical motion from convection.
# In-cloud updrafts are 1-5 m/s, but convection covers only ~5-10% of area,
# so grid-mean contribution is ~0.05-0.5 m/s.
CONVECTIVE_UPDRAFT_VELOCITY = 0.10  # m/s (grid-mean effective)

# Sub-cloud evaporation (virga) parameters.
# In subsidence zones, rain falling from isolated convective cells evaporates
# in the dry, descending environmental air before reaching the surface.
VIRGA_FLOOR = 0.25    # minimum fraction reaching surface in strongest descent
VIRGA_SCALE = 0.005   # m/s, sigmoid transition width
W_BOOST_MAX = 1.5     # maximum amplification in strongest ascent

# Eddy precipitation: moisture wrung out during baroclinic eddy transport.
# C_EDDY has units 1/m — fraction of eddy moisture flux that precipitates
# per meter of transport distance.  Moisture e-folding distance ~2000 km
# gives C ≈ 5e-7 /m.
EDDY_PRECIP_COEFFICIENT = 5e-7  # 1/m


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


def compute_precipitation_rh_gate(rh: np.ndarray) -> np.ndarray:
    """Sundqvist (1989) RH gate for precipitation.

    C = 1 - sqrt((1 - RH) / (1 - RH_crit)) for RH > RH_crit, else 0.
    Smooth, concave-up onset — zero below RH_crit=0.65, reaches 1.0 at RH=1.0.
    """
    rh_clipped = np.clip(rh, 0.0, 0.999)
    RH_CRIT = 0.40  # sub-grid saturation: parts of 5° cell saturate before grid-mean
    return np.where(
        rh_clipped > RH_CRIT,
        1.0 - np.sqrt((1.0 - rh_clipped) / (1.0 - RH_CRIT)),
        0.0,
    )


def compute_convective_precipitation(
    rh: np.ndarray,
    q: np.ndarray,
    w_updraft: float = CONVECTIVE_UPDRAFT_VELOCITY,
    efficiency: float = CONVECTIVE_PRECIP_EFFICIENCY,
    rho: float = RHO_AIR,
    vertical_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Compute precipitation from moisture flux × RH gate.

    P = rh_gate × efficiency × w_updraft × q × ρ × virga_factor

    The RH gate is the sum of convective (Gompertz) and stratiform (Sundqvist)
    RH factors, applied directly rather than through cloud fractions to avoid
    double-counting moisture dependence.

    Parameters
    ----------
    rh : np.ndarray
        Relative humidity (0-1).
    q : np.ndarray
        Specific humidity (kg/kg).
    w_updraft : float
        Effective grid-mean updraft velocity (m/s). Default 0.10.
    efficiency : float
        Precipitation efficiency (0-1). Default 0.30.
    rho : float
        Air density (kg/m³). Default 1.2.
    vertical_velocity : np.ndarray | None
        Large-scale vertical velocity (m/s, positive = rising). When
        negative (descent), sub-cloud evaporation reduces precipitation
        reaching the surface.

    """
    rh_gate = compute_precipitation_rh_gate(rh)
    P_convective = rh_gate * efficiency * w_updraft * q * rho

    # Vertical velocity scaling: smooth factor from VIRGA_FLOOR (strong descent)
    # through ~0.5 at w=0 to W_BOOST_MAX (strong ascent).
    # Replaces both the old virga (descent suppression) and cloud-fraction
    # w_factor (ascent amplification) with a single continuous function.
    if vertical_velocity is not None:
        w_factor = VIRGA_FLOOR + (W_BOOST_MAX - VIRGA_FLOOR) / (
            1.0 + np.exp(-vertical_velocity / VIRGA_SCALE)
        )
        P_convective = P_convective * w_factor

    return P_convective


def compute_eddy_precipitation(
    q: np.ndarray,
    grad_q: np.ndarray,
    kappa: np.ndarray,
    rh: np.ndarray,
    coefficient: float = EDDY_PRECIP_COEFFICIENT,
) -> np.ndarray:
    """Precipitation from baroclinic eddy moisture transport.

    As eddies transport moisture along gradients, rising air in frontal
    zones saturates and precipitates.  The rate scales with the eddy
    moisture flux magnitude:

        P_eddy = C × κ × |∇q| × rh_gate
    """
    rh_gate = compute_precipitation_rh_gate(rh)
    return coefficient * kappa * grad_q * rh_gate


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
    rh: np.ndarray,
    q: np.ndarray,
    T_bl: np.ndarray,
    vertical_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobian of precipitation rate w.r.t. temperatures and humidity.

    The RH gate is treated as frozen (lagged) — only the q dependence in
    P = rh_gate × eff × w × q × ρ is differentiated.

    Parameters
    ----------
    rh : np.ndarray
        Relative humidity (0-1), frozen for Jacobian.
    q : np.ndarray
        Specific humidity (kg/kg).
    T_bl : np.ndarray
        Boundary layer temperature (K).
    vertical_velocity : np.ndarray | None
        Large-scale vertical velocity (m/s).

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
    # Convective precipitation Jacobian
    # P = rh_gate × eff × w_updraft × q × ρ  (rh_gate frozen)
    # ∂P/∂q = rh_gate × eff × w_updraft × ρ
    # =========================================================================
    rh_gate = compute_precipitation_rh_gate(rh)
    dP_conv_dq = (
        rh_gate
        * CONVECTIVE_PRECIP_EFFICIENCY
        * CONVECTIVE_UPDRAFT_VELOCITY
        * RHO_AIR
    )
    # Apply w_factor (descent suppression + ascent amplification)
    if vertical_velocity is not None:
        w_factor = VIRGA_FLOOR + (W_BOOST_MAX - VIRGA_FLOOR) / (
            1.0 + np.exp(-vertical_velocity / VIRGA_SCALE)
        )
        dP_conv_dq = dP_conv_dq * w_factor
    dP_dq += dP_conv_dq

    return dP_dT_bl, dP_dT_atm, dP_dq


def compute_precipitation_recycling(
    evap_rate: np.ndarray,
    humidity_q: np.ndarray,
    wind_speed: np.ndarray,
    land_mask: np.ndarray,
    resolution_deg: float = 5.0,
    vertical_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Precipitation recycling (Eltahir & Bras 1996): ρ = E·L / (q·U·H + E·L).

    Local convection recaptures evaporated moisture before wind transports
    it out of the grid cell.  Land only.  Suppressed by sub-cloud evaporation
    (virga) in subsidence zones, same as convective precipitation.
    """
    L = R_EARTH_METERS * np.deg2rad(resolution_deg)
    H = 2000.0  # moisture scale height (m)

    E = np.maximum(evap_rate, 0.0)
    moisture_flux = np.maximum(humidity_q, 1e-6) * np.maximum(wind_speed, 0.5) * H
    local_supply = E * L

    recycled = E * local_supply / (moisture_flux + local_supply)

    # Sub-cloud evaporation suppresses recycling in descent zones only
    if vertical_velocity is not None:
        sigmoid_w = 1.0 / (1.0 + np.exp(-vertical_velocity / VIRGA_SCALE))
        descent_factor = VIRGA_FLOOR + (1.0 - VIRGA_FLOOR) * sigmoid_w
        virga_factor = np.where(vertical_velocity >= 0.0, 1.0, descent_factor)
        recycled = recycled * virga_factor

    return np.where(land_mask, recycled, 0.0)


def compute_precipitation_recycling_jacobian(
    evap_rate: np.ndarray,
    humidity_q: np.ndarray,
    wind_speed: np.ndarray,
    land_mask: np.ndarray,
    resolution_deg: float = 5.0,
    vertical_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Jacobian of precipitation recycling w.r.t. humidity.

    dP_rec/dq = dR/dq × virga_factor

    The virga factor depends only on vertical_velocity (lagged), so it's
    a constant multiplier on dR/dq.  dR/dq is always negative (more q →
    larger denominator → lower recycling ratio).
    """
    L = R_EARTH_METERS * np.deg2rad(resolution_deg)
    H = 2000.0

    E = np.maximum(evap_rate, 0.0)
    V = np.maximum(wind_speed, 0.5)
    q = np.maximum(humidity_q, 1e-6)
    denom = q * V * H + E * L
    dR_dq = -E * E * L * V * H / np.maximum(denom * denom, 1e-30)

    if vertical_velocity is not None:
        sigmoid_w = 1.0 / (1.0 + np.exp(-vertical_velocity / VIRGA_SCALE))
        descent_factor = VIRGA_FLOOR + (1.0 - VIRGA_FLOOR) * sigmoid_w
        virga_factor = np.where(vertical_velocity >= 0.0, 1.0, descent_factor)
        result = dR_dq * virga_factor
    else:
        result = dR_dq

    return np.where(land_mask, result, 0.0)
