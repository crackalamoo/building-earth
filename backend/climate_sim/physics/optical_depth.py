"""Optical depth based radiation calculations.

This module computes effective emission temperatures by finding where τ=1
in the atmospheric column, properly accounting for water vapor distribution
and cloud optical depth.

Key concepts:
- τ is additive through the column
- Emission to space comes from τ≈1 level (looking down from TOA)
- Emission to surface comes from τ≈1 level (looking up from surface)
- T_eff is the temperature at the τ=1 level

The τ=1 level is found analytically using the Lambert W function.
"""

import numpy as np
from scipy.special import lambertw

from climate_sim.data.constants import (
    ATMOSPHERE_LAYER_HEIGHT_M,
    BOUNDARY_LAYER_HEIGHT_M,
    STANDARD_LAPSE_RATE_K_PER_M,
    STEFAN_BOLTZMANN_W_M2_K4,
)


# Physical constants for optical depth calculations
WATER_VAPOR_SCALE_HEIGHT_M = 2000.0  # ~2 km observed globally

# Mass absorption coefficient for liquid water in thermal IR (m²/kg)
LIQUID_WATER_MASS_ABSORPTION_COEF = 130.0  # m²/kg

# Optical depth from CO2 and other well-mixed gases
# τ = -ln(1 - ε) → τ_CO2 ≈ 0.60 for ε = 0.45
TAU_CO2_BASELINE = 0.60

# Mass absorption coefficient for water vapor (m²/kg effective for broadband LW)
WATER_VAPOR_MASS_ABSORPTION_COEF = 0.033  # m²/kg


def compute_water_vapor_optical_depth(
    humidity_q: np.ndarray,
    layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute total column water vapor optical depth from specific humidity.

    Parameters
    ----------
    humidity_q : np.ndarray
        Specific humidity at surface/BL level (kg/kg).
    layer_height_m : float
        Total atmospheric column height (m).

    Returns
    -------
    np.ndarray
        Water vapor optical depth (dimensionless).
    """
    rho_air_surface = 1.2  # kg/m³
    rho_wv_surface = rho_air_surface * humidity_q

    # Water vapor path: WVP = ρ_wv_0 × H_wv × (1 - exp(-z_top/H_wv))
    H = WATER_VAPOR_SCALE_HEIGHT_M
    wvp = rho_wv_surface * H * (1.0 - np.exp(-layer_height_m / H))

    return WATER_VAPOR_MASS_ABSORPTION_COEF * wvp


def compute_cloud_optical_depth(
    lwc: np.ndarray,
    thickness_m: float,
) -> np.ndarray:
    """Compute cloud optical depth from liquid water content.

    Parameters
    ----------
    lwc : np.ndarray
        Liquid water content (kg/m³).
    thickness_m : float
        Cloud geometric thickness (m).

    Returns
    -------
    np.ndarray
        Cloud optical depth (dimensionless).
    """
    lwp = lwc * thickness_m
    return LIQUID_WATER_MASS_ABSORPTION_COEF * lwp


def find_tau_one_height_from_top(
    tau_co2: float,
    tau_h2o: np.ndarray,
    z_top: float,
    H: float = WATER_VAPOR_SCALE_HEIGHT_M,
) -> np.ndarray:
    """Find height where τ=1 looking down from TOA using Lambert W.

    The optical depth from TOA to height z is:
        τ(z) = τ_CO2 × (z_top - z)/z_top + τ_H2O × (exp(-z/H) - β)/(1-β)
    where β = exp(-z_top/H).

    Setting τ = 1 and solving for z using Lambert W.

    Parameters
    ----------
    tau_co2 : float
        CO2 optical depth (uniform distribution).
    tau_h2o : np.ndarray
        Water vapor optical depth (exponential distribution).
    z_top : float
        Top of atmosphere height (m).
    H : float
        Water vapor scale height (m).

    Returns
    -------
    np.ndarray
        Height z where τ=1 (m). Clipped to [0, z_top].
    """
    beta = np.exp(-z_top / H)

    # Coefficients for the equation: B×u + E×exp(u) = F
    # where u = c/H, c = z_top - z (depth from TOA)
    a = tau_co2 / z_top  # τ per unit depth for CO2
    B = a * H
    E = tau_h2o * beta / (1.0 - beta + 1e-10)
    F = 1.0 + E

    # Handle edge cases
    tau_total = tau_co2 + tau_h2o
    result = np.zeros_like(tau_h2o)

    # Case 1: Optically thin (τ_total < 1) - τ=1 never reached
    thin_mask = tau_total < 1.0
    # For thin atmospheres, use τ_total/2 as effective level
    # This corresponds to the emission-weighted mean
    if np.any(thin_mask):
        # Approximate: find z where τ = τ_total/2
        # For small τ, most emission is from near surface
        result[thin_mask] = 0.0  # Surface

    # Case 2: CO2 dominated (τ_H2O ≈ 0)
    co2_dominated = (~thin_mask) & (tau_h2o < 0.01)
    if np.any(co2_dominated):
        # τ = τ_CO2 × (z_top - z)/z_top = 1
        # z = z_top × (1 - 1/τ_CO2)
        z_co2 = z_top * (1.0 - 1.0 / (tau_co2 + 1e-10))
        result[co2_dominated] = np.maximum(z_co2, 0.0)

    # Case 3: H2O dominated (τ_CO2 ≈ 0)
    h2o_dominated = (~thin_mask) & (~co2_dominated) & (tau_co2 < 0.01)
    if np.any(h2o_dominated):
        # τ = τ_H2O × (exp(-z/H) - β)/(1-β) = 1
        # exp(-z/H) = β + (1-β)/τ_H2O
        arg = beta + (1.0 - beta) / tau_h2o[h2o_dominated]
        arg = np.clip(arg, 1e-10, 1.0)  # Must be > 0 and < 1 for valid z > 0
        result[h2o_dominated] = -H * np.log(arg)

    # Case 4: General case - use Lambert W
    general = (~thin_mask) & (~co2_dominated) & (~h2o_dominated)
    if np.any(general):
        B_g = B
        E_g = E[general]
        F_g = F[general]

        # Solve: B×u + E×exp(u) = F
        # Lambert W form: w×exp(w) = (E/B)×exp(F/B)
        # Solution: w = W((E/B)×exp(F/B)), u = F/B - w

        # Compute argument to Lambert W
        # Use log to avoid overflow: log(arg) = log(E/B) + F/B
        log_arg = np.log(E_g / B_g + 1e-300) + F_g / B_g

        # Lambert W of exp(log_arg)
        # W(exp(x)) for large x ≈ x - ln(x) + ln(x)/x + ...
        # But scipy.special.lambertw handles this
        arg = np.exp(np.clip(log_arg, -700, 700))  # Avoid overflow
        w = np.real(lambertw(arg))

        u = F_g / B_g - w
        c = H * u  # Depth from TOA
        z = z_top - c

        result[general] = np.clip(z, 0.0, z_top)

    return result


def find_tau_one_height_from_bottom(
    tau_co2: float,
    tau_h2o: np.ndarray,
    z_top: float,
    H: float = WATER_VAPOR_SCALE_HEIGHT_M,
) -> np.ndarray:
    """Find height where τ=1 looking up from surface using Lambert W.

    The optical depth from surface to height z is:
        τ(z) = τ_CO2 × z/z_top + τ_H2O × (1 - exp(-z/H))/(1-β)
    where β = exp(-z_top/H).

    Parameters
    ----------
    tau_co2 : float
        CO2 optical depth.
    tau_h2o : np.ndarray
        Water vapor optical depth.
    z_top : float
        Top of atmosphere height (m).
    H : float
        Water vapor scale height (m).

    Returns
    -------
    np.ndarray
        Height z where τ=1 (m). Clipped to [0, z_top].
    """
    beta = np.exp(-z_top / H)

    # Coefficients: B×v + D×exp(v) = G
    # where v = -z/H (so z = -H×v)
    a = tau_co2 / z_top
    B = a * H
    D = tau_h2o / (1.0 - beta + 1e-10)
    G = D - 1.0

    tau_total = tau_co2 + tau_h2o
    result = np.zeros_like(tau_h2o)

    # Case 1: Optically thin
    thin_mask = tau_total < 1.0
    if np.any(thin_mask):
        result[thin_mask] = z_top  # TOA (looking up, never reach τ=1)

    # Case 2: CO2 dominated
    co2_dominated = (~thin_mask) & (tau_h2o < 0.01)
    if np.any(co2_dominated):
        # τ = τ_CO2 × z/z_top = 1 → z = z_top/τ_CO2
        z_co2 = z_top / (tau_co2 + 1e-10)
        result[co2_dominated] = np.minimum(z_co2, z_top)

    # Case 3: H2O dominated
    h2o_dominated = (~thin_mask) & (~co2_dominated) & (tau_co2 < 0.01)
    if np.any(h2o_dominated):
        # τ = τ_H2O × (1 - exp(-z/H))/(1-β) = 1
        # 1 - exp(-z/H) = (1-β)/τ_H2O
        # exp(-z/H) = 1 - (1-β)/τ_H2O
        arg = 1.0 - (1.0 - beta) / tau_h2o[h2o_dominated]
        arg = np.clip(arg, 1e-10, 1.0)
        result[h2o_dominated] = -H * np.log(arg)

    # Case 4: General case
    general = (~thin_mask) & (~co2_dominated) & (~h2o_dominated)
    if np.any(general):
        B_g = B
        D_g = D[general]
        G_g = G[general]

        # Solve: B×v + D×exp(v) = G where v = -z/H
        # Same Lambert W form: w×exp(w) = (D/B)×exp(G/B)
        log_arg = np.log(D_g / B_g + 1e-300) + G_g / B_g
        arg = np.exp(np.clip(log_arg, -700, 700))
        w = np.real(lambertw(arg))

        v = G_g / B_g - w
        z = -H * v

        result[general] = np.clip(z, 0.0, z_top)

    return result


def compute_temperature_at_height(
    z: np.ndarray,
    T_surface: np.ndarray,
    T_bl: np.ndarray,
    T_atm: np.ndarray | None = None,
    z_bl: float = BOUNDARY_LAYER_HEIGHT_M,
    z_atm: float = ATMOSPHERE_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute temperature at height z using a three-layer profile.

    The temperature profile is:
    - z < z_bl: Linear interpolation from T_surface to T_bl
    - z_bl < z < z_bl + z_atm: Linear interpolation from T_bl to T_atm
    - z > z_bl + z_atm: Continue with standard lapse rate from T_atm

    Parameters
    ----------
    z : np.ndarray
        Height (m).
    T_surface : np.ndarray
        Surface temperature (K).
    T_bl : np.ndarray
        Boundary layer temperature (K).
    T_atm : np.ndarray, optional
        Free atmosphere temperature (K). If None, uses lapse rate from T_bl.
    z_bl : float
        Boundary layer height (m).
    z_atm : float
        Atmosphere layer height (m).

    Returns
    -------
    np.ndarray
        Temperature at height z (K).
    """
    z_top = z_bl + z_atm

    # Below BL: linear interpolation from surface to BL
    in_bl = z <= z_bl
    frac_bl = np.minimum(z / z_bl, 1.0)
    T_in_bl = T_surface * (1 - frac_bl) + T_bl * frac_bl

    if T_atm is None:
        # No atmosphere temp provided - use lapse rate from BL
        z_above_bl = np.maximum(z - z_bl, 0.0)
        T_above_bl = T_bl - STANDARD_LAPSE_RATE_K_PER_M * z_above_bl
    else:
        # In free atmosphere: linear interpolation from T_bl to T_atm
        in_atm = (z > z_bl) & (z <= z_top)
        frac_atm = np.clip((z - z_bl) / z_atm, 0.0, 1.0)
        T_in_atm = T_bl * (1 - frac_atm) + T_atm * frac_atm

        # Above atmosphere layer: continue with lapse rate from T_atm
        z_above_atm = np.maximum(z - z_top, 0.0)
        T_above_atm = T_atm - STANDARD_LAPSE_RATE_K_PER_M * z_above_atm

        T_above_bl = np.where(in_atm, T_in_atm, T_above_atm)

    T = np.where(in_bl, T_in_bl, T_above_bl)

    # Minimum temperature (stratosphere)
    return np.maximum(T, 200.0)


def compute_temperature_derivatives_at_height(
    z: np.ndarray,
    z_bl: float = BOUNDARY_LAYER_HEIGHT_M,
    z_atm: float = ATMOSPHERE_LAYER_HEIGHT_M,
    use_T_atm: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute derivatives of T(z) with respect to T_surface, T_bl, and T_atm.

    Using chain rule: T(z) = f(T_surface, T_bl, T_atm, z)

    For z <= z_bl:
        T = T_surface × (1 - z/z_bl) + T_bl × (z/z_bl)
        dT/dT_surface = 1 - z/z_bl
        dT/dT_bl = z/z_bl
        dT/dT_atm = 0

    For z_bl < z <= z_bl + z_atm:
        T = T_bl × (1 - frac) + T_atm × frac, where frac = (z - z_bl) / z_atm
        dT/dT_surface = 0
        dT/dT_bl = 1 - frac
        dT/dT_atm = frac

    For z > z_bl + z_atm:
        T = T_atm - lapse × (z - z_top)
        dT/dT_surface = 0
        dT/dT_bl = 0
        dT/dT_atm = 1

    Parameters
    ----------
    z : np.ndarray
        Height (m).
    z_bl : float
        Boundary layer height (m).
    z_atm : float
        Atmosphere layer height (m).
    use_T_atm : bool
        If True, use three-layer profile with T_atm. If False, use two-layer.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (dT/dT_surface, dT/dT_bl, dT/dT_atm) - derivatives with same shape as z.
    """
    z_top = z_bl + z_atm

    # Region masks
    in_bl = z <= z_bl
    in_atm = (z > z_bl) & (z <= z_top)

    # Below BL: interpolation between surface and BL
    frac_bl = np.minimum(z / z_bl, 1.0)
    dT_dTs_bl = 1.0 - frac_bl
    dT_dTbl_bl = frac_bl
    dT_dTatm_bl = np.zeros_like(z)

    if use_T_atm:
        # In free atmosphere: interpolation between BL and atm
        frac_atm = np.clip((z - z_bl) / z_atm, 0.0, 1.0)
        dT_dTs_atm = np.zeros_like(z)
        dT_dTbl_atm = 1.0 - frac_atm
        dT_dTatm_atm = frac_atm

        # Above atmosphere: lapse rate from T_atm
        dT_dTs_above = np.zeros_like(z)
        dT_dTbl_above = np.zeros_like(z)
        dT_dTatm_above = np.ones_like(z)

        # Combine by region
        dT_dTsurface = np.where(in_bl, dT_dTs_bl, np.where(in_atm, dT_dTs_atm, dT_dTs_above))
        dT_dTbl = np.where(in_bl, dT_dTbl_bl, np.where(in_atm, dT_dTbl_atm, dT_dTbl_above))
        dT_dTatm = np.where(in_bl, dT_dTatm_bl, np.where(in_atm, dT_dTatm_atm, dT_dTatm_above))
    else:
        # Two-layer mode: above BL uses lapse rate from T_bl
        dT_dTsurface = np.where(in_bl, dT_dTs_bl, 0.0)
        dT_dTbl = np.where(in_bl, dT_dTbl_bl, 1.0)
        dT_dTatm = np.zeros_like(z)

    return dT_dTsurface, dT_dTbl, dT_dTatm


def compute_effective_temperatures_analytical(
    humidity_q: np.ndarray,
    T_surface: np.ndarray,
    T_bl: np.ndarray,
    T_atm: np.ndarray | None = None,
    tau_cloud_from_top: np.ndarray | None = None,
    cloud_top_m: float | None = None,
    tau_cloud_from_bottom: np.ndarray | None = None,
    cloud_base_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute effective emission temperatures using analytical τ=1 finding.

    Parameters
    ----------
    humidity_q : np.ndarray
        Specific humidity (kg/kg).
    T_surface : np.ndarray
        Surface temperature (K).
    T_bl : np.ndarray
        Boundary layer temperature (K).
    T_atm : np.ndarray, optional
        Free atmosphere temperature (K). If None, uses lapse rate from T_bl.
    tau_cloud_from_top : np.ndarray, optional
        Additional cloud τ for downward-looking emission (OLR).
    cloud_top_m : float, optional
        Cloud top height for OLR calculation.
    tau_cloud_from_bottom : np.ndarray, optional
        Additional cloud τ for upward-looking emission.
    cloud_base_m : float, optional
        Cloud base height for downward emission calculation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (T_eff_up, T_eff_down, emissivity)
    """
    z_top = BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M
    H = WATER_VAPOR_SCALE_HEIGHT_M

    # Clear-sky optical depths
    tau_co2 = TAU_CO2_BASELINE
    tau_h2o = compute_water_vapor_optical_depth(humidity_q, z_top)

    # Total optical depth (for emissivity calculation)
    tau_total_clear = tau_co2 + tau_h2o

    # === Upward emission (OLR) ===
    if tau_cloud_from_top is not None and cloud_top_m is not None:
        # Cloud adds τ at cloud top level
        # If τ_clear(cloud_top) + τ_cloud > 1, τ=1 is at or above cloud top
        # Otherwise, τ=1 is below cloud top (in clear air below cloud)

        # τ from TOA to cloud top (clear sky only)
        z_cloud_top = cloud_top_m
        c_cloud = z_top - z_cloud_top  # Depth from TOA to cloud top
        tau_co2_to_cloud = tau_co2 * c_cloud / z_top
        beta = np.exp(-z_top / H)
        exp_cloud = np.exp(-z_cloud_top / H)
        tau_h2o_to_cloud = tau_h2o * (exp_cloud - beta) / (1.0 - beta + 1e-10)
        tau_clear_to_cloud = tau_co2_to_cloud + tau_h2o_to_cloud

        # Check if τ=1 is above or below cloud
        tau_at_cloud_top = tau_clear_to_cloud + tau_cloud_from_top
        above_cloud = tau_clear_to_cloud >= 1.0  # τ=1 reached before cloud
        in_or_below_cloud = ~above_cloud

        z_tau1_up = np.zeros_like(humidity_q)

        if np.any(above_cloud):
            # τ=1 is in clear sky above cloud - use clear-sky solution
            z_tau1_up[above_cloud] = find_tau_one_height_from_top(
                tau_co2, tau_h2o[above_cloud], z_top, H
            )

        if np.any(in_or_below_cloud):
            # τ=1 is at cloud top or below
            # For simplicity: if τ_at_cloud_top >= 1, τ=1 is at cloud top
            # (thick cloud approximation - emission from cloud top)
            at_cloud_top = in_or_below_cloud & (tau_at_cloud_top >= 1.0)
            z_tau1_up[at_cloud_top] = z_cloud_top

            # If τ < 1 even with cloud, continue to surface (rare for thick clouds)
            below_cloud = in_or_below_cloud & (tau_at_cloud_top < 1.0)
            if np.any(below_cloud):
                # Need τ = 1 - tau_at_cloud_top more below cloud
                # For simplicity, use surface (cloud is too thin to matter)
                z_tau1_up[below_cloud] = 0.0

        tau_total_up = tau_total_clear + tau_cloud_from_top
    else:
        z_tau1_up = find_tau_one_height_from_top(tau_co2, tau_h2o, z_top, H)
        tau_total_up = tau_total_clear

    T_eff_up = compute_temperature_at_height(z_tau1_up, T_surface, T_bl, T_atm)

    # === Downward emission ===
    if tau_cloud_from_bottom is not None and cloud_base_m is not None:
        # Cloud adds τ at cloud base level
        z_cloud_base = cloud_base_m
        tau_co2_to_cloud = tau_co2 * z_cloud_base / z_top
        beta = np.exp(-z_top / H)
        tau_h2o_to_cloud = tau_h2o * (1.0 - np.exp(-z_cloud_base / H)) / (1.0 - beta + 1e-10)
        tau_clear_to_cloud = tau_co2_to_cloud + tau_h2o_to_cloud

        tau_at_cloud_base = tau_clear_to_cloud + tau_cloud_from_bottom
        below_cloud = tau_clear_to_cloud >= 1.0  # τ=1 below cloud base
        in_or_above_cloud = ~below_cloud

        z_tau1_down = np.zeros_like(humidity_q)

        if np.any(below_cloud):
            z_tau1_down[below_cloud] = find_tau_one_height_from_bottom(
                tau_co2, tau_h2o[below_cloud], z_top, H
            )

        if np.any(in_or_above_cloud):
            at_cloud_base = in_or_above_cloud & (tau_at_cloud_base >= 1.0)
            z_tau1_down[at_cloud_base] = z_cloud_base

            above_cloud = in_or_above_cloud & (tau_at_cloud_base < 1.0)
            if np.any(above_cloud):
                z_tau1_down[above_cloud] = z_top

        tau_total_down = tau_total_clear + tau_cloud_from_bottom
    else:
        z_tau1_down = find_tau_one_height_from_bottom(tau_co2, tau_h2o, z_top, H)
        tau_total_down = tau_total_clear

    T_eff_down = compute_temperature_at_height(z_tau1_down, T_surface, T_bl, T_atm)

    # Emissivity from total τ (average of up and down for simplicity)
    tau_total = 0.5 * (tau_total_up + tau_total_down)
    emissivity = 1.0 - np.exp(-tau_total)

    return T_eff_up, T_eff_down, emissivity


def compute_column_radiation_with_clouds(
    humidity_q: np.ndarray,
    T_surface: np.ndarray,
    T_bl: np.ndarray,
    T_atm: np.ndarray,
    conv_frac: np.ndarray,
    conv_lwc: np.ndarray,
    conv_base_m: float,
    conv_top_m: float,
    strat_frac: np.ndarray,
    strat_lwc: np.ndarray,
    strat_base_m: float,
    strat_top_m: float,
    n_levels: int = 50,  # Unused, kept for API compatibility
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute OLR and downward emission using analytical optical depth approach.

    Uses Lambert W function to find τ=1 levels analytically rather than
    numerically iterating through a discretized profile.

    Parameters
    ----------
    humidity_q : np.ndarray
        Specific humidity (kg/kg).
    T_surface : np.ndarray
        Surface temperature (K).
    T_bl : np.ndarray
        Boundary layer temperature (K).
    T_atm : np.ndarray
        Free atmosphere temperature (K). (Currently unused - T profile from lapse rate)
    conv_frac : np.ndarray
        Convective cloud fraction (0-1).
    conv_lwc : np.ndarray
        Convective cloud LWC (kg/m³).
    conv_base_m : float
        Convective cloud base height (m).
    conv_top_m : float
        Convective cloud top height (m).
    strat_frac : np.ndarray
        Stratiform cloud fraction (0-1).
    strat_lwc : np.ndarray
        Stratiform cloud LWC (kg/m³).
    strat_base_m : float
        Stratiform cloud base height (m).
    strat_top_m : float
        Stratiform cloud top height (m).
    n_levels : int
        Unused (kept for API compatibility).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (OLR, downward_emission, total_emissivity) in W/m².
    """
    sigma = STEFAN_BOLTZMANN_W_M2_K4

    # Compute cloud optical depths
    conv_thickness = conv_top_m - conv_base_m
    strat_thickness = strat_top_m - strat_base_m
    tau_conv = compute_cloud_optical_depth(conv_lwc, conv_thickness)
    tau_strat = compute_cloud_optical_depth(strat_lwc, strat_thickness)

    # Fractions
    clear_frac = np.maximum(1.0 - conv_frac - strat_frac, 0.0)

    # === Clear sky ===
    T_eff_up_clear, T_eff_down_clear, eps_clear = compute_effective_temperatures_analytical(
        humidity_q, T_surface, T_bl, T_atm
    )

    # === Convective clouds ===
    # OLR: cloud adds τ at cloud top
    # Downward: cloud adds τ at cloud base
    T_eff_up_conv, T_eff_down_conv, eps_conv = compute_effective_temperatures_analytical(
        humidity_q,
        T_surface,
        T_bl,
        T_atm,
        tau_cloud_from_top=tau_conv,
        cloud_top_m=conv_top_m,
        tau_cloud_from_bottom=tau_conv,
        cloud_base_m=conv_base_m,
    )

    # === Stratiform clouds ===
    T_eff_up_strat, T_eff_down_strat, eps_strat = compute_effective_temperatures_analytical(
        humidity_q,
        T_surface,
        T_bl,
        T_atm,
        tau_cloud_from_top=tau_strat,
        cloud_top_m=strat_top_m,
        tau_cloud_from_bottom=tau_strat,
        cloud_base_m=strat_base_m,
    )

    # === Compute fluxes for each column type ===
    # OLR = ε × σ × T_eff_up⁴ + (1-ε) × σ × T_surface⁴
    olr_clear = eps_clear * sigma * T_eff_up_clear**4 + (1.0 - eps_clear) * sigma * T_surface**4
    olr_conv = eps_conv * sigma * T_eff_up_conv**4 + (1.0 - eps_conv) * sigma * T_surface**4
    olr_strat = eps_strat * sigma * T_eff_up_strat**4 + (1.0 - eps_strat) * sigma * T_surface**4

    # Downward = ε × σ × T_eff_down⁴
    down_clear = eps_clear * sigma * T_eff_down_clear**4
    down_conv = eps_conv * sigma * T_eff_down_conv**4
    down_strat = eps_strat * sigma * T_eff_down_strat**4

    # === Area-weighted averages ===
    olr = clear_frac * olr_clear + conv_frac * olr_conv + strat_frac * olr_strat
    downward = clear_frac * down_clear + conv_frac * down_conv + strat_frac * down_strat
    total_emissivity = clear_frac * eps_clear + conv_frac * eps_conv + strat_frac * eps_strat

    return olr, downward, total_emissivity


def compute_radiation_jacobian_with_clouds(
    humidity_q: np.ndarray,
    T_surface: np.ndarray,
    T_bl: np.ndarray,
    T_atm: np.ndarray,
    conv_frac: np.ndarray,
    conv_lwc: np.ndarray,
    conv_base_m: float,
    conv_top_m: float,
    strat_frac: np.ndarray,
    strat_lwc: np.ndarray,
    strat_base_m: float,
    strat_top_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobian of radiation with respect to T_surface, T_bl, and T_atm.

    Uses chain rule:
        d(OLR)/d(T_x) = d(OLR)/d(T_eff) × d(T_eff)/d(T_x)

    where T_eff depends on T_surface, T_bl, and T_atm through the temperature profile.

    Parameters
    ----------
    humidity_q : np.ndarray
        Specific humidity (kg/kg).
    T_surface : np.ndarray
        Surface temperature (K).
    T_bl : np.ndarray
        Boundary layer temperature (K).
    T_atm : np.ndarray
        Free atmosphere temperature (K).
    conv_frac, strat_frac : np.ndarray
        Cloud fractions.
    conv_lwc, strat_lwc : np.ndarray
        Liquid water contents.
    conv_base_m, conv_top_m, strat_base_m, strat_top_m : float
        Cloud geometry.

    Returns
    -------
    tuple of 6 np.ndarray
        (d_olr_dTs, d_olr_dTbl, d_olr_dTatm, d_down_dTs, d_down_dTbl, d_down_dTatm)
    """
    sigma = STEFAN_BOLTZMANN_W_M2_K4
    z_top = BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M
    H = WATER_VAPOR_SCALE_HEIGHT_M

    # Compute cloud optical depths
    conv_thickness = conv_top_m - conv_base_m
    strat_thickness = strat_top_m - strat_base_m
    tau_conv = compute_cloud_optical_depth(conv_lwc, conv_thickness)
    tau_strat = compute_cloud_optical_depth(strat_lwc, strat_thickness)

    # Clear-sky optical depths
    tau_co2 = TAU_CO2_BASELINE
    tau_h2o = compute_water_vapor_optical_depth(humidity_q, z_top)

    # Fractions
    clear_frac = np.maximum(1.0 - conv_frac - strat_frac, 0.0)

    # === Find z_tau1 levels for each column type ===

    # Clear sky
    z_tau1_up_clear = find_tau_one_height_from_top(tau_co2, tau_h2o, z_top, H)
    z_tau1_down_clear = find_tau_one_height_from_bottom(tau_co2, tau_h2o, z_top, H)
    T_eff_up_clear = compute_temperature_at_height(z_tau1_up_clear, T_surface, T_bl, T_atm)
    T_eff_down_clear = compute_temperature_at_height(z_tau1_down_clear, T_surface, T_bl, T_atm)
    eps_clear = 1.0 - np.exp(-(tau_co2 + tau_h2o))

    # Convective clouds - approximate: τ=1 at cloud top/base for thick clouds
    z_tau1_up_conv = np.full_like(humidity_q, conv_top_m)
    z_tau1_down_conv = np.full_like(humidity_q, conv_base_m)
    T_eff_up_conv = compute_temperature_at_height(z_tau1_up_conv, T_surface, T_bl, T_atm)
    T_eff_down_conv = compute_temperature_at_height(z_tau1_down_conv, T_surface, T_bl, T_atm)
    eps_conv = 1.0 - np.exp(-(tau_co2 + tau_h2o + tau_conv))

    # Stratiform clouds
    z_tau1_up_strat = np.full_like(humidity_q, strat_top_m)
    z_tau1_down_strat = np.full_like(humidity_q, strat_base_m)
    T_eff_up_strat = compute_temperature_at_height(z_tau1_up_strat, T_surface, T_bl, T_atm)
    T_eff_down_strat = compute_temperature_at_height(z_tau1_down_strat, T_surface, T_bl, T_atm)
    eps_strat = 1.0 - np.exp(-(tau_co2 + tau_h2o + tau_strat))

    # === Compute temperature derivatives at each z_tau1 level ===
    # d(T_eff)/d(T_surface), d(T_eff)/d(T_bl), d(T_eff)/d(T_atm)

    dT_up_clear_dTs, dT_up_clear_dTbl, dT_up_clear_dTatm = (
        compute_temperature_derivatives_at_height(z_tau1_up_clear)
    )
    dT_down_clear_dTs, dT_down_clear_dTbl, dT_down_clear_dTatm = (
        compute_temperature_derivatives_at_height(z_tau1_down_clear)
    )

    dT_up_conv_dTs, dT_up_conv_dTbl, dT_up_conv_dTatm = compute_temperature_derivatives_at_height(
        z_tau1_up_conv
    )
    dT_down_conv_dTs, dT_down_conv_dTbl, dT_down_conv_dTatm = (
        compute_temperature_derivatives_at_height(z_tau1_down_conv)
    )

    dT_up_strat_dTs, dT_up_strat_dTbl, dT_up_strat_dTatm = (
        compute_temperature_derivatives_at_height(z_tau1_up_strat)
    )
    dT_down_strat_dTs, dT_down_strat_dTbl, dT_down_strat_dTatm = (
        compute_temperature_derivatives_at_height(z_tau1_down_strat)
    )

    # === Compute flux derivatives using chain rule ===

    # OLR = ε × σ × T_eff_up⁴ + (1-ε) × σ × T_surface⁴
    # d(OLR)/d(T_x) = ε × 4σT_eff³ × d(T_eff)/d(T_x) + [additional term for T_surface]

    # Clear sky
    d_olr_clear_dTs = (
        eps_clear * 4.0 * sigma * T_eff_up_clear**3 * dT_up_clear_dTs
        + (1.0 - eps_clear) * 4.0 * sigma * T_surface**3
    )
    d_olr_clear_dTbl = eps_clear * 4.0 * sigma * T_eff_up_clear**3 * dT_up_clear_dTbl
    d_olr_clear_dTatm = eps_clear * 4.0 * sigma * T_eff_up_clear**3 * dT_up_clear_dTatm

    # Convective
    d_olr_conv_dTs = (
        eps_conv * 4.0 * sigma * T_eff_up_conv**3 * dT_up_conv_dTs
        + (1.0 - eps_conv) * 4.0 * sigma * T_surface**3
    )
    d_olr_conv_dTbl = eps_conv * 4.0 * sigma * T_eff_up_conv**3 * dT_up_conv_dTbl
    d_olr_conv_dTatm = eps_conv * 4.0 * sigma * T_eff_up_conv**3 * dT_up_conv_dTatm

    # Stratiform
    d_olr_strat_dTs = (
        eps_strat * 4.0 * sigma * T_eff_up_strat**3 * dT_up_strat_dTs
        + (1.0 - eps_strat) * 4.0 * sigma * T_surface**3
    )
    d_olr_strat_dTbl = eps_strat * 4.0 * sigma * T_eff_up_strat**3 * dT_up_strat_dTbl
    d_olr_strat_dTatm = eps_strat * 4.0 * sigma * T_eff_up_strat**3 * dT_up_strat_dTatm

    # Area-weighted
    d_olr_dTs = (
        clear_frac * d_olr_clear_dTs + conv_frac * d_olr_conv_dTs + strat_frac * d_olr_strat_dTs
    )
    d_olr_dTbl = (
        clear_frac * d_olr_clear_dTbl + conv_frac * d_olr_conv_dTbl + strat_frac * d_olr_strat_dTbl
    )
    d_olr_dTatm = (
        clear_frac * d_olr_clear_dTatm
        + conv_frac * d_olr_conv_dTatm
        + strat_frac * d_olr_strat_dTatm
    )

    # Downward = ε × σ × T_eff_down⁴
    # d(Down)/d(T_x) = ε × 4σT_eff³ × d(T_eff)/d(T_x)

    d_down_clear_dTs = eps_clear * 4.0 * sigma * T_eff_down_clear**3 * dT_down_clear_dTs
    d_down_clear_dTbl = eps_clear * 4.0 * sigma * T_eff_down_clear**3 * dT_down_clear_dTbl
    d_down_clear_dTatm = eps_clear * 4.0 * sigma * T_eff_down_clear**3 * dT_down_clear_dTatm

    d_down_conv_dTs = eps_conv * 4.0 * sigma * T_eff_down_conv**3 * dT_down_conv_dTs
    d_down_conv_dTbl = eps_conv * 4.0 * sigma * T_eff_down_conv**3 * dT_down_conv_dTbl
    d_down_conv_dTatm = eps_conv * 4.0 * sigma * T_eff_down_conv**3 * dT_down_conv_dTatm

    d_down_strat_dTs = eps_strat * 4.0 * sigma * T_eff_down_strat**3 * dT_down_strat_dTs
    d_down_strat_dTbl = eps_strat * 4.0 * sigma * T_eff_down_strat**3 * dT_down_strat_dTbl
    d_down_strat_dTatm = eps_strat * 4.0 * sigma * T_eff_down_strat**3 * dT_down_strat_dTatm

    d_down_dTs = (
        clear_frac * d_down_clear_dTs + conv_frac * d_down_conv_dTs + strat_frac * d_down_strat_dTs
    )
    d_down_dTbl = (
        clear_frac * d_down_clear_dTbl
        + conv_frac * d_down_conv_dTbl
        + strat_frac * d_down_strat_dTbl
    )
    d_down_dTatm = (
        clear_frac * d_down_clear_dTatm
        + conv_frac * d_down_conv_dTatm
        + strat_frac * d_down_strat_dTatm
    )

    return d_olr_dTs, d_olr_dTbl, d_olr_dTatm, d_down_dTs, d_down_dTbl, d_down_dTatm
