"""Radiative column model components with optional atmospheric layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from climate_sim.core.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.data.constants import (
    ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
    ATMOSPHERE_LAYER_HEIGHT_M,
    ATMOSPHERE_LAYER_MIDPOINT_M,
    BOUNDARY_LAYER_EMISSIVITY,
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    BOUNDARY_LAYER_HEIGHT_M,
    R_EARTH_METERS,
    SHORTWAVE_ABSORPTANCE_ATMOSPHERE,
    STANDARD_LAPSE_RATE_K_PER_M,
    STEFAN_BOLTZMANN_W_M2_K4,
)
from climate_sim.physics.atmosphere.hadley import compute_itcz_latitude
from climate_sim.physics.atmosphere.pressure import compute_pressure
from climate_sim.physics.humidity import (
    compute_humidity_q,
    compute_saturation_specific_humidity,
    specific_humidity_to_relative_humidity,
)

if TYPE_CHECKING:
    from climate_sim.physics.clouds import CloudPrecipOutput

# Water vapor scale height (~2 km observed globally). Used to compute effective
# emission height since water vapor is exponentially distributed.
WATER_VAPOR_SCALE_HEIGHT_M = 2000.0

# ---------------------------------------------------------------------------
# Two-band water vapor greenhouse (Geen/Isca gray radiation scheme)
# Splits LW spectrum into absorption bands (logarithmic in q) and
# atmospheric window (linear + quadratic continuum in q).
# ---------------------------------------------------------------------------
_R_WINDOW = 0.3732   # Window fraction of LW spectrum

# Non-window band: line absorption saturates logarithmically
_A_NW = 0.10         # Well-mixed gases (CO2, O3)
_B_NW = 23.8         # H2O line absorption
_C_NW = 254.0        # Offset to prevent log(0)

# Window band: continuum absorption (linear foreign + quadratic self)
_A_WIN = 0.215       # Well-mixed gases in window
_B_WIN = 147.11      # H2O-N2 foreign continuum
_C_WIN = 1.0814e4    # H2O-H2O self-continuum

# Layer pressure thicknesses (fraction of surface pressure)
_DP_P0_BL = 0.15     # BL: 1000-850 hPa
_DP_P0_ATM = 0.65    # Free atm: 850-200 hPa

# Free atmosphere humidity as fraction of BL humidity
# H2O scale height ~2 km, integrated over 1-8.5 km
_Q_ATM_FRACTION = 0.26

_EPS_DRY_ATM = (
    (1 - _R_WINDOW) * (1 - np.exp(-_DP_P0_ATM * _A_NW))
    + _R_WINDOW * (1 - np.exp(-_DP_P0_ATM * _A_WIN))
)

# Import cloud height constants from clouds module
from climate_sim.physics.clouds import (
    CONVECTIVE_CLOUD_BASE_HEIGHT_M,
    CONVECTIVE_CLOUD_TOP_HEIGHT_M,
    HIGH_CLOUD_BASE_HEIGHT_M,
    HIGH_CLOUD_TOP_HEIGHT_M,
    MARINE_SC_CLOUD_BASE_HEIGHT_M,
    MARINE_SC_CLOUD_TOP_HEIGHT_M,
    STRATIFORM_CLOUD_BASE_HEIGHT_M,
    STRATIFORM_CLOUD_TOP_HEIGHT_M,
)


@dataclass(frozen=True)
class RadiationConfig:
    """Container for radiative transfer parameters.

    Emissivity is humidity-dependent: water vapor is the primary greenhouse gas.
    Dry air has low emissivity (transparent), moist air has high emissivity (strong greenhouse).
    """

    stefan_boltzmann: float = STEFAN_BOLTZMANN_W_M2_K4
    emissivity_surface: float = 1.0
    emissivity_atmosphere_dry: float = 0.45  # Includes CO2, O3, well-mixed gases
    emissivity_atmosphere_moist: float = 0.85
    emissivity_bl_dry: float = 0.55  # Higher than free atm due to water vapor density
    emissivity_bl_moist: float = 0.80
    emissivity_atmosphere: float = 0.60  # Default when humidity not available
    include_atmosphere: bool = True
    atmosphere_heat_capacity: float = ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K
    temperature_floor: float = 10.0  # K
    boundary_layer_heat_capacity: float = BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K
    boundary_layer_emissivity: float = BOUNDARY_LAYER_EMISSIVITY
    shortwave_absorptance_atmosphere: float = SHORTWAVE_ABSORPTANCE_ATMOSPHERE
    cloud_top_delta_z_m: float = ATMOSPHERE_LAYER_HEIGHT_M / 2.0


def _with_floor(values: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(values, floor)


def compute_humidity_emissivity(
    humidity_q: np.ndarray,
    eps_dry: float,
    eps_moist: float,
    q_reference: float = 0.018,  # kg/kg, typical tropical BL humidity
) -> np.ndarray:
    """Compute emissivity based on specific humidity.

    Emissivity depends on absolute humidity (q), not relative humidity,
    because it's the number of water vapor molecules in the optical path
    that determines absorption/emission.

    Linear interpolation from eps_dry (at q=0) to eps_moist (at q=q_reference).
    """
    humidity_fraction = np.clip(humidity_q / q_reference, 0.0, 1.0)
    return eps_dry + (eps_moist - eps_dry) * humidity_fraction


def compute_two_band_emissivity(
    humidity_q: np.ndarray,
    dp_p0: float,
) -> np.ndarray:
    """Compute layer emissivity using two-band gray radiation (Geen/Isca).

    Non-window band (~63%): line absorption saturates logarithmically.
    Window band (~37%): continuum absorption, linear + quadratic in q.
    """
    q = np.maximum(humidity_q, 0.0)
    tau_nw = dp_p0 * (_A_NW + _B_NW * np.log(_C_NW * q + 1.0))
    tau_win = dp_p0 * (_A_WIN + _B_WIN * q + _C_WIN * q**2)
    eps_nw = 1.0 - np.exp(-tau_nw)
    eps_win = 1.0 - np.exp(-tau_win)
    return (1.0 - _R_WINDOW) * eps_nw + _R_WINDOW * eps_win


def compute_emission_asymmetry_factor(emissivity: np.ndarray | float) -> np.ndarray | float:
    """Compute the emission asymmetry factor for a grey slab with lapse rate.

    For a layer with linear temperature profile, upward emission comes from
    colder upper levels, downward from warmer lower levels. The asymmetry
    factor f satisfies:
        T_eff_up = T_mid - (dT/2) * f
        T_eff_down = T_mid + (dT/2) * f

    Derived from grey slab radiative transfer: f = 1 - 2g/eps where
    g = (1/tau)[1 - (tau + 1)exp(-tau)].

    Limits: optically thin (eps -> 0): f -> 0; optically thick (eps -> 1): f -> 1.
    """
    eps = np.asarray(emissivity)
    eps_safe = np.clip(eps, 1e-6, 1.0 - 1e-6)

    tau_H = -np.log(1.0 - eps_safe)
    exp_neg_tau = 1.0 - eps_safe
    g = (1.0 - (tau_H + 1.0) * exp_neg_tau) / tau_H
    f = 1.0 - 2.0 * g / eps_safe

    return np.clip(f, 0.0, 1.0)


def compute_effective_emission_height(
    emissivity: np.ndarray | float,
    layer_height_m: float,
    emissivity_dry: float = 0.0,
) -> np.ndarray | float:
    """Compute effective emission height for asymmetry calculation.

    The atmosphere has two IR absorber types with different vertical profiles:
    1. Well-mixed gases (CO2, O3): uniform, H_eff = 0.16 * tau * layer_height
    2. Water vapor: exponential (scale height ~2 km), H_eff = 0.5 * tau * H_wv

    If emissivity_dry <= 0, uses geometric height (appropriate for boundary layer).
    Otherwise computes emissivity-weighted average of both components.
    """
    eps_total = np.asarray(emissivity)

    if emissivity_dry <= 0:
        return layer_height_m

    # Decompose into well-mixed (CO2) and water vapor components
    eps_dry = np.minimum(emissivity_dry, eps_total - 1e-6)
    eps_H2O = np.maximum(eps_total - eps_dry, 1e-6)

    # Well-mixed component: uniform absorber formula
    eps_dry_safe = np.clip(eps_dry, 1e-6, 1.0 - 1e-6)
    tau_dry = -np.log(1.0 - eps_dry_safe)
    H_eff_dry = layer_height_m * 0.16 * tau_dry

    # Water vapor component: exponential absorber formula
    eps_H2O_safe = np.clip(eps_H2O, 1e-6, 1.0 - 1e-6)
    tau_H2O = -np.log(1.0 - eps_H2O_safe)
    H_eff_H2O = WATER_VAPOR_SCALE_HEIGHT_M * 0.5 * tau_H2O

    # Emissivity-weighted average
    eps_total_safe = np.maximum(eps_total, 1e-6)
    return (eps_dry * H_eff_dry + eps_H2O * H_eff_H2O) / eps_total_safe


def compute_effective_emission_temperatures(
    temperature_mid: np.ndarray,
    emissivity: np.ndarray | float,
    layer_height_m: float,
    lapse_rate_K_per_m: float = STANDARD_LAPSE_RATE_K_PER_M,
    emissivity_dry: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute effective emission temperatures for upward and downward radiation.

    Due to lapse rate within the layer, upward emission comes from colder upper
    levels and downward emission from warmer lower levels. Set emissivity_dry=0
    for boundary layer (uses geometric height), or ~0.55 for free atmosphere
    (combines CO2 + H2O absorbers).
    """
    H_eff = compute_effective_emission_height(emissivity, layer_height_m, emissivity_dry)
    delta_T = lapse_rate_K_per_m * H_eff
    f = compute_emission_asymmetry_factor(emissivity)

    T_eff_up = temperature_mid - 0.5 * delta_T * f
    T_eff_down = temperature_mid + 0.5 * delta_T * f

    return T_eff_up, T_eff_down


def _compute_pressure_anomaly(
    surface_temp: np.ndarray,
    itcz_rad: np.ndarray,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
) -> np.ndarray:
    pressure = compute_pressure(surface_temp, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)

    nlat, nlon = surface_temp.shape
    lat_spacing = 180.0 / nlat
    lat_centers = -90.0 + (np.arange(nlat) + 0.5) * lat_spacing
    cos_lat = np.clip(np.cos(np.deg2rad(lat_centers)), 1.0e-6, None)
    weights = np.broadcast_to(cos_lat[:, None], (nlat, nlon))

    mean_pressure = area_weighted_mean(pressure, weights)
    dp = pressure - mean_pressure

    dp_norm = np.clip(dp / 1000.0, -1.0, 1.0)
    return dp_norm


def compute_cloud_coverage(
    rh: np.ndarray,
    dp_norm: np.ndarray,
    lat_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cloud coverage from relative humidity and pressure pattern.

    Based on observations: ITCZ 50-60%, subtropics 30-40%, polar 60-70%.
    """
    rh_min = 0.35
    rh_max = 0.85
    rh_smooth = np.clip((rh - rh_min) / (rh_max - rh_min), 0, 1)
    rh_smooth = 3 * rh_smooth**2 - 2 * rh_smooth**3

    c_conv = 0.40 + 0.20 * rh_smooth
    c_strat = 0.20 + 0.20 * rh_smooth

    weight_strat = 0.5 * (1.0 + dp_norm)
    coverage = c_conv + (c_strat - c_conv) * weight_strat

    if lat_deg is not None:
        lat_abs = np.abs(lat_deg)
        polar_factor = np.clip((lat_abs - 60.0) / 15.0, 0, 1)
        polar_factor = 3 * polar_factor**2 - 2 * polar_factor**3
        polar_coverage = 0.50 + 0.20 * rh_smooth
        coverage = coverage * (1 - polar_factor) + polar_coverage * polar_factor

    return np.array(coverage)


def compute_cloud_top_height(
    dp_norm: np.ndarray,
) -> np.ndarray:
    h_strat = 1500.0  # Stratocumulus top height (m)
    h_conv = 12000.0  # Deep convective top height (m)

    # Low pressure (dp_norm = -1) → convective → high tops
    # High pressure (dp_norm = +1) → stratocumulus → low tops
    weight_conv = 0.5 * (1.0 - dp_norm)
    h_cloud = h_strat + (h_conv - h_strat) * weight_conv

    return np.array(h_cloud)


def compute_cloud_albedo(
    rh: np.ndarray,
    dp_norm: np.ndarray,
    lat_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cloud albedo (reflectivity when present) from RH and pressure.

    ITCZ clouds are mostly thin cirrus/anvils (low albedo), subtropical regions
    have localized bright stratocumulus patches, polar clouds are thin stratus.
    """
    rh_min = 0.35
    rh_max = 0.85
    rh_smooth = np.clip((rh - rh_min) / (rh_max - rh_min), 0, 1)
    rh_smooth = 3 * rh_smooth**2 - 2 * rh_smooth**3

    alpha_conv = 0.20 + 0.10 * rh_smooth
    alpha_strat = 0.30 + 0.15 * rh_smooth

    weight_strat = 0.5 * (1.0 + dp_norm)
    alpha_cloud = alpha_conv + (alpha_strat - alpha_conv) * weight_strat

    if lat_deg is not None:
        lat_abs = np.abs(lat_deg)
        polar_factor = np.clip((lat_abs - 60.0) / 15.0, 0, 1)
        polar_factor = 3 * polar_factor**2 - 2 * polar_factor**3
        alpha_polar = 0.20 + 0.10 * rh_smooth
        alpha_cloud = alpha_cloud * (1 - polar_factor) + alpha_polar * polar_factor

    return alpha_cloud


def radiative_balance_rhs(
    temperature_K: np.ndarray,
    insolation_W_m2: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    albedo_field: np.ndarray,
    config: RadiationConfig,
    land_mask: np.ndarray | None = None,
    humidity_q: np.ndarray | None = None,
    log_diagnostics: bool = False,
    cell_area_m2: np.ndarray | None = None,
    itcz_rad: np.ndarray | None = None,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
    cloud_output: "CloudPrecipOutput | None" = None,
) -> np.ndarray:
    """Column energy-balance tendency for the configured radiative model."""

    floor = config.temperature_floor

    if not config.include_atmosphere:
        surface = _with_floor(temperature_K[0], floor)
        emitted = config.emissivity_surface * config.stefan_boltzmann * np.power(surface, 4)
        absorbed = insolation_W_m2 * (1.0 - albedo_field)
        return ((absorbed - emitted) / heat_capacity_field)[np.newaxis, :, :]

    surface = _with_floor(temperature_K[0], floor)

    nlayers = temperature_K.shape[0]
    atmosphere = _with_floor(temperature_K[2], floor)  # Free atmosphere (layer 2)

    # Compute cloud properties - use unified CloudPrecipOutput if available
    if cloud_output is not None:
        # Use unified cloud-precipitation module output
        # Separate convective, stratiform, marine Sc, and high clouds for accurate radiation
        conv_frac = cloud_output.convective_frac
        conv_albedo = cloud_output.convective_albedo
        conv_top_K = cloud_output.convective_top_K
        strat_frac = cloud_output.stratiform_frac
        strat_albedo = cloud_output.stratiform_albedo
        strat_top_K = cloud_output.stratiform_top_K
        marine_sc_frac = cloud_output.marine_sc_frac
        marine_sc_albedo = cloud_output.marine_sc_albedo
        marine_sc_top_K = cloud_output.marine_sc_top_K
        high_frac = cloud_output.high_cloud_frac
        high_albedo = cloud_output.high_cloud_albedo
        high_top_K = cloud_output.high_cloud_top_K

        # Low clouds (conv, strat, marine Sc) are mutually exclusive regimes:
        # - Convective: rising motion + unstable (low LTS)
        # - Stratiform: rising motion + stable (high LTS)
        # - Marine Sc: subsidence + stable + ocean
        # So we sum them but cap at 1.0
        low_cloud_frac = np.minimum(conv_frac + strat_frac + marine_sc_frac, 1.0)

        # Effective low cloud albedo (weighted average by relative fractions)
        low_cloud_frac_safe = np.maximum(low_cloud_frac, 1e-10)
        low_cloud_albedo = np.where(
            low_cloud_frac > 0.01,
            (conv_frac * conv_albedo + strat_frac * strat_albedo + marine_sc_frac * marine_sc_albedo) / low_cloud_frac_safe,
            0.30  # default if no low clouds
        )

        # High clouds at different level - random overlap with low clouds
        # High clouds only reflect SW that wasn't already reflected by low clouds
        clear_of_low = 1.0 - low_cloud_frac
        high_effective_frac = high_frac * clear_of_low

        # Total cloud fraction for diagnostics (with overlap correction)
        total_cloud_frac = low_cloud_frac + high_effective_frac
        cloud_coverage = total_cloud_frac

        # Weighted average albedo for diagnostics
        total_frac_safe = np.maximum(total_cloud_frac, 1e-10)
        cloud_albedo = np.where(
            total_cloud_frac > 0.01,
            (low_cloud_frac * low_cloud_albedo + high_effective_frac * high_albedo) / total_frac_safe,
            0.30
        )

        # Atmospheric albedo with proper overlap:
        # - Clear sky baseline (Rayleigh + aerosols): 0.05
        # - Low clouds reflect: low_cloud_frac × low_cloud_albedo
        # - High clouds reflect remaining SW: high_effective_frac × high_albedo
        atm_albedo_field = 0.05 + low_cloud_frac * low_cloud_albedo + high_effective_frac * high_albedo

    else:
        # Fallback: compute simple cloud cover from humidity/pressure patterns
        # This path is used during initialization or when full cloud physics isn't available
        if humidity_q is not None and itcz_rad is not None and lat2d is not None and lon2d is not None:
            rh = specific_humidity_to_relative_humidity(
                humidity_q, surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d
            )
            dp_norm = _compute_pressure_anomaly(
                surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d
            )
            cloud_coverage = compute_cloud_coverage(rh, dp_norm, lat2d)
            cloud_albedo = compute_cloud_albedo(rh, dp_norm, lat2d)
        elif humidity_q is not None:
            # Minimal fallback: use humidity-based cloud cover
            rh = humidity_q / np.maximum(
                compute_saturation_specific_humidity(surface), 1e-10
            )
            cloud_coverage = np.clip(rh - 0.5, 0, 0.5) * 2.0  # Linear ramp from RH=0.5 to 1.0
            cloud_albedo = np.full_like(cloud_coverage, 0.35)
        else:
            # No humidity info: assume 50% cloud cover globally
            cloud_coverage = np.full_like(surface, 0.50)
            cloud_albedo = np.full_like(surface, 0.35)

        # Treat all clouds as low stratiform for the fallback
        low_cloud_frac = cloud_coverage
        low_cloud_albedo = cloud_albedo
        high_frac = np.zeros_like(surface)
        high_albedo = np.zeros_like(surface)
        high_effective_frac = np.zeros_like(surface)
        conv_frac = np.zeros_like(surface)
        strat_frac = cloud_coverage
        marine_sc_frac = np.zeros_like(surface)

        atm_albedo_field = 0.05 + cloud_coverage * cloud_albedo

    # Log cloud diagnostics
    if log_diagnostics and cell_area_m2 is not None:
        total_area = np.sum(cell_area_m2)
        mean_coverage = area_weighted_mean(cloud_coverage, cell_area_m2 / total_area)
        mean_albedo_per_cloud = area_weighted_mean(cloud_albedo, cell_area_m2 / total_area)
        mean_atm_albedo = area_weighted_mean(atm_albedo_field, cell_area_m2 / total_area)
        print(f"[CLOUD] Global mean coverage: {mean_coverage:.3f}")
        print(f"[CLOUD] Global mean albedo per cloud: {mean_albedo_per_cloud:.3f}")
        print(f"[CLOUD] Global mean atmospheric albedo: {mean_atm_albedo:.3f}")

    sigma = config.stefan_boltzmann
    eps_sfc = config.emissivity_surface

    # Clear-sky emissivity: two-band (logarithmic line + quadratic continuum)
    # Free atmosphere uses q_atm ≈ 26% of BL humidity (H2O scale height ~2 km)
    if humidity_q is not None:
        eps_clear = compute_two_band_emissivity(
            humidity_q * _Q_ATM_FRACTION, _DP_P0_ATM
        )
    else:
        eps_clear = config.emissivity_atmosphere

    # Cloud contribution: sqrt saturation since clouds saturate LW quickly
    eps_cloud = (1.0 - eps_clear) * np.sqrt(cloud_coverage)
    eps_atm = eps_clear + eps_cloud

    emitted_surface = eps_sfc * sigma * np.power(surface, 4)

    # Effective emission temperatures with up/down asymmetry due to lapse rate.
    # Clouds emit separately, so atmosphere uses eps_clear only.
    T_atm_up, T_atm_down = compute_effective_emission_temperatures(
        atmosphere, eps_clear, ATMOSPHERE_LAYER_HEIGHT_M, emissivity_dry=_EPS_DRY_ATM
    )
    atm_own_emission_up = eps_clear * sigma * np.power(T_atm_up, 4)
    atm_own_emission_down = eps_clear * sigma * np.power(T_atm_down, 4)

    # Clear fraction: use same overlap logic as SW for consistency
    # Low clouds are mutually exclusive regimes, sum capped at 1.0
    low_cloud_frac = np.minimum(conv_frac + strat_frac + marine_sc_frac, 1.0)
    clear_of_low = 1.0 - low_cloud_frac
    high_effective_frac = high_frac * clear_of_low
    clear_frac = 1.0 - low_cloud_frac - high_effective_frac

    emitted_clear_up = eps_clear * sigma * np.power(T_atm_up, 4)
    emitted_clear_down = eps_clear * sigma * np.power(T_atm_down, 4)

    # ============================================================
    # LONGWAVE CLOUD EMISSIVITIES (from τ_LW via Beer-Lambert)
    # ============================================================
    # Use τ-derived emissivities from cloud_output if available
    if cloud_output is not None and cloud_output.eps_convective is not None:
        eps_conv_cloud = cloud_output.eps_convective
        eps_strat_cloud = cloud_output.eps_stratiform
        eps_marine_sc_cloud = cloud_output.eps_marine_sc
        eps_high_cloud = cloud_output.eps_high
    else:
        # Fallback to τ-derived values (ε = 1 - exp(-τ_LW))
        eps_conv_cloud = 1.0 - np.exp(-40.0)   # τ_LW = 40 → ε ≈ 1.0
        eps_strat_cloud = 1.0 - np.exp(-7.0)   # τ_LW = 7 → ε ≈ 0.999
        eps_marine_sc_cloud = 1.0 - np.exp(-10.0)  # τ_LW = 10 → ε ≈ 1.0
        eps_high_cloud = 1.0 - np.exp(-1.5)    # τ_LW = 1.5 → ε ≈ 0.78

    # Cloud temperatures: TOPs lapsed from atmosphere midpoint, BASEs from boundary layer.
    # T_atm represents temperature at ATMOSPHERE_LAYER_MIDPOINT_M (~5 km).
    # Cloud tops extend into free atmosphere, so lapse from T_atm.
    # Cloud bases are in/near the boundary layer, so lapse UP from T_BL (in 3-layer model).
    z_atm_mid = ATMOSPHERE_LAYER_MIDPOINT_M
    z_bl_mid = BOUNDARY_LAYER_HEIGHT_M / 2.0  # ~500m, midpoint of BL layer

    # Cloud TOP temperatures: lapse from atmosphere midpoint
    # Convective tops at 10 km (above midpoint = colder)
    current_conv_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (CONVECTIVE_CLOUD_TOP_HEIGHT_M - z_atm_mid),
        floor,
    )
    # Stratiform tops at 1.5 km (below midpoint, but still above BL)
    current_strat_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (STRATIFORM_CLOUD_TOP_HEIGHT_M - z_atm_mid),
        floor,
    )
    # Marine Sc tops at 1 km (at BL top)
    current_marine_sc_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (MARINE_SC_CLOUD_TOP_HEIGHT_M - z_atm_mid),
        floor,
    )
    # High cloud tops at 12 km (well above midpoint = cold)
    current_high_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (HIGH_CLOUD_TOP_HEIGHT_M - z_atm_mid),
        floor,
    )
    # High cloud bases at 8 km (above midpoint = cold)
    current_high_base_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (HIGH_CLOUD_BASE_HEIGHT_M - z_atm_mid),
        floor,
    )

    # Cloud BASE temperatures for low clouds: lapse UP from boundary layer.
    # BL temp represents midpoint (~500m). Cloud bases are at various heights within BL.
    # T_base = T_bl - Γ × (z_base - z_bl_mid)  [negative dz means warmer]
    # Will be recomputed in 3-layer section with actual boundary temp
    # Use atmosphere-based estimate here as placeholder
    current_conv_base_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (CONVECTIVE_CLOUD_BASE_HEIGHT_M - z_atm_mid),
        floor,
    )
    current_strat_base_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (STRATIFORM_CLOUD_BASE_HEIGHT_M - z_atm_mid),
        floor,
    )
    current_marine_sc_base_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (MARINE_SC_CLOUD_BASE_HEIGHT_M - z_atm_mid),
        floor,
    )

    # Upward cloud emissions (from tops) - computed here since tops use atm temp
    emitted_conv_up = eps_conv_cloud * sigma * np.power(current_conv_top_K, 4)
    emitted_strat_up = eps_strat_cloud * sigma * np.power(current_strat_top_K, 4)
    emitted_marine_sc_up = eps_marine_sc_cloud * sigma * np.power(current_marine_sc_top_K, 4)
    emitted_high_up = eps_high_cloud * sigma * np.power(current_high_top_K, 4)
    emitted_high_down = eps_high_cloud * sigma * np.power(current_high_base_K, 4)

    # Downward cloud emissions (from bases) - will be recomputed in 3-layer section
    emitted_conv_down = eps_conv_cloud * sigma * np.power(current_conv_base_K, 4)
    emitted_strat_down = eps_strat_cloud * sigma * np.power(current_strat_base_K, 4)
    emitted_marine_sc_down = eps_marine_sc_cloud * sigma * np.power(current_marine_sc_base_K, 4)

    # Area-weighted OLR (upward emission to space)
    # High clouds emit MUCH less than clear sky due to cold tops
    #
    # Low clouds (stratiform, marine Sc) have nearly the full atmosphere above them
    # (tops at ~1-1.5 km). Their upward emission is absorbed by atmosphere above
    # and re-emitted at colder temperature.
    #
    # Convective and high clouds have very little atmosphere above (tops at 8-12 km)
    # so their emission goes directly to space.

    # Atmosphere absorbs and re-emits cloud radiation
    # transmittance = 1 - eps_clear (what gets through without absorption)
    transmittance_atm = 1.0 - eps_clear
    # For OLR: atmosphere above emits upward at T_atm (using raw T, not effective T)
    atm_emission_up = eps_clear * sigma * np.power(atmosphere, 4)
    # For downward: atmosphere below high clouds re-emits at T_atm_down (warmer)
    atm_emission_down = atm_own_emission_down

    # Low cloud OLR: transmitted cloud emission + atmosphere-above emission
    strat_olr = transmittance_atm * emitted_strat_up + atm_emission_up
    marine_sc_olr = transmittance_atm * emitted_marine_sc_up + atm_emission_up

    emitted_toa = (
        clear_frac * emitted_clear_up
        + conv_frac * emitted_conv_up           # High tops - direct to space
        + strat_frac * strat_olr                # Low tops - atm above absorbs/re-emits
        + marine_sc_frac * marine_sc_olr        # Low tops - atm above absorbs/re-emits
        + high_effective_frac * emitted_high_up # High tops - direct to space (with overlap)
    )

    # Downward emission to BL and surface
    # High cloud downward emission (from base at 8 km) is absorbed by atmosphere
    # below and re-emitted at warmer temperature. Use same transmittance as upward.
    high_down_effective = transmittance_atm * emitted_high_down + atm_emission_down

    downward_lw_weighted = (
        clear_frac * emitted_clear_down
        + conv_frac * emitted_conv_down
        + strat_frac * emitted_strat_down
        + marine_sc_frac * emitted_marine_sc_down
        + high_effective_frac * high_down_effective  # absorbed/re-emitted by atm below
    )

    # ============================================================
    # SHORTWAVE RADIATION - TOP DOWN
    # ============================================================
    # SW travels: space -> high clouds -> low clouds -> surface
    # At each layer, some is reflected (albedo) and some is absorbed.
    #
    # Clear-sky absorption split between atmosphere layers:
    # - Free atmosphere: 12% (O3, some H2O)
    # - Boundary layer: 8% (H2O-rich)
    #
    # Cloud SW absorptance derived from τ_SW and single-scattering albedo ω₀:
    # absorptance ≈ (1 - ω₀) × τ / (1 + τ)
    # Water clouds (ω₀≈0.99) absorb very little despite high τ
    # Ice clouds (ω₀≈0.97) absorb slightly more

    alpha_atm = atm_albedo_field

    # Import clear-sky absorptance constants
    from climate_sim.physics.clouds import (
        CLEAR_SKY_SW_ABSORPTANCE_ATM,
        CLEAR_SKY_SW_ABSORPTANCE_BL,
    )

    if cloud_output is not None:
        # Use τ-derived SW absorptances from cloud_output if available
        if cloud_output.abs_sw_high is not None:
            abs_high = cloud_output.abs_sw_high
            abs_conv = cloud_output.abs_sw_convective
            abs_strat = cloud_output.abs_sw_stratiform
            abs_marine_sc = cloud_output.abs_sw_marine_sc
        else:
            # Fallback: compute from τ inline
            # absorptance = (1 - ω₀) × τ / (1 + τ)
            abs_high = 0.03 * 0.8 / 1.8       # τ=0.8, ω₀=0.97 → ~0.013
            abs_conv = 0.01 * 40.0 / 41.0     # τ=40, ω₀=0.99 → ~0.0098
            abs_strat = 0.01 * 7.0 / 8.0      # τ=7, ω₀=0.99 → ~0.0087
            abs_marine_sc = 0.01 * 10.0 / 11.0  # τ=10, ω₀=0.99 → ~0.0091

        # High clouds absorb from full incoming
        high_absorbed_frac = high_frac * abs_high

        # SW reaching low cloud level = (1 - high_albedo - abs_high) per high cloud fraction
        sw_reaching_low = 1.0 - high_frac * (high_albedo + abs_high)

        # Low clouds absorb from what reaches them
        low_absorbed_frac = sw_reaching_low * (
            conv_frac * abs_conv
            + strat_frac * abs_strat
            + marine_sc_frac * abs_marine_sc
        )

        cloud_sw_absorptance = high_absorbed_frac + low_absorbed_frac
    else:
        # Fallback: use simple cloud coverage with τ-derived absorptance
        cloud_sw_absorptance = 0.01 * cloud_coverage

    # Total atmospheric SW absorptance (clear-sky + clouds)
    clear_sky_sw_abs_total = CLEAR_SKY_SW_ABSORPTANCE_ATM + CLEAR_SKY_SW_ABSORPTANCE_BL
    beta_atm = clear_sky_sw_abs_total + cloud_sw_absorptance
    absorbed_shortwave_atm = beta_atm * insolation_W_m2
    sw_down_surface = (1.0 - alpha_atm - beta_atm) * insolation_W_m2
    absorbed_shortwave_sfc = sw_down_surface * (1.0 - albedo_field)

    # Longwave budget
    downward_longwave = downward_lw_weighted
    absorbed_from_surface = eps_atm * emitted_surface

    boundary = _with_floor(temperature_K[1], floor)

    # Recompute low cloud BASE temperatures using boundary layer temp.
    # BL temp represents midpoint of BL (~500m = z_bl_mid).
    # Cloud bases within BL: T_base = T_bl - Γ × (z_base - z_bl_mid)
    # Bases below BL midpoint are warmer, above are cooler.
    current_conv_base_K = _with_floor(
        boundary - STANDARD_LAPSE_RATE_K_PER_M * (CONVECTIVE_CLOUD_BASE_HEIGHT_M - z_bl_mid),
        floor,
    )
    current_strat_base_K = _with_floor(
        boundary - STANDARD_LAPSE_RATE_K_PER_M * (STRATIFORM_CLOUD_BASE_HEIGHT_M - z_bl_mid),
        floor,
    )
    current_marine_sc_base_K = _with_floor(
        boundary - STANDARD_LAPSE_RATE_K_PER_M * (MARINE_SC_CLOUD_BASE_HEIGHT_M - z_bl_mid),
        floor,
    )

    # Recompute downward cloud emissions with corrected base temps
    emitted_conv_down = eps_conv_cloud * sigma * np.power(current_conv_base_K, 4)
    emitted_strat_down = eps_strat_cloud * sigma * np.power(current_strat_base_K, 4)
    emitted_marine_sc_down = eps_marine_sc_cloud * sigma * np.power(current_marine_sc_base_K, 4)

    # Recompute downward_lw_weighted with corrected emissions
    high_down_effective = transmittance_atm * emitted_high_down + atm_emission_down
    downward_lw_weighted = (
        clear_frac * emitted_clear_down
        + conv_frac * emitted_conv_down
        + strat_frac * emitted_strat_down
        + marine_sc_frac * emitted_marine_sc_down
        + high_effective_frac * high_down_effective
    )

    # BL emissivity: two-band (same physics as free atm, using BL humidity directly)
    if humidity_q is not None:
        eps_bl = compute_two_band_emissivity(humidity_q, _DP_P0_BL)
    else:
        eps_bl = BOUNDARY_LAYER_EMISSIVITY

    # BL emission with up/down asymmetry (well-mixed, so emissivity_dry=0)
    T_bl_up, T_bl_down = compute_effective_emission_temperatures(
        boundary, eps_bl, BOUNDARY_LAYER_HEIGHT_M, emissivity_dry=0.0
    )
    emitted_bl_up = eps_bl * sigma * np.power(T_bl_up, 4)
    emitted_bl_down = eps_bl * sigma * np.power(T_bl_down, 4)

    # Transmittance = 1 - emissivity (for grey slab)
    trans_bl = 1.0 - eps_bl
    trans_atm = 1.0 - eps_atm

    # ============================================================
    # SW ABSORPTION PARTITIONING BETWEEN ATMOSPHERE AND BL
    # ============================================================
    # Clear-sky absorption is split based on where absorbers are:
    # - Free atmosphere: O3 (stratosphere), some H2O → 12%
    # - Boundary layer: H2O-rich → 8%
    #
    # Cloud absorption goes to:
    # - High clouds: absorbed in free atmosphere
    # - Low clouds (conv, strat, marine_sc): absorbed in/near BL
    #
    # Compute clear-sky SW reaching each layer
    sw_after_albedo = (1.0 - alpha_atm) * insolation_W_m2

    # Clear-sky absorption
    clear_sw_abs_atm = CLEAR_SKY_SW_ABSORPTANCE_ATM * insolation_W_m2
    clear_sw_abs_bl = CLEAR_SKY_SW_ABSORPTANCE_BL * insolation_W_m2

    # Cloud absorption partitioning
    # High cloud absorption goes to free atmosphere
    high_cloud_sw_abs = high_frac * abs_high * insolation_W_m2 if cloud_output is not None else 0.0
    # Low cloud absorption goes to boundary layer
    if cloud_output is not None:
        sw_reaching_low = 1.0 - high_frac * (high_albedo + abs_high)
        low_cloud_sw_abs = sw_reaching_low * (
            conv_frac * abs_conv
            + strat_frac * abs_strat
            + marine_sc_frac * abs_marine_sc
        ) * insolation_W_m2
    else:
        low_cloud_sw_abs = cloud_sw_absorptance * insolation_W_m2

    absorbed_shortwave_atm_upper = clear_sw_abs_atm + high_cloud_sw_abs
    absorbed_shortwave_bl = clear_sw_abs_bl + low_cloud_sw_abs

    downward_longwave_to_surface = emitted_bl_down + trans_bl * downward_lw_weighted
    surface_tendency = (
        absorbed_shortwave_sfc + downward_longwave_to_surface - emitted_surface
    ) / heat_capacity_field

    absorbed_from_surface_bl = eps_bl * emitted_surface
    absorbed_from_atm_bl = eps_bl * downward_lw_weighted
    boundary_tendency = (
        absorbed_shortwave_bl + absorbed_from_surface_bl + absorbed_from_atm_bl
        - emitted_bl_up - emitted_bl_down
    ) / config.boundary_layer_heat_capacity

    transmitted_surface_to_atm = trans_bl * emitted_surface
    upwelling_lw = transmitted_surface_to_atm + emitted_bl_up

    # === Atmosphere LW budget ===
    #
    # For atmosphere tendency:
    #   IN: LW absorbed from below (portion of upwelling_lw that's absorbed)
    #   OUT: emitted_toa (upward emissions) + downward_lw_weighted (downward emissions)
    #
    # The transmitted portion of upwelling_lw that escapes to space never
    # enters the atmosphere's energy budget - it just passes through.
    #
    # Absorption path depends on cloud type and height:
    #
    # CLEAR SKY: gas absorbs eps_clear
    #
    # LOW CLOUDS (strat, marine_sc, conv bases near BL):
    #   - Cloud absorbs eps_cloud at base (before gas)
    #   - Transmitted (1-eps_cloud) reaches gas, which absorbs eps_clear of that
    #   - Total absorbed: eps_cloud + (1-eps_cloud) * eps_clear
    #
    # HIGH CLOUDS (bases at 8km, within upper atmosphere):
    #   - Gas below absorbs eps_clear first
    #   - Transmitted (1-eps_clear) reaches high cloud, which absorbs eps_high of that
    #   - Total absorbed: eps_clear + (1-eps_clear) * eps_high

    # Clear sky absorption
    clear_absorbed = clear_frac * eps_clear * upwelling_lw

    # Low cloud absorption (cloud first, then gas above)
    # Note: conv clouds have bases near BL like low clouds
    conv_total_abs = eps_conv_cloud + (1.0 - eps_conv_cloud) * eps_clear
    strat_total_abs = eps_strat_cloud + (1.0 - eps_strat_cloud) * eps_clear
    marine_sc_total_abs = eps_marine_sc_cloud + (1.0 - eps_marine_sc_cloud) * eps_clear

    low_cloud_absorbed = (
        conv_frac * conv_total_abs
        + strat_frac * strat_total_abs
        + marine_sc_frac * marine_sc_total_abs
    ) * upwelling_lw

    # High cloud absorption (gas first, then cloud above)
    high_total_abs = eps_clear + (1.0 - eps_clear) * eps_high_cloud
    high_cloud_absorbed = high_frac * high_total_abs * upwelling_lw

    lw_absorbed_from_below = clear_absorbed + low_cloud_absorbed + high_cloud_absorbed

    atmosphere_tendency = (
        absorbed_shortwave_atm_upper
        + lw_absorbed_from_below
        - emitted_toa
        - downward_lw_weighted
    ) / config.atmosphere_heat_capacity

    if log_diagnostics:
        if cell_area_m2 is None:
            raise ValueError("cell_area_m2 must be provided when log_diagnostics=True")

        reflected_by_atm = alpha_atm * insolation_W_m2
        reflected_by_surface = albedo_field * sw_down_surface

        # Total outgoing longwave radiation (OLR) at TOA
        olr_total = emitted_toa + trans_atm * emitted_bl_up + trans_atm * trans_bl * emitted_surface

        # Decompose atmosphere LW absorption by source:
        # lw_absorbed_from_below absorbs from upwelling_lw = transmitted_surface_to_atm + emitted_bl_up
        # The same absorption coefficients apply to both components, so decompose proportionally.
        _upwelling_safe = np.where(upwelling_lw > 0, upwelling_lw, 1.0)
        _surface_fraction = np.where(upwelling_lw > 0, transmitted_surface_to_atm / _upwelling_safe, 0.5)
        absorbed_from_surface_atm = lw_absorbed_from_below * _surface_fraction
        absorbed_from_boundary_atm = lw_absorbed_from_below * (1.0 - _surface_fraction)

        absorbed_shortwave_atm_total = absorbed_shortwave_atm_upper + absorbed_shortwave_bl

        print("\n=== Radiation Diagnostics (W/m²) ===")
        print("\nLayer Temperatures (K):")
        print(f"  Surface:                         {area_weighted_mean(surface, cell_area_m2):7.2f}")
        print(f"  Boundary layer:                  {area_weighted_mean(boundary, cell_area_m2):7.2f}")
        print(f"  Atmosphere:                      {area_weighted_mean(atmosphere, cell_area_m2):7.2f}")
        print("\nShortwave Fluxes:")
        print(f"Incoming solar (TOA):              {area_weighted_mean(insolation_W_m2, cell_area_m2):7.2f}")
        print(f"Reflected by atmosphere:           {area_weighted_mean(reflected_by_atm, cell_area_m2):7.2f}")
        print(f"Absorbed by atmosphere (SW total): {area_weighted_mean(absorbed_shortwave_atm_total, cell_area_m2):7.2f}")
        print(f"  - absorbed by boundary layer:    {area_weighted_mean(absorbed_shortwave_bl, cell_area_m2):7.2f}")
        print(f"  - absorbed by free atmosphere:   {area_weighted_mean(absorbed_shortwave_atm_upper, cell_area_m2):7.2f}")
        print(f"SW reaching surface:               {area_weighted_mean(sw_down_surface, cell_area_m2):7.2f}")
        print(f"Reflected by surface:              {area_weighted_mean(reflected_by_surface, cell_area_m2):7.2f}")
        print(f"Absorbed by surface (SW):          {area_weighted_mean(absorbed_shortwave_sfc, cell_area_m2):7.2f}")
        print("\nLongwave Emissions:")
        print(f"Surface emission:                  {area_weighted_mean(emitted_surface, cell_area_m2):7.2f}")
        print(f"Boundary layer emission (total):   {area_weighted_mean(emitted_bl_up + emitted_bl_down, cell_area_m2):7.2f}")
        print(f"  - downward (to surface):         {area_weighted_mean(emitted_bl_down, cell_area_m2):7.2f}")
        print(f"  - upward (to atmosphere):        {area_weighted_mean(emitted_bl_up, cell_area_m2):7.2f}")
        print(f"Atmosphere emission (total):       {area_weighted_mean(atm_own_emission_up + atm_own_emission_down, cell_area_m2):7.2f}")
        print(f"  - downward (to BL):              {area_weighted_mean(atm_own_emission_down, cell_area_m2):7.2f}")
        print(f"  - upward (to space):             {area_weighted_mean(atm_own_emission_up, cell_area_m2):7.2f}")
        print(f"OLR (to space):                    {area_weighted_mean(olr_total, cell_area_m2):7.2f}")
        print("\nBoundary layer absorbs from:")
        print(f"  - solar (SW):                    {area_weighted_mean(absorbed_shortwave_bl, cell_area_m2):7.2f}")
        print(f"  - surface (LW):                  {area_weighted_mean(absorbed_from_surface_bl, cell_area_m2):7.2f}")
        print(f"  - atmosphere (LW):               {area_weighted_mean(absorbed_from_atm_bl, cell_area_m2):7.2f}")
        print("Atmosphere absorbs from:")
        print(f"  - surface emission (absorbed):   {area_weighted_mean(absorbed_from_surface_atm, cell_area_m2):7.2f}")
        print(f"  - boundary layer emission:       {area_weighted_mean(absorbed_from_boundary_atm, cell_area_m2):7.2f}")
        print("\nNet Radiation Balances:")
        print(f"Net surface radiation balance:     {area_weighted_mean(absorbed_shortwave_sfc + downward_longwave_to_surface - emitted_surface, cell_area_m2):7.2f}")
        print(f"Net boundary layer balance:        {area_weighted_mean(absorbed_shortwave_bl + absorbed_from_surface_bl + absorbed_from_atm_bl - emitted_bl_up - emitted_bl_down, cell_area_m2):7.2f}")
        print(f"Net atmosphere balance:            {area_weighted_mean(absorbed_shortwave_atm_upper + absorbed_from_surface_atm + absorbed_from_boundary_atm - atm_own_emission_up - atm_own_emission_down, cell_area_m2):7.2f}")
        print("\nGlobal Energy Balance:")
        print(f"Net TOA balance (SW_in - OLR):     {area_weighted_mean(insolation_W_m2 - reflected_by_atm - reflected_by_surface - olr_total, cell_area_m2):7.2f}")
        print("=" * 40)

    return np.stack([surface_tendency, boundary_tendency, atmosphere_tendency])


def radiative_balance_rhs_temperature_derivative(
    temperature_K: np.ndarray,
    *,
    heat_capacity_field: np.ndarray,
    config: RadiationConfig,
    land_mask: np.ndarray | None = None,
    humidity_q: np.ndarray | None = None,
    itcz_rad: np.ndarray | None = None,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
    cloud_output: "CloudPrecipOutput | None" = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the Jacobian of radiative tendency with respect to temperature.

    Returns diagonal terms (self-feedback) and cross-layer coupling terms.
    Cloud fractions and heights are frozen during Newton iterations; only
    cloud temperatures (via lapse rate from surface) contribute to derivatives.

    Returns
    -------
    For 1-layer: np.ndarray of shape (1, nlat, nlon) - just diagonal
    For 3-layer: tuple of (diag, cross) where:
      - diag: (nlayers, nlat, nlon) diagonal terms
      - cross: (nlayers, nlayers, nlat, nlon) off-diagonal coupling terms
    """

    floor = config.temperature_floor
    sigma = config.stefan_boltzmann

    if not config.include_atmosphere:
        surface = _with_floor(temperature_K[0], floor)
        coeff = -4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
        return (coeff / heat_capacity_field)[np.newaxis, :, :]

    surface = _with_floor(temperature_K[0], floor)
    nlayers = temperature_K.shape[0]
    atmosphere = _with_floor(temperature_K[2], floor)  # Free atmosphere (layer 2)

    # Use cloud_output if available, otherwise compute simple fallback
    if cloud_output is not None:
        conv_frac = cloud_output.convective_frac
        strat_frac = cloud_output.stratiform_frac
        marine_sc_frac = cloud_output.marine_sc_frac
        high_frac = cloud_output.high_cloud_frac
        cloud_coverage = cloud_output.total_frac
    else:
        # Fallback: compute simple cloud cover (frozen for linearization)
        if humidity_q is not None and itcz_rad is not None and lat2d is not None and lon2d is not None:
            rh = specific_humidity_to_relative_humidity(
                humidity_q, surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d
            )
            dp_norm = _compute_pressure_anomaly(
                surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d
            )
            cloud_coverage = compute_cloud_coverage(rh, dp_norm, lat2d)
        elif humidity_q is not None:
            rh = humidity_q / np.maximum(
                compute_saturation_specific_humidity(surface), 1e-10
            )
            cloud_coverage = np.clip(rh - 0.5, 0, 0.5) * 2.0
        else:
            cloud_coverage = np.full_like(surface, 0.50)

        # Treat all as stratiform for fallback
        conv_frac = np.zeros_like(surface)
        strat_frac = cloud_coverage
        marine_sc_frac = np.zeros_like(surface)
        high_frac = np.zeros_like(surface)

    # Clear-sky emissivity (must match RHS)
    if humidity_q is not None:
        eps_clear = compute_two_band_emissivity(
            humidity_q * _Q_ATM_FRACTION, _DP_P0_ATM
        )
    else:
        eps_clear = config.emissivity_atmosphere
    eps_cloud = (1.0 - eps_clear) * np.sqrt(cloud_coverage)
    eps_atm = eps_clear + eps_cloud

    surface_diag = (
        -4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
    ) / heat_capacity_field

    T_atm_up, T_atm_down = compute_effective_emission_temperatures(
        atmosphere, eps_atm, ATMOSPHERE_LAYER_HEIGHT_M, emissivity_dry=_EPS_DRY_ATM
    )

    # Clear fraction (must match RHS calculation)
    low_cloud_frac = np.minimum(conv_frac + strat_frac + marine_sc_frac, 1.0)
    clear_of_low = 1.0 - low_cloud_frac
    high_effective_frac = high_frac * clear_of_low
    clear_frac = 1.0 - low_cloud_frac - high_effective_frac

    # ============================================================
    # LONGWAVE CLOUD EMISSIVITIES (must match RHS - from τ_LW via Beer-Lambert)
    # ============================================================
    if cloud_output is not None and cloud_output.eps_convective is not None:
        eps_conv_cloud = cloud_output.eps_convective
        eps_strat_cloud = cloud_output.eps_stratiform
        eps_marine_sc_cloud = cloud_output.eps_marine_sc
        eps_high_cloud = cloud_output.eps_high
    else:
        # Fallback to τ-derived values (ε = 1 - exp(-τ_LW))
        eps_conv_cloud = 1.0 - np.exp(-40.0)   # τ_LW = 40 → ε ≈ 1.0
        eps_strat_cloud = 1.0 - np.exp(-7.0)   # τ_LW = 7 → ε ≈ 0.999
        eps_marine_sc_cloud = 1.0 - np.exp(-10.0)  # τ_LW = 10 → ε ≈ 1.0
        eps_high_cloud = 1.0 - np.exp(-1.5)    # τ_LW = 1.5 → ε ≈ 0.78

    # Cloud TOP temperatures lapsed from atmosphere layer midpoint (must match RHS)
    z_atm_mid = ATMOSPHERE_LAYER_MIDPOINT_M
    z_bl_mid = BOUNDARY_LAYER_HEIGHT_M / 2.0  # BL midpoint at ~500m

    current_conv_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (CONVECTIVE_CLOUD_TOP_HEIGHT_M - z_atm_mid),
        floor,
    )
    current_strat_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (STRATIFORM_CLOUD_TOP_HEIGHT_M - z_atm_mid),
        floor,
    )
    current_marine_sc_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (MARINE_SC_CLOUD_TOP_HEIGHT_M - z_atm_mid),
        floor,
    )
    current_high_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (HIGH_CLOUD_TOP_HEIGHT_M - z_atm_mid),
        floor,
    )
    current_high_base_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * (HIGH_CLOUD_BASE_HEIGHT_M - z_atm_mid),
        floor,
    )

    # Cloud BASE temperatures: depend on boundary layer temperature
    boundary = _with_floor(temperature_K[1], floor)
    current_conv_base_K = _with_floor(
        boundary - STANDARD_LAPSE_RATE_K_PER_M * (CONVECTIVE_CLOUD_BASE_HEIGHT_M - z_bl_mid),
        floor,
    )
    current_strat_base_K = _with_floor(
        boundary - STANDARD_LAPSE_RATE_K_PER_M * (STRATIFORM_CLOUD_BASE_HEIGHT_M - z_bl_mid),
        floor,
    )
    current_marine_sc_base_K = _with_floor(
        boundary - STANDARD_LAPSE_RATE_K_PER_M * (MARINE_SC_CLOUD_BASE_HEIGHT_M - z_bl_mid),
        floor,
    )

    # Low cloud OLR goes through atmosphere above (transmittance = 1 - eps_clear)
    # OLR = transmittance * cloud_emission + eps_clear * atm_emission
    transmittance_atm = 1.0 - eps_clear

    # Cloud temperatures now depend on T_atm, not T_surface
    # d(cloud_temp)/d(T_atm) = 1, d(cloud_temp)/d(T_surface) = 0
    _d_olr_dTs = np.zeros_like(surface)
    _d_olr_dTbl = np.zeros_like(surface)

    # OLR derivatives with respect to T_atm (all clouds depend on T_atm via lapse rate)
    # Clear sky contribution
    d_clear_olr_dTatm = clear_frac * 4.0 * eps_clear * sigma * np.power(T_atm_up, 3)

    # Convective clouds: direct to space
    d_conv_olr_dTatm = eps_conv_cloud * conv_frac * 4.0 * sigma * np.power(current_conv_top_K, 3)

    # Low clouds: transmitted fraction + atmosphere-above re-emission
    d_strat_olr_dTatm = strat_frac * (
        transmittance_atm * eps_strat_cloud * 4.0 * sigma * np.power(current_strat_top_K, 3)
        + eps_clear * 4.0 * sigma * np.power(atmosphere, 3)
    )
    d_marine_sc_olr_dTatm = marine_sc_frac * (
        transmittance_atm * eps_marine_sc_cloud * 4.0 * sigma * np.power(current_marine_sc_top_K, 3)
        + eps_clear * 4.0 * sigma * np.power(atmosphere, 3)
    )

    # High clouds: direct to space
    d_high_olr_dTatm = eps_high_cloud * high_effective_frac * 4.0 * sigma * np.power(current_high_top_K, 3)

    _d_olr_dTatm = d_clear_olr_dTatm + d_conv_olr_dTatm + d_strat_olr_dTatm + d_marine_sc_olr_dTatm + d_high_olr_dTatm

    # Downward emission derivatives
    # Clear sky and high cloud downward depend on T_atm
    d_clear_down_dTatm = clear_frac * 4.0 * eps_clear * sigma * np.power(T_atm_down, 3)
    # High cloud downward: transmitted cloud emission + atmosphere re-emission (both depend on T_atm)
    d_high_down_dTatm = high_effective_frac * (
        transmittance_atm * eps_high_cloud * 4.0 * sigma * np.power(current_high_base_K, 3)
        + eps_clear * 4.0 * sigma * np.power(T_atm_down, 3)
    )

    # Low cloud base temps depend on T_bl
    d_conv_down_dTbase = eps_conv_cloud * conv_frac * 4.0 * sigma * np.power(current_conv_base_K, 3)
    d_strat_down_dTbase = eps_strat_cloud * strat_frac * 4.0 * sigma * np.power(current_strat_base_K, 3)
    d_marine_sc_down_dTbase = eps_marine_sc_cloud * marine_sc_frac * 4.0 * sigma * np.power(current_marine_sc_base_K, 3)
    _d_low_cloud_down_dTbase = d_conv_down_dTbase + d_strat_down_dTbase + d_marine_sc_down_dTbase

    _d_down_dTatm = d_clear_down_dTatm + d_high_down_dTatm
    _d_down_dTbl = _d_low_cloud_down_dTbase
    _d_down_dTs = np.zeros_like(surface)

    # Atmosphere diagonal: d(tendency)/dT_atm
    # tendency = SW + lw_absorbed - emitted_toa - downward_lw_weighted
    # d(tendency)/dT_atm = -d(emitted_toa)/dT_atm - d(downward_lw_weighted)/dT_atm
    # (absorption doesn't depend on T_atm)
    atmosphere_diag = (
        -_d_olr_dTatm - _d_down_dTatm
    ) / config.atmosphere_heat_capacity

    boundary = _with_floor(temperature_K[1], floor)

    # BL emissivity: two-band (must match RHS)
    if humidity_q is not None:
        eps_bl = compute_two_band_emissivity(humidity_q, _DP_P0_BL)
    else:
        eps_bl = BOUNDARY_LAYER_EMISSIVITY

    trans_bl = 1.0 - eps_bl

    T_bl_up, T_bl_down = compute_effective_emission_temperatures(
        boundary, eps_bl, BOUNDARY_LAYER_HEIGHT_M, emissivity_dry=0.0
    )

    d_emitted_bl_up_dT = 4.0 * eps_bl * sigma * np.power(T_bl_up, 3)
    d_emitted_bl_down_dT = 4.0 * eps_bl * sigma * np.power(T_bl_down, 3)
    boundary_diag = (
        -(d_emitted_bl_up_dT + d_emitted_bl_down_dT)
    ) / config.boundary_layer_heat_capacity

    cross = np.zeros((3, 3) + surface.shape, dtype=float)

    # Surface <- BL (downward emission)
    cross[0, 1] = d_emitted_bl_down_dT / heat_capacity_field
    # BL <- surface
    cross[1, 0] = 4.0 * eps_bl * config.emissivity_surface * sigma * np.power(surface, 3) / config.boundary_layer_heat_capacity

    # BL absorbs downwelling LW from atmosphere
    cross[1, 0] += eps_bl * _d_down_dTs / config.boundary_layer_heat_capacity
    boundary_diag += eps_bl * _d_down_dTbl / config.boundary_layer_heat_capacity
    cross[1, 2] = eps_bl * _d_down_dTatm / config.boundary_layer_heat_capacity

    # Surface receives transmitted fraction of downwelling atmosphere LW
    surface_diag += (1.0 - eps_bl) * _d_down_dTs / heat_capacity_field
    cross[0, 1] += (1.0 - eps_bl) * _d_down_dTbl / heat_capacity_field
    cross[0, 2] = (1.0 - eps_bl) * _d_down_dTatm / heat_capacity_field

    # === Atmosphere tendency derivatives ===
    # tendency = SW + lw_absorbed - emitted_toa - downward_lw_weighted
    #
    # lw_absorbed depends on upwelling_lw = trans_bl * emitted_surface + emitted_bl_up
    # Total absorption fraction (matching tendency calculation):
    conv_total_abs = eps_conv_cloud + (1.0 - eps_conv_cloud) * eps_clear
    strat_total_abs = eps_strat_cloud + (1.0 - eps_strat_cloud) * eps_clear
    marine_sc_total_abs = eps_marine_sc_cloud + (1.0 - eps_marine_sc_cloud) * eps_clear
    high_total_abs = eps_clear + (1.0 - eps_clear) * eps_high_cloud

    total_abs_frac = (
        clear_frac * eps_clear
        + conv_frac * conv_total_abs
        + strat_frac * strat_total_abs
        + marine_sc_frac * marine_sc_total_abs
        + high_frac * high_total_abs
    )

    # d(upwelling_lw)/dT_surface = trans_bl * d(emitted_surface)/dT_surface
    d_emitted_surface_dT = 4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
    d_upwelling_dTs = trans_bl * d_emitted_surface_dT

    # d(upwelling_lw)/dT_bl = d(emitted_bl_up)/dT_bl
    d_upwelling_dTbl = d_emitted_bl_up_dT

    # Atmosphere <- surface: absorption of upwelling from surface
    cross[2, 0] = total_abs_frac * d_upwelling_dTs / config.atmosphere_heat_capacity

    # Atmosphere <- BL: absorption of upwelling from BL, minus downward emission derivative
    cross[2, 1] = (total_abs_frac * d_upwelling_dTbl - _d_down_dTbl) / config.atmosphere_heat_capacity

    diag = np.stack([surface_diag, boundary_diag, atmosphere_diag])

    return diag, cross


def radiative_equilibrium_initial_guess(
    monthly_insolation: np.ndarray,
    *,
    albedo_field: np.ndarray,
    config: RadiationConfig,
    land_mask: np.ndarray | None = None,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
) -> np.ndarray:
    """Initial temperature guess via local radiative equilibrium.

    Two-pass approach: compute cloud cover from simple temperature estimate,
    then recompute with updated cloud cover from first-pass temperatures.
    """
    sigma = config.stefan_boltzmann
    insolation = monthly_insolation.mean(axis=0)

    if not config.include_atmosphere:
        absorbed = insolation * (1.0 - albedo_field)
        surface = np.power(absorbed / (config.emissivity_surface * sigma), 0.25)
        return _with_floor(surface[np.newaxis, :, :], config.temperature_floor)

    def compute_cloud_properties(temp_surface: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute cloud properties from surface temperature."""
        if lat2d is None or lon2d is None:
            raise ValueError("lat2d and lon2d are required for radiative_equilibrium_initial_guess")
        cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=R_EARTH_METERS)
        itcz = compute_itcz_latitude(temp_surface, lat2d, cell_areas)

        rh = compute_humidity_q(
            lat2d, temp_surface, return_rh=True, land_mask=land_mask, lon_2d=lon2d, itcz_rad=itcz
        )
        dp_norm = _compute_pressure_anomaly(temp_surface, itcz_rad=itcz, lat2d=lat2d, lon2d=lon2d)
        coverage = compute_cloud_coverage(rh, dp_norm, lat2d)
        albedo = compute_cloud_albedo(rh, dp_norm, lat2d)
        top_height = compute_cloud_top_height(dp_norm)
        return coverage, albedo, top_height

    def radiative_equilibrium_temps(
        cloud_coverage: np.ndarray,
        cloud_albedo: np.ndarray,
        cloud_top_height: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve analytical radiative equilibrium for given cloud properties."""
        atm_albedo = 0.05 + cloud_coverage * cloud_albedo
        atm_absorptance = config.shortwave_absorptance_atmosphere + 0.05 * cloud_coverage

        absorbed_sw_atm = atm_absorptance * insolation
        sw_down_surface = (1.0 - atm_albedo - atm_absorptance) * insolation
        absorbed_sw_surface = sw_down_surface * (1.0 - albedo_field)

        eps_clear = config.emissivity_atmosphere
        eps_cloud = (1.0 - eps_clear) * np.sqrt(cloud_coverage)
        eps_atm = eps_clear + eps_cloud
        denom = np.maximum(2.0 - eps_atm, 1e-6)

        surface = np.power((absorbed_sw_surface + 0.5 * absorbed_sw_atm) / (0.5 * denom * sigma), 0.25)
        atmosphere = np.power((absorbed_sw_atm / (eps_atm * sigma)) + 0.5 * np.power(surface, 4), 0.25)
        return surface, atmosphere

    # First pass: estimate cloud properties from simple surface temperature
    absorbed_initial = insolation * (1.0 - albedo_field)
    temp_guess = np.power(absorbed_initial / (config.emissivity_surface * sigma), 0.25)
    temp_guess = np.maximum(temp_guess, config.temperature_floor)
    coverage, albedo, top_height = compute_cloud_properties(temp_guess)

    surface, atmosphere = radiative_equilibrium_temps(coverage, albedo, top_height)

    boundary = 0.7 * surface + 0.3 * atmosphere
    first_pass_temp = np.stack([surface, boundary, atmosphere])

    first_pass_temp = _with_floor(first_pass_temp, config.temperature_floor)

    # Second pass: update cloud properties with first-pass surface temperature
    coverage, albedo, top_height = compute_cloud_properties(first_pass_temp[0])
    surface, atmosphere = radiative_equilibrium_temps(coverage, albedo, top_height)

    boundary = 0.7 * surface + 0.3 * atmosphere
    stacked = np.stack([surface, boundary, atmosphere])

    return _with_floor(stacked, config.temperature_floor)
