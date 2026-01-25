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

    # In 3-layer mode: layer 1 is boundary, layer 2 is atmosphere
    # In 2-layer mode: layer 1 is atmosphere
    if nlayers == 3:
        atmosphere = _with_floor(temperature_K[2], floor)  # Free atmosphere
    else:
        atmosphere = _with_floor(temperature_K[1], floor)  # Atmosphere in 2-layer

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

    # Clear-sky emissivity: humidity-dependent with higher dry floor and lower
    # saturation threshold for free atmosphere vs boundary layer
    if humidity_q is not None:
        eps_clear = compute_humidity_emissivity(
            humidity_q,
            eps_dry=0.55,
            eps_moist=0.80,
            q_reference=0.008,
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
        atmosphere, eps_clear, ATMOSPHERE_LAYER_HEIGHT_M, emissivity_dry=0.55
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

    # Cloud LW emissivities by type
    eps_high_cloud = 0.40
    eps_conv_cloud = 0.50
    eps_strat_cloud = 0.85
    eps_marine_sc_cloud = 0.90

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
    # In 2-layer model, use surface as proxy for BL.
    # T_base = T_bl - Γ × (z_base - z_bl_mid)  [negative dz means warmer]
    if nlayers == 3:
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
    else:
        # 2-layer: use surface as proxy, lapse up from there
        # Surface represents z=0, so T_base = T_surface - Γ × z_base
        current_conv_base_K = _with_floor(
            surface - STANDARD_LAPSE_RATE_K_PER_M * CONVECTIVE_CLOUD_BASE_HEIGHT_M,
            floor,
        )
        current_strat_base_K = _with_floor(
            surface - STANDARD_LAPSE_RATE_K_PER_M * STRATIFORM_CLOUD_BASE_HEIGHT_M,
            floor,
        )
        current_marine_sc_base_K = _with_floor(
            surface - STANDARD_LAPSE_RATE_K_PER_M * MARINE_SC_CLOUD_BASE_HEIGHT_M,
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

    # Shortwave partitioning
    alpha_atm = atm_albedo_field
    beta_atm = config.shortwave_absorptance_atmosphere + 0.05 * cloud_coverage
    absorbed_shortwave_atm = beta_atm * insolation_W_m2
    sw_down_surface = (1.0 - alpha_atm - beta_atm) * insolation_W_m2
    absorbed_shortwave_sfc = sw_down_surface * (1.0 - albedo_field)

    # Longwave budget
    downward_longwave = downward_lw_weighted
    absorbed_from_surface = eps_atm * emitted_surface

    if nlayers == 2:
        surface_tendency = (
            absorbed_shortwave_sfc + downward_longwave - emitted_surface
        ) / heat_capacity_field

        # Atmosphere loses its own emission only (clouds emit from condensed water)
        atmosphere_tendency = (
            absorbed_shortwave_atm + absorbed_from_surface - atm_own_emission_up - atm_own_emission_down
        ) / config.atmosphere_heat_capacity

        return np.stack([surface_tendency, atmosphere_tendency])

    elif nlayers == 3:
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

        # BL emissivity: humidity-dependent
        if humidity_q is not None:
            eps_bl = compute_humidity_emissivity(
                humidity_q,
                config.emissivity_bl_dry,
                config.emissivity_bl_moist,
            )
        else:
            eps_bl = config.boundary_layer_emissivity

        # BL emission with up/down asymmetry (well-mixed, so emissivity_dry=0)
        T_bl_up, T_bl_down = compute_effective_emission_temperatures(
            boundary, eps_bl, BOUNDARY_LAYER_HEIGHT_M, emissivity_dry=0.0
        )
        emitted_bl_up = eps_bl * sigma * np.power(T_bl_up, 4)
        emitted_bl_down = eps_bl * sigma * np.power(T_bl_down, 4)

        tau_bl = 1.0 - eps_bl
        tau_atm = 1.0 - eps_atm

        # BL gets ~60% of atmospheric SW absorption (water vapor, aerosols, low clouds)
        bl_sw_fraction = 0.60
        absorbed_shortwave_bl = bl_sw_fraction * absorbed_shortwave_atm
        absorbed_shortwave_atm_upper = (1.0 - bl_sw_fraction) * absorbed_shortwave_atm

        downward_longwave_to_surface = emitted_bl_down + tau_bl * downward_lw_weighted
        surface_tendency = (
            absorbed_shortwave_sfc + downward_longwave_to_surface - emitted_surface
        ) / heat_capacity_field

        absorbed_from_surface_bl = eps_bl * emitted_surface
        absorbed_from_atm_bl = eps_bl * downward_lw_weighted
        boundary_tendency = (
            absorbed_shortwave_bl + absorbed_from_surface_bl + absorbed_from_atm_bl
            - emitted_bl_up - emitted_bl_down
        ) / config.boundary_layer_heat_capacity

        transmitted_surface_to_atm = tau_bl * emitted_surface
        absorbed_from_surface_atm = eps_atm * transmitted_surface_to_atm
        absorbed_from_boundary_atm = eps_atm * emitted_bl_up

        # Atmosphere loses its own emission only (clouds emit from condensed water)
        atmosphere_tendency = (
            absorbed_shortwave_atm_upper + absorbed_from_surface_atm + absorbed_from_boundary_atm
            - atm_own_emission_up - atm_own_emission_down
        ) / config.atmosphere_heat_capacity

        if log_diagnostics:
            if cell_area_m2 is None:
                raise ValueError("cell_area_m2 must be provided when log_diagnostics=True")

            reflected_by_atm = alpha_atm * insolation_W_m2
            reflected_by_surface = albedo_field * sw_down_surface

            # Total outgoing longwave radiation (OLR) at TOA
            olr_total = emitted_toa + tau_atm * emitted_bl_up + tau_atm * tau_bl * emitted_surface

            print("\n=== Radiation Diagnostics (W/m²) ===")
            print("\nLayer Temperatures (K):")
            print(f"  Surface:                         {area_weighted_mean(surface, cell_area_m2):7.2f}")
            print(f"  Boundary layer:                  {area_weighted_mean(boundary, cell_area_m2):7.2f}")
            print(f"  Atmosphere:                      {area_weighted_mean(atmosphere, cell_area_m2):7.2f}")
            print("\nShortwave Fluxes:")
            print(f"Incoming solar (TOA):              {area_weighted_mean(insolation_W_m2, cell_area_m2):7.2f}")
            print(f"Reflected by atmosphere:           {area_weighted_mean(reflected_by_atm, cell_area_m2):7.2f}")
            print(f"Absorbed by atmosphere (SW total): {area_weighted_mean(absorbed_shortwave_atm, cell_area_m2):7.2f}")
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
            print(f"  - boundary layer emission:       {area_weighted_mean(absorbed_from_boundary_atm, weights=cell_area_m2):7.2f}")
            print("\nNet Radiation Balances:")
            print(f"Net surface radiation balance:     {area_weighted_mean(absorbed_shortwave_sfc + downward_longwave_to_surface - emitted_surface, cell_area_m2):7.2f}")
            print(f"Net boundary layer balance:        {area_weighted_mean(absorbed_shortwave_bl + absorbed_from_surface_bl + absorbed_from_atm_bl - emitted_bl_up - emitted_bl_down, cell_area_m2):7.2f}")
            print(f"Net atmosphere balance:            {area_weighted_mean(absorbed_shortwave_atm_upper + absorbed_from_surface_atm + absorbed_from_boundary_atm - atm_own_emission_up - atm_own_emission_down, cell_area_m2):7.2f}")
            print("\nGlobal Energy Balance:")
            print(f"Net TOA balance (SW_in - OLR):     {area_weighted_mean(insolation_W_m2 - reflected_by_atm - reflected_by_surface - olr_total, cell_area_m2):7.2f}")
            print("=" * 40)

        return np.stack([surface_tendency, boundary_tendency, atmosphere_tendency])

    else:
        raise ValueError(f"Unsupported number of atmosphere layers: {nlayers}")


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
    For 2+ layers: tuple of (diag, cross) where:
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

    if nlayers == 3:
        atmosphere = _with_floor(temperature_K[2], floor)
    else:
        atmosphere = _with_floor(temperature_K[1], floor)

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

    # Clear-sky emissivity (same parameters as RHS)
    if humidity_q is not None:
        eps_clear = compute_humidity_emissivity(
            humidity_q,
            eps_dry=0.55,
            eps_moist=0.80,
            q_reference=0.008,
        )
    else:
        eps_clear = config.emissivity_atmosphere
    eps_cloud = (1.0 - eps_clear) * np.sqrt(cloud_coverage)
    eps_atm = eps_clear + eps_cloud

    surface_diag = (
        -4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
    ) / heat_capacity_field

    T_atm_up, T_atm_down = compute_effective_emission_temperatures(
        atmosphere, eps_atm, ATMOSPHERE_LAYER_HEIGHT_M, emissivity_dry=0.55
    )

    # Clear fraction (must match RHS calculation)
    low_cloud_frac = np.minimum(conv_frac + strat_frac + marine_sc_frac, 1.0)
    clear_of_low = 1.0 - low_cloud_frac
    high_effective_frac = high_frac * clear_of_low
    clear_frac = 1.0 - low_cloud_frac - high_effective_frac

    # Cloud LW emissivities (must match RHS)
    eps_high_cloud = 0.40
    eps_conv_cloud = 0.50
    eps_strat_cloud = 0.85
    eps_marine_sc_cloud = 0.90

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

    # Cloud BASE temperatures: depend on boundary layer in 3-layer, surface in 2-layer
    if nlayers == 3:
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
    else:
        # 2-layer: use surface as proxy, lapse up from z=0
        current_conv_base_K = _with_floor(
            surface - STANDARD_LAPSE_RATE_K_PER_M * CONVECTIVE_CLOUD_BASE_HEIGHT_M,
            floor,
        )
        current_strat_base_K = _with_floor(
            surface - STANDARD_LAPSE_RATE_K_PER_M * STRATIFORM_CLOUD_BASE_HEIGHT_M,
            floor,
        )
        current_marine_sc_base_K = _with_floor(
            surface - STANDARD_LAPSE_RATE_K_PER_M * MARINE_SC_CLOUD_BASE_HEIGHT_M,
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

    # Low cloud base temps: depend on T_bl in 3-layer, T_surface in 2-layer
    d_conv_down_dTbase = eps_conv_cloud * conv_frac * 4.0 * sigma * np.power(current_conv_base_K, 3)
    d_strat_down_dTbase = eps_strat_cloud * strat_frac * 4.0 * sigma * np.power(current_strat_base_K, 3)
    d_marine_sc_down_dTbase = eps_marine_sc_cloud * marine_sc_frac * 4.0 * sigma * np.power(current_marine_sc_base_K, 3)
    _d_low_cloud_down_dTbase = d_conv_down_dTbase + d_strat_down_dTbase + d_marine_sc_down_dTbase

    # Partition derivatives based on layer count
    if nlayers == 3:
        # Low cloud bases depend on T_bl
        _d_down_dTatm = d_clear_down_dTatm + d_high_down_dTatm
        _d_down_dTbl = _d_low_cloud_down_dTbase
        _d_down_dTs = np.zeros_like(surface)
    else:
        # 2-layer: low cloud bases depend on T_surface
        _d_down_dTatm = d_clear_down_dTatm + d_high_down_dTatm
        _d_down_dTbl = np.zeros_like(surface)
        _d_down_dTs = _d_low_cloud_down_dTbase

    # Atmosphere loses its own emission only (clouds emit separately)
    d_atm_own_emission_dT = 4.0 * eps_clear * sigma * (np.power(T_atm_up, 3) + np.power(T_atm_down, 3))

    atmosphere_diag = (
        -d_atm_own_emission_dT
    ) / config.atmosphere_heat_capacity

    if nlayers == 2:
        surface_coupling = _d_down_dTatm / heat_capacity_field
        surface_diag += _d_down_dTs / heat_capacity_field

        atmosphere_coupling = (
            4.0 * eps_atm * config.emissivity_surface * sigma * np.power(surface, 3)
        ) / config.atmosphere_heat_capacity

        diag = np.stack([surface_diag, atmosphere_diag])
        cross = np.zeros((2, 2) + surface.shape, dtype=float)
        cross[0, 1] = surface_coupling
        cross[1, 0] = atmosphere_coupling
        return diag, cross

    elif nlayers == 3:
        boundary = _with_floor(temperature_K[1], floor)

        if humidity_q is not None:
            eps_bl = compute_humidity_emissivity(
                humidity_q,
                config.emissivity_bl_dry,
                config.emissivity_bl_moist,
            )
        else:
            eps_bl = config.boundary_layer_emissivity

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

        # Atmosphere <- BL (upwelling)
        cross[2, 1] = 4.0 * eps_atm * eps_bl * sigma * np.power(T_bl_up, 3) / config.atmosphere_heat_capacity

        # Surface receives transmitted fraction of downwelling atmosphere LW
        surface_diag += (1.0 - eps_bl) * _d_down_dTs / heat_capacity_field
        cross[0, 1] += (1.0 - eps_bl) * _d_down_dTbl / heat_capacity_field
        cross[0, 2] = (1.0 - eps_bl) * _d_down_dTatm / heat_capacity_field

        # Atmosphere <- surface (transmitted through BL)
        cross[2, 0] = 4.0 * eps_atm * (1.0 - eps_bl) * config.emissivity_surface * sigma * np.power(surface, 3) / config.atmosphere_heat_capacity

        diag = np.stack([surface_diag, boundary_diag, atmosphere_diag])

        return diag, cross

    else:
        raise ValueError(f"Unsupported number of atmosphere layers: {nlayers}")


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

    if config.boundary_layer_heat_capacity != config.atmosphere_heat_capacity:
        boundary = 0.7 * surface + 0.3 * atmosphere
        first_pass_temp = np.stack([surface, boundary, atmosphere])
    else:
        first_pass_temp = np.stack([surface, atmosphere])

    first_pass_temp = _with_floor(first_pass_temp, config.temperature_floor)

    # Second pass: update cloud properties with first-pass surface temperature
    coverage, albedo, top_height = compute_cloud_properties(first_pass_temp[0])
    surface, atmosphere = radiative_equilibrium_temps(coverage, albedo, top_height)

    if config.boundary_layer_heat_capacity != config.atmosphere_heat_capacity:
        boundary = 0.7 * surface + 0.3 * atmosphere
        stacked = np.stack([surface, boundary, atmosphere])
    else:
        stacked = np.stack([surface, atmosphere])

    return _with_floor(stacked, config.temperature_floor)
