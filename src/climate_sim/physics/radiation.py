"""Radiative column model components with optional atmospheric layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.humidity import (
    compute_cloud_cover,
    specific_humidity_to_relative_humidity,
    compute_humidity_q,
)
from climate_sim.physics.atmosphere.pressure import compute_pressure
from climate_sim.physics.atmosphere.hadley import compute_itcz_latitude
from climate_sim.core.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.data.constants import (
    ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
    ATMOSPHERE_LAYER_HEIGHT_M,
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    BOUNDARY_LAYER_EMISSIVITY,
    BOUNDARY_LAYER_HEIGHT_M,
    R_EARTH_METERS,
    SHORTWAVE_ABSORPTANCE_ATMOSPHERE,
    STANDARD_LAPSE_RATE_K_PER_M,
)

# Water vapor scale height (~2 km observed globally)
# Used to compute effective emission height for atmosphere layer
# Since water vapor is exponentially distributed, emission comes from
# a thin layer near τ ≈ 1, not uniformly from the full geometric height
WATER_VAPOR_SCALE_HEIGHT_M = 2000.0


@dataclass(frozen=True)
class RadiationConfig:
    """Container for radiative transfer parameters.

    Emissivity is humidity-dependent:
    - Water vapor is the primary greenhouse gas
    - Dry air (deserts): low emissivity → atmosphere is transparent → surface radiates to space
    - Moist air (tropics, ocean): high emissivity → strong greenhouse effect
    - Clouds: essentially blackbody (eps ≈ 1.0)
    """

    stefan_boltzmann: float = 5.670374419e-8  # W m-2 K-4
    emissivity_surface: float = 1.0
    # Base emissivity for dry air (minimum) - includes CO2, O3, and other well-mixed gases
    # Even bone-dry air has significant greenhouse effect from CO2 (~0.4-0.5)
    emissivity_atmosphere_dry: float = 0.45
    # Maximum emissivity for saturated air
    emissivity_atmosphere_moist: float = 0.85
    # BL emissivity range (higher than free atm due to higher water vapor density)
    emissivity_bl_dry: float = 0.55  # Dry BL (desert) - still has CO2 + some water vapor
    emissivity_bl_moist: float = 0.80  # Saturated BL (tropical ocean)
    # Legacy constant (used as fallback if humidity not available)
    emissivity_atmosphere: float = 0.60
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

    Parameters
    ----------
    humidity_q : np.ndarray
        Specific humidity (kg/kg).
    eps_dry : float
        Emissivity for completely dry air.
    eps_moist : float
        Emissivity for saturated tropical air.
    q_reference : float
        Reference humidity for scaling (typical moist BL value).

    Returns
    -------
    np.ndarray
        Emissivity field, clamped to [eps_dry, eps_moist].
    """
    # Linear interpolation with clamping
    # At q=0: eps = eps_dry
    # At q=q_reference: eps = eps_moist
    humidity_fraction = np.clip(humidity_q / q_reference, 0.0, 1.0)
    emissivity = eps_dry + (eps_moist - eps_dry) * humidity_fraction
    return emissivity


def compute_emission_asymmetry_factor(emissivity: np.ndarray | float) -> np.ndarray | float:
    """Compute the emission asymmetry factor for a grey slab with lapse rate.

    For a layer with a linear temperature profile (lapse rate), emission in
    different directions comes from different effective heights:
    - Upward emission comes from colder upper levels
    - Downward emission comes from warmer lower levels

    The asymmetry factor f satisfies:
        T_eff_up = T_mid - (ΔT/2) × f
        T_eff_down = T_mid + (ΔT/2) × f

    where ΔT = lapse_rate × layer_height.

    Derivation: For a grey slab with optical depth τ_H and linear temperature
    profile, integrating the Planck-weighted emission gives:
        f = 1 - 2g/ε
    where:
        g = (1/τ_H)[1 - (τ_H + 1)exp(-τ_H)]
        ε = 1 - exp(-τ_H) (emissivity)
        τ_H = -ln(1 - ε) (optical depth)

    Limits:
    - Optically thin (ε → 0): f → 0 (uniform emission, no asymmetry)
    - Optically thick (ε → 1): f → 1 (emission from boundaries)

    Parameters
    ----------
    emissivity : np.ndarray | float
        Layer emissivity (0 to 1).

    Returns
    -------
    np.ndarray | float
        Asymmetry factor f (0 to 1).
    """
    eps = np.asarray(emissivity)
    # Avoid numerical issues at ε = 0 or 1
    eps_safe = np.clip(eps, 1e-6, 1.0 - 1e-6)

    # Optical depth from emissivity: ε = 1 - exp(-τ), so τ = -ln(1 - ε)
    tau_H = -np.log(1.0 - eps_safe)

    # g = (1/τ)[1 - (τ + 1)exp(-τ)]
    exp_neg_tau = 1.0 - eps_safe  # = exp(-τ)
    g = (1.0 - (tau_H + 1.0) * exp_neg_tau) / tau_H

    # f = 1 - 2g/ε
    f = 1.0 - 2.0 * g / eps_safe

    # Clamp to valid range
    return np.clip(f, 0.0, 1.0)


def compute_effective_emission_height(
    emissivity: np.ndarray | float,
    layer_height_m: float,
    emissivity_dry: float = 0.0,
) -> np.ndarray | float:
    """Compute effective emission height for asymmetry calculation.

    The atmosphere contains two types of IR absorbers with different profiles:
    1. Well-mixed gases (CO2, O3, CH4, etc.) - uniform with height
    2. Water vapor - exponentially distributed (scale height ~2 km)

    For the well-mixed component, effective height uses the uniform absorber
    formula: Δz/H ≈ 0.16 × τ (derived from weighting function integral).

    For water vapor, effective height uses the exponential formula:
    H_eff = H_wv × 0.5 × τ.

    The combined H_eff is the emissivity-weighted average.

    Parameters
    ----------
    emissivity : np.ndarray | float
        Total layer emissivity (including both components).
    layer_height_m : float
        Geometric layer thickness (m).
    emissivity_dry : float
        Baseline emissivity from well-mixed gases (CO2, etc.).
        If 0, uses geometric height (appropriate for BL).
        If > 0, computes combined H_eff for atmosphere.

    Returns
    -------
    np.ndarray | float
        Effective height for ΔT calculation (m).
    """
    eps_total = np.asarray(emissivity)

    if emissivity_dry <= 0:
        # No well-mixed component specified: use geometric height (BL case)
        return layer_height_m

    # Decompose into well-mixed (CO2) and water vapor components
    eps_dry = np.minimum(emissivity_dry, eps_total - 1e-6)
    eps_H2O = np.maximum(eps_total - eps_dry, 1e-6)

    # Well-mixed component (CO2, O3, etc.): uniform absorber formula
    # For uniform absorber: Δz/H ≈ 0.16 × τ (linear approximation for τ < 2)
    eps_dry_safe = np.clip(eps_dry, 1e-6, 1.0 - 1e-6)
    tau_dry = -np.log(1.0 - eps_dry_safe)
    H_eff_dry = layer_height_m * 0.16 * tau_dry

    # Water vapor component: exponential absorber formula
    # H_eff = H_wv × 0.5 × τ
    eps_H2O_safe = np.clip(eps_H2O, 1e-6, 1.0 - 1e-6)
    tau_H2O = -np.log(1.0 - eps_H2O_safe)
    H_eff_H2O = WATER_VAPOR_SCALE_HEIGHT_M * 0.5 * tau_H2O

    # Weighted average by emissivity contribution
    eps_total_safe = np.maximum(eps_total, 1e-6)
    H_eff = (eps_dry * H_eff_dry + eps_H2O * H_eff_H2O) / eps_total_safe

    return H_eff


def compute_effective_emission_temperatures(
    temperature_mid: np.ndarray,
    emissivity: np.ndarray | float,
    layer_height_m: float,
    lapse_rate_K_per_m: float = STANDARD_LAPSE_RATE_K_PER_M,
    emissivity_dry: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute effective emission temperatures for upward and downward radiation.

    For layers with lapse rate, upward emission comes from colder upper levels
    and downward emission from warmer lower levels. The effective temperatures
    depend on the layer's optical structure:
    - BL (well-mixed): use geometric height for ΔT (emissivity_dry=0)
    - Free atmosphere: combine well-mixed (CO2) and exponential (H2O) absorbers

    Parameters
    ----------
    temperature_mid : np.ndarray
        Mid-layer temperature (K).
    emissivity : np.ndarray | float
        Total layer emissivity.
    layer_height_m : float
        Geometric layer thickness (m).
    lapse_rate_K_per_m : float
        Temperature lapse rate (K/m), default 6.5 K/km.
    emissivity_dry : float
        Baseline emissivity from well-mixed gases (CO2, etc.).
        Set to 0 for BL (uses geometric height).
        Set to ~0.55 for atmosphere (combines CO2 + H2O).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (T_eff_up, T_eff_down) - effective emission temperatures for
        upward and downward radiation respectively.
    """
    H_eff = compute_effective_emission_height(
        emissivity, layer_height_m, emissivity_dry
    )
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

    Observations show:
    - ITCZ (low pressure): 50-60% coverage (all cloud types combined)
    - Subtropics (high pressure): 30-40% average (mostly clear with patches of stratocumulus)
    - Polar (|lat| > 60°): 60-70% coverage (low stratus/fog)
    - Cloud cover generally higher over ocean than land
    """
    rh_min = 0.35
    rh_max = 0.85
    rh_smooth = np.clip((rh - rh_min) / (rh_max - rh_min), 0, 1)
    rh_smooth = 3 * rh_smooth**2 - 2 * rh_smooth**3

    # Convective: ITCZ with all cloud types (50-60%)
    c_conv = 0.40 + 0.20 * rh_smooth  # [0.40, 0.60]

    # Stratocumulus: subtropical dry zones (30-40%)
    # (Note: localized marine stratocumulus regions can be 70-90%, but
    # subtropical average is lower due to large dry areas)
    c_strat = 0.20 + 0.20 * rh_smooth  # [0.20, 0.40]

    # Interpolate based on pressure
    weight_strat = 0.5 * (1.0 + dp_norm)
    coverage = c_conv + (c_strat - c_conv) * weight_strat

    # Polar regime: boost coverage at high latitudes (|lat| > 60°)
    if lat_deg is not None:
        lat_abs = np.abs(lat_deg)
        # Smooth transition starting at 60°, full effect at 75°
        polar_factor = np.clip((lat_abs - 60.0) / 15.0, 0, 1)
        polar_factor = 3 * polar_factor**2 - 2 * polar_factor**3  # Smoothstep
        # Boost to 60-70% at poles
        polar_coverage = 0.50 + 0.20 * rh_smooth  # [0.50, 0.70]
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

    Effective atmospheric albedo (CERES observations):
    - Tropical ITCZ: ~0.15-0.25 (60% coverage × 0.25 albedo + 0.05 clear sky)
    - Subtropical average: ~0.19 (34% coverage × 0.41 albedo + 0.05 clear sky)
    - Polar: ~0.20 (65% coverage × 0.25 albedo + 0.05 clear sky, thin stratus)

    Key insight: ITCZ clouds are mostly thin cirrus/anvils (low albedo per cloud)
    despite occasional thick convective cores. Subtropical regions are mostly
    clear with localized bright stratocumulus patches. Polar clouds are thin
    low stratus/fog with low albedo per cloud.
    """
    rh_min = 0.35
    rh_max = 0.85
    rh_smooth = np.clip((rh - rh_min) / (rh_max - rh_min), 0, 1)
    rh_smooth = 3 * rh_smooth**2 - 2 * rh_smooth**3

    # Convective: mostly thin anvils/cirrus with some thick cores
    alpha_conv = 0.20 + 0.10 * rh_smooth  # [0.20, 0.30]

    # Stratocumulus: thicker, more uniform decks
    alpha_strat = 0.30 + 0.15 * rh_smooth  # [0.30, 0.45]

    # Interpolate based on pressure
    weight_strat = 0.5 * (1.0 + dp_norm)
    alpha_cloud = alpha_conv + (alpha_strat - alpha_conv) * weight_strat

    # Polar regime: thin stratus/fog at high latitudes (|lat| > 60°)
    if lat_deg is not None:
        lat_abs = np.abs(lat_deg)
        polar_factor = np.clip((lat_abs - 60.0) / 15.0, 0, 1)
        polar_factor = 3 * polar_factor**2 - 2 * polar_factor**3  # Smoothstep
        # Polar clouds: thin stratus with low albedo
        alpha_polar = 0.20 + 0.10 * rh_smooth  # [0.20, 0.30]
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

    # Compute cloud properties from humidity and pressure
    if humidity_q is not None and itcz_rad is not None:
        rh = specific_humidity_to_relative_humidity(humidity_q, surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)
        dp_norm = _compute_pressure_anomaly(surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)

        cloud_coverage = compute_cloud_coverage(rh, dp_norm, lat2d)
        cloud_albedo = compute_cloud_albedo(rh, dp_norm, lat2d)
        cloud_top_height = compute_cloud_top_height(dp_norm)
    else:
        # Fallback to old cloud cover parameterization
        if humidity_q is not None:
            rh = specific_humidity_to_relative_humidity(humidity_q, surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)
            cloud_coverage = compute_cloud_cover(relative_humidity=rh, land_mask=land_mask)
        else:
            cloud_coverage = compute_cloud_cover(temperature=temperature_K, land_mask=land_mask)
        cloud_albedo = np.array(0.35)
        cloud_top_height = config.cloud_top_delta_z_m

    # Shortwave: atmospheric albedo
    atm_albedo_field = 0.05 + cloud_coverage * cloud_albedo

    # Log cloud diagnostics if requested
    if log_diagnostics and cell_area_m2 is not None:
        total_area = np.sum(cell_area_m2)
        mean_coverage = area_weighted_mean(cloud_coverage, cell_area_m2 / total_area)
        mean_albedo_per_cloud = area_weighted_mean(cloud_albedo, cell_area_m2 / total_area)
        mean_atm_albedo = area_weighted_mean(atm_albedo_field, cell_area_m2 / total_area)
        print(f"[CLOUD] Global mean coverage: {mean_coverage:.3f}")
        print(f"[CLOUD] Global mean albedo per cloud: {mean_albedo_per_cloud:.3f}")
        print(f"[CLOUD] Global mean atmospheric albedo: {mean_atm_albedo:.3f}")

    # Longwave: atmospheric emissivity (humidity-dependent)
    sigma = config.stefan_boltzmann
    eps_sfc = config.emissivity_surface

    # Clear-sky free atmosphere emissivity depends on water vapor content
    # Free atm is drier than BL, but has longer path length (~8 km vs 1 km)
    # so even at lower q it has significant optical depth.
    # CO2/O3 contribute ~0.25-0.30 baseline regardless of humidity.
    # Use BL humidity directly but with parameters tuned for free atm:
    # - Higher dry floor (0.55 vs 0.45) due to CO2/O3 + path length
    # - Lower q_reference (8 g/kg) since free atm saturates at lower humidity
    if humidity_q is not None:
        eps_clear = compute_humidity_emissivity(
            humidity_q,
            eps_dry=0.55,    # CO2/O3 baseline + path length effect
            eps_moist=0.80,
            q_reference=0.008,  # 8 g/kg - free atm saturates at lower q than BL
        )
    else:
        eps_clear = config.emissivity_atmosphere

    # Cloud contribution: sqrt saturation (clouds saturate LW quickly)
    # Individual clouds ≈ blackbody, but grid-cell doesn't fill perfectly
    eps_cloud = (1.0 - eps_clear) * np.sqrt(cloud_coverage)
    eps_atm = eps_clear + eps_cloud

    emitted_surface = eps_sfc * sigma * np.power(surface, 4)

    # Compute effective emission temperatures for atmosphere (up vs down asymmetry)
    # Due to lapse rate within the layer, upward emission comes from colder levels,
    # downward emission from warmer levels.
    # The atmosphere has two absorber types:
    # - Well-mixed (CO2, O3): baseline emissivity ~0.55, uniform with height
    # - Water vapor: humidity-dependent, exponentially distributed (scale height ~2 km)
    # The emissivity_dry parameter triggers the combined H_eff calculation.
    T_atm_up, T_atm_down = compute_effective_emission_temperatures(
        atmosphere, eps_atm, ATMOSPHERE_LAYER_HEIGHT_M, emissivity_dry=0.55
    )
    emitted_atm_up = eps_atm * sigma * np.power(T_atm_up, 4)
    emitted_atm_down = eps_atm * sigma * np.power(T_atm_down, 4)

    # Cloud-top correction for TOA longwave emission
    cloud_top_K = _with_floor(
        T_atm_up - STANDARD_LAPSE_RATE_K_PER_M * cloud_top_height,
        floor,
    )
    emitted_toa = (
        eps_clear * sigma * np.power(T_atm_up, 4)
        + eps_cloud * sigma * np.power(cloud_top_K, 4)
    )

    # Shortwave partitioning
    alpha_atm = atm_albedo_field
    beta_atm = config.shortwave_absorptance_atmosphere + 0.05 * cloud_coverage

    # SW absorbed in atmosphere
    absorbed_shortwave_atm = beta_atm * insolation_W_m2

    # SW reaching surface, then partially absorbed
    sw_down_surface = (1.0 - alpha_atm - beta_atm) * insolation_W_m2
    absorbed_shortwave_sfc = sw_down_surface * (1.0 - albedo_field)

    # Longwave
    downward_longwave = emitted_atm_down
    absorbed_from_surface = eps_atm * emitted_surface

    if nlayers == 2:
        # Two-layer system: surface + atmosphere (no boundary layer)
        surface_tendency = (
            absorbed_shortwave_sfc + downward_longwave - emitted_surface
        ) / heat_capacity_field

        atmosphere_tendency = (
            absorbed_shortwave_atm + absorbed_from_surface - emitted_atm_down - emitted_toa
        ) / config.atmosphere_heat_capacity

        return np.stack([surface_tendency, atmosphere_tendency])

    elif nlayers == 3:
        # Three-layer system: surface + boundary layer + atmosphere
        boundary = _with_floor(temperature_K[1], floor)

        # BL emissivity depends on humidity (water vapor is the absorber)
        if humidity_q is not None:
            eps_bl = compute_humidity_emissivity(
                humidity_q,
                config.emissivity_bl_dry,
                config.emissivity_bl_moist,
            )
        else:
            eps_bl = config.boundary_layer_emissivity

        # Compute effective emission temperatures for BL (up vs down asymmetry)
        # BL is well-mixed, so use geometric height directly (emissivity_dry=0)
        T_bl_up, T_bl_down = compute_effective_emission_temperatures(
            boundary, eps_bl, BOUNDARY_LAYER_HEIGHT_M, emissivity_dry=0.0
        )
        emitted_bl_up = eps_bl * sigma * np.power(T_bl_up, 4)
        emitted_bl_down = eps_bl * sigma * np.power(T_bl_down, 4)

        # Longwave transmissivities (grey, no-scattering assumption)
        tau_bl = 1.0 - eps_bl
        tau_atm = 1.0 - eps_atm

        # Split atmospheric SW absorption between BL and free atmosphere
        # BL contains most water vapor, aerosols, and low clouds → gets ~60% of SW absorption
        bl_sw_fraction = 0.60
        absorbed_shortwave_bl = bl_sw_fraction * absorbed_shortwave_atm
        absorbed_shortwave_atm_upper = (1.0 - bl_sw_fraction) * absorbed_shortwave_atm

        downward_longwave_to_surface = emitted_bl_down + tau_bl * emitted_atm_down
        surface_tendency = (
            absorbed_shortwave_sfc + downward_longwave_to_surface - emitted_surface
        ) / heat_capacity_field

        absorbed_from_surface_bl = eps_bl * emitted_surface
        absorbed_from_atm_bl = eps_bl * emitted_atm_down
        boundary_tendency = (
            absorbed_shortwave_bl + absorbed_from_surface_bl + absorbed_from_atm_bl
            - emitted_bl_up - emitted_bl_down
        ) / config.boundary_layer_heat_capacity

        transmitted_surface_to_atm = tau_bl * emitted_surface
        absorbed_from_surface_atm = eps_atm * transmitted_surface_to_atm
        absorbed_from_boundary_atm = eps_atm * emitted_bl_up
        atmosphere_tendency = (
            absorbed_shortwave_atm_upper + absorbed_from_surface_atm + absorbed_from_boundary_atm
            - emitted_atm_down - emitted_toa
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
            print(f"Atmosphere emission (total):       {area_weighted_mean(emitted_atm_up + emitted_atm_down, cell_area_m2):7.2f}")
            print(f"  - downward (to BL):              {area_weighted_mean(emitted_atm_down, cell_area_m2):7.2f}")
            print(f"  - upward (to space):             {area_weighted_mean(emitted_atm_up, cell_area_m2):7.2f}")
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
            print(f"Net atmosphere balance:            {area_weighted_mean(absorbed_shortwave_atm_upper + absorbed_from_surface_atm + absorbed_from_boundary_atm - emitted_atm_down - emitted_toa, cell_area_m2):7.2f}")
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
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Partial derivatives of the radiative tendency with respect to temperature."""

    floor = config.temperature_floor
    sigma = config.stefan_boltzmann

    if not config.include_atmosphere:
        surface = _with_floor(temperature_K[0], floor)
        coeff = -4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
        return (coeff / heat_capacity_field)[np.newaxis, :, :]

    surface = _with_floor(temperature_K[0], floor)

    # Check nlayers early to use correct atmosphere layer
    nlayers = temperature_K.shape[0]

    if nlayers == 3:
        atmosphere = _with_floor(temperature_K[2], floor)
    else:
        atmosphere = _with_floor(temperature_K[1], floor)

    # Compute cloud properties (frozen for linearization)
    if humidity_q is not None and itcz_rad is not None:
        rh = specific_humidity_to_relative_humidity(humidity_q, surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)
        dp_norm = _compute_pressure_anomaly(surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)
        cloud_coverage = compute_cloud_coverage(rh, dp_norm, lat2d)
        cloud_top_height = compute_cloud_top_height(dp_norm)
    else:
        if humidity_q is not None:
            rh = specific_humidity_to_relative_humidity(humidity_q, surface, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)
            cloud_coverage = compute_cloud_cover(relative_humidity=rh, land_mask=land_mask)
        else:
            cloud_coverage = compute_cloud_cover(temperature=temperature_K, land_mask=land_mask)
        cloud_top_height = config.cloud_top_delta_z_m

    # Humidity-dependent clear-sky emissivity (frozen for linearization)
    # Same parameters as RHS: higher dry floor, lower q_reference for free atm
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

    cloud_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * cloud_top_height,
        floor,
    )

    surface_diag = (
        -4.0 * config.emissivity_surface * sigma * np.power(surface, 3)
    ) / heat_capacity_field

    # Compute effective emission temperatures for atmosphere (same as RHS)
    # These are needed for correct Jacobian with asymmetric emission
    T_atm_up, T_atm_down = compute_effective_emission_temperatures(
        atmosphere, eps_atm, ATMOSPHERE_LAYER_HEIGHT_M, emissivity_dry=0.55
    )

    # Atmosphere loses LW by emitting up and down, but TOA emission is corrected
    # to include a colder cloud-top contribution.
    # Use effective temperatures for correct derivatives
    d_emitted_atm_down_dT = 4.0 * eps_atm * sigma * np.power(T_atm_down, 3)
    d_emitted_atm_up_dT = 4.0 * eps_atm * sigma * np.power(T_atm_up, 3)
    d_emitted_toa_dT = (
        4.0 * eps_clear * sigma * np.power(T_atm_up, 3)
        + 4.0 * eps_cloud * sigma * np.power(cloud_top_K, 3)
    )
    atmosphere_diag = (
        -(d_emitted_atm_down_dT + d_emitted_toa_dT)
    ) / config.atmosphere_heat_capacity

    if nlayers == 2:
        surface_coupling = (
            4. * eps_atm * sigma * np.power(T_atm_down, 3)
        ) / heat_capacity_field
        atmosphere_coupling = (
            4. * eps_atm * config.emissivity_surface * sigma * np.power(surface, 3)
        ) / config.atmosphere_heat_capacity

        diag = np.stack([surface_diag, atmosphere_diag])
        cross = np.zeros((2, 2) + surface.shape, dtype=float)
        cross[0, 1] = surface_coupling
        cross[1, 0] = atmosphere_coupling
        return diag, cross

    elif nlayers == 3:
        boundary = _with_floor(temperature_K[1], floor)

        # Humidity-dependent BL emissivity (frozen for linearization)
        if humidity_q is not None:
            eps_bl = compute_humidity_emissivity(
                humidity_q,
                config.emissivity_bl_dry,
                config.emissivity_bl_moist,
            )
        else:
            eps_bl = config.boundary_layer_emissivity

        # Compute effective emission temperatures for BL (same as RHS)
        T_bl_up, T_bl_down = compute_effective_emission_temperatures(
            boundary, eps_bl, BOUNDARY_LAYER_HEIGHT_M, emissivity_dry=0.0
        )

        # BL emits both up and down
        d_emitted_bl_up_dT = 4.0 * eps_bl * sigma * np.power(T_bl_up, 3)
        d_emitted_bl_down_dT = 4.0 * eps_bl * sigma * np.power(T_bl_down, 3)
        boundary_diag = (
            -(d_emitted_bl_up_dT + d_emitted_bl_down_dT)
        ) / config.boundary_layer_heat_capacity

        diag = np.stack([surface_diag, boundary_diag, atmosphere_diag])

        cross = np.zeros((3, 3) + surface.shape, dtype=float)

        # Surface receives downward BL emission
        cross[0, 1] = d_emitted_bl_down_dT / heat_capacity_field
        # BL absorbs surface emission
        cross[1, 0] = 4.0 * eps_bl * config.emissivity_surface * sigma * np.power(surface, 3) / config.boundary_layer_heat_capacity

        # Boundary absorbs eps_bl of downwelling LW from atmosphere
        cross[1, 2] = 4.0 * eps_bl * eps_atm * sigma * np.power(T_atm_down, 3) / config.boundary_layer_heat_capacity

        # Atmosphere absorbs eps_atm of upwelling LW from boundary
        cross[2, 1] = 4.0 * eps_atm * eps_bl * sigma * np.power(T_bl_up, 3) / config.atmosphere_heat_capacity

        # Surface receives the transmitted fraction of downwelling atmosphere LW
        cross[0, 2] = 4.0 * (1.0 - eps_bl) * eps_atm * sigma * np.power(T_atm_down, 3) / heat_capacity_field

        # Atmosphere absorbs eps_atm of the surface LW transmitted through boundary
        cross[2, 0] = 4.0 * eps_atm * (1.0 - eps_bl) * config.emissivity_surface * sigma * np.power(surface, 3) / config.atmosphere_heat_capacity

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

    Uses a two-pass approach:
    1. First pass: compute cloud cover from RH based on simple temperature estimate
    2. Second pass: recompute with updated cloud cover from first-pass temperatures
    """
    sigma = config.stefan_boltzmann
    insolation = monthly_insolation.mean(axis=0)

    if not config.include_atmosphere:
        absorbed = insolation * (1.0 - albedo_field)
        surface = np.power(absorbed / (config.emissivity_surface * sigma), 0.25)
        return _with_floor(surface[np.newaxis, :, :], config.temperature_floor)

    def compute_cloud_properties(temp_surface: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute cloud properties from surface temperature via RH and pressure."""
        if lat2d is not None and lon2d is not None:
            # Compute ITCZ once from temperature
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
        else:
            dummy_temp = np.zeros((2,) + albedo_field.shape, dtype=float)
            coverage = compute_cloud_cover(temperature=dummy_temp, land_mask=land_mask)
            albedo = np.array(0.35)
            top_height = np.array(config.cloud_top_delta_z_m)
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
