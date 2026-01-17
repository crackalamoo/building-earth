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
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    BOUNDARY_LAYER_EMISSIVITY,
    SHORTWAVE_ABSORPTANCE_ATMOSPHERE,
    STANDARD_LAPSE_RATE_K_PER_M,
    ATMOSPHERE_LAYER_HEIGHT_M,
    R_EARTH_METERS,
)


@dataclass(frozen=True)
class RadiationConfig:
    """Container for radiative transfer parameters."""

    stefan_boltzmann: float = 5.670374419e-8  # W m-2 K-4
    emissivity_surface: float = 1.0
    emissivity_atmosphere: float = 0.78
    include_atmosphere: bool = True
    atmosphere_heat_capacity: float = ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K
    temperature_floor: float = 10.0  # K
    boundary_layer_heat_capacity: float = BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K
    boundary_layer_emissivity: float = BOUNDARY_LAYER_EMISSIVITY
    shortwave_absorptance_atmosphere: float = SHORTWAVE_ABSORPTANCE_ATMOSPHERE
    cloud_top_delta_z_m: float = ATMOSPHERE_LAYER_HEIGHT_M / 2.0


def _with_floor(values: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(values, floor)


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

    # Longwave: atmospheric emissivity
    sigma = config.stefan_boltzmann
    eps_sfc = config.emissivity_surface
    eps_clear = config.emissivity_atmosphere
    # Cloud contribution: sqrt saturation to account for overlap/gaps
    # Individual clouds ≈ blackbody, but grid-cell doesn't fill perfectly
    eps_cloud = (1.0 - eps_clear) * np.sqrt(cloud_coverage)
    eps_atm = eps_clear + eps_cloud

    emitted_surface = eps_sfc * sigma * np.power(surface, 4)
    emitted_atmosphere = eps_atm * sigma * np.power(atmosphere, 4)

    # Cloud-top correction for TOA longwave emission
    cloud_top_K = _with_floor(
        atmosphere - STANDARD_LAPSE_RATE_K_PER_M * cloud_top_height,
        floor,
    )
    emitted_toa = (
        eps_clear * sigma * np.power(atmosphere, 4)
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
    downward_longwave = emitted_atmosphere
    absorbed_from_surface = eps_atm * emitted_surface

    if nlayers == 2:
        # Two-layer system: surface + atmosphere (no boundary layer)
        surface_tendency = (
            absorbed_shortwave_sfc + downward_longwave - emitted_surface
        ) / heat_capacity_field

        atmosphere_tendency = (
            absorbed_shortwave_atm + absorbed_from_surface - emitted_atmosphere - emitted_toa
        ) / config.atmosphere_heat_capacity

        return np.stack([surface_tendency, atmosphere_tendency])

    elif nlayers == 3:
        # Three-layer system: surface + boundary layer + atmosphere
        boundary = _with_floor(temperature_K[1], floor)

        eps_bl = config.boundary_layer_emissivity
        emitted_boundary = eps_bl * sigma * np.power(boundary, 4)

        # Longwave transmissivities (grey, no-scattering assumption)
        tau_bl = 1.0 - eps_bl
        tau_atm = 1.0 - eps_atm

        downward_longwave_to_surface = emitted_boundary + tau_bl * emitted_atmosphere
        surface_tendency = (
            absorbed_shortwave_sfc + downward_longwave_to_surface - emitted_surface
        ) / heat_capacity_field

        absorbed_from_surface_bl = eps_bl * emitted_surface
        absorbed_from_atm_bl = eps_bl * emitted_atmosphere
        boundary_tendency = (
            absorbed_from_surface_bl + absorbed_from_atm_bl - 2.0 * emitted_boundary
        ) / config.boundary_layer_heat_capacity

        transmitted_surface_to_atm = tau_bl * emitted_surface
        absorbed_from_surface_atm = eps_atm * transmitted_surface_to_atm
        absorbed_from_boundary_atm = eps_atm * emitted_boundary
        atmosphere_tendency = (
            absorbed_shortwave_atm + absorbed_from_surface_atm + absorbed_from_boundary_atm
            - emitted_atmosphere - emitted_toa
        ) / config.atmosphere_heat_capacity

        if log_diagnostics:
            if cell_area_m2 is None:
                raise ValueError("cell_area_m2 must be provided when log_diagnostics=True")

            reflected_by_atm = alpha_atm * insolation_W_m2
            reflected_by_surface = albedo_field * sw_down_surface

            # Total outgoing longwave radiation (OLR) at TOA
            olr_total = emitted_toa + tau_atm * emitted_boundary + tau_atm * tau_bl * emitted_surface

            print("\n=== Radiation Diagnostics (W/m²) ===")
            print("\nLayer Temperatures (K):")
            print(f"  Surface:                         {area_weighted_mean(surface, cell_area_m2):7.2f}")
            print(f"  Boundary layer:                  {area_weighted_mean(boundary, cell_area_m2):7.2f}")
            print(f"  Atmosphere:                      {area_weighted_mean(atmosphere, cell_area_m2):7.2f}")
            print("\nShortwave Fluxes:")
            print(f"Incoming solar (TOA):              {area_weighted_mean(insolation_W_m2, cell_area_m2):7.2f}")
            print(f"Reflected by atmosphere:           {area_weighted_mean(reflected_by_atm, cell_area_m2):7.2f}")
            print(f"Absorbed by atmosphere (SW):       {area_weighted_mean(absorbed_shortwave_atm, cell_area_m2):7.2f}")
            print(f"SW reaching surface:               {area_weighted_mean(sw_down_surface, cell_area_m2):7.2f}")
            print(f"Reflected by surface:              {area_weighted_mean(reflected_by_surface, cell_area_m2):7.2f}")
            print(f"Absorbed by surface (SW):          {area_weighted_mean(absorbed_shortwave_sfc, cell_area_m2):7.2f}")
            print("\nLongwave Emissions:")
            print(f"Surface emission:                  {area_weighted_mean(emitted_surface, cell_area_m2):7.2f}")
            print(f"Boundary layer emission (total):   {area_weighted_mean(2.0 * emitted_boundary, cell_area_m2):7.2f}")
            print(f"  - absorbed by surface:           {area_weighted_mean(emitted_boundary, cell_area_m2):7.2f}")
            print(f"  - absorbed by atmosphere:        {area_weighted_mean(absorbed_from_boundary_atm, cell_area_m2):7.2f}")
            print(f"Atmosphere emission (total):       {area_weighted_mean(2.0 * emitted_atmosphere, cell_area_m2):7.2f}")
            print(f"  - downward to surface:           {area_weighted_mean(tau_bl * emitted_atmosphere, cell_area_m2):7.2f}")
            print(f"  - downward to boundary:          {area_weighted_mean(emitted_atmosphere, cell_area_m2):7.2f}")
            print(f"OLR (to space):                    {area_weighted_mean(olr_total, cell_area_m2):7.2f}")
            print("\nBoundary layer absorbs from:")
            print(f"  - surface:                       {area_weighted_mean(absorbed_from_surface_bl, cell_area_m2):7.2f}")
            print(f"  - atmosphere:                    {area_weighted_mean(absorbed_from_atm_bl, cell_area_m2):7.2f}")
            print("Atmosphere absorbs from:")
            print(f"  - surface emission (absorbed):   {area_weighted_mean(absorbed_from_surface_atm, cell_area_m2):7.2f}")
            print(f"  - boundary layer emission:       {area_weighted_mean(absorbed_from_boundary_atm, weights=cell_area_m2):7.2f}")
            print("\nNet Radiation Balances:")
            print(f"Net surface radiation balance:     {area_weighted_mean(absorbed_shortwave_sfc + downward_longwave_to_surface - emitted_surface, cell_area_m2):7.2f}")
            print(f"Net boundary layer balance:        {area_weighted_mean(absorbed_from_surface_bl + absorbed_from_atm_bl - 2.0 * emitted_boundary, cell_area_m2):7.2f}")
            print(f"Net atmosphere balance:            {area_weighted_mean(absorbed_shortwave_atm + absorbed_from_surface_atm + absorbed_from_boundary_atm - emitted_atmosphere - emitted_toa, cell_area_m2):7.2f}")
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


    # Atmosphere loses LW by emitting up and down, but TOA emission is corrected
    # to include a colder cloud-top contribution.
    d_emitted_atmosphere_dT = 4.0 * eps_atm * sigma * np.power(atmosphere, 3)
    d_emitted_toa_dT = (
        4.0 * eps_clear * sigma * np.power(atmosphere, 3)
        + 4.0 * eps_cloud * sigma * np.power(cloud_top_K, 3)
    )
    atmosphere_diag = (
        -(d_emitted_atmosphere_dT + d_emitted_toa_dT)
    ) / config.atmosphere_heat_capacity

    if nlayers == 2:
        surface_coupling = (
            4. * eps_atm * sigma * np.power(atmosphere, 3)
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
        eps_bl = config.boundary_layer_emissivity

        boundary_diag = (
            -8.0 * eps_bl * sigma * np.power(boundary, 3)
        ) / config.boundary_layer_heat_capacity

        diag = np.stack([surface_diag, boundary_diag, atmosphere_diag])

        cross = np.zeros((3, 3) + surface.shape, dtype=float)

        cross[0, 1] = 4.0 * eps_bl * sigma * np.power(boundary, 3) / heat_capacity_field
        cross[1, 0] = 4.0 * eps_bl * config.emissivity_surface * sigma * np.power(surface, 3) / config.boundary_layer_heat_capacity

        # Boundary absorbs eps_bl of downwelling LW from atmosphere: eps_bl * (eps_atm * sigma T_atm^4)
        cross[1, 2] = 4.0 * eps_bl * eps_atm * sigma * np.power(atmosphere, 3) / config.boundary_layer_heat_capacity

        # Atmosphere absorbs eps_atm of upwelling LW from boundary: eps_atm * (eps_bl * sigma T_bl^4)
        cross[2, 1] = 4.0 * eps_atm * eps_bl * sigma * np.power(boundary, 3) / config.atmosphere_heat_capacity

        # Surface receives the transmitted fraction of downwelling atmosphere LW:
        # (1 - eps_bl) * (eps_atm * sigma T_atm^4)
        cross[0, 2] = 4.0 * (1.0 - eps_bl) * eps_atm * sigma * np.power(atmosphere, 3) / heat_capacity_field

        # Atmosphere absorbs eps_atm of the surface LW transmitted through boundary:
        # eps_atm * (1 - eps_bl) * (eps_sfc * sigma T_s^4)
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
