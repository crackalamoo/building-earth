"""Snow and surface albedo parameterisation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.data.landmask import compute_ocean_albedo_direct, OCEAN_ALBEDO_DIFFUSE
from climate_sim.physics.solar import solar_declination, monthly_midpoint_days

ARCTIC_CIRCLE_LATITUDE_DEG = 66.5

# Seawater freezes at about -1.8°C due to salinity
SEAWATER_FREEZE_C = -1.8

@dataclass(frozen=True)
class SnowAlbedoConfig:
    """Configuration for the diagnostic snow albedo scheme."""

    enabled: bool = True
    latent_heat_enabled: bool = True
    snow_albedo: float = 0.45  # Seasonal snow at low elevations
    ice_sheet_albedo: float = 0.80  # Permanent ice sheets (Antarctica, Greenland)
    freeze_temperature_c: float = -5.0
    melt_temperature_c: float = 1.0  # Snow melts above 1°C

    latent_melt_center_K: float = 273.15
    latent_melt_halfwidth_K: float = 2.0
    latent_energy_J_per_m2: float = 3.34e7

    # Sea ice parameters
    sea_ice_enabled: bool = True
    sea_ice_albedo: float = 0.60  # Multi-year ice with some melt ponds (base, overhead sun)

    # Zenith-angle correction for snow/ice albedo
    # At grazing angles, snow/ice reflects more due to increased specular reflection
    # Formula: alpha = base + zenith_correction * (1 - mu), where mu = cos(zenith)
    # Based on Briegleb & Ramanathan (1982), Warren (1982)
    snow_ice_zenith_correction: float = 0.10  # Adds up to 0.10 at grazing incidence
    sea_ice_max_fraction: float = 0.70  # Seasonal average (winter max ~85%, summer min ~35%)
    sea_ice_freeze_c: float = SEAWATER_FREEZE_C  # Start freezing at -1.8°C
    sea_ice_full_c: float = -8.0  # Full ice formation by -8°C
    # Heat capacity reduction factor when fully ice-covered
    # Real ice is ~1700x lower, but we use gentler factor for stability
    sea_ice_heat_capacity_factor: float = 0.08  # 8% of ocean = ~12x reduction

    # Vegetation and soil moisture albedo effects
    vegetation_albedo_enabled: bool = True
    # Bare soil albedo depends on soil type (mineralogy) and moisture
    # Soil type: sand deserts (hyperarid) are bright; clay/rock/organic soils are darker
    desert_soil_albedo: float = 0.30   # Global desert average (Sahara 0.35, Gobi 0.22, etc.)
    normal_soil_albedo: float = 0.22   # Clay, rock, laterite, loam
    soil_type_precip_threshold: float = 150.0  # mm/yr: below = sandy desert
    # Moisture darkening: wet soil ~0.05 darker than dry (pore water reduces scattering)
    soil_moisture_darkening: float = 0.05
    # Vegetation albedo (forests, grasslands)
    vegetation_albedo: float = 0.18  # Typical vegetated surface

    # Vegetation fraction = ground cover (bare soil vs any plant cover)
    # Low threshold: 400-500 mm/yr gives full grass/shrub cover
    veg_precip_min_mm_year: float = 50.0    # Below this: hyperarid, ~0% cover
    veg_precip_max_mm_year: float = 1000.0  # Above this: full ground cover

    # Growing season: caps max achievable ground cover when short
    # Only matters below ~5 warm months; above that, precip is sole driver
    veg_growing_threshold_c: float = 5.0
    veg_growing_season_full_months: float = 5.0  # months above threshold for uncapped cover
    veg_tundra_floor: float = 0.0  # 0 warm months = no vegetation


class AlbedoModel:
    def __init__(
        self,
        lat2d: np.ndarray,
        lon2d: np.ndarray,
        land_mask: np.ndarray,
        config: SnowAlbedoConfig | None = None,
    ) -> None:
        """Initialize the snow albedo model.

        Args:
            lat2d: Latitude grid in degrees.
            lon2d: Longitude grid in degrees.
            land_mask: Boolean array of shape (lat, lon) indicating land grid cells.
            config: Snow albedo configuration.
        """
        self.lat2d = lat2d
        self.lon2d = lon2d
        self.land_mask = land_mask
        if config is not None:
            self.config = config
        else:
            self.config = SnowAlbedoConfig()

    def guess_albedo_field(self) -> np.ndarray:
        """Initial albedo guess with high values at polar latitudes.

        Polar regions (|lat| >= 66.5°) get ice sheet albedo to ensure
        the model starts in the cold/icy state for Antarctica and Arctic.
        """
        albedo = 0.3 * np.ones_like(self.lat2d)
        in_polar = np.abs(self.lat2d) >= ARCTIC_CIRCLE_LATITUDE_DEG
        albedo = np.where(in_polar & self.land_mask, self.config.ice_sheet_albedo, albedo)
        return albedo

    def compute_vegetation_fraction(
        self,
        annual_precip_mm_year: np.ndarray,
        monthly_temperatures_c: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute vegetation ground cover fraction.

        Represents visible ground cover (bare soil vs any plant material),
        not biomass or productivity. Precipitation drives ground cover with
        a low saturation threshold (~500 mm/yr). Growing season acts as a
        cap on maximum achievable cover when very short (<5 warm months).

        Parameters
        ----------
        annual_precip_mm_year : np.ndarray
            Annual precipitation in mm/year.
        monthly_temperatures_c : np.ndarray | None
            Monthly surface temperatures in Celsius, shape (12, lat, lon).
            Used to compute growing season cap.

        Returns
        -------
        np.ndarray
            Vegetation fraction (0-1), only meaningful for land cells.
        """
        if not self.config.vegetation_albedo_enabled:
            return np.zeros_like(annual_precip_mm_year)

        p_min = self.config.veg_precip_min_mm_year
        p_max = self.config.veg_precip_max_mm_year

        # Precipitation → ground cover: power-law ramp 50-1000 mm/yr
        # Exponent 0.6 gives concave shape matching observed semi-arid vegetation:
        # 200mm→25%, 400mm→55%, 650mm→76%, 800mm→87% (RMSE 0.048 vs observations)
        denom = p_max - p_min
        if abs(denom) < 1e-6:
            u = np.where(annual_precip_mm_year > p_min, 1.0, 0.0)
        else:
            u = (annual_precip_mm_year - p_min) / denom
        u_clamped = np.clip(u, 0.0, 1.0)
        veg_frac = np.power(u_clamped, 0.6)

        # Growing season cap: short seasons limit max achievable ground cover
        if monthly_temperatures_c is not None:
            warm_months = np.sum(
                monthly_temperatures_c > self.config.veg_growing_threshold_c, axis=0,
            )
            # Hermite ramp from tundra_floor (0 months) to 1.0 (full_months)
            # Hermite keeps cap low for 1-2 months, rises steeply toward 5
            full = self.config.veg_growing_season_full_months
            floor = self.config.veg_tundra_floor
            gs_u = np.clip(warm_months / full, 0.0, 1.0)
            gs_cap = floor + (1.0 - floor) * gs_u * gs_u * (3.0 - 2.0 * gs_u)
            veg_frac = np.minimum(veg_frac, gs_cap)

        # Cap at 95% - even rainforests have some bare ground
        veg_frac = np.clip(veg_frac, 0.0, 0.95)

        # Only apply to land
        veg_frac = np.where(self.land_mask, veg_frac, 0.0)

        return veg_frac

    def compute_bare_soil_albedo(
        self, soil_moisture: np.ndarray, annual_precip_mm_year: np.ndarray,
    ) -> np.ndarray:
        """Compute bare soil albedo from soil type (mineralogy) and moisture.

        Two independent effects:
        1. Soil type: hyperarid regions have bright quartz sand (~0.35),
           wetter regions have darker clay/organic soils (~0.22).
           Threshold at ~150 mm/yr (smooth transition).
        2. Moisture darkening: wet soil ~0.05 darker (water fills pore spaces).
        """
        # Soil type: smooth transition from desert sand to normal soil
        thresh = self.config.soil_type_precip_threshold
        # Hermite ramp over 0 → 2*threshold
        t = np.clip(annual_precip_mm_year / (2.0 * thresh), 0.0, 1.0)
        soil_frac = t * t * (3.0 - 2.0 * t)  # 0=desert, 1=normal
        dry_albedo = (
            (1.0 - soil_frac) * self.config.desert_soil_albedo
            + soil_frac * self.config.normal_soil_albedo
        )

        # Moisture darkening
        return dry_albedo - self.config.soil_moisture_darkening * soil_moisture

    def compute_sea_ice_fraction(self, temperatures_c: np.ndarray) -> np.ndarray:
        """Compute sea ice fraction based on temperature.

        Uses smooth hermite interpolation between freeze and full-ice temperatures.
        Returns fraction in [0, max_fraction] for ocean cells, 0 for land.
        """
        if not self.config.sea_ice_enabled:
            return np.zeros_like(temperatures_c)

        # Smooth transition from freeze temp to full-ice temp
        denom = self.config.sea_ice_freeze_c - self.config.sea_ice_full_c
        if abs(denom) < 1e-6:
            # Avoid division by zero
            u = np.where(temperatures_c < self.config.sea_ice_freeze_c, 1.0, 0.0)
        else:
            u = (self.config.sea_ice_freeze_c - temperatures_c) / denom
        u_clamped = np.clip(u, 0.0, 1.0)

        # Smooth hermite interpolation
        ice_fraction = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)

        # Apply max fraction and mask to ocean only
        ice_fraction = ice_fraction * self.config.sea_ice_max_fraction
        ice_fraction = np.where(self.land_mask, 0.0, ice_fraction)

        return ice_fraction

    def apply_snow_albedo(
        self,
        base_albedo: np.ndarray,
        monthly_temperatures_K: np.ndarray,
        soil_moisture: np.ndarray | None = None,
        vegetation_fraction: np.ndarray | None = None,
        annual_precip_mm_year: np.ndarray | None = None,
        effective_mu: np.ndarray | None = None,
        cloud_fraction: np.ndarray | None = None,
        ocean_albedo: np.ndarray | None = None,
        ice_sheet_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an albedo field with snow, sea ice, and vegetation adjustments applied."""

        if not self.config.enabled:
            return base_albedo

        if self.config.freeze_temperature_c >= self.config.melt_temperature_c:
            raise ValueError("Snow melt temperature must exceed freeze temperature")

        if monthly_temperatures_K.ndim == 2 or monthly_temperatures_K.ndim == 3:
            surface_temperatures = monthly_temperatures_K
        elif monthly_temperatures_K.ndim == 4:
            surface_temperatures = monthly_temperatures_K[:, 0]
        else:
            raise ValueError(
                "Expected 2-D (lat, lon), 3-D (month, lat, lon), or 4-D (month, layer, lat, lon) "
                "temperature arrays for snow albedo calculation."
            )

        temperatures_c = surface_temperatures - 273.15

        # =====================================================================
        # 1. Compute snow-free land surface albedo from vegetation and soil moisture
        # =====================================================================
        if self.config.vegetation_albedo_enabled and vegetation_fraction is not None:
            # Use default soil moisture if not provided
            if soil_moisture is None:
                soil_moisture = np.full_like(base_albedo, 0.3)

            # Bare soil albedo from soil type + moisture
            if annual_precip_mm_year is None:
                annual_precip_mm_year = np.full_like(base_albedo, 500.0)
            bare_soil_albedo = self.compute_bare_soil_albedo(soil_moisture, annual_precip_mm_year)

            # Vegetation has fixed low albedo
            veg_albedo = self.config.vegetation_albedo

            # Blend bare soil and vegetation
            land_base_albedo = (
                (1.0 - vegetation_fraction) * bare_soil_albedo
                + vegetation_fraction * veg_albedo
            )

            # Apply only to land cells, keep base_albedo for ocean
            snow_free_albedo = np.where(self.land_mask, land_base_albedo, base_albedo)
        else:
            # No vegetation tracking - use original base_albedo
            snow_free_albedo = base_albedo

        # =====================================================================
        # 1b. Apply ocean albedo (properly flux-weighted)
        # =====================================================================
        if ocean_albedo is not None:
            # Use precomputed flux-weighted ocean albedo
            # Blend with diffuse albedo under clouds if cloud_fraction provided
            if cloud_fraction is not None:
                effective_ocean_albedo = (
                    cloud_fraction * OCEAN_ALBEDO_DIFFUSE
                    + (1.0 - cloud_fraction) * ocean_albedo
                )
            else:
                effective_ocean_albedo = ocean_albedo

            # Apply to ocean cells only (where not land)
            snow_free_albedo = np.where(self.land_mask, snow_free_albedo, effective_ocean_albedo)

        # =====================================================================
        # 2. Compute snow fraction from temperature
        # =====================================================================
        denom = self.config.melt_temperature_c - self.config.freeze_temperature_c
        u = (self.config.melt_temperature_c - temperatures_c) / denom
        u_clamped = np.clip(u, 0.0, 1.0)
        snow_fraction = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)
        snow_fraction = np.clip(snow_fraction, 0.0, 1.0)
        land_snow_fraction = np.where(self.land_mask, snow_fraction, 0.0)

        # =====================================================================
        # 3. Snow albedo: seasonal snow everywhere, ice_sheet_albedo forced
        #    for geographic ice sheet cells (Antarctica, Greenland)
        # =====================================================================
        base_snow_albedo = self.config.snow_albedo

        # Apply zenith-angle correction to snow/ice albedo
        # At low sun angles, specular reflection increases albedo
        # Formula: alpha = base + correction * (1 - mu)
        if effective_mu is not None and self.config.snow_ice_zenith_correction > 0:
            zenith_boost = self.config.snow_ice_zenith_correction * (1.0 - effective_mu)
            effective_snow_albedo = np.clip(base_snow_albedo + zenith_boost, 0.0, 0.95)
        else:
            effective_snow_albedo = base_snow_albedo

        # =====================================================================
        # 4. Sea ice fraction and albedo
        # =====================================================================
        sea_ice_fraction = self.compute_sea_ice_fraction(temperatures_c)

        # Apply zenith-angle correction to sea ice albedo
        if effective_mu is not None and self.config.snow_ice_zenith_correction > 0:
            zenith_boost = self.config.snow_ice_zenith_correction * (1.0 - effective_mu)
            effective_sea_ice_albedo = np.clip(
                self.config.sea_ice_albedo + zenith_boost, 0.0, 0.95
            )
        else:
            effective_sea_ice_albedo = self.config.sea_ice_albedo

        # =====================================================================
        # 5. Combine all effects
        # =====================================================================
        # Start with snow-free (vegetation + soil moisture adjusted) albedo
        adjusted = snow_free_albedo.copy()

        # Apply snow albedo to land
        adjusted = adjusted + (effective_snow_albedo - adjusted) * land_snow_fraction

        # Force ice sheet albedo for geographic ice sheet cells (bypasses snow fraction)
        if ice_sheet_mask is not None:
            ice_sheet_albedo = self.config.ice_sheet_albedo
            if effective_mu is not None and self.config.snow_ice_zenith_correction > 0:
                zenith_boost = self.config.snow_ice_zenith_correction * (1.0 - effective_mu)
                ice_sheet_albedo = np.clip(ice_sheet_albedo + zenith_boost, 0.0, 0.95)
            adjusted = np.where(ice_sheet_mask, ice_sheet_albedo, adjusted)

        # Apply sea ice albedo to ocean
        adjusted = adjusted + (effective_sea_ice_albedo - adjusted) * sea_ice_fraction

        return adjusted

    def effective_heat_capacity_surface(
        self,
        T_surface: np.ndarray,
        *,
        land_mask: np.ndarray,
        base_C_land: float,
        base_C_ocean: float,
        ice_sheet_mask: np.ndarray | None = None,
        ice_sheet_heat_capacity_multiplier: float = 100.0,
    ) -> np.ndarray:
        """Return the latent-heat-adjusted surface heat capacity field.

        Includes:
        - Latent heat effects near freezing for land
        - Reduced heat capacity for sea ice covered ocean
        - Massive heat capacity for ice sheet cells near/above freezing
        """
        ceff = np.where(land_mask, base_C_land, base_C_ocean).astype(float)

        # Ice sheet cells: massive effective heat capacity near/above freezing
        # This physically represents the latent heat of kilometers of ice,
        # clamping surface temperature at ~0°C
        if ice_sheet_mask is not None and np.any(ice_sheet_mask):
            T_c = T_surface - 273.15
            # Ramp multiplier from 1x at -10°C to full at -2°C and above
            ice_ramp = np.clip((T_c + 10.0) / 8.0, 0.0, 1.0)
            ice_multiplier = 1.0 + (ice_sheet_heat_capacity_multiplier - 1.0) * ice_ramp
            ceff = np.where(ice_sheet_mask, ceff * ice_multiplier, ceff)

        # Sea ice heat capacity reduction for ocean cells
        if self.config.sea_ice_enabled:
            temperatures_c = T_surface - 273.15
            ice_fraction = self.compute_sea_ice_fraction(temperatures_c)

            # Blend between full ocean and reduced ice heat capacity
            # ice_factor interpolates from 1.0 (open ocean) to sea_ice_heat_capacity_factor (ice)
            ocean_mask = ~land_mask
            ice_factor = 1.0 - ice_fraction * (1.0 - self.config.sea_ice_heat_capacity_factor)

            # Apply only to ocean cells
            ceff = np.where(ocean_mask, ceff * ice_factor, ceff)

        # Latent heat effects for land near freezing (existing logic)
        if not self.config.latent_heat_enabled:
            return ceff

        melt_halfwidth = self.config.latent_melt_halfwidth_K
        latent_energy = self.config.latent_energy_J_per_m2
        if melt_halfwidth <= 0.0 or latent_energy <= 0.0:
            return ceff

        melt_center = self.config.latent_melt_center_K
        band_lo = melt_center - melt_halfwidth
        band_hi = melt_center + melt_halfwidth
        in_band = (T_surface >= band_lo) & (T_surface <= band_hi) & land_mask
        if not np.any(in_band):
            return ceff

        added_capacity = latent_energy / (2.0 * melt_halfwidth)
        ceff = ceff.copy()
        ceff[in_band] += added_capacity
        return ceff


def _compute_flux_weighted_ocean_albedo(
    lat_rad: np.ndarray,
    declination_rad: np.ndarray,
    n_hour_angles: int = 48,
) -> np.ndarray:
    """Compute flux-weighted daily mean ocean albedo by integrating over the day.

    The CESM ocean albedo formula is nonlinear in cos(zenith), so we cannot
    simply apply it to the mean cos(zenith). Instead, we integrate:

        albedo_mean = ∫ albedo(μ(h)) × μ(h) dh / ∫ μ(h) dh

    where μ(h) = cos(zenith) at hour angle h, and the integrals are over
    the sunlit portion of the day.
    """
    nlat = lat_rad.shape[0]
    nmonth = declination_rad.shape[0]

    # Precompute trig values
    sin_lat = np.sin(lat_rad)[:, None]  # (nlat, 1)
    cos_lat = np.cos(lat_rad)[:, None]
    sin_dec = np.sin(declination_rad)[None, :]  # (1, nmonth)
    cos_dec = np.cos(declination_rad)[None, :]

    # Compute hour angle at sunrise/sunset for each lat/month
    tan_lat = np.tan(lat_rad)[:, None]
    tan_dec = np.tan(declination_rad)[None, :]
    cos_H0 = np.clip(-tan_lat * tan_dec, -1.0, 1.0)
    H0 = np.arccos(cos_H0)  # (nlat, nmonth)

    # Handle polar night (H0 = 0) and polar day (H0 = π)
    polar_night = cos_H0 >= 1.0
    polar_day = cos_H0 <= -1.0
    H0[polar_night] = 0.0
    H0[polar_day] = np.pi

    # Integrate over hour angles
    result = np.zeros((nlat, nmonth))

    for i_lat in range(nlat):
        for i_month in range(nmonth):
            h0 = H0[i_lat, i_month]
            if h0 < 1e-6:
                # Polar night: use diffuse albedo
                result[i_lat, i_month] = OCEAN_ALBEDO_DIFFUSE
                continue

            # Integration points from -H0 to +H0
            h = np.linspace(-h0, h0, n_hour_angles)

            # cos(zenith) at each hour angle
            mu = (
                sin_lat[i_lat, 0] * sin_dec[0, i_month]
                + cos_lat[i_lat, 0] * cos_dec[0, i_month] * np.cos(h)
            )
            mu = np.maximum(mu, 0.001)

            # Albedo at each point
            albedo = compute_ocean_albedo_direct(mu)

            # Flux-weighted mean: ∫(α × μ)dh / ∫μ dh
            flux_weighted_albedo = np.sum(albedo * mu) / np.sum(mu)
            result[i_lat, i_month] = flux_weighted_albedo

    return result


def compute_monthly_flux_weighted_ocean_albedo(lat2d: np.ndarray) -> np.ndarray:
    """Compute monthly flux-weighted mean ocean albedo for each latitude.

    This properly accounts for the nonlinearity of the CESM ocean albedo
    formula by integrating over the diurnal cycle rather than applying
    the formula to the mean cos(zenith).
    """
    lat_rad = np.deg2rad(lat2d[:, 0])
    declinations = solar_declination(monthly_midpoint_days())
    albedo = _compute_flux_weighted_ocean_albedo(lat_rad, declinations)
    return albedo.T  # (nmonth, nlat)
