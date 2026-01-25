"""Snow albedo parameterisation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    # Ice sheet vs seasonal snow discrimination
    # Based on actual surface temperature: very cold = permanent ice, warmer = seasonal snow
    # Smooth hermite blend between these thresholds
    ice_sheet_temp_c: float = -35.0  # Below this: 100% ice sheet albedo
    seasonal_snow_temp_c: float = -15.0  # Above this: 100% seasonal snow albedo

    latent_melt_center_K: float = 273.15
    latent_melt_halfwidth_K: float = 2.0
    latent_energy_J_per_m2: float = 3.34e7

    # Sea ice parameters
    sea_ice_enabled: bool = True
    sea_ice_albedo: float = 0.60  # Multi-year ice with some melt ponds
    sea_ice_max_fraction: float = 0.70  # Seasonal average (winter max ~85%, summer min ~35%)
    sea_ice_freeze_c: float = SEAWATER_FREEZE_C  # Start freezing at -1.8°C
    sea_ice_full_c: float = -8.0  # Full ice formation by -8°C
    # Heat capacity reduction factor when fully ice-covered
    # Real ice is ~1700x lower, but we use gentler factor for stability
    sea_ice_heat_capacity_factor: float = 0.08  # 8% of ocean = ~12x reduction

    # Vegetation and soil moisture albedo effects
    vegetation_albedo_enabled: bool = True
    # Bare soil albedo depends on moisture: dry soil is brighter
    dry_soil_albedo: float = 0.35  # Dry sand/desert
    wet_soil_albedo: float = 0.20  # Wet soil (darker)
    # Vegetation albedo (forests, grasslands)
    vegetation_albedo: float = 0.18  # Typical vegetated surface

    # Vegetation fraction from annual precipitation
    # Desert: < 250 mm/year -> veg_frac ~ 0
    # Grassland: 250-750 mm/year -> veg_frac ~ 0.3-0.6
    # Forest: > 1000 mm/year -> veg_frac ~ 0.8-0.9
    veg_precip_min_mm_year: float = 200.0  # Below this: bare desert
    veg_precip_max_mm_year: float = 1200.0  # Above this: full vegetation

    # Growing season temperature threshold
    # Vegetation needs warmth to grow - months below this don't contribute
    # 5°C is a common threshold for temperate vegetation
    veg_growing_threshold_c: float = 5.0
    # Minimum growing season fraction for any vegetation
    # Below this (e.g., < 2 months), essentially no vegetation can establish
    veg_min_growing_season: float = 0.15  # ~2 months


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
        """Compute vegetation fraction from precipitation and growing season.

        Vegetation coverage depends on:
        1. Annual precipitation (water availability)
        2. Growing season length (months with T > threshold)

        Parameters
        ----------
        annual_precip_mm_year : np.ndarray
            Annual precipitation in mm/year.
        monthly_temperatures_c : np.ndarray | None
            Monthly surface temperatures in Celsius, shape (12, lat, lon).
            If None, growing season factor is not applied.

        Returns
        -------
        np.ndarray
            Vegetation fraction (0-1), only meaningful for land cells.
        """
        if not self.config.vegetation_albedo_enabled:
            return np.zeros_like(annual_precip_mm_year)

        p_min = self.config.veg_precip_min_mm_year
        p_max = self.config.veg_precip_max_mm_year

        # Normalize precipitation to [0, 1] range
        denom = p_max - p_min
        if abs(denom) < 1e-6:
            u = np.where(annual_precip_mm_year > p_min, 1.0, 0.0)
        else:
            u = (annual_precip_mm_year - p_min) / denom
        u_clamped = np.clip(u, 0.0, 1.0)

        # Smooth hermite interpolation for natural transition
        veg_frac_precip = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)

        # Apply growing season factor if temperatures provided
        if monthly_temperatures_c is not None:
            # Count fraction of year with T > threshold
            threshold = self.config.veg_growing_threshold_c
            months_above = np.sum(monthly_temperatures_c > threshold, axis=0)
            growing_season_frac = months_above / 12.0

            # Below minimum growing season, no vegetation can establish
            min_gs = self.config.veg_min_growing_season
            # Scale from 0 at min_gs to 1 at full year
            # Using smooth hermite for gradual transition
            denom = 1.0 - min_gs
            if abs(denom) < 1e-6:
                # Full year required - only year-round warm areas qualify
                gs_factor = np.where(growing_season_frac >= 1.0 - 1e-6, 1.0, 0.0)
            else:
                gs_scaled = (growing_season_frac - min_gs) / denom
                gs_scaled = np.clip(gs_scaled, 0.0, 1.0)
                gs_factor = gs_scaled * gs_scaled * (3.0 - 2.0 * gs_scaled)

            veg_frac = veg_frac_precip * gs_factor
        else:
            veg_frac = veg_frac_precip

        # Cap at 95% - even rainforests have some bare ground
        veg_frac = np.clip(veg_frac, 0.0, 0.95)

        # Only apply to land
        veg_frac = np.where(self.land_mask, veg_frac, 0.0)

        return veg_frac

    def compute_bare_soil_albedo(self, soil_moisture: np.ndarray) -> np.ndarray:
        """Compute bare soil albedo based on soil moisture.

        Wet soil is darker than dry soil due to:
        - Water filling pore spaces reduces light scattering
        - Typical change: ~0.10-0.15 albedo reduction when wet

        Parameters
        ----------
        soil_moisture : np.ndarray
            Soil moisture fraction (0-1).

        Returns
        -------
        np.ndarray
            Bare soil albedo.
        """
        dry_albedo = self.config.dry_soil_albedo
        wet_albedo = self.config.wet_soil_albedo

        # Linear interpolation between dry and wet
        bare_soil_albedo = dry_albedo - (dry_albedo - wet_albedo) * soil_moisture

        return bare_soil_albedo

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

            # Bare soil albedo depends on moisture
            bare_soil_albedo = self.compute_bare_soil_albedo(soil_moisture)

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
        # 2. Compute snow fraction from temperature
        # =====================================================================
        denom = self.config.melt_temperature_c - self.config.freeze_temperature_c
        u = (self.config.melt_temperature_c - temperatures_c) / denom
        u_clamped = np.clip(u, 0.0, 1.0)
        snow_fraction = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)
        snow_fraction = np.clip(snow_fraction, 0.0, 1.0)
        land_snow_fraction = np.where(self.land_mask, snow_fraction, 0.0)

        # =====================================================================
        # 3. Ice sheet vs seasonal snow discrimination
        # =====================================================================
        if abs(self.config.ice_sheet_albedo - self.config.snow_albedo) > 1e-6:
            ice_sheet_temp = self.config.ice_sheet_temp_c
            seasonal_snow_temp = self.config.seasonal_snow_temp_c
            temp_range = seasonal_snow_temp - ice_sheet_temp
            if abs(temp_range) < 1e-6:
                ice_sheet_frac = np.zeros_like(temperatures_c)
            else:
                u_ice = (seasonal_snow_temp - temperatures_c) / temp_range
                u_ice_clamped = np.clip(u_ice, 0.0, 1.0)
                ice_sheet_frac = u_ice_clamped * u_ice_clamped * (3.0 - 2.0 * u_ice_clamped)

            effective_snow_albedo = (
                self.config.snow_albedo
                + (self.config.ice_sheet_albedo - self.config.snow_albedo) * ice_sheet_frac
            )
        else:
            effective_snow_albedo = self.config.snow_albedo

        # =====================================================================
        # 4. Sea ice fraction
        # =====================================================================
        sea_ice_fraction = self.compute_sea_ice_fraction(temperatures_c)

        # =====================================================================
        # 5. Combine all effects
        # =====================================================================
        # Start with snow-free (vegetation + soil moisture adjusted) albedo
        adjusted = snow_free_albedo.copy()

        # Apply snow albedo to land
        adjusted = adjusted + (effective_snow_albedo - adjusted) * land_snow_fraction

        # Apply sea ice albedo to ocean
        adjusted = adjusted + (self.config.sea_ice_albedo - adjusted) * sea_ice_fraction

        return adjusted

    def effective_heat_capacity_surface(
        self,
        T_surface: np.ndarray,
        *,
        land_mask: np.ndarray,
        base_C_land: float,
        base_C_ocean: float,
    ) -> np.ndarray:
        """Return the latent-heat-adjusted surface heat capacity field.

        Includes:
        - Latent heat effects near freezing for land
        - Reduced heat capacity for sea ice covered ocean
        """
        ceff = np.where(land_mask, base_C_land, base_C_ocean).astype(float)

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
