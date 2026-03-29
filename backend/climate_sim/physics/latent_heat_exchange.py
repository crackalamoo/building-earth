"""Latent heat exchange between the surface and atmosphere."""

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.atmosphere.wind import WindModel
from climate_sim.data.constants import (
    GAS_CONSTANT_J_KG_K,
    STANDARD_LAPSE_RATE_K_PER_M,
    BOUNDARY_LAYER_HEIGHT_M,
    LATENT_HEAT_VAPORIZATION_J_KG,
)


# Seawater freezes at about -1.8°C due to salinity
SEAWATER_FREEZE_C = -1.8


@dataclass(frozen=True)
class LatentHeatExchangeConfig:
    """Configuration for the latent heat exchange model."""

    enabled: bool = True
    minimum_wind_speed_m_s: float = 2.0
    # Manabe (1969) beta function: β = min(θ/θ_crit, 1)
    # Below field capacity, evapotranspiration scales linearly with soil moisture.
    # Above field capacity, evapotranspiration is at full potential rate.
    manabe_theta_crit: float = 0.75
    # No evaporation below freezing (ice-covered surface)
    freeze_threshold_c: float = SEAWATER_FREEZE_C


class LatentHeatExchangeModel:
    """Compute tendencies from latent heat exchange."""

    def __init__(
        self,
        *,
        land_mask: np.ndarray,
        surface_heat_capacity_J_m2_K: np.ndarray,
        atmosphere_heat_capacity_J_m2_K: np.ndarray | float,
        wind_model: WindModel | None = None,
        config: LatentHeatExchangeConfig | None = None,
        boundary_layer_heat_capacity_J_m2_K: np.ndarray | float | None = None,
    ) -> None:
        self._config = config or LatentHeatExchangeConfig()

        land_mask_bool = np.asarray(land_mask, dtype=bool)
        heat_capacity_surface = np.asarray(surface_heat_capacity_J_m2_K, dtype=float)
        heat_capacity_atmosphere = np.asarray(atmosphere_heat_capacity_J_m2_K, dtype=float)

        if land_mask_bool.shape != heat_capacity_surface.shape:
            raise ValueError("Surface heat capacity must match the land mask shape")
        if heat_capacity_atmosphere.shape not in ((), land_mask_bool.shape):
            raise ValueError(
                "Atmospheric heat capacity must be scalar or match the land mask shape"
            )

        if heat_capacity_atmosphere.shape == ():
            heat_capacity_atmosphere = np.full(
                land_mask_bool.shape, float(heat_capacity_atmosphere)
            )

        self._land_mask = land_mask_bool
        self._surface_heat_capacity = np.maximum(heat_capacity_surface, 1.0e-9)
        self._atmosphere_heat_capacity = np.maximum(heat_capacity_atmosphere, 1.0e-9)
        self._wind_model = wind_model

        # Boundary layer heat capacity (optional, for 3-layer system)
        if boundary_layer_heat_capacity_J_m2_K is not None:
            heat_capacity_boundary = np.asarray(boundary_layer_heat_capacity_J_m2_K, dtype=float)
            if heat_capacity_boundary.shape == ():
                heat_capacity_boundary = np.full(
                    land_mask_bool.shape, float(heat_capacity_boundary)
                )
            self._boundary_layer_heat_capacity = np.maximum(heat_capacity_boundary, 1.0e-9)
        else:
            self._boundary_layer_heat_capacity = None

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def compute_tendencies(
        self,
        surface_temperature_K: np.ndarray,
        atmosphere_temperature_K: np.ndarray,
        humidity_q: np.ndarray,
        *,
        wind_speed_reference_m_s: np.ndarray | None,
        itcz_rad: np.ndarray | None = None,
        boundary_layer_temperature_K: np.ndarray | None = None,
        precipitation_rate: np.ndarray | None = None,
        soil_moisture: np.ndarray | None = None,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Return surface and atmospheric tendencies from latent heat exchange.

        Latent heat routing:
        - Surface loses heat via evaporation: dT_sfc/dt = -E * Lv / C_sfc
        - Atmosphere gains heat via precipitation (condensation): dT_atm/dt = +P * Lv / C_atm

        If boundary_layer_temperature_K is provided, returns 3 tendencies:
        (surface, boundary_layer, atmosphere). Latent heat only couples surface
        and boundary layer for evaporation; atmosphere gains heat from precipitation.

        Also returns the evaporation rate (kg/m²/s) as the last element for
        prognostic humidity calculations.
        """

        if not self.enabled:
            zeros = np.zeros_like(surface_temperature_K, dtype=float)
            if boundary_layer_temperature_K is not None:
                return zeros, zeros, zeros, zeros  # surface, BL, atm, evap_rate
            return zeros, zeros, zeros  # surface, atm, evap_rate

        surface_temperature = np.asarray(surface_temperature_K, dtype=float)
        atmosphere_temperature = np.asarray(atmosphere_temperature_K, dtype=float)

        if surface_temperature.shape != self._surface_heat_capacity.shape:
            raise ValueError("Surface temperature must match the surface heat capacity field shape")
        if atmosphere_temperature.shape != self._surface_heat_capacity.shape:
            raise ValueError(
                "Atmosphere temperature must match the surface heat capacity field shape"
            )

        # Use advection model for atmospheric properties if available
        if self._wind_model is not None:
            pressure, rho, wind_speed_10m, ch = self._wind_model.compute_atmospheric_properties(
                surface_temperature,
                atmosphere_temperature,
                wind_speed_reference_m_s,
                itcz_rad=itcz_rad,
            )
        else:
            # Fallback: no wind, no exchange
            zeros = np.zeros_like(surface_temperature_K, dtype=float)
            if boundary_layer_temperature_K is not None:
                return zeros, zeros, zeros, zeros  # surface, BL, atm, evap_rate
            return zeros, zeros, zeros  # surface, atm, evap_rate

        wind_abs = np.maximum(np.abs(wind_speed_10m), self._config.minimum_wind_speed_m_s)

        # Magnus formula gives e_sat in hPa, pressure is in Pa - convert to hPa
        # Cap temperature to prevent overflow in exp (max realistic surface T ~ 80C)
        surface_temperature_C = np.clip(surface_temperature_K - 273.15, -100.0, 80.0)
        e_sat = 6.112 * np.exp(17.67 * surface_temperature_C / (surface_temperature_C + 243.5))
        pressure_hPa = pressure / 100.0
        # Ensure denominator doesn't go negative or zero
        denom = np.maximum(pressure_hPa - (1 - 0.622) * e_sat, 1.0)
        q_sat = (0.622 * e_sat) / denom

        humidity_q = np.minimum(humidity_q, q_sat)
        heat_flux = rho * 2.5e6 * ch * wind_abs * (q_sat - humidity_q)
        # Manabe (1969) beta: β = min(θ/θ_crit, 1)
        theta_crit = self._config.manabe_theta_crit
        if soil_moisture is not None:
            land_factor = np.minimum(soil_moisture / theta_crit, 1.0)
        else:
            # Fallback when soil moisture not yet initialized: use RH as proxy
            rh = humidity_q / np.maximum(q_sat, 1e-10)
            land_factor = np.minimum(rh / theta_crit, 1.0)
        heat_flux = np.where(self._land_mask, heat_flux * land_factor, heat_flux)

        # No evaporation below freezing (ice-covered surface can't evaporate liquid water)
        frozen = surface_temperature_C < self._config.freeze_threshold_c
        heat_flux = np.where(frozen, 0.0, heat_flux)

        # Evaporation rate (kg/m²/s) for prognostic humidity
        evaporation_rate = heat_flux / LATENT_HEAT_VAPORIZATION_J_KG

        # Surface always loses heat via evaporation
        surface_tendency = -heat_flux / self._surface_heat_capacity

        # Atmosphere gains heat via PRECIPITATION (condensation), not evaporation
        # This is the key physics fix: latent heat is released where it RAINS
        if precipitation_rate is not None:
            precip_heating = precipitation_rate * LATENT_HEAT_VAPORIZATION_J_KG  # W/m²
            atmosphere_tendency = precip_heating / self._atmosphere_heat_capacity
        else:
            # Fallback: assume local E ≈ P (old behavior for backward compatibility)
            atmosphere_tendency = heat_flux / self._atmosphere_heat_capacity

        # Three-layer system: BL gets fraction of precipitation heating
        # This compensates for cooling from vertical ascent at the ITCZ
        if boundary_layer_temperature_K is not None:
            # Over ocean, shallow cumulus and stratocumulus release ~47% of
            # latent heat in the 1-4 km range (Nelson 2018, TRMM).  Our BL
            # (0-1 km) captures the lowest portion: drizzle re-evaporation,
            # sub-cloud evaporative cooling reversal, and shallow warm rain.
            # Over land, deep convection dominates → most heating aloft.
            BL_LATENT_FRACTION_OCEAN = 0.20
            BL_LATENT_FRACTION_LAND = 0.05
            bl_frac = np.where(self._land_mask, BL_LATENT_FRACTION_LAND, BL_LATENT_FRACTION_OCEAN)
            if precipitation_rate is not None:
                precip_heating = precipitation_rate * LATENT_HEAT_VAPORIZATION_J_KG
                boundary_tendency = bl_frac * precip_heating / self._boundary_layer_heat_capacity
                atmosphere_tendency = (
                    (1 - bl_frac) * precip_heating / self._atmosphere_heat_capacity
                )
            else:
                boundary_tendency = np.zeros_like(surface_tendency)
            return surface_tendency, boundary_tendency, atmosphere_tendency, evaporation_rate

        # Two-layer system
        return surface_tendency, atmosphere_tendency, evaporation_rate

    def compute_jacobian(
        self,
        surface_temperature_K: np.ndarray,
        atmosphere_temperature_K: np.ndarray,
        humidity_q: np.ndarray,
        *,
        wind_speed_reference_m_s: np.ndarray | None,
        itcz_rad: np.ndarray | None = None,
        boundary_layer_temperature_K: np.ndarray | None = None,
        precipitation_rate: np.ndarray | None = None,
        soil_moisture: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Jacobian (diagonal and cross-coupling) for latent heat exchange.

        With prognostic humidity, atmosphere heating comes from precipitation, not
        evaporation. The atmosphere Jacobian w.r.t. temperature is zero since
        precipitation is diagnostic (depends on humidity, not current temperature).

        Returns
        -------
        (diag, cross) where diag is (3, nlat, nlon), cross is (3, 3, nlat, nlon)
        """
        if not self.enabled:
            if boundary_layer_temperature_K is not None:
                zeros = np.zeros_like(surface_temperature_K)
                diag = np.stack([zeros, zeros, zeros])
                cross = np.zeros((3, 3) + surface_temperature_K.shape)
                return diag, cross
            else:
                zeros = np.zeros_like(surface_temperature_K)
                diag = np.stack([zeros, zeros])
                cross = np.zeros((2, 2) + surface_temperature_K.shape)
                return diag, cross

        surface_temperature = np.asarray(surface_temperature_K, dtype=float)
        atmosphere_temperature = np.asarray(atmosphere_temperature_K, dtype=float)
        humidity_q = np.asarray(humidity_q, dtype=float)

        if self._wind_model is not None:
            pressure, rho, wind_speed_10m, ch = self._wind_model.compute_atmospheric_properties(
                surface_temperature,
                atmosphere_temperature,
                wind_speed_reference_m_s,
                itcz_rad=itcz_rad,
            )
        else:
            if boundary_layer_temperature_K is not None:
                zeros = np.zeros_like(surface_temperature_K)
                diag = np.stack([zeros, zeros, zeros])
                cross = np.zeros((3, 3) + surface_temperature_K.shape)
                return diag, cross
            else:
                zeros = np.zeros_like(surface_temperature_K)
                diag = np.stack([zeros, zeros])
                cross = np.zeros((2, 2) + surface_temperature_K.shape)
                return diag, cross

        wind_abs = np.maximum(np.abs(wind_speed_10m), self._config.minimum_wind_speed_m_s)
        R = GAS_CONSTANT_J_KG_K

        # Compute saturation vapor pressure and its derivative
        # Magnus formula gives e_sat in hPa, pressure is in Pa - convert to hPa
        # Cap temperature to prevent overflow in exp
        surface_temperature_C = np.clip(surface_temperature_K - 273.15, -100.0, 80.0)
        e_sat = 6.112 * np.exp(17.67 * surface_temperature_C / (surface_temperature_C + 243.5))
        pressure_hPa = pressure / 100.0

        # Derivative of e_sat with respect to T_C
        # d(e_sat)/dT_C = e_sat * 17.67 * 243.5 / (T_C + 243.5)^2
        de_sat_dT_C = e_sat * 17.67 * 243.5 / np.power(surface_temperature_C + 243.5, 2)

        # Compute q_sat and its derivatives (using hPa for consistency)
        # Ensure denominator doesn't go negative or zero
        denom = np.maximum(pressure_hPa - (1 - 0.622) * e_sat, 1.0)
        q_sat = (0.622 * e_sat) / denom
        humidity_q_clamped = np.minimum(humidity_q, q_sat)
        q_deficit = q_sat - humidity_q_clamped

        # Derivative of q_sat with respect to T_surf (via e_sat)
        # d(q_sat)/dT_surf = 0.622 * pressure * d(e_sat)/dT_surf / (pressure - (1-0.622)*e_sat)^2
        dq_sat_dT_surf = 0.622 * pressure_hPa * de_sat_dT_C / np.power(denom, 2)

        # Derivative of q_sat with respect to T_atm (via pressure)
        # Approximate dp/dT_atm ≈ -30 Pa/K = -0.30 hPa/K (from thermal wind scaling)
        dp_dT_atm_approx = -0.30  # hPa/K
        dq_sat_dT_atm = (
            -0.622
            * e_sat
            * (1.0 - (1.0 - 0.622) * e_sat / np.maximum(pressure_hPa, 1.0))
            * dp_dT_atm_approx
            / np.power(denom, 2)
        )

        # Compute near-surface air temperature and its derivatives
        # T_2m = T_boundary + lapse_correction (constant correction)
        if boundary_layer_temperature_K is not None:
            boundary_temperature = np.asarray(boundary_layer_temperature_K, dtype=float)
            mid_layer_height_m = BOUNDARY_LAYER_HEIGHT_M / 2.0  # 375m
            height_difference_m = mid_layer_height_m - 2.0  # 373m
            lapse_correction_K = STANDARD_LAPSE_RATE_K_PER_M * height_difference_m
            near_surface_air_K = boundary_temperature + lapse_correction_K
            dT_2m_dT_surf = np.zeros_like(surface_temperature)
            dT_2m_dT_atm = np.zeros_like(surface_temperature)
        else:
            near_surface_air_K = surface_temperature.copy()
            dT_2m_dT_surf = np.ones_like(surface_temperature)
            dT_2m_dT_atm = np.zeros_like(surface_temperature)

        # Density: rho = pressure / (R * T_2m)
        # Derivatives:
        # ∂rho/∂T_surf = -pressure / (R * T_2m^2) * dT_2m/dT_surf
        # ∂rho/∂T_atm = (1/R) * (dp/dT_atm / T_2m - p * dT_2m/dT_atm / T_2m^2)
        drho_dT_surf = -pressure / (R * np.power(near_surface_air_K, 2)) * dT_2m_dT_surf
        drho_dT_atm = (1.0 / R) * (
            dp_dT_atm_approx / near_surface_air_K
            - pressure * dT_2m_dT_atm / np.power(near_surface_air_K, 2)
        )

        # Latent heat flux: F = rho * L_v * ch * wind_abs * (q_sat - humidity_q)
        # Full derivatives:
        # ∂F/∂T_surf = L_v * ch * wind_abs * [rho * dq_sat/dT_surf + q_deficit * drho/dT_surf]
        # ∂F/∂T_atm = L_v * ch * wind_abs * [rho * dq_sat/dT_atm + q_deficit * drho/dT_atm]
        dheat_flux_dT_surf = (
            LATENT_HEAT_VAPORIZATION_J_KG
            * ch
            * wind_abs
            * (rho * dq_sat_dT_surf + q_deficit * drho_dT_surf)
        )
        dheat_flux_dT_atm = (
            LATENT_HEAT_VAPORIZATION_J_KG
            * ch
            * wind_abs
            * (rho * dq_sat_dT_atm + q_deficit * drho_dT_atm)
        )

        # Apply Manabe beta land factor (frozen during Newton iterations)
        theta_crit = self._config.manabe_theta_crit
        if soil_moisture is not None:
            land_factor = np.minimum(soil_moisture / theta_crit, 1.0)
        else:
            q_sat_safe = np.maximum(q_sat, 1e-10)
            rh = humidity_q_clamped / q_sat_safe
            land_factor = np.minimum(rh / theta_crit, 1.0)
        dheat_flux_dT_surf = np.where(
            self._land_mask, dheat_flux_dT_surf * land_factor, dheat_flux_dT_surf
        )
        dheat_flux_dT_atm = np.where(
            self._land_mask, dheat_flux_dT_atm * land_factor, dheat_flux_dT_atm
        )

        # No evaporation below freezing (zero derivatives)
        frozen = surface_temperature_C < self._config.freeze_threshold_c
        dheat_flux_dT_surf = np.where(frozen, 0.0, dheat_flux_dT_surf)
        dheat_flux_dT_atm = np.where(frozen, 0.0, dheat_flux_dT_atm)

        # Three-layer system: latent heat only between surface and boundary layer
        if boundary_layer_temperature_K is not None:
            if self._boundary_layer_heat_capacity is None:
                raise ValueError(
                    "boundary_layer_heat_capacity must be provided when using boundary_layer_temperature_K"
                )

            # For 3-layer, need derivative with respect to boundary temperature
            # T_2m = T_boundary + lapse_correction, so dT_2m/dT_boundary = 1
            # ∂rho/∂T_boundary = -pressure / (R * T_2m^2) * dT_2m/dT_boundary = -rho / T_2m
            drho_dT_boundary = -rho / near_surface_air_K

            # q_sat doesn't depend on T_boundary directly, only via pressure (which depends on T_atm)
            # So ∂q_sat/∂T_boundary = 0
            dq_sat_dT_boundary = np.zeros_like(surface_temperature)

            # ∂F/∂T_boundary = L_v * ch * wind_abs * [rho * dq_sat/dT_boundary + q_deficit * drho/dT_boundary]
            dheat_flux_dT_boundary = (
                LATENT_HEAT_VAPORIZATION_J_KG
                * ch
                * wind_abs
                * (rho * dq_sat_dT_boundary + q_deficit * drho_dT_boundary)
            )
            # Apply land factor (same as for other derivatives)
            dheat_flux_dT_boundary = np.where(
                self._land_mask, dheat_flux_dT_boundary * land_factor, dheat_flux_dT_boundary
            )
            # No evaporation below freezing (zero derivatives)
            dheat_flux_dT_boundary = np.where(frozen, 0.0, dheat_flux_dT_boundary)

            # Surface tendency: -evap_heat_flux / C_surf
            # ∂/∂T_surf = -dheat_flux_dT_surf / C_surf
            # ∂/∂T_boundary = -dheat_flux_dT_boundary / C_surf
            # ∂/∂T_atm = -dheat_flux_dT_atm / C_surf
            surface_diag = -dheat_flux_dT_surf / self._surface_heat_capacity
            surface_boundary_coupling = -dheat_flux_dT_boundary / self._surface_heat_capacity
            surface_atm_coupling = -dheat_flux_dT_atm / self._surface_heat_capacity

            # Boundary tendency: 0 (BL just transports moisture)
            boundary_diag = np.zeros_like(surface_diag)

            # Atmosphere: precipitation-based heating has zero Jacobian (lagged diagnostic)
            atmosphere_diag = np.zeros_like(surface_diag)
            atmosphere_surface_coupling = np.zeros_like(surface_diag)
            atmosphere_boundary_coupling = np.zeros_like(surface_diag)

            diag = np.stack([surface_diag, boundary_diag, atmosphere_diag])
            cross = np.zeros((3, 3) + surface_temperature.shape)
            cross[0, 1] = surface_boundary_coupling  # Surface-boundary coupling
            cross[0, 2] = surface_atm_coupling  # Surface-atmosphere coupling
            cross[2, 0] = atmosphere_surface_coupling  # Atmosphere-surface coupling
            cross[2, 1] = atmosphere_boundary_coupling  # Atmosphere-boundary coupling

            return diag, cross

        # Two-layer system: latent heat between surface and atmosphere
        # Surface tendency: -evap_heat_flux / C_surf
        # ∂/∂T_surf = -dheat_flux_dT_surf / C_surf
        # ∂/∂T_atm = -dheat_flux_dT_atm / C_surf
        surface_diag = -dheat_flux_dT_surf / self._surface_heat_capacity
        surface_atm_coupling = -dheat_flux_dT_atm / self._surface_heat_capacity

        # Atmosphere tendency: P * Lv / C_atm (from precipitation)
        # Jacobian is zero since precipitation is diagnostic (lagged)
        atmosphere_diag = np.zeros_like(surface_diag)
        atmosphere_surface_coupling = np.zeros_like(surface_diag)

        diag = np.stack([surface_diag, atmosphere_diag])
        cross = np.zeros((2, 2) + surface_temperature.shape)
        cross[0, 1] = surface_atm_coupling  # Surface-atmosphere coupling
        cross[1, 0] = atmosphere_surface_coupling  # Atmosphere-surface coupling

        return diag, cross

    def compute_evaporation_jacobian_wrt_humidity(
        self,
        surface_temperature_K: np.ndarray,
        atmosphere_temperature_K: np.ndarray,
        humidity_q: np.ndarray,
        *,
        wind_speed_reference_m_s: np.ndarray | None,
        itcz_rad: np.ndarray | None = None,
        boundary_layer_temperature_K: np.ndarray | None = None,
        soil_moisture: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute derivative of evaporation rate w.r.t. specific humidity.

        E = rho * Ch * |V| * (q_sat - q)
        dE/dq = -rho * Ch * |V|

        This is needed for the humidity Jacobian in the implicit solver.

        Returns
        -------
        np.ndarray
            dE/dq in units of (kg/m²/s) / (kg/kg) = 1/s
        """
        if not self.enabled or self._wind_model is None:
            return np.zeros_like(surface_temperature_K, dtype=float)

        surface_temperature = np.asarray(surface_temperature_K, dtype=float)
        atmosphere_temperature = np.asarray(atmosphere_temperature_K, dtype=float)

        pressure, rho, wind_speed_10m, ch = self._wind_model.compute_atmospheric_properties(
            surface_temperature,
            atmosphere_temperature,
            wind_speed_reference_m_s,
            itcz_rad=itcz_rad,
        )

        wind_abs = np.maximum(np.abs(wind_speed_10m), self._config.minimum_wind_speed_m_s)

        # E = rho * Ch * |V| * (q_sat - q)
        # dE/dq = -rho * Ch * |V| (negative: higher q means less evaporation)
        dE_dq = -rho * ch * wind_abs

        # Apply Manabe beta land factor
        surface_temperature_C = np.clip(surface_temperature_K - 273.15, -100.0, 80.0)
        theta_crit = self._config.manabe_theta_crit
        if soil_moisture is not None:
            land_factor = np.minimum(soil_moisture / theta_crit, 1.0)
        else:
            e_sat = 6.112 * np.exp(17.67 * surface_temperature_C / (surface_temperature_C + 243.5))
            pressure_hPa = pressure / 100.0
            denom = np.maximum(pressure_hPa - (1 - 0.622) * e_sat, 1.0)
            q_sat = (0.622 * e_sat) / denom
            humidity_q_clamped = np.minimum(np.asarray(humidity_q, dtype=float), q_sat)
            rh = humidity_q_clamped / np.maximum(q_sat, 1e-10)
            land_factor = np.minimum(rh / theta_crit, 1.0)
        dE_dq = np.where(self._land_mask, dE_dq * land_factor, dE_dq)

        # No evaporation below freezing
        frozen = surface_temperature_C < self._config.freeze_threshold_c
        dE_dq = np.where(frozen, 0.0, dE_dq)

        return dE_dq

    def compute_evaporation_jacobian_wrt_soil_moisture(
        self,
        surface_temperature_K: np.ndarray,
        atmosphere_temperature_K: np.ndarray,
        humidity_q: np.ndarray,
        *,
        wind_speed_reference_m_s: np.ndarray | None,
        itcz_rad: np.ndarray | None = None,
        boundary_layer_temperature_K: np.ndarray | None = None,
        soil_moisture: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute derivative of evaporation rate w.r.t. soil moisture.

        Only non-zero over land cells; zero over ocean and below freezing.
        """
        if not self.enabled or self._wind_model is None:
            return np.zeros_like(surface_temperature_K, dtype=float)

        surface_temperature = np.asarray(surface_temperature_K, dtype=float)
        atmosphere_temperature = np.asarray(atmosphere_temperature_K, dtype=float)

        pressure, rho, wind_speed_10m, ch = self._wind_model.compute_atmospheric_properties(
            surface_temperature,
            atmosphere_temperature,
            wind_speed_reference_m_s,
            itcz_rad=itcz_rad,
        )

        wind_abs = np.maximum(np.abs(wind_speed_10m), self._config.minimum_wind_speed_m_s)

        # Compute q_sat at surface temperature
        surface_temperature_C = np.clip(surface_temperature_K - 273.15, -100.0, 80.0)
        e_sat = 6.112 * np.exp(17.67 * surface_temperature_C / (surface_temperature_C + 243.5))
        pressure_hPa = pressure / 100.0
        denom = np.maximum(pressure_hPa - (1 - 0.622) * e_sat, 1.0)
        q_sat = (0.622 * e_sat) / denom

        humidity_q_clamped = np.minimum(np.asarray(humidity_q, dtype=float), q_sat)
        q_deficit = q_sat - humidity_q_clamped

        theta_crit = self._config.manabe_theta_crit

        # dE/dSM = rho * ch * |V| * (q_sat - q) / theta_crit when SM < theta_crit
        dE_dSM = rho * ch * wind_abs * q_deficit / theta_crit

        # Zero where SM >= theta_crit (beta already saturated at 1)
        if soil_moisture is not None:
            saturated = soil_moisture >= theta_crit
            dE_dSM = np.where(saturated, 0.0, dE_dSM)

        # Land only — zero over ocean
        dE_dSM = np.where(self._land_mask, dE_dSM, 0.0)

        # No evaporation below freezing
        frozen = surface_temperature_C < self._config.freeze_threshold_c
        dE_dSM = np.where(frozen, 0.0, dE_dSM)

        return dE_dSM
