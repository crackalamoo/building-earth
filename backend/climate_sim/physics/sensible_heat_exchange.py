"""Neutral sensible heat exchange between the surface and atmosphere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.atmosphere.atmosphere import (
    compute_two_meter_temperature,
)
from climate_sim.data.constants import (
    GAS_CONSTANT_J_KG_K,
    HEAT_CAPACITY_AIR_J_KG_K,
)
from climate_sim.data.elevation import VON_KARMAN_CONSTANT
from climate_sim.data.calendar import SECONDS_PER_DAY
from climate_sim.core.math_core import area_weighted_mean

from climate_sim.physics.atmosphere.wind import WindModel


@dataclass(frozen=True)
class SensibleHeatExchangeConfig:
    """Configuration for the neutral sensible heat exchange model."""

    enabled: bool = True
    von_karman: float = VON_KARMAN_CONSTANT
    gas_constant_dry_air_J_kg_K: float = GAS_CONSTANT_J_KG_K
    minimum_wind_speed_m_s: float = 2
    reference_height_surface_m: float = 2.0
    include_lapse_rate_elevation: bool = False


class SensibleHeatExchangeModel:
    """Compute tendencies from neutral sensible heat exchange."""

    def __init__(
        self,
        *,
        land_mask: np.ndarray,
        surface_heat_capacity_J_m2_K: np.ndarray,
        atmosphere_heat_capacity_J_m2_K: np.ndarray | float,
        wind_model: WindModel | None = None,
        config: SensibleHeatExchangeConfig | None = None,
        boundary_layer_heat_capacity_J_m2_K: np.ndarray | float | None = None,
        topographic_elevation: np.ndarray | None = None,
    ) -> None:
        self._config = config or SensibleHeatExchangeConfig()

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
        self._topographic_elevation = topographic_elevation

        # Boundary layer heat capacity (optional, for 3-layer system)
        if boundary_layer_heat_capacity_J_m2_K is not None:
            heat_capacity_boundary = np.asarray(boundary_layer_heat_capacity_J_m2_K, dtype=float)
            if heat_capacity_boundary.shape == ():
                heat_capacity_boundary = np.full(land_mask_bool.shape, float(heat_capacity_boundary))
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
        *,
        wind_speed_reference_m_s: np.ndarray | None,
        itcz_rad: np.ndarray | None = None,
        boundary_layer_temperature_K: np.ndarray | None = None,
        log_diagnostics: bool = False,
        cell_area_m2: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return surface and atmospheric tendencies from sensible heat exchange.

        If boundary_layer_temperature_K is provided, returns 3 tendencies:
        (surface, boundary_layer, atmosphere). Otherwise returns 2 tendencies.
        """

        if not self.enabled:
            zeros = np.zeros_like(surface_temperature_K, dtype=float)
            if boundary_layer_temperature_K is not None:
                return zeros, zeros, zeros
            return zeros, zeros

        surface_temperature = np.asarray(surface_temperature_K, dtype=float)
        atmosphere_temperature = np.asarray(atmosphere_temperature_K, dtype=float)

        if surface_temperature.shape != self._surface_heat_capacity.shape:
            raise ValueError(
                "Surface temperature must match the surface heat capacity field shape"
            )
        if atmosphere_temperature.shape != self._surface_heat_capacity.shape:
            raise ValueError(
                "Atmosphere temperature must match the surface heat capacity field shape"
            )

        # Use advection model for atmospheric properties if available
        if self._wind_model is not None:
            _pressure, rho, wind_speed_10m, ch = self._wind_model.compute_atmospheric_properties(
                surface_temperature,
                atmosphere_temperature,
                wind_speed_reference_m_s,
                itcz_rad=itcz_rad,
            )
        else:
            # Fallback: no wind, no exchange
            zeros = np.zeros_like(surface_temperature_K, dtype=float)
            if boundary_layer_temperature_K is not None:
                return zeros, zeros, zeros
            return zeros, zeros

        Cbl = 1.2e5  # J m-2 K-1

        # Three-layer system with boundary layer
        if boundary_layer_temperature_K is not None:
            if self._boundary_layer_heat_capacity is None:
                raise ValueError(
                    "boundary_layer_heat_capacity must be provided when using boundary_layer_temperature_K"
                )
            boundary_temperature = np.asarray(boundary_layer_temperature_K, dtype=float)

            # Surface ↔ Boundary layer: use existing roughness-based exchange
            near_surface_air_K = compute_two_meter_temperature(
                boundary_temperature,
                surface_temperature,
                topographic_elevation=self._topographic_elevation,
            )
            wind_abs = np.maximum(np.abs(wind_speed_10m), self._config.minimum_wind_speed_m_s)
            cp = HEAT_CAPACITY_AIR_J_KG_K

            # Surface-boundary exchange: bulk aerodynamic formula
            # Boundary layer is thin (750m) and well-mixed, use direct coupling
            g_surf = rho * cp * ch * wind_abs

            delta_surf_bl = surface_temperature - near_surface_air_K
            heat_flux_surf_bl = g_surf * delta_surf_bl

            # BL ↔ Atmosphere: No sensible heat flux here.
            # In reality, BL-atmosphere heat exchange occurs via:
            # 1. Vertical motion (subsidence/convection) - handled in vertical_motion.py
            # 2. Entrainment at BL top - implicitly included in vertical motion
            # 3. Radiation - handled in radiation.py
            # There is no turbulent sensible heat flux at the BL top like there is
            # at the surface, so this term is zero.
            heat_flux_bl_atm = np.zeros_like(heat_flux_surf_bl)

            surface_tendency = -heat_flux_surf_bl / self._surface_heat_capacity
            boundary_tendency = (heat_flux_surf_bl - heat_flux_bl_atm) / self._boundary_layer_heat_capacity
            atmosphere_tendency = heat_flux_bl_atm / self._atmosphere_heat_capacity

            if log_diagnostics:
                if cell_area_m2 is None:
                    raise ValueError("cell_area_m2 must be provided when log_diagnostics=True")

                # Compute effective timescales
                tau_bl_atm_days = area_weighted_mean(tau_bl_atm, cell_area_m2) / SECONDS_PER_DAY

                print("\n=== Sensible Heat Exchange Diagnostics ===")
                print(f"\nHeat Fluxes (W/m²):")
                print(f"Surface → Boundary layer:          {area_weighted_mean(heat_flux_surf_bl, cell_area_m2):7.2f}")
                print(f"Boundary layer → Atmosphere:       {area_weighted_mean(heat_flux_bl_atm, cell_area_m2):7.2f}")
                print(f"\nConductances (W/m²/K):")
                print(f"Surface-boundary (g_surf):         {area_weighted_mean(g_surf, cell_area_m2):7.2f}")
                print(f"Boundary-atmosphere (g_mix):       {area_weighted_mean(g_mix_bl_atm, cell_area_m2):7.2f}")
                print(f"\nTemperature Differences (K):")
                print(f"Surface - Near-surface air:        {area_weighted_mean(delta_surf_bl, cell_area_m2):7.2f}")
                print(f"Boundary layer - Atmosphere:       {area_weighted_mean(delta_bl_atm, cell_area_m2):7.2f}")
                print(f"\nEffective Timescales:")
                print(f"Boundary-atmosphere coupling:      {tau_bl_atm_days:7.2f} days")
                print(f"Mean wind speed (10m):             {area_weighted_mean(wind_abs, cell_area_m2):7.2f} m/s")
                print("=" * 40)

            return surface_tendency, boundary_tendency, atmosphere_tendency

        # Two-layer system: disabled (boundary layer is needed for proper physics)
        zeros = np.zeros_like(surface_temperature, dtype=float)
        return zeros, zeros

    def compute_jacobian(
        self,
        surface_temperature_K: np.ndarray,
        atmosphere_temperature_K: np.ndarray,
        *,
        wind_speed_reference_m_s: np.ndarray | None,
        itcz_rad: np.ndarray | None = None,
        boundary_layer_temperature_K: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Jacobian (diagonal and cross-coupling) for sensible heat exchange.

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

        if self._wind_model is not None:
            _pressure, rho, wind_speed_10m, ch = self._wind_model.compute_atmospheric_properties(
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
        cp = HEAT_CAPACITY_AIR_J_KG_K

        # Three-layer system
        if boundary_layer_temperature_K is not None:
            if self._boundary_layer_heat_capacity is None:
                raise ValueError(
                    "boundary_layer_heat_capacity must be provided when using boundary_layer_temperature_K"
                )

            # For the surface-boundary exchange, the heat flux is:
            # heat_flux_surf_bl = g_surf * (T_surf - T_2m)
            # where T_2m = T_boundary + lapse_correction - LAPSE_RATE * (elevation - 2.0)
            # The elevation term is constant, so:
            # ∂T_2m/∂T_surf = 0
            # ∂T_2m/∂T_boundary = 1
            # Therefore:
            # ∂(heat_flux)/∂T_surf = g_surf
            # ∂(heat_flux)/∂T_boundary = -g_surf

            g_surf = rho * cp * ch * wind_abs

            # Surface tendency: -heat_flux / C_surf
            # ∂/∂T_surf = -g_surf / C_surf
            # ∂/∂T_boundary = g_surf / C_surf
            surface_diag = -g_surf / self._surface_heat_capacity
            surface_boundary_coupling = g_surf / self._surface_heat_capacity

            # Boundary tendency: (heat_flux_surf_bl - heat_flux_bl_atm) / C_bl
            # From surface-boundary flux:
            #   ∂/∂T_surf = g_surf / C_bl
            #   ∂/∂T_boundary = -g_surf / C_bl
            # From boundary-atmosphere flux: g_mix_bl_atm * (T_bl - T_atm)
            # where g_mix_bl_atm = 1 / r_mix_bl_atm (turbulent mixing conductance)
            Cbl = self._boundary_layer_heat_capacity
            tau_bl_atm = (self._boundary_layer_heat_capacity * self._atmosphere_heat_capacity) / (
                self._boundary_layer_heat_capacity + self._atmosphere_heat_capacity
            ) / (rho * cp * ch * wind_abs)
            r_mix_bl_atm = tau_bl_atm / Cbl
            bl_atm_conductance = 1.0 / np.maximum(r_mix_bl_atm, 1e-9)

            boundary_diag = (-g_surf - bl_atm_conductance) / self._boundary_layer_heat_capacity
            boundary_surface_coupling = g_surf / self._boundary_layer_heat_capacity
            boundary_atm_coupling = bl_atm_conductance / self._boundary_layer_heat_capacity
            boundary_atm_coupling = 0

            # Atmosphere tendency: heat_flux_bl_atm / C_atm
            atmosphere_diag = -bl_atm_conductance / self._atmosphere_heat_capacity
            atm_boundary_coupling = bl_atm_conductance / self._atmosphere_heat_capacity
            atm_boundary_coupling = 0

            diag = np.stack([surface_diag, boundary_diag, atmosphere_diag])
            cross = np.zeros((3, 3) + surface_temperature.shape)
            cross[0, 1] = surface_boundary_coupling
            cross[1, 0] = boundary_surface_coupling
            cross[1, 2] = boundary_atm_coupling
            cross[2, 1] = atm_boundary_coupling

            return diag, cross

        # Two-layer system: disabled (boundary layer is needed for proper physics)
        zeros = np.zeros_like(surface_temperature_K)
        diag = np.stack([zeros, zeros])
        cross = np.zeros((2, 2) + surface_temperature_K.shape)
        return diag, cross
