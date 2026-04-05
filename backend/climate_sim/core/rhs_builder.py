"""Right-hand-side factory for energy-balance model physics assembly."""

import os

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from dataclasses import dataclass

import climate_sim.physics.radiation as radiation
from climate_sim.physics.sensible_heat_exchange import (
    SensibleHeatExchangeModel,
    SensibleHeatExchangeConfig,
)
from climate_sim.physics.latent_heat_exchange import (
    LatentHeatExchangeModel,
    LatentHeatExchangeConfig,
)
from climate_sim.physics.ocean_currents import OceanAdvectionConfig
from climate_sim.physics.atmosphere.wind import WindModel
from climate_sim.physics.atmosphere.advection import AdvectionOperator
from climate_sim.physics.diffusion import LayeredDiffusionOperator, DiffusionOperator
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.vertical_motion import (
    VerticalMotionConfig,
    compute_vertical_motion_tendencies,
    compute_bl_atm_mixing_tendencies,
    _bl_atm_mixing_tau,
    _BL_ATM_MIXING_T_SCALE,
    compute_hadley_subsidence_velocity,
    compute_hadley_upper_velocity,
    hadley_moisture_tendency_jacobian,
    _ALPHA,
)
from climate_sim.physics.orographic_effects import OrographicModel
from climate_sim.physics.surface_albedo import AlbedoModel
from climate_sim.physics.precipitation import (
    compute_precipitation_jacobian,
    compute_precipitation_recycling_jacobian,
    compute_precipitation_rh_gate,
)
from climate_sim.physics.humidity import (
    COLUMN_MASS_KG_M2,
    compute_saturation_specific_humidity,
    specific_humidity_to_relative_humidity,
)
from climate_sim.physics.clouds import (
    compute_cloud_fraction_jacobian,
    compute_clouds_and_precipitation,
    compute_vertical_velocity_from_divergence,
    CloudPrecipOutput,
)
from climate_sim.core.state import ModelState
from climate_sim.core.math_core import spherical_cell_area, compute_divergence
from climate_sim.data.constants import (
    R_EARTH_METERS,
    LATENT_HEAT_VAPORIZATION_J_KG,
    GAS_CONSTANT_WATER_VAPOR_J_KG_K,
    HEAT_CAPACITY_AIR_J_KG_K,
    BOUNDARY_LAYER_HEIGHT_M,
)
from typing import Callable


@dataclass
class Linearization:
    diag: np.ndarray
    cross: np.ndarray | None = None
    surface_diffusion_matrix: sparse.csc_matrix | None = None
    surface_advection_matrix: sparse.csc_matrix | None = None  # Ocean heat advection
    atmosphere_diffusion_matrix: sparse.csc_matrix | None = None
    atmosphere_advection_matrix: sparse.csc_matrix | None = None
    boundary_layer_diffusion_matrix: sparse.csc_matrix | None = None
    boundary_layer_advection_matrix: sparse.csc_matrix | None = None
    solver_fingerprint: str | None = None
    # Humidity Jacobian components (for prognostic humidity in Newton solver)
    humidity_advection_matrix: sparse.csc_matrix | None = None  # -V·∇q operator
    humidity_diffusion_matrix: sparse.csc_matrix | None = None  # Humidity diffusion operator
    humidity_diag: np.ndarray | None = None  # dE/dq - dP/dq diagonal
    humidity_temp_coupling: tuple[np.ndarray, ...] | None = (
        None  # dR_q/dT terms (dR_q/dTsfc, dR_q/dTbl, dR_q/dTatm)
    )
    temp_humidity_coupling: tuple[np.ndarray, ...] | None = (
        None  # dR_T/dq terms (dR_Tsfc/dq, dR_Tbl/dq, dR_Tatm/dq)
    )
    # Soil moisture Jacobian components (for prognostic SM in Newton solver)
    soil_diag: np.ndarray | None = None  # d(SM_tendency)/dSM diagonal
    soil_humidity_coupling: np.ndarray | None = None  # d(SM_tendency)/dq
    soil_temp_coupling: tuple[np.ndarray, ...] | None = (
        None  # d(SM_tendency)/dT terms (dTsfc, dTbl, dTatm)
    )
    temp_soil_coupling: tuple[np.ndarray, ...] | None = (
        None  # d(T_tendency)/dSM terms (dTsfc, dTbl, dTatm)
    )
    humidity_soil_coupling: np.ndarray | None = None  # d(q_tendency)/dSM


type FloatArray = NDArray[np.floating]

type RhsFn = Callable[[ModelState, FloatArray, FloatArray], FloatArray]
type RhsDerivativeFn = Callable[[ModelState, FloatArray, FloatArray], Linearization]


@dataclass(frozen=True, slots=True)
class RhsBuildInputs:
    """Structured inputs for RHS factory functions."""

    heat_capacity_field: FloatArray
    diffusion_operator: LayeredDiffusionOperator
    radiation_config: RadiationConfig
    land_mask: FloatArray
    roughness_length: FloatArray
    topographic_elevation: FloatArray
    sensible_heat_cfg: SensibleHeatExchangeConfig
    latent_heat_cfg: LatentHeatExchangeConfig
    wind_model: WindModel | None
    advection_operator: AdvectionOperator | None
    lon2d: FloatArray
    lat2d: FloatArray
    ocean_advection_cfg: OceanAdvectionConfig | None = None
    vertical_motion_cfg: VerticalMotionConfig | None = None
    humidity_diffusion_operator: DiffusionOperator | None = None
    orographic_model: OrographicModel | None = None
    amoc_velocity: tuple[FloatArray, FloatArray] | None = None
    albedo_model: AlbedoModel | None = None
    base_albedo_field: FloatArray | None = None
    ice_sheet_mask: FloatArray | None = None


def create_rhs_functions(inputs: RhsBuildInputs) -> tuple[RhsFn, RhsDerivativeFn]:
    """Build RHS and Jacobian functions from physics configuration.

    This factory assembles the right-hand-side tendency function and its
    temperature derivative (Jacobian) from the provided physics operators.

    Parameters
    ----------
    inputs : RhsBuildInputs
        Structured inputs containing all physics operators and configuration.

    Returns
    -------
    tuple[RhsFn, RhsDerivativeFn]
        The RHS function and its derivative for use in the solver.
    """
    # Extract and linearize diffusion operators
    surface_diffusion_diag: np.ndarray | None = None
    surface_matrix = None
    if inputs.diffusion_operator.surface.enabled:
        surface_diffusion_diag, surface_offdiag = (
            inputs.diffusion_operator.surface.linearised_tendency()
        )
        if surface_offdiag is not None and surface_offdiag.nnz > 0:
            surface_matrix = (
                surface_offdiag
                if sparse.isspmatrix_csc(surface_offdiag)
                else surface_offdiag.tocsc()
            )

    atmosphere_diffusion_diag: np.ndarray | None = None
    atmosphere_matrix = None
    if inputs.radiation_config.include_atmosphere and inputs.diffusion_operator.atmosphere.enabled:
        atmosphere_diffusion_diag, atmosphere_offdiag = (
            inputs.diffusion_operator.atmosphere.linearised_tendency()
        )
        if atmosphere_offdiag is not None and atmosphere_offdiag.nnz > 0:
            atmosphere_matrix = (
                atmosphere_offdiag
                if sparse.isspmatrix_csc(atmosphere_offdiag)
                else atmosphere_offdiag.tocsc()
            )

    boundary_layer_diffusion_diag: np.ndarray | None = None
    boundary_layer_matrix = None
    if inputs.diffusion_operator.boundary_layer is not None:
        boundary_layer_diffusion_diag, boundary_layer_offdiag = (
            inputs.diffusion_operator.boundary_layer.linearised_tendency()
        )
        if boundary_layer_offdiag is not None and boundary_layer_offdiag.nnz > 0:
            boundary_layer_matrix = (
                boundary_layer_offdiag
                if sparse.isspmatrix_csc(boundary_layer_offdiag)
                else boundary_layer_offdiag.tocsc()
            )

    # Build heat exchange models
    sensible_heat_model: SensibleHeatExchangeModel | None = None
    if inputs.radiation_config.include_atmosphere and inputs.sensible_heat_cfg.enabled:
        sensible_heat_model = SensibleHeatExchangeModel(
            land_mask=inputs.land_mask,
            surface_heat_capacity_J_m2_K=inputs.heat_capacity_field,
            atmosphere_heat_capacity_J_m2_K=inputs.radiation_config.atmosphere_heat_capacity,
            wind_model=inputs.wind_model,
            config=inputs.sensible_heat_cfg,
            boundary_layer_heat_capacity_J_m2_K=inputs.radiation_config.boundary_layer_heat_capacity,
            topographic_elevation=inputs.topographic_elevation,
        )

    latent_heat_model: LatentHeatExchangeModel | None = None
    if inputs.radiation_config.include_atmosphere and inputs.latent_heat_cfg.enabled:
        latent_heat_model = LatentHeatExchangeModel(
            land_mask=inputs.land_mask,
            surface_heat_capacity_J_m2_K=inputs.heat_capacity_field,
            atmosphere_heat_capacity_J_m2_K=inputs.radiation_config.atmosphere_heat_capacity,
            wind_model=inputs.wind_model,
            config=inputs.latent_heat_cfg,
            boundary_layer_heat_capacity_J_m2_K=inputs.radiation_config.boundary_layer_heat_capacity,
        )

    def rhs(state: ModelState, insolation: np.ndarray, itcz_rad: np.ndarray) -> np.ndarray:
        """Compute temperature tendencies from all physics."""
        log_radiation = os.environ.get("LOG_RADIATION", "0") == "1"

        # Compute cell areas for area-weighted diagnostics
        cell_areas = None
        if log_radiation:
            cell_areas = spherical_cell_area(
                inputs.lon2d,
                inputs.lat2d,
                earth_radius_m=R_EARTH_METERS,
            )

        # Unified cloud physics: compute cloud fractions for separate cloud types
        # (convective, stratiform, marine Sc, high clouds) with proper emission temperatures
        cloud_output: CloudPrecipOutput | None = state.cloud_output
        nlayers = state.temperature.shape[0]
        humidity_field = state.humidity_field

        # Compute unified clouds if not already provided in state
        if (
            cloud_output is None
            and humidity_field is not None
            and inputs.radiation_config.include_atmosphere
        ):
            # Get wind for vertical velocity computation
            if nlayers == 3 and state.boundary_layer_wind_field is not None:
                wind_u, wind_v, _ = state.boundary_layer_wind_field
            elif state.wind_field is not None:
                wind_u, wind_v, _ = state.wind_field
            else:
                wind_u = wind_v = None

            # Compute vertical velocity from wind divergence + orographic lifting
            if wind_u is not None and wind_v is not None:
                divergence = compute_divergence(wind_u, wind_v, inputs.lat2d, inputs.lon2d)
                vertical_velocity = compute_vertical_velocity_from_divergence(divergence)
                if state.orographic_w is not None:
                    vertical_velocity = vertical_velocity + state.orographic_w
                elif inputs.orographic_model is not None:
                    vertical_velocity = (
                        vertical_velocity
                        + inputs.orographic_model.compute_orographic_vertical_velocity(
                            wind_u, wind_v
                        )
                    )
            else:
                vertical_velocity = np.zeros_like(humidity_field)

            # Get temperature fields
            surface_temp = state.temperature[0]
            T_bl = state.temperature[1]
            T_atm = state.temperature[2]

            # Compute relative humidity
            rh = specific_humidity_to_relative_humidity(
                humidity_field, T_bl, itcz_rad=itcz_rad, lat2d=inputs.lat2d, lon2d=inputs.lon2d
            )

            # Compute unified cloud-precipitation
            cloud_output = compute_clouds_and_precipitation(
                T_bl_K=T_bl,
                T_atm_K=T_atm,
                q=humidity_field,
                rh=rh,
                vertical_velocity=vertical_velocity,
                T_surface_K=surface_temp,
                ocean_mask=~inputs.land_mask if inputs.land_mask is not None else None,
            )

        radiative = radiation.radiative_balance_rhs(
            state.temperature,
            insolation,
            heat_capacity_field=inputs.heat_capacity_field,
            albedo_field=state.albedo_field,
            config=inputs.radiation_config,
            land_mask=inputs.land_mask,
            humidity_q=state.humidity_field,
            log_diagnostics=log_radiation,
            cell_area_m2=cell_areas,
            itcz_rad=itcz_rad,
            lat2d=inputs.lat2d,
            lon2d=inputs.lon2d,
            cloud_output=cloud_output,  # Use unified cloud physics with optical depth
        )
        if inputs.radiation_config.include_atmosphere:
            radiative = radiative.copy()
            nlayers = state.temperature.shape[0]
            surface_temperature = state.temperature[0]
            # Use boundary layer wind (with drag) for wind speed reference
            if nlayers == 3 and state.boundary_layer_wind_field is not None:
                wind_speed_ref = state.boundary_layer_wind_field[2]
            elif state.wind_field is not None:
                wind_speed_ref = state.wind_field[2]
            else:
                wind_speed_ref = None
            humidity_field = state.humidity_field

            if inputs.diffusion_operator.surface.enabled:
                radiative[0] += inputs.diffusion_operator.surface.tendency(surface_temperature)

            # Ocean heat transport: use advection operator with ocean currents
            ocean_advection_enabled = (
                inputs.ocean_advection_cfg is not None and inputs.ocean_advection_cfg.enabled
            )
            if ocean_advection_enabled and state.ocean_current_field is not None:
                ocean_u, ocean_v = state.ocean_current_field
                # Create advection operator for ocean surface if not already available
                if inputs.advection_operator is not None:
                    # Replace NaN (land cells) with 0 for the advection operator
                    ocean_u_safe = np.where(np.isnan(ocean_u), 0.0, ocean_u)
                    ocean_v_safe = np.where(np.isnan(ocean_v), 0.0, ocean_v)
                    # Use the existing advection operator infrastructure
                    ocean_advection_tendency = inputs.advection_operator.tendency(
                        surface_temperature, ocean_u_safe, ocean_v_safe
                    )
                    # Mask out land cells
                    ocean_mask = ~inputs.land_mask
                    ocean_advection_tendency = np.where(ocean_mask, ocean_advection_tendency, 0.0)
                    radiative[0] += ocean_advection_tendency

            # Ekman pumping: upwelling entrainment of deep ocean water
            # w_E from offshore Ekman transport at coastal cells (positive = upwelling)
            if state.ocean_ekman_pumping is not None and state.deep_ocean_temperature is not None:
                w_E = np.nan_to_num(state.ocean_ekman_pumping, nan=0.0)  # m/s
                H_mix = 70.0  # m, consistent with ocean heat capacity
                T_deep = state.deep_ocean_temperature  # K
                # Cap upwelling velocity to avoid extreme values near equator
                w_E_up = np.minimum(np.maximum(w_E, 0.0), 5e-6)
                # dT/dt = w_E/H * (T_deep - T_sfc) [K/s]
                entrainment = (w_E_up / H_mix) * (T_deep - surface_temperature)
                # Only apply where upwelling cools (T_deep < T_sfc); at high
                # latitudes T_deep > T_sfc which would spuriously warm
                ocean_mask = ~inputs.land_mask
                cooling_mask = ocean_mask & (T_deep < surface_temperature)
                radiative[0] += np.where(cooling_mask, entrainment, 0.0)

            # AMOC thermohaline advection (separate from wind-driven currents)
            if inputs.amoc_velocity is not None and inputs.advection_operator is not None:
                u_amoc, v_amoc = inputs.amoc_velocity
                u_safe = np.nan_to_num(u_amoc, nan=0.0)
                v_safe = np.nan_to_num(v_amoc, nan=0.0)
                amoc_tendency = inputs.advection_operator.tendency(
                    surface_temperature, u_safe, v_safe
                )
                # Only apply on ocean cells within the AMOC region
                amoc_mask = ~np.isnan(u_amoc)
                radiative[0] += np.where(amoc_mask, amoc_tendency, 0.0)

            if nlayers == 3:
                # Three-layer system
                boundary_temperature = state.temperature[1]
                atmosphere_temperature = state.temperature[2]

                if inputs.diffusion_operator.boundary_layer is not None:
                    radiative[1] += inputs.diffusion_operator.boundary_layer.tendency(
                        boundary_temperature
                    )
                if inputs.diffusion_operator.atmosphere.enabled:
                    radiative[2] += inputs.diffusion_operator.atmosphere.tendency(
                        atmosphere_temperature
                    )

                # Apply advection to both layers with appropriate wind fields
                if inputs.advection_operator is not None:
                    # Boundary layer advection: use boundary layer wind (with drag, already blocked)
                    if state.boundary_layer_wind_field is not None:
                        wind_u_bl, wind_v_bl, _ = state.boundary_layer_wind_field
                        advection_boundary = inputs.advection_operator.tendency(
                            boundary_temperature, wind_u_bl, wind_v_bl
                        )
                        radiative[1] += advection_boundary

                    # Free atmosphere advection: use free atmosphere wind (geostrophic)
                    if state.wind_field is not None:
                        wind_u_atm, wind_v_atm, _ = state.wind_field
                        advection_atmosphere = inputs.advection_operator.tendency(
                            atmosphere_temperature, wind_u_atm, wind_v_atm
                        )
                        radiative[2] += advection_atmosphere

                    # Upper-branch Hadley advection of T_atm
                    if (
                        inputs.vertical_motion_cfg is not None
                        and inputs.vertical_motion_cfg.enabled
                        and inputs.vertical_motion_cfg.hadley_descent_velocity_m_s > 0
                    ):
                        lat_rad_2d = np.deg2rad(inputs.lat2d)
                        w_hadley_fwd = compute_hadley_subsidence_velocity(
                            lat_rad_2d,
                            itcz_rad,
                            peak_velocity_m_s=inputs.vertical_motion_cfg.hadley_descent_velocity_m_s,
                        )
                        v_upper = compute_hadley_upper_velocity(w_hadley_fwd, lat_rad_2d, itcz_rad)
                        u_zero = np.zeros_like(v_upper)
                        radiative[2] += inputs.advection_operator.tendency(
                            atmosphere_temperature, u_zero, v_upper
                        )

                if sensible_heat_model is not None:
                    tendencies = sensible_heat_model.compute_tendencies(
                        surface_temperature_K=surface_temperature,
                        atmosphere_temperature_K=atmosphere_temperature,
                        wind_speed_reference_m_s=wind_speed_ref,
                        itcz_rad=itcz_rad,
                        boundary_layer_temperature_K=boundary_temperature,
                        log_diagnostics=log_radiation,
                        cell_area_m2=cell_areas,
                    )
                    assert isinstance(tendencies, tuple)
                    assert len(tendencies) == 3
                    surface_tendency, boundary_tendency, atmosphere_tendency = tendencies
                    radiative[0] += surface_tendency
                    radiative[1] += boundary_tendency
                    radiative[2] += atmosphere_tendency

                if humidity_field is not None and latent_heat_model is not None:
                    # Pass precipitation rate for correct latent heat routing:
                    # - Surface loses heat via evaporation
                    # - Atmosphere gains heat via precipitation (condensation)
                    precip_rate = state.precipitation_field
                    tendencies = latent_heat_model.compute_tendencies(
                        surface_temperature_K=surface_temperature,
                        atmosphere_temperature_K=atmosphere_temperature,
                        humidity_q=humidity_field,
                        wind_speed_reference_m_s=wind_speed_ref,
                        itcz_rad=itcz_rad,
                        boundary_layer_temperature_K=boundary_temperature,
                        precipitation_rate=precip_rate,
                        soil_moisture=state.soil_moisture,
                        vegetation_fraction=state.vegetation_fraction,
                    )
                    assert isinstance(tendencies, tuple)
                    assert len(tendencies) == 4  # surface, BL, atm, evap_rate
                    surface_tendency, boundary_tendency, atmosphere_tendency, _evap_rate = (
                        tendencies
                    )
                    radiative[0] += surface_tendency
                    radiative[1] += boundary_tendency
                    radiative[2] += atmosphere_tendency

                # Vertical motion: energy-conserving heat exchange between BL and atmosphere
                vertical_motion_enabled = (
                    inputs.vertical_motion_cfg is not None and inputs.vertical_motion_cfg.enabled
                )
                if vertical_motion_enabled and state.boundary_layer_wind_field is not None:
                    wind_u_vm, wind_v_vm, _ = state.boundary_layer_wind_field
                    divergence = compute_divergence(
                        wind_u_vm, wind_v_vm, inputs.lat2d, inputs.lon2d
                    )
                    bl_tendency, atm_tendency = compute_vertical_motion_tendencies(
                        divergence, boundary_temperature, atmosphere_temperature
                    )
                    radiative[1] += bl_tendency
                    radiative[2] += atm_tendency

                # Background BL-atmosphere mixing (global subsidence/entrainment)
                tau_mix = (
                    inputs.vertical_motion_cfg.tau_bl_atm_mixing_s
                    if inputs.vertical_motion_cfg is not None
                    else 0.0
                )
                if tau_mix > 0:
                    bl_mix, atm_mix = compute_bl_atm_mixing_tendencies(
                        boundary_temperature,
                        atmosphere_temperature,
                        C_bl=inputs.radiation_config.boundary_layer_heat_capacity,
                        C_atm=inputs.radiation_config.atmosphere_heat_capacity,
                        tau_s=tau_mix,
                    )
                    radiative[1] += bl_mix
                    radiative[2] += atm_mix

            return radiative
        if inputs.diffusion_operator.surface.enabled:
            radiative = radiative + inputs.diffusion_operator.surface.tendency(state.temperature)
        return radiative

    def rhs_derivative(
        state: ModelState, insolation: np.ndarray, itcz_rad: np.ndarray
    ) -> Linearization:
        """Compute Jacobian of RHS with respect to temperature."""

        # Unified cloud physics: compute cloud fractions for separate cloud types
        # (convective, stratiform, marine Sc, high clouds) with proper emission temperatures
        cloud_output: CloudPrecipOutput | None = state.cloud_output
        computed_vertical_velocity: np.ndarray | None = None
        nlayers = state.temperature.shape[0]
        humidity_field = state.humidity_field

        # Compute unified clouds if not already provided in state
        if (
            cloud_output is None
            and humidity_field is not None
            and inputs.radiation_config.include_atmosphere
        ):
            # Get wind for vertical velocity computation
            if nlayers == 3 and state.boundary_layer_wind_field is not None:
                wind_u, wind_v, _ = state.boundary_layer_wind_field
            elif state.wind_field is not None:
                wind_u, wind_v, _ = state.wind_field
            else:
                wind_u = wind_v = None

            # Compute vertical velocity from wind divergence + orographic lifting
            if wind_u is not None and wind_v is not None:
                divergence = compute_divergence(wind_u, wind_v, inputs.lat2d, inputs.lon2d)
                vertical_velocity = compute_vertical_velocity_from_divergence(divergence)
                if state.orographic_w is not None:
                    vertical_velocity = vertical_velocity + state.orographic_w
                elif inputs.orographic_model is not None:
                    vertical_velocity = (
                        vertical_velocity
                        + inputs.orographic_model.compute_orographic_vertical_velocity(
                            wind_u, wind_v
                        )
                    )
            else:
                vertical_velocity = np.zeros_like(humidity_field)
            computed_vertical_velocity = vertical_velocity

            surface_temp = state.temperature[0]
            T_bl = state.temperature[1]
            T_atm = state.temperature[2]

            rh = specific_humidity_to_relative_humidity(
                humidity_field, T_bl, itcz_rad=itcz_rad, lat2d=inputs.lat2d, lon2d=inputs.lon2d
            )

            cloud_output = compute_clouds_and_precipitation(
                T_bl_K=T_bl,
                T_atm_K=T_atm,
                q=humidity_field,
                rh=rh,
                vertical_velocity=vertical_velocity,
                T_surface_K=surface_temp,
                ocean_mask=~inputs.land_mask if inputs.land_mask is not None else None,
            )

        if inputs.radiation_config.include_atmosphere:
            # Cloud fraction Jacobian for chain rule
            cld_jac = None
            if cloud_output is not None and humidity_field is not None:
                T_bl_jac = state.temperature[1]
                T_atm_jac = state.temperature[2]
                rh_jac = specific_humidity_to_relative_humidity(
                    humidity_field,
                    T_bl_jac,
                    itcz_rad=itcz_rad,
                    lat2d=inputs.lat2d,
                    lon2d=inputs.lon2d,
                )
                w_jac = (
                    computed_vertical_velocity
                    if computed_vertical_velocity is not None
                    else np.zeros_like(humidity_field)
                )
                cld_jac = compute_cloud_fraction_jacobian(
                    T_bl_jac,
                    T_atm_jac,
                    humidity_field,
                    rh_jac,
                    w_jac,
                    cloud_output,
                )

            # Compute snow/ice albedo derivative dα/dT_sfc for chain rule.
            # Snow fraction uses T_surface, so the derivative goes into
            # surface_diag (self-feedback on surface temperature).
            albedo_derivative: np.ndarray | None = None
            if (
                inputs.albedo_model is not None
                and inputs.base_albedo_field is not None
            ):
                T_sfc_C = state.temperature[0] - 273.15
                albedo_derivative = inputs.albedo_model.compute_albedo_derivative(
                    T_sfc_C,
                    inputs.base_albedo_field,
                    ice_sheet_mask=inputs.ice_sheet_mask,
                )

            radiative_derivative = radiation.radiative_balance_rhs_temperature_derivative(
                state.temperature,
                heat_capacity_field=inputs.heat_capacity_field,
                config=inputs.radiation_config,
                land_mask=inputs.land_mask,
                humidity_q=state.humidity_field,
                lat2d=inputs.lat2d,
                lon2d=inputs.lon2d,
                itcz_rad=itcz_rad,
                cloud_output=cloud_output,
                cloud_jacobian=cld_jac,
                insolation_W_m2=insolation,
                albedo_field=state.albedo_field,
                albedo_derivative=albedo_derivative,
            )
            assert isinstance(radiative_derivative, tuple)
            radiative_diag, cross = radiative_derivative
            diag = radiative_diag.copy()

            if surface_diffusion_diag is not None:
                diag[0] = diag[0] + surface_diffusion_diag

            if nlayers == 3:
                # 3-layer: boundary is layer 1, atmosphere is layer 2
                if boundary_layer_diffusion_diag is not None:
                    diag[1] = diag[1] + boundary_layer_diffusion_diag
                if atmosphere_diffusion_diag is not None:
                    diag[2] = diag[2] + atmosphere_diffusion_diag

            # Ocean advection Jacobian
            surface_advection_matrix = None
            ocean_advection_enabled = (
                inputs.ocean_advection_cfg is not None and inputs.ocean_advection_cfg.enabled
            )
            if (
                ocean_advection_enabled
                and state.ocean_current_field is not None
                and inputs.advection_operator is not None
            ):
                ocean_u, ocean_v = state.ocean_current_field
                # Replace NaN (land cells) with 0 for the advection operator
                ocean_u_safe = np.where(np.isnan(ocean_u), 0.0, ocean_u)
                ocean_v_safe = np.where(np.isnan(ocean_v), 0.0, ocean_v)
                _, ocean_advection_mat = inputs.advection_operator.linearised_tendency(
                    ocean_u_safe, ocean_v_safe
                )
                if ocean_advection_mat is not None:
                    surface_advection_matrix = (
                        ocean_advection_mat
                        if sparse.isspmatrix_csc(ocean_advection_mat)
                        else ocean_advection_mat.tocsc()
                    )

            # Compute atmosphere/boundary layer advection Jacobians if enabled
            atmosphere_advection_matrix = None
            boundary_layer_advection_matrix = None
            if inputs.advection_operator is not None:
                if nlayers == 3:
                    # Three-layer: use separate wind fields for boundary layer and atmosphere
                    if state.boundary_layer_wind_field is not None:
                        wind_u_bl, wind_v_bl, _ = state.boundary_layer_wind_field
                        _, bl_matrix_csr = inputs.advection_operator.linearised_tendency(
                            wind_u_bl, wind_v_bl
                        )
                        if bl_matrix_csr is not None:
                            boundary_layer_advection_matrix = (
                                bl_matrix_csr
                                if sparse.isspmatrix_csc(bl_matrix_csr)
                                else bl_matrix_csr.tocsc()
                            )
                    if state.wind_field is not None:
                        wind_u_atm, wind_v_atm, _ = state.wind_field
                        _, atm_matrix_csr = inputs.advection_operator.linearised_tendency(
                            wind_u_atm, wind_v_atm
                        )
                        if atm_matrix_csr is not None:
                            atmosphere_advection_matrix = (
                                atm_matrix_csr
                                if sparse.isspmatrix_csc(atm_matrix_csr)
                                else atm_matrix_csr.tocsc()
                            )

                    # Upper-branch Hadley advection Jacobian for T_atm
                    if (
                        inputs.vertical_motion_cfg is not None
                        and inputs.vertical_motion_cfg.enabled
                        and inputs.vertical_motion_cfg.hadley_descent_velocity_m_s > 0
                    ):
                        lat_rad_2d = np.deg2rad(inputs.lat2d)
                        w_hadley_jac = compute_hadley_subsidence_velocity(
                            lat_rad_2d,
                            itcz_rad,
                            peak_velocity_m_s=inputs.vertical_motion_cfg.hadley_descent_velocity_m_s,
                        )
                        v_upper_jac = compute_hadley_upper_velocity(
                            w_hadley_jac, lat_rad_2d, itcz_rad
                        )
                        u_zero = np.zeros_like(v_upper_jac)
                        _, hadley_atm_mat = inputs.advection_operator.linearised_tendency(
                            u_zero, v_upper_jac
                        )
                        if hadley_atm_mat is not None:
                            hadley_csc = (
                                hadley_atm_mat
                                if sparse.isspmatrix_csc(hadley_atm_mat)
                                else hadley_atm_mat.tocsc()
                            )
                            if atmosphere_advection_matrix is not None:
                                atmosphere_advection_matrix = (
                                    atmosphere_advection_matrix + hadley_csc
                                )
                            else:
                                atmosphere_advection_matrix = hadley_csc

            # Compute sensible heat exchange Jacobian if enabled
            if sensible_heat_model is not None:
                sh_diag, sh_cross = sensible_heat_model.compute_jacobian(
                    state.temperature[0],
                    state.temperature[2],
                    wind_speed_reference_m_s=state.wind_field[2]
                    if state.wind_field is not None
                    else None,
                    itcz_rad=None,
                    boundary_layer_temperature_K=state.temperature[1],
                )
                # Add sensible heat directly to diagonal and cross terms
                diag = diag + sh_diag
                if cross is not None:
                    cross = cross + sh_cross
                else:
                    cross = sh_cross

            # Compute latent heat exchange Jacobian if enabled
            if latent_heat_model is not None and state.humidity_field is not None:
                precip_rate = state.precipitation_field
                lh_diag, lh_cross = latent_heat_model.compute_jacobian(
                    state.temperature[0],
                    state.temperature[2],
                    state.humidity_field,
                    wind_speed_reference_m_s=state.wind_field[2]
                    if state.wind_field is not None
                    else None,
                    itcz_rad=None,
                    boundary_layer_temperature_K=state.temperature[1],
                    precipitation_rate=precip_rate,
                    soil_moisture=state.soil_moisture,
                    vegetation_fraction=state.vegetation_fraction,
                )
                # Add latent heat directly to diagonal and cross terms
                diag = diag + lh_diag
                if cross is not None:
                    cross = cross + lh_cross
                else:
                    cross = lh_cross

            # Vertical motion Jacobian (3-layer only)
            # Tendency: Q = ρ*cp*w*f*(θ_atm - θ_bl), symmetric for ascent/descent.
            # dT_bl = Q/C_bl, dT_atm = -Q/C_atm (energy conserving).
            # Note: ascent (w<0) gives positive BL diagonal (destabilizing),
            # but total diagonal remains negative from radiation + other terms.
            vertical_motion_enabled = (
                inputs.vertical_motion_cfg is not None
                and inputs.vertical_motion_cfg.enabled
                and nlayers == 3
            )
            if vertical_motion_enabled and state.boundary_layer_wind_field is not None:
                wind_u_bl_vm, wind_v_bl_vm, _ = state.boundary_layer_wind_field
                divergence = compute_divergence(
                    wind_u_bl_vm, wind_v_bl_vm, inputs.lat2d, inputs.lon2d
                )

                h_bl = BOUNDARY_LAYER_HEIGHT_M
                w = divergence * h_bl  # m/s, positive = subsidence

                rho = 1.0  # kg/m³
                cp = HEAT_CAPACITY_AIR_J_KG_K

                # Interpolation factor and potential temperature ratio
                P0 = 1013.25  # hPa
                P_ATM = 500.0  # hPa
                P_EXCHANGE = 850.0  # hPa
                KAPPA = 0.286
                f = (np.log(P0) - np.log(P_EXCHANGE)) / (np.log(P0) - np.log(P_ATM))
                alpha = (P0 / P_ATM) ** KAPPA  # ≈ 1.22

                coeff = rho * cp * w * f  # signed: positive for subsidence, negative for ascent

                C_bl = inputs.radiation_config.boundary_layer_heat_capacity
                C_atm = inputs.radiation_config.atmosphere_heat_capacity

                # BL: dT_bl = coeff*(α*T_atm - T_bl) / C_bl
                diag[1] += -coeff / C_bl
                cross[1, 2] += coeff * alpha / C_bl

                # Atm: dT_atm = -coeff*(α*T_atm - T_bl) / C_atm
                diag[2] += -coeff * alpha / C_atm
                cross[2, 1] += coeff / C_atm

            # Background BL-atmosphere mixing Jacobian
            # Stability-dependent tau (lagged): tau_eff = tau_base * (1 + (inv/T_SCALE)^2)
            tau_mix = (
                inputs.vertical_motion_cfg.tau_bl_atm_mixing_s
                if inputs.vertical_motion_cfg is not None
                else 0.0
            )
            if tau_mix > 0 and nlayers == 3:
                C_bl = inputs.radiation_config.boundary_layer_heat_capacity
                C_atm = inputs.radiation_config.atmosphere_heat_capacity
                theta_atm_jac = state.temperature[2] * _ALPHA
                tau = _bl_atm_mixing_tau(theta_atm_jac, state.temperature[1], tau_mix)
                # Full derivative of f(T_bl) = (θ-T_bl) / τ(θ-T_bl) w.r.t. T_bl:
                #   df/dT_bl = [(x/Ts)²-1] / [τ_base × (1+(x/Ts)²)²]
                # where x = max(θ-T_bl, 0).  This is negative (stabilizing)
                # for x < Ts and positive (destabilizing) for x > Ts.
                # Clamp to non-positive to maintain solver stability.
                x = np.maximum(theta_atm_jac - state.temperature[1], 0.0)
                x_norm_sq = (x / _BL_ATM_MIXING_T_SCALE) ** 2
                denom = tau_mix * (1.0 + x_norm_sq) ** 2
                df_dTbl = np.where(
                    x > 0,
                    np.minimum((x_norm_sq - 1.0) / denom, 0.0),
                    -1.0 / tau,
                )
                diag[1] += df_dTbl
                cross[1, 2] += _ALPHA / tau
                # dT_atm/dt = -(C_bl/C_atm) * f => Jacobian mirrors BL terms
                diag[2] += (C_bl / C_atm) * df_dTbl
                cross[2, 1] += (C_bl / C_atm) / tau

            # Use updated surface diffusion matrix if ocean ψ was applied
            actual_surface_diffusion_matrix = surface_matrix
            if inputs.diffusion_operator.surface.enabled and state.ocean_current_psi is not None:
                _, new_surface_offdiag = inputs.diffusion_operator.surface.linearised_tendency()
                if new_surface_offdiag is not None and new_surface_offdiag.nnz > 0:
                    actual_surface_diffusion_matrix = (
                        new_surface_offdiag
                        if sparse.isspmatrix_csc(new_surface_offdiag)
                        else new_surface_offdiag.tocsc()
                    )

            # =========================================================================
            # Humidity Jacobian components (for prognostic humidity in Newton solver)
            # =========================================================================
            humidity_advection_matrix = None
            humidity_diffusion_matrix = None
            humidity_diag = None
            humidity_temp_coupling = None
            temp_humidity_coupling = None

            # Soil moisture Jacobian defaults (set inside humidity block if available)
            soil_diag = None
            soil_humidity_coupling = None
            soil_temp_coupling = None
            temp_soil_coupling = None
            humidity_soil_coupling = None

            # Build humidity diffusion matrix if operator is provided
            # Use the full matrix (not just off-diagonal) for proper Jacobian
            if (
                inputs.humidity_diffusion_operator is not None
                and inputs.humidity_diffusion_operator.enabled
            ):
                full_matrix = inputs.humidity_diffusion_operator.matrix
                if full_matrix is not None:
                    humidity_diffusion_matrix = (
                        full_matrix if sparse.isspmatrix_csc(full_matrix) else full_matrix.tocsc()
                    )

            # Physical constants for humidity Jacobian
            L_v = LATENT_HEAT_VAPORIZATION_J_KG
            R_v = GAS_CONSTANT_WATER_VAPOR_J_KG_K

            if state.humidity_field is not None and inputs.advection_operator is not None:
                # Get wind field for humidity advection (prefer boundary layer wind)
                if nlayers == 3 and state.boundary_layer_wind_field is not None:
                    wind_u_q, wind_v_q, _ = state.boundary_layer_wind_field
                elif state.wind_field is not None:
                    wind_u_q, wind_v_q, _ = state.wind_field
                else:
                    wind_u_q, wind_v_q = None, None

                if wind_u_q is not None and wind_v_q is not None:
                    # Flux-form humidity advection matrix: -∇·(uq) for conservation
                    _, humidity_adv_mat = inputs.advection_operator.linearised_flux_tendency(
                        wind_u_q, wind_v_q
                    )
                    if humidity_adv_mat is not None:
                        humidity_advection_matrix = (
                            humidity_adv_mat
                            if sparse.isspmatrix_csc(humidity_adv_mat)
                            else humidity_adv_mat.tocsc()
                        )

                    # Compute E-P derivatives w.r.t. humidity
                    # dE/dq from latent heat model
                    if latent_heat_model is not None:
                        dE_dq = latent_heat_model.compute_evaporation_jacobian_wrt_humidity(
                            state.temperature[0],
                            state.temperature[-1],
                            state.humidity_field,
                            wind_speed_reference_m_s=state.wind_field[2]
                            if state.wind_field is not None
                            else None,
                            itcz_rad=None,
                            boundary_layer_temperature_K=state.temperature[1],
                            soil_moisture=state.soil_moisture,
                            vegetation_fraction=state.vegetation_fraction,
                        )
                    else:
                        dE_dq = np.zeros_like(state.humidity_field)

                    # dP/dq from precipitation model
                    t_bl = state.temperature[1]

                    # Compute vertical velocity from divergence + orographic (must match forward RHS)
                    if wind_u_q is not None and wind_v_q is not None:
                        divergence = compute_divergence(
                            wind_u_q, wind_v_q, inputs.lat2d, inputs.lon2d
                        )
                        w_largescale = compute_vertical_velocity_from_divergence(divergence)
                        if state.orographic_w is not None:
                            w_largescale = w_largescale + state.orographic_w
                        elif inputs.orographic_model is not None:
                            w_largescale = (
                                w_largescale
                                + inputs.orographic_model.compute_orographic_vertical_velocity(
                                    wind_u_q, wind_v_q
                                )
                            )
                    else:
                        w_largescale = np.zeros_like(state.humidity_field)

                    # Compute RH for precipitation Jacobian (rh_gate treated as frozen)
                    q_sat_jac = compute_saturation_specific_humidity(t_bl)
                    rh_jac = np.clip(state.humidity_field / np.maximum(q_sat_jac, 1e-10), 0.0, 1.0)

                    dP_dT_bl, dP_dT_atm, dP_dq = compute_precipitation_jacobian(
                        rh_jac,
                        state.humidity_field,
                        t_bl,
                        vertical_velocity=w_largescale,
                    )

                    # Orographic precipitation Jacobian: dP_oro/dq = rh_gate * η * max(w, 0) * ρ
                    if state.orographic_w is not None and inputs.orographic_model is not None:
                        eff = inputs.orographic_model.config.orographic_precip_efficiency
                        rho = 101325.0 / (287.05 * state.temperature[1])
                        rh_gate_oro = compute_precipitation_rh_gate(rh_jac)
                        dP_oro_dq = rh_gate_oro * eff * np.maximum(state.orographic_w, 0.0) * rho
                        dP_dq = dP_dq + dP_oro_dq

                    # Precipitation recycling Jacobian: dR/dq (always negative, stabilising)
                    if inputs.land_mask is not None:
                        grid_deg = abs(inputs.lat2d[1, 0] - inputs.lat2d[0, 0])
                        if state.boundary_layer_wind_field is not None:
                            ws_recycle = state.boundary_layer_wind_field[2]
                        elif state.wind_field is not None:
                            ws_recycle = state.wind_field[2]
                        else:
                            ws_recycle = np.full_like(state.humidity_field, 3.0)
                        # Approximate evaporation for recycling Jacobian
                        q_sat_sfc = compute_saturation_specific_humidity(state.temperature[0])
                        rho_sfc = 101325.0 / (287.05 * state.temperature[0])
                        ch_approx = np.where(inputs.land_mask, 1.0e-3, 1.2e-3)
                        evap_approx = (
                            rho_sfc
                            * ch_approx
                            * np.maximum(ws_recycle, 1.0)
                            * np.maximum(q_sat_sfc - state.humidity_field, 0.0)
                        )
                        if state.soil_moisture is not None:
                            manabe_beta = np.minimum(state.soil_moisture / 0.75, 1.0)
                            evap_approx = np.where(
                                inputs.land_mask, evap_approx * manabe_beta, evap_approx
                            )
                        dP_recycling_dq = compute_precipitation_recycling_jacobian(
                            evap_approx,
                            state.humidity_field,
                            ws_recycle,
                            inputs.land_mask,
                            resolution_deg=grid_deg,
                        )
                        dP_dq = dP_dq + dP_recycling_dq

                    # Humidity diagonal: dE/dq / M - dP/dq / M + Hadley subsidence
                    # (dE/dq is negative, dP/dq is positive)
                    # Clamp: total dP/dq must stay non-negative to keep humidity_diag
                    # strictly stabilising.  The recycling Jacobian can make dP/dq negative
                    # at dry cells where recycling dominates cloud precipitation.
                    dP_dq = np.maximum(dP_dq, 0.0)
                    humidity_diag = dE_dq / COLUMN_MASS_KG_M2 - dP_dq / COLUMN_MASS_KG_M2

                    # Hadley moisture redistribution Jacobian (descent + ascent)
                    if (
                        inputs.vertical_motion_cfg is not None
                        and inputs.vertical_motion_cfg.enabled
                        and inputs.vertical_motion_cfg.hadley_descent_velocity_m_s > 0
                    ):
                        lat_rad = np.deg2rad(inputs.lat2d)
                        w_hadley = compute_hadley_subsidence_velocity(
                            lat_rad,
                            itcz_rad,
                            peak_velocity_m_s=inputs.vertical_motion_cfg.hadley_descent_velocity_m_s,
                        )
                        humidity_diag += hadley_moisture_tendency_jacobian(
                            w_hadley,
                            humidity_field,
                            upper_troposphere_q_fraction=inputs.vertical_motion_cfg.upper_troposphere_q_fraction,
                            lat_rad=lat_rad,
                        )

                    # Humidity-temperature coupling terms already computed above

                    # Humidity-temperature coupling via Clausius-Clapeyron
                    if latent_heat_model is not None and latent_heat_model.enabled:
                        T_sfc = state.temperature[0]
                        dln_qsat_dT = L_v / (R_v * T_sfc * T_sfc)
                        q_sat = compute_saturation_specific_humidity(T_sfc)
                        dq_sat_dT_sfc = q_sat * dln_qsat_dT
                        bulk_coef = 0.006  # kg/m²/s per (kg/kg) humidity deficit
                        dE_dT_sfc = bulk_coef * dq_sat_dT_sfc
                    else:
                        dE_dT_sfc = np.zeros_like(state.humidity_field)

                    # dR_q/dT terms (to be multiplied by -dt in solver)
                    dR_q_dT_sfc = dE_dT_sfc / COLUMN_MASS_KG_M2
                    dR_q_dT_bl = -dP_dT_bl / COLUMN_MASS_KG_M2
                    dR_q_dT_atm = -dP_dT_atm / COLUMN_MASS_KG_M2

                    humidity_temp_coupling = (dR_q_dT_sfc, dR_q_dT_bl, dR_q_dT_atm)

                    # Temperature-humidity coupling via latent heat release
                    C_sfc = inputs.heat_capacity_field
                    C_atm = inputs.radiation_config.atmosphere_heat_capacity
                    C_bl = inputs.radiation_config.boundary_layer_heat_capacity

                    # Evaporation Jacobian: surface cools when q increases (less evap)
                    dR_Tsfc_dq = -dE_dq * L_v / C_sfc

                    # Precipitation heating split: ocean 20% / land 5% to BL
                    bl_frac = np.where(inputs.land_mask, 0.05, 0.20)
                    dR_Tatm_dq = (1 - bl_frac) * dP_dq * L_v / C_atm
                    dR_Tbl_dq = bl_frac * dP_dq * L_v / C_bl

                    temp_humidity_coupling = (dR_Tsfc_dq, dR_Tbl_dq, dR_Tatm_dq)

                    # Precipitation latent heating feedback on temperature:
                    # P depends on T through dP/dT_bl and dP/dT_atm.
                    # When T_bl warms → P increases → latent heat warms BL and atm.
                    # This positive feedback is the dominant missing Jacobian term.
                    diag[1] += bl_frac * dP_dT_bl * L_v / C_bl
                    diag[2] += (1 - bl_frac) * dP_dT_atm * L_v / C_atm
                    # Cross-layer: BL P-heating from atm T change and vice versa
                    cross[1, 2] += bl_frac * dP_dT_atm * L_v / C_bl
                    cross[2, 1] += (1 - bl_frac) * dP_dT_bl * L_v / C_atm

                    # =============================================================
                    # Soil moisture Jacobian (for prognostic SM in Newton solver)
                    # =============================================================
                    SOIL_CAPACITY_KG_M2 = 300.0
                    TAU_SLOW_SECONDS = 365.0 * 86400.0

                    if (
                        latent_heat_model is not None
                        and latent_heat_model.enabled
                        and state.soil_moisture is not None
                    ):
                        # Compute dE/dSM from latent heat model
                        dE_dSM = latent_heat_model.compute_evaporation_jacobian_wrt_soil_moisture(
                            state.temperature[0],
                            state.temperature[-1],
                            state.humidity_field,
                            wind_speed_reference_m_s=state.wind_field[2]
                            if state.wind_field is not None
                            else None,
                            itcz_rad=None,
                            boundary_layer_temperature_K=state.temperature[1],
                            soil_moisture=state.soil_moisture,
                            vegetation_fraction=state.vegetation_fraction,
                        )

                        # SM tendency: dSM/dt = (P - E) / capacity - SM / tau_slow
                        # Tendency derivatives stored in Linearization (solver multiplies by -dt)

                        # d(Tsfc_tendency)/dSM = -dE_dSM * L_v / C_sfc
                        # (negative: more SM -> more E -> surface cools)
                        dR_Tsfc_dSM = -dE_dSM * L_v / C_sfc

                        # d(q_tendency)/dSM = +dE_dSM / COLUMN_MASS
                        # (positive: more SM -> more E -> more q)
                        dR_q_dSM = dE_dSM / COLUMN_MASS_KG_M2

                        # Land mask for SM coupling (SM is land-only)
                        land = inputs.land_mask

                        # d(SM_tendency)/dTsfc = -dE_dT_sfc / CAPACITY
                        # (negative: warmer surface -> more E -> less SM)
                        dR_SM_dTsfc = np.where(land, -dE_dT_sfc / SOIL_CAPACITY_KG_M2, 0.0)

                        # d(SM_tendency)/dTbl = +dP_dT_bl / CAPACITY
                        # (positive: warmer BL -> more P -> more SM)
                        dR_SM_dTbl = np.where(land, dP_dT_bl / SOIL_CAPACITY_KG_M2, 0.0)

                        # d(SM_tendency)/dTatm = +dP_dT_atm / CAPACITY
                        dR_SM_dTatm = np.where(land, dP_dT_atm / SOIL_CAPACITY_KG_M2, 0.0)

                        # d(SM_tendency)/dq = +dP_dq / CAPACITY
                        # (positive: more q -> more P -> more SM)
                        dR_SM_dq = np.where(land, dP_dq / SOIL_CAPACITY_KG_M2, 0.0)

                        # d(SM_tendency)/dSM = -dE_dSM / CAPACITY - 1/tau_slow
                        # Ocean cells: only the slow drainage term (identity + small offset)
                        # since dE_dSM is already zero over ocean from the latent heat model
                        soil_diag = np.where(
                            land,
                            -dE_dSM / SOIL_CAPACITY_KG_M2 - 1.0 / TAU_SLOW_SECONDS,
                            -1.0 / TAU_SLOW_SECONDS,
                        )

                        soil_humidity_coupling = dR_SM_dq
                        soil_temp_coupling = (dR_SM_dTsfc, dR_SM_dTbl, dR_SM_dTatm)
                        temp_soil_coupling = (
                            dR_Tsfc_dSM,
                            np.zeros_like(dR_Tsfc_dSM),
                            np.zeros_like(dR_Tsfc_dSM),
                        )
                        humidity_soil_coupling = dR_q_dSM

            return Linearization(
                diag=diag,
                cross=cross,
                surface_diffusion_matrix=actual_surface_diffusion_matrix,
                surface_advection_matrix=surface_advection_matrix,
                atmosphere_diffusion_matrix=atmosphere_matrix,
                atmosphere_advection_matrix=atmosphere_advection_matrix,
                boundary_layer_diffusion_matrix=boundary_layer_matrix,
                boundary_layer_advection_matrix=boundary_layer_advection_matrix,
                humidity_advection_matrix=humidity_advection_matrix,
                humidity_diffusion_matrix=humidity_diffusion_matrix,
                humidity_diag=humidity_diag,
                humidity_temp_coupling=humidity_temp_coupling,
                temp_humidity_coupling=temp_humidity_coupling,
                soil_diag=soil_diag,
                soil_humidity_coupling=soil_humidity_coupling,
                soil_temp_coupling=soil_temp_coupling,
                temp_soil_coupling=temp_soil_coupling,
                humidity_soil_coupling=humidity_soil_coupling,
            )
        # No atmosphere case - cloud_output not used
        radiative_derivative = radiation.radiative_balance_rhs_temperature_derivative(
            state.temperature,
            heat_capacity_field=inputs.heat_capacity_field,
            config=inputs.radiation_config,
            land_mask=inputs.land_mask,
            humidity_q=state.humidity_field,
            lat2d=inputs.lat2d,
            itcz_rad=itcz_rad,
            cloud_output=None,
        )
        assert isinstance(radiative_derivative, np.ndarray)
        diag = radiative_derivative.copy()
        if surface_diffusion_diag is not None:
            diag = diag + surface_diffusion_diag
        return Linearization(diag=diag, surface_diffusion_matrix=surface_matrix)

    return rhs, rhs_derivative
