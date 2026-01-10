"""Post-processing utilities for solver results."""

from __future__ import annotations

from typing import Dict

import numpy as np

from climate_sim.physics.atmosphere.atmosphere import compute_two_meter_temperature
from climate_sim.physics.atmosphere.wind import WindConfig, WindModel
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.solar import compute_monthly_declinations
from climate_sim.core.state import ModelState, select_wind_temperature


def postprocess_periodic_cycle_results(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    monthly_states: list[ModelState],
    *,
    resolved_wind: WindConfig | None,
    resolved_radiation: RadiationConfig,
    return_layer_map: bool = False,
    topographic_elevation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Post-process solver results: convert units, compute diagnostics, build output dictionaries.

    Parameters
    ----------
    lon2d, lat2d
        Longitude and latitude grids
    monthly_states
        List of 12 ModelState objects (one per month)
    resolved_wind
        Wind configuration (used for geostrophic wind computation)
    resolved_radiation
        Radiation configuration (used to check if atmosphere is enabled)
    return_layer_map
        If True, return a dictionary of all layers; if False, return only surface temperature
    topographic_elevation
        Elevation of each cell in metres. Used for lapse rate correction in 2m temperature.

    Returns
    -------
    tuple
        Either (lon2d, lat2d, surface_temp_C) or (lon2d, lat2d, layers_map) depending on
        return_layer_map flag
    """
    monthly_T = np.array([state.temperature for state in monthly_states])
    temperature_2m_c: np.ndarray | None = None
    if monthly_T.ndim == 3:
        # Single-layer (no atmosphere)
        monthly_surface_K = monthly_T
        layers_map = {"surface": monthly_surface_K - 273.15}
    elif monthly_T.shape[1] == 2:
        # Two-layer (surface + atmosphere)
        monthly_surface_K = monthly_T[:, 0]
        monthly_atmosphere_K = monthly_T[:, 1]
        temperature_2m_c = compute_two_meter_temperature(
            None,
            monthly_surface_K,
            topographic_elevation=topographic_elevation,
        ) - 273.15
        layers_map = {
            "surface": monthly_surface_K - 273.15,
            "atmosphere": monthly_atmosphere_K - 273.15,
        }
    elif monthly_T.shape[1] == 3:
        # Three-layer (surface + boundary layer + atmosphere)
        monthly_surface_K = monthly_T[:, 0]
        monthly_boundary_K = monthly_T[:, 1]
        monthly_atmosphere_K = monthly_T[:, 2]
        # Compute 2m temperature using the standard function with lapse rate correction
        temperature_2m_K = compute_two_meter_temperature(
            monthly_boundary_K,
            monthly_surface_K,
            topographic_elevation=topographic_elevation,
        )
        temperature_2m_c = temperature_2m_K - 273.15
        layers_map = {
            "surface": monthly_surface_K - 273.15,
            "boundary_layer": monthly_boundary_K - 273.15,
            "atmosphere": monthly_atmosphere_K - 273.15,
        }
    else:
        raise ValueError(f"Unexpected temperature shape: {monthly_T.shape}")

    if temperature_2m_c is None:
        temperature_2m_c = layers_map["surface"].copy()

    layers_map["temperature_2m"] = temperature_2m_c

    layers_map["albedo"] = np.array([state.albedo_field for state in monthly_states])

    wind_fields = [state.wind_field for state in monthly_states]
    if all(wind is not None for wind in wind_fields):
        wind_u = np.stack([wind[0] for wind in wind_fields if wind is not None], axis=0)
        wind_v = np.stack([wind[1] for wind in wind_fields if wind is not None], axis=0)
        wind_speed = np.stack([wind[2] for wind in wind_fields if wind is not None], axis=0)
        layers_map["wind_u"] = wind_u
        layers_map["wind_v"] = wind_v
        layers_map["wind_speed"] = wind_speed
        
        # Compute geostrophic wind fields
        if resolved_wind is not None and resolved_wind.enabled and resolved_radiation.include_atmosphere:
            geostrophic_fields = []
            monthly_declinations = compute_monthly_declinations()
            wind_model_temp = WindModel(lon2d, lat2d, config=resolved_wind)
            for idx, state in enumerate(monthly_states):
                wind_temperature = select_wind_temperature(state.temperature)
                geo_field = wind_model_temp.wind_field(
                    wind_temperature, 
                    declination_rad=monthly_declinations[idx], 
                    ekman_drag=False
                )
                geostrophic_fields.append(geo_field)
            
            geostrophic_u = np.stack([geo[0] for geo in geostrophic_fields], axis=0)
            geostrophic_v = np.stack([geo[1] for geo in geostrophic_fields], axis=0)
            geostrophic_speed = np.stack([geo[2] for geo in geostrophic_fields], axis=0)
            layers_map["wind_u_geostrophic"] = geostrophic_u
            layers_map["wind_v_geostrophic"] = geostrophic_v
            layers_map["wind_speed_geostrophic"] = geostrophic_speed

    humidity_fields = [state.humidity_field for state in monthly_states]
    if all(humidity is not None for humidity in humidity_fields):
        humidity_q = np.stack([humidity for humidity in humidity_fields if humidity is not None], axis=0)
        layers_map["humidity"] = humidity_q

    if return_layer_map:
        return lon2d, lat2d, layers_map

    return lon2d, lat2d, layers_map["surface"]
