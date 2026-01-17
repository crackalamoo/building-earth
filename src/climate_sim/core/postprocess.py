"""Post-processing utilities for solver results."""

from __future__ import annotations

from typing import Dict

import numpy as np

from climate_sim.physics.atmosphere.atmosphere import compute_two_meter_temperature, log_law_map_wind_speed
from climate_sim.physics.atmosphere.wind import WindConfig, WindModel
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.core.state import ModelState, select_wind_temperature
from climate_sim.data.landmask import compute_land_mask
from climate_sim.data.elevation import compute_cell_roughness_length, load_elevation_data


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

    # Extract wind fields from state
    # For 3-layer: state.wind_field = geostrophic (free atmosphere), state.boundary_layer_wind_field = Ekman (BL)
    # For 2-layer: state.wind_field = Ekman (atmosphere), state.boundary_layer_wind_field = None
    wind_fields_geostrophic = [state.wind_field for state in monthly_states]
    wind_fields_ekman = []
    for state in monthly_states:
        # Use boundary layer wind if available (3-layer), otherwise use atmosphere wind (2-layer)
        if state.boundary_layer_wind_field is not None:
            wind_fields_ekman.append(state.boundary_layer_wind_field)
        else:
            wind_fields_ekman.append(state.wind_field)

    if all(wind is not None for wind in wind_fields_geostrophic):
        # Geostrophic wind (free atmosphere, no drag)
        geostrophic_u = np.stack([wind[0] for wind in wind_fields_geostrophic if wind is not None], axis=0)
        geostrophic_v = np.stack([wind[1] for wind in wind_fields_geostrophic if wind is not None], axis=0)
        geostrophic_speed = np.stack([wind[2] for wind in wind_fields_geostrophic if wind is not None], axis=0)
        layers_map["wind_u_geostrophic"] = geostrophic_u
        layers_map["wind_v_geostrophic"] = geostrophic_v
        layers_map["wind_speed_geostrophic"] = geostrophic_speed

    if all(wind is not None for wind in wind_fields_ekman):
        # Ekman wind (boundary layer with drag, or atmosphere for 2-layer)
        ekman_u = np.stack([wind[0] for wind in wind_fields_ekman if wind is not None], axis=0)
        ekman_v = np.stack([wind[1] for wind in wind_fields_ekman if wind is not None], axis=0)
        ekman_speed = np.stack([wind[2] for wind in wind_fields_ekman if wind is not None], axis=0)
        layers_map["wind_u"] = ekman_u
        layers_map["wind_v"] = ekman_v
        layers_map["wind_speed"] = ekman_speed

        # Compute 10m wind from Ekman wind using log law
        land_mask = compute_land_mask(lon2d, lat2d)
        elevation_data = load_elevation_data()
        roughness_length = compute_cell_roughness_length(lon2d, lat2d, land_mask=land_mask, data=elevation_data)

        # Reference height: Ekman boundary layer effective height (200m land, 500m ocean)
        height_ref_m = np.where(land_mask, 200.0, 500.0)

        wind_10m_fields = []
        for ekman_wind in wind_fields_ekman:
            if ekman_wind is not None:
                ekman_speed_2d = ekman_wind[2]
                wind_10m_speed = log_law_map_wind_speed(
                    ekman_speed_2d,
                    height_ref_m=height_ref_m,
                    height_target_m=10.0,
                    roughness_length_m=roughness_length,
                )
                # Scale u and v components by the same ratio
                scale_factor = np.zeros_like(ekman_speed_2d)
                mask_nonzero = ekman_speed_2d > 1.0e-6
                scale_factor[mask_nonzero] = wind_10m_speed[mask_nonzero] / ekman_speed_2d[mask_nonzero]

                wind_10m_u = ekman_wind[0] * scale_factor
                wind_10m_v = ekman_wind[1] * scale_factor
                wind_10m_fields.append((wind_10m_u, wind_10m_v, wind_10m_speed))
            else:
                wind_10m_fields.append(None)

        if all(wind is not None for wind in wind_10m_fields):
            wind_10m_u = np.stack([wind[0] for wind in wind_10m_fields if wind is not None], axis=0)
            wind_10m_v = np.stack([wind[1] for wind in wind_10m_fields if wind is not None], axis=0)
            wind_10m_speed = np.stack([wind[2] for wind in wind_10m_fields if wind is not None], axis=0)
            layers_map["wind_u_10m"] = wind_10m_u
            layers_map["wind_v_10m"] = wind_10m_v
            layers_map["wind_speed_10m"] = wind_10m_speed

    humidity_fields = [state.humidity_field for state in monthly_states]
    if all(humidity is not None for humidity in humidity_fields):
        humidity_q = np.stack([humidity for humidity in humidity_fields if humidity is not None], axis=0)
        layers_map["humidity"] = humidity_q

    # Extract precipitation fields from state
    precipitation_fields = [state.precipitation_field for state in monthly_states]
    if all(precip is not None for precip in precipitation_fields):
        precipitation = np.stack([precip for precip in precipitation_fields if precip is not None], axis=0)
        layers_map["precipitation"] = precipitation

    # Extract ocean current fields from state
    ocean_current_fields = [state.ocean_current_field for state in monthly_states]
    if all(current is not None for current in ocean_current_fields):
        ocean_u = np.stack([current[0] for current in ocean_current_fields if current is not None], axis=0)
        ocean_v = np.stack([current[1] for current in ocean_current_fields if current is not None], axis=0)
        layers_map["ocean_u"] = ocean_u
        layers_map["ocean_v"] = ocean_v

    if return_layer_map:
        return lon2d, lat2d, layers_map

    return lon2d, lat2d, layers_map["surface"]
