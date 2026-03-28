"""Atmospheric utility functions."""

from __future__ import annotations

import numpy as np

from climate_sim.data.constants import BOUNDARY_LAYER_HEIGHT_M
from climate_sim.data.constants import STANDARD_LAPSE_RATE_K_PER_M


def adjust_temperature_by_elevation(
    temperature_c: np.ndarray,
    elevation_delta_m: np.ndarray | float,
) -> np.ndarray:
    """Apply a fixed lapse-rate correction for an elevation change.

    Parameters
    ----------
    temperature_c:
        Base temperature field expressed in degrees Celsius.
    elevation_delta_m:
        Elevation difference in metres (positive for moving upward). Can be a
        scalar or array broadcastable to ``temperature_c``.

    Returns
    -------
    numpy.ndarray
        Temperature field adjusted for the specified elevation change.
    """

    temp = np.asarray(temperature_c, dtype=float)
    delta = np.asarray(elevation_delta_m, dtype=float)
    adjusted = temp - STANDARD_LAPSE_RATE_K_PER_M * delta
    return adjusted.astype(float, copy=False)


def compute_two_meter_temperature(
    boundary_layer_K: np.ndarray | None,
    surface_K: np.ndarray,
    topographic_elevation: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 2 m air temperature from available layers.

    Parameters
    ----------
    boundary_layer_K:
        Boundary layer temperature in K (represents mid-layer ~375m), or None if unavailable.
    surface_K:
        Surface temperature in K.
    topographic_elevation:
        Elevation of each cell in metres. Used for lapse rate correction when
        boundary_layer_K is not None. Can be a scalar or array broadcastable to surface_K.

    Returns
    -------
    numpy.ndarray
        2 m temperature in K.
    """
    if boundary_layer_K is not None:
        mid_layer_height_m = BOUNDARY_LAYER_HEIGHT_M / 2.0  # 375m
        height_difference_m = mid_layer_height_m - 2.0  # 373m
        lapse_correction_K = STANDARD_LAPSE_RATE_K_PER_M * height_difference_m
        result = boundary_layer_K + lapse_correction_K

        # Apply elevation adjustment if available
        if topographic_elevation is not None:
            elevation_delta_m = topographic_elevation - 2.0
            result = adjust_temperature_by_elevation(result, elevation_delta_m)

        return result

    return surface_K.copy()


def log_law_map_wind_speed(
    wind_speed_ref_m_s: np.ndarray,
    height_ref_m: np.ndarray | float,
    height_target_m: np.ndarray | float,
    roughness_length_m: np.ndarray,
) -> np.ndarray:
    """Map wind speeds between two heights using the logarithmic wind profile law.

    Parameters
    ----------
    wind_speed_ref_m_s:
        Wind speed at the reference height in m/s.
    height_ref_m:
        Reference height in metres.
    height_target_m:
        Target height in metres.
    roughness_length_m:
        Surface roughness length in metres.
    Returns
    -------
    numpy.ndarray
        Wind speed at the target height in m/s.
    """
    u_ref = wind_speed_ref_m_s
    z_ref = height_ref_m
    z_target = height_target_m
    z0 = roughness_length_m

    log_ref = np.log(z_ref / z0)
    log_target = np.log(z_target / z0)

    u_target = u_ref * (log_target / log_ref)
    return u_target.astype(float, copy=False)
