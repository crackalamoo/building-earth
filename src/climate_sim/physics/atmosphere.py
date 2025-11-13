"""Atmospheric utility functions."""

from __future__ import annotations

import numpy as np

STANDARD_LAPSE_RATE_K_PER_M = 6.5 / 1000.0


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
    atmosphere_c: np.ndarray | None,
    surface_c: np.ndarray,
) -> np.ndarray:
    """Compute 2 m air temperature from available layers.

    Behavior matches the solver's previous inline logic:
    - If an atmosphere layer exists, start from it (in °C) and adjust from the
      reference atmosphere height down to 2 m using a fixed lapse rate. If
      ``include_lapse_rate_elevation`` is True, add topographic elevation to the
      vertical delta (moving further downward to the local surface).
    - If no atmosphere layer is provided, fall back to the surface temperature.

    Parameters
    ----------
    atmosphere_c:
        Monthly cycle of atmosphere temperature in °C, or None if unavailable.
    surface_c:
        Monthly cycle of surface temperature in °C.

    Returns
    -------
    numpy.ndarray
        Monthly cycle of 2 m temperature in °C.
    """
    if atmosphere_c is None:
        return surface_c.copy()
    Ks = 6
    Ke = 0.4
    surface_K = surface_c + 273.15
    atmosphere_K = atmosphere_c + 273.15
    Tbl = (Ks*surface_K + Ke*atmosphere_K) / (Ks + Ke) - 273.15
    return Tbl

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
