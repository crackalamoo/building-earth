"""Humidity utilities."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from climate_sim.data.constants import OCEAN_CLOUD_COVER_BOOST
from climate_sim.physics.pressure import pressure_from_temperature_elevation

RH_POLES = 0.85
RH_SUBTROPICS= 0.35
RH_ITCZ = 0.80
RH_POLES_WATER = 0.90
RH_SUBTROPICS_WATER = 0.63
RH_ITCZ_WATER = 0.88
STORM_BAND = np.deg2rad([30, 60])
DELTA_SUBTROPICS = np.deg2rad(20)
LAT_POLES = np.deg2rad(70)


def _guess_humidity_rh_function(itcz: np.ndarray,
    rh_poles=RH_POLES, rh_subtropics=RH_SUBTROPICS, rh_itcz=RH_ITCZ) -> PchipInterpolator:
    phi = np.array([
        -LAT_POLES,
        itcz - DELTA_SUBTROPICS,
        itcz,
        itcz + DELTA_SUBTROPICS,
        LAT_POLES
    ])

    target_rh = np.array([
        rh_poles,
        rh_subtropics,
        rh_itcz,
        rh_subtropics,
        rh_poles
    ])

    interp = PchipInterpolator(phi, target_rh)

    return interp

def compute_humidity_q(
    lat_2d: np.ndarray,
    temperature: np.ndarray,
    declination_rad: np.ndarray,
    *,
    return_rh: bool = False,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute humidity field (specific humidity or relative humidity).
    
    Parameters
    ----------
    lat_2d : np.ndarray
        Latitude field in degrees.
    temperature : np.ndarray
        Temperature field in Kelvin.
    declination_rad : np.ndarray
        Solar declination angle in radians.
    return_rh : bool, optional
        If True, return relative humidity (0-1). If False, return specific humidity (kg/kg).
        Default is False.
    land_mask : np.ndarray | None, optional
        Boolean array indicating land cells (True) vs ocean cells (False).
    
    Returns
    -------
    np.ndarray
        Humidity field. If return_rh is False, returns specific humidity in kg/kg.
        If return_rh is True, returns relative humidity as a fraction (0-1).
    """
    # Convert latitude to radians for internal computation
    lat_2d_rad = np.deg2rad(lat_2d)
    
    itcz = declination_rad * 0.7
    rh_function = _guess_humidity_rh_function(itcz)
    rh = rh_function(lat_2d_rad)

    # Apply ocean boost if land mask is provided
    if land_mask is not None:
        ocean_mask = ~land_mask
        rh_function_ocean = _guess_humidity_rh_function(declination_rad * 0.3,
        RH_POLES_WATER, RH_SUBTROPICS_WATER, RH_ITCZ_WATER)
        rh_ocean = rh_function_ocean(lat_2d_rad)
        rh = np.where(ocean_mask, rh_ocean, rh)

    lat_rel = lat_2d_rad - itcz
    storm_bump = 0.5 * (1 - np.cos(2*np.pi * (lat_rel - STORM_BAND[0])/(STORM_BAND[1] - STORM_BAND[0])))
    storm_bump = np.where(
        ((STORM_BAND[0] < lat_rel) & (lat_rel < STORM_BAND[1]))
        | ((-STORM_BAND[1] < lat_rel) & (lat_rel < -STORM_BAND[0])),
        storm_bump, 0)
    storm_A = 0.2
    rh = rh + storm_A * (1-rh) * storm_bump

    rh = np.clip(rh, 0.1, 0.98)

    if return_rh:
        return rh

    p_guess = pressure_from_temperature_elevation(temperature)
    e_sat = 6.112 * np.exp(17.67 * temperature / (temperature + 243.5))
    q_sat = (0.622 * e_sat)/(p_guess - (1-0.622)*e_sat)

    return q_sat * rh

def compute_cloud_cover(temperature: np.ndarray, land_mask: np.ndarray | None = None) -> np.ndarray:
    """Compute latitude-dependent cloud cover fraction.
    
    Uses a simple parameterization based on latitude only:
        C = 0.65 - 2.59 * sin²(lat) + 3.55 * sin⁴(lat)
    
    The result is clamped to a maximum of 0.9, then boosted over ocean cells
    by OCEAN_CLOUD_COVER_BOOST if a land mask is provided.
    
    Parameters
    ----------
    temperature : np.ndarray
        Temperature field (K) used to infer grid shape. The temperature values
        themselves are not used in the computation; only the shape is needed
        to reconstruct the latitude grid.
    land_mask : np.ndarray | None, optional
        Boolean array indicating land cells (True) vs ocean cells (False).
        If provided, cloud cover is boosted over ocean cells by
        OCEAN_CLOUD_COVER_BOOST. If None, no boost is applied.
    
    Returns
    -------
    np.ndarray
        Cloud cover fraction (0-1) at each grid point, with the same shape
        as the innermost spatial dimensions of the temperature field.
    
    Notes
    -----
    The latitude grid is derived from the temperature field shape assuming
    a regular lat/lon grid with cell centers at the midpoints.
    """
    # Extract the spatial shape from the temperature field
    if temperature.ndim == 2:
        nlat, nlon = temperature.shape
    elif temperature.ndim == 3:
        nlat, nlon = temperature.shape[1], temperature.shape[2]
    else:
        raise ValueError(
            f"Temperature field must be 2D or 3D, got shape {temperature.shape}"
        )

    # Reconstruct the latitude grid (cell centers)
    lat_spacing = 180.0 / float(nlat)
    lat_centres = -90.0 + (np.arange(nlat, dtype=float) + 0.5) * lat_spacing

    # Create 2D latitude field matching the grid
    lat2d = lat_centres[:, np.newaxis]  # Shape: (nlat, 1)
    lat2d = np.broadcast_to(lat2d, (nlat, nlon))  # Shape: (nlat, nlon)

    # Convert to radians for trigonometric functions
    lat_rad = np.deg2rad(lat2d)

    # Compute the cloud cover formula
    sin_lat = np.sin(lat_rad)
    sin2_lat = sin_lat * sin_lat
    sin4_lat = sin2_lat * sin2_lat

    cloud_cover = 0.7 - 2.59 * sin2_lat + 3.55 * sin4_lat

    # Clamp to maximum of 0.9
    cloud_cover = np.minimum(cloud_cover, 0.9)

    # Apply ocean boost if land mask is provided
    if land_mask is not None:
        ocean_mask = ~land_mask
        cloud_cover = np.where(ocean_mask, cloud_cover * OCEAN_CLOUD_COVER_BOOST, cloud_cover)

    return cloud_cover

