"""Humidity utilities."""

from __future__ import annotations

import numpy as np

from climate_sim.physics.atmosphere.pressure import compute_pressure
from climate_sim.physics.atmosphere.hadley import LAT_POLES, LAT_SUBPOLAR, DELTA_SUBTROPICS, compute_itcz_latitude
from climate_sim.core.math_core import spherical_cell_area
from climate_sim.core.timing import time_block
from climate_sim.data.constants import R_EARTH_METERS

# Mean RH and anomalies at key latitude bands (like pressure anomalies)
RH_MEAN = 0.65
DRH_ITCZ = +0.15           # Humid at ITCZ (convergence, rising air)
DRH_SUBTROPICS = -0.30     # Dry at subtropics (descending air, deserts)
DRH_SUBPOLAR = +0.10       # Moderately humid (storm tracks)
DRH_POLES = +0.20          # Humid at poles

# Ocean has higher RH overall
RH_MEAN_WATER = 0.75
DRH_ITCZ_WATER = +0.13
DRH_SUBTROPICS_WATER = -0.12
DRH_SUBPOLAR_WATER = +0.05
DRH_POLES_WATER = +0.15

# Width of humidity features (radians)
SIGMA_RH_ITCZ = np.deg2rad(10.0)       # ITCZ humid zone width
SIGMA_RH_SUBTROPICS = np.deg2rad(15.0)  # Subtropical dry zone width
SIGMA_RH_SUBPOLAR = np.deg2rad(12.0)   # Subpolar storm track width
SIGMA_RH_POLES = np.deg2rad(10.0)      # Polar humid zone width


def specific_humidity_to_relative_humidity(
    q: np.ndarray,
    temperature_K: np.ndarray,
    itcz_rad: np.ndarray | None = None,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
) -> np.ndarray:
    """Convert specific humidity to relative humidity.

    Parameters
    ----------
    q : np.ndarray
        Specific humidity (kg/kg).
    temperature_K : np.ndarray
        Temperature field in Kelvin.
    itcz_rad : np.ndarray | None
        ITCZ latitude in radians (optional).
    lat2d : np.ndarray | None
        Latitude field in degrees.
    lon2d : np.ndarray | None
        Longitude field in degrees.

    Returns
    -------
    np.ndarray
        Relative humidity as a fraction (0-1).
    """
    # Compute saturation vapor pressure (Magnus formula)
    # Magnus formula requires temperature in Celsius
    temperature_C = temperature_K - 273.15

    p_Pa = compute_pressure(temperature_K, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)
    p_hPa = p_Pa / 100.0  # Convert Pa to hPa

    # Magnus formula: e_sat in hPa
    e_sat = 6.112 * np.exp(17.67 * temperature_C / (temperature_C + 243.5))
    q_sat = (0.622 * e_sat) / (p_hPa - (1 - 0.622) * e_sat)

    # Compute RH = q / q_sat, clamped to valid range
    rh = q / np.maximum(q_sat, 1e-10)
    return np.clip(rh, 0.0, 1.0)


def relative_humidity_pattern(
    lat_rad: np.ndarray,
    itcz_rad: np.ndarray,
    rh_mean: float = RH_MEAN,
    drh_itcz: float = DRH_ITCZ,
    drh_subtropics: float = DRH_SUBTROPICS,
    drh_subpolar: float = DRH_SUBPOLAR,
    drh_poles: float = DRH_POLES,
) -> np.ndarray:
    """Compute relative humidity pattern using anomalies from mean.

    Like pressure, RH is computed as mean + sum of Gaussian anomalies:
    - ITCZ: positive anomaly (humid, rising air)
    - Subtropics: negative anomaly (dry, descending air)
    - Subpolar: positive anomaly (storm tracks)
    - Poles: positive anomaly (humid)

    Parameters
    ----------
    lat_rad : np.ndarray
        Latitude field in radians, shape (nlat, nlon).
    itcz_rad : np.ndarray
        ITCZ latitude in radians, shape (nlon,) or broadcast-compatible.
    rh_mean : float
        Mean relative humidity.
    drh_itcz, drh_subtropics, drh_subpolar, drh_poles : float
        RH anomalies at key latitude bands.

    Returns
    -------
    np.ndarray
        Relative humidity (0-1), same shape as lat_rad.
    """
    # Subtropical dry zones follow ITCZ (descending branch of Hadley cell)
    lat_subtrop_south = 0.5 * itcz_rad - DELTA_SUBTROPICS
    lat_subtrop_north = 0.5 * itcz_rad + DELTA_SUBTROPICS

    # ITCZ humid anomaly
    rh_itcz = drh_itcz * np.exp(-((lat_rad - itcz_rad) / SIGMA_RH_ITCZ) ** 2)

    # Subtropical dry anomaly (both hemispheres)
    rh_subtrop = drh_subtropics * (
        np.exp(-((lat_rad - lat_subtrop_south) / SIGMA_RH_SUBTROPICS) ** 2)
        + np.exp(-((lat_rad - lat_subtrop_north) / SIGMA_RH_SUBTROPICS) ** 2)
    )

    # Subpolar humid anomaly (fixed latitudes, both hemispheres)
    rh_subpolar = drh_subpolar * (
        np.exp(-((lat_rad + LAT_SUBPOLAR) / SIGMA_RH_SUBPOLAR) ** 2)
        + np.exp(-((lat_rad - LAT_SUBPOLAR) / SIGMA_RH_SUBPOLAR) ** 2)
    )

    # Polar humid anomaly (fixed latitudes, both hemispheres)
    rh_polar = drh_poles * (
        np.exp(-((lat_rad + LAT_POLES) / SIGMA_RH_POLES) ** 2)
        + np.exp(-((lat_rad - LAT_POLES) / SIGMA_RH_POLES) ** 2)
    )

    rh = rh_mean + rh_itcz + rh_subtrop + rh_subpolar + rh_polar

    return np.clip(rh, 0.1, 0.98)

def compute_humidity_q(
    lat_2d: np.ndarray,
    temperature: np.ndarray,
    *,
    return_rh: bool = False,
    land_mask: np.ndarray | None = None,
    lon_2d: np.ndarray | None = None,
    itcz_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Compute humidity field (specific humidity or relative humidity).

    Parameters
    ----------
    lat_2d : np.ndarray
        Latitude field in degrees.
    temperature : np.ndarray
        Temperature field in Kelvin.
    return_rh : bool, optional
        If True, return relative humidity (0-1). If False, return specific humidity (kg/kg).
        Default is False.
    land_mask : np.ndarray | None, optional
        Boolean array indicating land cells (True) vs ocean cells (False).
    lon_2d : np.ndarray | None
        Longitude field in degrees. Used to compute ITCZ if itcz_rad not provided.
    itcz_rad : np.ndarray | None
        Pre-computed ITCZ latitude in radians, shape (nlon,).

    Returns
    -------
    np.ndarray
        Humidity field. If return_rh is False, returns specific humidity in kg/kg.
        If return_rh is True, returns relative humidity as a fraction (0-1).
    """
    # Convert latitude to radians for internal computation
    lat_2d_rad = np.deg2rad(lat_2d)

    # Compute ITCZ from temperature or use pre-computed
    if itcz_rad is None:
        with time_block("compute_itcz_in_humidity"):
            cell_areas = spherical_cell_area(lon_2d, lat_2d, earth_radius_m=R_EARTH_METERS)
            itcz = compute_itcz_latitude(temperature, lat_2d, cell_areas)
    else:
        itcz = itcz_rad

    nlat, nlon = temperature.shape

    # Broadcast ITCZ to 2D grid
    itcz_2d = np.broadcast_to(itcz[np.newaxis, :], (nlat, nlon))

    # Compute RH using analytical formula for land
    rh_land = relative_humidity_pattern(lat_2d_rad, itcz_2d)

    # Blend with ocean values at cell level if land mask is provided
    if land_mask is not None:
        ocean_mask = ~land_mask  # True for ocean cells

        if ocean_mask.any():
            # Ocean ITCZ is scaled by 0.75 (less land-ocean contrast)
            itcz_ocean_2d = np.broadcast_to((itcz * 0.75)[np.newaxis, :], (nlat, nlon))

            # Compute ocean RH with ocean-specific parameters
            rh_ocean = relative_humidity_pattern(
                lat_2d_rad, itcz_ocean_2d,
                RH_MEAN_WATER, DRH_ITCZ_WATER, DRH_SUBTROPICS_WATER,
                DRH_SUBPOLAR_WATER, DRH_POLES_WATER
            )

            # Blend: use ocean values for ocean cells, land values for land cells
            rh = np.where(ocean_mask, rh_ocean, rh_land)
        else:
            rh = rh_land
    else:
        rh = rh_land

    if return_rh:
        return rh

    # Magnus formula requires temperature in Celsius
    temperature_C = temperature - 273.15

    p_Pa = compute_pressure(temperature, lat2d=lat_2d, lon2d=lon_2d, itcz_rad=itcz)
    p_hPa = p_Pa / 100.0  # Convert Pa to hPa

    e_sat = 6.112 * np.exp(17.67 * temperature_C / (temperature_C + 243.5))
    q_sat = (0.622 * e_sat) / (p_hPa - (1 - 0.622) * e_sat)

    return q_sat * rh

def compute_cloud_cover(
    relative_humidity: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
    temperature: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cloud cover fraction from relative humidity.
    
    Uses a physically-based parameterization where cloud cover depends on
    relative humidity with different thresholds for land and ocean:
        C = max(0, (RH - RH_threshold) / (1 - RH_threshold)) ^ power
    
    Parameters
    ----------
    relative_humidity : np.ndarray | None, optional
        Relative humidity field (0-1). If None, falls back to latitude-based
        prescription for backward compatibility.
    land_mask : np.ndarray | None, optional
        Boolean array indicating land cells (True) vs ocean cells (False).
        If provided, different values are used for land vs ocean in the fallback.
    temperature : np.ndarray | None, optional
        Temperature field (K).
    Returns
    -------
    np.ndarray
        Cloud cover fraction (0-1) at each grid point.
    """
    # Fallback to latitude-based prescription if RH not provided
    if relative_humidity is None:
        if temperature is None:
            raise ValueError("Either relative_humidity or temperature must be provided")
        return _compute_cloud_cover_latitude_fallback(temperature, land_mask)
    
    rh_crit = 0.35
    rh_max = 0.85
    cloud_param = np.clip((relative_humidity - rh_crit)/(rh_max - rh_crit), 0, 1)

    cloud_cover = 3*cloud_param**2 - 2*cloud_param**3
    cloud_cover = 0.07 + (0.8 - 0.07) * cloud_cover
    
    return cloud_cover


def _compute_cloud_cover_latitude_fallback(
    temperature: np.ndarray, land_mask: np.ndarray | None = None
) -> np.ndarray:
    """Fallback latitude-dependent cloud cover (for backward compatibility).
    
    Uses a simple parameterization based on latitude only:
        C = 0.65 - 2.59 * sin²(lat) + 3.55 * sin⁴(lat)
    """
    nlat, nlon = temperature.shape[1], temperature.shape[2]

    # Reconstruct the latitude grid (cell centers)
    lat_spacing = 180.0 / float(nlat)
    lat_centers = -90.0 + (np.arange(nlat, dtype=float) + 0.5) * lat_spacing

    # Create 2D latitude field matching the grid
    lat2d = lat_centers[:, np.newaxis]  # Shape: (nlat, 1)
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

    return cloud_cover
