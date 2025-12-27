"""Surface pressure estimation utilities driven by climate model fields."""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator
from climate_sim.core.math_core import area_weighted_mean
from climate_sim.data.constants import ATMOSPHERE_MASS, EARTH_SURFACE_AREA_M2, GAS_CONSTANT_J_KG_K, R_EARTH_METERS

# Hadley cell pressure anomalies (Pa)
# Low pressure at ITCZ (rising air), high pressure at subtropics (descending air)
DP_ITCZ = -1000.0  # Low pressure at equatorial trough
DP_SUBTROPICS = 800.0  # High pressure at subtropical highs (~30°)
DP_SUBPOLAR = -500.0  # Low pressure at subpolar lows (~60°)
DP_POLES = 300.0  # Weak high pressure at polar highs

# Key latitude bands (radians)
LAT_SUBTROPICS = np.deg2rad(30.0)
LAT_SUBPOLAR = np.deg2rad(60.0)
LAT_POLES = np.deg2rad(80.0)

def _get_latitude_centers(nlat: int) -> np.ndarray:
    """Return latitude centers (deg) for a grid with nlat latitude points."""
    if nlat <= 0:
        raise ValueError("Number of latitude points must be positive")

    lat_spacing = 180.0 / float(nlat)
    lat_centers = -90.0 + (np.arange(nlat, dtype=float) + 0.5) * lat_spacing
    return lat_centers


def _hadley_cell_pressure_function(itcz_lat_rad: float) -> PchipInterpolator:
    """Create pressure anomaly interpolator for Hadley cell circulation.

    Parameters
    ----------
    itcz_lat_rad : float
        ITCZ latitude in radians (follows solar declination).

    Returns
    -------
    PchipInterpolator
        Interpolator mapping latitude (radians) to pressure anomaly (Pa).
    """
    # Control points for pressure pattern
    # Symmetric pattern shifted by ITCZ position
    phi = np.array([
        -LAT_POLES,          # South polar high
        -LAT_SUBPOLAR,       # South subpolar low
        -LAT_SUBTROPICS,     # South subtropical high
        itcz_lat_rad,        # ITCZ (equatorial trough)
        LAT_SUBTROPICS,      # North subtropical high
        LAT_SUBPOLAR,        # North subpolar low
        LAT_POLES            # North polar high
    ])

    dp = np.array([
        DP_POLES,
        DP_SUBPOLAR,
        DP_SUBTROPICS,
        DP_ITCZ,
        DP_SUBTROPICS,
        DP_SUBPOLAR,
        DP_POLES
    ])

    return PchipInterpolator(phi, dp)


def _smooth_temperature_field(
    field: np.ndarray,
    lat_centers: np.ndarray,
    *,
    smoothing_length_km: float = 1000.0,
) -> np.ndarray:
    """Apply latitude-dependent Gaussian smoothing with longitude wrapping.

    Uses isotropic smoothing in physical space - constant smoothing length in km
    regardless of latitude. This means more grid cells are smoothed at high latitudes
    where cells are smaller in the zonal direction.
    """

    field = np.asarray(field, dtype=float)
    nlat, nlon = field.shape

    # Calculate grid spacing in km
    lat_spacing_deg = 180.0 / nlat
    lat_spacing_km = R_EARTH_METERS * np.deg2rad(lat_spacing_deg) / 1000.0

    # Meridional sigma in grid cells (constant)
    sigma_lat = smoothing_length_km / lat_spacing_km

    # Zonal spacing varies with latitude
    lon_spacing_deg = 360.0 / nlon
    cos_lat = np.cos(np.deg2rad(lat_centers))
    lon_spacing_km = R_EARTH_METERS * np.deg2rad(lon_spacing_deg) * np.abs(cos_lat) / 1000.0

    # Avoid division by zero at poles
    lon_spacing_km = np.maximum(lon_spacing_km, lat_spacing_km * 0.1)

    # Sigma varies by latitude - more grid cells smoothed near poles
    sigma_lon_by_lat = smoothing_length_km / lon_spacing_km

    # Maximum padding needed (at equator where sigma_lon is smallest)
    max_sigma_lon = np.max(sigma_lon_by_lat)
    pad_width = int(np.ceil(3 * max_sigma_lon))

    # Wrap field in longitude for periodic boundary
    field_wrapped = np.pad(field, ((0, 0), (pad_width, pad_width)), mode='wrap')

    # Apply latitude-dependent smoothing row by row
    smoothed_wrapped = np.zeros_like(field_wrapped)
    for i in range(nlat):
        sigma_lon = sigma_lon_by_lat[i]
        smoothed_wrapped[i, :] = gaussian_filter(
            field_wrapped[i, :],
            sigma=sigma_lon,
            mode='nearest'
        )

    # Apply meridional smoothing
    smoothed_wrapped = gaussian_filter(smoothed_wrapped, sigma=(sigma_lat, 0), mode='nearest')

    # Extract the central portion (unwrap)
    smoothed = smoothed_wrapped[:, pad_width:-pad_width]

    return smoothed


def compute_pressure(
    temperature_K: np.ndarray,
    elevation_m: np.ndarray | None = None,
    humidity_q: np.ndarray | None = None,
    gravity_m_s2: float = 9.81,
    declination_rad: float | None = None,
) -> np.ndarray:
    """Compute surface pressure (Pa) from temperature and elevation using hydrostatic balance.

    Parameters
    ----------
    temperature_K : np.ndarray
        Surface temperature field in Kelvin.
    elevation_m : np.ndarray | None
        Surface elevation in meters (optional).
    humidity_q : np.ndarray | None
        Specific humidity in kg/kg (optional).
    gravity_m_s2 : float
        Gravitational acceleration in m/s².
    declination_rad : float | None
        Solar declination angle in radians. If provided, adds Hadley cell circulation
        pattern with ITCZ following the thermal equator.

    Returns
    -------
    np.ndarray
        Surface pressure field in Pascals.
    """

    temperature = np.asarray(temperature_K, dtype=float)
    if temperature.ndim != 2:
        raise ValueError("temperature_K must be a 2-D latitude/longitude field")

    shape = temperature.shape
    if elevation_m is None:
        elevation = np.zeros(shape, dtype=float)
    else:
        elevation = np.asarray(elevation_m, dtype=float)
        if elevation.shape != shape:
            raise ValueError("Temperature and elevation fields must share the same shape")

    if humidity_q is not None:
        humidity = np.asarray(humidity_q, dtype=float)
        if humidity.shape != shape:
            raise ValueError("Temperature and humidity fields must share the same shape")

    mean_p = ATMOSPHERE_MASS * gravity_m_s2 / EARTH_SURFACE_AREA_M2

    nlat, nlon = shape
    lat_deg = _get_latitude_centers(nlat)
    cos_lat = np.clip(np.cos(np.deg2rad(lat_deg)), 1.0e-6, None)
    weights = np.asarray(np.broadcast_to(cos_lat[:, None], shape), dtype=float)

    if humidity_q is not None:
        virtual_temperature = temperature * (1 + 0.61 * humidity_q)
        temp_smooth = _smooth_temperature_field(virtual_temperature, lat_deg, smoothing_length_km=1000.0)
        target_mean = area_weighted_mean(virtual_temperature, weights)
    else:
        temp_smooth = _smooth_temperature_field(temperature, lat_deg, smoothing_length_km=1000.0)
        target_mean = area_weighted_mean(temperature, weights)

    smooth_mean = area_weighted_mean(temp_smooth, weights)
    temp_smooth = temp_smooth + (target_mean - smooth_mean)

    dT = temp_smooth - area_weighted_mean(temp_smooth, weights)

    beta = 30.0
    dp_th = -beta * dT
    dp_th = dp_th - area_weighted_mean(dp_th, weights)

    t_ref_lat = area_weighted_mean(temp_smooth, weights, axis=1)
    t_ref_lat_2d = np.broadcast_to(t_ref_lat[:, None], shape)
    t_ref_safe = np.maximum(t_ref_lat_2d, 1.0)

    p_orog = mean_p * np.exp(-gravity_m_s2 * elevation / (GAS_CONSTANT_J_KG_K * t_ref_safe))

    # Add Hadley cell circulation pattern if declination is provided
    if declination_rad is None:
        declination_rad = 0.0
    # ITCZ follows thermal equator (scaled by solar declination)
    itcz_lat_rad = declination_rad * 0.65
    hadley_function = _hadley_cell_pressure_function(itcz_lat_rad)

    # Create 2D latitude field in radians
    lat_2d_rad = np.deg2rad(np.broadcast_to(lat_deg[:, None], shape))

    dp_hadley = hadley_function(lat_2d_rad)
    dp_hadley = dp_hadley - area_weighted_mean(dp_hadley, weights)

    p_surface = p_orog + dp_th + dp_hadley
    p_surface = p_surface * (mean_p / area_weighted_mean(p_surface, weights))
    # p_surface = mean_p + dp_hadley

    return p_surface
