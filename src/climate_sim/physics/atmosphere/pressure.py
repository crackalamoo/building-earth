"""Surface pressure estimation utilities driven by climate model fields."""

import numpy as np
from scipy.ndimage import gaussian_filter
from climate_sim.core.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.core.timing import time_block
from climate_sim.data.constants import ATMOSPHERE_MASS, EARTH_SURFACE_AREA_M2, GAS_CONSTANT_J_KG_K, R_EARTH_METERS
from climate_sim.physics.atmosphere.hadley import LAT_POLES, LAT_SUBPOLAR, DELTA_SUBTROPICS, compute_itcz_latitude

# Hadley cell pressure anomalies (Pa)
# Low pressure at ITCZ (rising air), high pressure at subtropics (descending air)
DP_ITCZ = -800.0  # Low pressure at equatorial trough
DP_SUBTROPICS = 800.0  # High pressure at subtropical highs (ITCZ ± DELTA_SUBTROPICS)
DP_SUBPOLAR = -500.0  # Low pressure at subpolar lows (~60°)
DP_POLES = 300.0  # Weak high pressure at polar highs

# Width of pressure features (radians) - controls smoothness of transitions
SIGMA_ITCZ = np.deg2rad(8.0)       # ITCZ trough width
SIGMA_SUBTROPICS = np.deg2rad(12.0)  # Subtropical high width
SIGMA_SUBPOLAR = np.deg2rad(10.0)   # Subpolar low width
SIGMA_POLES = np.deg2rad(8.0)       # Polar high width

def _get_latitude_centers(nlat: int) -> np.ndarray:
    """Return latitude centers (deg) for a grid with nlat latitude points."""
    if nlat <= 0:
        raise ValueError("Number of latitude points must be positive")

    lat_spacing = 180.0 / float(nlat)
    lat_centers = -90.0 + (np.arange(nlat, dtype=float) + 0.5) * lat_spacing
    return lat_centers


def hadley_pressure_anomaly(lat_rad: np.ndarray, itcz_rad: np.ndarray) -> np.ndarray:
    """Compute Hadley cell pressure anomaly using analytical Gaussian formula.

    This is a fully vectorized analytical formula - no interpolation needed.
    Each pressure feature (ITCZ, subtropical highs, subpolar lows, polar highs)
    is represented as a Gaussian bump/dip centered at its characteristic latitude.

    Parameters
    ----------
    lat_rad : np.ndarray
        Latitude field in radians, shape (nlat, nlon).
    itcz_rad : np.ndarray
        ITCZ latitude in radians, shape (nlon,) or broadcast-compatible.

    Returns
    -------
    np.ndarray
        Pressure anomaly in Pa, same shape as lat_rad.
    """
    # Subtropical highs follow ITCZ (descending branch of Hadley cell)
    lat_subtrop_south = 0.5 * itcz_rad - DELTA_SUBTROPICS
    lat_subtrop_north = 0.5 * itcz_rad + DELTA_SUBTROPICS

    # ITCZ low pressure trough
    dp_itcz = DP_ITCZ * np.exp(-((lat_rad - itcz_rad) / SIGMA_ITCZ) ** 2)

    # Subtropical highs (follow ITCZ)
    dp_subtrop = DP_SUBTROPICS * (
        np.exp(-((lat_rad - lat_subtrop_south) / SIGMA_SUBTROPICS) ** 2)
        + np.exp(-((lat_rad - lat_subtrop_north) / SIGMA_SUBTROPICS) ** 2)
    )

    # Subpolar lows (fixed latitudes)
    dp_subpolar = DP_SUBPOLAR * (
        np.exp(-((lat_rad + LAT_SUBPOLAR) / SIGMA_SUBPOLAR) ** 2)
        + np.exp(-((lat_rad - LAT_SUBPOLAR) / SIGMA_SUBPOLAR) ** 2)
    )

    # Polar highs (fixed latitudes)
    dp_poles = DP_POLES * (
        np.exp(-((lat_rad + LAT_POLES) / SIGMA_POLES) ** 2)
        + np.exp(-((lat_rad - LAT_POLES) / SIGMA_POLES) ** 2)
    )

    return dp_itcz + dp_subtrop + dp_subpolar + dp_poles


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
    skip_smoothing: bool = False,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
    itcz_rad: np.ndarray | None = None,
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
    skip_smoothing : bool
        If True, assume temperature_K is already smoothed and skip the smoothing step.
        Use this when calling from wind calculations to avoid double smoothing.
    lat2d : np.ndarray | None
        2D latitude grid in degrees (optional). Used to compute ITCZ if itcz_rad not provided.
    lon2d : np.ndarray | None
        2D longitude grid in degrees (optional). Used to compute ITCZ if itcz_rad not provided.
    itcz_rad : np.ndarray | None
        Pre-computed ITCZ latitude in radians, shape (nlon,). If provided, uses this.
        Otherwise computes from temperature if lat2d/lon2d provided.

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

    if skip_smoothing:
        # Temperature is already smoothed, use it directly
        temp_smooth = temperature
        target_mean = area_weighted_mean(temperature, weights)
    elif humidity_q is not None:
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

    # Compute ITCZ from temperature or use pre-computed
    if itcz_rad is None:
        if lat2d is None or lon2d is None:
            raise ValueError("lat2d and lon2d must be provided when itcz_rad is None")
        # Type narrowing: after the check above, we know these are not None
        lat2d_nonnull: np.ndarray = lat2d  # type: ignore[assignment]
        lon2d_nonnull: np.ndarray = lon2d  # type: ignore[assignment]
        with time_block("compute_itcz_in_pressure"):
            cell_areas = spherical_cell_area(lon2d_nonnull, lat2d_nonnull, earth_radius_m=R_EARTH_METERS)
            itcz_lat_rad = compute_itcz_latitude(temperature, lat2d_nonnull, cell_areas)
    else:
        itcz_lat_rad = itcz_rad

    # Create 2D latitude field in radians
    lat_2d_rad = np.deg2rad(np.broadcast_to(lat_deg[:, None], shape))

    # Broadcast ITCZ to 2D grid (same value for all latitudes in a longitude column)
    itcz_2d = np.broadcast_to(itcz_lat_rad[np.newaxis, :], shape)

    # Apply Hadley cell pressure pattern using analytical Gaussian formula
    dp_hadley = hadley_pressure_anomaly(lat_2d_rad, itcz_2d)
    dp_hadley = dp_hadley - area_weighted_mean(dp_hadley, weights)

    p_surface = p_orog + dp_th + dp_hadley
    p_surface = p_surface * (mean_p / area_weighted_mean(p_surface, weights))
    # p_surface = mean_p + dp_hadley

    return p_surface


def compute_geopotential_height(
    temperature_K: np.ndarray,
    reference_pressure_pa: float,
    surface_pressure_pa: np.ndarray,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    """Compute geopotential height (m) of a pressure surface.

    For a given pressure level p₀, computes the altitude where pressure equals p₀.
    Uses hydrostatic balance with isothermal atmosphere assumption:
        Z = (RT/g) * ln(p_surface/p₀)
    """
    temperature = np.asarray(temperature_K, dtype=float)
    if temperature.ndim != 2:
        raise ValueError("temperature_K must be a 2-D latitude/longitude field")

    T_safe = np.maximum(temperature, 150.0)
    scale_height = GAS_CONSTANT_J_KG_K * T_safe / gravity_m_s2

    # Where surface pressure < reference pressure (e.g., high mountains), the
    # reference pressure level doesn't exist. Clip to avoid log of values < 1.
    pressure_ratio = np.maximum(surface_pressure_pa / reference_pressure_pa, 1.0)
    geopotential_height = scale_height * np.log(pressure_ratio)

    return geopotential_height
