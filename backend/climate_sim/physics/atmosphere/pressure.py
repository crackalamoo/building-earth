"""Surface pressure estimation utilities driven by climate model fields."""

import numpy as np
from scipy.ndimage import gaussian_filter
from climate_sim.core.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.core.timing import time_block
from climate_sim.data.constants import (
    GAS_CONSTANT_J_KG_K,
    R_EARTH_METERS,
)
from climate_sim.physics.atmosphere.hadley import LAT_POLES, LAT_SUBPOLAR, compute_itcz_latitude

# Base latitude of subtropical highs (radians) - relatively fixed at ~30°
LAT_SUBTROPICS_BASE = np.deg2rad(29.0)
# How much subtropical highs track ITCZ movement (0 = fixed, 1 = moves with ITCZ)
# Reduced from 0.325: real subtropical high doesn't migrate as far as ITCZ,
# keeping descent zone near 30° rather than shifting to 35°+ in summer.
SUBTROPICS_ITCZ_COUPLING = 0.15

# Hadley cell pressure anomalies (Pa)
# These represent CIRCULATION-ONLY effects, separate from thermal pressure.
# Thermal effects (dp = -β*dT) are computed separately from temperature anomalies.
DP_ITCZ = -700.0  # Low pressure at ITCZ (rising air in Hadley cell)
DP_SUBTROPICS = 800.0  # High pressure at subtropical highs (descending air)
DP_POLES = 0.0  # No explicit polar high (prevents excessive polar easterlies)

# Subpolar lows: hemisphere-dependent because eddy momentum transport
# (which maintains these lows) differs fundamentally between hemispheres.
# SH: uninterrupted circumpolar ocean → continuous storm track → deep broad low.
# NH: continents fragment the westerlies → weaker zonal-mean low, concentrated
#     in localized features (Icelandic Low, Aleutian Low) that the thermal
#     term partially captures via T anomalies.
DP_SUBPOLAR_NH = -600.0  # Pa — weaker (continental disruption of storm track)
DP_SUBPOLAR_SH = -1500.0  # Pa — stronger (uninterrupted circumpolar storm track)

# Width of pressure features (radians) - controls smoothness of transitions
SIGMA_ITCZ = np.deg2rad(6.0)  # ITCZ trough width
SIGMA_SUBTROPICS = np.deg2rad(12.0)  # Subtropical high width
SIGMA_SUBPOLAR_NH = np.deg2rad(8.0)  # NH subpolar width
SIGMA_SUBPOLAR_SH = np.deg2rad(10.0)  # SH subpolar: slightly broader
SIGMA_POLES = np.deg2rad(8.0)  # Polar high width

# Thermal pressure coefficient: δp = -β δT
# When a column warms by δT, it expands, upper-level mass diverges, and surface
# pressure drops.  The steady-state fractional change scales as δp/p ≈ -δT/T,
# giving β ≈ p_mean / T_mean as an upper bound.  In practice the response is
# weaker because the BL is only ~15% of the column depth, the upper troposphere
# partially compensates, and friction limits the steady-state deficit.
# Calibrated to observed thermal lows: Sahara ~10 hPa for ~5 K column ΔT.
# Calibrated to observed thermal lows: Sahara ~10 hPa for ~5 K column ΔT.
THERMAL_PRESSURE_COEFFICIENT = 200.0  # Pa/K

# Rossby deformation radius: L_R = N*H / f, the minimum scale at which
# temperature anomalies can organise upper-level mass redistribution and
# create surface pressure anomalies.  Used for latitude-dependent smoothing.
_BRUNT_VAISALA_FREQ = 0.01  # s⁻¹, typical tropospheric N
_TROPOPAUSE_HEIGHT_M = 10_000.0  # m, effective scale height
_OMEGA = 7.2921e-5  # rad/s, Earth's angular velocity
_MIN_ROSSBY_RADIUS_KM = 500.0  # floor near poles (prevents blow-up)
_MAX_ROSSBY_RADIUS_KM = 4000.0  # cap in deep tropics


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
    # Subtropical highs: base at ~30° with small ITCZ-tracking component
    # When ITCZ moves north, subtropical highs shift slightly poleward in NH, equatorward in SH
    lat_subtrop_north = LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad
    lat_subtrop_south = -LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad

    subtrop_strength_north = DP_SUBTROPICS
    subtrop_strength_south = DP_SUBTROPICS

    # ITCZ low pressure trough
    dp_itcz = DP_ITCZ * np.exp(-(((lat_rad - itcz_rad) / SIGMA_ITCZ) ** 2))

    # Subtropical highs (follow ITCZ, strength modulated by cell width)
    dp_subtrop = subtrop_strength_south * np.exp(
        -(((lat_rad - lat_subtrop_south) / SIGMA_SUBTROPICS) ** 2)
    ) + subtrop_strength_north * np.exp(-(((lat_rad - lat_subtrop_north) / SIGMA_SUBTROPICS) ** 2))

    # Subpolar lows (fixed latitudes, hemisphere-dependent amplitude and width)
    dp_subpolar = DP_SUBPOLAR_SH * np.exp(
        -(((lat_rad + LAT_SUBPOLAR) / SIGMA_SUBPOLAR_SH) ** 2)
    ) + DP_SUBPOLAR_NH * np.exp(-(((lat_rad - LAT_SUBPOLAR) / SIGMA_SUBPOLAR_NH) ** 2))

    # Polar highs (fixed latitudes)
    dp_poles = DP_POLES * (
        np.exp(-(((lat_rad + LAT_POLES) / SIGMA_POLES) ** 2))
        + np.exp(-(((lat_rad - LAT_POLES) / SIGMA_POLES) ** 2))
    )

    return dp_itcz + dp_subtrop + dp_subpolar + dp_poles


def _rossby_radius_km(lat_deg: np.ndarray) -> np.ndarray:
    """Rossby deformation radius L_R = N*H / |f|, clamped to physical bounds.

    At the equator f→0, so L_R diverges; we cap at _MAX_ROSSBY_RADIUS_KM.
    Near the poles f is large and L_R shrinks; we floor at _MIN_ROSSBY_RADIUS_KM.
    """
    f = np.abs(2.0 * _OMEGA * np.sin(np.deg2rad(lat_deg)))
    f_safe = np.maximum(f, 1e-10)  # avoid division by zero
    lr_m = _BRUNT_VAISALA_FREQ * _TROPOPAUSE_HEIGHT_M / f_safe
    lr_km = lr_m / 1000.0
    return np.clip(lr_km, _MIN_ROSSBY_RADIUS_KM, _MAX_ROSSBY_RADIUS_KM)


def _smooth_temperature_field(
    field: np.ndarray,
    lat_centers: np.ndarray,
    *,
    smoothing_length_km: float | None = None,
) -> np.ndarray:
    """Low-pass filter temperature at the local Rossby deformation radius.

    The Rossby radius L_R = N*H/f sets the minimum horizontal scale at which
    temperature anomalies can drive organised upper-level mass redistribution
    and hence surface pressure anomalies.  The Gaussian sigma is set to L_R/3,
    which preserves features at L_R (H ≈ 0.80) while filtering sub-Rossby
    noise (H ≈ 0.02 at L_R/3).
    """

    field = np.asarray(field, dtype=float)
    nlat, nlon = field.shape

    # Latitude-dependent smoothing length: σ = L_R / 3.
    # This preserves Rossby-scale features while filtering grid noise.
    if smoothing_length_km is not None:
        smooth_km = np.full_like(lat_centers, smoothing_length_km)
    else:
        smooth_km = _rossby_radius_km(lat_centers) / 3.0

    # Grid spacing in km
    lat_spacing_deg = 180.0 / nlat
    lat_spacing_km = R_EARTH_METERS * np.deg2rad(lat_spacing_deg) / 1000.0

    lon_spacing_deg = 360.0 / nlon
    cos_lat = np.cos(np.deg2rad(lat_centers))
    lon_spacing_km = R_EARTH_METERS * np.deg2rad(lon_spacing_deg) * np.abs(cos_lat) / 1000.0
    lon_spacing_km = np.maximum(lon_spacing_km, lat_spacing_km * 0.1)

    # Zonal sigma in grid cells (varies with latitude via both L_R and cell size)
    sigma_lon_by_lat = smooth_km / lon_spacing_km

    # Meridional sigma in grid cells (varies with latitude via L_R only)
    sigma_lat_by_lat = smooth_km / lat_spacing_km

    # Maximum padding needed
    max_sigma_lon = np.max(sigma_lon_by_lat)
    pad_width = int(np.ceil(3 * max_sigma_lon))

    # Wrap field in longitude for periodic boundary
    field_wrapped = np.pad(field, ((0, 0), (pad_width, pad_width)), mode="wrap")

    # Apply latitude-dependent smoothing row by row (zonal + meridional sigma both vary)
    smoothed_wrapped = np.zeros_like(field_wrapped)
    for i in range(nlat):
        sigma_lon = sigma_lon_by_lat[i]
        smoothed_wrapped[i, :] = gaussian_filter(
            field_wrapped[i, :], sigma=sigma_lon, mode="nearest"
        )

    # Meridional smoothing: apply row-by-row with varying sigma
    # We approximate by using the mean sigma (the variation is modest: ~1-7 cells)
    mean_sigma_lat = np.mean(sigma_lat_by_lat)
    smoothed_wrapped = gaussian_filter(smoothed_wrapped, sigma=(mean_sigma_lat, 0), mode="nearest")

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

    mean_p = 101325.0  # Pa, standard mean sea level pressure

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
        temp_smooth = _smooth_temperature_field(
            virtual_temperature, lat_deg, smoothing_length_km=None
        )
        target_mean = area_weighted_mean(virtual_temperature, weights)
    else:
        temp_smooth = _smooth_temperature_field(temperature, lat_deg, smoothing_length_km=None)
        target_mean = area_weighted_mean(temperature, weights)

    smooth_mean = area_weighted_mean(temp_smooth, weights)
    temp_smooth = temp_smooth + (target_mean - smooth_mean)

    # Thermal pressure responds only to ZONAL anomalies (departures from the
    # zonal mean).  The meridional pressure structure (equator-to-pole gradient)
    # is already captured by dp_hadley, which represents the Hadley/Ferrel/Polar
    # cell response to the meridional temperature gradient.  Applying dp_th to
    # the full temperature field would double-count the meridional component and
    # produce ~60-100 hPa equator-to-pole gradients (obs ~20-25 hPa).
    zonal_mean = np.mean(temp_smooth, axis=1, keepdims=True)
    dT_zonal = temp_smooth - zonal_mean

    dp_th = -THERMAL_PRESSURE_COEFFICIENT * dT_zonal
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
            cell_areas = spherical_cell_area(
                lon2d_nonnull, lat2d_nonnull, earth_radius_m=R_EARTH_METERS
            )
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
    Uses hydrostatic balance: Z = (R*T_local/g) * ln(p_surface/p₀)

    Using local (smoothed) temperature captures the thermal wind effect:
    warm columns have larger scale heights, so a given pressure surface is
    higher in warm air than cold air. The gradient of Z then includes both
    the SLP gradient contribution and the thermal wind contribution, which
    is the dominant term at upper levels.

    The input temperature should already be spatially smoothed (1000 km) to
    remove mesoscale thermal lows while preserving the large-scale meridional
    gradient that drives the thermal wind.
    """
    temperature = np.asarray(temperature_K, dtype=float)
    if temperature.ndim != 2:
        raise ValueError("temperature_K must be a 2-D latitude/longitude field")

    # Use local temperature for scale height to capture thermal wind effect.
    # dZ/dy = (R/g) * [dT/dy * ln(p_sfc/p_ref) + T/p_sfc * dp_sfc/dy]
    #          ^^^^ thermal wind term              ^^^^ SLP gradient term
    T_local = np.maximum(temperature, 150.0)
    scale_height = GAS_CONSTANT_J_KG_K * T_local / gravity_m_s2

    # Where surface pressure < reference pressure (e.g., high mountains), the
    # reference pressure level doesn't exist. Clip to avoid log of values < 1.
    pressure_ratio = np.maximum(surface_pressure_pa / reference_pressure_pa, 1.0)
    geopotential_height = scale_height * np.log(pressure_ratio)

    return geopotential_height
