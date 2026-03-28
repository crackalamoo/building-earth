"""Hadley cell circulation utilities.

This module contains utilities for computing ITCZ position and other
Hadley cell-related diagnostics based on temperature fields.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Key latitude bands (radians)
# These define the structure of the Hadley circulation
LAT_POLES = np.deg2rad(80.0)  # Polar highs
LAT_SUBPOLAR = np.deg2rad(60.0)  # Subpolar lows (Ferrel cell boundary)
DELTA_SUBTROPICS = np.deg2rad(30.0)  # Offset from ITCZ to subtropical highs/dry zones


def compute_itcz_latitude(
    temperature: np.ndarray,
    lat2d: np.ndarray,
    cell_areas: np.ndarray,
    tau: float = 0.5,
) -> np.ndarray:
    """Compute ITCZ latitude from area-weighted maximum temperature position.

    Uses a softmax centroid with sharpness parameter tau. Smaller tau tracks
    the thermal equator more tightly; larger tau gives a broader, more stable
    centroid.
    """
    # Create tropical mask: only consider latitudes between -30° and 30°
    tropical_mask = (lat2d >= -30.0) & (lat2d <= 30.0)

    # Apply mask to temperature for finding max (set non-tropical to -inf)
    temp_masked = np.where(tropical_mask, temperature, -np.inf)

    # Find max temperature per longitude within tropical band: shape (nlon,)
    temp_max = np.max(temp_masked, axis=0)

    # Broadcast to compute temperature weights: exp((T - T_max) / tau)
    temp_weights = np.exp((temperature - temp_max[np.newaxis, :]) / tau)

    # Zero out weights outside tropical band
    temp_weights = np.where(tropical_mask, temp_weights, 0.0)

    # Compute weighted sum: sum over latitude dimension
    numerator = np.sum(lat2d * cell_areas * temp_weights, axis=0)
    denominator = np.sum(cell_areas * temp_weights, axis=0)

    # Compute centroid latitude per longitude
    max_temp_lat = numerator / np.maximum(denominator, 1e-10)

    # Smooth ITCZ longitudinally to avoid jaggedness
    nlon = len(max_temp_lat)
    lon_spacing_deg = 360.0 / nlon
    sigma_lon = 10.0 / lon_spacing_deg

    max_temp_lat = np.clip(max_temp_lat, -30.0, 30.0)
    max_temp_lat = gaussian_filter1d(max_temp_lat, sigma=sigma_lon, mode="wrap")

    itcz_lat_rad = np.deg2rad(max_temp_lat)

    return itcz_lat_rad
