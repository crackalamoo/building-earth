"""Hadley cell circulation utilities.

This module contains utilities for computing ITCZ position and other
Hadley cell-related diagnostics based on temperature fields.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Key latitude bands (radians)
# These define the structure of the Hadley circulation
LAT_POLES = np.deg2rad(80.0)           # Polar highs
LAT_SUBPOLAR = np.deg2rad(60.0)        # Subpolar lows (Ferrel cell boundary)
DELTA_SUBTROPICS = np.deg2rad(20.0)    # Offset from ITCZ to subtropical highs/dry zones


def compute_itcz_latitude(
    temperature: np.ndarray,
    lat2d: np.ndarray,
    cell_areas: np.ndarray,
) -> np.ndarray:
    """Compute ITCZ latitude from area-weighted maximum temperature position.

    For each longitude column, computes a weighted centroid latitude using:
    - Area weighting: cell areas
    - Temperature weighting: exp((T - T_max) / tau) where T_max is the max temp for that longitude

    This gives more weight to warmer cells relative to the maximum temperature at each longitude.
    The warmest cell gets weight 1.0, and cooler cells get exponentially smaller weights.
    """
    # Temperature scale for exponential weighting (K or °C, doesn't matter for relative weights)
    tau = 2.0

    # Vectorized computation across all longitudes
    # Find max temperature per longitude: shape (nlon,)
    temp_max = np.max(temperature, axis=0)  # (nlon,)

    # Broadcast to compute temperature weights: exp((T - T_max) / tau)
    # Shape: (nlat, nlon) - (1, nlon) -> (nlat, nlon)
    temp_weights = np.exp((temperature - temp_max[np.newaxis, :]) / tau)

    # Compute weighted sum: sum over latitude dimension
    numerator = np.sum(lat2d * cell_areas * temp_weights, axis=0)  # (nlon,)
    denominator = np.sum(cell_areas * temp_weights, axis=0)  # (nlon,)

    # Compute centroid latitude per longitude
    max_temp_lat = numerator / np.maximum(denominator, 1e-10)

    # Smooth ITCZ longitudinally to avoid jaggedness
    # Use sigma = 15 degrees in longitude space
    nlon = len(max_temp_lat)
    lon_spacing_deg = 360.0 / nlon
    sigma_lon = 10.0 / lon_spacing_deg  # sigma in grid cells

    # ITCZ is typically between 30°S and 30°N
    max_temp_lat = np.clip(max_temp_lat, -30.0, 30.0)

    # Use mode='wrap' for periodic boundary conditions
    max_temp_lat = gaussian_filter1d(max_temp_lat, sigma=sigma_lon, mode='wrap')

    itcz_lat_rad = np.deg2rad(max_temp_lat)

    return itcz_lat_rad
