"""Cloud cover parameterization utilities."""

from __future__ import annotations

import numpy as np


def compute_cloud_cover(temperature: np.ndarray) -> np.ndarray:
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

    cloud_cover = 0.65 - 2.59 * sin2_lat + 3.55 * sin4_lat

    # Clamp to maximum of 0.9
    cloud_cover = np.minimum(cloud_cover, 0.9)

    return cloud_cover
