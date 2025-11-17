"""Cloud cover parameterization utilities."""

from __future__ import annotations

import numpy as np

from climate_sim.data.constants import OCEAN_CLOUD_COVER_BOOST


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
