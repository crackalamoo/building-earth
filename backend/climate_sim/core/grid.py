"""Grid generation helpers for regular latitude/longitude meshes."""

import numpy as np


def create_lat_lon_grid(resolution_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D lon/lat arrays on a regular grid centered on cell middles."""
    lats = np.arange(-90.0 + resolution_deg / 2, 90.0, resolution_deg)
    lons = np.arange(resolution_deg / 2, 360.0, resolution_deg)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d


def expand_latitude_field(latitude_field: np.ndarray, lon_size: int) -> np.ndarray:
    """Broadcast a latitude-only monthly field across longitude indices."""
    return np.repeat(latitude_field[..., None], lon_size, axis=2)
