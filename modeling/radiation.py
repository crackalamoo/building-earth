"""Radiative equilibrium helpers based on a simple black-body balance.

This module offers a coarse-grained planetary skin-temperature estimate using
top-of-atmosphere solar forcing and the Stefan–Boltzmann law.  It ignores
horizontal heat transport, greenhouse effects, and seasonal/diurnal variations,
but serves as a physically motivated baseline field for testing the rest of the
pipeline.
"""

from __future__ import annotations

import numpy as np

SOLAR_CONSTANT = 1361.0  # W m-2, top-of-atmosphere solar irradiance
STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4


def create_lat_lon_grid(resolution_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D lon/lat arrays on a regular grid centred on cell middles."""
    lats = np.arange(-90.0 + resolution_deg / 2, 90.0, resolution_deg)
    lons = np.arange(resolution_deg / 2, 360.0, resolution_deg)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d


def daily_mean_insolation(lat_deg: np.ndarray, *, solar_constant: float = SOLAR_CONSTANT) -> np.ndarray:
    """Approximate equinox daily-mean insolation as a function of latitude.

    Uses the classic result for declination zero, giving:
        Q(φ) = (S0 / π) * cos φ  for |φ| ≤ 90°
    Negative values (polar night) are clipped to zero.
    """
    lat_rad = np.deg2rad(lat_deg)
    mean_flux = (solar_constant / np.pi) * np.cos(lat_rad)
    return np.clip(mean_flux, a_min=0.0, a_max=None)


def compute_blackbody_temperature_field(
    resolution_deg: float = 1.0,
    *,
    solar_constant: float = SOLAR_CONSTANT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lon/lat grids and the corresponding radiative equilibrium temperature (K).

    The calculation assumes:
    * Daily-mean equinox insolation at the top of the atmosphere.
    * Instantaneous radiative balance with no horizontal energy transport.
    """
    lon2d, lat2d = create_lat_lon_grid(resolution_deg)
    incoming = daily_mean_insolation(lat2d, solar_constant=solar_constant)

    # Avoid division-by-zero by inserting a tiny floor where insolation vanishes.
    absorbed = np.maximum(incoming, 1e-6)
    temperature = (absorbed / STEFAN_BOLTZMANN) ** 0.25
    return lon2d, lat2d, temperature


def compute_temperature_celsius(
    resolution_deg: float = 1.0,
    *,
    solar_constant: float = SOLAR_CONSTANT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper that returns radiative-equilibrium temperature in °C."""
    lon2d, lat2d, temperature_K = compute_blackbody_temperature_field(
        resolution_deg=resolution_deg,
        solar_constant=solar_constant,
    )
    temperature_C = temperature_K - 273.15
    return lon2d, lat2d, temperature_C
