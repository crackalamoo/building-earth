"""Seasonal insolation utilities and calendar helpers."""

from __future__ import annotations

import numpy as np
from typing import Optional

ECCENTRICITY_AMPLITUDE = 0.033  # Approximate fractional variation (≈2e) in solar flux

SOLAR_CONSTANT = 1361.0  # W m-2
OBLIQUITY_DEGREES = 23.44
SECONDS_PER_DAY = 86400.0
DAYS_PER_MONTH = np.array([31, 28.2425, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
ANNUAL_DAYS = DAYS_PER_MONTH.sum()


def monthly_midpoint_days() -> np.ndarray:
    """Return the day-of-year for the midpoint of each calendar month."""
    starts = np.concatenate(([0.0], np.cumsum(DAYS_PER_MONTH)[:-1]))
    return starts + 0.5 * DAYS_PER_MONTH


def solar_declination(day_of_year: np.ndarray) -> np.ndarray:
    """Compute solar declination angle (radians) for given day of year."""
    mean_longitude = 2.0 * np.pi * (day_of_year - 80.0) / ANNUAL_DAYS
    obliquity_rad = np.deg2rad(OBLIQUITY_DEGREES)
    return np.arcsin(np.sin(obliquity_rad) * np.sin(mean_longitude))


def daily_mean_insolation(
    lat_rad: np.ndarray,
    declination_rad: np.ndarray,
    *,
    solar_constant: float,
    orbital_distance_factor: np.ndarray | None = None,
) -> np.ndarray:
    """Compute daily-mean top-of-atmosphere insolation for a given declination."""
    tan_lat = np.tan(lat_rad)
    tan_dec = np.tan(declination_rad)
    cos_hour_angle = -tan_lat[:, None] * tan_dec[None, :]

    cos_hour_angle = np.clip(cos_hour_angle, -1.0, 1.0)
    hour_angle = np.arccos(cos_hour_angle)

    polar_night = cos_hour_angle >= 1.0
    polar_day = cos_hour_angle <= -1.0
    hour_angle[polar_night] = 0.0
    hour_angle[polar_day] = np.pi

    sin_lat = np.sin(lat_rad)[:, None]
    cos_lat = np.cos(lat_rad)[:, None]
    sin_dec = np.sin(declination_rad)[None, :]
    cos_dec = np.cos(declination_rad)[None, :]

    flux = (
        solar_constant
        / np.pi
        * (hour_angle * sin_lat * sin_dec + cos_lat * cos_dec * np.sin(hour_angle))
    )
    if orbital_distance_factor is not None:
        flux = flux * orbital_distance_factor[None, :]
    return np.maximum(flux, 0.0)


def compute_monthly_insolation_field(
    lat2d: np.ndarray,
    *,
    solar_constant: Optional[float] = SOLAR_CONSTANT,
    use_elliptical_orbit: bool = True,
) -> np.ndarray:
    """Return monthly-averaged insolation (W m-2) at the top of the atmosphere."""
    solar_constant = solar_constant if solar_constant is not None else SOLAR_CONSTANT
    lat_rad = np.deg2rad(lat2d[:, 0])
    midpoints = monthly_midpoint_days()
    declinations = solar_declination(midpoints)
    distance_factor = None
    if use_elliptical_orbit:
        distance_factor = orbital_distance_correction(midpoints)
    insolation_by_lat_month = daily_mean_insolation(
        lat_rad,
        declinations,
        solar_constant=solar_constant,
        orbital_distance_factor=distance_factor,
    )
    return insolation_by_lat_month.T  # month, lat


def orbital_distance_correction(day_of_year: np.ndarray) -> np.ndarray:
    """Return multiplicative correction for solar flux due to orbital eccentricity."""

    anomaly = 2.0 * np.pi * (day_of_year - 3.0) / ANNUAL_DAYS
    return 1.0 + ECCENTRICITY_AMPLITUDE * np.cos(anomaly)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from grid import create_lat_lon_grid, expand_latitude_field
    lon2d, lat2d = create_lat_lon_grid(1.0)
    insolation = compute_monthly_insolation_field(lat2d)
    print(insolation.shape)
    insolation = expand_latitude_field(insolation, lon2d.shape[1])
    print(insolation.shape)
    for i in range(insolation.shape[0]):
        plt.imshow(insolation[i], origin='lower', extent=(-180, 180, -90, 90), aspect='auto')
        plt.colorbar(label='W m$^{-2}$')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
