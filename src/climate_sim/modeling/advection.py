"""Geostrophic advection utilities for atmospheric heat transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.utils.elevation import load_elevation_data, compute_cell_elevation, pressure_from_temperature_elevation

def _compute_geostrophic_wind_components(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    temperature: np.ndarray,
    *,
    abs_coriolis: np.ndarray,
    config: "GeostrophicAdvectionConfig",
    pressure: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the geostrophic wind given horizontal temperature gradients."""

    coriolis = np.maximum(abs_coriolis, config.coriolis_floor_s)
    if pressure is not None:
        pressure_safe = np.maximum(pressure, 100.0)
        velocity_x = grad_y / (coriolis * pressure_safe)
        velocity_y = -grad_x / (coriolis * pressure_safe)
        speed = np.hypot(velocity_x, velocity_y)
        return velocity_x, velocity_y, speed

    temp_safe = np.maximum(temperature, config.minimum_temperature_K)
    scale = config.gravity_m_s2 * config.troposphere_scale_height_m

    velocity_x = scale * grad_y / (coriolis * temp_safe)
    velocity_y = -scale * grad_x / (coriolis * temp_safe)
    speed = np.hypot(velocity_x, velocity_y)

    return velocity_x, velocity_y, speed


@dataclass(frozen=True)
class GeostrophicAdvectionConfig:
    """Configuration for geostrophic advection tendencies."""

    enabled: bool = True
    earth_radius_m: float = 6.371e6
    earth_rotation_rate_rad_s: float = 7.2921e-5
    gravity_m_s2: float = 9.81
    troposphere_scale_height_m: float = 8000.0
    coriolis_floor_s: float = 1.0e-5
    minimum_temperature_K: float = 150.0
    use_pressure_gradients: bool = True

class GeostrophicAdvectionOperator:
    """Evaluate geostrophic advection tendencies on a fixed longitude/latitude grid."""

    def __init__(
        self,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        *,
        config: GeostrophicAdvectionConfig,
    ) -> None:
        if lon2d.shape != lat2d.shape:
            raise ValueError("Longitude and latitude grids must share the same shape")
        if lon2d.ndim != 2:
            raise ValueError("Longitude and latitude grids must be two-dimensional")

        self._lon2d = np.asarray(lon2d, dtype=float)
        self._lat2d = np.asarray(lat2d, dtype=float)
        self._config = config

        nlat, nlon = self._lon2d.shape
        if nlat < 1 or nlon < 1:
            raise ValueError("Longitude/latitude grids must be non-empty")

        lat_centers = self._lat2d[:, 0]
        lon_centers = self._lon2d[0, :]

        lat_spacing = np.diff(lat_centers)
        if not np.allclose(lat_spacing, lat_spacing[0]):
            raise ValueError("Latitude grid must have constant spacing for gradients")
        self._delta_y = config.earth_radius_m * np.deg2rad(float(lat_spacing[0]))

        lon_spacing = np.diff(lon_centers)
        if not np.allclose(lon_spacing, lon_spacing[0]):
            raise ValueError("Longitude grid must have constant spacing for gradients")
        delta_lon_rad = np.deg2rad(float(lon_spacing[0]))
        cos_lat = np.cos(np.deg2rad(lat_centers))[:, np.newaxis]
        delta_x = config.earth_radius_m * cos_lat * delta_lon_rad
        with np.errstate(divide="ignore", invalid="ignore"):
            self._inv_two_delta_x = np.zeros_like(delta_x)
            valid = np.abs(delta_x) > 0.0
            self._inv_two_delta_x[valid] = 1.0 / (2.0 * delta_x[valid])

        coriolis = 2.0 * config.earth_rotation_rate_rad_s * np.sin(
            np.deg2rad(self._lat2d)
        )
        self._abs_coriolis = np.abs(coriolis)

        if config.use_pressure_gradients:
            elevation_data = load_elevation_data()
            assert elevation_data is not None, "Elevation data could not be loaded"
            self.elevation_m = compute_cell_elevation(
                self._lon2d, self._lat2d, data=elevation_data, sample_method="center"
            )
            self.elevation_m = np.maximum(self.elevation_m, 0.0)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def wind_field(
        self, temperature: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        """Compute the geostrophic wind field (u, v, speed) for the given temperatures."""

        if temperature.shape != self._lon2d.shape:
            raise ValueError(
                "Temperature field must match the longitude/latitude grid shape"
            )

        if not self.enabled:
            zeros = np.zeros_like(temperature)
            return (zeros, zeros, zeros), zeros, zeros

        pressure = None
        if self._config.use_pressure_gradients:
            pressure = pressure_from_temperature_elevation(temperature)
            grad_x, grad_y = self._horizontal_gradient(pressure)
        else:
            grad_x, grad_y = self._horizontal_gradient(temperature)
        return _compute_geostrophic_wind_components(
            grad_x,
            grad_y,
            temperature,
            abs_coriolis=self._abs_coriolis,
            config=self._config,
            pressure=pressure,
        ), grad_x, grad_y

    def _horizontal_gradient(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if field.shape != self._lon2d.shape:
            raise ValueError("Field must match the longitude/latitude grid shape")

        grad_y = np.zeros_like(field)
        if np.isfinite(self._delta_y) and self._delta_y > 0.0 and field.shape[0] > 1:
            inv_delta_y = 1.0 / self._delta_y
            inv_two_delta_y = 0.5 * inv_delta_y
            grad_y[1:-1] = (field[2:] - field[:-2]) * inv_two_delta_y
            grad_y[0] = (field[1] - field[0]) * inv_delta_y
            grad_y[-1] = (field[-1] - field[-2]) * inv_delta_y

        if field.shape[1] > 1:
            diff_east = np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)
            grad_x = diff_east * self._inv_two_delta_x
        else:
            grad_x = np.zeros_like(field)

        return grad_x, grad_y

    def tendency(self, temperature: np.ndarray) -> np.ndarray:
        """Return the geostrophic advection tendency (K/s) for the given field."""

        if not self.enabled:
            return np.zeros_like(temperature)

        pressure = None
        if self._config.use_pressure_gradients:
            pressure = pressure_from_temperature_elevation(temperature)
            grad_x, grad_y = self._horizontal_gradient(pressure)
        else:
            grad_x, grad_y = self._horizontal_gradient(temperature)
        velocity_x, velocity_y, _speed = _compute_geostrophic_wind_components(
            grad_x,
            grad_y,
            temperature,
            abs_coriolis=self._abs_coriolis,
            config=self._config,
            pressure=pressure,
        )

        tendency = -(velocity_x * grad_x + velocity_y * grad_y)
        return tendency
