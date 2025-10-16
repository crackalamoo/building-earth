"""Geostrophic advection utilities for atmospheric heat transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.utils.elevation import (
    compute_cell_elevation,
    pressure_from_temperature_elevation,
)


def _compute_geostrophic_wind_components(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    temperature: np.ndarray,
    *,
    abs_coriolis: np.ndarray,
    lat_sign: np.ndarray,
    config: "GeostrophicAdvectionConfig",
    pressure: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the geostrophic wind given horizontal gradients of temperature or pressure."""

    grad_mag = np.hypot(grad_x, grad_y)

    coriolis = np.maximum(abs_coriolis, config.coriolis_floor_s)

    if config.use_pressure_gradients and pressure is not None:
        pressure_safe = np.maximum(pressure, config.minimum_pressure_pa)
        temp_safe = np.maximum(temperature, config.minimum_temperature_K)
        density = pressure_safe / (config.dry_air_gas_constant_J_kgK * temp_safe)
        density = np.maximum(density, config.minimum_density_kg_m3)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = grad_mag / (density * coriolis)
    else:
        temp_safe = np.maximum(temperature, config.minimum_temperature_K)
        scale = config.gravity_m_s2 * config.troposphere_scale_height_m
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = scale * grad_mag / (coriolis * temp_safe)

    speed = np.where(np.isfinite(speed), speed, 0.0)
    speed = np.where(grad_mag > 0.0, speed, 0.0)
    speed = np.where(abs_coriolis >= config.coriolis_floor_s, speed, 0.0)

    unit_x = np.zeros_like(grad_x)
    unit_y = np.zeros_like(grad_y)
    nonzero = grad_mag > 0.0
    unit_x[nonzero] = grad_x[nonzero] / grad_mag[nonzero]
    unit_y[nonzero] = grad_y[nonzero] / grad_mag[nonzero]

    velocity_x = np.zeros_like(unit_x)
    velocity_y = np.zeros_like(unit_y)

    nh = lat_sign > 0.0
    sh = lat_sign < 0.0

    velocity_x[nh] = unit_y[nh]
    velocity_y[nh] = -unit_x[nh]

    velocity_x[sh] = -unit_y[sh]
    velocity_y[sh] = unit_x[sh]

    velocity_x *= speed
    velocity_y *= speed

    return velocity_x, velocity_y, speed


@dataclass(frozen=True)
class GeostrophicAdvectionConfig:
    """Configuration for geostrophic advection tendencies."""

    enabled: bool = True
    use_pressure_gradients: bool = True
    earth_radius_m: float = 6.371e6
    earth_rotation_rate_rad_s: float = 7.2921e-5
    gravity_m_s2: float = 9.81
    troposphere_scale_height_m: float = 8000.0
    coriolis_floor_s: float = 1.0e-5
    minimum_temperature_K: float = 150.0
    reference_sea_level_pressure_pa: float = 101_325.0
    dry_air_gas_constant_J_kgK: float = 287.0
    minimum_density_kg_m3: float = 0.1
    minimum_pressure_pa: float = 100.0


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
        self._elevation_m: np.ndarray | None = None

        nlat, nlon = self._lon2d.shape
        if nlat < 1 or nlon < 1:
            raise ValueError("Longitude/latitude grids must be non-empty")

        lat_centers = self._lat2d[:, 0]
        lon_centers = self._lon2d[0, :]

        if nlat > 1:
            lat_spacing = np.diff(lat_centers)
            if not np.allclose(lat_spacing, lat_spacing[0]):
                raise ValueError("Latitude grid must have constant spacing for gradients")
            self._delta_y = config.earth_radius_m * np.deg2rad(float(lat_spacing[0]))
        else:
            self._delta_y = np.inf

        if nlon > 1:
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
        else:
            self._inv_two_delta_x = np.zeros_like(self._lon2d)

        self._lat_sign = np.sign(self._lat2d)
        coriolis = 2.0 * config.earth_rotation_rate_rad_s * np.sin(
            np.deg2rad(self._lat2d)
        )
        self._abs_coriolis = np.abs(coriolis)

        if self._config.use_pressure_gradients:
            self._elevation_m = compute_cell_elevation(self._lon2d, self._lat2d)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def wind_field(
        self, temperature: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the geostrophic wind field (u, v, speed) for the given temperatures."""

        if temperature.shape != self._lon2d.shape:
            raise ValueError(
                "Temperature field must match the longitude/latitude grid shape"
            )

        if not self.enabled:
            zeros = np.zeros_like(temperature)
            return zeros, zeros, zeros

        pressure = None
        if self._config.use_pressure_gradients and self._elevation_m is not None:
            pressure = pressure_from_temperature_elevation(
                temperature,
                self._elevation_m,
                sea_level_pressure_pa=self._config.reference_sea_level_pressure_pa,
                gravity_m_s2=self._config.gravity_m_s2,
                gas_constant_J_kgK=self._config.dry_air_gas_constant_J_kgK,
            )
            grad_x, grad_y = self._horizontal_gradient(pressure)
        else:
            grad_x, grad_y = self._horizontal_gradient(temperature)
        return _compute_geostrophic_wind_components(
            grad_x,
            grad_y,
            temperature,
            abs_coriolis=self._abs_coriolis,
            lat_sign=self._lat_sign,
            config=self._config,
            pressure=pressure,
        )

    def zero_wind_field(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a wind field with zero velocity everywhere."""

        zeros = np.zeros_like(self._lon2d)
        return zeros.copy(), zeros.copy(), zeros.copy()

    def tendency_from_wind(
        self, temperature: np.ndarray, wind_field: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    ) -> np.ndarray:
        """Return the advection tendency for a prescribed wind field."""

        if temperature.shape != self._lon2d.shape:
            raise ValueError("Temperature field must match the longitude/latitude grid shape")

        if not self.enabled or wind_field is None:
            return np.zeros_like(temperature)

        velocity_x, velocity_y, _speed = wind_field
        if velocity_x.shape != temperature.shape or velocity_y.shape != temperature.shape:
            raise ValueError("Wind field must match the temperature grid shape")

        grad_x, grad_y = self._horizontal_gradient(temperature)
        tendency = -(velocity_x * grad_x + velocity_y * grad_y)
        return tendency

    def _horizontal_gradient(self, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if temperature.shape != self._lon2d.shape:
            raise ValueError("Temperature field must match the longitude/latitude grid shape")

        grad_y = np.zeros_like(temperature)
        if np.isfinite(self._delta_y) and self._delta_y > 0.0 and temperature.shape[0] > 1:
            inv_delta_y = 1.0 / self._delta_y
            inv_two_delta_y = 0.5 * inv_delta_y
            grad_y[1:-1] = (temperature[2:] - temperature[:-2]) * inv_two_delta_y
            grad_y[0] = (temperature[1] - temperature[0]) * inv_delta_y
            grad_y[-1] = (temperature[-1] - temperature[-2]) * inv_delta_y

        if temperature.shape[1] > 1:
            diff_east = np.roll(temperature, -1, axis=1) - np.roll(temperature, 1, axis=1)
            grad_x = diff_east * self._inv_two_delta_x
        else:
            grad_x = np.zeros_like(temperature)

        return grad_x, grad_y

    def tendency(self, temperature: np.ndarray) -> np.ndarray:
        """Return the geostrophic advection tendency (K/s) for the given field."""

        if not self.enabled:
            return np.zeros_like(temperature)

        wind = self.wind_field(temperature)
        return self.tendency_from_wind(temperature, wind)
