"""Geostrophic advection utilities for atmospheric heat transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeostrophicAdvectionConfig:
    """Configuration for geostrophic advection tendencies."""

    enabled: bool = True
    ekman_friction: bool = True
    earth_radius_m: float = 6.371e6
    earth_rotation_rate_rad_s: float = 7.2921e-5
    gravity_m_s2: float = 9.81
    troposphere_scale_height_m: float = 8000.0
    coriolis_floor_s: float = 1.0e-5
    minimum_temperature_K: float = 150.0
    boundary_layer_depth_m: float = 1000.0
    land_eddy_viscosity_m2_s: float = 1.0
    ocean_eddy_viscosity_m2_s: float = 2.0


class GeostrophicAdvectionOperator:
    """Evaluate geostrophic advection tendencies on a fixed longitude/latitude grid."""

    def __init__(
        self,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        *,
        ocean_mask: np.ndarray | None = None,
        config: GeostrophicAdvectionConfig,
    ) -> None:
        if lon2d.shape != lat2d.shape:
            raise ValueError("Longitude and latitude grids must share the same shape")
        if lon2d.ndim != 2:
            raise ValueError("Longitude and latitude grids must be two-dimensional")

        self._lon2d = np.asarray(lon2d, dtype=float)
        self._lat2d = np.asarray(lat2d, dtype=float)
        self._config = config

        if ocean_mask is not None:
            if ocean_mask.shape != self._lon2d.shape:
                raise ValueError("Ocean mask must match the longitude/latitude grid shape")
            self._ocean_mask = np.asarray(ocean_mask, dtype=bool)
        else:
            self._ocean_mask = np.ones_like(self._lon2d, dtype=bool)

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

        coriolis = 2.0 * config.earth_rotation_rate_rad_s * np.sin(
            np.deg2rad(self._lat2d)
        )
        self._coriolis = coriolis
        self._abs_coriolis = np.abs(coriolis)

        if config.ekman_friction:
            boundary_layer_depth = float(config.boundary_layer_depth_m)
            if boundary_layer_depth <= 0.0:
                raise ValueError("Boundary-layer depth must be positive when using Ekman friction")

            land_viscosity = float(config.land_eddy_viscosity_m2_s)
            ocean_viscosity = float(config.ocean_eddy_viscosity_m2_s)
            if land_viscosity < 0.0 or ocean_viscosity < 0.0:
                raise ValueError("Eddy viscosities must be non-negative")

            viscosity = np.where(
                self._ocean_mask, ocean_viscosity, land_viscosity
            )
            tau = np.full_like(self._lon2d, np.inf, dtype=float)
            valid = viscosity > 0.0
            if np.any(valid):
                denom = (np.pi**2) * viscosity[valid]
                tau[valid] = (boundary_layer_depth**2) / denom
            inverse_tau = np.zeros_like(self._lon2d, dtype=float)
            finite = np.isfinite(tau) & (tau > 0.0)
            inverse_tau[finite] = 1.0 / tau[finite]
        else:
            inverse_tau = np.zeros_like(self._lon2d, dtype=float)

        self._inverse_tau = inverse_tau

    @property
    def enabled(self) -> bool:
        return self._config.enabled

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

        grad_x, grad_y = self._horizontal_gradient(temperature)
        temp_safe = np.maximum(temperature, self._config.minimum_temperature_K)
        scale = self._config.gravity_m_s2 * self._config.troposphere_scale_height_m
        forcing_x = -scale * grad_x / temp_safe
        forcing_y = -scale * grad_y / temp_safe
        forcing_x = np.where(np.isfinite(forcing_x), forcing_x, 0.0)
        forcing_y = np.where(np.isfinite(forcing_y), forcing_y, 0.0)

        velocity_x = np.zeros_like(temperature)
        velocity_y = np.zeros_like(temperature)

        abs_coriolis = self._abs_coriolis
        valid = abs_coriolis >= self._config.coriolis_floor_s
        if np.any(valid):
            coriolis = self._coriolis[valid]
            inverse_tau = self._inverse_tau[valid]
            px = forcing_x[valid]
            py = forcing_y[valid]
            det = inverse_tau**2 + coriolis**2
            nonzero = det > 0.0
            if np.any(nonzero):
                ux = np.zeros_like(px)
                uy = np.zeros_like(py)
                ux[nonzero] = (-px[nonzero] * inverse_tau[nonzero] - coriolis[nonzero] * py[nonzero]) / det[nonzero]
                uy[nonzero] = (
                    coriolis[nonzero] * px[nonzero] - py[nonzero] * inverse_tau[nonzero]
                ) / det[nonzero]
                velocity_x[valid] = ux
                velocity_y[valid] = uy

        tendency = -(velocity_x * grad_x + velocity_y * grad_y)
        return tendency
