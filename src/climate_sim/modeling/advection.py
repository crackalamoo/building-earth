"""Geostrophic advection utilities for atmospheric heat transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


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

        self._cos_lat = np.cos(np.deg2rad(self._lat2d))

        nlat, nlon = self._lon2d.shape
        if nlat < 1 or nlon < 1:
            raise ValueError("Longitude/latitude grids must be non-empty")

        lat_centers = self._lat2d[:, 0]
        lon_centers = self._lon2d[0, :]

        self._delta_phi_rad: float | None
        if nlat > 1:
            lat_spacing = np.diff(lat_centers)
            if not np.allclose(lat_spacing, lat_spacing[0]):
                raise ValueError("Latitude grid must have constant spacing for gradients")
            self._delta_phi_rad = float(np.deg2rad(lat_spacing[0]))
            self._delta_y = config.earth_radius_m * self._delta_phi_rad
        else:
            self._delta_y = np.inf
            self._delta_phi_rad = None

        self._delta_lambda_rad: float | None
        if nlon > 1:
            lon_spacing = np.diff(lon_centers)
            if not np.allclose(lon_spacing, lon_spacing[0]):
                raise ValueError("Longitude grid must have constant spacing for gradients")
            delta_lon_rad = float(np.deg2rad(lon_spacing[0]))
            cos_lat = np.cos(np.deg2rad(lat_centers))[:, np.newaxis]
            delta_x = config.earth_radius_m * cos_lat * delta_lon_rad
            with np.errstate(divide="ignore", invalid="ignore"):
                self._inv_two_delta_x = np.zeros_like(delta_x)
                valid = np.abs(delta_x) > 0.0
                self._inv_two_delta_x[valid] = 1.0 / (2.0 * delta_x[valid])
            self._delta_lambda_rad = delta_lon_rad
        else:
            self._inv_two_delta_x = np.zeros_like(self._lon2d)
            self._delta_lambda_rad = None

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

    def velocity_field(self, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the (u, v) geostrophic wind components for ``temperature``."""

        if not self.enabled:
            shape = temperature.shape
            return np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)

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

        return velocity_x, velocity_y

    def _flux_form_tendency(
        self, temperature: np.ndarray, velocity_x: np.ndarray, velocity_y: np.ndarray
    ) -> np.ndarray:
        cos_lat = self._cos_lat
        tendency = np.zeros_like(temperature)

        if temperature.shape[1] > 1 and self._delta_lambda_rad is not None:
            u_east = 0.5 * (velocity_x + np.roll(velocity_x, -1, axis=1))
            temp_east = np.where(
                u_east >= 0.0, temperature, np.roll(temperature, -1, axis=1)
            )
            flux_east = temp_east * u_east * cos_lat
            flux_west = np.roll(flux_east, 1, axis=1)
            div_lambda = (flux_east - flux_west) / self._delta_lambda_rad
        else:
            div_lambda = np.zeros_like(temperature)

        if temperature.shape[0] > 1 and self._delta_phi_rad is not None:
            v_faces = np.zeros((temperature.shape[0] + 1, temperature.shape[1]), dtype=float)
            v_faces[1:-1] = 0.5 * (velocity_y[:-1] + velocity_y[1:])

            temp_faces = np.zeros_like(v_faces)
            temp_faces[1:-1] = np.where(
                v_faces[1:-1] >= 0.0, temperature[:-1], temperature[1:]
            )
            flux_phi = temp_faces * v_faces
            div_phi = (flux_phi[1:] - flux_phi[:-1]) / self._delta_phi_rad
        else:
            div_phi = np.zeros_like(temperature)

        denom = self._config.earth_radius_m * cos_lat
        with np.errstate(divide="ignore", invalid="ignore"):
            tendency = -(div_lambda + div_phi) / denom

        tendency = np.where(np.isfinite(tendency), tendency, 0.0)
        tendency = np.where(np.abs(denom) > 0.0, tendency, 0.0)
        return tendency

    def _flux_form_matrix(
        self, velocity_x: np.ndarray, velocity_y: np.ndarray
    ) -> sparse.csr_matrix | None:
        nlat, nlon = velocity_x.shape
        size = nlat * nlon
        if size == 0:
            return sparse.csr_matrix((0, 0))

        cos_lat = self._cos_lat
        valid_cos = np.abs(cos_lat) > 1.0e-12
        cell_index = (
            np.arange(nlat, dtype=int)[:, np.newaxis] * nlon
            + np.arange(nlon, dtype=int)[np.newaxis, :]
        )

        row_entries: list[np.ndarray] = []
        col_entries: list[np.ndarray] = []
        data_entries: list[np.ndarray] = []

        earth_radius = self._config.earth_radius_m

        if nlon > 1 and self._delta_lambda_rad is not None:
            u_east = 0.5 * (velocity_x + np.roll(velocity_x, -1, axis=1))
            east_source = np.where(
                u_east >= 0.0,
                cell_index,
                np.roll(cell_index, -1, axis=1),
            )
            coeff = -u_east / (earth_radius * self._delta_lambda_rad)
            coeff = np.where(valid_cos, coeff, 0.0)

            row_entries.append(cell_index.ravel())
            col_entries.append(east_source.ravel())
            data_entries.append(coeff.ravel())

            u_west = np.roll(u_east, 1, axis=1)
            west_from_neighbor = np.roll(u_east >= 0.0, 1, axis=1)
            west_source = np.where(
                west_from_neighbor,
                np.roll(cell_index, 1, axis=1),
                cell_index,
            )
            coeff_west = u_west / (earth_radius * self._delta_lambda_rad)
            coeff_west = np.where(valid_cos, coeff_west, 0.0)

            row_entries.append(cell_index.ravel())
            col_entries.append(west_source.ravel())
            data_entries.append(coeff_west.ravel())

        if nlat > 1 and self._delta_phi_rad is not None:
            v_faces = np.zeros((nlat + 1, nlon), dtype=float)
            v_faces[1:-1] = 0.5 * (velocity_y[:-1] + velocity_y[1:])

            inv_cos = np.zeros_like(cos_lat)
            inv_cos[valid_cos] = 1.0 / cos_lat[valid_cos]
            base = 1.0 / (earth_radius * self._delta_phi_rad)

            north_faces = v_faces[1:]
            north_source = np.where(
                north_faces >= 0.0,
                cell_index,
                np.roll(cell_index, -1, axis=0),
            )
            coeff_north = -north_faces * base * inv_cos
            coeff_north = np.where(valid_cos, coeff_north, 0.0)

            row_entries.append(cell_index.ravel())
            col_entries.append(north_source.ravel())
            data_entries.append(coeff_north.ravel())

            south_faces = v_faces[:-1]
            south_source = np.where(
                south_faces >= 0.0,
                np.roll(cell_index, 1, axis=0),
                cell_index,
            )
            coeff_south = south_faces * base * inv_cos
            coeff_south = np.where(valid_cos, coeff_south, 0.0)

            row_entries.append(cell_index.ravel())
            col_entries.append(south_source.ravel())
            data_entries.append(coeff_south.ravel())

        if not row_entries:
            return sparse.csr_matrix((size, size))

        rows = np.concatenate(row_entries)
        cols = np.concatenate(col_entries)
        data = np.concatenate(data_entries)

        nonzero = np.abs(data) > 0.0
        if not np.any(nonzero):
            return sparse.csr_matrix((size, size))

        matrix = sparse.csr_matrix((data[nonzero], (rows[nonzero], cols[nonzero])), shape=(size, size))
        matrix.sum_duplicates()
        return matrix

    def tendency(self, temperature: np.ndarray) -> np.ndarray:
        """Return the geostrophic advection tendency (K/s) for the given field."""

        if not self.enabled:
            return np.zeros_like(temperature)

        velocity_x, velocity_y = self.velocity_field(temperature)
        return self._flux_form_tendency(temperature, velocity_x, velocity_y)

    def linearised_matrix(self, temperature: np.ndarray) -> sparse.csr_matrix | None:
        """Return the linearised advection operator for ``temperature``."""

        if not self.enabled:
            return None

        velocity_x, velocity_y = self.velocity_field(temperature)
        return self._flux_form_matrix(velocity_x, velocity_y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lats = np.linspace(-80.0, 80.0, 33)
    lons = np.linspace(0.0, 360.0, 65, endpoint=False)
    lon2d, lat2d = np.meshgrid(lons, lats)

    base_temp = 288.0 - 40.0 * np.sin(np.deg2rad(lat2d)) ** 2
    wave = 5.0 * np.cos(np.deg2rad(lon2d)) * np.cos(np.deg2rad(lat2d))
    temperature = base_temp + wave

    ocean_mask = lat2d < 0.0
    config = GeostrophicAdvectionConfig()
    operator = GeostrophicAdvectionOperator(
        lon2d, lat2d, ocean_mask=ocean_mask, config=config
    )

    u, v = operator.velocity_field(temperature)
    tendency = operator.tendency(temperature)
    speed = np.hypot(u, v)

    plt.figure(figsize=(10, 5))
    quiver = plt.quiver(lon2d, lat2d, u, v, speed, scale=200.0, pivot="middle")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title("Diagnostic geostrophic velocity field")
    plt.colorbar(quiver, label="Speed (m s⁻¹)")
    plt.contour(lon2d, lat2d, temperature, colors="k", linewidths=0.5)
    plt.tight_layout()
    plt.show()
