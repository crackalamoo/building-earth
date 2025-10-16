"""Geostrophic advection utilities for atmospheric heat transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _compute_geostrophic_wind_components(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    temperature: np.ndarray,
    *,
    abs_coriolis: np.ndarray,
    lat_sign: np.ndarray,
    config: "GeostrophicAdvectionConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the geostrophic wind given horizontal temperature gradients."""

    grad_mag = np.hypot(grad_x, grad_y)

    temp_safe = np.maximum(temperature, config.minimum_temperature_K)
    scale = config.gravity_m_s2 * config.troposphere_scale_height_m
    coriolis = np.maximum(abs_coriolis, config.coriolis_floor_s)

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
    earth_radius_m: float = 6.371e6
    earth_rotation_rate_rad_s: float = 7.2921e-5
    gravity_m_s2: float = 9.81
    troposphere_scale_height_m: float = 8000.0
    coriolis_floor_s: float = 1.0e-5
    minimum_temperature_K: float = 150.0


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

        grad_x, grad_y = self._horizontal_gradient(temperature)
        return _compute_geostrophic_wind_components(
            grad_x,
            grad_y,
            temperature,
            abs_coriolis=self._abs_coriolis,
            lat_sign=self._lat_sign,
            config=self._config,
        )

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
        velocity_x, velocity_y, _speed = _compute_geostrophic_wind_components(
            grad_x,
            grad_y,
            temperature,
            abs_coriolis=self._abs_coriolis,
            lat_sign=self._lat_sign,
            config=self._config,
        )

        tendency = -(velocity_x * grad_x + velocity_y * grad_y)
        return tendency


if __name__ == "__main__":
    import argparse

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.widgets import Slider

    from climate_sim.modeling.diffusion import DiffusionConfig
    from climate_sim.modeling.radiation import RadiationConfig
    from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
    from climate_sim.utils.solver import compute_periodic_cycle_celsius

    parser = argparse.ArgumentParser(
        description="Visualise the monthly geostrophic wind diagnosed from the model."
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Grid resolution in degrees (must be positive).",
    )
    args = parser.parse_args()

    if args.resolution <= 0:
        raise ValueError("resolution must be positive")

    radiation_config = RadiationConfig()
    diffusion_config = DiffusionConfig()
    advection_config = GeostrophicAdvectionConfig()
    snow_config = SnowAlbedoConfig()

    lon2d, lat2d, layers = compute_periodic_cycle_celsius(
        resolution_deg=args.resolution,
        radiation_config=radiation_config,
        diffusion_config=diffusion_config,
        advection_config=advection_config,
        snow_config=snow_config,
        return_layer_map=True,
    )

    if "atmosphere" in layers:
        temperature_cycle_c = layers["atmosphere"]
        layer_label = "Atmosphere"
    else:
        temperature_cycle_c = layers["surface"]
        layer_label = "Surface"

    temperature_cycle_k = temperature_cycle_c + 273.15

    operator = GeostrophicAdvectionOperator(
        lon2d,
        lat2d,
        config=advection_config,
    )

    months = temperature_cycle_k.shape[0]
    wind_u = np.zeros_like(temperature_cycle_k)
    wind_v = np.zeros_like(temperature_cycle_k)
    wind_speed = np.zeros_like(temperature_cycle_k)

    for idx in range(months):
        u_field, v_field, speed_field = operator.wind_field(temperature_cycle_k[idx])
        wind_u[idx] = u_field
        wind_v[idx] = v_field
        wind_speed[idx] = speed_field

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    max_speed = float(np.max(wind_speed))
    if not np.isfinite(max_speed) or max_speed <= 0.0:
        max_speed = 1.0

    stride = max(1, int(round(1.0 / args.resolution)))
    lon_sampled = lon2d[::stride, ::stride]
    lat_sampled = lat2d[::stride, ::stride]

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=projection))
    fig.subplots_adjust(bottom=0.18)

    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#555555")
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "lakes", "110m", edgecolor="#333333", facecolor="none"
        ),
        linewidth=0.2,
    )
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)

    norm = Normalize(vmin=0.0, vmax=max_speed)
    cmap = plt.cm.viridis

    u0 = wind_u[0, ::stride, ::stride]
    v0 = wind_v[0, ::stride, ::stride]
    s0 = wind_speed[0, ::stride, ::stride]

    quiver = ax.quiver(
        lon_sampled,
        lat_sampled,
        u0,
        v0,
        s0,
        cmap=cmap,
        norm=norm,
        transform=projection,
        scale_units="xy",
        scale=max_speed * 1.5,
        pivot="middle",
    )

    cbar = fig.colorbar(quiver, ax=ax, orientation="vertical", pad=0.04, fraction=0.046)
    cbar.set_label("Wind speed (m/s)")

    reference_speed = min(max_speed, max(5.0, 0.25 * max_speed))
    ax.quiverkey(
        quiver,
        0.85,
        1.05,
        reference_speed,
        f"{reference_speed:.1f} m/s",
        labelpos="E",
        color="#333333",
    )

    initial_label = month_names[0 % len(month_names)]
    ax.set_title(f"Geostrophic Wind ({layer_label}) – {initial_label}")

    slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    month_slider = Slider(
        slider_ax,
        label="Month",
        valmin=0,
        valmax=months - 1,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
    )

    def update(month_idx: float) -> None:
        index = int(month_idx)
        u_field = wind_u[index, ::stride, ::stride]
        v_field = wind_v[index, ::stride, ::stride]
        speed_field = wind_speed[index, ::stride, ::stride]
        quiver.set_UVC(u_field, v_field, speed_field)
        month_label = month_names[index % len(month_names)]
        ax.set_title(f"Geostrophic Wind ({layer_label}) – {month_label}")
        fig.canvas.draw_idle()

    month_slider.on_changed(update)

    plt.show()
