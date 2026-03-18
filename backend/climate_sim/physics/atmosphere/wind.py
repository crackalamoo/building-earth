"""Advection utilities for atmospheric heat transport via wind."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.data.elevation import (
    load_elevation_data,
    compute_cell_elevation,
    compute_cell_roughness_length,
    neutral_drag_from_roughness_length,
    WATER_ROUGHNESS_LENGTH_M,
)
from climate_sim.data.constants import BOUNDARY_LAYER_HEIGHT_M, ATMOSPHERE_LAYER_HEIGHT_M
from climate_sim.physics.atmosphere.pressure import (
    compute_pressure,
    compute_geopotential_height,
    _smooth_temperature_field,
    _get_latitude_centers,
)
from climate_sim.physics.atmosphere.atmosphere import (
    compute_two_meter_temperature,
    log_law_map_wind_speed,
)
from climate_sim.data.constants import GAS_CONSTANT_J_KG_K, R_EARTH_METERS
from climate_sim.core.math_core import (
    regular_latitude_edges,
    regular_longitude_edges,
    spherical_cell_area,
    area_weighted_mean,
)
from climate_sim.data.landmask import compute_land_mask
from climate_sim.data.elevation import VON_KARMAN_CONSTANT

def compute_surface_roughness(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    """Return the neutral 10 m drag coefficient over the provided grid."""

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")
    if lon2d.shape != land_mask.shape:
        raise ValueError("Land mask must match the longitude/latitude grid shape")

    land_mask_bool = np.asarray(land_mask, dtype=bool)

    elevation_data = load_elevation_data()
    assert elevation_data is not None, "Elevation data could not be loaded"
    land_shape = land_mask_bool.shape

    lat_centers = lat2d[:, 0]
    lon_centers = lon2d[0, :]

    lat_edges = regular_latitude_edges(lat_centers)
    lon_edges = regular_longitude_edges(lon_centers)

    high_res_deg = 0.05

    def _aligned_edges(min_edge: float, max_edge: float) -> np.ndarray:
        start = np.floor(min_edge / high_res_deg) * high_res_deg
        end = np.ceil(max_edge / high_res_deg) * high_res_deg
        return np.arange(start, end + 1e-9, high_res_deg)

    lat_edges_hr = _aligned_edges(lat_edges[0], lat_edges[-1])
    lon_edges_hr = _aligned_edges(lon_edges[0], lon_edges[-1])

    if lat_edges_hr.size < 2 or lon_edges_hr.size < 2:
        raise ValueError("High-resolution grid could not be constructed for roughness aggregation")

    lat_centers_hr = lat_edges_hr[:-1] + 0.5 * high_res_deg
    lon_centers_hr = lon_edges_hr[:-1] + 0.5 * high_res_deg

    lon_hr2d, lat_hr2d = np.meshgrid(lon_centers_hr, lat_centers_hr)

    roughness_hr = compute_cell_roughness_length(
        lon_hr2d, lat_hr2d, data=elevation_data
    )
    area_hr = spherical_cell_area(lon_hr2d, lat_hr2d, earth_radius_m=R_EARTH_METERS)

    lat_idx_hr = np.searchsorted(lat_edges, lat_centers_hr, side="right") - 1
    lon_idx_hr = np.searchsorted(lon_edges, lon_centers_hr, side="right") - 1

    lat_idx_matrix = lat_idx_hr[:, np.newaxis]
    lon_idx_matrix = lon_idx_hr[np.newaxis, :]

    lat_idx_flat = np.broadcast_to(lat_idx_matrix, roughness_hr.shape).ravel()
    lon_idx_flat = np.broadcast_to(lon_idx_matrix, roughness_hr.shape).ravel()
    values_flat = roughness_hr.ravel()
    weights_flat = area_hr.ravel()

    valid = (
        (lat_idx_flat >= 0)
        & (lat_idx_flat < land_shape[0])
        & (lon_idx_flat >= 0)
        & (lon_idx_flat < land_shape[1])
    )

    weighted_sum = np.zeros(land_shape, dtype=float)
    weight_sum = np.zeros(land_shape, dtype=float)

    np.add.at(
        weighted_sum,
        (lat_idx_flat[valid], lon_idx_flat[valid]),
        values_flat[valid] * weights_flat[valid],
    )
    np.add.at(
        weight_sum,
        (lat_idx_flat[valid], lon_idx_flat[valid]),
        weights_flat[valid],
    )

    if np.any(weight_sum <= 0.0):
        raise ValueError("Encountered zero total area during roughness aggregation")

    with np.errstate(divide="ignore", invalid="ignore"):
        aggregated_roughness = weighted_sum / weight_sum

    roughness_map = np.where(
        land_mask_bool, aggregated_roughness, WATER_ROUGHNESS_LENGTH_M
    )
    return neutral_drag_from_roughness_length(roughness_map)


@dataclass(frozen=True)
class WindConfig:
    """Configuration for wind model."""
    enabled: bool = True
    earth_rotation_rate_rad_s: float = 7.2921e-5
    gravity_m_s2: float = 9.81
    troposphere_scale_height_m: float = 8000.0
    coriolis_floor_s: float = 1e-5
    minimum_temperature_K: float = 150.0

class WindModel:
    """Evaluate advection on a fixed longitude/latitude grid."""

    def __init__(
        self,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        *,
        config: WindConfig,
        roughness_length: np.ndarray | None = None,
        land_mask: np.ndarray | None = None,
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
        self._delta_y = R_EARTH_METERS * np.deg2rad(float(lat_spacing[0]))

        lon_spacing = np.diff(lon_centers)
        if not np.allclose(lon_spacing, lon_spacing[0]):
            raise ValueError("Longitude grid must have constant spacing for gradients")
        delta_lon_rad = np.deg2rad(float(lon_spacing[0]))
        cos_lat = np.cos(np.deg2rad(lat_centers))[:, np.newaxis]
        delta_x = R_EARTH_METERS * cos_lat * delta_lon_rad
        with np.errstate(divide="ignore", invalid="ignore"):
            self._inv_two_delta_x = np.zeros_like(delta_x)
            valid = np.abs(delta_x) > 0.0
            self._inv_two_delta_x[valid] = 1.0 / (2.0 * delta_x[valid])

        coriolis = 2.0 * config.earth_rotation_rate_rad_s * np.sin(
            np.deg2rad(self._lat2d)
        )
        self._coriolis = coriolis

        self._land_mask = (
            land_mask if land_mask is not None
            else compute_land_mask(self._lon2d, self._lat2d)
        )

        elevation_data = load_elevation_data()
        assert elevation_data is not None, "Elevation data could not be loaded"

        if roughness_length is not None:
            self._roughness_length = roughness_length
            self._drag_coefficient = neutral_drag_from_roughness_length(roughness_length)
        else:
            self._drag_coefficient = compute_surface_roughness(
                self._lon2d, self._lat2d, self._land_mask
            )
            self._roughness_length = compute_cell_roughness_length(
                self._lon2d, self._lat2d, data=elevation_data, land_mask=self._land_mask
            )

        self.elevation_m = compute_cell_elevation(
            self._lon2d, self._lat2d, data=elevation_data
        )
        self.elevation_m = np.maximum(self.elevation_m, 0.0)

        # Pre-compute bulk transfer coefficient (constant for the grid)
        log_height_surface = 10.0
        roughness_momentum = self._roughness_length
        roughness_heat = np.maximum(roughness_momentum / 10.0, 1.0e-9)

        lm = np.log(
            np.maximum(log_height_surface / roughness_momentum, 1.0 + 1.0e-9)
        )
        lh = np.log(
            np.maximum(log_height_surface / roughness_heat, 1.0 + 1.0e-9)
        )
        ch_raw = (VON_KARMAN_CONSTANT**2) / (lm * lh)

        ch_land = np.clip(ch_raw, 1e-4, 2.0e-3)
        ch_ocean = np.clip(ch_raw, 3e-4, 3.0e-3)
        self._bulk_transfer_coefficient = np.where(self._land_mask, ch_land, ch_ocean)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def wind_field(
        self,
        temperature: np.ndarray,
        temperature_boundary_layer: np.ndarray | None = None,
        itcz_rad: np.ndarray | None = None,
        ekman_drag: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the geostrophic wind field (u, v, speed) for the given temperatures."""

        if temperature.shape != self._lon2d.shape:
            raise ValueError(
                "Temperature field must match the longitude/latitude grid shape"
            )

        if not self.enabled:
            zeros = np.zeros_like(temperature)
            return zeros, zeros, zeros

        # Determine which pressure level this wind represents
        # BL wind: ~925 hPa, Free atmosphere: ~700 hPa

        if ekman_drag:
            reference_pressure_pa = 92500.0  # 925 hPa
        else:
            reference_pressure_pa = 70000.0  # 700 hPa

        if temperature_boundary_layer is not None:
            # Mass-weighted column mean for pressure gradient calculation.
            # SLP reflects the weight of the entire column; density decays
            # exponentially with scale height H ≈ 8500 m, so mass fraction
            # is proportional to ∫ exp(-z/H) dz over each layer's extent.
            _H = 8500.0  # scale height (m)
            bl_mass = _H * (1.0 - np.exp(-BOUNDARY_LAYER_HEIGHT_M / _H))
            atm_top = BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M
            atm_mass = _H * (
                np.exp(-BOUNDARY_LAYER_HEIGHT_M / _H)
                - np.exp(-atm_top / _H)
            )
            total_mass = bl_mass + atm_mass
            column_temperature = (
                temperature_boundary_layer * bl_mass +
                temperature * atm_mass
            ) / total_mass
        else:
            # Single-layer case or missing BL temp
            column_temperature = temperature

        # Compute geopotential height using smoothed temperature and full pressure field
        # This removes spurious small-scale variations while preserving large-scale patterns

        # Smooth the column temperature to get large-scale patterns (same as compute_pressure)
        nlat, nlon = column_temperature.shape
        lat_deg = _get_latitude_centers(nlat)
        cos_lat = np.clip(np.cos(np.deg2rad(lat_deg)), 1.0e-6, None)
        weights = np.broadcast_to(cos_lat[:, None], column_temperature.shape)

        column_temperature_smooth = _smooth_temperature_field(
            column_temperature, lat_deg,
        )

        # Preserve global mean (same as compute_pressure does)
        target_mean = area_weighted_mean(column_temperature, weights)
        smooth_mean = area_weighted_mean(column_temperature_smooth, weights)
        column_temperature_smooth = column_temperature_smooth + (target_mean - smooth_mean)

        # Use full pressure field (with Hadley + thermal patterns) for p_SLP
        # This is what provides the large-scale circulation patterns
        # Skip smoothing since we already smoothed the temperature
        p_SLP = compute_pressure(
            column_temperature_smooth, itcz_rad=itcz_rad, skip_smoothing=True,
            lat2d=self._lat2d, lon2d=self._lon2d
        )

        # Compute geopotential height of the reference pressure surface
        geopotential = compute_geopotential_height(
            column_temperature_smooth,
            reference_pressure_pa,
            p_SLP,
            gravity_m_s2=self._config.gravity_m_s2,
        )

        geostrophic = self._compute_geostrophic_wind_components(
            temperature,
            config=self._config,
            geopotential=geopotential,
            layer_pressure=reference_pressure_pa,
        )

        u_geo, v_geo, speed_geo, weight_geo = geostrophic

        if ekman_drag:
            u_final, v_final, speed_final = self._apply_surface_drag(
                u_geo, v_geo, speed_geo, weight_geo=weight_geo,
            )
            return u_final, v_final, speed_final
        else:
            return u_geo, v_geo, speed_geo

    def _compute_geostrophic_wind_components(
        self,
        temperature: np.ndarray,
        *,
        config: WindConfig,
        geopotential: np.ndarray,
        layer_pressure: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return wind from blended geostrophic and friction-balanced dynamics.

        At mid-latitudes (|f| >> friction), uses geostrophic balance.
        Near the equator (|f| ~ friction), blends toward friction-balanced flow
        where wind blows directly down the pressure gradient.
        """
        grad_x, grad_y = self._horizontal_gradient(geopotential)

        # Pressure gradient from geopotential: ∇p = ρg∇Z
        # For wind calculation, we need ∇p/ρ = g∇Z
        # grad_x, grad_y are ∂Z/∂x, ∂Z/∂y in meters
        g = config.gravity_m_s2
        dp_dx_over_rho = g * grad_x  # (1/ρ) ∂p/∂x, units m/s²
        dp_dy_over_rho = g * grad_y  # (1/ρ) ∂p/∂y, units m/s²

        # Friction parameter k = Cd / h_m (same as in _apply_surface_drag)
        # For this calculation, use ocean value as representative
        k = 3e-4 / 150.0  # ~2e-6 /m

        # === Geostrophic wind (mid-latitudes) ===
        # u_geo = -(1/f) * (1/ρ) ∂p/∂y = -(g/f) * ∂Z/∂y
        # v_geo = (1/f) * (1/ρ) ∂p/∂x = (g/f) * ∂Z/∂x
        coriolis_abs = np.abs(self._coriolis)
        sign = np.sign(self._coriolis)
        sign[sign == 0.0] = 1.0

        # Use a small floor just to avoid division by zero (blending handles equator)
        f_safe = np.maximum(coriolis_abs, 1e-10)

        u_geo = -dp_dy_over_rho / (sign * f_safe)
        v_geo = dp_dx_over_rho / (sign * f_safe)

        # === Friction-balanced wind (equator) ===
        # At equator: k|u|u = -(1/ρ)∇p, so wind blows down pressure gradient
        # |u| = sqrt(|∇p/ρ| / k)
        # Direction: opposite to pressure gradient (toward low pressure)
        grad_mag = np.sqrt(dp_dx_over_rho**2 + dp_dy_over_rho**2)
        speed_fric = np.sqrt(np.maximum(grad_mag / k, 0))

        # Unit vector pointing down pressure gradient (toward low pressure)
        grad_mag_safe = np.maximum(grad_mag, 1e-10)
        u_fric = -dp_dx_over_rho / grad_mag_safe * speed_fric
        v_fric = -dp_dy_over_rho / grad_mag_safe * speed_fric

        # === Blend based on Ekman number E = k|u|/|f| ===
        # When E << 1: geostrophic dominates
        # When E >> 1: friction dominates
        # Use a smooth transition based on |f| relative to a critical value
        # Critical |f| where friction and Coriolis are comparable:
        # k|u| ~ |f| → |f|_crit ~ k * typical_speed ~ 2e-6 * 10 = 2e-5
        # This corresponds to ~8° latitude
        f_crit = 2e-5  # Transition centered around this |f|

        # Smooth blending weight: 1 = pure geostrophic, 0 = pure friction
        # Use tanh for smooth transition over ~5° latitude
        transition_width = 1e-5  # Controls sharpness of transition
        weight_geo = 0.5 * (1 + np.tanh((coriolis_abs - f_crit) / transition_width))

        # Blend winds
        velocity_x = weight_geo * u_geo + (1 - weight_geo) * u_fric
        velocity_y = weight_geo * v_geo + (1 - weight_geo) * v_fric
        speed = np.hypot(velocity_x, velocity_y)

        return velocity_x, velocity_y, speed, weight_geo

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

    def _apply_surface_drag(
        self,
        u_geo: np.ndarray,
        v_geo: np.ndarray,
        speed_geo: np.ndarray,
        weight_geo: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Ekman drag (speed reduction + turning) scaled by weight_geo.

        At mid-latitudes (weight_geo ≈ 1) the input is geostrophic, so full
        Rayleigh drag and Coriolis turning are applied.  Near the equator
        (weight_geo ≈ 0) the input is already friction-balanced (speed and
        direction already account for drag), so drag and turning are
        suppressed to avoid double-counting.
        """

        drag_coeff = self._drag_coefficient
        coriolis = self._coriolis
        coriolis_abs = np.maximum(np.abs(coriolis), self._config.coriolis_floor_s)

        # Effective depth over which surface stress is distributed.
        h_m = np.where(self._land_mask, 400.0, 500.0)
        k = drag_coeff / h_m

        Ug = np.maximum(speed_geo, 0.0)
        a = (k / coriolis_abs) ** 2  # (k/f)^2

        eps = 1e-16
        general = a > eps
        y = np.empty_like(Ug)

        if np.any(general):
            sqrt_term = np.sqrt(1.0 + 4.0 * a[general] * (Ug[general] ** 2))
            y[general] = (-1.0 + sqrt_term) / (2.0 * a[general])

        if np.any(~general):
            y[~general] = Ug[~general] ** 2

        y = np.clip(y, 0.0, None)
        u_mag_full = np.sqrt(y)

        zero_geo = Ug <= 1.0e-12
        if np.any(zero_geo):
            u_mag_full[zero_geo] = 0.0

        # Blend speed reduction: geostrophic part gets full drag,
        # friction-balanced part keeps its original speed.
        u_mag = weight_geo * u_mag_full + (1.0 - weight_geo) * Ug

        r = k * u_mag_full
        if np.any(zero_geo):
            r[zero_geo] = 0.0

        r_over_f = r / coriolis_abs
        alpha = np.arctan(r_over_f)
        if np.any(zero_geo):
            alpha[zero_geo] = 0.0

        # Scale turning angle by weight_geo: friction-balanced wind
        # already points in the correct (down-gradient) direction.
        alpha = alpha * weight_geo

        # Ekman turning: wind rotates toward low pressure in the friction layer.
        # NH (f > 0): counterclockwise turn; SH (f < 0): clockwise turn.
        rotation_angle = np.where(coriolis >= 0.0, alpha, -alpha)
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)

        Ug_safe = np.maximum(Ug, 1e-12)
        ux = np.where(Ug_safe > 0.0, u_geo / Ug_safe, 0.0)
        vy = np.where(Ug_safe > 0.0, v_geo / Ug_safe, 0.0)

        # Counterclockwise rotation: [cos -sin; sin cos]
        ux_rot = ux * cos_a - vy * sin_a
        vy_rot = ux * sin_a + vy * cos_a

        u_final = u_mag * ux_rot
        v_final = u_mag * vy_rot
        speed_final = u_mag

        if np.any(zero_geo):
            speed_final[zero_geo] = 0.0

        return u_final, v_final, speed_final

    def compute_atmospheric_properties(
        self,
        surface_temperature_K: np.ndarray,
        atmosphere_temperature_K: np.ndarray,
        wind_speed_reference_m_s: np.ndarray | None,
        *,
        itcz_rad: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute atmospheric properties needed for heat exchange calculations.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - pressure: Local surface pressure in Pa (elevation-corrected)
            - air_density: Air density in kg/m³ at 2m height
            - wind_speed_10m: Wind speed at 10 m in m/s
            - bulk_transfer_coefficient: Dimensionless bulk transfer coefficient
        """
        if surface_temperature_K.shape != self._lon2d.shape:
            raise ValueError("Surface temperature must match grid shape")
        if atmosphere_temperature_K.shape != self._lon2d.shape:
            raise ValueError("Atmosphere temperature must match grid shape")

        # Compute pressure from atmosphere temperature, accounting for elevation
        # This gives local surface pressure, not sea-level pressure
        pressure = compute_pressure(
            atmosphere_temperature_K,
            elevation_m=self.elevation_m,
            itcz_rad=itcz_rad,
            lat2d=self._lat2d,
            lon2d=self._lon2d,
        )

        # Map wind speed to 10 m height
        # Reference height corresponds to Ekman boundary layer: 0.5 * h_m
        # where h_m = 400m (land) or 1000m (ocean) from _apply_surface_drag
        if wind_speed_reference_m_s is None:
            wind_speed_10m = np.zeros_like(surface_temperature_K)
        else:
            height_ref_m = np.where(self._land_mask, 200.0, 500.0)
            wind_speed_10m = log_law_map_wind_speed(
                wind_speed_reference_m_s,
                height_ref_m=height_ref_m,
                height_target_m=10,
                roughness_length_m=self._roughness_length,
            )

        # Compute near-surface air temperature for density calculation
        near_surface_air_K = compute_two_meter_temperature(
            atmosphere_temperature_K,
            surface_temperature_K,
            topographic_elevation=self.elevation_m,
        )

        # Air density
        air_density = pressure / (GAS_CONSTANT_J_KG_K * near_surface_air_K)

        # Return pre-computed bulk transfer coefficient
        return pressure, air_density, wind_speed_10m, self._bulk_transfer_coefficient
