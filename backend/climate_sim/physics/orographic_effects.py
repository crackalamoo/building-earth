"""Orographic effects on winds: terrain-induced vertical velocity and flow blocking.

Physics:
- Resolved orographic lifting: w = V · ∇h (wind forced upward over terrain slopes)
- Sub-grid orographic lifting: w = |V| × σ_h / L_subgrid (unresolved terrain variability)
- Flow blocking: when Froude number Fr = |V_n| / (N × h_eff) < 1, the cross-barrier
  wind component is reduced by Fr², representing flow going around rather than over terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.data.constants import R_EARTH_METERS


@dataclass(frozen=True)
class OrographicConfig:
    """Configuration for orographic wind effects."""

    enabled: bool = True
    subgrid_length_scale_factor: float = 0.5


class OrographicModel:
    """Pre-computes terrain gradients and sub-grid statistics for orographic effects.

    Parameters
    ----------
    lon2d, lat2d : np.ndarray
        2D coordinate arrays (degrees).
    elevation : np.ndarray
        Cell-mean elevation (m).
    elevation_std : np.ndarray
        Sub-grid elevation standard deviation (m).
    face_stats : dict[str, np.ndarray]
        Precomputed directional blockage ratios from
        ``compute_face_elevation_statistics``.  Values are open fractions
        (0 = fully blocked, 1 = fully open).
    config : OrographicConfig
        Physics configuration.
    """

    def __init__(
        self,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        elevation: np.ndarray,
        elevation_std: np.ndarray,
        face_stats: dict[str, np.ndarray],
        config: OrographicConfig,
        land_mask: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.lon2d = lon2d
        self.lat2d = lat2d
        self.elevation = elevation
        self.land_mask = land_mask if land_mask is not None else np.ones_like(elevation, dtype=bool)

        # Zero out sub-grid terrain stats over ocean (coastal variance is noise)
        self.elevation_std = np.where(self.land_mask, elevation_std, 0.0)

        # Pre-compute terrain gradients ∂h/∂x, ∂h/∂y (m/m)
        nlat, nlon = elevation.shape
        self.grad_x = np.zeros_like(elevation)
        self.grad_y = np.zeros_like(elevation)

        lat_centers = lat2d[:, 0]
        lon_centers = lon2d[0, :]

        if nlon > 1:
            dlon_rad = np.deg2rad(lon_centers[1] - lon_centers[0])
            dx = R_EARTH_METERS * np.cos(np.deg2rad(lat_centers)) * dlon_rad
            inv_2dx = np.zeros_like(dx)
            valid = np.abs(dx) > 0.0
            inv_2dx[valid] = 1.0 / (2.0 * dx[valid])
            padded = np.pad(elevation, ((0, 0), (1, 1)), mode="wrap")
            self.grad_x = (padded[:, 2:] - padded[:, :-2]) * inv_2dx[:, np.newaxis]

        if nlat > 1:
            dlat_rad = np.deg2rad(lat_centers[1] - lat_centers[0])
            dy = R_EARTH_METERS * dlat_rad
            if dy != 0.0:
                inv_2dy = 1.0 / (2.0 * dy)
                padded = np.pad(elevation, ((1, 1), (0, 0)), mode="edge")
                self.grad_y = (padded[2:, :] - padded[:-2, :]) * inv_2dy

        # Pre-compute sub-grid length scale (m)
        if nlon > 1:
            mean_dx = R_EARTH_METERS * np.cos(np.deg2rad(np.mean(np.abs(lat_centers)))) * dlon_rad
        else:
            mean_dx = R_EARTH_METERS * np.deg2rad(5.0)
        self.l_subgrid = config.subgrid_length_scale_factor * mean_dx

        # --- Directional blockage ratios (precomputed from fine-res data) ---
        # Open fraction: 0 = fully blocked, 1 = fully open
        self.r_east_pos = face_stats["r_east_pos"]      # for u > 0
        self.r_east_neg = face_stats["r_east_neg"]      # for u < 0
        self.r_north_pos = face_stats["r_north_pos"]    # for v > 0
        self.r_north_neg = face_stats["r_north_neg"]    # for v < 0

        # Diffusion barrier: use min open fraction of both directions (conservative)
        self.diffusion_barrier_east = np.minimum(self.r_east_pos, self.r_east_neg)
        self.diffusion_barrier_north = np.minimum(self.r_north_pos, self.r_north_neg)

    def compute_orographic_vertical_velocity(
        self, wind_u: np.ndarray, wind_v: np.ndarray
    ) -> np.ndarray:
        """Compute terrain-induced vertical velocity (m/s).

        Returns w > 0 for upward motion (upslope flow).
        """
        # Resolved: w = u·∂h/∂x + v·∂h/∂y
        w_resolved = wind_u * self.grad_x + wind_v * self.grad_y

        # Sub-grid: w = |V| × σ_h / L_subgrid (always upward — represents mean lifting)
        wind_speed = np.hypot(wind_u, wind_v)
        w_subgrid = wind_speed * self.elevation_std / self.l_subgrid

        return w_resolved + w_subgrid

    def apply_flow_blocking(
        self, wind_u: np.ndarray, wind_v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce wind by precomputed cross-sectional blockage ratios.

        Each face has a directional open fraction (0=blocked, 1=open)
        computed from fine-resolution terrain.  Eastward flow at cell j
        is scaled by the east face's open fraction for eastward flow;
        westward flow uses the west face's (= cell j-1's east face)
        open fraction for westward flow.

        Returns blocked (u, v).
        """
        u_out = wind_u * np.where(
            wind_u >= 0,
            self.r_east_pos,
            np.roll(self.r_east_neg, 1, axis=1),
        )
        v_out = wind_v * np.where(
            wind_v >= 0,
            self.r_north_pos,
            np.roll(self.r_north_neg, 1, axis=0),
        )
        return u_out, v_out

    def compute_effects(
        self, wind_u: np.ndarray, wind_v: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Main entry point: compute all orographic effects.

        Returns dict with:
        - w_orographic: terrain-induced vertical velocity (m/s)
        - wind_u_blocked, wind_v_blocked: flow-blocked wind components
        """
        u_blocked, v_blocked = self.apply_flow_blocking(wind_u, wind_v)
        w_oro = self.compute_orographic_vertical_velocity(u_blocked, v_blocked)

        return {
            "w_orographic": w_oro,
            "wind_u_blocked": u_blocked,
            "wind_v_blocked": v_blocked,
        }
