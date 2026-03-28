"""Orographic effects on winds: terrain-induced vertical velocity and flow blocking.

Physics:
- Resolved orographic lifting: w = V · ∇h (wind forced upward over terrain slopes)
- Sub-grid orographic lifting: w = |V| × σ_h / L_subgrid (unresolved terrain variability)
- Flow blocking: when Froude number Fr = |V_n| / (N × h_eff) < 1, the cross-barrier
  wind component is reduced by Fr², representing flow going around rather than over terrain.
"""

from dataclasses import dataclass

import numpy as np

from climate_sim.physics.precipitation import compute_precipitation_rh_gate


@dataclass(frozen=True)
class OrographicConfig:
    """Configuration for orographic wind effects."""

    enabled: bool = True
    # Efficiency of orographic precipitation: fraction of condensable moisture
    # removed per unit ascent.  P_oro = efficiency * max(w_oro, 0) * q * rho.
    orographic_precip_efficiency: float = 0.1


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

        # Directional orographic gradients from fine-res data (m/m).
        # Separated into positive (upslope) and negative (downslope) so that
        # orographic w reflects the actual mountain slope, not the smoothed
        # cell-mean gradient which cancels windward and leeward within one cell.
        self.grad_x_pos = face_stats["grad_x_pos"]  # mean(max(∂h/∂x, 0))
        self.grad_x_neg = face_stats["grad_x_neg"]  # mean(max(-∂h/∂x, 0))
        self.grad_y_pos = face_stats["grad_y_pos"]
        self.grad_y_neg = face_stats["grad_y_neg"]

        # --- Directional blockage ratios (precomputed from fine-res data) ---
        # Wind blocking (H=1000m BL depth): for advection
        self.r_east_pos = face_stats["r_east_pos"]  # for u > 0
        self.r_east_neg = face_stats["r_east_neg"]  # for u < 0
        self.r_north_pos = face_stats["r_north_pos"]  # for v > 0
        self.r_north_neg = face_stats["r_north_neg"]  # for v < 0

        # Eddy blocking (H=5000m moisture-weighted tropo depth): for diffusion
        # Use min of both directions (eddies are bidirectional)
        self.diffusion_barrier_east = np.minimum(
            face_stats["r_east_pos_eddy"], face_stats["r_east_neg_eddy"]
        )
        self.diffusion_barrier_north = np.minimum(
            face_stats["r_north_pos_eddy"], face_stats["r_north_neg_eddy"]
        )

    def compute_orographic_vertical_velocity(
        self, wind_u: np.ndarray, wind_v: np.ndarray
    ) -> np.ndarray:
        """Compute terrain-induced vertical velocity (m/s).

        Uses directional gradients from fine-resolution terrain data.
        For eastward flow (u > 0), the air encounters the mean positive
        ∂h/∂x (upslope terrain) within the cell.  For westward flow (u < 0),
        it encounters the mean positive -∂h/∂x.  This avoids the cancellation
        of windward and leeward slopes that plagues cell-mean gradients at
        coarse resolution.

        Returns w > 0 for upward motion (upslope flow).
        """
        # Eastward (u>0) flow climbs positive gradients; westward climbs negative
        u_pos = np.maximum(wind_u, 0.0)
        u_neg = np.minimum(wind_u, 0.0)  # negative values
        w_x = u_pos * self.grad_x_pos + (-u_neg) * self.grad_x_neg

        # Northward (v>0) flow climbs positive gradients; southward climbs negative
        v_pos = np.maximum(wind_v, 0.0)
        v_neg = np.minimum(wind_v, 0.0)
        w_y = v_pos * self.grad_y_pos + (-v_neg) * self.grad_y_neg

        return w_x + w_y

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
        # Direction-independent: average both faces so blocking doesn't depend
        # on wind direction (avoids solver oscillation from direction switching).
        r_u = 0.5 * (self.r_east_pos + np.roll(self.r_east_neg, 1, axis=1))
        r_v = 0.5 * (self.r_north_pos + np.roll(self.r_north_neg, 1, axis=0))
        u_out = wind_u * r_u
        v_out = wind_v * r_v
        return u_out, v_out

    def compute_orographic_precipitation(
        self,
        w_orographic: np.ndarray,
        humidity_q: np.ndarray,
        temperature_K: np.ndarray,
        rh: np.ndarray,
    ) -> np.ndarray:
        """Direct orographic precipitation rate (kg/m²/s).

        When air is forced upward over terrain, it cools adiabatically and
        moisture condenses.  The rate scales as  P = rh_gate · η · max(w, 0) · q · ρ,
        where η is the precipitation efficiency and rh_gate is the combined
        Gompertz + Sundqvist RH factor.

        Returns precipitation rate in kg/m²/s (same units as cloud precip).
        """
        eff = self.config.orographic_precip_efficiency
        rho = 101325.0 / (287.05 * temperature_K)
        rh_gate = compute_precipitation_rh_gate(rh)
        return rh_gate * eff * np.maximum(w_orographic, 0.0) * humidity_q * rho

    def compute_effects(self, wind_u: np.ndarray, wind_v: np.ndarray) -> dict[str, np.ndarray]:
        """Main entry point: compute all orographic effects.

        Returns dict with:
        - w_orographic: terrain-induced vertical velocity (m/s)
        - wind_u_blocked, wind_v_blocked: flow-blocked wind components
        """
        # Orographic uplift uses UNBLOCKED wind: the approaching air is forced
        # upward over terrain, causing condensation and precipitation.
        # Wind blocking is a separate effect: surface flow deflected around terrain.
        w_oro = self.compute_orographic_vertical_velocity(wind_u, wind_v)
        u_blocked, v_blocked = self.apply_flow_blocking(wind_u, wind_v)

        return {
            "w_orographic": w_oro,
            "wind_u_blocked": u_blocked,
            "wind_v_blocked": v_blocked,
        }
