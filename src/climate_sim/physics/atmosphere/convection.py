"""Convective adjustment based on lapse rate.

Implements vertical heat transfer when the atmospheric lapse rate exceeds
the standard dry adiabatic lapse rate, representing convective mixing between
the boundary layer and free atmosphere.
"""

from dataclasses import dataclass

import numpy as np

from climate_sim.data.constants import (
    ATMOSPHERE_LAYER_HEIGHT_M,
    ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
    BOUNDARY_LAYER_HEIGHT_M,
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    STANDARD_LAPSE_RATE_K_PER_M,
)


@dataclass(frozen=True)
class ConvectionConfig:
    """Configuration for convective lapse rate adjustment.

    When the observed lapse rate between atmosphere and boundary layer exceeds
    the standard adiabatic lapse rate, convection redistributes heat to restore
    stability. Only active when boundary layer is enabled.

    Convection is applied as a physical constraint after solver steps, not as
    a tendency in the equations.
    """

    enabled: bool = True
    lapse_rate_K_per_m: float = STANDARD_LAPSE_RATE_K_PER_M
    atmosphere_height_m: float = ATMOSPHERE_LAYER_HEIGHT_M
    boundary_layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M


class ConvectionModel:
    """Convective heat redistribution between boundary layer and atmosphere.

    Applies instantaneous adjustment when the atmospheric temperature gradient
    exceeds the stable lapse rate, representing convective mixing at monthly
    timescales. This is applied as a physical constraint, not as a tendency.
    """

    def __init__(
        self,
        *,
        atmosphere_heat_capacity_J_m2_K: float,
        boundary_layer_heat_capacity_J_m2_K: float | None,
        config: ConvectionConfig | None = None,
    ):
        """Initialize convection model.

        Args:
            atmosphere_heat_capacity_J_m2_K: Heat capacity of atmosphere layer
            boundary_layer_heat_capacity_J_m2_K: Heat capacity of boundary layer
                (None if boundary layer disabled)
            config: Configuration parameters
        """
        self._config = config or ConvectionConfig()
        self._C_atm = atmosphere_heat_capacity_J_m2_K
        self._C_boundary = boundary_layer_heat_capacity_J_m2_K

        # Calculate the vertical distance between layer centers
        self._delta_z = (
            0.5 * self._config.atmosphere_height_m
            + 0.5 * self._config.boundary_layer_height_m
        )

        # Precompute heat capacity ratio for efficiency
        if self._C_boundary is not None:
            self._capacity_ratio = self._C_atm / self._C_boundary
        else:
            self._capacity_ratio = None

    @property
    def enabled(self) -> bool:
        """Whether convection is enabled and applicable (requires boundary layer)."""
        return self._config.enabled and self._C_boundary is not None

    def apply_convective_adjustment(
        self,
        atmosphere_temp_K: np.ndarray,
        boundary_layer_temp_K: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply convective adjustment as a physical constraint.

        Adjusts temperatures to enforce the stable lapse rate constraint when exceeded.
        This should be called after the solver computes a solution from other physics.

        Args:
            atmosphere_temp_K: Temperature of free atmosphere [K]
            boundary_layer_temp_K: Temperature of boundary layer [K] or None

        Returns:
            Tuple of (adjusted_atmosphere_temp, adjusted_boundary_temp) [K]
            Returns unchanged temperatures if disabled or boundary layer absent
        """
        if not self.enabled or boundary_layer_temp_K is None:
            return atmosphere_temp_K.copy(), boundary_layer_temp_K.copy()

        # Current temperature difference (boundary - atmosphere)
        current_diff = boundary_layer_temp_K - atmosphere_temp_K

        # Target stable temperature difference based on lapse rate
        target_diff = self._config.lapse_rate_K_per_m * self._delta_z

        # Only adjust where unstable (current_diff > target_diff)
        excess_diff = current_diff - target_diff
        unstable_mask = excess_diff > 0.0

        # Where unstable, redistribute the excess to restore stable lapse rate
        # Energy conservation: C_atm * dT_atm + C_boundary * dT_boundary = 0
        # Target: (T_boundary + dT_boundary) - (T_atm + dT_atm) = target_diff
        denominator = 1.0 + self._capacity_ratio

        dT_atm = np.where(unstable_mask, excess_diff / denominator, 0.0)
        dT_boundary = np.where(unstable_mask, -self._capacity_ratio * dT_atm, 0.0)

        adjusted_atm = atmosphere_temp_K + dT_atm
        adjusted_boundary = boundary_layer_temp_K + dT_boundary

        return adjusted_atm, adjusted_boundary
