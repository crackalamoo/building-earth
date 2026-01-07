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
    """

    enabled: bool = True
    lapse_rate_K_per_m: float = STANDARD_LAPSE_RATE_K_PER_M
    atmosphere_height_m: float = ATMOSPHERE_LAYER_HEIGHT_M
    boundary_layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M


class ConvectionModel:
    """Convective heat redistribution between boundary layer and atmosphere.

    Applies instantaneous adjustment when the atmospheric temperature gradient
    exceeds the stable lapse rate, representing convective mixing at monthly
    timescales.
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

    def compute_tendencies(
        self,
        atmosphere_temp_K: np.ndarray,
        boundary_layer_temp_K: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute convective heat tendencies for atmosphere and boundary layer.

        When the observed lapse rate exceeds the standard rate, redistributes heat
        to restore the stable temperature difference while conserving total energy.

        Args:
            atmosphere_temp_K: Temperature of free atmosphere [K]
            boundary_layer_temp_K: Temperature of boundary layer [K] or None

        Returns:
            Tuple of (atmosphere_tendency, boundary_tendency) [K/month]
            Returns zeros if disabled or boundary layer absent
        """
        if not self.enabled or boundary_layer_temp_K is None:
            # Return zero tendencies
            zeros = np.zeros_like(atmosphere_temp_K)
            return zeros, zeros

        # Current temperature difference (boundary - atmosphere)
        # Positive because boundary layer is warmer (at lower altitude)
        current_diff = boundary_layer_temp_K - atmosphere_temp_K

        # Target stable temperature difference based on lapse rate
        # This is how much cooler the atmosphere should be than the boundary
        target_diff = self._config.lapse_rate_K_per_m * self._delta_z

        # Only apply convection when atmosphere is too cold (unstable lapse rate)
        # i.e., when the temperature decrease exceeds the stable lapse rate
        unstable_mask = current_diff > target_diff

        # Calculate required temperature adjustments to reach target difference
        # Energy conservation: C_atm * dT_atm + C_boundary * dT_boundary = 0
        # Target constraint: (T_boundary + dT_boundary) - (T_atm + dT_atm) = target_diff
        #
        # Solving:
        # dT_atm = (current_diff - target_diff) / (1 + C_atm / C_boundary)
        # dT_boundary = -(C_atm / C_boundary) * dT_atm

        excess_diff = current_diff - target_diff
        denominator = 1.0 + self._capacity_ratio

        # Atmosphere warms (positive tendency) where unstable
        dT_atm = np.where(unstable_mask, excess_diff / denominator, 0.0)

        # Boundary layer cools (negative tendency) where unstable
        dT_boundary = np.where(unstable_mask, -self._capacity_ratio * dT_atm, 0.0)

        # At monthly timestep, this adjustment happens "instantaneously"
        # Return as tendency per month (adjustment happens within the month)
        return dT_atm, dT_boundary

    def compute_jacobian(
        self,
        atmosphere_temp_K: np.ndarray,
        boundary_layer_temp_K: np.ndarray | None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Compute Jacobian of convective tendencies.

        Returns linearization of tendency with respect to temperatures for
        Newton solver convergence.

        Args:
            atmosphere_temp_K: Temperature of free atmosphere [K]
            boundary_layer_temp_K: Temperature of boundary layer [K] or None

        Returns:
            ((d(atm_tendency)/dT_atm, d(atm_tendency)/dT_boundary),
             (d(boundary_tendency)/dT_atm, d(boundary_tendency)/dT_boundary))
            Returns zeros if disabled or boundary layer absent
        """
        if not self.enabled or boundary_layer_temp_K is None:
            zeros = np.zeros_like(atmosphere_temp_K)
            return (zeros, zeros), (zeros, zeros)

        # Determine where convection is active
        current_diff = boundary_layer_temp_K - atmosphere_temp_K
        target_diff = self._config.lapse_rate_K_per_m * self._delta_z
        unstable_mask = current_diff > target_diff

        # Jacobian coefficients (constant where convection active, zero elsewhere)
        denominator = 1.0 + self._capacity_ratio
        coeff = 1.0 / denominator

        # d(atm_tendency)/dT_atm = -1/denominator (where unstable)
        # d(atm_tendency)/dT_boundary = +1/denominator (where unstable)
        d_atm_d_Tatm = np.where(unstable_mask, -coeff, 0.0)
        d_atm_d_Tboundary = np.where(unstable_mask, coeff, 0.0)

        # d(boundary_tendency)/dT_atm = (C_atm/C_boundary)/denominator (where unstable)
        # d(boundary_tendency)/dT_boundary = -(C_atm/C_boundary)/denominator (where unstable)
        coeff_boundary = self._capacity_ratio / denominator
        d_boundary_d_Tatm = np.where(unstable_mask, coeff_boundary, 0.0)
        d_boundary_d_Tboundary = np.where(unstable_mask, -coeff_boundary, 0.0)

        return (d_atm_d_Tatm, d_atm_d_Tboundary), (
            d_boundary_d_Tboundary,
            d_boundary_d_Tatm,
        )
