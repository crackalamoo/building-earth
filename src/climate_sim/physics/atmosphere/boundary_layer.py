"""Boundary layer configuration for the climate model."""

from __future__ import annotations

from dataclasses import dataclass

from climate_sim.data.constants import (
    BOUNDARY_LAYER_EMISSIVITY,
    BOUNDARY_LAYER_HEIGHT_M,
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
)


@dataclass(frozen=True)
class BoundaryLayerConfig:
    """Configuration for the atmospheric boundary layer.

    When enabled, adds a third layer between surface and free atmosphere.
    The boundary layer is optically thin (low emissivity, zero albedo) and
    exchanges heat with adjacent layers through radiation and sensible heat.

    Requires `RadiationConfig.include_atmosphere=True`.
    """

    enabled: bool = True
    height_m: float = BOUNDARY_LAYER_HEIGHT_M
    heat_capacity: float = BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K
    emissivity: float = BOUNDARY_LAYER_EMISSIVITY
