"""Dataclasses for assembling model configurations used by the solver entry points."""

from __future__ import annotations

from dataclasses import dataclass, field

from climate_sim.physics.atmosphere.advection import AdvectionConfig
from climate_sim.physics.atmosphere.wind import WindConfig
from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.ocean_currents import OceanAdvectionConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig
from climate_sim.physics.vertical_motion import VerticalMotionConfig
from climate_sim.physics.orographic_effects import OrographicConfig
from climate_sim.physics.empirical_corrections import EmpiricalCorrectionsConfig


@dataclass(frozen=True)
class ModelConfig:
    """Aggregate container for all model knobs passed into the solver."""

    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    advection: AdvectionConfig = field(default_factory=AdvectionConfig)
    snow: SnowAlbedoConfig = field(default_factory=SnowAlbedoConfig)
    sensible_heat: SensibleHeatExchangeConfig = field(default_factory=SensibleHeatExchangeConfig)
    latent_heat: LatentHeatExchangeConfig = field(default_factory=LatentHeatExchangeConfig)
    ocean_advection: OceanAdvectionConfig = field(default_factory=OceanAdvectionConfig)
    vertical_motion: VerticalMotionConfig = field(default_factory=VerticalMotionConfig)
    orographic: OrographicConfig = field(default_factory=OrographicConfig)
    empirical: EmpiricalCorrectionsConfig = field(default_factory=EmpiricalCorrectionsConfig)
    solar_constant: float | None = None
    use_elliptical_orbit: bool = True
    ocean_heat_capacity: float | None = None
    land_heat_capacity: float | None = None
