"""Operator and grid construction for the climate model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.core.grid import create_lat_lon_grid, expand_latitude_field
from climate_sim.core.math_core import LinearSolveCache
from climate_sim.data.calendar import DAYS_PER_MONTH, SECONDS_PER_DAY
from climate_sim.data.elevation import compute_cell_elevation, compute_cell_roughness_length
from climate_sim.data.landmask import (
    compute_albedo_field,
    compute_heat_capacity_field,
    compute_land_mask,
)
from climate_sim.physics.atmosphere.advection import AdvectionOperator
from climate_sim.physics.atmosphere.boundary_layer import BoundaryLayerConfig
from climate_sim.physics.atmosphere.wind import WindModel
from climate_sim.physics.diffusion import (
    LayeredDiffusionOperator,
    create_diffusion_operator,
)
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.snow_albedo import AlbedoModel
from climate_sim.physics.solar import compute_monthly_insolation_field
from climate_sim.runtime.config import ModelConfig


@dataclass(frozen=True)
class SurfaceHeatCapacityContext:
    """Container for surface-layer heat-capacity metadata."""

    lon2d: np.ndarray
    lat2d: np.ndarray
    albedo_model: AlbedoModel
    wind_model: WindModel | None
    advection_operator: AdvectionOperator | None
    base_albedo: np.ndarray
    land_mask: np.ndarray
    base_C_land: float
    base_C_ocean: float
    baseline_capacity: np.ndarray


@dataclass(frozen=True)
class ModelOperators:
    """Complete set of operators and grids for the climate model."""

    lon2d: np.ndarray
    lat2d: np.ndarray
    heat_capacity_field: np.ndarray
    monthly_insolation: np.ndarray
    month_durations: np.ndarray
    diffusion_operator: LayeredDiffusionOperator
    wind_model: WindModel
    advection_operator: AdvectionOperator
    albedo_model: AlbedoModel
    base_albedo_field: np.ndarray
    land_mask: np.ndarray
    roughness_length: np.ndarray
    topographic_elevation: np.ndarray
    surface_context: SurfaceHeatCapacityContext
    solver_cache: LinearSolveCache
    # Configs
    radiation_config: RadiationConfig
    sensible_heat_cfg: SensibleHeatExchangeConfig
    latent_heat_cfg: LatentHeatExchangeConfig
    boundary_layer_cfg: BoundaryLayerConfig


def build_model_operators(
    resolution_deg: float,
    model_config: ModelConfig,
) -> ModelOperators:
    """Construct all operators, grids, and fields for the climate model.

    This function handles all the setup work: creating grids, computing
    masks and fields, building spatial operators (diffusion, advection),
    and initializing models (wind, albedo).

    Parameters
    ----------
    resolution_deg : float
        Grid resolution in degrees.
    model_config : ModelConfig
        Complete model configuration.

    Returns
    -------
    ModelOperators
        All operators and grids needed for the solver.
    """
    # Create grids
    lon2d, lat2d = create_lat_lon_grid(resolution_deg)

    # Compute insolation
    monthly_insolation_lat = compute_monthly_insolation_field(
        lat2d,
        solar_constant=model_config.solar_constant,
        use_elliptical_orbit=model_config.use_elliptical_orbit,
    )
    monthly_insolation = expand_latitude_field(monthly_insolation_lat, lon2d.shape[1])

    # Compute heat capacity field
    heat_capacity_field = compute_heat_capacity_field(
        lon2d,
        lat2d,
        ocean_heat_capacity=model_config.ocean_heat_capacity,
        land_heat_capacity=model_config.land_heat_capacity,
    )

    # Get configs
    snow_cfg = model_config.snow
    sensible_heat_cfg = model_config.sensible_heat
    latent_heat_cfg = model_config.latent_heat
    boundary_layer_cfg = model_config.boundary_layer
    radiation_config = model_config.radiation
    diffusion_config = model_config.diffusion
    wind_config = model_config.wind
    advection_config = model_config.advection

    # Update radiation config with boundary layer if enabled
    if boundary_layer_cfg.enabled:
        if not radiation_config.include_atmosphere:
            raise ValueError("Boundary layer requires include_atmosphere=True in radiation config")
        from dataclasses import replace
        from climate_sim.data.constants import ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K

        radiation_config = replace(
            radiation_config,
            atmosphere_heat_capacity=ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
            boundary_layer_heat_capacity=boundary_layer_cfg.heat_capacity,
            boundary_layer_emissivity=boundary_layer_cfg.emissivity,
        )

    # Compute land mask
    land_mask = compute_land_mask(lon2d, lat2d)

    # Build albedo model
    albedo_model = AlbedoModel(
        lat2d,
        lon2d,
        config=snow_cfg,
        land_mask=land_mask,
    )

    albedo_kwargs: dict[str, float] = {}
    if snow_cfg.enabled:
        albedo_kwargs = {"land_albedo": 0.18, "ocean_albedo": 0.06}

    base_albedo_field = compute_albedo_field(lon2d, lat2d, **albedo_kwargs)

    # Compute roughness and elevation
    roughness_length = compute_cell_roughness_length(
        lon2d,
        lat2d,
        land_mask=land_mask,
    )

    if sensible_heat_cfg.include_lapse_rate_elevation:
        topographic_elevation = compute_cell_elevation(lon2d, lat2d)
        topographic_elevation = np.maximum(topographic_elevation, 0.0)
    else:
        topographic_elevation = np.zeros_like(lon2d, dtype=float)

    # Build diffusion operator
    bl_heat_cap = None
    if boundary_layer_cfg.enabled:
        bl_heat_cap = radiation_config.boundary_layer_heat_capacity

    diffusion_operator = create_diffusion_operator(
        lon2d,
        lat2d,
        heat_capacity_field,
        land_mask=land_mask,
        atmosphere_heat_capacity=radiation_config.atmosphere_heat_capacity,
        boundary_layer_heat_capacity=bl_heat_cap,
        config=diffusion_config,
    )

    # Build wind and advection models
    wind_model = WindModel(
        lon2d,
        lat2d,
        config=wind_config,
    )

    advection_operator = AdvectionOperator(
        lon2d,
        lat2d,
        config=advection_config,
    )

    # Compute month durations
    month_durations = DAYS_PER_MONTH * SECONDS_PER_DAY

    # Compute baseline heat capacities for surface context
    land_values = heat_capacity_field[land_mask]
    if land_values.size == 0:
        base_C_land = float(np.mean(heat_capacity_field))
    else:
        base_C_land = float(np.mean(land_values))

    ocean_mask = ~land_mask
    ocean_values = heat_capacity_field[ocean_mask]
    if ocean_values.size == 0:
        base_C_ocean = base_C_land
    else:
        base_C_ocean = float(np.mean(ocean_values))

    baseline_capacity = np.where(land_mask, base_C_land, base_C_ocean).astype(float)

    # Build surface context
    surface_context = SurfaceHeatCapacityContext(
        lat2d=lat2d,
        lon2d=lon2d,
        albedo_model=albedo_model,
        wind_model=wind_model,
        advection_operator=advection_operator,
        base_albedo=base_albedo_field,
        land_mask=land_mask,
        base_C_land=base_C_land,
        base_C_ocean=base_C_ocean,
        baseline_capacity=baseline_capacity,
    )

    # Create solver cache
    solver_cache = LinearSolveCache()

    return ModelOperators(
        lon2d=lon2d,
        lat2d=lat2d,
        heat_capacity_field=heat_capacity_field,
        monthly_insolation=monthly_insolation,
        month_durations=month_durations,
        diffusion_operator=diffusion_operator,
        wind_model=wind_model,
        advection_operator=advection_operator,
        albedo_model=albedo_model,
        base_albedo_field=base_albedo_field,
        land_mask=land_mask,
        roughness_length=roughness_length,
        topographic_elevation=topographic_elevation,
        surface_context=surface_context,
        solver_cache=solver_cache,
        radiation_config=radiation_config,
        sensible_heat_cfg=sensible_heat_cfg,
        latent_heat_cfg=latent_heat_cfg,
        boundary_layer_cfg=boundary_layer_cfg,
    )
