# type: ignore[attr-defined]
"""Full model export — writes main.bin + main.manifest.json."""

import argparse
import os
from pathlib import Path

import numpy as np

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.core.solver import solve_periodic_climate
from climate_sim.runtime.cli import add_common_model_arguments
from climate_sim.runtime.config import ModelConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.atmosphere.advection import AdvectionConfig
from climate_sim.physics.atmosphere.wind import WindConfig
from climate_sim.physics.ocean_currents import OceanAdvectionConfig
from climate_sim.physics.vertical_motion import VerticalMotionConfig
from climate_sim.physics.orographic_effects import OrographicConfig

from .shared import write_binary_export, write_landmask_1deg


def add_full_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Load from cached main.npz instead of running simulation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: frontend/public/)",
    )
    add_common_model_arguments(
        parser,
        default_atmosphere=RadiationConfig().include_atmosphere,
    )


def run_full_export(args: argparse.Namespace) -> None:
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir.parent / "frontend" / "public"

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache:
        data_dir = os.getenv("DATA_DIR")
        if not data_dir:
            raise ValueError("DATA_DIR environment variable must be set to use --cache")
        cache_path = Path(data_dir) / "main.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}\n"
                "Run the simulation first with: uv run python -m backend.main"
            )
        print(f"Loading from cache: {cache_path}")
        with np.load(cache_path) as cached:
            layers = {k: cached[k] for k in cached}
        sample_field = layers.get("temperature_2m", layers.get("surface"))
        simulation_resolution = 180.0 / sample_field.shape[1]
        lon2d, lat2d = create_lat_lon_grid(simulation_resolution)
    else:
        print(f"Running climate simulation at {args.resolution}° resolution...")
        radiation_config = RadiationConfig(include_atmosphere=args.atmosphere)
        diffusion_config = DiffusionConfig(enabled=args.diffusion)
        snow_config = SnowAlbedoConfig(
            enabled=args.snow,
            latent_heat_enabled=args.latent_heat,
        )
        sensible_heat_config = SensibleHeatExchangeConfig(
            enabled=args.bulk_exchange,
            include_lapse_rate_elevation=args.lapse_rate_elevation,
        )
        latent_heat_config = LatentHeatExchangeConfig(
            enabled=args.latent_heat_exchange,
        )
        advection_config = AdvectionConfig(enabled=args.advection)
        wind_config = WindConfig()
        ocean_advection_config = OceanAdvectionConfig(enabled=args.ocean_advection)
        vertical_motion_config = VerticalMotionConfig(enabled=args.vertical_motion)
        orographic_config = OrographicConfig(enabled=args.orographic)

        model_config = ModelConfig(
            radiation=radiation_config,
            diffusion=diffusion_config,
            wind=wind_config,
            advection=advection_config,
            snow=snow_config,
            sensible_heat=sensible_heat_config,
            latent_heat=latent_heat_config,
            ocean_advection=ocean_advection_config,
            vertical_motion=vertical_motion_config,
            orographic=orographic_config,
            solar_constant=args.solar_constant,
            use_elliptical_orbit=args.elliptical_orbit,
        )

        lon2d, lat2d, layers = solve_periodic_climate(
            resolution_deg=args.resolution,
            model_config=model_config,
            return_layer_map=True,
        )
        assert isinstance(layers, dict)

    print("Exporting fields:")
    write_binary_export(output_dir, layers, lon2d, lat2d, interpolate=args.interpolate)
    write_landmask_1deg(output_dir)
    print("Done!")
