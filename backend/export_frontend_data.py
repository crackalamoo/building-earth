"""Export climate simulation data to JSON for the frontend visualization."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.core.interpolation import interpolate_layer_map
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
from climate_sim.physics.atmosphere.boundary_layer import BoundaryLayerConfig
from climate_sim.physics.ocean_currents import OceanAdvectionConfig
from climate_sim.physics.vertical_motion import VerticalMotionConfig

from dotenv import load_dotenv
load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export climate simulation data to JSON for frontend visualization."
    )
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
        help="Output path for JSON file (default: frontend/public/main.json)",
    )
    add_common_model_arguments(
        parser,
        default_atmosphere=RadiationConfig().include_atmosphere,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default to frontend/public/main.json relative to this script
        script_dir = Path(__file__).parent
        output_path = script_dir.parent / "frontend" / "public" / "main.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.cache:
        # Load from cached npz file
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
        # Infer resolution from cached data shape
        # Shape is (12, nlat, nlon), resolution = 180 / nlat
        sample_field = layers.get("temperature_2m", layers.get("surface"))
        simulation_resolution = 180.0 / sample_field.shape[1]
        lon2d, lat2d = create_lat_lon_grid(simulation_resolution)
    else:
        # Run simulation
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
        boundary_layer_config = BoundaryLayerConfig(enabled=args.boundary_layer)
        ocean_advection_config = OceanAdvectionConfig(enabled=args.ocean_advection)
        vertical_motion_config = VerticalMotionConfig(enabled=args.vertical_motion)

        model_config = ModelConfig(
            radiation=radiation_config,
            diffusion=diffusion_config,
            wind=wind_config,
            advection=advection_config,
            snow=snow_config,
            sensible_heat=sensible_heat_config,
            latent_heat=latent_heat_config,
            boundary_layer=boundary_layer_config,
            ocean_advection=ocean_advection_config,
            vertical_motion=vertical_motion_config,
            solar_constant=args.solar_constant,
            use_elliptical_orbit=args.elliptical_orbit,
        )

        lon2d, lat2d, layers = solve_periodic_climate(
            resolution_deg=args.resolution,
            model_config=model_config,
            return_layer_map=True,
        )
        assert isinstance(layers, dict)
        simulation_resolution = args.resolution

    # Interpolate if requested (follows main.py pattern: 0.25° output resolution)
    if args.interpolate:
        output_resolution = 0.25
        print(f"Interpolating temperature fields to {output_resolution}° resolution...")

        # Build layer map for interpolation
        interp_layers = {}
        if "surface" in layers:
            interp_layers["surface"] = layers["surface"]
        if "temperature_2m" in layers:
            interp_layers["temperature_2m"] = layers["temperature_2m"]

        _, _, interpolated = interpolate_layer_map(
            interp_layers,
            lon2d,
            lat2d,
            output_resolution_deg=output_resolution,
            apply_lapse_rate_to_2m=True,  # Always apply lapse rate for 2m temp
        )

        # Use interpolated data
        layers = interpolated
        print(f"Interpolation complete.")

    # Build JSON output
    # The frontend expects temperature_2m as [month][lat][lon] in Celsius
    output_data: dict[str, list] = {}

    if "temperature_2m" in layers:
        output_data["temperature_2m"] = layers["temperature_2m"].tolist()
        print(f"Exported temperature_2m: shape {layers['temperature_2m'].shape}")
    elif "surface" in layers:
        # Fall back to surface temperature if 2m not available
        output_data["temperature_2m"] = layers["surface"].tolist()
        print(f"Exported surface as temperature_2m: shape {layers['surface'].shape}")
    else:
        raise ValueError("No temperature data found in simulation output")

    # Optionally include other fields
    if "surface" in layers:
        output_data["surface"] = layers["surface"].tolist()

    # Write JSON
    print(f"Writing to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done! Output size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
