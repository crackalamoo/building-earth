# type: ignore[attr-defined]
"""Export onboarding stage binaries (stages 1-3)."""

import argparse
import os
from pathlib import Path

import numpy as np

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.core.solver import solve_periodic_climate
from climate_sim.runtime.config import ModelConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.atmosphere.advection import AdvectionConfig
from climate_sim.physics.atmosphere.wind import WindConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.ocean_currents import OceanAdvectionConfig
from climate_sim.physics.vertical_motion import VerticalMotionConfig
from climate_sim.physics.orographic_effects import OrographicConfig
from climate_sim.physics.empirical_corrections import EmpiricalCorrectionsConfig

from onboarding_stages import STAGE_NAMES
from .shared import write_binary_export


# --- Stage export definitions (physics configs, fields, quantization) ---

_STAGE_FIELDS: dict[int, list[str]] = {
    1: ["temperature_2m", "land_mask", "elevation", "surface"],
    2: ["temperature_2m", "land_mask", "elevation", "surface"],
    3: [
        "temperature_2m", "land_mask", "elevation",
        "wind_u_10m", "wind_v_10m", "wind_speed_10m", "surface",
        "precipitation", "humidity", "soil_moisture",
        "cloud_fraction", "cloud_high", "cloud_low", "cloud_convective",
        "vegetation_fraction", "snow_temperature",
        "land_mask_native", "land_mask_1deg",
    ],
}

_STAGE_QUANTIZATION: dict[int, tuple[float, float]] = {
    1: (-100, 100),  # wider range for extreme temps
    2: (-60, 60),
    3: (-60, 60),
}


def _get_stage_model_config(stage: int) -> ModelConfig:
    """Return a ModelConfig with physics appropriate for the given stage."""
    if stage == 1:
        return ModelConfig(
            radiation=RadiationConfig(include_atmosphere=False),
            diffusion=DiffusionConfig(enabled=False),
            wind=WindConfig(enabled=False),
            advection=AdvectionConfig(enabled=False),
            snow=SnowAlbedoConfig(enabled=False),
            sensible_heat=SensibleHeatExchangeConfig(enabled=False),
            latent_heat=LatentHeatExchangeConfig(enabled=False),
            ocean_advection=OceanAdvectionConfig(enabled=False),
            vertical_motion=VerticalMotionConfig(enabled=False),
            orographic=OrographicConfig(enabled=False),
        )
    elif stage == 2:
        return ModelConfig(
            radiation=RadiationConfig(),  # include_atmosphere=True (default)
            diffusion=DiffusionConfig(enabled=False),
            wind=WindConfig(enabled=False),
            advection=AdvectionConfig(enabled=False),
            snow=SnowAlbedoConfig(enabled=False),
            sensible_heat=SensibleHeatExchangeConfig(),
            latent_heat=LatentHeatExchangeConfig(enabled=False),
            ocean_advection=OceanAdvectionConfig(enabled=False),
            vertical_motion=VerticalMotionConfig(enabled=False),
            orographic=OrographicConfig(enabled=False),
        )
    elif stage == 3:
        # Full atmospheric dynamics but no ocean heat transport (neither
        # wind-driven gyres nor thermohaline/AMOC).
        return ModelConfig(
            radiation=RadiationConfig(),
            diffusion=DiffusionConfig(),
            wind=WindConfig(),
            advection=AdvectionConfig(),
            snow=SnowAlbedoConfig(),
            sensible_heat=SensibleHeatExchangeConfig(),
            latent_heat=LatentHeatExchangeConfig(),
            ocean_advection=OceanAdvectionConfig(enabled=False),
            vertical_motion=VerticalMotionConfig(),
            orographic=OrographicConfig(),
            empirical=EmpiricalCorrectionsConfig(amoc_enabled=False),
        )
    else:
        raise ValueError(f"Stage {stage} uses the default full model config (stage 4 = main.bin)")


def add_onboarding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Load from cached stageN.npz instead of running simulations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: frontend/public/)",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=float,
        default=5.0,
        help="Grid resolution in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        default=False,
        help="Interpolate to higher resolution",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="1,2,3",
        help="Comma-separated list of stages to export (default: 1,2,3)",
    )


def run_onboarding_export(args: argparse.Namespace) -> None:
    if args.output:
        output_dir = Path(args.output)
    else:
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir.parent / "frontend" / "public"

    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = os.getenv("DATA_DIR", "data")
    stages_to_export = [int(s.strip()) for s in args.stages.split(",")]

    for stage_num in stages_to_export:
        if stage_num not in (1, 2, 3):
            print(f"Skipping stage {stage_num} (only stages 1-3 need separate exports)")
            continue

        name = STAGE_NAMES[stage_num]
        print(f"\n{'='*60}")
        print(f"Stage {stage_num}: {name}")
        print(f"{'='*60}")

        cache_path = Path(data_dir) / f"stage{stage_num}.npz"

        if args.cache and cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            with np.load(cache_path) as cached:
                layers = {k: cached[k] for k in cached}
            sample_field = layers.get("temperature_2m", layers.get("surface"))
            simulation_resolution = 180.0 / sample_field.shape[1]
            lon2d, lat2d = create_lat_lon_grid(simulation_resolution)
        else:
            print(f"Running solver for stage {stage_num} at {args.resolution}° resolution...")
            model_config = _get_stage_model_config(stage_num)

            lon2d, lat2d, layers = solve_periodic_climate(
                resolution_deg=args.resolution,
                model_config=model_config,
                return_layer_map=True,
            )
            assert isinstance(layers, dict)

            # Cache the result
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, **layers)
            print(f"Cached solver output to {cache_path}")

        q_min, q_max = _STAGE_QUANTIZATION[stage_num]
        fields_filter = _STAGE_FIELDS[stage_num]

        print(f"Exporting stage {stage_num} fields (quantization: [{q_min}, {q_max}]):")
        write_binary_export(
            output_dir,
            layers,
            lon2d,
            lat2d,
            interpolate=args.interpolate,
            file_prefix=f"stage{stage_num}",
            fields_filter=fields_filter,
            quantization_min=float(q_min),
            quantization_max=float(q_max),
        )

    print("\nOnboarding export complete!")
