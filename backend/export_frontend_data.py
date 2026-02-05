"""Export climate simulation data as binary for the frontend visualization."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.core.interpolation import interpolate_layer_map
from climate_sim.core.solver import solve_periodic_climate
from climate_sim.data.landmask import compute_land_mask
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
        description="Export climate simulation data as binary for frontend visualization."
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
        help="Output directory (default: frontend/public/)",
    )
    add_common_model_arguments(
        parser,
        default_atmosphere=RadiationConfig().include_atmosphere,
    )
    return parser.parse_args()


def _write_binary_export(
    output_dir: Path,
    layers: dict[str, np.ndarray],
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    interpolate: bool,
) -> None:
    """Write main.bin + main.manifest.json to output_dir."""

    # Keep native-resolution copies before interpolation
    native_layers = dict(layers)

    # Interpolate temperature_2m if requested
    if interpolate:
        output_resolution = 0.25
        print(f"Interpolating temperature fields to {output_resolution}° resolution...")

        interp_layers: dict[str, np.ndarray] = {}
        if "surface" in layers:
            interp_layers["surface"] = layers["surface"]
        if "temperature_2m" in layers:
            interp_layers["temperature_2m"] = layers["temperature_2m"]

        _, _, interpolated = interpolate_layer_map(
            interp_layers,
            lon2d,
            lat2d,
            output_resolution_deg=output_resolution,
            apply_lapse_rate_to_2m=True,
        )
        print("Interpolation complete.")
    else:
        interpolated = {}

    # Compute land_mask at native resolution
    land_mask = compute_land_mask(lon2d, lat2d).astype(np.uint8)

    # Assemble fields for export
    # Each entry: (array, dtype_str) — dtype_str is 'float16' or 'uint8'
    fields: list[tuple[str, np.ndarray, str]] = []

    # temperature_2m: interpolated (0.25deg) or native
    if "temperature_2m" in interpolated:
        fields.append(("temperature_2m", interpolated["temperature_2m"], "float16"))
        print(f"  temperature_2m: {interpolated['temperature_2m'].shape} (interpolated)")
    elif "temperature_2m" in native_layers:
        fields.append(("temperature_2m", native_layers["temperature_2m"], "float16"))
        print(f"  temperature_2m: {native_layers['temperature_2m'].shape} (native)")
    elif "surface" in native_layers:
        fields.append(("temperature_2m", native_layers["surface"], "float16"))
        print(f"  temperature_2m: {native_layers['surface'].shape} (from surface)")

    # surface: always native resolution (for Blue Marble snow/ice detection)
    if "surface" in native_layers:
        fields.append(("surface", native_layers["surface"], "float16"))
        print(f"  surface: {native_layers['surface'].shape}")

    # land_mask: static, native resolution
    fields.append(("land_mask", land_mask, "uint8"))
    print(f"  land_mask: {land_mask.shape}")

    # vegetation_fraction: native resolution
    if "vegetation_fraction" in native_layers:
        fields.append(("vegetation_fraction", native_layers["vegetation_fraction"], "float16"))
        print(f"  vegetation_fraction: {native_layers['vegetation_fraction'].shape}")

    # Wind fields: native resolution
    for wind_key in ("wind_u_10m", "wind_v_10m", "wind_speed_10m"):
        if wind_key in native_layers:
            fields.append((wind_key, native_layers[wind_key], "float16"))
            print(f"  {wind_key}: {native_layers[wind_key].shape}")

    # Build binary blob and manifest
    manifest: dict = {"fields": []}
    blobs: list[bytes] = []
    offset = 0

    for name, arr, dtype_str in fields:
        # Convert to target dtype
        if dtype_str == "float16":
            encoded = arr.astype(np.float16)
        elif dtype_str == "uint8":
            encoded = arr.astype(np.uint8)
        else:
            encoded = arr.astype(np.float32)

        raw = encoded.tobytes()
        manifest["fields"].append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": dtype_str,
            "offset": offset,
            "bytes": len(raw),
        })
        blobs.append(raw)
        offset += len(raw)

    # Write files
    bin_path = output_dir / "main.bin"
    manifest_path = output_dir / "main.manifest.json"

    with open(bin_path, "wb") as f:
        for blob in blobs:
            f.write(blob)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    bin_size = bin_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {bin_path} ({bin_size:.2f} MB)")
    print(f"Wrote {manifest_path}")


def main() -> None:
    args = _parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        script_dir = Path(__file__).parent
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

    print("Exporting fields:")
    _write_binary_export(output_dir, layers, lon2d, lat2d, interpolate=args.interpolate)
    print("Done!")


if __name__ == "__main__":
    main()
