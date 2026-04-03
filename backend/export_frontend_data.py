# type: ignore[attr-defined]
"""Export climate simulation data as binary for the frontend visualization."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.export.temperature_interpolation import interpolate_layer_map
from climate_sim.export.orographic_interpolation import recompute_fields_at_1deg
from climate_sim.core.solver import solve_periodic_climate
from climate_sim.data.elevation import compute_cell_elevation
from climate_sim.data.landmask import compute_land_mask
from climate_sim.runtime.cli import add_common_model_arguments
from climate_sim.runtime.config import ModelConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.surface_albedo import SurfaceAlbedoConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.atmosphere.advection import AdvectionConfig
from climate_sim.physics.atmosphere.wind import WindConfig
from climate_sim.physics.ocean_currents import OceanAdvectionConfig
from climate_sim.physics.vertical_motion import VerticalMotionConfig
from climate_sim.physics.orographic_effects import OrographicConfig

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


def _fill_island_vegetation(
    veg: np.ndarray,
    native_layers: dict[str, np.ndarray],
    native_land_mask: np.ndarray,
    fine_land_mask: np.ndarray,
) -> np.ndarray:
    """Recompute vegetation for coarse ocean cells that contain fine-grid land.

    Islands like Hawaii sit in ocean cells where veg=0. We recompute veg from
    precipitation and temperature at these cells, assuming soil_moisture=1.0.
    """
    nlat, nlon = native_land_mask.shape
    fine_nlat, fine_nlon = fine_land_mask.shape

    # Find coarse ocean cells that contain at least one fine-grid land pixel
    lat_ratio = fine_nlat // nlat
    lon_ratio = fine_nlon // nlon

    precip = native_layers["precipitation"]  # (12, nlat, nlon) in kg/m²/s
    surface = native_layers["surface"]  # (12, nlat, nlon)

    # Convert precipitation rate (kg/m²/s) to annual total (mm/year)
    # Each month's rate × seconds in that month, then sum
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_seconds = np.array([d * 86400.0 for d in days_per_month])  # (12,)

    # Veg params (matching SurfaceAlbedoConfig defaults)
    p_min, p_max = 50.0, 1000.0
    growing_thresh = 5.0
    full_months = 5.0

    for i in range(nlat):
        for j in range(nlon):
            if native_land_mask[i, j]:
                continue  # already has valid land veg

            # Check if any fine-grid pixels in this coarse cell are land
            fi0 = i * lat_ratio
            fi1 = min((i + 1) * lat_ratio, fine_nlat)
            fj0 = j * lon_ratio
            fj1 = min((j + 1) * lon_ratio, fine_nlon)
            if not np.any(fine_land_mask[fi0:fi1, fj0:fj1]):
                continue  # pure ocean, skip

            # Recompute veg from this cell's precipitation and temperature
            # precip is kg/m²/s per month; multiply by seconds → kg/m² = mm
            annual_precip = float(np.sum(precip[:, i, j] * month_seconds))
            u = np.clip((annual_precip - p_min) / (p_max - p_min), 0.0, 1.0)
            veg_frac = u**0.6

            # Growing season cap
            warm_months = float(np.sum(surface[:, i, j] > growing_thresh))
            gs_u = np.clip(warm_months / full_months, 0.0, 1.0)
            gs_cap = gs_u * gs_u * (3.0 - 2.0 * gs_u)
            veg_frac = min(veg_frac, gs_cap, 0.95)

            # Write into all 12 months
            for m in range(12):
                veg[m, i, j] = veg_frac

    return veg


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
            compute_snow_temperature=True,
        )
        print("Interpolation complete.")

        # Recompute precipitation, humidity, soil_moisture at 1° with orographic detail
        print("Recomputing precipitation at 1° with orographic detail...")
        fine_fields = recompute_fields_at_1deg(native_layers, lon2d, lat2d)
    else:
        interpolated = {}
        fine_fields = {}

    # Compute land masks
    native_land_mask = compute_land_mask(lon2d, lat2d).astype(np.uint8)
    if interpolate:
        fine_lon2d, fine_lat2d = create_lat_lon_grid(output_resolution)
        land_mask = compute_land_mask(fine_lon2d, fine_lat2d).astype(np.uint8)
    else:
        land_mask = native_land_mask

    # Assemble fields for export
    # Each entry: (array, dtype_str) — dtype_str is 'float16' or 'uint8'
    fields: list[tuple[str, np.ndarray, str]] = []

    # temperature_2m: interpolated (0.25deg) or native — quantized to uint8
    # Range [-60, +60] °C → [0, 255], decode: val * (120/255) - 60
    t2m_src = None
    t2m_label = ""
    if "temperature_2m" in interpolated:
        t2m_src = interpolated["temperature_2m"]
        t2m_label = "interpolated"
    elif "temperature_2m" in native_layers:
        t2m_src = native_layers["temperature_2m"]
        t2m_label = "native"
    elif "surface" in native_layers:
        t2m_src = native_layers["surface"]
        t2m_label = "from surface"
    if t2m_src is not None:
        t2m_u8 = np.clip((t2m_src + 60.0) * (255.0 / 120.0), 0, 255).astype(np.uint8)
        fields.append(("temperature_2m", t2m_u8, "uint8"))
        print(f"  temperature_2m: {t2m_u8.shape} ({t2m_label}, uint8)")

    # surface: always native resolution (for Blue Marble snow/ice detection)
    if "surface" in native_layers:
        fields.append(("surface", native_layers["surface"], "float16"))
        print(f"  surface: {native_layers['surface'].shape}")

    # land_mask: high-res (0.25deg when interpolated) for rendering
    fields.append(("land_mask", land_mask, "uint8"))
    print(f"  land_mask: {land_mask.shape}")

    # land_mask_native: native 5deg for type-aware interpolation of low-res fields
    if interpolate and native_land_mask is not land_mask:
        fields.append(("land_mask_native", native_land_mask, "uint8"))
        print(f"  land_mask_native: {native_land_mask.shape}")

    # land_mask_1deg: 1° land mask matching soil_moisture/precipitation resolution
    if interpolate and fine_fields:
        lon1deg, lat1deg = create_lat_lon_grid(1.0)
        land_mask_1deg = compute_land_mask(lon1deg, lat1deg).astype(np.uint8)
        fields.append(("land_mask_1deg", land_mask_1deg, "uint8"))
        print(f"  land_mask_1deg: {land_mask_1deg.shape}")

    # vegetation_fraction: native resolution, with island fill
    if "vegetation_fraction" in native_layers:
        veg = native_layers["vegetation_fraction"].copy()
        if interpolate and "precipitation" in native_layers and "surface" in native_layers:
            veg = _fill_island_vegetation(
                veg,
                native_layers,
                native_land_mask,
                land_mask,
            )
        fields.append(("vegetation_fraction", veg, "float16"))
        print(f"  vegetation_fraction: {veg.shape}")

    # precipitation: 1° orographic detail when interpolated, else native
    if "precipitation" in fine_fields:
        fields.append(("precipitation", fine_fields["precipitation"], "float16"))
        print(f"  precipitation: {fine_fields['precipitation'].shape} (1° orographic)")
    elif "precipitation" in native_layers:
        fields.append(("precipitation", native_layers["precipitation"], "float16"))
        print(f"  precipitation: {native_layers['precipitation'].shape} (native)")

    # humidity: 1° interpolated when available, else native
    if "humidity" in fine_fields:
        fields.append(("humidity", fine_fields["humidity"], "float16"))
        print(f"  humidity: {fine_fields['humidity'].shape} (1° interpolated)")
    elif "humidity" in native_layers:
        fields.append(("humidity", native_layers["humidity"], "float16"))
        print(f"  humidity: {native_layers['humidity'].shape} (native)")

    # soil_moisture: 1° orographic-adjusted when available, else native
    if "soil_moisture" in fine_fields:
        fields.append(("soil_moisture", fine_fields["soil_moisture"], "float16"))
        print(f"  soil_moisture: {fine_fields['soil_moisture'].shape} (1° orographic)")
    elif "soil_moisture" in native_layers:
        fields.append(("soil_moisture", native_layers["soil_moisture"], "float16"))
        print(f"  soil_moisture: {native_layers['soil_moisture'].shape} (native)")

    # cloud_fraction: native resolution (combined from cloud components)
    conv = native_layers.get("convective_cloud_frac")
    strat = native_layers.get("stratiform_cloud_frac")
    marine = native_layers.get("marine_sc_cloud_frac")
    high = native_layers.get("high_cloud_frac")
    if all(x is not None for x in [conv, strat, marine, high]):
        low = 1 - (1 - conv) * (1 - strat) * (1 - marine)
        total_cloud = np.clip(low + high * (1 - low), 0, 1)
        fields.append(("cloud_fraction", total_cloud, "float16"))
        print(f"  cloud_fraction: {total_cloud.shape}")

        fields.append(("cloud_high", high, "float16"))
        print(f"  cloud_high: {high.shape}")
        low_nonconv = np.clip(1 - (1 - strat) * (1 - marine), 0, 1)
        fields.append(("cloud_low", low_nonconv, "float16"))
        print(f"  cloud_low: {low_nonconv.shape}")
        fields.append(("cloud_convective", conv, "float16"))
        print(f"  cloud_convective: {conv.shape}")

    # Elevation: high-res (0.25deg) for 3D terrain displacement
    if interpolate:
        elevation = compute_cell_elevation(fine_lon2d, fine_lat2d, cache=False)
    else:
        elevation = compute_cell_elevation(lon2d, lat2d, cache=False)
    # Keep negative values (bathymetry) for ocean depth shading
    fields.append(("elevation", elevation, "float16"))
    print(f"  elevation: {elevation.shape}")

    # Snow temperature: interpolated surface with lapse correction for snow/ice
    if "snow_temperature" in interpolated:
        snow_temp = interpolated["snow_temperature"]
        # Quantize to uint8: range [-60, +60] °C → [0, 255]
        snow_temp_u8 = np.clip((snow_temp + 60.0) * (255.0 / 120.0), 0, 255).astype(np.uint8)
        fields.append(("snow_temperature", snow_temp_u8, "uint8"))
        print(f"  snow_temperature: {snow_temp_u8.shape}")

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
        manifest["fields"].append(
            {
                "name": name,
                "shape": list(arr.shape),
                "dtype": dtype_str,
                "offset": offset,
                "bytes": len(raw),
            }
        )
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

    # Write standalone 1° land mask for fast initial load (primordial globe)
    lon1d, lat1d = create_lat_lon_grid(1.0)
    lm1deg = compute_land_mask(lon1d, lat1d).astype(np.uint8)
    lm_path = output_dir / "landmask1deg.bin"
    with open(lm_path, "wb") as f:
        f.write(lm1deg.tobytes())
    print(f"Wrote {lm_path} ({lm_path.stat().st_size / 1024:.1f} KB, {lm1deg.shape})")


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
        snow_config = SurfaceAlbedoConfig(
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
            surface_albedo=snow_config,
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
    _write_binary_export(output_dir, layers, lon2d, lat2d, interpolate=args.interpolate)
    print("Done!")


if __name__ == "__main__":
    main()
