import argparse

import numpy as np

from climate_sim.modeling.advection import GeostrophicAdvectionConfig
from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
from climate_sim.plotting import plot_layered_monthly_temperature_cycle
from climate_sim.utils.math_core import spherical_cell_area
from climate_sim.utils.solver import compute_periodic_cycle_celsius

from dotenv import load_dotenv
load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the climate model and plot the cycle.")
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Grid resolution in degrees",
    )
    parser.add_argument(
        "--solar-constant",
        type=float,
        default=None,
        help="Override the solar constant (W m^-2)",
    )
    parser.add_argument(
        "--diffusion",
        dest="diffusion",
        action="store_true",
        default=True,
        help="Enable lateral diffusion (default)",
    )
    parser.add_argument(
        "--no-diffusion",
        dest="diffusion",
        action="store_false",
        help="Disable lateral diffusion",
    )
    parser.add_argument(
        "--diffusion-geometry",
        dest="diffusion_geometry",
        action="store_true",
        default=True,
        help="Use spherical geometry scalings in the diffusion operator (default)",
    )
    parser.add_argument(
        "--no-diffusion-geometry",
        dest="diffusion_geometry",
        action="store_false",
        help="Treat diffusion with uniform planar geometry",
    )

    parser.add_argument(
        "--advection",
        dest="advection",
        action="store_true",
        default=True,
        help="Enable geostrophic atmospheric advection (default)",
    )
    parser.add_argument(
        "--no-advection",
        dest="advection",
        action="store_false",
        help="Disable geostrophic atmospheric advection",
    )

    default_atmosphere = RadiationConfig().include_atmosphere
    parser.add_argument(
        "--atmosphere",
        dest="atmosphere",
        action="store_true",
        default=default_atmosphere,
        help="Include an explicit atmospheric layer",
    )
    parser.add_argument(
        "--no-atmosphere",
        dest="atmosphere",
        action="store_false",
        help="Exclude the atmospheric layer",
    )

    parser.add_argument(
        "--snow",
        dest="snow",
        action="store_true",
        default=True,
        help="Enable diagnostic snow-albedo adjustments (default)",
    )
    parser.add_argument(
        "--no-snow",
        dest="snow",
        action="store_false",
        help="Disable snow-albedo adjustments",
    )
    parser.add_argument(
        "--fahrenheit",
        dest="fahrenheit",
        action="store_true",
        help="Display temperatures in degrees Fahrenheit instead of Celsius",
    )

    return parser.parse_args()


def _convert_temperature(
    values: np.ndarray | float, use_fahrenheit: bool
) -> np.ndarray | float:
    if not use_fahrenheit:
        return values
    converted = (np.asarray(values) * (9.0 / 5.0)) + 32.0
    if isinstance(values, np.ndarray):
        return converted
    if np.isscalar(values):
        return float(converted)
    return converted


def _temperature_unit(use_fahrenheit: bool) -> str:
    return "°F" if use_fahrenheit else "°C"


def main() -> None:
    args = _parse_args()

    radiation_config = RadiationConfig(include_atmosphere=args.atmosphere)
    diffusion_config = DiffusionConfig(
        enabled=args.diffusion, use_spherical_geometry=args.diffusion_geometry
    )
    advection_config = GeostrophicAdvectionConfig(enabled=args.advection)
    snow_config = SnowAlbedoConfig(enabled=args.snow)
    lon2d, lat2d, layers = compute_periodic_cycle_celsius(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        radiation_config=radiation_config,
        diffusion_config=diffusion_config,
        advection_config=advection_config,
        snow_config=snow_config,
        return_layer_map=True,
    )
    surface_cycle = layers["surface"]
    albedo_field = layers.get("albedo")

    cell_areas = spherical_cell_area(
        lon2d, lat2d, earth_radius_m=diffusion_config.earth_radius_m
    )
    surface_area_mean = float(
        np.average(surface_cycle.mean(axis=0), weights=cell_areas)
    )
    albedo_area_mean: float | None = None
    if albedo_field is not None:
        albedo_area_mean = float(np.average(albedo_field, weights=cell_areas))

    unit = _temperature_unit(args.fahrenheit)
    summary_parts = [
        "Surface layer:",
        f"Tmin={_convert_temperature(surface_cycle.min(), args.fahrenheit):.1f}{unit}",
        f"Tmax={_convert_temperature(surface_cycle.max(), args.fahrenheit):.1f}{unit}",
        f"simple mean={_convert_temperature(surface_cycle.mean(), args.fahrenheit):.1f}{unit}",
        f"area-weighted mean={_convert_temperature(surface_area_mean, args.fahrenheit):.1f}{unit}",
    ]
    if albedo_area_mean is not None:
        summary_parts.append(f"area-weighted mean albedo={albedo_area_mean:.3f}")

    print(" ".join(summary_parts))

    atmosphere_cycle = layers.get("atmosphere")
    if atmosphere_cycle is not None:
        atmosphere_area_mean = float(
            np.average(atmosphere_cycle.mean(axis=0), weights=cell_areas)
        )
        print(
            "Atmosphere layer: "
            f"Tmin={_convert_temperature(atmosphere_cycle.min(), args.fahrenheit):.1f}{unit}, "
            f"Tmax={_convert_temperature(atmosphere_cycle.max(), args.fahrenheit):.1f}{unit}, "
            f"simple mean={_convert_temperature(atmosphere_cycle.mean(), args.fahrenheit):.1f}{unit}, "
            f"area-weighted mean={_convert_temperature(atmosphere_area_mean, args.fahrenheit):.1f}{unit}"
        )

    layer_cycles: dict[str, np.ndarray] = {"Surface": surface_cycle}
    if atmosphere_cycle is not None:
        layer_cycles["Atmosphere"] = atmosphere_cycle

    plot_layered_monthly_temperature_cycle(
        lon2d,
        lat2d,
        layer_cycles,
        title="Temperature Cycle",
    )


if __name__ == "__main__":
    main()
