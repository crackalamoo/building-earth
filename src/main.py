import argparse

import numpy as np

from climate_sim.modeling.advection import GeostrophicAdvectionConfig
from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
from climate_sim.plotting import plot_layered_monthly_temperature_cycle
from climate_sim.utils.atmosphere import adjust_temperature_by_elevation
from climate_sim.utils.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.utils.solver import compute_periodic_cycle_celsius
from climate_sim.utils.temperature import convert_temperature, temperature_unit

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
        area_weighted_mean(surface_cycle.mean(axis=0), cell_areas)
    )
    albedo_area_mean: float | None = None
    if albedo_field is not None:
        albedo_area_mean = float(area_weighted_mean(albedo_field, cell_areas))

    unit = temperature_unit(args.fahrenheit)
    summary_parts = [
        "Surface layer:",
        f"Tmin={convert_temperature(surface_cycle.min(), args.fahrenheit):.1f}{unit}",
        f"Tmax={convert_temperature(surface_cycle.max(), args.fahrenheit):.1f}{unit}",
        f"simple mean={convert_temperature(surface_cycle.mean(), args.fahrenheit):.1f}{unit}",
        f"area-weighted mean={convert_temperature(surface_area_mean, args.fahrenheit):.1f}{unit}",
    ]
    if albedo_area_mean is not None:
        summary_parts.append(f"area-weighted mean albedo={albedo_area_mean:.3f}")

    print(" ".join(summary_parts))

    atmosphere_cycle = layers.get("atmosphere")
    if atmosphere_cycle is not None:
        atmosphere_area_mean = float(
            area_weighted_mean(atmosphere_cycle.mean(axis=0), cell_areas)
        )
        print(
            "Atmosphere layer: "
            f"Tmin={convert_temperature(atmosphere_cycle.min(), args.fahrenheit):.1f}{unit}, "
            f"Tmax={convert_temperature(atmosphere_cycle.max(), args.fahrenheit):.1f}{unit}, "
            f"simple mean={convert_temperature(atmosphere_cycle.mean(), args.fahrenheit):.1f}{unit}, "
            f"area-weighted mean={convert_temperature(atmosphere_area_mean, args.fahrenheit):.1f}{unit}"
        )

    layer_cycles: dict[str, np.ndarray] = {"Surface": surface_cycle}
    if atmosphere_cycle is not None:
        layer_cycles["Atmosphere"] = atmosphere_cycle

        # The single-layer atmosphere represents the tropospheric slab whose depth is
        # characterised by the geostrophic advection scale height. Treat the layer's
        # temperature as representative of the slab midpoint and lapse-rate adjust to
        # obtain a 2 m diagnostic (ignoring local topography).
        troposphere_midpoint_m = 0.5 * advection_config.troposphere_scale_height_m
        delta_to_two_m = 2.0 - troposphere_midpoint_m
        atmosphere_2m_cycle = adjust_temperature_by_elevation(
            atmosphere_cycle, delta_to_two_m
        )
        layer_cycles["Atmosphere (2 m)"] = atmosphere_2m_cycle

    plot_layered_monthly_temperature_cycle(
        lon2d,
        lat2d,
        layer_cycles,
        title=f"Temperature Cycle ({unit})",
        use_fahrenheit=args.fahrenheit,
    )


if __name__ == "__main__":
    main()
