import argparse

import numpy as np

from climate_sim.modeling.bulk_coupling import BulkCouplingConfig
from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
from climate_sim.plotting import plot_layered_monthly_temperature_cycle
from climate_sim.utils.math import spherical_cell_area
from climate_sim.utils.solver import compute_periodic_cycle_celsius


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
        "--bulk-coupling",
        dest="bulk_coupling",
        action="store_true",
        default=True,
        help="Enable atmosphere-ocean bulk coupling (default)",
    )
    parser.add_argument(
        "--no-bulk-coupling",
        dest="bulk_coupling",
        action="store_false",
        help="Disable atmosphere-ocean bulk coupling",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    radiation_config = RadiationConfig(include_atmosphere=args.atmosphere)
    diffusion_config = DiffusionConfig(
        enabled=args.diffusion, use_spherical_geometry=args.diffusion_geometry
    )
    snow_config = SnowAlbedoConfig(enabled=args.snow)
    bulk_config = BulkCouplingConfig(
        enabled=args.bulk_coupling,
        atmosphere_heat_capacity=radiation_config.atmosphere_heat_capacity,
    )

    lon2d, lat2d, layers = compute_periodic_cycle_celsius(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        radiation_config=radiation_config,
        diffusion_config=diffusion_config,
        bulk_coupling_config=bulk_config,
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

    summary_parts = [
        "Surface layer:",
        f"Tmin={surface_cycle.min():.1f}°C",
        f"Tmax={surface_cycle.max():.1f}°C",
        f"simple mean={surface_cycle.mean():.1f}°C",
        f"area-weighted mean={surface_area_mean:.1f}°C",
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
            f"Tmin={atmosphere_cycle.min():.1f}°C, "
            f"Tmax={atmosphere_cycle.max():.1f}°C, "
            f"simple mean={atmosphere_cycle.mean():.1f}°C, "
            f"area-weighted mean={atmosphere_area_mean:.1f}°C"
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
