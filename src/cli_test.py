"""Simple CLI to print predicted climates for selected locations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the model climate at a few reference locations.",
    )
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
        "--atmosphere",
        dest="atmosphere",
        action="store_true",
        default=True,
        help="Include an explicit atmospheric layer",
    )
    parser.add_argument(
        "--no-atmosphere",
        dest="atmosphere",
        action="store_false",
        help="Exclude the atmospheric layer",
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
    parser.add_argument(
        "--ekman-friction",
        dest="ekman_friction",
        action="store_true",
        default=True,
        help="Include Ekman friction in the advection balance (default)",
    )
    parser.add_argument(
        "--no-ekman-friction",
        dest="ekman_friction",
        action="store_false",
        help="Exclude Ekman friction from the advection balance",
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
    return parser.parse_args()


def _nearest_cell_indices(
    lon2d: "np.ndarray", lat2d: "np.ndarray", latitude: float, longitude: float
) -> tuple[int, int]:
    import numpy as np

    lon_wrapped = longitude % 360.0
    lat_idx = int(np.abs(lat2d[:, 0] - latitude).argmin())
    lon_idx = int(np.abs(lon2d[0] - lon_wrapped).argmin())
    return lat_idx, lon_idx


def _summarize_location(
    location: Location,
    monthly_surface_cycle: "np.ndarray",
    lon2d: "np.ndarray",
    lat2d: "np.ndarray",
) -> None:
    lat_idx, lon_idx = _nearest_cell_indices(
        lon2d, lat2d, location.latitude, location.longitude
    )
    monthly = monthly_surface_cycle[:, lat_idx, lon_idx]
    annual_mean = float(monthly.mean())
    annual_min = float(monthly.min())
    annual_max = float(monthly.max())
    monthly_str = ", ".join(f"{value:5.1f}" for value in monthly)

    print(f"{location.name} ({location.latitude:.2f}°, {location.longitude:.2f}°)")
    print(f"  Monthly temps (°C): {monthly_str}")
    print(
        "  Summary: "
        f"mean={annual_mean:4.1f} °C, "
        f"min={annual_min:4.1f} °C, "
        f"max={annual_max:4.1f} °C"
    )


def main() -> None:
    args = _parse_args()

    from climate_sim.modeling.advection import GeostrophicAdvectionConfig
    from climate_sim.modeling.diffusion import DiffusionConfig
    from climate_sim.modeling.radiation import RadiationConfig
    from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
    from climate_sim.utils.math import spherical_cell_area
    from climate_sim.utils.solver import compute_periodic_cycle_celsius

    radiation_config = RadiationConfig(include_atmosphere=args.atmosphere)
    diffusion_config = DiffusionConfig(
        enabled=args.diffusion, use_spherical_geometry=args.diffusion_geometry
    )
    advection_config = GeostrophicAdvectionConfig(
        enabled=args.advection, ekman_friction=args.ekman_friction
    )
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
    print(
        "Configuration: "
        f"atmosphere={'on' if args.atmosphere else 'off'}, "
        f"diffusion={'on' if args.diffusion else 'off'}, "
        f"advection={'on' if args.advection else 'off'}, "
        f"Ekman friction={'on' if args.ekman_friction else 'off'}, "
        f"snow={'on' if args.snow else 'off'}"
    )
    surface_cycle = layers["surface"]
    cell_areas = spherical_cell_area(
        lon2d, lat2d, earth_radius_m=diffusion_config.earth_radius_m
    )
    surface_area_mean = float(
        np.average(surface_cycle.mean(axis=0), weights=cell_areas)
    )

    print(
        "Global surface summary: "
        f"Tmin={surface_cycle.min():.1f} °C, "
        f"Tmax={surface_cycle.max():.1f} °C, "
        f"simple mean={surface_cycle.mean():.1f} °C, "
        f"area-weighted mean={surface_area_mean:.1f} °C"
    )

    locations = [
        Location("Chicago (IL)", 41.5, -87.6),
        Location("San Francisco (CA)", 37.8, -122.2),
        Location("Kinshasa (DRC)", -4.3, 15.3),
        Location("South Pole Vicinity", -85.0, 0.0),
    ]

    for location in locations:
        _summarize_location(location, surface_cycle, lon2d, lat2d)


if __name__ == "__main__":
    main()
