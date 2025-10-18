"""Simple CLI to print predicted climates for selected locations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

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
    parser.add_argument(
        "--sensible-heat",
        dest="sensible_heat",
        action="store_true",
        default=True,
        help="Enable the neutral sensible heat exchange model (default)",
    )
    parser.add_argument(
        "--no-sensible-heat",
        dest="sensible_heat",
        action="store_false",
        help="Disable the neutral sensible heat exchange model",
    )
    parser.add_argument(
        "--elliptical-orbit",
        dest="elliptical_orbit",
        action="store_true",
        default=True,
        help="Apply Earth's orbital eccentricity correction to insolation (default)",
    )
    parser.add_argument(
        "--circular-orbit",
        dest="elliptical_orbit",
        action="store_false",
        help="Disable the orbital eccentricity correction and assume a circular orbit",
    )
    return parser.parse_args()


def _convert_temperature(
    values: np.ndarray, use_fahrenheit: bool
) -> np.ndarray:
    import numpy as np

    if not use_fahrenheit:
        return values
    converted = (np.asarray(values) * (9.0 / 5.0)) + 32.0
    return converted


def _temperature_unit(use_fahrenheit: bool) -> str:
    return "°F" if use_fahrenheit else "°C"


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
    monthly_surface_cycle: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    use_fahrenheit: bool,
) -> None:
    lat_idx, lon_idx = _nearest_cell_indices(
        lon2d, lat2d, location.latitude, location.longitude
    )
    monthly = monthly_surface_cycle[:, lat_idx, lon_idx]
    annual_mean = monthly.mean()
    annual_min = monthly.min()
    annual_max = monthly.max()
    monthly_display = _convert_temperature(monthly, use_fahrenheit)
    monthly_str = ", ".join(f"{value:5.1f}" for value in monthly_display)
    unit = _temperature_unit(use_fahrenheit)

    print(f"{location.name} ({location.latitude:.2f}°, {location.longitude:.2f}°)")
    print(f"  Monthly temps ({unit}): {monthly_str}")
    print(
        "  Summary: "
        f"mean={_convert_temperature(annual_mean, use_fahrenheit):4.1f} {unit}, "
        f"min={_convert_temperature(annual_min, use_fahrenheit):4.1f} {unit}, "
        f"max={_convert_temperature(annual_max, use_fahrenheit):4.1f} {unit}"
    )


def main() -> None:
    args = _parse_args()

    from climate_sim.modeling.advection import AdvectionConfig
    from climate_sim.modeling.diffusion import DiffusionConfig
    from climate_sim.modeling.radiation import RadiationConfig
    from climate_sim.modeling.sensible_heat_exchange import SensibleHeatExchangeConfig
    from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
    from climate_sim.utils.solver import compute_periodic_cycle_results
    from climate_sim.utils.math_core import area_weighted_mean, spherical_cell_area

    radiation_config = RadiationConfig(include_atmosphere=args.atmosphere)
    diffusion_config = DiffusionConfig(enabled=args.diffusion)
    advection_config = AdvectionConfig(enabled=args.advection)
    snow_config = SnowAlbedoConfig(enabled=args.snow)
    sensible_heat_config = SensibleHeatExchangeConfig(enabled=args.sensible_heat)
    lon2d, lat2d, layers = compute_periodic_cycle_results(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.elliptical_orbit,
        radiation_config=radiation_config,
        diffusion_config=diffusion_config,
        advection_config=advection_config,
        snow_config=snow_config,
        sensible_heat_config=sensible_heat_config,
        return_layer_map=True,
    )
    surface_cycle = layers["surface"]

    cell_areas = spherical_cell_area(
        lon2d, lat2d, earth_radius_m=diffusion_config.earth_radius_m
    )
    surface_area_mean = area_weighted_mean(surface_cycle.mean(axis=0), cell_areas)
    unit = _temperature_unit(args.fahrenheit)
    print(
        "Global surface layer: ",
        f"Tmin={_convert_temperature(surface_cycle.min(), args.fahrenheit):.1f} {unit}, ",
        f"Tmax={_convert_temperature(surface_cycle.max(), args.fahrenheit):.1f} {unit}, ",
        f"area-weighted mean={_convert_temperature(surface_area_mean, args.fahrenheit):.1f} {unit}"
    )

    locations = [
        Location("Chicago (IL)", 41.5, -87.6),
        Location("San Francisco (CA)", 37.8, -122.2),
        Location("Kinshasa (DRC)", -4.3, 15.3),
        Location("South Pole Vicinity", -85.0, 0.0),
    ]

    for location in locations:
        _summarize_location(
            location, surface_cycle, lon2d, lat2d, args.fahrenheit
        )


if __name__ == "__main__":
    main()
