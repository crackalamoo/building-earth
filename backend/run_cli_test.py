"""Simple CLI to print predicted climates for selected locations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from dotenv import load_dotenv

import numpy as np

from climate_sim.runtime.cli import add_common_model_arguments
from climate_sim.data.constants import R_EARTH_METERS

load_dotenv()


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the model climate at a few reference locations.",
    )
    from climate_sim.physics.radiation import RadiationConfig

    default_atmosphere = RadiationConfig().include_atmosphere
    add_common_model_arguments(
        parser,
        default_atmosphere=default_atmosphere,
        fahrenheit_help="Display temperatures in degrees Fahrenheit instead of Celsius",
        fahrenheit_options=("--fahrenheit",),
    )
    return parser.parse_args()


def _convert_temperature(values: np.ndarray, use_fahrenheit: bool) -> np.ndarray:
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
    lat_idx, lon_idx = _nearest_cell_indices(lon2d, lat2d, location.latitude, location.longitude)
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

    from climate_sim.physics.diffusion import DiffusionConfig
    from climate_sim.physics.radiation import RadiationConfig
    from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
    from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
    from climate_sim.physics.snow_albedo import SnowAlbedoConfig
    from climate_sim.physics.atmosphere.advection import AdvectionConfig
    from climate_sim.physics.atmosphere.wind import WindConfig
    from climate_sim.physics.orographic_effects import OrographicConfig
    from climate_sim.core.solver import solve_periodic_climate
    from climate_sim.core.math_core import area_weighted_mean, spherical_cell_area
    from climate_sim.runtime.config import ModelConfig

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
    orographic_config = OrographicConfig(enabled=args.orographic)
    model_config = ModelConfig(
        radiation=radiation_config,
        diffusion=diffusion_config,
        wind=wind_config,
        advection=advection_config,
        snow=snow_config,
        sensible_heat=sensible_heat_config,
        latent_heat=latent_heat_config,
        orographic=orographic_config,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.elliptical_orbit,
    )
    lon2d, lat2d, layers = solve_periodic_climate(
        resolution_deg=args.resolution,
        model_config=model_config,
        return_layer_map=True,
    )
    surface_cycle = layers["surface"]

    cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=R_EARTH_METERS)
    surface_area_mean = area_weighted_mean(surface_cycle.mean(axis=0), cell_areas)
    unit = _temperature_unit(args.fahrenheit)
    print(
        "Global surface layer: ",
        f"Tmin={_convert_temperature(surface_cycle.min(), args.fahrenheit):.1f} {unit}, ",
        f"Tmax={_convert_temperature(surface_cycle.max(), args.fahrenheit):.1f} {unit}, ",
        f"area-weighted mean={_convert_temperature(surface_area_mean, args.fahrenheit):.1f} {unit}",
    )

    locations = [
        Location("Chicago (IL)", 41.5, -87.6),
        Location("San Francisco (CA)", 37.8, -122.2),
        Location("Kinshasa (DRC)", -4.3, 15.3),
        Location("South Pole Vicinity", -85.0, 0.0),
    ]

    for location in locations:
        _summarize_location(location, surface_cycle, lon2d, lat2d, args.fahrenheit)


if __name__ == "__main__":
    main()
