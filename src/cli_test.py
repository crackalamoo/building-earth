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

    from climate_sim.modeling.radiation import RadiationConfig
    from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
    from climate_sim.utils.solver import compute_periodic_cycle_celsius

    config = RadiationConfig(include_atmosphere=True)
    snow_config = SnowAlbedoConfig(enabled=args.snow)
    lon2d, lat2d, layers = compute_periodic_cycle_celsius(
        radiation_config=config,
        snow_config=snow_config,
        return_layer_map=True,
    )
    surface_cycle = layers["surface"]

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
