"""Compare climate model configurations by toggling model components."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.plotting import plot_monthly_temperature_cycle
from climate_sim.utils.solver import compute_periodic_cycle_celsius


def _build_configs(
    *,
    enable_diffusion: bool,
    include_atmosphere: bool,
) -> Tuple[RadiationConfig, DiffusionConfig]:
    radiation_cfg = RadiationConfig(include_atmosphere=include_atmosphere)
    diffusion_cfg = DiffusionConfig(enabled=enable_diffusion)
    return radiation_cfg, diffusion_cfg


def _summarise(flags: Dict[str, bool]) -> str:
    return ", ".join(f"{key}={'on' if value else 'off'}" for key, value in flags.items())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot anomalies between two climate model configurations.",
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
        "--base-diffusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable lateral diffusion in the baseline case",
    )
    parser.add_argument(
        "--base-atmosphere",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include an explicit atmospheric layer in the baseline case",
    )

    parser.add_argument(
        "--experiment-diffusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable lateral diffusion in the experiment case",
    )
    parser.add_argument(
        "--experiment-atmosphere",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include an explicit atmospheric layer in the experiment case",
    )

    args = parser.parse_args()

    base_rad, base_diff = _build_configs(
        enable_diffusion=args.base_diffusion,
        include_atmosphere=args.base_atmosphere,
    )
    exp_rad, exp_diff = _build_configs(
        enable_diffusion=args.experiment_diffusion,
        include_atmosphere=args.experiment_atmosphere,
    )

    lon2d, lat2d, base_layers = compute_periodic_cycle_celsius(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        radiation_config=base_rad,
        diffusion_config=base_diff,
        return_layer_map=True,
    )
    _, _, exp_layers = compute_periodic_cycle_celsius(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        radiation_config=exp_rad,
        diffusion_config=exp_diff,
        return_layer_map=True,
    )

    base_surface = base_layers["surface"]
    exp_surface = exp_layers["surface"]
    anomaly = exp_surface - base_surface

    base_summary = {
        "diffusion": args.base_diffusion,
        "atmosphere": args.base_atmosphere,
    }
    exp_summary = {
        "diffusion": args.experiment_diffusion,
        "atmosphere": args.experiment_atmosphere,
    }

    print("Baseline configuration:", _summarise(base_summary))
    print("Experiment configuration:", _summarise(exp_summary))
    print(
        f"Annual mean anomaly = {anomaly.mean():.2f} °C, "
        f"min = {anomaly.min():.2f} °C, max = {anomaly.max():.2f} °C",
    )

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        anomaly,
        title="Experiment − Baseline Surface Temperature (°C)",
    )


if __name__ == "__main__":
    main()
