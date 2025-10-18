"""Compare climate model configurations by toggling model components."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import cmocean
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from climate_sim.modeling.advection import GeostrophicAdvectionConfig
from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
from climate_sim.plotting import plot_monthly_temperature_cycle
from climate_sim.utils.solver import compute_periodic_cycle_celsius
from climate_sim.utils.temperature import convert_temperature, temperature_unit

from dotenv import load_dotenv
load_dotenv()


def _build_configs(
    *,
    enable_diffusion: bool,
    enable_advection: bool,
    include_atmosphere: bool,
    use_geometry: bool,
) -> Tuple[RadiationConfig, DiffusionConfig, GeostrophicAdvectionConfig]:
    radiation_cfg = RadiationConfig(include_atmosphere=include_atmosphere)
    diffusion_cfg = DiffusionConfig(
        enabled=enable_diffusion, use_spherical_geometry=use_geometry
    )
    advection_cfg = GeostrophicAdvectionConfig(enabled=enable_advection)
    return radiation_cfg, diffusion_cfg, advection_cfg


def _summarize(flags: Dict[str, bool]) -> str:
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
        "--base-snow",
        dest="base_snow",
        action="store_true",
        default=True,
        help="Enable snow-albedo adjustments in the baseline case (default)",
    )
    parser.add_argument(
        "--no-base-snow",
        dest="base_snow",
        action="store_false",
        help="Disable snow-albedo adjustments in the baseline case",
    )

    parser.add_argument(
        "--base-diffusion",
        dest="base_diffusion",
        action="store_true",
        default=True,
        help="Enable lateral diffusion in the baseline case (default)",
    )
    parser.add_argument(
        "--no-base-diffusion",
        dest="base_diffusion",
        action="store_false",
        help="Disable lateral diffusion in the baseline case",
    )
    parser.add_argument(
        "--base-advection",
        dest="base_advection",
        action="store_true",
        default=True,
        help="Enable geostrophic atmospheric advection in the baseline case (default)",
    )
    parser.add_argument(
        "--no-base-advection",
        dest="base_advection",
        action="store_false",
        help="Disable geostrophic atmospheric advection in the baseline case",
    )
    parser.add_argument(
        "--base-atmosphere",
        dest="base_atmosphere",
        action="store_true",
        default=True,
        help="Include an explicit atmospheric layer in the baseline case (default)",
    )
    parser.add_argument(
        "--no-base-atmosphere",
        dest="base_atmosphere",
        action="store_false",
        help="Exclude the atmospheric layer in the baseline case",
    )
    parser.add_argument(
        "--base-diffusion-geometry",
        dest="base_geometry",
        action="store_true",
        default=True,
        help="Use spherical geometry in baseline diffusion (default)",
    )
    parser.add_argument(
        "--no-base-diffusion-geometry",
        dest="base_geometry",
        action="store_false",
        help="Use planar diffusion geometry in the baseline case",
    )

    parser.add_argument(
        "--exp-diffusion",
        dest="experiment_diffusion",
        action="store_true",
        default=True,
        help="Enable lateral diffusion in the experiment case (default)",
    )
    parser.add_argument(
        "--no-exp-diffusion",
        dest="experiment_diffusion",
        action="store_false",
        help="Disable lateral diffusion in the experiment case",
    )
    parser.add_argument(
        "--exp-advection",
        dest="experiment_advection",
        action="store_true",
        default=True,
        help="Enable geostrophic atmospheric advection in the experiment case (default)",
    )
    parser.add_argument(
        "--no-exp-advection",
        dest="experiment_advection",
        action="store_false",
        help="Disable geostrophic atmospheric advection in the experiment case",
    )
    parser.add_argument(
        "--exp-atmosphere",
        dest="experiment_atmosphere",
        action="store_true",
        default=True,
        help="Include an explicit atmospheric layer in the experiment case (default)",
    )
    parser.add_argument(
        "--no-exp-atmosphere",
        dest="experiment_atmosphere",
        action="store_false",
        help="Exclude the atmospheric layer in the experiment case",
    )
    parser.add_argument(
        "--exp-diffusion-geometry",
        dest="experiment_geometry",
        action="store_true",
        default=True,
        help="Use spherical geometry in experiment diffusion (default)",
    )
    parser.add_argument(
        "--no-exp-diffusion-geometry",
        dest="experiment_geometry",
        action="store_false",
        help="Use planar diffusion geometry in the experiment case",
    )
    parser.add_argument(
        "--exp-snow",
        dest="experiment_snow",
        action="store_true",
        default=True,
        help="Enable snow-albedo adjustments in the experiment case (default)",
    )
    parser.add_argument(
        "--no-exp-snow",
        dest="experiment_snow",
        action="store_false",
        help="Disable snow-albedo adjustments in the experiment case",
    )
    parser.add_argument(
        "--fahrenheit",
        dest="fahrenheit",
        action="store_true",
        help="Display anomalies in degrees Fahrenheit instead of Celsius",
    )

    args = parser.parse_args()

    base_rad, base_diff, base_adv = _build_configs(
        enable_diffusion=args.base_diffusion,
        enable_advection=args.base_advection,
        include_atmosphere=args.base_atmosphere,
        use_geometry=args.base_geometry,
    )
    exp_rad, exp_diff, exp_adv = _build_configs(
        enable_diffusion=args.experiment_diffusion,
        enable_advection=args.experiment_advection,
        include_atmosphere=args.experiment_atmosphere,
        use_geometry=args.experiment_geometry,
    )

    base_snow = SnowAlbedoConfig(enabled=args.base_snow)
    exp_snow = SnowAlbedoConfig(enabled=args.experiment_snow)

    lon2d, lat2d, base_layers = compute_periodic_cycle_celsius(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        radiation_config=base_rad,
        diffusion_config=base_diff,
        advection_config=base_adv,
        snow_config=base_snow,
        return_layer_map=True,
    )
    _, _, exp_layers = compute_periodic_cycle_celsius(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        radiation_config=exp_rad,
        diffusion_config=exp_diff,
        advection_config=exp_adv,
        snow_config=exp_snow,
        return_layer_map=True,
    )

    base_surface = base_layers["surface"]
    exp_surface = exp_layers["surface"]
    anomaly = exp_surface - base_surface

    base_summary = {
        "diffusion": args.base_diffusion,
        "advection": args.base_advection,
        "atmosphere": args.base_atmosphere,
        "geometry": args.base_geometry,
        "snow": args.base_snow,
    }
    exp_summary = {
        "diffusion": args.experiment_diffusion,
        "advection": args.experiment_advection,
        "atmosphere": args.experiment_atmosphere,
        "geometry": args.experiment_geometry,
        "snow": args.experiment_snow,
    }

    print("Baseline configuration:", _summarize(base_summary))
    print("Experiment configuration:", _summarize(exp_summary))
    unit = temperature_unit(args.fahrenheit)
    display_anomaly = convert_temperature(anomaly, args.fahrenheit, is_delta=True)
    assert isinstance(display_anomaly, np.ndarray)
    print(
        f"Annual mean anomaly = {display_anomaly.mean():.2f} {unit}, "
        f"min = {display_anomaly.min():.2f} {unit}, max = {display_anomaly.max():.2f} {unit}",
    )

    anomaly_abs = float(np.max(np.abs(display_anomaly)))
    vmax = anomaly_abs if anomaly_abs > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        anomaly,
        title=f"Experiment − Baseline Surface Temperature ({unit})",
        cmap=cmocean.cm.balance,
        norm=norm,
        colorbar_label=f"Temperature anomaly ({unit})",
        use_fahrenheit=args.fahrenheit,
        value_is_delta=True,
    )


if __name__ == "__main__":
    main()
