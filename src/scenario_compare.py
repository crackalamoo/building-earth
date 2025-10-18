"""Compare climate model configurations by toggling model components."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import cmocean
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
from climate_sim.plotting import plot_layered_monthly_temperature_cycle
from climate_sim.utils.atmosphere import adjust_temperature_by_elevation
from climate_sim.utils.solver import compute_periodic_cycle_results
from climate_sim.utils.temperature import convert_temperature, temperature_unit

from dotenv import load_dotenv
load_dotenv()


def _build_configs(
    *,
    enable_diffusion: bool,
    include_atmosphere: bool,
) -> Tuple[RadiationConfig, DiffusionConfig]:
    radiation_cfg = RadiationConfig(include_atmosphere=include_atmosphere)
    diffusion_cfg = DiffusionConfig(enabled=enable_diffusion)
    return radiation_cfg, diffusion_cfg


def _summarize(flags: Dict[str, bool]) -> str:
    return ", ".join(f"{key}={'on' if value else 'off'}" for key, value in flags.items())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot anomalies between two climate model configurations.",
    )
    parser.add_argument(
        "--resolution", "-r",
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
        "--base-elliptical-orbit",
        dest="base_elliptical_orbit",
        action="store_true",
        default=True,
        help="Apply Earth's orbital eccentricity correction in the baseline case (default)",
    )
    parser.add_argument(
        "--base-circular-orbit",
        dest="base_elliptical_orbit",
        action="store_false",
        help="Disable the orbital eccentricity correction in the baseline case",
    )
    parser.add_argument(
        "--exp-elliptical-orbit",
        dest="experiment_elliptical_orbit",
        action="store_true",
        default=True,
        help="Apply Earth's orbital eccentricity correction in the experiment case (default)",
    )
    parser.add_argument(
        "--exp-circular-orbit",
        dest="experiment_elliptical_orbit",
        action="store_false",
        help="Disable the orbital eccentricity correction in the experiment case",
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
        "--base-bulk-exchange",
        dest="base_bulk_exchange",
        action="store_true",
        default=True,
        help="Enable bulk sensible heat exchange in the baseline case (default)",
    )
    parser.add_argument(
        "--no-base-bulk-exchange",
        dest="base_bulk_exchange",
        action="store_false",
        help="Disable bulk sensible heat exchange in the baseline case",
    )
    parser.add_argument(
        "--exp-bulk-exchange",
        dest="experiment_bulk_exchange",
        action="store_true",
        default=True,
        help="Enable bulk sensible heat exchange in the experiment case (default)",
    )
    parser.add_argument(
        "--no-exp-bulk-exchange",
        dest="experiment_bulk_exchange",
        action="store_false",
        help="Disable bulk sensible heat exchange in the experiment case",
    )
    parser.add_argument(
        "--fahrenheit", "-f",
        dest="fahrenheit",
        action="store_true",
        help="Display anomalies in degrees Fahrenheit instead of Celsius",
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

    base_snow = SnowAlbedoConfig(enabled=args.base_snow)
    exp_snow = SnowAlbedoConfig(enabled=args.experiment_snow)

    base_sensible_heat = SensibleHeatExchangeConfig(enabled=args.base_bulk_exchange)
    exp_sensible_heat = SensibleHeatExchangeConfig(enabled=args.experiment_bulk_exchange)

    lon2d, lat2d, base_layers = compute_periodic_cycle_results(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.base_elliptical_orbit,
        radiation_config=base_rad,
        diffusion_config=base_diff,
        snow_config=base_snow,
        sensible_heat_config=base_sensible_heat,
        return_layer_map=True,
    )
    _, _, exp_layers = compute_periodic_cycle_results(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.experiment_elliptical_orbit,
        radiation_config=exp_rad,
        diffusion_config=exp_diff,
        snow_config=exp_snow,
        sensible_heat_config=exp_sensible_heat,
        return_layer_map=True,
    )

    base_surface = base_layers["surface"]
    exp_surface = exp_layers["surface"]
    anomalies: Dict[str, np.ndarray] = {"Surface": exp_surface - base_surface}

    base_atmosphere = base_layers.get("atmosphere")
    exp_atmosphere = exp_layers.get("atmosphere")
    if base_atmosphere is not None and exp_atmosphere is not None:
        anomalies["Atmosphere"] = exp_atmosphere - base_atmosphere

        atmosphere_height = 5000  # effective emission layer height (m)
        delta_to_two_m = 2.0 - atmosphere_height
        base_two_meter = adjust_temperature_by_elevation(
            base_atmosphere, delta_to_two_m
        )
        exp_two_meter = adjust_temperature_by_elevation(
            exp_atmosphere, delta_to_two_m
        )
        anomalies["Two-meter"] = exp_two_meter - base_two_meter

    base_summary = {
        "elliptical_orbit": args.base_elliptical_orbit,
        "diffusion": args.base_diffusion,
        "atmosphere": args.base_atmosphere,
        "snow": args.base_snow,
        "bulk_exchange": args.base_bulk_exchange,
    }
    exp_summary = {
        "elliptical_orbit": args.experiment_elliptical_orbit,
        "diffusion": args.experiment_diffusion,
        "atmosphere": args.experiment_atmosphere,
        "snow": args.experiment_snow,
        "bulk_exchange": args.experiment_bulk_exchange,
    }

    print("Baseline configuration:", _summarize(base_summary))
    print("Experiment configuration:", _summarize(exp_summary))
    unit = temperature_unit(args.fahrenheit)

    for mode, field in anomalies.items():
        display_field = convert_temperature(field, args.fahrenheit, is_delta=True)
        assert isinstance(display_field, np.ndarray)
        print(
            f"{mode} anomaly – mean = {display_field.mean():.2f} {unit}, "
            f"min = {display_field.min():.2f} {unit}, max = {display_field.max():.2f} {unit}"
        )

    max_abs_c = max(float(np.max(np.abs(field))) for field in anomalies.values())
    if max_abs_c <= 0.0 or not np.isfinite(max_abs_c):
        max_abs_c = 1.0
    max_abs_display = float(
        convert_temperature(np.array([max_abs_c]), args.fahrenheit, is_delta=True)[0]
    )
    vmax = max_abs_display if max_abs_display > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    plot_layered_monthly_temperature_cycle(
        lon2d,
        lat2d,
        anomalies,
        title=f"Experiment − Baseline Temperature Anomalies ({unit})",
        cmap=cmocean.cm.balance,
        norm=norm,
        colorbar_label=f"Temperature anomaly ({unit})",
        use_fahrenheit=args.fahrenheit,
        value_is_delta=True,
    )


if __name__ == "__main__":
    main()
