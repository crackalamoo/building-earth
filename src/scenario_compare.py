"""Compare climate model configurations by toggling model components."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import cmocean
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig
from climate_sim.physics.atmosphere.advection import AdvectionConfig
from climate_sim.physics.atmosphere.boundary_layer import BoundaryLayerConfig
from climate_sim.plotting import plot_layered_monthly_temperature_cycle
from climate_sim.runtime.cli import (
    add_boolean_flag,
    add_resolution_argument,
    add_solar_constant_argument,
    add_temperature_unit_argument,
)
from climate_sim.runtime.config import ModelConfig
from climate_sim.core.solver import solve_periodic_climate
from climate_sim.core.units import convert_temperature, temperature_unit

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
    default_atmosphere = RadiationConfig().include_atmosphere
    add_resolution_argument(parser)
    add_solar_constant_argument(parser)
    add_boolean_flag(
        parser,
        dest="base_elliptical_orbit",
        default=True,
        enable_option="--base-elliptical-orbit",
        disable_option="--base-circular-orbit",
        help_enable="Apply Earth's orbital eccentricity correction in the baseline case (default)",
        help_disable="Disable the orbital eccentricity correction in the baseline case",
    )
    add_boolean_flag(
        parser,
        dest="experiment_elliptical_orbit",
        default=True,
        enable_option="--exp-elliptical-orbit",
        disable_option="--exp-circular-orbit",
        help_enable="Apply Earth's orbital eccentricity correction in the experiment case (default)",
        help_disable="Disable the orbital eccentricity correction in the experiment case",
    )
    add_boolean_flag(
        parser,
        dest="base_snow",
        default=True,
        enable_option="--base-snow",
        disable_option="--no-base-snow",
        help_enable="Enable snow-albedo adjustments in the baseline case (default)",
        help_disable="Disable snow-albedo adjustments in the baseline case",
    )
    add_boolean_flag(
        parser,
        dest="base_latent_heat",
        default=True,
        enable_option="--base-latent-heat",
        disable_option="--no-base-latent-heat",
        help_enable="Include latent heat of fusion in the baseline surface heat capacity (default)",
        help_disable="Disable the baseline latent heat of fusion adjustment",
    )
    add_boolean_flag(
        parser,
        dest="base_diffusion",
        default=True,
        enable_option="--base-diffusion",
        disable_option="--no-base-diffusion",
        help_enable="Enable lateral diffusion in the baseline case (default)",
        help_disable="Disable lateral diffusion in the baseline case",
    )
    add_boolean_flag(
        parser,
        dest="base_atmosphere",
        default=default_atmosphere,
        enable_option="--base-atmosphere",
        disable_option="--no-base-atmosphere",
        help_enable="Include an explicit atmospheric layer in the baseline case (default)",
        help_disable="Exclude the atmospheric layer in the baseline case",
    )
    add_boolean_flag(
        parser,
        dest="experiment_diffusion",
        default=True,
        enable_option="--exp-diffusion",
        disable_option="--no-exp-diffusion",
        help_enable="Enable lateral diffusion in the experiment case (default)",
        help_disable="Disable lateral diffusion in the experiment case",
    )
    add_boolean_flag(
        parser,
        dest="experiment_atmosphere",
        default=default_atmosphere,
        enable_option="--exp-atmosphere",
        disable_option="--no-exp-atmosphere",
        help_enable="Include an explicit atmospheric layer in the experiment case (default)",
        help_disable="Exclude the atmospheric layer in the experiment case",
    )
    add_boolean_flag(
        parser,
        dest="experiment_snow",
        default=True,
        enable_option="--exp-snow",
        disable_option="--no-exp-snow",
        help_enable="Enable snow-albedo adjustments in the experiment case (default)",
        help_disable="Disable snow-albedo adjustments in the experiment case",
    )
    add_boolean_flag(
        parser,
        dest="experiment_latent_heat",
        default=True,
        enable_option="--exp-latent-heat",
        disable_option="--no-exp-latent-heat",
        help_enable="Include latent heat of fusion in the experiment surface heat capacity (default)",
        help_disable="Disable the experiment latent heat of fusion adjustment",
    )
    add_boolean_flag(
        parser,
        dest="base_bulk_exchange",
        default=True,
        enable_option="--base-bulk-exchange",
        disable_option="--no-base-bulk-exchange",
        help_enable="Enable bulk sensible heat exchange in the baseline case (default)",
        help_disable="Disable bulk sensible heat exchange in the baseline case",
    )
    add_boolean_flag(
        parser,
        dest="base_lapse_rate_elevation",
        default=False,
        enable_option="--base-lapse-rate-elevation",
        disable_option="--no-base-lapse-rate-elevation",
        help_enable="Include lapse-rate elevation corrections in the baseline case",
        help_disable="Ignore lapse-rate elevation in the baseline case (default)",
    )
    add_boolean_flag(
        parser,
        dest="base_latent_heat_exchange",
        default=True,
        enable_option="--base-latent-heat-exchange",
        disable_option="--no-base-latent-heat-exchange",
        help_enable="Enable latent heat exchange in the baseline case (default)",
        help_disable="Disable latent heat exchange in the baseline case",
    )
    add_boolean_flag(
        parser,
        dest="experiment_bulk_exchange",
        default=True,
        enable_option="--exp-bulk-exchange",
        disable_option="--no-exp-bulk-exchange",
        help_enable="Enable bulk sensible heat exchange in the experiment case (default)",
        help_disable="Disable bulk sensible heat exchange in the experiment case",
    )
    add_boolean_flag(
        parser,
        dest="experiment_lapse_rate_elevation",
        default=False,
        enable_option="--exp-lapse-rate-elevation",
        disable_option="--no-exp-lapse-rate-elevation",
        help_enable="Include lapse-rate elevation corrections in the experiment case",
        help_disable="Ignore lapse-rate elevation in the experiment case (default)",
    )
    add_boolean_flag(
        parser,
        dest="experiment_latent_heat_exchange",
        default=True,
        enable_option="--exp-latent-heat-exchange",
        disable_option="--no-exp-latent-heat-exchange",
        help_enable="Enable latent heat exchange in the experiment case (default)",
        help_disable="Disable latent heat exchange in the experiment case",
    )
    add_boolean_flag(
        parser,
        dest="base_advection",
        default=True,
        enable_option="--base-advection",
        disable_option="--no-base-advection",
        help_enable="Enable atmospheric advection in the baseline case",
        help_disable="Disable atmospheric advection in the baseline case (default)",
    )
    add_boolean_flag(
        parser,
        dest="experiment_advection",
        default=True,
        enable_option="--exp-advection",
        disable_option="--no-exp-advection",
        help_enable="Enable atmospheric advection in the experiment case",
        help_disable="Disable atmospheric advection in the experiment case (default)",
    )
    add_boolean_flag(
        parser,
        dest="base_boundary_layer",
        default=True,
        enable_option="--base-boundary-layer",
        disable_option="--no-base-boundary-layer",
        help_enable="Enable boundary layer in the baseline case (default)",
        help_disable="Disable boundary layer in the baseline case",
    )
    add_boolean_flag(
        parser,
        dest="experiment_boundary_layer",
        default=True,
        enable_option="--exp-boundary-layer",
        disable_option="--no-exp-boundary-layer",
        help_enable="Enable boundary layer in the experiment case (default)",
        help_disable="Disable boundary layer in the experiment case",
    )
    add_temperature_unit_argument(
        parser,
        help_text="Display anomalies in degrees Fahrenheit instead of Celsius",
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

    base_snow = SnowAlbedoConfig(
        enabled=args.base_snow,
        latent_heat_enabled=args.base_latent_heat,
    )
    exp_snow = SnowAlbedoConfig(
        enabled=args.experiment_snow,
        latent_heat_enabled=args.experiment_latent_heat,
    )

    base_sensible_heat = SensibleHeatExchangeConfig(
        enabled=args.base_bulk_exchange,
        include_lapse_rate_elevation=args.base_lapse_rate_elevation,
    )
    exp_sensible_heat = SensibleHeatExchangeConfig(
        enabled=args.experiment_bulk_exchange,
        include_lapse_rate_elevation=args.experiment_lapse_rate_elevation,
    )
    base_latent_heat = LatentHeatExchangeConfig(
        enabled=args.base_latent_heat_exchange,
    )
    exp_latent_heat = LatentHeatExchangeConfig(
        enabled=args.experiment_latent_heat_exchange,
    )

    base_advection = AdvectionConfig(
        enabled=args.base_advection,
    )
    exp_advection = AdvectionConfig(
        enabled=args.experiment_advection,
    )

    base_boundary_layer = BoundaryLayerConfig(
        enabled=args.base_boundary_layer,
    )
    exp_boundary_layer = BoundaryLayerConfig(
        enabled=args.experiment_boundary_layer,
    )

    base_model_config = ModelConfig(
        radiation=base_rad,
        diffusion=base_diff,
        advection=base_advection,
        snow=base_snow,
        sensible_heat=base_sensible_heat,
        latent_heat=base_latent_heat,
        boundary_layer=base_boundary_layer,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.base_elliptical_orbit,
    )
    exp_model_config = ModelConfig(
        radiation=exp_rad,
        diffusion=exp_diff,
        advection=exp_advection,
        snow=exp_snow,
        sensible_heat=exp_sensible_heat,
        latent_heat=exp_latent_heat,
        boundary_layer=exp_boundary_layer,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.experiment_elliptical_orbit,
    )

    lon2d, lat2d, base_layers = solve_periodic_climate(
        resolution_deg=args.resolution,
        model_config=base_model_config,
        return_layer_map=True,
    )
    _, _, exp_layers = solve_periodic_climate(
        resolution_deg=args.resolution,
        model_config=exp_model_config,
        return_layer_map=True,
    )

    base_surface = base_layers["surface"]
    exp_surface = exp_layers["surface"]
    anomalies: Dict[str, np.ndarray] = {"Surface": exp_surface - base_surface}

    base_atmosphere = base_layers.get("atmosphere")
    exp_atmosphere = exp_layers.get("atmosphere")
    if base_atmosphere is not None and exp_atmosphere is not None:
        anomalies["Atmosphere"] = exp_atmosphere - base_atmosphere

    base_two_meter = base_layers.get("temperature_2m")
    exp_two_meter = exp_layers.get("temperature_2m")
    if base_two_meter is not None and exp_two_meter is not None:
        anomalies["Two-meter"] = exp_two_meter - base_two_meter

    base_summary = {
        "elliptical_orbit": args.base_elliptical_orbit,
        "diffusion": args.base_diffusion,
        "atmosphere": args.base_atmosphere,
        "snow": args.base_snow,
        "latent_heat": args.base_latent_heat,
        "bulk_exchange": args.base_bulk_exchange,
        "latent_heat_exchange": args.base_latent_heat_exchange,
        "lapse_rate_elevation": args.base_lapse_rate_elevation,
        "advection": args.base_advection,
    }
    exp_summary = {
        "elliptical_orbit": args.experiment_elliptical_orbit,
        "diffusion": args.experiment_diffusion,
        "atmosphere": args.experiment_atmosphere,
        "snow": args.experiment_snow,
        "latent_heat": args.experiment_latent_heat,
        "bulk_exchange": args.experiment_bulk_exchange,
        "latent_heat_exchange": args.experiment_latent_heat_exchange,
        "lapse_rate_elevation": args.experiment_lapse_rate_elevation,
        "advection": args.experiment_advection,
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

    for anomaly in anomalies:
        anomalies[anomaly] = np.concatenate([anomalies[anomaly], np.mean(anomalies[anomaly], axis=0, keepdims=True)], axis=0)

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
