"""Helpers for building consistent command-line interfaces."""

import argparse
from typing import Iterable, Sequence


def add_resolution_argument(
    parser: argparse.ArgumentParser,
    *,
    default: float = 1.0,
    options: Sequence[str] = ("--resolution", "-r"),
    help_text: str = "Grid resolution in degrees",
) -> None:
    """Attach a resolution argument to *parser*."""

    parser.add_argument(*options, type=float, default=default, help=help_text)


def add_solar_constant_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = "Override the solar constant (W m^-2)",
) -> None:
    """Attach a solar-constant override argument."""

    parser.add_argument("--solar-constant", type=float, default=None, help=help_text)


def add_boolean_flag(
    parser: argparse.ArgumentParser,
    *,
    dest: str,
    default: bool,
    enable_option: str,
    disable_option: str,
    help_enable: str,
    help_disable: str,
    enable_aliases: Iterable[str] | None = None,
    disable_aliases: Iterable[str] | None = None,
) -> None:
    """Register paired ``--feature``/``--no-feature`` style flags."""

    enable_names = [enable_option]
    if enable_aliases:
        enable_names.extend(enable_aliases)
    disable_names = [disable_option]
    if disable_aliases:
        disable_names.extend(disable_aliases)

    parser.add_argument(
        *enable_names,
        dest=dest,
        action="store_true",
        default=default,
        help=help_enable,
    )
    parser.add_argument(
        *disable_names,
        dest=dest,
        action="store_false",
        help=help_disable,
    )


def add_temperature_unit_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
    options: Sequence[str] = ("--fahrenheit", "-f"),
    dest: str = "fahrenheit",
) -> None:
    """Add a Fahrenheit toggle."""

    parser.add_argument(*options, dest=dest, action="store_true", help=help_text)


def add_headless_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = "Disable plotting (headless mode for non-interactive execution)",
) -> None:
    """Add a headless mode toggle."""

    parser.add_argument("--headless", dest="headless", action="store_true", help=help_text)


def add_interpolation_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add interpolation-related arguments."""

    add_boolean_flag(
        parser,
        dest="interpolate",
        default=False,
        enable_option="--interpolate",
        disable_option="--no-interpolate",
        help_enable="Interpolate temperature fields to 1° resolution for display (includes lapse rate correction for land)",
        help_disable="Display at solver resolution (default)",
    )


def add_common_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_atmosphere: bool,
    include_temperature_unit: bool = True,
    fahrenheit_help: str | None = None,
    fahrenheit_options: Sequence[str] = ("--fahrenheit", "-f"),
) -> None:
    """Attach the standard set of model configuration switches."""

    add_resolution_argument(parser)
    add_solar_constant_argument(parser)
    add_boolean_flag(
        parser,
        dest="diffusion",
        default=True,
        enable_option="--diffusion",
        disable_option="--no-diffusion",
        help_enable="Enable lateral diffusion (default)",
        help_disable="Disable lateral diffusion",
    )
    add_boolean_flag(
        parser,
        dest="atmosphere",
        default=default_atmosphere,
        enable_option="--atmosphere",
        disable_option="--no-atmosphere",
        help_enable="Include an explicit atmospheric layer",
        help_disable="Exclude the atmospheric layer",
    )
    add_boolean_flag(
        parser,
        dest="snow",
        default=True,
        enable_option="--snow",
        disable_option="--no-snow",
        help_enable="Enable diagnostic snow-albedo adjustments (default)",
        help_disable="Disable snow-albedo adjustments",
    )
    add_boolean_flag(
        parser,
        dest="latent_heat",
        default=True,
        enable_option="--latent-heat",
        disable_option="--no-latent-heat",
        help_enable="Include latent heat of fusion in the surface heat capacity (default)",
        help_disable="Disable the latent heat of fusion adjustment",
    )
    add_boolean_flag(
        parser,
        dest="bulk_exchange",
        default=True,
        enable_option="--bulk-exchange",
        disable_option="--no-bulk-exchange",
        help_enable="Enable the neutral bulk sensible heat exchange model (default)",
        help_disable="Disable the neutral bulk sensible heat exchange model",
    )
    add_boolean_flag(
        parser,
        dest="latent_heat_exchange",
        default=True,
        enable_option="--latent-heat-exchange",
        disable_option="--no-latent-heat-exchange",
        help_enable="Enable latent heat exchange between surface and atmosphere (default)",
        help_disable="Disable latent heat exchange between surface and atmosphere",
    )
    add_boolean_flag(
        parser,
        dest="lapse_rate_elevation",
        default=True,
        enable_option="--lapse-rate-elevation",
        disable_option="--no-lapse-rate-elevation",
        help_enable="Apply lapse-rate adjustments using topographic elevation",
        help_disable="Ignore topographic elevation in lapse-rate adjustments (default)",
    )
    add_boolean_flag(
        parser,
        dest="advection",
        default=True,
        enable_option="--advection",
        disable_option="--no-advection",
        help_enable="Enable atmospheric advection by wind (default)",
        help_disable="Disable atmospheric advection",
    )
    add_boolean_flag(
        parser,
        dest="elliptical_orbit",
        default=True,
        enable_option="--elliptical-orbit",
        disable_option="--circular-orbit",
        help_enable="Apply Earth's orbital eccentricity correction to insolation (default)",
        help_disable="Disable the orbital eccentricity correction and assume a circular orbit",
    )
    add_boolean_flag(
        parser,
        dest="ocean_advection",
        default=True,
        enable_option="--ocean-advection",
        disable_option="--no-ocean-advection",
        help_enable="Enable ocean heat advection by Sverdrup-Stommel currents (default)",
        help_disable="Disable ocean heat advection",
    )
    add_boolean_flag(
        parser,
        dest="vertical_motion",
        default=True,
        enable_option="--vertical-motion",
        disable_option="--no-vertical-motion",
        help_enable="Enable vertical motion physics (subsidence warming + latent heat release) (default)",
        help_disable="Disable vertical motion physics",
    )
    add_boolean_flag(
        parser,
        dest="orographic",
        default=True,
        enable_option="--orographic",
        disable_option="--no-orographic",
        help_enable="Enable orographic effects (terrain-induced vertical velocity and flow blocking) (default)",
        help_disable="Disable orographic effects",
    )
    if include_temperature_unit:
        add_temperature_unit_argument(
            parser,
            help_text=(
                fahrenheit_help
                if fahrenheit_help is not None
                else "Display temperatures in degrees Fahrenheit instead of Celsius"
            ),
            options=fahrenheit_options,
        )
    add_headless_argument(parser)
    add_interpolation_arguments(parser)
