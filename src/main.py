import argparse
import os
import time

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.widgets import RadioButtons, Slider
from pathlib import Path

from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
from climate_sim.plotting import (
    plot_layered_monthly_temperature_cycle,
    save_monthly_temperature_gif,
)
from climate_sim.utils.atmosphere import (
    adjust_temperature_by_elevation,
    log_law_map_wind_speed,
)
from climate_sim.utils.constants import R_EARTH_METERS
from climate_sim.utils.elevation import compute_cell_roughness_length
from climate_sim.utils.landmask import compute_land_mask
from climate_sim.utils.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.utils.solver import compute_periodic_cycle_results
from climate_sim.utils.temperature import convert_temperature, temperature_unit

from dotenv import load_dotenv
load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the climate model and plot the cycle.")
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
        "--latent-heat",
        dest="latent_heat",
        action="store_true",
        default=True,
        help="Include latent heat of fusion in the surface heat capacity (default)",
    )
    parser.add_argument(
        "--no-latent-heat",
        dest="latent_heat",
        action="store_false",
        help="Disable the latent heat of fusion adjustment",
    )
    parser.add_argument(
        "--bulk-exchange",
        dest="bulk_exchange",
        action="store_true",
        default=True,
        help="Enable the neutral bulk sensible heat exchange model (default)",
    )
    parser.add_argument(
        "--no-bulk-exchange",
        dest="bulk_exchange",
        action="store_false",
        help="Disable the neutral bulk sensible heat exchange model",
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
    parser.add_argument(
        "--fahrenheit", "-f",
        dest="fahrenheit",
        action="store_true",
        help="Display temperatures in degrees Fahrenheit instead of Celsius",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    start = time.time()
    radiation_config = RadiationConfig(include_atmosphere=args.atmosphere)
    diffusion_config = DiffusionConfig(enabled=args.diffusion)
    snow_config = SnowAlbedoConfig(
        enabled=args.snow,
        latent_heat_enabled=args.latent_heat,
    )
    sensible_heat_config = SensibleHeatExchangeConfig(enabled=args.bulk_exchange)
    print(f"Configuration setup took {time.time() - start:.2f} seconds")

    start = time.time()
    lon2d, lat2d, layers = compute_periodic_cycle_results(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.elliptical_orbit,
        radiation_config=radiation_config,
        diffusion_config=diffusion_config,
        snow_config=snow_config,
        sensible_heat_config=sensible_heat_config,
        return_layer_map=True,
    )
    print(f"Model run took {time.time() - start:.2f} seconds")
    assert type(layers) is dict
    surface_cycle = layers["surface"]
    albedo_field = layers.get("albedo")

    cell_areas = spherical_cell_area(
        lon2d, lat2d, earth_radius_m=diffusion_config.earth_radius_m
    )
    surface_area_mean = float(
        area_weighted_mean(surface_cycle.mean(axis=0), cell_areas)
    )
    albedo_area_mean: float | None = None
    if albedo_field is not None:
        albedo_area_mean = float(area_weighted_mean(albedo_field, cell_areas))

    unit = temperature_unit(args.fahrenheit)
    summary_parts = [
        "Surface layer:",
        f"Tmin={convert_temperature(surface_cycle.min(), args.fahrenheit):.1f}{unit}",
        f"Tmax={convert_temperature(surface_cycle.max(), args.fahrenheit):.1f}{unit}",
        f"area-weighted mean={convert_temperature(surface_area_mean, args.fahrenheit):.1f}{unit}",
    ]
    if albedo_area_mean is not None:
        summary_parts.append(f"area-weighted mean albedo={albedo_area_mean:.3f}")

    print(" ".join(summary_parts))

    atmosphere_cycle = layers.get("atmosphere")
    if atmosphere_cycle is not None:
        atmosphere_area_mean = float(
            area_weighted_mean(atmosphere_cycle.mean(axis=0), cell_areas)
        )
        print(
            "Atmosphere layer: "
            f"Tmin={convert_temperature(atmosphere_cycle.min(), args.fahrenheit):.1f}{unit}, "
            f"Tmax={convert_temperature(atmosphere_cycle.max(), args.fahrenheit):.1f}{unit}, "
            f"area-weighted mean={convert_temperature(atmosphere_area_mean, args.fahrenheit):.1f}{unit}"
        )

    land_mask_bool = compute_land_mask(lon2d, lat2d)
    land_mask = land_mask_bool.astype(float)
    ocean_mask = 1.0 - land_mask
    land_weights = cell_areas * land_mask
    ocean_weights = cell_areas * ocean_mask

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    layer_cycles: dict[str, np.ndarray] = {"Surface": surface_cycle}
    if atmosphere_cycle is not None:
        layer_cycles["Atmosphere"] = atmosphere_cycle

        atmosphere_height = 5000  # height of effective emission layer in meters
        delta_to_two_m = 2.0 - atmosphere_height
        atmosphere_2m_cycle = adjust_temperature_by_elevation(
            atmosphere_cycle, delta_to_two_m
        )
        layer_cycles["Atmosphere (2 m)"] = atmosphere_2m_cycle

    wind_u = layers.get("wind_u")
    wind_v = layers.get("wind_v")
    wind_speed = layers.get("wind_speed")

    wind_speed_10 = None
    if wind_u is not None and wind_v is not None and wind_speed is not None:
        roughness_length = compute_cell_roughness_length(
            lon2d,
            lat2d,
            land_mask=land_mask_bool,
        )
        height_ref_land_m = sensible_heat_config.land_reference_height_m
        height_ref_ocean_m = sensible_heat_config.ocean_reference_height_m
        height_target_m = sensible_heat_config.reference_height_surface_m
        height_ref_field = np.where(
            land_mask_bool,
            height_ref_land_m,
            height_ref_ocean_m,
        )
        wind_speed_10 = log_law_map_wind_speed(
            wind_speed,
            height_ref_m=height_ref_field,
            height_target_m=height_target_m,
            roughness_length_m=roughness_length,
        )
        scale_factor = np.zeros_like(wind_speed)
        mask_nonzero = wind_speed > 1.0e-6
        scale_factor[mask_nonzero] = wind_speed_10[mask_nonzero] / wind_speed[mask_nonzero]
        wind_u_10 = wind_u * scale_factor
        wind_v_10 = wind_v * scale_factor

        projection = ccrs.PlateCarree()
        fig_wind, ax_wind = plt.subplots(
            figsize=(12, 6), subplot_kw=dict(projection=projection)
        )
        ax_wind.set_global()
        ax_wind.coastlines(linewidth=0.4)
        ax_wind.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
        ax_wind.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "lakes", "110m", edgecolor="#000000", facecolor="none"
            ),
            linewidth=0.2,
        )
        ax_wind.add_feature(
            cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0
        )

        lon_full = lon2d[0]
        lon_wrapped = ((lon_full + 180.0) % 360.0) - 180.0
        lon_sort_idx = np.argsort(lon_wrapped)
        lon_sorted = lon_wrapped[lon_sort_idx]

        wind_u_sorted = wind_u[:, :, lon_sort_idx]
        wind_v_sorted = wind_v[:, :, lon_sort_idx]
        wind_speed_sorted = wind_speed[:, :, lon_sort_idx]
        wind_u_10_sorted = wind_u_10[:, :, lon_sort_idx]
        wind_v_10_sorted = wind_v_10[:, :, lon_sort_idx]
        wind_speed_10_sorted = wind_speed_10[:, :, lon_sort_idx]

        max_speed = float(np.max(np.maximum(wind_speed_sorted, wind_speed_10_sorted)))
        if not np.isfinite(max_speed) or max_speed <= 0.0:
            max_speed = 1.0

        stride = max(1, int(round(1.0 / args.resolution)))
        lat_coords = lat2d[::stride, 0]
        lon_coords = lon_sorted[::stride]

        meters_per_deg_lat = np.pi / 180.0 * R_EARTH_METERS
        cosphi = np.cos(np.deg2rad(lat_coords))
        meters_per_deg_lon_vec = meters_per_deg_lat * np.clip(cosphi, 1e-6, None)

        def _to_deg_per_sec(
            u_slice: np.ndarray, v_slice: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            u_deg = u_slice / meters_per_deg_lon_vec[:, None]
            v_deg = v_slice / meters_per_deg_lat
            return u_deg, v_deg

        cmap = plt.cm.viridis
        norm = Normalize(vmin=0.0, vmax=max_speed)

        wind_levels = {
            "100 m": {
                "u": wind_u_sorted,
                "v": wind_v_sorted,
                "speed": wind_speed_sorted,
            },
            "10 m": {
                "u": wind_u_10_sorted,
                "v": wind_v_10_sorted,
                "speed": wind_speed_10_sorted,
            },
        }

        level_names = list(wind_levels.keys())
        current_state = {"level": level_names[0], "month": 0}
        stream_container: dict[str, object | None] = {"obj": None}
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig_wind.colorbar(
            scalar_mappable,
            ax=ax_wind,
            orientation="vertical",
            pad=0.04,
            fraction=0.046,
        )
        cbar.set_label("Wind speed (m/s)")

        def _clear_streamplot(stream_set) -> None:
            if stream_set is None:
                return
            stream_set.lines.set_segments([])
            stream_set.lines.set_array(np.array([]))
            stream_set.lines.set_visible(False)
            for art in list(ax_wind.get_children()):
                if isinstance(art, mpatches.FancyArrowPatch):
                    art.remove()

        def _draw_streamplot() -> None:
            level_data = wind_levels[current_state["level"]]
            idx = current_state["month"]
            u_slice = level_data["u"][idx, ::stride, ::stride]
            v_slice = level_data["v"][idx, ::stride, ::stride]
            speed_slice = level_data["speed"][idx, ::stride, ::stride]
            u_deg_slice, v_deg_slice = _to_deg_per_sec(u_slice, v_slice)

            current_stream = stream_container["obj"]
            _clear_streamplot(current_stream)

            new_stream = ax_wind.streamplot(
                lon_coords,
                lat_coords,
                u_deg_slice,
                v_deg_slice,
                color=speed_slice,
                cmap=cmap,
                norm=norm,
                transform=projection,
                density=1.8,
                linewidth=1.2,
                arrowsize=1.4,
            )

            stream_container["obj"] = new_stream
            ax_wind.set_title(
                f"Geostrophic Wind ({current_state['level']}) – {month_names[idx]}"
            )
            fig_wind.canvas.draw_idle()

        slider_ax = fig_wind.add_axes([0.2, 0.08, 0.6, 0.03])
        month_slider = Slider(
            slider_ax,
            label="Month",
            valmin=0,
            valmax=11,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )

        def _on_month_change(val: float) -> None:
            current_state["month"] = int(val)
            _draw_streamplot()

        month_slider.on_changed(_on_month_change)

        radio_ax = fig_wind.add_axes([0.02, 0.55, 0.12, 0.18])
        radio_ax.set_title("Wind Level", fontsize=9)
        level_selector = RadioButtons(radio_ax, level_names, active=0)

        def _on_level_change(label: str) -> None:
            current_state["level"] = label
            _draw_streamplot()

        level_selector.on_clicked(_on_level_change)
        _draw_streamplot()

    atmosphere_2m_cycle = layer_cycles.get("Atmosphere (2 m)")

    data_dir_value = os.getenv("DATA_DIR")
    if not data_dir_value:
        raise ValueError("DATA_DIR environment variable must be set to save GIFs.")
    data_dir = Path(data_dir_value).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    def _save_cycle(name: str, field: np.ndarray, filename: str) -> None:
        output_path = data_dir / filename
        save_monthly_temperature_gif(
            lon2d,
            lat2d,
            field,
            output_path=output_path,
            title=f"{name} Temperature ({unit})",
            colorbar_label=f"Temperature ({unit})",
            use_fahrenheit=args.fahrenheit,
        )
        print(f"Saved {name.lower()} temperature animation to {output_path}")

    _save_cycle("Surface", surface_cycle, "surface_temperature_cycle.gif")
    if atmosphere_2m_cycle is not None:
        _save_cycle("Two-meter", atmosphere_2m_cycle, "two_meter_temperature_cycle.gif")
    if atmosphere_cycle is not None:
        _save_cycle("Atmosphere", atmosphere_cycle, "atmosphere_temperature_cycle.gif")

    def _print_mean(
        label: str,
        field: np.ndarray | None,
        weights: np.ndarray,
        fallback: str,
    ) -> str:
        if field is None:
            return fallback
        return f"{area_weighted_mean(field, weights):.2f}"

    for idx in range(12):
        print(f"{month_names[idx]} statistics:")
        surface_land = area_weighted_mean(surface_cycle[idx], land_weights)
        surface_ocean = area_weighted_mean(surface_cycle[idx], ocean_weights)
        print(f"  Surface mean (land/ocean) [°C]: {surface_land:.2f} / {surface_ocean:.2f}")

        two_m_land = None
        two_m_ocean = None
        if atmosphere_2m_cycle is not None:
            two_m_land = area_weighted_mean(atmosphere_2m_cycle[idx], land_weights)
            two_m_ocean = area_weighted_mean(atmosphere_2m_cycle[idx], ocean_weights)
            print(f"  2 m mean (land/ocean) [°C]: {two_m_land:.2f} / {two_m_ocean:.2f}")
        else:
            print("  2 m mean (land/ocean) [°C]: N/A (no atmosphere layer)")

        if atmosphere_cycle is not None:
            atm_land = area_weighted_mean(atmosphere_cycle[idx], land_weights)
            atm_ocean = area_weighted_mean(atmosphere_cycle[idx], ocean_weights)
            print(f"  Atmosphere mean (land/ocean) [°C]: {atm_land:.2f} / {atm_ocean:.2f}")
        else:
            print("  Atmosphere mean (land/ocean) [°C]: N/A (no atmosphere layer)")

        if wind_speed is not None:
            wind_land = area_weighted_mean(wind_speed[idx], land_weights)
            wind_ocean = area_weighted_mean(wind_speed[idx], ocean_weights)
            print(f"  Wind speed mean (land/ocean) [m/s]: {wind_land:.2f} / {wind_ocean:.2f}")
        else:
            reason = (
                "bulk exchange disabled"
                if not sensible_heat_config.enabled
                else "winds unavailable"
            )
            print(f"  Wind speed mean (land/ocean) [m/s]: N/A ({reason})")

        if wind_speed_10 is not None:
            wind10_land = area_weighted_mean(wind_speed_10[idx], land_weights)
            wind10_ocean = area_weighted_mean(wind_speed_10[idx], ocean_weights)
            print(f"  Wind speed 10 m mean (land/ocean) [m/s]: {wind10_land:.2f} / {wind10_ocean:.2f}")
        else:
            reason = (
                "bulk exchange disabled"
                if not sensible_heat_config.enabled
                else "winds unavailable"
            )
            print(f"  Wind speed 10 m mean (land/ocean) [m/s]: N/A ({reason})")

    print("Annual statistics:")
    surface_land_annual = area_weighted_mean(surface_cycle.mean(axis=0), land_weights)
    surface_ocean_annual = area_weighted_mean(surface_cycle.mean(axis=0), ocean_weights)
    print(f"  Surface mean (land/ocean) [°C]: {surface_land_annual:.2f} / {surface_ocean_annual:.2f}")

    if atmosphere_2m_cycle is not None:
        two_m_land_annual = area_weighted_mean(atmosphere_2m_cycle.mean(axis=0), land_weights)
        two_m_ocean_annual = area_weighted_mean(atmosphere_2m_cycle.mean(axis=0), ocean_weights)
        print(f"  2 m mean (land/ocean) [°C]: {two_m_land_annual:.2f} / {two_m_ocean_annual:.2f}")
    else:
        print("  2 m mean (land/ocean) [°C]: N/A (no atmosphere layer)")

    if atmosphere_cycle is not None:
        atm_land_annual = area_weighted_mean(atmosphere_cycle.mean(axis=0), land_weights)
        atm_ocean_annual = area_weighted_mean(atmosphere_cycle.mean(axis=0), ocean_weights)
        print(f"  Atmosphere mean (land/ocean) [°C]: {atm_land_annual:.2f} / {atm_ocean_annual:.2f}")
    else:
        print("  Atmosphere mean (land/ocean) [°C]: N/A (no atmosphere layer)")

    if wind_speed is not None:
        wind_land_annual = area_weighted_mean(wind_speed.mean(axis=0), land_weights)
        wind_ocean_annual = area_weighted_mean(wind_speed.mean(axis=0), ocean_weights)
        print(f"  Wind speed mean (land/ocean) [m/s]: {wind_land_annual:.2f} / {wind_ocean_annual:.2f}")
    else:
        reason = (
            "bulk exchange disabled" if not sensible_heat_config.enabled else "winds unavailable"
        )
        print(f"  Wind speed mean (land/ocean) [m/s]: N/A ({reason})")

    if wind_speed_10 is not None:
        wind10_land_annual = area_weighted_mean(wind_speed_10.mean(axis=0), land_weights)
        wind10_ocean_annual = area_weighted_mean(wind_speed_10.mean(axis=0), ocean_weights)
        print(f"  Wind speed 10 m mean (land/ocean) [m/s]: {wind10_land_annual:.2f} / {wind10_ocean_annual:.2f}")
    else:
        reason = (
            "bulk exchange disabled" if not sensible_heat_config.enabled else "winds unavailable"
        )
        print(f"  Wind speed 10 m mean (land/ocean) [m/s]: N/A ({reason})")

    plot_layered_monthly_temperature_cycle(
        lon2d,
        lat2d,
        layer_cycles,
        title=f"Temperature Cycle ({unit})",
        use_fahrenheit=args.fahrenheit,
    )


if __name__ == "__main__":
    main()
