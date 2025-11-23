"""CLI entry point for running and visualizing the climate simulator."""

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

from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig
from climate_sim.plotting import (
    plot_layered_monthly_temperature_cycle,
    save_monthly_temperature_gif,
)
from climate_sim.data.calendar import MONTH_NAMES
from climate_sim.runtime.cli import add_common_model_arguments
from climate_sim.physics.atmosphere import (
    log_law_map_wind_speed,
)
from climate_sim.data.constants import R_EARTH_METERS
from climate_sim.data.elevation import (
    compute_cell_roughness_length,
)
from climate_sim.physics.pressure import compute_pressure
from climate_sim.data.landmask import compute_land_mask
from climate_sim.core.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.core.solver import compute_periodic_cycle_results
from climate_sim.core.units import convert_temperature, temperature_unit
from climate_sim.physics.humidity import compute_cloud_cover, specific_humidity_to_relative_humidity

from dotenv import load_dotenv
load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the climate model and plot the cycle.")
    default_atmosphere = RadiationConfig().include_atmosphere
    add_common_model_arguments(
        parser,
        default_atmosphere=default_atmosphere,
        fahrenheit_help="Display temperatures in degrees Fahrenheit instead of Celsius",
    )
    return parser.parse_args()

def _print_mean(
    label: str,
    field: np.ndarray | None,
    weights: np.ndarray,
    fallback: str,
) -> str:
    if field is None:
        return fallback
    return f"{area_weighted_mean(field, weights):.2f}"

def main() -> None:
    args = _parse_args()

    start = time.time()
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
        latent_heat_config=latent_heat_config,
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

    month_names = list(MONTH_NAMES)

    layer_cycles: dict[str, np.ndarray] = {"Surface": surface_cycle}
    if atmosphere_cycle is not None:
        layer_cycles["Atmosphere"] = atmosphere_cycle

        atmosphere_2m_cycle = layers.get("temperature_2m")
        if atmosphere_2m_cycle is not None:
            layer_cycles["Atmosphere (2 m)"] = atmosphere_2m_cycle

    wind_u = layers.get("wind_u")
    wind_v = layers.get("wind_v")
    wind_speed = layers.get("wind_speed")

    Tatm_field = layers.get("Tatm")
    Tatm_cycle_K: np.ndarray | None = None
    humidity_q_cycle = layers.get("humidity")
    if Tatm_field is not None:
        Tatm_arr = np.asarray(Tatm_field, dtype=float)
        if Tatm_arr.ndim == 3:
            mean_Tatm = float(np.nanmean(Tatm_arr))
            if mean_Tatm < 200.0:
                Tatm_arr = Tatm_arr + 273.15
            Tatm_cycle_K = Tatm_arr
    elif atmosphere_cycle is not None:
        Tatm_cycle_K = atmosphere_cycle + 273.15

    slp_cycle_hpa: np.ndarray | None = None
    if Tatm_cycle_K is not None:
        pressure_monthly = np.empty_like(Tatm_cycle_K, dtype=float)
        for idx in range(Tatm_cycle_K.shape[0]):
            pressure_monthly[idx] = compute_pressure(
                Tatm_cycle_K[idx],
            )
        slp_cycle_hpa = pressure_monthly * 0.01

    wind_speed_10 = None
    if wind_u is not None and wind_v is not None and wind_speed is not None:
        roughness_length = compute_cell_roughness_length(
            lon2d,
            lat2d,
            land_mask=land_mask_bool,
        )
        height_ref_field = np.where(land_mask_bool, 400.0, 1000.0)
        height_target_m = sensible_heat_config.reference_height_surface_m
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

        slp_sorted = None
        if slp_cycle_hpa is not None:
            slp_sorted = slp_cycle_hpa[:, :, lon_sort_idx]

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

        pressure_container: dict[str, object | None] = {"artist": None, "data": None}
        if slp_sorted is not None:
            slp_min = float(np.nanmin(slp_sorted))
            slp_max = float(np.nanmax(slp_sorted))
            if not np.isfinite(slp_min) or not np.isfinite(slp_max):
                slp_min, slp_max = 980.0, 1030.0
            elif slp_max - slp_min < 1.0e-3:
                slp_min -= 1.0
                slp_max += 1.0

            slp_norm = Normalize(vmin=slp_min, vmax=slp_max)
            pressure_extent = (
                float(lon_sorted[0]),
                float(lon_sorted[-1]),
                float(np.min(lat2d[:, 0])),
                float(np.max(lat2d[:, 0])),
            )
            pressure_artist = ax_wind.imshow(
                slp_sorted[0],
                extent=pressure_extent,
                origin="lower",
                transform=projection,
                cmap=plt.cm.cividis,
                norm=slp_norm,
                alpha=0.65,
                zorder=0.1,
                interpolation="nearest",
            )
            pressure_container["artist"] = pressure_artist
            pressure_container["data"] = slp_sorted

            pressure_cax = fig_wind.add_axes([0.2, 0.02, 0.6, 0.02])
            pressure_cbar = fig_wind.colorbar(
                pressure_artist,
                cax=pressure_cax,
                orientation="horizontal",
            )
            pressure_cbar.set_label("Sea-level pressure (hPa)")

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
            pressure_artist = pressure_container.get("artist")
            slp_data = pressure_container.get("data")
            if pressure_artist is not None and slp_data is not None:
                pressure_artist.set_data(slp_data[idx])
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

    # Humidity plot
    if humidity_q_cycle is not None:
        # Use stored humidity from solver state
        # Compute relative humidity from stored specific humidity
        # Use surface temperature (matching solver's _select_humidity_temperature behavior)
        temp_for_humidity = surface_cycle + 273.15  # Convert to Kelvin
        
        # Compute RH from q: rh = q / q_sat
        # where q_sat is computed from temperature and pressure
        monthly_humidity_rh = []
        for month_idx in range(12):
            temp_K = temp_for_humidity[month_idx]
            q = humidity_q_cycle[month_idx]
            rh = specific_humidity_to_relative_humidity(q, temp_K)
            monthly_humidity_rh.append(rh)
        
        humidity_rh_cycle = np.stack(monthly_humidity_rh, axis=0)
        
        # Compute cloud cover for each month from relative humidity
        monthly_cloud_cover = []
        for month_idx in range(12):
            rh = humidity_rh_cycle[month_idx]
            cloud_cover = compute_cloud_cover(
                relative_humidity=rh,
                land_mask=land_mask_bool,
            )
            monthly_cloud_cover.append(cloud_cover)
        
        cloud_cover_cycle = np.stack(monthly_cloud_cover, axis=0)
        
        # Setup humidity plot
        projection = ccrs.PlateCarree()
        fig_humidity, ax_humidity = plt.subplots(
            figsize=(12, 6), subplot_kw=dict(projection=projection)
        )
        ax_humidity.set_global()
        ax_humidity.coastlines(linewidth=0.4)
        ax_humidity.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
        ax_humidity.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "lakes", "110m", edgecolor="#000000", facecolor="none"
            ),
            linewidth=0.2,
        )
        ax_humidity.add_feature(
            cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0
        )
        
        # Wrap longitude for proper display
        lon_full = lon2d[0]
        lon_wrapped = ((lon_full + 180.0) % 360.0) - 180.0
        lon_sort_idx = np.argsort(lon_wrapped)
        lon_sorted = lon_wrapped[lon_sort_idx]
        
        humidity_q_sorted = humidity_q_cycle[:, :, lon_sort_idx]
        humidity_rh_sorted = humidity_rh_cycle[:, :, lon_sort_idx]
        cloud_cover_sorted = cloud_cover_cycle[:, :, lon_sort_idx]
        
        humidity_data = {
            "Specific Humidity (q)": humidity_q_sorted,
            "Relative Humidity (RH)": humidity_rh_sorted,
            "Cloud Cover": cloud_cover_sorted,
        }
        
        # Initial state
        current_state_humidity = {"month": 0, "type": "Specific Humidity (q)"}
        
        def get_norm_and_label(humidity_type: str) -> tuple[Normalize, str]:
            if humidity_type == "Specific Humidity (q)":
                vmin = float(humidity_q_cycle.min())
                vmax = float(humidity_q_cycle.max())
                label = "Specific Humidity (kg/kg)"
            elif humidity_type == "Relative Humidity (RH)":
                vmin = float(humidity_rh_cycle.min())
                vmax = float(humidity_rh_cycle.max())
                label = "Relative Humidity"
            else:  # Cloud Cover
                vmin = float(cloud_cover_cycle.min())
                vmax = float(cloud_cover_cycle.max())
                label = "Cloud Cover Fraction"
            return Normalize(vmin=vmin, vmax=vmax), label
        
        norm_humidity, colorbar_label_humidity = get_norm_and_label(current_state_humidity["type"])
        cmap_humidity = plt.cm.viridis
        
        # Create sorted lat/lon grids for pcolormesh
        lat_sorted = lat2d[:, 0]
        lon_sorted_2d, lat_sorted_2d = np.meshgrid(lon_sorted, lat_sorted)
        
        humidity_mesh = ax_humidity.pcolormesh(
            lon_sorted_2d,
            lat_sorted_2d,
            humidity_data[current_state_humidity["type"]][current_state_humidity["month"]],
            cmap=cmap_humidity,
            norm=norm_humidity,
            shading="auto",
            transform=projection,
        )
        
        ax_humidity.set_title(
            f"{current_state_humidity['type']} – {month_names[current_state_humidity['month']]}"
        )
        
        # Colorbar
        cbar_humidity = fig_humidity.colorbar(
            humidity_mesh,
            ax=ax_humidity,
            orientation="vertical",
            pad=0.04,
            fraction=0.046,
        )
        cbar_humidity.set_label(colorbar_label_humidity)
        
        def _update_humidity_plot() -> None:
            data = humidity_data[current_state_humidity["type"]][current_state_humidity["month"]]
            norm_humidity, colorbar_label_humidity = get_norm_and_label(current_state_humidity["type"])
            humidity_mesh.set_norm(norm_humidity)
            humidity_mesh.set_array(data.ravel())
            cbar_humidity.set_label(colorbar_label_humidity)
            cbar_humidity.update_normal(humidity_mesh)
            ax_humidity.set_title(
                f"{current_state_humidity['type']} – {month_names[current_state_humidity['month']]}"
            )
            fig_humidity.canvas.draw_idle()
        
        slider_humidity_ax = fig_humidity.add_axes([0.2, 0.08, 0.6, 0.03])
        month_slider_humidity = Slider(
            slider_humidity_ax,
            label="Month",
            valmin=0,
            valmax=11,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )
        
        def _on_humidity_month_change(val: float) -> None:
            current_state_humidity["month"] = int(val)
            _update_humidity_plot()
        
        month_slider_humidity.on_changed(_on_humidity_month_change)
        
        radio_humidity_ax = fig_humidity.add_axes([0.02, 0.55, 0.12, 0.18])
        radio_humidity_ax.set_title("Variable", fontsize=9)
        type_names_humidity = list(humidity_data.keys())
        type_selector_humidity = RadioButtons(radio_humidity_ax, type_names_humidity, active=0)
        
        def _on_humidity_type_change(label: str) -> None:
            current_state_humidity["type"] = label
            _update_humidity_plot()
        
        type_selector_humidity.on_clicked(_on_humidity_type_change)
        _update_humidity_plot()

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
