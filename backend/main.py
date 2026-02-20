# type: ignore[attr-defined]
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
import cmocean
from pathlib import Path

from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig
from climate_sim.physics.vertical_motion import VerticalMotionConfig
from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.plotting import (
    plot_layered_monthly_temperature_cycle,
    save_monthly_temperature_gif,
    add_dynamic_status_readout,
    _format_lat,
    _format_lon,
)
from climate_sim.data.calendar import MONTH_NAMES
from climate_sim.runtime.cli import add_common_model_arguments
from climate_sim.runtime.config import ModelConfig
from climate_sim.physics.atmosphere.advection import AdvectionConfig
from climate_sim.physics.atmosphere.wind import WindConfig
from climate_sim.physics.ocean_currents import OceanAdvectionConfig
from climate_sim.physics.orographic_effects import OrographicConfig
from climate_sim.data.constants import R_EARTH_METERS
from climate_sim.physics.atmosphere.pressure import compute_pressure
from climate_sim.physics.atmosphere.hadley import compute_itcz_latitude
from climate_sim.data.landmask import compute_land_mask
from climate_sim.core.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.core.solver import solve_periodic_climate
from climate_sim.core.units import convert_temperature, temperature_unit
from climate_sim.export.temperature_interpolation import interpolate_layer_map
from climate_sim.core.timing import time_block, get_profiler
from climate_sim.physics.humidity import specific_humidity_to_relative_humidity
from climate_sim.physics.radiation import compute_cloud_coverage, _compute_pressure_anomaly

from dotenv import load_dotenv
load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the climate model and plot the cycle.")
    default_atmosphere = RadiationConfig().include_atmosphere
    parser.add_argument(
        "--cache",
        dest="cache",
        action="store_true",
        default=False,
        help="load from cache",
    )
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
    advection_config = AdvectionConfig(enabled=args.advection)
    wind_config = WindConfig()
    ocean_advection_config = OceanAdvectionConfig(enabled=args.ocean_advection)
    vertical_motion_config = VerticalMotionConfig(enabled=args.vertical_motion)
    orographic_config = OrographicConfig(enabled=args.orographic)

    model_config = ModelConfig(
        radiation=radiation_config,
        diffusion=diffusion_config,
        wind=wind_config,
        advection=advection_config,
        snow=snow_config,
        sensible_heat=sensible_heat_config,
        latent_heat=latent_heat_config,
        ocean_advection=ocean_advection_config,
        vertical_motion=vertical_motion_config,
        orographic=orographic_config,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.elliptical_orbit,
    )
    print(f"Configuration setup took {time.time() - start:.2f} seconds")

    start = time.time()
    data_dir = os.getenv("DATA_DIR")
    assert data_dir is not None, "Please set the DATA_DIR environment variable to enable caching."
    data_dir = Path(data_dir)
    cache_path = data_dir / "main.npz"
    if args.cache:
        lon2d, lat2d = create_lat_lon_grid(args.resolution)
        with np.load(cache_path) as cached:
            layers = {k: cached[k] for k in cached}
    else:
        lon2d, lat2d, layers = solve_periodic_climate(
            resolution_deg=args.resolution,
            model_config=model_config,
            return_layer_map=True,
        )
        np.savez_compressed(cache_path, **layers)
    print(f"Model run took {time.time() - start:.2f} seconds")
    assert type(layers) is dict
    surface_cycle = layers["surface"]
    albedo_field = layers.get("albedo")

    cell_areas = spherical_cell_area(
        lon2d, lat2d, earth_radius_m=R_EARTH_METERS
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

    # Wind fields at different levels:
    # - geostrophic: free atmosphere, no drag
    # - ekman (wind_u/v/speed): boundary layer with drag
    # - 10m: ekman wind scaled to 10m via log law
    wind_u_geo = layers.get("wind_u_geostrophic")
    wind_v_geo = layers.get("wind_v_geostrophic")
    wind_speed_geo = layers.get("wind_speed_geostrophic")
    wind_u = layers.get("wind_u")
    wind_v = layers.get("wind_v")
    wind_speed = layers.get("wind_speed")
    wind_u_10m = layers.get("wind_u_10m")
    wind_v_10m = layers.get("wind_v_10m")
    wind_speed_10m = layers.get("wind_speed_10m")

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

    # Get boundary layer temperature if available (3-layer model)
    boundary_layer_cycle = layers.get("boundary_layer")

    # Compute ITCZ from boundary layer temp
    # Using boundary layer avoids cold high-elevation surfaces (e.g., Tibet) biasing ITCZ
    slp_cycle_hpa: np.ndarray | None = None
    if Tatm_cycle_K is not None:
        cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=R_EARTH_METERS)
        pressure_monthly = np.empty_like(Tatm_cycle_K, dtype=float)
        for idx in range(Tatm_cycle_K.shape[0]):
            # Use boundary layer temp if available, otherwise surface temp
            if boundary_layer_cycle is not None:
                itcz_temp_K = boundary_layer_cycle[idx] + 273.15
            else:
                itcz_temp_K = surface_cycle[idx] + 273.15
            itcz_rad = compute_itcz_latitude(itcz_temp_K, lat2d, cell_areas)
            pressure_monthly[idx] = compute_pressure(
                Tatm_cycle_K[idx],
                itcz_rad=itcz_rad,
            )
        slp_cycle_hpa = pressure_monthly * 0.01

    if wind_u is not None and wind_v is not None and wind_speed is not None and not args.headless:
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

            # Ekman wind (boundary layer with drag)
            wind_u_sorted = wind_u[:, :, lon_sort_idx]
            wind_v_sorted = wind_v[:, :, lon_sort_idx]
            wind_speed_sorted = wind_speed[:, :, lon_sort_idx]

            # 10m wind (from Ekman via log law)
            wind_u_10m_sorted = wind_u_10m[:, :, lon_sort_idx] if wind_u_10m is not None else None
            wind_v_10m_sorted = wind_v_10m[:, :, lon_sort_idx] if wind_v_10m is not None else None
            wind_speed_10m_sorted = wind_speed_10m[:, :, lon_sort_idx] if wind_speed_10m is not None else None

            # Geostrophic wind (free atmosphere, no drag)
            wind_u_geo_sorted = wind_u_geo[:, :, lon_sort_idx] if wind_u_geo is not None else None
            wind_v_geo_sorted = wind_v_geo[:, :, lon_sort_idx] if wind_v_geo is not None else None
            wind_speed_geo_sorted = wind_speed_geo[:, :, lon_sort_idx] if wind_speed_geo is not None else None

            slp_sorted = None
            if slp_cycle_hpa is not None:
                slp_sorted = slp_cycle_hpa[:, :, lon_sort_idx]

            # Compute max speed across all available wind levels
            all_speeds = [wind_speed_sorted]
            if wind_speed_10m_sorted is not None:
                all_speeds.append(wind_speed_10m_sorted)
            if wind_speed_geo_sorted is not None:
                all_speeds.append(wind_speed_geo_sorted)

            max_speed = float(np.max([np.max(s) for s in all_speeds]))
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

            cmap = cmocean.cm.speed
            norm = Normalize(vmin=0.0, vmax=max_speed)

            # Build wind levels dict with available fields
            wind_levels = {}
            if wind_u_geo_sorted is not None:
                wind_levels["Geostrophic"] = {
                    "u": wind_u_geo_sorted,
                    "v": wind_v_geo_sorted,
                    "speed": wind_speed_geo_sorted,
                }
            wind_levels["Ekman (BL)"] = {
                "u": wind_u_sorted,
                "v": wind_v_sorted,
                "speed": wind_speed_sorted,
            }
            if wind_u_10m_sorted is not None:
                wind_levels["10 m"] = {
                    "u": wind_u_10m_sorted,
                    "v": wind_v_10m_sorted,
                    "speed": wind_speed_10m_sorted,
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
                    cmap=plt.cm.magma,
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
                # Store current wind data for hover
                stream_container["u_data"] = u_slice
                stream_container["v_data"] = v_slice
                stream_container["speed_data"] = speed_slice
                stream_container["lon_coords"] = lon_coords
                stream_container["lat_coords"] = lat_coords

                pressure_artist = pressure_container.get("artist")
                slp_data = pressure_container.get("data")
                if pressure_artist is not None and slp_data is not None:
                    pressure_artist.set_data(slp_data[idx])
                ax_wind.set_title(
                    f"Wind ({current_state['level']}) – {month_names[idx]}"
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

            # Add hover functionality using helper
            add_dynamic_status_readout(
                fig=fig_wind,
                ax=ax_wind,
                lon_coords=stream_container.get("lon_coords", lon_coords),
                lat_coords=stream_container.get("lat_coords", lat_coords),
                data_container=stream_container,
                format_message=lambda lon, lat, data, lon_idx, lat_idx: (
                    f"{_format_lat(lat)}  {_format_lon(lon)}  "
                    f"Speed: {data.get('speed_data', [[0]])[lat_idx, lon_idx]:.1f} m/s  "
                    f"U: {data.get('u_data', [[0]])[lat_idx, lon_idx]:.1f} m/s  "
                    f"V: {data.get('v_data', [[0]])[lat_idx, lon_idx]:.1f} m/s"
                ) if data.get("u_data") is not None else ""
            )

            _draw_streamplot()

    # Humidity plot
    if humidity_q_cycle is not None and not args.headless:
            # Use stored humidity from solver state
            # Compute relative humidity from stored specific humidity
            # Use boundary layer temperature if available, fall back to surface
            if boundary_layer_cycle is not None:
                temp_for_humidity = boundary_layer_cycle + 273.15  # Convert to Kelvin
            else:
                temp_for_humidity = surface_cycle + 273.15  # Convert to Kelvin

            # Compute RH from q: rh = q / q_sat
            # where q_sat is computed from temperature and pressure
            monthly_humidity_rh = []
            for month_idx in range(12):
                temp_K = temp_for_humidity[month_idx]
                q = humidity_q_cycle[month_idx]
                if boundary_layer_cycle is not None:
                    itcz_temp_K = boundary_layer_cycle[month_idx] + 273.15
                else:
                    itcz_temp_K = temp_K
                itcz_rad = compute_itcz_latitude(itcz_temp_K, lat2d, cell_areas)
                rh = specific_humidity_to_relative_humidity(q, temp_K, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)
                monthly_humidity_rh.append(rh)

            humidity_rh_cycle = np.stack(monthly_humidity_rh, axis=0)

            # Compute cloud cover for each month from relative humidity and pressure
            monthly_cloud_cover = []
            for month_idx in range(12):
                rh = humidity_rh_cycle[month_idx]
                if boundary_layer_cycle is not None:
                    itcz_temp_K = boundary_layer_cycle[month_idx] + 273.15
                else:
                    itcz_temp_K = surface_cycle[month_idx] + 273.15
                itcz_rad = compute_itcz_latitude(itcz_temp_K, lat2d, cell_areas)
                dp_norm = _compute_pressure_anomaly(itcz_temp_K, itcz_rad, lat2d=lat2d, lon2d=lon2d)
                cloud_cover = compute_cloud_coverage(rh, dp_norm, lat2d)
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

            # Get precipitation if available
            precipitation_cycle = layers.get("precipitation")
            precipitation_sorted = None
            if precipitation_cycle is not None:
                # Convert from kg/m²/s to mm/day for visualization
                precipitation_mm_day = precipitation_cycle * 86400  # 1 kg/m² = 1 mm
                precipitation_sorted = precipitation_mm_day[:, :, lon_sort_idx]

            # Get cloud fractions for all four cloud types
            convective_frac_cycle = layers.get("convective_cloud_frac")
            convective_frac_sorted = None
            if convective_frac_cycle is not None:
                convective_frac_sorted = convective_frac_cycle[:, :, lon_sort_idx]

            stratiform_frac_cycle = layers.get("stratiform_cloud_frac")
            stratiform_frac_sorted = None
            if stratiform_frac_cycle is not None:
                stratiform_frac_sorted = stratiform_frac_cycle[:, :, lon_sort_idx]

            marine_sc_frac_cycle = layers.get("marine_sc_cloud_frac")
            marine_sc_frac_sorted = None
            if marine_sc_frac_cycle is not None:
                marine_sc_frac_sorted = marine_sc_frac_cycle[:, :, lon_sort_idx]

            high_cloud_frac_cycle = layers.get("high_cloud_frac")
            high_cloud_frac_sorted = None
            if high_cloud_frac_cycle is not None:
                high_cloud_frac_sorted = high_cloud_frac_cycle[:, :, lon_sort_idx]

            humidity_data = {
                "Specific Humidity (q)": humidity_q_sorted,
                "Relative Humidity (RH)": humidity_rh_sorted,
                "Cloud Cover": cloud_cover_sorted,
            }
            if precipitation_sorted is not None:
                humidity_data["Precipitation"] = precipitation_sorted
            if convective_frac_sorted is not None:
                humidity_data["Convective Clouds"] = convective_frac_sorted
            if stratiform_frac_sorted is not None:
                humidity_data["Stratiform Clouds"] = stratiform_frac_sorted
            if marine_sc_frac_sorted is not None:
                humidity_data["Marine Stratocumulus"] = marine_sc_frac_sorted
            if high_cloud_frac_sorted is not None:
                humidity_data["High Clouds"] = high_cloud_frac_sorted
        
            # Initial state
            current_state_humidity = {"month": 0, "type": "Specific Humidity (q)"}
        
            def get_norm_and_label(humidity_type: str) -> tuple[Normalize, str]:
                if humidity_type == "Specific Humidity (q)":
                    vmin = 0
                    vmax = float(humidity_q_cycle.max())
                    label = "Specific Humidity (kg/kg)"
                elif humidity_type == "Relative Humidity (RH)":
                    vmin = 0
                    vmax = 1
                    label = "Relative Humidity"
                elif humidity_type == "Precipitation":
                    vmin = 0
                    vmax = 15  # mm/day, typical tropical max
                    label = "Precipitation (mm/day)"
                elif humidity_type == "Convective Clouds":
                    vmin = 0
                    vmax = 0.6  # Max convective fraction from clouds.py
                    label = "Convective Cloud Fraction"
                elif humidity_type == "Stratiform Clouds":
                    vmin = 0
                    vmax = 0.8  # Max stratiform fraction from clouds.py
                    label = "Stratiform Cloud Fraction"
                elif humidity_type == "Marine Stratocumulus":
                    vmin = 0
                    vmax = 0.9  # Max marine Sc fraction from clouds.py
                    label = "Marine Sc Cloud Fraction"
                elif humidity_type == "High Clouds":
                    vmin = 0
                    vmax = 0.7  # Max high cloud fraction from clouds.py
                    label = "High Cloud Fraction"
                else:  # Cloud Cover (old/fallback)
                    vmin = 0
                    vmax = 1
                    label = "Cloud Cover Fraction"
                return Normalize(vmin=vmin, vmax=vmax), label
        
            norm_humidity, colorbar_label_humidity = get_norm_and_label(current_state_humidity["type"])
            cmap_humidity = cmocean.cm.rain
        
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

            # Add hover functionality for humidity plot
            humidity_hover_data = {
                "field": humidity_data[current_state_humidity["type"]][current_state_humidity["month"]],
                "type": current_state_humidity["type"],
            }

            def format_humidity_message(lon: float, lat: float, data: dict, lon_idx: int, lat_idx: int) -> str:
                value = data["field"][lat_idx, lon_idx]
                var_type = data["type"]
                if var_type == "Specific Humidity (q)":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  q: {value:.4f} kg/kg"
                elif var_type == "Relative Humidity (RH)":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  RH: {value:.2%}"
                elif var_type == "Precipitation":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Precip: {value:.1f} mm/day"
                elif var_type == "Convective Clouds":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Conv. Cloud: {value:.2%}"
                elif var_type == "Stratiform Clouds":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Strat. Cloud: {value:.2%}"
                elif var_type == "Marine Stratocumulus":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Marine Sc: {value:.2%}"
                elif var_type == "High Clouds":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  High Cloud: {value:.2%}"
                else:  # Cloud Cover (old/fallback)
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Cloud: {value:.2%}"

            add_dynamic_status_readout(
                fig=fig_humidity,
                ax=ax_humidity,
                lon_coords=lon_sorted,
                lat_coords=lat_sorted,
                data_container=humidity_hover_data,
                format_message=format_humidity_message,
            )

            # Update hover data when plot updates
            original_update = _update_humidity_plot
            def _update_humidity_plot() -> None:
                original_update()
                humidity_hover_data["field"] = humidity_data[current_state_humidity["type"]][current_state_humidity["month"]]
                humidity_hover_data["type"] = current_state_humidity["type"]

            _update_humidity_plot()

    # Surface Properties plot
    if not args.headless:
        # Collect available surface properties
        soil_moisture_cycle = layers.get("soil_moisture")
        vegetation_fraction_cycle = layers.get("vegetation_fraction")
        snow_ice_fraction_cycle = layers.get("snow_ice_fraction")
        # albedo_field is already loaded earlier (2D, not 3D monthly cycle)

        # Check if we have any surface properties to plot
        has_soil_moisture = soil_moisture_cycle is not None
        has_vegetation = vegetation_fraction_cycle is not None
        has_snow_ice = snow_ice_fraction_cycle is not None
        has_albedo = albedo_field is not None

        if has_soil_moisture or has_vegetation or has_snow_ice or has_albedo:
            projection = ccrs.PlateCarree()
            fig_surface, ax_surface = plt.subplots(
                figsize=(12, 6), subplot_kw=dict(projection=projection)
            )
            ax_surface.set_global()
            ax_surface.coastlines(linewidth=0.4)
            ax_surface.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
            ax_surface.add_feature(
                cfeature.NaturalEarthFeature(
                    "physical", "lakes", "110m", edgecolor="#000000", facecolor="none"
                ),
                linewidth=0.2,
            )
            ax_surface.add_feature(
                cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0
            )

            # Wrap longitude for proper display
            lon_full = lon2d[0]
            lon_wrapped = ((lon_full + 180.0) % 360.0) - 180.0
            lon_sort_idx = np.argsort(lon_wrapped)
            lon_sorted = lon_wrapped[lon_sort_idx]

            # Prepare data dictionary with available fields
            surface_props_data: dict[str, np.ndarray] = {}
            surface_props_is_monthly: dict[str, bool] = {}

            if has_soil_moisture:
                surface_props_data["Soil Moisture"] = soil_moisture_cycle[:, :, lon_sort_idx]
                surface_props_is_monthly["Soil Moisture"] = True
            if has_vegetation:
                surface_props_data["Vegetation Fraction"] = vegetation_fraction_cycle[:, :, lon_sort_idx]
                surface_props_is_monthly["Vegetation Fraction"] = True
            if has_snow_ice:
                surface_props_data["Snow/Ice Fraction"] = snow_ice_fraction_cycle[:, :, lon_sort_idx]
                surface_props_is_monthly["Snow/Ice Fraction"] = True
            if has_albedo:
                # albedo_field is 3D (monthly, lat, lon)
                surface_props_data["Surface Albedo"] = albedo_field[:, :, lon_sort_idx]
                surface_props_is_monthly["Surface Albedo"] = True

            # Create sorted lat/lon grids for pcolormesh
            lat_sorted = lat2d[:, 0]
            lon_sorted_2d, lat_sorted_2d = np.meshgrid(lon_sorted, lat_sorted)

            # Initial state
            prop_names = list(surface_props_data.keys())
            current_state_surface = {"month": 0, "type": prop_names[0]}

            def get_surface_norm_and_label(prop_type: str) -> tuple[Normalize, str]:
                if prop_type == "Soil Moisture":
                    return Normalize(vmin=0, vmax=1), "Soil Moisture Fraction"
                elif prop_type == "Vegetation Fraction":
                    return Normalize(vmin=0, vmax=1), "Vegetation Fraction"
                elif prop_type == "Snow/Ice Fraction":
                    return Normalize(vmin=0, vmax=1), "Snow/Ice Fraction"
                elif prop_type == "Surface Albedo":
                    return Normalize(vmin=0, vmax=1), "Surface Albedo"
                else:
                    return Normalize(vmin=0, vmax=1), prop_type

            def get_surface_data(prop_type: str, month_idx: int) -> np.ndarray:
                """Get the data for a surface property, handling monthly vs static fields."""
                data = surface_props_data[prop_type]
                if surface_props_is_monthly[prop_type]:
                    return data[month_idx]
                else:
                    # Static field (like albedo), return as-is
                    return data

            norm_surface, colorbar_label_surface = get_surface_norm_and_label(current_state_surface["type"])
            cmap_surface = cmocean.cm.turbid

            surface_mesh = ax_surface.pcolormesh(
                lon_sorted_2d,
                lat_sorted_2d,
                get_surface_data(current_state_surface["type"], current_state_surface["month"]),
                cmap=cmap_surface,
                norm=norm_surface,
                shading="auto",
                transform=projection,
            )

            ax_surface.set_title(
                f"{current_state_surface['type']} – {month_names[current_state_surface['month']]}"
                if surface_props_is_monthly.get(current_state_surface["type"], True)
                else f"{current_state_surface['type']} (Annual)"
            )

            # Colorbar
            cbar_surface = fig_surface.colorbar(
                surface_mesh,
                ax=ax_surface,
                orientation="vertical",
                pad=0.04,
                fraction=0.046,
            )
            cbar_surface.set_label(colorbar_label_surface)

            def _update_surface_plot() -> None:
                data = get_surface_data(current_state_surface["type"], current_state_surface["month"])
                norm_surface, colorbar_label_surface = get_surface_norm_and_label(current_state_surface["type"])
                surface_mesh.set_norm(norm_surface)
                surface_mesh.set_array(data.ravel())
                cbar_surface.set_label(colorbar_label_surface)
                cbar_surface.update_normal(surface_mesh)

                # Update title based on whether field is monthly or static
                is_monthly = surface_props_is_monthly.get(current_state_surface["type"], True)
                if is_monthly:
                    ax_surface.set_title(
                        f"{current_state_surface['type']} – {month_names[current_state_surface['month']]}"
                    )
                else:
                    ax_surface.set_title(f"{current_state_surface['type']} (Annual)")

                fig_surface.canvas.draw_idle()

            slider_surface_ax = fig_surface.add_axes([0.2, 0.08, 0.6, 0.03])
            month_slider_surface = Slider(
                slider_surface_ax,
                label="Month",
                valmin=0,
                valmax=11,
                valinit=0,
                valstep=1,
                valfmt="%0.0f",
            )

            def _on_surface_month_change(val: float) -> None:
                current_state_surface["month"] = int(val)
                _update_surface_plot()

            month_slider_surface.on_changed(_on_surface_month_change)

            radio_surface_ax = fig_surface.add_axes([0.02, 0.55, 0.14, 0.18])
            radio_surface_ax.set_title("Variable", fontsize=9)
            type_selector_surface = RadioButtons(radio_surface_ax, prop_names, active=0)

            def _on_surface_type_change(label: str) -> None:
                current_state_surface["type"] = label
                _update_surface_plot()

            type_selector_surface.on_clicked(_on_surface_type_change)

            # Add hover functionality for surface properties plot
            surface_hover_data = {
                "field": get_surface_data(current_state_surface["type"], current_state_surface["month"]),
                "type": current_state_surface["type"],
            }

            def format_surface_message(lon: float, lat: float, data: dict, lon_idx: int, lat_idx: int) -> str:
                value = data["field"][lat_idx, lon_idx]
                var_type = data["type"]
                if var_type == "Soil Moisture":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Soil Moist: {value:.2%}"
                elif var_type == "Vegetation Fraction":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Vegetation: {value:.2%}"
                elif var_type == "Snow/Ice Fraction":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Snow/Ice: {value:.2%}"
                elif var_type == "Surface Albedo":
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  Albedo: {value:.3f}"
                else:
                    return f"{_format_lat(lat)}  {_format_lon(lon)}  {var_type}: {value:.3f}"

            add_dynamic_status_readout(
                fig=fig_surface,
                ax=ax_surface,
                lon_coords=lon_sorted,
                lat_coords=lat_sorted,
                data_container=surface_hover_data,
                format_message=format_surface_message,
            )

            # Update hover data when plot updates
            original_surface_update = _update_surface_plot
            def _update_surface_plot() -> None:
                original_surface_update()
                surface_hover_data["field"] = get_surface_data(current_state_surface["type"], current_state_surface["month"])
                surface_hover_data["type"] = current_state_surface["type"]

            _update_surface_plot()

    # Ocean currents plot
    ocean_u = layers.get("ocean_u")
    ocean_v = layers.get("ocean_v")
    if ocean_u is not None and ocean_v is not None and not args.headless:
        projection = ccrs.PlateCarree()
        fig_ocean, ax_ocean = plt.subplots(
            figsize=(12, 6), subplot_kw=dict(projection=projection)
        )
        ax_ocean.set_global()
        ax_ocean.coastlines(linewidth=0.4)
        ax_ocean.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
        ax_ocean.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "lakes", "110m", edgecolor="#000000", facecolor="none"
            ),
            linewidth=0.2,
        )
        ax_ocean.add_feature(
            cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0
        )

        # Wrap longitude for display
        lon_full = lon2d[0]
        lon_wrapped = ((lon_full + 180.0) % 360.0) - 180.0
        lon_sort_idx = np.argsort(lon_wrapped)
        lon_sorted = lon_wrapped[lon_sort_idx]

        ocean_u_sorted = ocean_u[:, :, lon_sort_idx]
        ocean_v_sorted = ocean_v[:, :, lon_sort_idx]

        # Compute ocean current speed (with NaN handling)
        ocean_speed = np.sqrt(
            np.nan_to_num(ocean_u_sorted, nan=0.0)**2 +
            np.nan_to_num(ocean_v_sorted, nan=0.0)**2
        )
        # Set land cells back to NaN for display
        ocean_speed = np.where(
            np.isnan(ocean_u_sorted) | np.isnan(ocean_v_sorted),
            np.nan,
            ocean_speed
        )

        max_speed = float(np.nanmax(ocean_speed))
        if not np.isfinite(max_speed) or max_speed <= 0.0:
            max_speed = 0.1

        stride = max(1, int(round(1.0 / args.resolution)))
        lat_coords = lat2d[::stride, 0]
        lon_coords = lon_sorted[::stride]

        meters_per_deg_lat = np.pi / 180.0 * R_EARTH_METERS
        cosphi = np.cos(np.deg2rad(lat_coords))
        meters_per_deg_lon_vec = meters_per_deg_lat * np.clip(cosphi, 1e-6, None)

        def _to_deg_per_sec_ocean(
            u_slice: np.ndarray, v_slice: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            u_deg = u_slice / meters_per_deg_lon_vec[:, None]
            v_deg = v_slice / meters_per_deg_lat
            return u_deg, v_deg

        cmap_ocean = cmocean.cm.tempo
        norm_ocean = Normalize(vmin=0.0, vmax=max_speed)

        current_state_ocean = {"month": 0}
        stream_container_ocean: dict[str, object | None] = {"obj": None}
        speed_mesh_container: dict[str, object | None] = {"mesh": None}

        # Create background mesh for speed coloring
        lat_sorted = lat2d[:, 0]
        lon_sorted_2d, lat_sorted_2d = np.meshgrid(lon_sorted, lat_sorted)

        speed_mesh = ax_ocean.pcolormesh(
            lon_sorted_2d,
            lat_sorted_2d,
            ocean_speed[0],
            cmap=cmap_ocean,
            norm=norm_ocean,
            shading="auto",
            transform=projection,
            zorder=0,
        )
        speed_mesh_container["mesh"] = speed_mesh

        cbar_ocean = fig_ocean.colorbar(
            speed_mesh,
            ax=ax_ocean,
            orientation="vertical",
            pad=0.04,
            fraction=0.046,
        )
        cbar_ocean.set_label("Current speed (m/s)")

        def _clear_streamplot_ocean(stream_set) -> None:
            if stream_set is None:
                return
            stream_set.lines.set_segments([])
            stream_set.lines.set_array(np.array([]))
            stream_set.lines.set_visible(False)
            for art in list(ax_ocean.get_children()):
                if isinstance(art, mpatches.FancyArrowPatch):
                    art.remove()

        def _draw_streamplot_ocean() -> None:
            idx = current_state_ocean["month"]
            u_slice = ocean_u_sorted[idx, ::stride, ::stride]
            v_slice = ocean_v_sorted[idx, ::stride, ::stride]
            speed_slice = ocean_speed[idx, ::stride, ::stride]

            # Update background speed mesh
            speed_mesh = speed_mesh_container["mesh"]
            if speed_mesh is not None:
                speed_mesh.set_array(ocean_speed[idx].ravel())

            # Replace NaN with 0 for streamplot (it can't handle NaN)
            u_safe = np.nan_to_num(u_slice, nan=0.0)
            v_safe = np.nan_to_num(v_slice, nan=0.0)

            u_deg_slice, v_deg_slice = _to_deg_per_sec_ocean(u_safe, v_safe)

            current_stream = stream_container_ocean["obj"]
            _clear_streamplot_ocean(current_stream)

            new_stream = ax_ocean.streamplot(
                lon_coords,
                lat_coords,
                u_deg_slice,
                v_deg_slice,
                color="#1a1a1a",
                transform=projection,
                density=1.8,
                linewidth=0.8,
                arrowsize=1.2,
            )

            stream_container_ocean["obj"] = new_stream
            stream_container_ocean["u_data"] = u_slice
            stream_container_ocean["v_data"] = v_slice
            stream_container_ocean["speed_data"] = speed_slice
            stream_container_ocean["lon_coords"] = lon_coords
            stream_container_ocean["lat_coords"] = lat_coords

            ax_ocean.set_title(f"Ocean Currents – {month_names[idx]}")
            fig_ocean.canvas.draw_idle()

        slider_ocean_ax = fig_ocean.add_axes([0.2, 0.08, 0.6, 0.03])
        month_slider_ocean = Slider(
            slider_ocean_ax,
            label="Month",
            valmin=0,
            valmax=11,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )

        def _on_ocean_month_change(val: float) -> None:
            current_state_ocean["month"] = int(val)
            _draw_streamplot_ocean()

        month_slider_ocean.on_changed(_on_ocean_month_change)

        # Add hover functionality
        add_dynamic_status_readout(
            fig=fig_ocean,
            ax=ax_ocean,
            lon_coords=stream_container_ocean.get("lon_coords", lon_coords),
            lat_coords=stream_container_ocean.get("lat_coords", lat_coords),
            data_container=stream_container_ocean,
            format_message=lambda lon, lat, data, lon_idx, lat_idx: (
                f"{_format_lat(lat)}  {_format_lon(lon)}  "
                f"Speed: {np.nan_to_num(data.get('speed_data', [[0]])[lat_idx, lon_idx], nan=0.0):.3f} m/s  "
                f"U: {np.nan_to_num(data.get('u_data', [[0]])[lat_idx, lon_idx], nan=0.0):.3f} m/s  "
                f"V: {np.nan_to_num(data.get('v_data', [[0]])[lat_idx, lon_idx], nan=0.0):.3f} m/s"
            ) if data.get("u_data") is not None else ""
        )

        _draw_streamplot_ocean()

    atmosphere_2m_cycle = layer_cycles.get("Atmosphere (2 m)")

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

        if atmosphere_cycle is not None:
            atm_land = area_weighted_mean(atmosphere_cycle[idx], land_weights)
            atm_ocean = area_weighted_mean(atmosphere_cycle[idx], ocean_weights)
            print(f"  Atmosphere mean (land/ocean) [°C]: {atm_land:.2f} / {atm_ocean:.2f}")

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

        if wind_speed_10m is not None:
            wind10_land = area_weighted_mean(wind_speed_10m[idx], land_weights)
            wind10_ocean = area_weighted_mean(wind_speed_10m[idx], ocean_weights)
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

    if wind_speed_10m is not None:
        wind10_land_annual = area_weighted_mean(wind_speed_10m.mean(axis=0), land_weights)
        wind10_ocean_annual = area_weighted_mean(wind_speed_10m.mean(axis=0), ocean_weights)
        print(f"  Wind speed 10 m mean (land/ocean) [m/s]: {wind10_land_annual:.2f} / {wind10_ocean_annual:.2f}")
    else:
        reason = (
            "bulk exchange disabled" if not sensible_heat_config.enabled else "winds unavailable"
        )
        print(f"  Wind speed 10 m mean (land/ocean) [m/s]: N/A ({reason})")

    # Apply interpolation if requested (works in both headless and interactive mode)
    plot_lon2d = lon2d
    plot_lat2d = lat2d
    plot_layer_cycles = layer_cycles

    if args.interpolate:
        output_resolution = 0.25
        print(f"Interpolating temperature fields to {output_resolution}° resolution...")
        with time_block("interpolation"):
            # Build layer map for interpolation (need lowercase keys)
            interp_layers = {
                "surface": layer_cycles["Surface"],
            }
            if "Atmosphere (2 m)" in layer_cycles:
                interp_layers["temperature_2m"] = layer_cycles["Atmosphere (2 m)"]

            plot_lon2d, plot_lat2d, interpolated = interpolate_layer_map(
                interp_layers,
                lon2d,
                lat2d,
                output_resolution_deg=output_resolution,
                apply_lapse_rate_to_2m=True,
            )

            # Rebuild layer_cycles with interpolated data
            plot_layer_cycles = {}
            if "surface" in interpolated:
                plot_layer_cycles["Surface"] = interpolated["surface"]
            if "temperature_2m" in interpolated:
                plot_layer_cycles["Atmosphere (2 m)"] = interpolated["temperature_2m"]

        get_profiler().print_summary()
        print(f"Interpolation complete. Output grid: {plot_lat2d.shape}")

        # Save interpolated data to npz
        data_dir_value = os.getenv("DATA_DIR")
        if data_dir_value:
            data_dir = Path(data_dir_value).expanduser()
            data_dir.mkdir(parents=True, exist_ok=True)
            interp_path = data_dir / "interpolated.npz"
            np.savez(
                interp_path,
                surface=interpolated.get("surface"),
                temperature_2m=interpolated.get("temperature_2m"),
            )
            print(f"Saved interpolated data to {interp_path}")

    if not args.headless:
        # Save GIFs using the (potentially interpolated) data
        data_dir_value = os.getenv("DATA_DIR")
        if data_dir_value:
            data_dir = Path(data_dir_value).expanduser()
            data_dir.mkdir(parents=True, exist_ok=True)

            def _save_cycle(name: str, field: np.ndarray, filename: str) -> None:
                output_path = data_dir / filename
                save_monthly_temperature_gif(
                    plot_lon2d,
                    plot_lat2d,
                    field,
                    output_path=output_path,
                    title=f"{name} Temperature ({unit})",
                    colorbar_label=f"Temperature ({unit})",
                    use_fahrenheit=args.fahrenheit,
                )
                print(f"Saved {name.lower()} temperature animation to {output_path}")

            if "Surface" in plot_layer_cycles:
                _save_cycle("Surface", plot_layer_cycles["Surface"], "surface_temperature_cycle.gif")
            if "Atmosphere (2 m)" in plot_layer_cycles:
                _save_cycle("Two-meter", plot_layer_cycles["Atmosphere (2 m)"], "two_meter_temperature_cycle.gif")
            if "Atmosphere" in plot_layer_cycles:
                _save_cycle("Atmosphere", plot_layer_cycles["Atmosphere"], "atmosphere_temperature_cycle.gif")

        plot_layered_monthly_temperature_cycle(
            plot_lon2d,
            plot_lat2d,
            plot_layer_cycles,
            title=f"Temperature Cycle ({unit})",
            use_fahrenheit=args.fahrenheit,
        )


if __name__ == "__main__":
    main()
