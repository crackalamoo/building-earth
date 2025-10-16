import argparse
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from matplotlib.streamplot import StreamplotSet

from climate_sim.modeling.advection import (
    GeostrophicAdvectionConfig,
    GeostrophicAdvectionOperator,
)
from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
from climate_sim.utils.solver import compute_periodic_cycle_celsius

parser = argparse.ArgumentParser(
    description="Visualise the monthly geostrophic wind diagnosed from the model."
)
parser.add_argument(
    "--resolution",
    type=float,
    default=1.0,
    help="Grid resolution in degrees (must be positive).",
)
args = parser.parse_args()

if args.resolution <= 0:
    raise ValueError("resolution must be positive")

radiation_config = RadiationConfig()
diffusion_config = DiffusionConfig()
advection_config = GeostrophicAdvectionConfig()
snow_config = SnowAlbedoConfig()

lon2d, lat2d, layers = compute_periodic_cycle_celsius(
    resolution_deg=args.resolution,
    radiation_config=radiation_config,
    diffusion_config=diffusion_config,
    advection_config=advection_config,
    snow_config=snow_config,
    return_layer_map=True,
)

if "atmosphere" in layers:
    temperature_cycle_c = layers["atmosphere"]
    layer_label = "Atmosphere"
else:
    temperature_cycle_c = layers["surface"]
    layer_label = "Surface"

temperature_cycle_k = temperature_cycle_c + 273.15

operator = GeostrophicAdvectionOperator(
    lon2d,
    lat2d,
    config=advection_config,
)

months = temperature_cycle_k.shape[0]
wind_u = np.zeros_like(temperature_cycle_k)
wind_v = np.zeros_like(temperature_cycle_k)
wind_speed = np.zeros_like(temperature_cycle_k)

for idx in range(months):
    u_field, v_field, speed_field = operator.wind_field(temperature_cycle_k[idx])
    wind_u[idx] = u_field
    wind_v[idx] = v_field
    wind_speed[idx] = speed_field

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

lon_full = lon2d[0]
lon_wrapped = ((lon_full + 180.0) % 360.0) - 180.0
lon_sort_idx = np.argsort(lon_wrapped)
lon_sorted = lon_wrapped[lon_sort_idx]

wind_u = wind_u[:, :, lon_sort_idx]
wind_v = wind_v[:, :, lon_sort_idx]
wind_speed = wind_speed[:, :, lon_sort_idx]

max_speed = float(np.max(wind_speed))
if not np.isfinite(max_speed) or max_speed <= 0.0:
    max_speed = 1.0

stride = max(1, int(round(1.0 / args.resolution)))
lat_coords = lat2d[::stride, 0]
lon_coords = lon_sorted[::stride]

R_EARTH_METERS = 6.371e6
meters_per_deg_lat = np.pi / 180.0 * R_EARTH_METERS

cosphi = np.cos(np.deg2rad(lat_coords))
meters_per_deg_lon_vec = meters_per_deg_lat * np.clip(cosphi, 1e-6, None)


def to_deg_per_sec(u_slice: np.ndarray, v_slice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_deg = u_slice / meters_per_deg_lon_vec[:, None]
    v_deg = v_slice / meters_per_deg_lat
    return u_deg, v_deg

projection = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=projection))
fig.subplots_adjust(bottom=0.18)

ax.set_global()
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#555555")
ax.add_feature(
    cfeature.NaturalEarthFeature(
        "physical", "lakes", "110m", edgecolor="#333333", facecolor="none"
    ),
    linewidth=0.2,
)
ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)

norm = Normalize(vmin=0.0, vmax=max_speed)
cmap = plt.cm.viridis

u0 = wind_u[0, ::stride, ::stride]
v0 = wind_v[0, ::stride, ::stride]
s0 = wind_speed[0, ::stride, ::stride]
u0_deg, v0_deg = to_deg_per_sec(u0, v0)

def _draw_streamplot(
    u_field: np.ndarray, v_field: np.ndarray, magnitude: np.ndarray
) -> StreamplotSet:
    return ax.streamplot(
        lon_coords,
        lat_coords,
        u_field,
        v_field,
        color=magnitude,
        cmap=cmap,
        norm=norm,
        transform=projection,
        density=1.8,
        linewidth=1.2,
        arrowsize=1.4,
    )


stream = _draw_streamplot(
    u0_deg,
    v0_deg,
    s0,
)

cbar = fig.colorbar(stream.lines, ax=ax, orientation="vertical", pad=0.04, fraction=0.046)
cbar.set_label("Wind speed (m/s)")

stream_container = {"obj": stream}


def _clear_streamplot_artists(stream_set: StreamplotSet) -> None:
    """Blank existing streamplot artists so they no longer draw."""
    # Clear line segments
    stream_set.lines.set_segments([])
    stream_set.lines.set_array(np.array([]))
    stream_set.lines.set_visible(False)

    # Clear arrow patches
    if stream_set.arrows is not None:
        stream_set.arrows.set_paths([])
        stream_set.arrows.set_array(np.array([]))
        stream_set.arrows.set_visible(False)

initial_label = month_names[0 % len(month_names)]
ax.set_title(f"Geostrophic Wind ({layer_label}) – {initial_label}")

slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
month_slider = Slider(
    slider_ax,
    label="Month",
    valmin=0,
    valmax=months - 1,
    valinit=0,
    valstep=1,
    valfmt="%0.0f",
)

def update(month_idx: float) -> None:
    index = int(month_idx)
    u_field = wind_u[index, ::stride, ::stride]
    v_field = wind_v[index, ::stride, ::stride]
    speed_field = wind_speed[index, ::stride, ::stride]
    u_deg, v_deg = to_deg_per_sec(u_field, v_field)

    # Remove existing stream artists before drawing the updated field.
    current_stream = stream_container["obj"]
    _clear_streamplot_artists(current_stream)

    new_stream = _draw_streamplot(
        u_deg,
        v_deg,
        speed_field,
    )

    stream_container["obj"] = new_stream
    cbar.update_normal(new_stream.lines)
    month_label = month_names[index % len(month_names)]
    ax.set_title(f"Geostrophic Wind ({layer_label}) – {month_label}")
    fig.canvas.draw_idle()

month_slider.on_changed(update)

plt.show()
