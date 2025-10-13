import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
from typing import Iterable, Optional, Tuple


def build_temperature_cmap() -> Tuple[LinearSegmentedColormap, np.ndarray]:
    """Custom temperature ramp with category-aware transitions."""
    bounds = np.array([-30.0, 0.0, 15.0, 20.0, 25.0, 30.0, 45.0])
    colors = [
        "#0B1E6D",  # deep cold blue (<0 °C)
        "#1E88E5",  # cooler blue
        "#64B5F6",  # light blue (0–15 °C)
        "#66BB6A",  # green (15–20 °C)
        "#FFEB3B",  # yellow (20–25 °C)
        "#FB8C00",  # orange (25–30 °C)
        "#D32F2F",  # red (30+ °C)
    ]
    normalized = (bounds - bounds[0]) / (bounds[-1] - bounds[0])
    return LinearSegmentedColormap.from_list(
        "custom_temperature_categories", list(zip(normalized, colors))
    ), bounds


def plot_field(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field: np.ndarray,
    *,
    title: str = "Scalar Field",
    cmap: Optional[LinearSegmentedColormap] = None,
    norm: Optional[Normalize] = None,
    colorbar_label: str = "",
    colorbar_ticks: Optional[Iterable[float]] = None,
    colorbar_orientation: str = "vertical",
    stats_text: Optional[str] = None,
) -> None:
    """Render a scalar field on an equirectangular map with land outlines."""
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    divider = make_axes_locatable(ax)

    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        field,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    land_outline = cfeature.NaturalEarthFeature(
        "physical",
        "land",
        scale="110m",
        edgecolor="black",
        facecolor="none",
    )
    ax.add_feature(land_outline, linewidth=0.6)

    ax.set_global()
    ax.set_title(title, fontsize=14, weight="bold")

    if colorbar_orientation == "vertical":
        cax = divider.append_axes("right", size="3%", pad=0.08, axes_class=Axes)
        cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
    else:
        cax = divider.append_axes("bottom", size="5%", pad=0.4, axes_class=Axes)
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    if colorbar_label:
        cbar.set_label(colorbar_label)
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
        labels = [f"{int(tick)}" for tick in colorbar_ticks]
        if colorbar_orientation == "horizontal":
            cbar.ax.set_xticklabels(labels)
        else:
            cbar.ax.set_yticklabels(labels)

    if stats_text:
        if colorbar_orientation == "vertical":
            stats_ax = divider.append_axes("left", size="9%", pad=0.25, axes_class=Axes)
        else:
            stats_ax = divider.append_axes("top", size="12%", pad=0.7, axes_class=Axes)
        stats_ax.axis("off")
        stats_ax.text(
            0.5,
            0.5,
            stats_text,
            ha="center",
            va="center",
            fontsize=8,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
            transform=stats_ax.transAxes,
        )

    add_status_readout(fig, ax, lon2d, lat2d, field)

    plt.show()


def plot_temperature_field(
    lon2d: np.ndarray, lat2d: np.ndarray, field: np.ndarray, *, title: str = "Temperature Field"
) -> None:
    """Specialised helper for °C temperature fields."""
    cmap, bounds = build_temperature_cmap()
    vmin, vmax = bounds[0], bounds[-1]
    norm = Normalize(vmin=vmin, vmax=vmax)
    stats_text = (
        f"min {field.min():.1f}°C\n"
        f"mean {field.mean():.1f}°C\n"
        f"max {field.max():.1f}°C"
    )

    plot_field(
        lon2d,
        lat2d,
        field,
        title=title,
        cmap=cmap,
        norm=norm,
        colorbar_label="Temperature (°C)",
        colorbar_ticks=bounds,
        colorbar_orientation="vertical",
        stats_text=stats_text,
    )


def add_status_readout(
    fig: plt.Figure,
    ax: plt.Axes,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field: np.ndarray,
) -> None:
    """Push hover readout to the interactive toolbar/status bar."""
    manager = getattr(fig.canvas, "manager", None)
    toolbar = getattr(manager, "toolbar", None)
    if toolbar is None or not hasattr(toolbar, "set_message"):
        return

    lons = lon2d[0, :]
    lats = lat2d[:, 0]

    def format_lat(lat_deg: float) -> str:
        hemisphere = "N" if lat_deg >= 0 else "S"
        return f"{abs(lat_deg):.1f}°{hemisphere}"

    def format_lon(lon_deg: float) -> str:
        lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
        hemisphere = "E" if lon_wrapped >= 0 else "W"
        return f"{abs(lon_wrapped):.1f}°{hemisphere}"

    def on_move(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            toolbar.set_message("")
            return

        lon = event.xdata % 360.0
        lat = event.ydata

        if not np.isfinite(lon) or not np.isfinite(lat):
            return

        lon_idx = int(np.abs(lons - lon).argmin())
        lat_idx = int(np.abs(lats - lat).argmin())

        sample_lon = lon2d[lat_idx, lon_idx]
        sample_lat = lat2d[lat_idx, lon_idx]
        temperature = field[lat_idx, lon_idx]

        toolbar.set_message(
            f"{format_lat(sample_lat)}  {format_lon(sample_lon)}  {temperature:.1f} °C"
        )

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("figure_leave_event", lambda _evt: toolbar.set_message(""))


def plot_monthly_temperature_cycle(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    monthly_field: np.ndarray,
    *,
    title: str = "Monthly Temperature Cycle",
) -> None:
    """Interactive monthly cycle viewer driven by a slider."""
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
    cmap, bounds = build_temperature_cmap()
    vmin, vmax = bounds[0], bounds[-1]
    norm = Normalize(vmin=vmin, vmax=vmax)

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=projection))
    ax.set_global()
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2)
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)

    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        monthly_field[0],
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=projection,
    )
    ax.set_title(f"{title} – {month_names[0]}")

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.04, fraction=0.046)
    cbar.set_label("Temperature (°C)")
    cbar.set_ticks(bounds)

    slider_ax = fig.add_axes([0.12, 0.1, 0.76, 0.03])
    slider = Slider(
        slider_ax,
        label="Month",
        valmin=0,
        valmax=11,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
    )

    current_field = {"data": monthly_field[0]}

    def update_status_handler() -> None:
        lons = lon2d[0, :]
        lats = lat2d[:, 0]
        manager = getattr(fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        if toolbar is None or not hasattr(toolbar, "set_message"):
            return

        def format_lat(lat_deg: float) -> str:
            hemisphere = "N" if lat_deg >= 0 else "S"
            return f"{abs(lat_deg):.1f}°{hemisphere}"

        def format_lon(lon_deg: float) -> str:
            lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
            hemisphere = "E" if lon_wrapped >= 0 else "W"
            return f"{abs(lon_wrapped):.1f}°{hemisphere}"

        def on_move(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                toolbar.set_message("")
                return

            lon = event.xdata % 360.0
            lat = event.ydata

            if not np.isfinite(lon) or not np.isfinite(lat):
                return

            lon_idx = int(np.abs(lons - lon).argmin())
            lat_idx = int(np.abs(lats - lat).argmin())

            sample_lon = lon2d[lat_idx, lon_idx]
            sample_lat = lat2d[lat_idx, lon_idx]
            temperature = current_field["data"][lat_idx, lon_idx]

            toolbar.set_message(
                f"{format_lat(sample_lat)}  {format_lon(sample_lon)}  {temperature:.1f} °C"
            )

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig.canvas.mpl_connect("figure_leave_event", lambda _evt: toolbar.set_message(""))

    update_status_handler()

    def on_update(idx: float) -> None:
        month_index = int(idx)
        data = monthly_field[month_index]
        mesh.set_array(data.ravel())
        ax.set_title(f"{title} – {month_names[month_index]}")
        current_field["data"] = data
        fig.canvas.draw_idle()

    slider.on_changed(on_update)
    plt.show()
