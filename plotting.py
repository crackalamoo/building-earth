import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
) -> None:
    """Render a scalar field on an equirectangular map with land outlines."""
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

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

    cbar = fig.colorbar(mesh, orientation="horizontal", pad=0.07)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
        cbar.ax.set_xticklabels([f"{int(tick)}" for tick in colorbar_ticks])

    add_status_readout(fig, ax, lon2d, lat2d, field)

    fig.tight_layout()
    plt.show()


def plot_temperature_field(
    lon2d: np.ndarray, lat2d: np.ndarray, field: np.ndarray, *, title: str = "Temperature Field"
) -> None:
    """Specialised helper for °C temperature fields."""
    cmap, bounds = build_temperature_cmap()
    vmin, vmax = bounds[0], bounds[-1]
    norm = Normalize(vmin=vmin, vmax=vmax)

    plot_field(
        lon2d,
        lat2d,
        field,
        title=title,
        cmap=cmap,
        norm=norm,
        colorbar_label="Temperature (°C)",
        colorbar_ticks=bounds,
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
