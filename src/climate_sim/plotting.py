import io
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import RadioButtons, Slider
from PIL import Image
from typing import Iterable, Mapping, Sequence

from climate_sim.utils.temperature import convert_temperature, temperature_unit


def build_temperature_cmap(
    unit: str = "C",
) -> tuple[LinearSegmentedColormap, np.ndarray]:
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
    cmap = LinearSegmentedColormap.from_list(
        "custom_temperature_categories", list(zip(normalized, colors))
    )
    if unit.upper().startswith("F"):
        bounds = (bounds * (9.0 / 5.0)) + 32.0
    return cmap, bounds


def plot_field(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field: np.ndarray,
    *,
    title: str = "Scalar Field",
    cmap: LinearSegmentedColormap | None = None,
    norm: Normalize | None = None,
    colorbar_label: str = "",
    colorbar_ticks: Iterable[float] | None = None,
    colorbar_orientation: str = "vertical",
    stats_text: str | None = None,
    status_unit: str | None = None,
) -> None:
    """Render a scalar field on an equirectangular map with land outlines."""
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    divider = make_axes_locatable(ax)

    display_field = np.ma.masked_invalid(field)

    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        display_field,
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

    add_status_readout(fig, ax, lon2d, lat2d, display_field, unit_label=status_unit)

    plt.show()


def plot_temperature_field(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field: np.ndarray,
    *,
    title: str = "Temperature Field",
    use_fahrenheit: bool = False,
    value_is_delta: bool = False,
) -> None:
    """Specialised helper for temperature fields with unit conversions."""
    unit = temperature_unit(use_fahrenheit)
    cmap, bounds = build_temperature_cmap(unit=unit[-1])
    display_field = convert_temperature(field, use_fahrenheit, is_delta=value_is_delta)
    vmin, vmax = bounds[0], bounds[-1]
    norm = Normalize(vmin=vmin, vmax=vmax)
    stats_text = (
        f"min {display_field.min():.1f}{unit}\n"
        f"mean {display_field.mean():.1f}{unit}\n"
        f"max {display_field.max():.1f}{unit}"
    )

    plot_field(
        lon2d,
        lat2d,
        display_field,
        title=title,
        cmap=cmap,
        norm=norm,
        colorbar_label=f"Temperature ({unit})",
        colorbar_ticks=bounds,
        colorbar_orientation="vertical",
        stats_text=stats_text,
        status_unit=unit,
    )


def add_status_readout(
    fig: plt.Figure,
    ax: plt.Axes,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field: np.ndarray,
    *,
    unit_label: str | None = None,
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

        if np.ma.is_masked(temperature) or not np.isfinite(temperature):
            toolbar.set_message(
                f"{format_lat(sample_lat)}  {format_lon(sample_lon)}  missing"
            )
            return

        suffix = f" {unit_label}" if unit_label else ""
        toolbar.set_message(
            f"{format_lat(sample_lat)}  {format_lon(sample_lon)}  {float(temperature):.1f}{suffix}"
        )

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("figure_leave_event", lambda _evt: toolbar.set_message(""))


def plot_monthly_temperature_cycle(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    monthly_field: np.ndarray,
    *,
    title: str = "Monthly Temperature Cycle",
    cmap: LinearSegmentedColormap | None = None,
    norm: Normalize | None = None,
    colorbar_label: str = "Temperature (°C)",
    colorbar_ticks: Iterable[float] | None = None,
    use_fahrenheit: bool = False,
    value_is_delta: bool = False,
    missing_color: str = "#4A1486",
    missing_label: str | None = "Missing data",
    month_labels: Sequence[str] | None = None,
) -> None:
    """Interactive monthly cycle viewer driven by a slider."""
    plot_layered_monthly_temperature_cycle(
        lon2d,
        lat2d,
        {"Surface": monthly_field},
        title=title,
        cmap=cmap,
        norm=norm,
        colorbar_label=colorbar_label,
        colorbar_ticks=colorbar_ticks,
        use_fahrenheit=use_fahrenheit,
        value_is_delta=value_is_delta,
        missing_color=missing_color,
        missing_label=missing_label,
        month_labels=month_labels,
    )


def plot_layered_monthly_temperature_cycle(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    layer_fields: Mapping[str, np.ndarray] | Sequence[tuple[str, np.ndarray]],
    *,
    title: str = "Monthly Temperature Cycle",
    cmap: LinearSegmentedColormap | None = None,
    norm: Normalize | None = None,
    colorbar_label: str = "Temperature (°C)",
    colorbar_ticks: Iterable[float] | None = None,
    use_fahrenheit: bool = False,
    value_is_delta: bool = False,
    missing_color: str = "#4A1486",
    missing_label: str | None = "Missing data",
    month_labels: Sequence[str] | None = None,
) -> None:
    """Interactive monthly cycle viewer with month slider and layer selector."""

    if isinstance(layer_fields, Mapping):
        items = list(layer_fields.items())
    else:
        items = list(layer_fields)

    if not items:
        raise ValueError("layer_fields must contain at least one entry")

    layer_names = [str(name) for name, _ in items]
    data_stack = [np.asarray(field) for _, field in items]

    reference_shape = data_stack[0].shape
    if len(reference_shape) != 3:
        raise ValueError("Each layer field must be a 3-D array of shape (month, lat, lon)")

    for field in data_stack:
        if field.shape != reference_shape:
            raise ValueError("All layer fields must share the same shape")

    n_months = reference_shape[0]
    if month_labels is None:
        if n_months != 12:
            raise ValueError("Expected 12 monthly slices per layer when month_labels is not provided")
        month_labels = [
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

    month_label_list = [str(label) for label in month_labels]
    if len(month_label_list) != n_months:
        raise ValueError("Number of month_labels must match the leading dimension of the layer fields")

    unit = temperature_unit(use_fahrenheit)
    default_cmap, bounds = build_temperature_cmap(unit=unit[-1])
    if cmap is None:
        cmap = default_cmap
    if norm is None:
        vmin, vmax = bounds[0], bounds[-1]
        norm = Normalize(vmin=vmin, vmax=vmax)
    if use_fahrenheit and colorbar_label.endswith("°C)"):
        colorbar_label = colorbar_label[:-3] + "°F)"

    cmap = cmap.with_extremes(bad=missing_color)

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=projection))
    ax.set_global()
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
    ax.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'lakes', '110m',
        edgecolor='#000000', facecolor='none'),
        linewidth=0.2)
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)

    current_state = {"layer": 0, "month": 0}

    data_stack_display = [
        np.ma.masked_invalid(
            convert_temperature(field, use_fahrenheit, is_delta=value_is_delta)
        )
        for field in data_stack
    ]

    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        data_stack_display[current_state["layer"]][current_state["month"]],
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=projection,
    )

    if missing_label is not None and any(
        np.ma.getmaskarray(field).any() for field in data_stack_display
    ):
        legend = ax.legend(
            handles=[Patch(facecolor=missing_color, label=missing_label)],
            loc="lower left",
            fontsize=8,
            framealpha=0.85,
        )
        legend.set_zorder(mesh.get_zorder() + 0.5)

    def format_title() -> str:
        month_label = month_label_list[current_state["month"]]
        if len(layer_names) == 1:
            return f"{title} – {month_label}"
        layer_label = layer_names[current_state["layer"]]
        return f"{title} – {layer_label} – {month_label}"

    ax.set_title(format_title())

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.04, fraction=0.046)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
    elif cmap is default_cmap:
        cbar.set_ticks(bounds)

    slider_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        label="Month",
        valmin=0,
        valmax=n_months - 1,
        valinit=current_state["month"],
        valstep=1,
        valfmt="%0.0f",
    )
    slider.valtext.set_text(month_label_list[current_state["month"]])

    current_field = {
        "data": data_stack_display[current_state["layer"]][current_state["month"]]
    }

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

            if np.ma.is_masked(temperature) or not np.isfinite(temperature):
                toolbar.set_message(
                    f"{format_lat(sample_lat)}  {format_lon(sample_lon)}  missing"
                )
                return

            toolbar.set_message(
                f"{format_lat(sample_lat)}  {format_lon(sample_lon)}  {float(temperature):.1f} {unit}"
            )

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig.canvas.mpl_connect("figure_leave_event", lambda _evt: toolbar.set_message(""))

    update_status_handler()

    def update_plot() -> None:
        data = data_stack_display[current_state["layer"]][current_state["month"]]
        mesh.set_array(np.ma.ravel(data))
        ax.set_title(format_title())
        current_field["data"] = data
        fig.canvas.draw_idle()

    def on_update(idx: float) -> None:
        current_state["month"] = int(idx)
        update_plot()
        slider.valtext.set_text(month_label_list[current_state["month"]])

    slider.on_changed(on_update)

    if len(layer_names) > 1:
        radio_ax = fig.add_axes([0.01, 0.45, 0.11, 0.25])
        radio_ax.set_title("Layer", fontsize=10)
        _layer_selector = RadioButtons(radio_ax, layer_names, active=current_state["layer"])

        def on_layer(label: str) -> None:
            current_state["layer"] = layer_names.index(label)
            update_plot()

        _layer_selector.on_clicked(on_layer)

    plt.show()


def save_monthly_temperature_gif(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    monthly_field: np.ndarray,
    *,
    output_path: str | Path,
    title: str = "Monthly Temperature Cycle",
    cmap: LinearSegmentedColormap | None = None,
    norm: Normalize | None = None,
    colorbar_label: str = "Temperature (°C)",
    colorbar_ticks: Iterable[float] | None = None,
    use_fahrenheit: bool = False,
    value_is_delta: bool = False,
    fps: float = 2.0,
) -> Path:
    """Render a monthly temperature cycle to an animated GIF."""
    monthly_array = np.asarray(monthly_field)
    if monthly_array.ndim != 3 or monthly_array.shape[0] != 12:
        raise ValueError("monthly_field must have shape (12, n_lat, n_lon)")

    unit = temperature_unit(use_fahrenheit)
    default_cmap, bounds = build_temperature_cmap(unit=unit[-1])
    if cmap is None:
        cmap = default_cmap
    if norm is None:
        vmin, vmax = bounds[0], bounds[-1]
        norm = Normalize(vmin=vmin, vmax=vmax)
    if use_fahrenheit and colorbar_label.endswith("°C)"):
        colorbar_label = colorbar_label[:-3] + "°F)"

    display_monthly = convert_temperature(
        monthly_array,
        use_fahrenheit,
        is_delta=value_is_delta,
    )

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

    projection = ccrs.PlateCarree()
    frames: list[Image.Image] = []

    for month_idx, month_name in enumerate(month_names):
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw=dict(projection=projection),
        )
        ax.set_global()
        ax.coastlines(linewidth=0.4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "lakes",
                "110m",
                edgecolor="#000000",
                facecolor="none",
            ),
            linewidth=0.2,
        )
        ax.add_feature(
            cfeature.LAND,
            facecolor="#f5f5f5",
            edgecolor="none",
            zorder=0,
        )

        mesh = ax.pcolormesh(
            lon2d,
            lat2d,
            display_monthly[month_idx],
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=projection,
        )
        ax.set_title(f"{title} – {month_name}")

        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.04, fraction=0.046)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks)
        elif cmap is default_cmap:
            cbar.set_ticks(bounds)

        fig.tight_layout()
        fig.canvas.draw()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        buffer.seek(0)
        with Image.open(buffer) as image:
            frames.append(image.convert("P", palette=Image.ADAPTIVE))
        buffer.close()

    if not frames:
        raise RuntimeError("No frames were generated for the GIF.")

    duration_ms = 500
    if fps > 0:
        duration_ms = max(20, int(round(1000.0 / fps)))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=True,
        disposal=2,
    )

    for frame in frames:
        frame.close()

    return output_path
