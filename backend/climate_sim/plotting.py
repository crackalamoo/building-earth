# type: ignore[attr-defined]
"""Visualization helpers for rendering climate model outputs."""

import io
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.widgets import RadioButtons, Slider
from PIL import Image

from climate_sim.data.calendar import MONTH_NAMES
from climate_sim.core.units import convert_temperature, temperature_unit


def build_temperature_cmap(
    unit: str = "C",
) -> tuple[LinearSegmentedColormap, np.ndarray]:
    """Custom temperature ramp with category-aware transitions."""
    bounds = np.array([-30.0, -15.0, 0.0, 0.0, 10.0, 21.0, 25.0, 30.0, 35.0, 40.0])
    colors = [
        "#3B1E6D",  # deep purple (-30 °C)
        "#0B1E6D",  # deep cold blue (-15 °C)
        "#1E88E5",  # cooler blue (0 °C, freezing)
        "#64B5F6",  # light blue (0 °C)
        "#66BB6A",  # green (10 °C)
        "#FFEB3B",  # yellow (21 °C)
        "#FB8C00",  # orange (25 °C)
        "#D32F2F",  # red (30 °C)
        "#B5382A",  # deep red (35 °C)
        "#8A0000",  # deeper red (40 °C)
    ]
    normalized = (bounds - bounds[0]) / (bounds[-1] - bounds[0])
    cmap = LinearSegmentedColormap.from_list(
        "custom_temperature_categories", list(zip(normalized, colors))
    )
    if unit.upper().startswith("F"):
        bounds = (bounds * (9.0 / 5.0)) + 32.0
    return cmap, bounds


def _format_lat(lat_deg: float) -> str:
    """Format latitude with hemisphere indicator."""
    hemisphere = "N" if lat_deg >= 0 else "S"
    return f"{abs(lat_deg):.1f}°{hemisphere}"


def _format_lon(lon_deg: float) -> str:
    """Format longitude with hemisphere indicator."""
    lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
    hemisphere = "E" if lon_wrapped >= 0 else "W"
    return f"{abs(lon_wrapped):.1f}°{hemisphere}"


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
    add_dynamic_status_readout(
        fig=fig,
        ax=ax,
        lon_coords=lon2d[0, :],
        lat_coords=lat2d[:, 0],
        data_container={"field": field},
        format_message=lambda lon, lat, data, lon_idx, lat_idx: (
            f"{_format_lat(lat)}  {_format_lon(lon)}  "
            f"{data['field'][lat_idx, lon_idx]:.1f}" + (f" {unit_label}" if unit_label else "")
        ),
    )


def add_dynamic_status_readout(
    fig: plt.Figure,
    ax: plt.Axes,
    lon_coords: np.ndarray,
    lat_coords: np.ndarray,
    data_container: dict,
    format_message: callable,
) -> None:
    """Add hover readout with custom data and formatting.

    Args:
        fig: The matplotlib figure
        ax: The axes to attach hover events to
        lon_coords: 1-D array of longitude coordinates
        lat_coords: 1-D array of latitude coordinates
        data_container: Dictionary containing data arrays that can be updated dynamically
        format_message: Callable(lon, lat, data_container, lon_idx, lat_idx) -> str
                       Function to format the status message from sampled data
    """
    manager = getattr(fig.canvas, "manager", None)
    toolbar = getattr(manager, "toolbar", None)
    if toolbar is None or not hasattr(toolbar, "set_message"):
        return

    def on_move(event):
        if not ax.get_visible() or event.x is None or event.y is None:
            return

        # Use coordinate transform to check if cursor is within this axes,
        # since event.inaxes may point to an invisible overlapping axes.
        try:
            coords = ax.transData.inverted().transform((event.x, event.y))
        except Exception:
            return
        x_data, y_data = coords
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if not (xlim[0] <= x_data <= xlim[1] and ylim[0] <= y_data <= ylim[1]):
            toolbar.set_message("")
            return

        lon = x_data % 360.0
        lat = y_data

        if not np.isfinite(lon) or not np.isfinite(lat):
            return

        lon_idx = int(np.argmin(np.abs(lon_coords - lon)))
        lat_idx = int(np.argmin(np.abs(lat_coords - lat)))

        sample_lon = lon_coords[lon_idx]
        sample_lat = lat_coords[lat_idx]

        message = format_message(sample_lon, sample_lat, data_container, lon_idx, lat_idx)
        toolbar.set_message(message)

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
    per_layer_styles: Sequence[dict] | None = None,
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
        per_layer_styles=per_layer_styles,
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
    per_layer_styles: Sequence[dict] | None = None,
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

    month_names = list(MONTH_NAMES) + ["Average"]

    unit = temperature_unit(use_fahrenheit)
    default_cmap, bounds = build_temperature_cmap(unit=unit[-1])
    if cmap is None:
        cmap = default_cmap
    if norm is None:
        vmin, vmax = bounds[0], bounds[-1]
        norm = Normalize(vmin=vmin, vmax=vmax)
    if use_fahrenheit and colorbar_label.endswith("°C)"):
        colorbar_label = colorbar_label[:-3] + "°F)"

    # Build per-layer style list (defaults to the shared cmap/norm/label)
    layer_styles: list[dict] = []
    for i in range(len(items)):
        style: dict = {
            "cmap": cmap,
            "norm": norm,
            "colorbar_label": colorbar_label,
            "unit": unit,
        }
        if per_layer_styles is not None and i < len(per_layer_styles):
            style.update(per_layer_styles[i])
        layer_styles.append(style)

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=projection))
    ax.set_global()
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "lakes", "110m", edgecolor="#000000", facecolor="none"
        ),
        linewidth=0.2,
    )
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)

    current_state = {"layer": 0, "month": 0}

    data_stack_display = [
        convert_temperature(field, use_fahrenheit, is_delta=value_is_delta) for field in data_stack
    ]

    style0 = layer_styles[0]
    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        data_stack_display[current_state["layer"]][current_state["month"]],
        cmap=style0["cmap"],
        norm=style0["norm"],
        shading="auto",
        transform=projection,
    )

    def format_title() -> str:
        month_label = month_names[current_state["month"]]
        if len(layer_names) == 1:
            return f"{title} – {month_label}"
        layer_label = layer_names[current_state["layer"]]
        return f"{title} – {layer_label} – {month_label}"

    ax.set_title(format_title())

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.04, fraction=0.046)
    if style0["colorbar_label"]:
        cbar.set_label(style0["colorbar_label"])
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
    elif style0["cmap"] is default_cmap:
        cbar.set_ticks(bounds)

    slider_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        label="Month",
        valmin=0,
        valmax=reference_shape[0] - 1,
        valinit=current_state["month"],
        valstep=1,
        valfmt="%0.0f",
    )

    current_field = {"data": data_stack_display[current_state["layer"]][current_state["month"]]}

    add_dynamic_status_readout(
        fig=fig,
        ax=ax,
        lon_coords=lon2d[0, :],
        lat_coords=lat2d[:, 0],
        data_container=current_field,
        format_message=lambda lon, lat, data, lon_idx, lat_idx: (
            f"{_format_lat(lat)}  {_format_lon(lon)}  "
            f"{data['data'][lat_idx, lon_idx]:.1f} {layer_styles[current_state['layer']]['unit']}"
        ),
    )

    def update_plot() -> None:
        li = current_state["layer"]
        data = data_stack_display[li][current_state["month"]]
        style = layer_styles[li]
        mesh.set_cmap(style["cmap"])
        mesh.set_norm(style["norm"])
        mesh.set_array(data.ravel())
        ax.set_title(format_title())
        cbar.update_normal(mesh)
        cbar.set_label(style["colorbar_label"])
        current_field["data"] = data
        fig.canvas.draw_idle()

    def on_update(idx: float) -> None:
        current_state["month"] = int(idx)
        update_plot()

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


def plot_obs_vs_sim_grid(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    columns: Sequence[dict],
    *,
    title: str = "Observed vs Simulated",
    use_fahrenheit: bool = False,
) -> None:
    """2-row × N-column grid: observed on top, simulated on bottom.

    Each entry in *columns* is a dict with keys:
        obs, sim : (nmonths, nlat, nlon) arrays
        cmap, norm : colormap and normalization
        colorbar_label : str
        label : column title (e.g. "Temperature")
        unit : str for hover readout
        is_temperature : bool (default False) — apply F conversion if needed
        colorbar_ticks : optional list
        quiver_uv : optional dict with "obs" and "sim" keys, each (u, v) tuple
                     of (nmonths, nlat, nlon) arrays for wind direction arrows
    """
    ncols = len(columns)
    month_names = list(MONTH_NAMES) + ["Average"]

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(6 * ncols + 1, 9),
        subplot_kw=dict(projection=projection),
    )
    if ncols == 1:
        axes = axes.reshape(2, 1)
    fig.subplots_adjust(bottom=0.12, hspace=0.12, wspace=0.05)

    meshes: list[list] = [[], []]  # meshes[row][col]
    display_data: list[list[np.ndarray]] = [[], []]  # display_data[row][col]
    quivers: list[list] = [[], []]  # quivers[row][col] — None or quiver artist
    quiver_data: list[list] = [[], []]  # quiver_data[row][col] — None or (u, v)

    # Subsample stride for quiver arrows
    nlat, nlon = lon2d.shape
    stride_lat = max(1, nlat // 18)
    stride_lon = max(1, nlon // 36)
    q_lat = slice(None, None, stride_lat)
    q_lon = slice(None, None, stride_lon)
    q_lon2d = lon2d[q_lat, q_lon]
    q_lat2d = lat2d[q_lat, q_lon]

    for ci, col in enumerate(columns):
        is_temp = col.get("is_temperature", False)
        qdata = col.get("quiver_uv")
        for ri, key in enumerate(("obs", "sim")):
            ax = axes[ri, ci]
            ax.set_global()
            ax.coastlines(linewidth=0.4)
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    "physical", "lakes", "110m", edgecolor="#000000", facecolor="none"
                ),
                linewidth=0.2,
            )
            ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)

            raw = col[key]
            data = convert_temperature(raw, use_fahrenheit, is_delta=False) if is_temp else raw
            display_data[ri].append(data)

            mesh = ax.pcolormesh(
                lon2d,
                lat2d,
                data[0],
                cmap=col["cmap"],
                norm=col["norm"],
                shading="auto",
                transform=projection,
            )
            meshes[ri].append(mesh)

            if qdata is not None:
                u_full, v_full = qdata[key]
                u_sub = u_full[0, q_lat, q_lon]
                v_sub = v_full[0, q_lat, q_lon]
                qv = ax.quiver(
                    q_lon2d,
                    q_lat2d,
                    u_sub,
                    v_sub,
                    transform=projection,
                    scale=120,
                    width=0.003,
                    headwidth=3,
                    headlength=4,
                    color="k",
                    alpha=0.6,
                )
                quivers[ri].append(qv)
                quiver_data[ri].append((u_full, v_full))
            else:
                quivers[ri].append(None)
                quiver_data[ri].append(None)

        axes[0, ci].set_title(col["label"], fontsize=11)

    axes[0, 0].text(
        -0.05,
        0.5,
        "Observed",
        transform=axes[0, 0].transAxes,
        fontsize=11,
        va="center",
        ha="right",
        rotation=90,
    )
    axes[1, 0].text(
        -0.05,
        0.5,
        "Simulated",
        transform=axes[1, 0].transAxes,
        fontsize=11,
        va="center",
        ha="right",
        rotation=90,
    )

    # One colorbar per column, placed manually to the right of each column
    for ci, col in enumerate(columns):
        pos = axes[0, ci].get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.005, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(meshes[0][ci], cax=cbar_ax)
        cbar.set_label(col.get("colorbar_label", ""))
        ticks = col.get("colorbar_ticks")
        if ticks is not None:
            cbar.set_ticks(ticks)

    fig.suptitle(f"{title} – {month_names[0]}", fontsize=14)

    slider_ax = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        label="Month",
        valmin=0,
        valmax=display_data[0][0].shape[0] - 1,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
    )

    def on_update(idx: float) -> None:
        m = int(idx)
        for ri in range(2):
            for ci in range(ncols):
                meshes[ri][ci].set_array(display_data[ri][ci][m].ravel())
                if quivers[ri][ci] is not None:
                    u_full, v_full = quiver_data[ri][ci]
                    quivers[ri][ci].set_UVC(
                        u_full[m, q_lat, q_lon],
                        v_full[m, q_lat, q_lon],
                    )
        fig.suptitle(f"{title} – {month_names[m]}", fontsize=14)
        fig.canvas.draw_idle()

    slider.on_changed(on_update)
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

    month_names = list(MONTH_NAMES)

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
