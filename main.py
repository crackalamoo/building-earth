import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize


def build_temperature_cmap() -> tuple[LinearSegmentedColormap, np.ndarray]:
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


def plot_dummy_field(lon2d: np.ndarray, lat2d: np.ndarray, field: np.ndarray) -> None:
    """Render the dummy temperature field on a global map with land outlines only."""
    cmap, bounds = build_temperature_cmap()
    vmin, vmax = bounds[0], bounds[-1]
    norm = Normalize(vmin=vmin, vmax=vmax)

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
    ax.set_title("Dummy Temperature Field", fontsize=14, weight="bold")

    cbar = fig.colorbar(mesh, orientation="horizontal", pad=0.07)
    cbar.set_label("Temperature (°C)")
    cbar.set_ticks(bounds)
    cbar.ax.set_xticklabels([f"{int(b)}" for b in bounds])

    add_status_readout(fig, ax, lon2d, lat2d, field)

    fig.tight_layout()
    plt.show()


def main() -> None:
    # Grid (1° resolution, cell centers)
    lats = np.linspace(-89.5, 89.5, 180)
    lons = np.linspace(0.5, 359.5, 360)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Convert to radians
    lat_r = np.deg2rad(lat2d)
    lon_r = np.deg2rad(lon2d)

    # Construct a more realistic climatology-inspired temperature field (°C)
    polar_gradient = 23.0 * np.sin(np.abs(lat_r))**1.3
    tropical_boost = 2.0 * np.cos(lat_r)**4

    rng = np.random.default_rng(42)
    raw_noise = rng.standard_normal(lon2d.shape)
    kernel_lon = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel_lon /= kernel_lon.sum()
    smoothed_lon = np.apply_along_axis(
        lambda m: np.convolve(m, kernel_lon, mode="same"), axis=1, arr=raw_noise
    )
    kernel_lat = np.array([1.0, 2.0, 1.0])
    kernel_lat /= kernel_lat.sum()
    smoothed_noise = np.apply_along_axis(
        lambda m: np.convolve(m, kernel_lat, mode="same"), axis=0, arr=smoothed_lon
    )
    longitudinal_noise = 4.0 * smoothed_noise * np.cos(lat_r) ** 2

    T = 28.0 - polar_gradient + tropical_boost + longitudinal_noise

    # Optional: print stats
    print(f"Tmin={T.min():.1f}°C, Tmax={T.max():.1f}°C, mean={T.mean():.1f}°C")

    plot_dummy_field(lon2d, lat2d, T)


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


if __name__ == "__main__":
    main()
