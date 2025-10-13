import numpy as np

from plotting import plot_temperature_field


def generate_dummy_temperature_field(lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """Return a smooth, realistic-looking temperature field (°C)."""
    lat_r = np.deg2rad(lat2d)

    polar_gradient = 23.0 * np.sin(np.abs(lat_r))**1.3
    tropical_boost = 2.0 * np.cos(lat_r)**4

    rng = np.random.default_rng(42)
    raw_noise = rng.standard_normal(lon2d.shape)
    kernel_lon = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel_lon /= kernel_lon.sum()
    smoothed_lon = np.apply_along_axis(
        lambda m: np.convolve(m, kernel_lon, mode="same"), axis=1, arr=raw_noise
    )
    kernel_lat = np.array([1.0, 2.0, 1.0], dtype=float)
    kernel_lat /= kernel_lat.sum()
    smoothed_noise = np.apply_along_axis(
        lambda m: np.convolve(m, kernel_lat, mode="same"), axis=0, arr=smoothed_lon
    )
    longitudinal_noise = 4.0 * smoothed_noise * np.cos(lat_r) ** 2

    return 28.0 - polar_gradient + tropical_boost + longitudinal_noise


def main() -> None:
    # Grid (1° resolution, cell centers)
    lats = np.linspace(-89.5, 89.5, 180)
    lons = np.linspace(0.5, 359.5, 360)
    lon2d, lat2d = np.meshgrid(lons, lats)

    temperature = generate_dummy_temperature_field(lon2d, lat2d)

    # Optional: print stats
    print(
        f"Tmin={temperature.min():.1f}°C, "
        f"Tmax={temperature.max():.1f}°C, "
        f"mean={temperature.mean():.1f}°C"
    )

    plot_temperature_field(lon2d, lat2d, temperature, title="Dummy Temperature Field")


if __name__ == "__main__":
    main()
