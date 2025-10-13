from modeling.radiation import compute_temperature_celsius
from plotting import plot_temperature_field


def main() -> None:
    lon2d, lat2d, temperature = compute_temperature_celsius()

    # Optional: print stats
    print(
        f"Tmin={temperature.min():.1f}°C, "
        f"Tmax={temperature.max():.1f}°C, "
        f"mean={temperature.mean():.1f}°C"
    )

    plot_temperature_field(lon2d, lat2d, temperature, title="Black-body Radiative Temperature Field")


if __name__ == "__main__":
    main()
