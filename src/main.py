from climate_sim.plotting import plot_monthly_temperature_cycle
from climate_sim.utils.solver import compute_periodic_cycle_celsius


def main() -> None:
    lon2d, lat2d, layers = compute_periodic_cycle_celsius(return_layer_map=True)
    surface_cycle = layers["surface"]

    print(
        f"Annual Tmin={surface_cycle.min():.1f}°C, "
        f"Annual Tmax={surface_cycle.max():.1f}°C, "
        f"Annual mean={surface_cycle.mean():.1f}°C"
    )

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        surface_cycle,
        title="Surface Temperature Cycle",
    )


if __name__ == "__main__":
    main()
