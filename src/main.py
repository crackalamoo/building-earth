from climate_sim.modeling.radiation import compute_periodic_radiative_cycle_celsius
from climate_sim.plotting import plot_monthly_temperature_cycle


def main() -> None:
    lon2d, lat2d, monthly_cycle = compute_periodic_radiative_cycle_celsius()

    # Optional: print stats
    print(
        f"Annual Tmin={monthly_cycle.min():.1f}°C, "
        f"Annual Tmax={monthly_cycle.max():.1f}°C, "
        f"Annual mean={monthly_cycle.mean():.1f}°C"
    )

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        monthly_cycle,
        title="Temperature Cycle",
    )


if __name__ == "__main__":
    main()
