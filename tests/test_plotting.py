import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("cartopy")

from climate_sim.plotting import plot_monthly_temperature_cycle
from climate_sim.utils import solver


def test_plot_monthly_temperature_cycle_runs():
    lon2d, lat2d, monthly_temps = solver.compute_periodic_cycle_celsius(resolution_deg=90.0)
    plot_monthly_temperature_cycle(lon2d, lat2d, monthly_temps)
    plt.close("all")
