import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("cartopy")

from climate_sim.plotting import (
    plot_layered_monthly_temperature_cycle,
    plot_monthly_temperature_cycle,
)
from climate_sim.utils import solver


def test_plot_monthly_temperature_cycle_runs():
    lon2d, lat2d, monthly_temps = solver.compute_periodic_cycle_results(resolution_deg=90.0)
    fig = plot_monthly_temperature_cycle(lon2d, lat2d, monthly_temps, show=False)
    assert fig is not None
    plt.close("all")


def test_plot_layered_monthly_temperature_cycle_runs():
    lon2d, lat2d, layers = solver.compute_periodic_cycle_results(
        resolution_deg=90.0, return_layer_map=True
    )
    layer_fields = {"Surface": layers["surface"]}
    if "atmosphere" in layers:
        layer_fields["Atmosphere"] = layers["atmosphere"]
    fig = plot_layered_monthly_temperature_cycle(lon2d, lat2d, layer_fields, show=False)
    assert fig is not None
    plt.close("all")
