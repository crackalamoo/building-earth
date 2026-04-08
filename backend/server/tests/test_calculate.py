"""Tests for the calculate expression evaluator and reductions."""

import numpy as np
import pytest

from backend.server.calculate import resolve_fields
from backend.server.climate_data import ClimateDataStore


class _FakeStore(ClimateDataStore):
    """In-memory ClimateDataStore for tests. Skips file loading and lets each
    test stuff arrays directly. Grid shape is (12 months, nlat, nlon)."""

    def __init__(self, fields: dict[str, np.ndarray]) -> None:
        # Bypass ClimateDataStore.__init__ — no file I/O.
        self._data = dict(fields)


def _grid(nlat: int = 4, nlon: int = 8) -> tuple[int, int, int]:
    """Standard tiny test grid: 12 months × 4 lat bands × 8 lon bands."""
    return 12, nlat, nlon


def test_global_mean_constant_field() -> None:
    nm, nlat, nlon = _grid()
    store = _FakeStore({"temperature_2m": np.full((nm, nlat, nlon), 42.0)})
    result = resolve_fields("global_mean(T)", store, lat=None, lon=None, month=0)
    assert result["result"] == pytest.approx(42.0)


def test_global_mean_is_cos_lat_weighted() -> None:
    """A field that's nonzero only at high latitudes should weight less than
    naive averaging predicts, because cos(67.5°) ~= 0.38."""
    nm, nlat, nlon = _grid()
    # T = 100 in the polar bands (rows 0 and 3), 0 in the mid-latitude bands.
    field = np.zeros((nm, nlat, nlon))
    field[:, [0, 3], :] = 100.0
    store = _FakeStore({"temperature_2m": field})

    # Hand-computed expected:
    # lats = [-67.5, -22.5, 22.5, 67.5]; w = cos(deg2rad(lats))
    # Per-row sums of T*w (then summed across rows) over total weight:
    #   numerator   = 8*(100*0.3827 + 0*0.9239 + 0*0.9239 + 100*0.3827)
    #               = 8*76.54 = 612.4
    #   denominator = 8*(0.3827 + 0.9239 + 0.9239 + 0.3827) = 8*2.6131 = 20.91
    # weighted_mean ~= 612.4 / 20.91 ~= 29.29
    result = resolve_fields("global_mean(T)", store, lat=None, lon=None, month=0)
    assert result["result"] == pytest.approx(29.29, abs=0.05)
    # And critically: NOT the naive 50 from unweighted averaging.
    assert result["result"] < 40.0


def test_zonal_mean_depends_on_requested_latitude() -> None:
    """zonal_mean(T) at lat=67.5 should average only the polar row's lons,
    not the mid-latitude row's. Different lats → different results."""
    nm, nlat, nlon = _grid()
    # Each lat row holds a distinct constant value across all lons.
    field = np.zeros((nm, nlat, nlon))
    field[:, 0, :] = 10.0   # lat = -67.5
    field[:, 1, :] = 20.0   # lat = -22.5
    field[:, 2, :] = 30.0   # lat =  22.5
    field[:, 3, :] = 40.0   # lat =  67.5
    store = _FakeStore({"temperature_2m": field})

    polar = resolve_fields("zonal_mean(T)", store, lat=67.5, lon=None, month=0)
    midlat = resolve_fields("zonal_mean(T)", store, lat=22.5, lon=None, month=0)
    south = resolve_fields("zonal_mean(T)", store, lat=-67.5, lon=None, month=0)

    assert polar["result"] == pytest.approx(40.0)
    assert midlat["result"] == pytest.approx(30.0)
    assert south["result"] == pytest.approx(10.0)


def test_lat_band_mean_selects_correct_band() -> None:
    """lat_band_mean(T, 0, 90) should average only the northern bands,
    skipping the southern ones."""
    nm, nlat, nlon = _grid()
    field = np.zeros((nm, nlat, nlon))
    field[:, 0, :] = -100.0  # -67.5  (skipped by [0, 90])
    field[:, 1, :] = -50.0   # -22.5  (skipped)
    field[:, 2, :] = 50.0    #  22.5  (included)
    field[:, 3, :] = 100.0   #  67.5  (included)
    store = _FakeStore({"temperature_2m": field})

    # Northern hemisphere band: [0, 90]. cos(22.5°)=0.924, cos(67.5°)=0.383
    # Mean = (50*0.924 + 100*0.383) / (0.924 + 0.383)
    #      = (46.2 + 38.3) / 1.307 ≈ 64.65
    nh = resolve_fields(
        "lat_band_mean(T, 0, 90)", store, lat=None, lon=None, month=0
    )
    assert nh["result"] == pytest.approx(64.65, abs=0.05)

    # Symmetric southern band → negated result by construction.
    sh = resolve_fields(
        "lat_band_mean(T, -90, 0)", store, lat=None, lon=None, month=0
    )
    assert sh["result"] == pytest.approx(-64.65, abs=0.05)


def test_box_mean_isolates_single_cell() -> None:
    """A field with a single hot cell — box_mean over a box that contains
    only that cell returns its value; a box that excludes it returns 0."""
    nm, nlat, nlon = _grid()
    # Lats: [-67.5, -22.5, 22.5, 67.5]; lons: [22.5, 67.5, 112.5, ..., 337.5]
    field = np.zeros((nm, nlat, nlon))
    # Hot cell at lat=22.5 (index 2), lon=112.5 (index 2).
    field[:, 2, 2] = 1000.0
    store = _FakeStore({"temperature_2m": field})

    # Box (lat_min=10, lon_min=100, lat_max=30, lon_max=130) contains only
    # the (22.5, 112.5) cell. With one cell, the cos-weighted mean is just
    # the cell value.
    inside = resolve_fields(
        "box_mean(T, 10, 100, 30, 130)", store, lat=None, lon=None, month=0
    )
    assert inside["result"] == pytest.approx(1000.0)

    # Same lat band but a different lon range → no cells contain the hot one.
    outside = resolve_fields(
        "box_mean(T, 10, 200, 30, 250)", store, lat=None, lon=None, month=0
    )
    assert outside["result"] == pytest.approx(0.0)


def test_box_mean_longitude_wraparound() -> None:
    """A box that crosses the antimeridian (lon_min > lon_max) should select
    cells in both [lon_min, 360) and [0, lon_max], not the giant complement."""
    nm, nlat, nlon = _grid()
    # Lons at 4-band-aligned positions: [22.5, 67.5, 112.5, 157.5, 202.5,
    # 247.5, 292.5, 337.5]. Set the first and last lon columns hot — these
    # are the closest cells to the antimeridian (lon=22.5 just east of 0,
    # lon=337.5 just west of 360).
    field = np.zeros((nm, nlat, nlon))
    field[:, 2, 0] = 100.0   # lat=22.5, lon=22.5
    field[:, 2, 7] = 100.0   # lat=22.5, lon=337.5
    store = _FakeStore({"temperature_2m": field})

    # Wraparound box from lon=300 to lon=60 (crossing 0/360). This should
    # capture lons 337.5 and 22.5 (both = 100), nothing else in row 2.
    # The lat band [10, 30] only includes row index 2 (lat=22.5).
    # Two cells, both 100 → mean = 100.
    wrap = resolve_fields(
        "box_mean(T, 10, 300, 30, 60)", store, lat=None, lon=None, month=0
    )
    assert wrap["result"] == pytest.approx(100.0)

    # Non-wrapping box from lon=60 to lon=300 (the great middle). Row 2 has
    # lons 67.5, 112.5, 157.5, 202.5, 247.5, 292.5 — six zero-valued cells.
    middle = resolve_fields(
        "box_mean(T, 10, 60, 30, 300)", store, lat=None, lon=None, month=0
    )
    assert middle["result"] == pytest.approx(0.0)


def test_composition_inside_reduction() -> None:
    """global_mean of a composite expression — sqrt(u² + v²) — should
    evaluate the composite cell-by-cell and then reduce."""
    nm, nlat, nlon = _grid()
    store = _FakeStore({
        "wind_u_10m": np.full((nm, nlat, nlon), 3.0),
        "wind_v_10m": np.full((nm, nlat, nlon), 4.0),
    })
    result = resolve_fields(
        "global_mean(sqrt(u**2 + v**2))", store, lat=None, lon=None, month=0
    )
    # 3-4-5 triangle: every cell is 5, so the global mean is 5.
    assert result["result"] == pytest.approx(5.0)


def test_anomaly_from_zonal_mean() -> None:
    """T - zonal_mean(T) should be a per-cell anomaly. With a constant lon
    profile, the local value equals the zonal mean and the anomaly is 0.
    With a hot cell, that cell's anomaly is positive."""
    nm, nlat, nlon = _grid()
    field = np.zeros((nm, nlat, nlon))
    # All cells in row lat=22.5 (index 2) are 10, except cell at lon index 2
    # which is 80. Zonal mean of that row = (7*10 + 80) / 8 = 18.75.
    field[:, 2, :] = 10.0
    field[:, 2, 2] = 80.0
    store = _FakeStore({"temperature_2m": field})

    # At the hot cell: T=80, zonal_mean=18.75, anomaly=61.25
    hot = resolve_fields(
        "T - zonal_mean(T)", store, lat=22.5, lon=112.5, month=0
    )
    assert hot["result"] == pytest.approx(61.25)

    # At a cold cell in the same row: T=10, zonal_mean=18.75, anomaly=-8.75
    cold = resolve_fields(
        "T - zonal_mean(T)", store, lat=22.5, lon=22.5, month=0
    )
    assert cold["result"] == pytest.approx(-8.75)


def test_omitted_month_means_annual_mean_at_point() -> None:
    """When `month` is None, sampling a single cell should return the mean
    of the field over all 12 months at that cell."""
    nm, nlat, nlon = _grid()
    field = np.zeros((nm, nlat, nlon))
    # At cell (lat=22.5, lon=112.5), monthly values are 0..11 (mean 5.5).
    field[:, 2, 2] = np.arange(12)
    store = _FakeStore({"temperature_2m": field})

    # July (month=6) → just the July value, 6.0.
    july = resolve_fields("T", store, lat=22.5, lon=112.5, month=6)
    assert july["result"] == pytest.approx(6.0)

    # No month → annual mean, 5.5.
    annual = resolve_fields("T", store, lat=22.5, lon=112.5, month=None)
    assert annual["result"] == pytest.approx(5.5)


def test_omitted_month_means_annual_mean_inside_reduction() -> None:
    """global_mean(T) with no month should average over all 12 months
    AND all cells."""
    nm, nlat, nlon = _grid()
    field = np.zeros((nm, nlat, nlon))
    # Constant 10 for months 0-5, constant 20 for months 6-11. Annual = 15.
    field[:6, :, :] = 10.0
    field[6:, :, :] = 20.0
    store = _FakeStore({"temperature_2m": field})

    july = resolve_fields("global_mean(T)", store, lat=None, lon=None, month=6)
    annual = resolve_fields("global_mean(T)", store, lat=None, lon=None, month=None)

    assert july["result"] == pytest.approx(20.0)
    assert annual["result"] == pytest.approx(15.0)


def test_single_point_eval_still_works() -> None:
    """Regression: pre-reductions behavior — bare expression evaluated at
    one cell, including field aliases and math composition."""
    nm, nlat, nlon = _grid()
    store = _FakeStore({
        "temperature_2m": np.full((nm, nlat, nlon), 25.0),
        "wind_u_10m": np.full((nm, nlat, nlon), 3.0),
        "wind_v_10m": np.full((nm, nlat, nlon), 4.0),
    })
    # Bare alias.
    t = resolve_fields("T", store, lat=22.5, lon=112.5, month=0)
    assert t["result"] == pytest.approx(25.0)

    # Composite arithmetic.
    speed = resolve_fields("sqrt(u**2 + v**2)", store, lat=22.5, lon=112.5, month=0)
    assert speed["result"] == pytest.approx(5.0)


def test_missing_lat_lon_for_single_cell_reference_errors() -> None:
    """A bare field reference requires lat and lon — clean error otherwise."""
    nm, nlat, nlon = _grid()
    store = _FakeStore({"temperature_2m": np.full((nm, nlat, nlon), 25.0)})
    with pytest.raises(Exception, match="lat/lon"):
        resolve_fields("T", store, lat=None, lon=None, month=0)


def test_zonal_mean_without_lat_errors() -> None:
    """zonal_mean needs to know which latitude row to average."""
    nm, nlat, nlon = _grid()
    store = _FakeStore({"temperature_2m": np.full((nm, nlat, nlon), 25.0)})
    with pytest.raises(Exception, match="zonal_mean requires lat"):
        resolve_fields("zonal_mean(T)", store, lat=None, lon=None, month=0)
