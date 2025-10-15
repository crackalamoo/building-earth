"""Tests for math utility helpers."""

import numpy as np
import pytest

from climate_sim.utils.math import (
    regular_latitude_edges,
    regular_longitude_edges,
    spherical_cell_area,
    spherical_meridional_metrics,
    spherical_zonal_metrics,
)


def _build_regular_grid(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray]:
    lat_centers = np.linspace(-90.0 + 90.0 / nlat, 90.0 - 90.0 / nlat, nlat)
    lon_centers = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)
    return lon2d, lat2d


def test_spherical_cell_area_reuses_cached_arrays() -> None:
    lon2d, lat2d = _build_regular_grid(18, 36)
    radius = 6_371_000.0

    first = spherical_cell_area(lon2d, lat2d, earth_radius_m=radius)
    second = spherical_cell_area(lon2d, lat2d, earth_radius_m=radius)

    assert first is second
    assert first.flags.writeable is False


def test_spherical_cell_area_handles_copies_and_multiple_radii() -> None:
    lon2d, lat2d = _build_regular_grid(9, 18)
    lon2d_copy = lon2d.copy()
    lat2d_copy = lat2d.copy()

    radius = 6_371_000.0
    bigger_radius = radius * 1.1

    original = spherical_cell_area(lon2d, lat2d, earth_radius_m=radius)
    copy_cached = spherical_cell_area(lon2d_copy, lat2d_copy, earth_radius_m=radius)
    scaled_first = spherical_cell_area(
        lon2d, lat2d, earth_radius_m=bigger_radius
    )
    scaled_second = spherical_cell_area(
        lon2d, lat2d, earth_radius_m=bigger_radius
    )
    original_again = spherical_cell_area(lon2d, lat2d, earth_radius_m=radius)

    assert original is copy_cached
    assert scaled_first is scaled_second
    np.testing.assert_allclose(scaled_first, (bigger_radius / radius) ** 2 * original)
    np.testing.assert_allclose(original_again, original)

    with pytest.raises(ValueError):
        original[0, 0] = 0.0


def test_spherical_metrics_match_direct_calculation() -> None:
    lon2d, lat2d = _build_regular_grid(6, 12)
    radius = 6_371_000.0

    boundary_north, delta_y = spherical_meridional_metrics(
        lon2d, lat2d, earth_radius_m=radius
    )
    boundary_east, delta_x = spherical_zonal_metrics(
        lon2d, lat2d, earth_radius_m=radius
    )

    lon_edges = regular_longitude_edges(lon2d[0])
    lat_edges = regular_latitude_edges(lat2d[:, 0])

    lon_edges_rad = np.deg2rad(lon_edges)
    lat_edges_rad = np.deg2rad(lat_edges)
    lat_centers_rad = np.deg2rad(lat2d[:, 0])

    expected_delta_lon = lon_edges_rad[1:] - lon_edges_rad[:-1]
    expected_delta_lat_cell = lat_edges_rad[1:] - lat_edges_rad[:-1]
    expected_delta_lat_centers = lat_centers_rad[1:] - lat_centers_rad[:-1]
    interface_lat_rad = lat_edges_rad[1:-1]

    expected_boundary_north = (
        radius * np.cos(interface_lat_rad)[:, np.newaxis] * expected_delta_lon
    )
    expected_delta_y = radius * expected_delta_lat_centers
    expected_boundary_east = radius * expected_delta_lat_cell[:, np.newaxis]
    expected_delta_x = (
        radius * np.cos(lat_centers_rad)[:, np.newaxis] * expected_delta_lon
    )

    np.testing.assert_allclose(boundary_north, expected_boundary_north)
    np.testing.assert_allclose(delta_y, expected_delta_y)
    np.testing.assert_allclose(boundary_east, expected_boundary_east)
    np.testing.assert_allclose(delta_x, expected_delta_x)
