"""Tests for math utility helpers."""

import numpy as np
import pytest

from climate_sim.utils.math import spherical_cell_area


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
    scaled = spherical_cell_area(lon2d, lat2d, earth_radius_m=bigger_radius)

    assert original is copy_cached
    assert scaled is not original
    np.testing.assert_allclose(scaled, (bigger_radius / radius) ** 2 * original)

    with pytest.raises(ValueError):
        original[0, 0] = 0.0
