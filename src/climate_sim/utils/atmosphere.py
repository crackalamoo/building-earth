"""Atmospheric utility functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

STANDARD_LAPSE_RATE_K_PER_M = 6.5 / 1000.0


def adjust_temperature_by_elevation(
    temperature_c: NDArray[np.floating],
    elevation_delta_m: NDArray[np.floating] | float,
) -> NDArray[np.floating]:
    """Apply a fixed lapse-rate correction for an elevation change.

    Parameters
    ----------
    temperature_c:
        Base temperature field expressed in degrees Celsius.
    elevation_delta_m:
        Elevation difference in metres (positive for moving upward). Can be a
        scalar or array broadcastable to ``temperature_c``.

    Returns
    -------
    numpy.ndarray
        Temperature field adjusted for the specified elevation change.
    """

    temp = np.asarray(temperature_c, dtype=float)
    delta = np.asarray(elevation_delta_m, dtype=float)
    adjusted = temp - STANDARD_LAPSE_RATE_K_PER_M * delta
    return adjusted.astype(float, copy=False)
