"""Utilities for handling temperature unit conversions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

ArrayLike = NDArray[np.floating[Any]]


def convert_temperature(
    values: ArrayLike | float,
    use_fahrenheit: bool,
    *,
    is_delta: bool = False,
) -> ArrayLike | float:
    """Convert temperatures between Celsius and Fahrenheit."""
    if not use_fahrenheit:
        return values

    as_array = np.asarray(values, dtype=float)
    converted = as_array * (9.0 / 5.0)
    if not is_delta:
        converted = converted + 32.0

    if isinstance(values, np.ndarray):
        return converted.astype(values.dtype, copy=False)
    if np.isscalar(values):
        return float(converted)
    return converted


def temperature_unit(use_fahrenheit: bool) -> str:
    """Return the temperature unit string for display purposes."""
    return "°F" if use_fahrenheit else "°C"
