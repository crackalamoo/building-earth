"""Calendar utilities shared across the project."""

import numpy as np

SECONDS_PER_DAY = 86_400.0
"""Number of seconds in a day."""

DAYS_PER_MONTH = np.array(
    [31.0, 28.2425, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0],
    dtype=float,
)
"""Mean length of each calendar month expressed in days."""

MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
"""Full names for each calendar month."""
