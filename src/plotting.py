"""Small shared plotting helpers."""

from typing import Tuple

import numpy as np


def ecdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sorted_values, cumulative_fraction) for an empirical CDF plot."""
    values = np.sort(np.asarray(values))
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y
