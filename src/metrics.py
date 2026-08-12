"""Evaluation metrics."""

import numpy as np


def asep(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Average Squared Error of Prediction.

    Ignores any NaN entries in `predicted` (kept for safety / backward
    compatibility -- with the pooled AR(1) fallback in models.ar1_predict,
    AR(1) should no longer produce NaNs, but Random Forest or future methods
    could still legitimately skip a row).
    """
    predicted = np.asarray(predicted, dtype=float)
    valid = ~np.isnan(predicted)
    return float(np.mean((actual[valid] - predicted[valid]) ** 2))
