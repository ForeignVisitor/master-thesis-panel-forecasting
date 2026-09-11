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


def aaep(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Average Absolute Error of Prediction.

    Same NaN handling as `asep`. Added for Part A (mean vs median): squared
    error (ASEP) is minimized by the mean, absolute error is minimized by
    the median -- looking at both side by side is what actually shows
    whether "median beats mean under skew" is true, instead of assuming it.
    """
    predicted = np.asarray(predicted, dtype=float)
    valid = ~np.isnan(predicted)
    return float(np.mean(np.abs(actual[valid] - predicted[valid])))
