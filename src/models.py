"""Prediction methods: Naive, AR(1) per unit (with pooled AR(1) fallback), Random Forest.

Bug fix (2026-08): `ar1_predict` used to fall back to the *naive* prediction
whenever a unit had fewer than `MIN_TRAIN_OBS_PER_UNIT_AR1` training rows.
In the `new_units` split design a held-out unit has exactly 0 training rows
by definition, so every single test row silently became a naive prediction
labelled "AR(1)" -- AR(1) was never actually tested there (confirmed by the
paired t-test showing zero variation between AR(1) and naive on new_units).

Fix: fall back to a single AR(1) fit pooled across ALL training units
instead. A pooled AR(1) doesn't need any rows from the specific unit being
predicted, so it produces a real, distinct prediction for unseen units too.
This also improves the (rare) case of a unit with 1-4 training rows, where
a naive fallback was a scientific placeholder more than a real forecast.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import MIN_TRAIN_OBS_PER_UNIT_AR1, RANDOM_SEED


def naive_predict(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["lag_1_target"].to_numpy()


def _fit_ar1(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    """OLS fit of target ~ intercept + phi * lag_1_target. None if degenerate."""
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return None
    phi = np.sum((x - x_mean) * (y - y_mean)) / denom
    alpha = y_mean - phi * x_mean
    return float(alpha), float(phi)


def fit_pooled_ar1(train_df: pd.DataFrame) -> Tuple[float, float]:
    """Fit one AR(1) pooled across every unit in `train_df`.

    Used as the fallback for units with too little (or no) unit-specific
    training data, including entirely unseen units in the `new_units`
    design. Degenerate case (no variation in the lag feature) falls back
    further to predicting the unconditional training-set mean.
    """
    x = train_df["lag_1_target"].to_numpy()
    y = train_df["target"].to_numpy()
    fitted = _fit_ar1(x, y)
    if fitted is None:
        return float(y.mean()), 0.0
    return fitted


def ar1_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_train_obs: int = MIN_TRAIN_OBS_PER_UNIT_AR1,
) -> np.ndarray:
    """Fit an AR(1) per unit on training rows only.

    For a unit with at least `min_train_obs` training rows and a
    non-degenerate fit, uses that unit's own AR(1). Otherwise (too few
    rows, including 0 for an unseen unit, or a degenerate per-unit fit)
    falls back to the pooled AR(1) fit across all training units -- see
    `fit_pooled_ar1`. This never falls back to the naive prediction, so
    AR(1) is now a real predictor in every split design.
    """
    preds = np.full(len(test_df), np.nan)
    test_df = test_df.reset_index(drop=True)

    pooled_alpha, pooled_phi = fit_pooled_ar1(train_df)

    for unit in test_df["unit_id"].unique():
        unit_train = train_df[train_df["unit_id"] == unit]
        unit_test_idx = test_df.index[test_df["unit_id"] == unit]
        x_test = test_df.loc[unit_test_idx, "lag_1_target"].to_numpy()

        fitted = None
        if len(unit_train) >= min_train_obs:
            fitted = _fit_ar1(
                unit_train["lag_1_target"].to_numpy(),
                unit_train["target"].to_numpy(),
            )

        if fitted is None:
            alpha, phi = pooled_alpha, pooled_phi
        else:
            alpha, phi = fitted

        preds[unit_test_idx] = alpha + phi * x_test

    return preds


def ar1_predict_per_unit_only(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_train_obs: int = MIN_TRAIN_OBS_PER_UNIT_AR1,
) -> np.ndarray:
    """Strictly per-unit AR(1), no pooled fallback at all.

    For comparison only (meeting follow-up: "what happens if we use a
    separate AR(1) for each unit"). A test row for a unit with fewer than
    `min_train_obs` training rows, or a degenerate per-unit fit, is left as
    NaN instead of falling back to anything -- `metrics.asep` skips NaNs,
    so those rows are simply excluded rather than silently guessed. This is
    the "mark as not applicable" alternative from the new_units fix,
    applied generally.
    """
    preds = np.full(len(test_df), np.nan)
    test_df = test_df.reset_index(drop=True)

    for unit in test_df["unit_id"].unique():
        unit_train = train_df[train_df["unit_id"] == unit]
        unit_test_idx = test_df.index[test_df["unit_id"] == unit]

        if len(unit_train) < min_train_obs:
            continue

        fitted = _fit_ar1(
            unit_train["lag_1_target"].to_numpy(),
            unit_train["target"].to_numpy(),
        )
        if fitted is None:
            continue

        alpha, phi = fitted
        x_test = test_df.loc[unit_test_idx, "lag_1_target"].to_numpy()
        preds[unit_test_idx] = alpha + phi * x_test

    return preds


def rf_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    features = ["lag_1_target", "unit_code", "time"]
    model = RandomForestRegressor(n_estimators=300, random_state=random_seed, n_jobs=-1)
    model.fit(train_df[features], train_df["target"])
    return model.predict(test_df[features])
