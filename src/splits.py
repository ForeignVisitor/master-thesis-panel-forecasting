"""Monte Carlo train/test split designs used across the pipeline scripts."""

from typing import Set, Tuple

import numpy as np
import pandas as pd

from .config import RANDOM_SEED, TEST_FRACTION


def split_random_rows(
    panel: pd.DataFrame,
    rep: int,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random 10% of rows held out -> mixed prediction (new unit-year cells)."""
    rep_rng = np.random.default_rng(seed + rep)
    n = len(panel)
    test_idx = rep_rng.choice(n, size=int(n * test_fraction), replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    return panel.loc[~mask], panel.loc[mask]


def split_new_periods(
    panel: pd.DataFrame,
    rep: int,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int]]:
    """Hold out a block of years for ALL units -> pure forecasting.

    Also returns the exact set of years held out in this replication, so
    callers can check whether a crisis year landed in the test set.
    """
    years = sorted(panel["time"].unique())
    n_hold = max(1, int(len(years) * test_fraction))
    rep_rng = np.random.default_rng(seed + rep)
    start = rep_rng.integers(0, max(1, len(years) - n_hold))
    hold_years = set(years[start:start + n_hold]) if rep % 2 == 0 else set(years[-n_hold:])
    mask = panel["time"].isin(hold_years)
    return panel.loc[~mask], panel.loc[mask], hold_years


def split_new_periods_pair(
    panel: pd.DataFrame,
    rep: int,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Same as split_new_periods but only returns (train, test) for callers
    that don't need the held-out years (keeps the SPLIT_FUNCTIONS dict
    interface uniform)."""
    train, test, _ = split_new_periods(panel, rep, test_fraction, seed)
    return train, test


def split_new_units(
    panel: pd.DataFrame,
    rep: int,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire units (countries) -> pure prediction for new cross-sectional units."""
    units = panel["unit_id"].unique()
    rep_rng = np.random.default_rng(seed + rep)
    n_hold = max(1, int(len(units) * test_fraction))
    hold_units = rep_rng.choice(units, size=n_hold, replace=False)
    mask = panel["unit_id"].isin(hold_units)
    return panel.loc[~mask], panel.loc[mask]


SPLIT_FUNCTIONS = {
    "random_rows": split_random_rows,
    "new_periods": split_new_periods_pair,
    "new_units": split_new_units,
}
