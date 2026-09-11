"""
Part A, from last week's meeting: "just one variable Y" -- forget covariates
and Random Forest for now, and answer the simplest question first: if all you
have is a unit's own history of one variable, what would you guess for a new
observation? Compare the historical mean, the historical median, and (as an
extra, optional check) AR(1) over time. No Random Forest here on purpose --
RF only makes sense once we bring in a covariate X (that's Part B, later).
Also explicitly avoiding "dynamic panel data" estimators (no GMM /
Arellano-Bond) -- these are just simple per-unit summary predictors.

Two datasets, both single-variable (Y only):

1. REAL data -- the GDP-growth panel already in data/processed/panel_data.csv.
   GDP growth isn't really a "monetary" variable and isn't especially skewed,
   so mean and median should behave fairly similarly here. Useful as the
   plain baseline case.

2. SYNTHETIC "monetary" data -- built from scratch below, deliberately
   skewed like a real dollar/euro variable (e.g. revenue, income): every
   unit has a typical level, most years are close to it, but occasionally a
   unit has one unusually large year (a windfall, a one-off spike). This is
   the case the prof's notes point at directly: "maybe a dollar or euro
   variable, or create skewed data."

A statistical note on what to expect (worth being upfront about, rather than
just asserting "median wins under skew"): for plain squared-error loss
(ASEP), the *population* mean is always the best constant guess -- that's
true whether or not the data is skewed. What actually can make the median
win in practice is a finite-sample effect: with only a handful of
observations per unit and an occasional large outlier, the *sample* mean
is dragged around a lot by that one big value, while the *sample* median
barely moves. So the median can end up with lower average squared error
than the mean even though, in the infinite-data limit, the mean would win.
That's why this script reports both ASEP (squared error) and AAEP (average
absolute error, which the median targets directly) -- looking at both
together is more honest than picking whichever one makes a cleaner story.

Run:
    python part_a_mean_vs_median.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED, RESULTS_DIR
from src.data import build_features
from src.metrics import aaep, asep
from src.models import ar1_predict, mean_predict, median_predict
from src.plotting import ecdf_xy
from src.splits import split_random_rows

OUT_DIR = RESULTS_DIR / "part_a_mean_vs_median"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPLICATIONS = 200
METHODS = {
    "mean": mean_predict,
    "median": median_predict,
    "ar1": ar1_predict,
}

# --- synthetic "monetary" panel: deliberately skewed, one variable only ---
SYN_N_UNITS = 300
SYN_N_YEARS = 15
SYN_MIN_YEARS_PER_UNIT = 10  # keep all units after the lag drops 1 obs each

SYN_LEVEL_LOW, SYN_LEVEL_HIGH = 20.0, 500.0  # each unit's typical "euro" level
SYN_BASE_SIGMA = 0.25       # normal-year multiplicative noise (log scale)
SYN_OUTLIER_PROB = 0.06     # ~6% of unit-years are a one-off spike/windfall
SYN_OUTLIER_SIGMA = 1.4     # much wider spread for those spike years


def simulate_skewed_monetary_panel() -> pd.DataFrame:
    """One euro/dollar-style variable per unit, right-skewed with rare spikes.

    target_it = level_i * shock_it, shock_it lognormal and mean-1 so the
    variable stays positive (money can't go negative) and most years sit
    close to the unit's own typical level -- except the ~6% of years that
    get a much wider (still mean-1, so not systematically biased) shock,
    which produces the occasional large outlier a skewed monetary variable
    actually has (a windfall year, a one-off loss, etc).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    levels = rng.uniform(SYN_LEVEL_LOW, SYN_LEVEL_HIGH, size=SYN_N_UNITS)

    rows = []
    for unit in range(SYN_N_UNITS):
        level = levels[unit]
        for t in range(SYN_N_YEARS):
            is_outlier_year = rng.random() < SYN_OUTLIER_PROB
            sigma = SYN_OUTLIER_SIGMA if is_outlier_year else SYN_BASE_SIGMA
            # lognormal shock scaled to have mean 1, so it's not a biased guess
            z = rng.normal(0, sigma)
            shock = np.exp(z - (sigma ** 2) / 2)
            rows.append({
                "unit_id": f"FIRM{unit:04d}",
                "country": f"Simulated firm {unit}",
                "time": 2000 + t,
                "target": level * shock,
            })
    return pd.DataFrame(rows)


def run_monte_carlo(panel: pd.DataFrame, methods: dict) -> pd.DataFrame:
    rows = []
    for rep in range(N_REPLICATIONS):
        train_df, test_df = split_random_rows(panel, rep)
        actual = test_df["target"].to_numpy()

        for name, fn in methods.items():
            preds = fn(train_df, test_df)
            rows.append({
                "replication": rep,
                "method": name,
                "asep": asep(actual, preds),
                "aaep": aaep(actual, preds),
            })

        if (rep + 1) % 50 == 0:
            print(f"  completed replication {rep + 1}/{N_REPLICATIONS}")

    return pd.DataFrame(rows)


def summarize_and_plot(results: pd.DataFrame, out_dir: Path, title_suffix: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "mc_results.csv", index=False)

    summary = results.groupby("method")[["asep", "aaep"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)

    for metric, ylabel in [("asep", "ASEP (squared error)"), ("aaep", "AAEP (absolute error)")]:
        plt.figure(figsize=(7, 5))
        for method in results["method"].unique():
            values, y = ecdf_xy(results.loc[results["method"] == method, metric].to_numpy())
            plt.plot(values, y, label=method)
        plt.xlabel(ylabel)
        plt.ylabel("Empirical distribution function")
        plt.title(f"Part A -- {ylabel} ECDF{title_suffix}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"ecdf_{metric}.png", dpi=150)
        plt.close()

    print(summary.to_string(index=False))
    print("  saved:", out_dir / "mc_results.csv")
    print("  saved:", out_dir / "summary.csv")
    print("  saved:", out_dir / "ecdf_asep.png", "and ecdf_aaep.png")
    return summary


def main() -> None:
    # --- 1. real data ---
    print("=== Part A on REAL data (GDP growth panel) ===")
    real_panel = pd.read_csv(PROCESSED_DATA_DIR / "panel_data.csv")
    print(f"Loaded real panel: {len(real_panel)} rows, {real_panel['unit_id'].nunique()} units")
    real_results = run_monte_carlo(real_panel, METHODS)
    summarize_and_plot(real_results, OUT_DIR / "real_data", " (real GDP-growth panel)")

    # --- 2. synthetic skewed "monetary" data ---
    print("\n=== Part A on SYNTHETIC skewed monetary data ===")
    syn_panel = simulate_skewed_monetary_panel()
    syn_panel = build_features(
        syn_panel,
        processed_dir=OUT_DIR / "synthetic_data",
        min_years_per_unit=SYN_MIN_YEARS_PER_UNIT,
    )
    print(f"Simulated skewed panel: {len(syn_panel)} rows, {syn_panel['unit_id'].nunique()} units")
    syn_results = run_monte_carlo(syn_panel, METHODS)
    summarize_and_plot(syn_results, OUT_DIR / "synthetic_data", " (synthetic skewed monetary panel)")

    print(
        "\nReminder: ASEP (squared error) is minimized in the population by the mean, "
        "not the median -- so mean beating median on ASEP even under skew is not a bug. "
        "What to look for is whether the GAP between mean and median narrows (or flips) "
        "on the skewed synthetic panel vs. the real one, and how AAEP (absolute error) "
        "compares -- that's the honest version of the 'median is more robust' story."
    )


if __name__ == "__main__":
    main()
