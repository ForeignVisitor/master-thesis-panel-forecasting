"""
Time-series cross-validation, requested at the meeting after reading Section 4.1
("Data generating processes") of Qu, Timmermann & Zhu (2024).

Two things from that meeting:

1. Instead of only random splits, evaluate AR(1) and Random Forest with a genuine
   expanding-window, one-step-ahead walk-forward design -- train on everything up
   to year Y, predict year Y+1, then move forward one year at a time. Mirrors the
   example from the meeting: train through year 46 -> predict 47, through 47 ->
   predict 48, ..., through 49 -> predict 50.

2. Try a strictly per-unit AR(1) (no pooled fallback at all) under this same
   design, and compare it to the current hybrid AR(1). The idea: in new_units
   the fallback had to cover *every* row because held-out countries start with
   zero history. Here nothing is held out -- every country already has years of
   its own data by the time we're predicting recent years -- so the fallback
   should barely ever trigger, and the two AR(1) versions should behave almost
   the same. If they don't, that's worth digging into.

Requires: data/processed/panel_data.csv

Run:
    python time_series_cross_validation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import MIN_TRAIN_OBS_PER_UNIT_AR1, PROCESSED_DATA_DIR, RESULTS_DIR
from src.metrics import asep
from src.models import ar1_predict, ar1_predict_per_unit_only, naive_predict, rf_predict
from src.splits import walk_forward_folds

DATA_PATH = PROCESSED_DATA_DIR / "panel_data.csv"
OUT_DIR = RESULTS_DIR / "time_series_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5  # last 5 years, one-step-ahead each -> mirrors the 46->47->...->50 example


def run_walk_forward(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for test_year, train_df, test_df in walk_forward_folds(panel, n_folds=N_FOLDS):
        actual = test_df["target"].to_numpy()

        pred_naive = naive_predict(test_df)
        pred_ar1_hybrid = ar1_predict(train_df, test_df)
        pred_ar1_per_unit = ar1_predict_per_unit_only(train_df, test_df)
        pred_rf = rf_predict(train_df, test_df)

        n_skipped = int(np.isnan(pred_ar1_per_unit).sum())

        for method, preds in [
            ("naive", pred_naive),
            ("ar1_hybrid", pred_ar1_hybrid),
            ("ar1_per_unit_only", pred_ar1_per_unit),
            ("random_forest", pred_rf),
        ]:
            rows.append({
                "test_year": test_year,
                "n_train_rows": len(train_df),
                "n_test_rows": len(test_df),
                "method": method,
                "asep": asep(actual, preds),
            })

        print(
            f"Year {test_year}: train={len(train_df)} rows -> test={len(test_df)} rows | "
            f"naive={asep(actual, pred_naive):.2f}  "
            f"ar1_hybrid={asep(actual, pred_ar1_hybrid):.2f}  "
            f"ar1_per_unit_only={asep(actual, pred_ar1_per_unit):.2f} "
            f"(skipped {n_skipped}/{len(test_df)} rows, <{MIN_TRAIN_OBS_PER_UNIT_AR1} training obs)  "
            f"rf={asep(actual, pred_rf):.2f}"
        )

    return pd.DataFrame(rows)


def summarize_and_plot(results: pd.DataFrame) -> None:
    results.to_csv(OUT_DIR / "walk_forward_asep_by_year.csv", index=False)

    summary = results.groupby("method")["asep"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(OUT_DIR / "walk_forward_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    for method in results["method"].unique():
        subset = results[results["method"] == method].sort_values("test_year")
        plt.plot(subset["test_year"], subset["asep"], marker="o", label=method)
    plt.xlabel("Test year")
    plt.ylabel("ASEP")
    plt.title("Walk-forward one-step-ahead ASEP by year\n(expanding training window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "walk_forward_asep_by_year.png", dpi=150)
    plt.close()

    print("\nAverage ASEP across folds:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(" -", OUT_DIR / "walk_forward_asep_by_year.csv")
    print(" -", OUT_DIR / "walk_forward_asep_by_year.png")
    print(" -", OUT_DIR / "walk_forward_summary.csv")


def main() -> None:
    panel = pd.read_csv(DATA_PATH)
    print(f"Loaded panel: {len(panel)} rows, {panel['unit_id'].nunique()} units")
    print(f"Running walk-forward validation over the last {N_FOLDS} years "
          f"(expanding window, one year predicted at a time)...")

    results = run_walk_forward(panel)
    summarize_and_plot(results)


if __name__ == "__main__":
    main()
