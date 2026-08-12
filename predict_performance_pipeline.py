"""
Prediction performance comparison for panel data.

Compares Naive, AR(1) per unit (with pooled-AR(1) fallback), and Random
Forest using a Monte Carlo random train/test split design, following the
ASEP-distribution comparison style of Figure 1 in Haupt, Schnurbus, and
Tschernig (2010).

Run:
    python predict_performance_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import N_REPLICATIONS, RESULTS_DIR
from src.data import build_features, download_panel_data
from src.metrics import asep
from src.models import ar1_predict, naive_predict, rf_predict
from src.plotting import ecdf_xy
from src.splits import split_random_rows

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_monte_carlo(panel: pd.DataFrame) -> pd.DataFrame:
    """Random train/test split repeated many times; ASEP per method per replication."""
    rows = []

    for rep in range(N_REPLICATIONS):
        train_df, test_df = split_random_rows(panel, rep)
        actual = test_df["target"].to_numpy()

        pred_naive = naive_predict(test_df)
        pred_ar1 = ar1_predict(train_df, test_df)
        pred_rf = rf_predict(train_df, test_df)

        rows.append({"replication": rep, "method": "naive", "asep": asep(actual, pred_naive)})
        rows.append({"replication": rep, "method": "ar1", "asep": asep(actual, pred_ar1)})
        rows.append({"replication": rep, "method": "random_forest", "asep": asep(actual, pred_rf)})

        if (rep + 1) % 20 == 0:
            print(f"Completed replication {rep + 1}/{N_REPLICATIONS}")

    return pd.DataFrame(rows)


def summarize_and_plot(asep_df: pd.DataFrame) -> None:
    asep_df.to_csv(RESULTS_DIR / "asep_results.csv", index=False)

    methods = asep_df["method"].unique()

    plt.figure(figsize=(8, 6))
    for method in methods:
        values, ecdf_y = ecdf_xy(asep_df.loc[asep_df["method"] == method, "asep"].to_numpy())
        plt.plot(values, ecdf_y, label=method)

    plt.xlabel("ASEP")
    plt.ylabel("Empirical distribution function")
    plt.title("Prediction error distribution by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "asep_ecdf.png", dpi=150)
    plt.close()

    pivot = asep_df.pivot(index="replication", columns="method", values="asep")

    win_rows = []
    for a in methods:
        for b in methods:
            if a == b:
                continue
            win_rate = float(np.mean(pivot[a] < pivot[b]))
            win_rows.append({"method_a": a, "method_b": b, "a_beats_b_rate": win_rate})
    pd.DataFrame(win_rows).to_csv(RESULTS_DIR / "win_rates.csv", index=False)

    test_rows = []
    for a in methods:
        for b in methods:
            if a >= b:
                continue
            t_stat, p_val = stats.ttest_rel(pivot[a], pivot[b])
            test_rows.append({
                "method_a": a,
                "method_b": b,
                "mean_asep_a": pivot[a].mean(),
                "mean_asep_b": pivot[b].mean(),
                "t_stat": t_stat,
                "p_value": p_val,
            })
    pd.DataFrame(test_rows).to_csv(RESULTS_DIR / "mean_asep_test.csv", index=False)

    print("\nSaved:")
    print(" -", RESULTS_DIR / "asep_results.csv")
    print(" -", RESULTS_DIR / "asep_ecdf.png")
    print(" -", RESULTS_DIR / "win_rates.csv")
    print(" -", RESULTS_DIR / "mean_asep_test.csv")


def main() -> None:
    print("Downloading panel data...")
    panel = download_panel_data()

    print("Building features...")
    panel = build_features(panel)
    print(f"Panel ready: {len(panel)} rows, {panel['unit_id'].nunique()} units")

    print(f"Running {N_REPLICATIONS} Monte Carlo replications...")
    asep_df = run_monte_carlo(panel)

    print("Summarizing results and plotting...")
    summarize_and_plot(asep_df)


if __name__ == "__main__":
    main()
