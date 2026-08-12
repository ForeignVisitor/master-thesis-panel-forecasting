"""
Extends the ASEP-distribution comparison with three split designs:

1. random_rows    - random 10% of rows held out (mixed prediction)
2. new_periods    - most recent years held out for ALL countries (pure forecast)
3. new_units      - entire countries held out, never seen in training (pure prediction
                     for new cross-sectional units)

This tests directly whether "prediction" (new units) behaves differently from
"forecasting" (new time periods), which is the distinction your supervisor raised.

Requires: data/processed/panel_data.csv already created by
predict_performance_pipeline.py (run that first).

Run:
    python predict_performance_by_split_type.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import N_REPLICATIONS, PROCESSED_DATA_DIR, RESULTS_DIR
from src.metrics import asep
from src.models import ar1_predict, naive_predict, rf_predict
from src.splits import SPLIT_FUNCTIONS

DATA_PATH = PROCESSED_DATA_DIR / "panel_data.csv"
OUT_DIR = RESULTS_DIR / "split_type_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_all_designs(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for design_name, split_fn in SPLIT_FUNCTIONS.items():
        print(f"\nRunning design: {design_name}")

        for rep in range(N_REPLICATIONS):
            train_df, test_df = split_fn(panel, rep)

            if len(test_df) == 0 or len(train_df) == 0:
                continue

            actual = test_df["target"].to_numpy()

            pred_naive = naive_predict(test_df)
            pred_ar1 = ar1_predict(train_df, test_df)
            pred_rf = rf_predict(train_df, test_df)

            rows.append({"design": design_name, "replication": rep, "method": "naive",
                         "asep": asep(actual, pred_naive)})
            rows.append({"design": design_name, "replication": rep, "method": "ar1",
                         "asep": asep(actual, pred_ar1)})
            rows.append({"design": design_name, "replication": rep, "method": "random_forest",
                         "asep": asep(actual, pred_rf)})

            if (rep + 1) % 50 == 0:
                print(f"  {design_name}: completed {rep + 1}/{N_REPLICATIONS}")

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    results.to_csv(OUT_DIR / "asep_by_design.csv", index=False)

    designs = results["design"].unique()
    methods = results["method"].unique()

    fig, axes = plt.subplots(1, len(designs), figsize=(6 * len(designs), 5), sharey=True)
    if len(designs) == 1:
        axes = [axes]

    for ax, design in zip(axes, designs):
        subset = results[results["design"] == design]
        for method in methods:
            values = np.sort(subset.loc[subset["method"] == method, "asep"].to_numpy())
            if len(values) == 0:
                continue
            ecdf_y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, ecdf_y, label=method)
        ax.set_title(design)
        ax.set_xlabel("ASEP")
        ax.legend()

    axes[0].set_ylabel("Empirical distribution function")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "asep_ecdf_by_design.png", dpi=150)
    plt.close()

    summary_rows = []
    for design in designs:
        for method in methods:
            values = results.loc[
                (results["design"] == design) & (results["method"] == method), "asep"
            ]
            summary_rows.append({
                "design": design,
                "method": method,
                "mean_asep": values.mean(),
                "std_asep": values.std(),
                "n_replications": len(values),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "asep_summary_by_design.csv", index=False)

    test_rows = []
    for design in designs:
        pivot = results[results["design"] == design].pivot(
            index="replication", columns="method", values="asep"
        )
        for a in methods:
            for b in methods:
                if a >= b:
                    continue
                common = pivot[[a, b]].dropna()
                if len(common) < 2:
                    continue
                t_stat, p_val = stats.ttest_rel(common[a], common[b])
                test_rows.append({
                    "design": design,
                    "method_a": a,
                    "method_b": b,
                    "mean_a": common[a].mean(),
                    "mean_b": common[b].mean(),
                    "t_stat": t_stat,
                    "p_value": p_val,
                })
    pd.DataFrame(test_rows).to_csv(OUT_DIR / "significance_by_design.csv", index=False)

    print("\nSaved:")
    print(" -", OUT_DIR / "asep_by_design.csv")
    print(" -", OUT_DIR / "asep_ecdf_by_design.png")
    print(" -", OUT_DIR / "asep_summary_by_design.csv")
    print(" -", OUT_DIR / "significance_by_design.csv")


def main() -> None:
    panel = pd.read_csv(DATA_PATH)
    print(f"Loaded panel: {len(panel)} rows, {panel['unit_id'].nunique()} units")

    results = run_all_designs(panel)
    summarize(results)


if __name__ == "__main__":
    main()
