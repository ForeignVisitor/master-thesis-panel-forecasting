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

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

RANDOM_SEED = 42
N_REPLICATIONS = 200
TEST_FRACTION = 0.10
MIN_TRAIN_OBS_PER_UNIT_AR1 = 5

DATA_PATH = Path("data/processed/panel_data.csv")
RESULTS_DIR = Path("results/split_type_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def naive_predict(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["lag_1_target"].to_numpy()


def ar1_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    preds = np.full(len(test_df), np.nan)
    test_df = test_df.reset_index(drop=True)

    for unit in test_df["unit_id"].unique():
        unit_train = train_df[train_df["unit_id"] == unit]
        unit_test_idx = test_df.index[test_df["unit_id"] == unit]

        if len(unit_train) < MIN_TRAIN_OBS_PER_UNIT_AR1:
            preds[unit_test_idx] = test_df.loc[unit_test_idx, "lag_1_target"].to_numpy()
            continue

        x = unit_train["lag_1_target"].to_numpy()
        y = unit_train["target"].to_numpy()
        x_mean, y_mean = x.mean(), y.mean()
        denom = np.sum((x - x_mean) ** 2)

        if denom == 0:
            preds[unit_test_idx] = test_df.loc[unit_test_idx, "lag_1_target"].to_numpy()
            continue

        phi = np.sum((x - x_mean) * (y - y_mean)) / denom
        alpha = y_mean - phi * x_mean
        x_test = test_df.loc[unit_test_idx, "lag_1_target"].to_numpy()
        preds[unit_test_idx] = alpha + phi * x_test

    return preds


def rf_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    features = ["lag_1_target", "unit_code", "time"]
    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(train_df[features], train_df["target"])
    return model.predict(test_df[features])


def asep(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = ~np.isnan(predicted)
    return float(np.mean((actual[valid] - predicted[valid]) ** 2))


def split_random_rows(panel: pd.DataFrame, rep: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_rng = np.random.default_rng(RANDOM_SEED + rep)
    n = len(panel)
    test_idx = rep_rng.choice(n, size=int(n * TEST_FRACTION), replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    return panel.loc[~mask], panel.loc[mask]


def split_new_periods(panel: pd.DataFrame, rep: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent years for ALL units -> pure forecasting."""
    years = sorted(panel["time"].unique())
    n_hold = max(1, int(len(years) * TEST_FRACTION))
    rep_rng = np.random.default_rng(RANDOM_SEED + rep)
    # small random jitter on which recent block to hold out, keeps it a real
    # Monte Carlo replication while staying a "future years" design
    start = rep_rng.integers(0, max(1, len(years) - n_hold))
    hold_years = set(years[start:start + n_hold]) if rep % 2 == 0 else set(years[-n_hold:])
    mask = panel["time"].isin(hold_years)
    return panel.loc[~mask], panel.loc[mask]


def split_new_units(panel: pd.DataFrame, rep: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire countries -> pure prediction for new cross-sectional units."""
    units = panel["unit_id"].unique()
    rep_rng = np.random.default_rng(RANDOM_SEED + rep)
    n_hold = max(1, int(len(units) * TEST_FRACTION))
    hold_units = rep_rng.choice(units, size=n_hold, replace=False)
    mask = panel["unit_id"].isin(hold_units)
    return panel.loc[~mask], panel.loc[mask]


SPLIT_FUNCTIONS = {
    "random_rows": split_random_rows,
    "new_periods": split_new_periods,
    "new_units": split_new_units,
}


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
    results.to_csv(RESULTS_DIR / "asep_by_design.csv", index=False)

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
    plt.savefig(RESULTS_DIR / "asep_ecdf_by_design.png", dpi=150)
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
    summary_df.to_csv(RESULTS_DIR / "asep_summary_by_design.csv", index=False)

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
    pd.DataFrame(test_rows).to_csv(RESULTS_DIR / "significance_by_design.csv", index=False)

    print("\nSaved:")
    print(" -", RESULTS_DIR / "asep_by_design.csv")
    print(" -", RESULTS_DIR / "asep_ecdf_by_design.png")
    print(" -", RESULTS_DIR / "asep_summary_by_design.csv")
    print(" -", RESULTS_DIR / "significance_by_design.csv")


def main() -> None:
    panel = pd.read_csv(DATA_PATH)
    print(f"Loaded panel: {len(panel)} rows, {panel['unit_id'].nunique()} units")

    results = run_all_designs(panel)
    summarize(results)


if __name__ == "__main__":
    main()
