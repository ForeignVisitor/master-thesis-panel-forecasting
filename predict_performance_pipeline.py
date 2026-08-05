"""
Prediction performance comparison for panel data.

Compares Naive, AR(1) per unit, and Random Forest using a Monte Carlo
random train/test split design, following the ASEP-distribution comparison
style of Figure 1 in Haupt, Schnurbus, and Tschernig (2010).

Run:
    python predict_performance_pipeline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

RANDOM_SEED = 42
N_REPLICATIONS = 200
TEST_FRACTION = 0.10
MIN_TRAIN_OBS_PER_UNIT_AR1 = 5

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


def download_panel_data() -> pd.DataFrame:
    """Download World Bank GDP growth panel data (country x year)."""
    url = (
        "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.KD.ZG"
        "?format=json&per_page=20000"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    api_data = response.json()
    records = api_data[1]

    panel = pd.DataFrame(records)[["countryiso3code", "country", "date", "value"]].copy()
    panel["country"] = panel["country"].apply(
        lambda x: x["value"] if isinstance(x, dict) else x
    )
    panel.columns = ["unit_id", "country", "time", "target"]

    panel["time"] = pd.to_numeric(panel["time"], errors="coerce")
    panel["target"] = pd.to_numeric(panel["target"], errors="coerce")

    panel = panel.dropna(subset=["unit_id", "time", "target"]).copy()
    panel = panel[panel["unit_id"].str.len() == 3].copy()
    panel = panel.sort_values(["unit_id", "time"]).reset_index(drop=True)

    panel.to_csv(DATA_DIR / "raw" / "world_bank_gdp_growth_raw.csv", index=False)
    return panel


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lagged target and drop units with too few observations."""
    panel = panel.sort_values(["unit_id", "time"]).copy()
    panel["lag_1_target"] = panel.groupby("unit_id")["target"].shift(1)
    panel = panel.dropna(subset=["lag_1_target"]).copy()

    counts = panel.groupby("unit_id")["time"].nunique()
    valid_units = counts[counts >= 15].index
    panel = panel[panel["unit_id"].isin(valid_units)].copy()

    panel["unit_code"] = pd.factorize(panel["unit_id"])[0]
    panel = panel.sort_values(["unit_id", "time"]).reset_index(drop=True)

    panel.to_csv(DATA_DIR / "processed" / "panel_data.csv", index=False)
    return panel


def naive_predict(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["lag_1_target"].to_numpy()


def ar1_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Fit AR(1) per unit on training rows only; fall back to naive if too few obs."""
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
    return float(np.mean((actual - predicted) ** 2))


def run_monte_carlo(panel: pd.DataFrame) -> pd.DataFrame:
    """Random train/test split repeated many times; ASEP per method per replication."""
    n = len(panel)
    rows = []

    for rep in range(N_REPLICATIONS):
        rep_rng = np.random.default_rng(RANDOM_SEED + rep)
        test_idx = rep_rng.choice(n, size=int(n * TEST_FRACTION), replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True

        train_df = panel.loc[~mask]
        test_df = panel.loc[mask]

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
        values = np.sort(asep_df.loc[asep_df["method"] == method, "asep"].to_numpy())
        ecdf_y = np.arange(1, len(values) + 1) / len(values)
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
