"""
Diagnostics requested by supervisor:
1. Plot raw GDP growth time series to visually inspect volatility / crisis years.
2. Plot global average GDP growth per year with a volatility band.
3. Re-run the "new_periods" design while tracking which years were held out
   in each replication, to check whether the ASEP jump around 50-55 is caused
   by crisis years (e.g. 2008-09, 2020) landing in the holdout set.

Requires: data/processed/panel_data.csv (created by predict_performance_pipeline.py)

Run:
    python timeseries_and_new_periods_diagnostics.py
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.config import CRISIS_YEARS, N_REPLICATIONS, PROCESSED_DATA_DIR, RESULTS_DIR
from src.metrics import asep
from src.models import ar1_predict, naive_predict, rf_predict
from src.splits import split_new_periods

DATA_PATH = PROCESSED_DATA_DIR / "panel_data.csv"
OUT_DIR = RESULTS_DIR / "timeseries_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_country_time_series(panel: pd.DataFrame, n_countries: int = 12) -> None:
    """Plot GDP growth over time for a sample of countries."""
    sample_units = (
        panel.groupby("unit_id")["time"].count().sort_values(ascending=False)
        .head(n_countries).index
    )

    plt.figure(figsize=(11, 7))
    for unit in sample_units:
        sub = panel[panel["unit_id"] == unit].sort_values("time")
        label = sub["country"].iloc[0]
        plt.plot(sub["time"], sub["target"], marker="o", markersize=2, label=label)

    for year in CRISIS_YEARS:
        plt.axvline(year, color="red", linestyle="--", alpha=0.3)

    plt.xlabel("Year")
    plt.ylabel("GDP growth (%)")
    plt.title("GDP growth over time (sample of countries)\nRed dashed lines = known crisis years")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "country_time_series.png", dpi=150)
    plt.close()


def plot_global_average_with_band(panel: pd.DataFrame) -> None:
    """Plot the cross-country average GDP growth per year with a std-dev band."""
    yearly = panel.groupby("time")["target"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(11, 6))
    plt.plot(yearly["time"], yearly["mean"], color="black", label="Mean GDP growth")
    plt.fill_between(
        yearly["time"],
        yearly["mean"] - yearly["std"],
        yearly["mean"] + yearly["std"],
        alpha=0.2,
        label="+/- 1 std dev across countries",
    )

    for year in CRISIS_YEARS:
        plt.axvline(year, color="red", linestyle="--", alpha=0.5)

    plt.xlabel("Year")
    plt.ylabel("GDP growth (%)")
    plt.title("Global average GDP growth by year\nRed dashed lines = known crisis years")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "global_average_by_year.png", dpi=150)
    plt.close()

    yearly.to_csv(OUT_DIR / "yearly_mean_std.csv", index=False)


def run_new_periods_diagnostic(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for rep in range(N_REPLICATIONS):
        train_df, test_df, hold_years = split_new_periods(panel, rep)
        if len(test_df) == 0 or len(train_df) == 0:
            continue

        actual = test_df["target"].to_numpy()
        pred_naive = naive_predict(test_df)
        pred_ar1 = ar1_predict(train_df, test_df)
        pred_rf = rf_predict(train_df, test_df)

        contains_crisis = bool(hold_years & CRISIS_YEARS)

        rows.append({
            "replication": rep,
            "hold_years": sorted(hold_years),
            "min_hold_year": min(hold_years),
            "max_hold_year": max(hold_years),
            "contains_crisis_year": contains_crisis,
            "asep_naive": asep(actual, pred_naive),
            "asep_ar1": asep(actual, pred_ar1),
            "asep_rf": asep(actual, pred_rf),
        })

        if (rep + 1) % 50 == 0:
            print(f"Completed {rep + 1}/{N_REPLICATIONS}")

    return pd.DataFrame(rows)


def plot_asep_vs_crisis(diag_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    methods = ["asep_naive", "asep_ar1", "asep_rf"]
    titles = ["Naive", "AR(1)", "Random Forest"]

    for ax, method, title in zip(axes, methods, titles):
        colors = diag_df["contains_crisis_year"].map({True: "red", False: "steelblue"})
        ax.scatter(diag_df["replication"], diag_df[method], c=colors, s=15)
        ax.set_title(title)
        ax.set_xlabel("Replication")

    axes[0].set_ylabel("ASEP")
    fig.suptitle(
        "ASEP per replication (new_periods design)\n"
        "Red = crisis year (2008-09 or 2020) included in the held-out years"
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "asep_vs_crisis_year.png", dpi=150)
    plt.close()

    diag_df.to_csv(OUT_DIR / "new_periods_diagnostic.csv", index=False)

    summary = diag_df.groupby("contains_crisis_year")[methods].mean()
    summary.to_csv(OUT_DIR / "asep_mean_by_crisis_flag.csv")
    print("\nMean ASEP with vs without a crisis year in the holdout:")
    print(summary)


def main() -> None:
    panel = pd.read_csv(DATA_PATH)
    print(f"Loaded panel: {len(panel)} rows, {panel['unit_id'].nunique()} units")

    print("Plotting country time series...")
    plot_country_time_series(panel)

    print("Plotting global average with volatility band...")
    plot_global_average_with_band(panel)

    print("Running new_periods diagnostic with crisis-year tracking...")
    diag_df = run_new_periods_diagnostic(panel)

    print("Plotting ASEP vs crisis-year flag...")
    plot_asep_vs_crisis(diag_df)

    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
