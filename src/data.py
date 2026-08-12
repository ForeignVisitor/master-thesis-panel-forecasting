"""Download and prepare the World Bank GDP-growth panel dataset."""

from pathlib import Path

import pandas as pd
import requests

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def download_panel_data(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Download World Bank GDP growth panel data (country x year)."""
    raw_dir.mkdir(parents=True, exist_ok=True)

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

    panel.to_csv(raw_dir / "world_bank_gdp_growth_raw.csv", index=False)
    return panel


def build_features(
    panel: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    min_years_per_unit: int = 15,
) -> pd.DataFrame:
    """Add lagged target and drop units with too few observations."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    panel = panel.sort_values(["unit_id", "time"]).copy()
    panel["lag_1_target"] = panel.groupby("unit_id")["target"].shift(1)
    panel = panel.dropna(subset=["lag_1_target"]).copy()

    counts = panel.groupby("unit_id")["time"].nunique()
    valid_units = counts[counts >= min_years_per_unit].index
    panel = panel[panel["unit_id"].isin(valid_units)].copy()

    panel["unit_code"] = pd.factorize(panel["unit_id"])[0]
    panel = panel.sort_values(["unit_id", "time"]).reset_index(drop=True)

    panel.to_csv(processed_dir / "panel_data.csv", index=False)
    return panel


def load_processed_panel(processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load the already-built processed panel (skips download + build_features)."""
    return pd.read_csv(processed_dir / "panel_data.csv")
