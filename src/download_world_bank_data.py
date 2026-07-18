from pathlib import Path
import pandas as pd
import requests

URL = (
    "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.KD.ZG"
    "?format=json&per_page=20000"
)

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

response = requests.get(URL, timeout=60)
response.raise_for_status()

api_data = response.json()
records = api_data[1]

panel = pd.DataFrame(records)

panel = panel[
    [
        "countryiso3code",
        "country",
        "date",
        "value",
    ]
].copy()

panel["country"] = panel["country"].apply(
    lambda x: x["value"] if isinstance(x, dict) else x
)

panel.columns = ["unit_id", "country", "time", "target"]

panel["time"] = pd.to_numeric(panel["time"], errors="coerce")
panel["target"] = pd.to_numeric(panel["target"], errors="coerce")

panel = panel.dropna(subset=["unit_id", "time", "target"]).copy()
panel = panel[panel["unit_id"].str.len() == 3].copy()
panel = panel.sort_values(["unit_id", "time"]).copy()

panel.to_csv(raw_dir / "world_bank_gdp_growth_raw.csv", index=False)

panel["lag_1_target"] = panel.groupby("unit_id")["target"].shift(1)

panel = panel.dropna(subset=["lag_1_target"]).copy()

unit_counts = panel.groupby("unit_id")["time"].nunique()
valid_units = unit_counts[unit_counts >= 20].index

panel = panel[panel["unit_id"].isin(valid_units)].copy()
panel = panel.sort_values(["unit_id", "time"]).reset_index(drop=True)

panel.to_csv(processed_dir / "panel_data.csv", index=False)

print("Saved raw data:", raw_dir / "world_bank_gdp_growth_raw.csv")
print("Saved processed data:", processed_dir / "panel_data.csv")
print("Rows:", len(panel))
print("Units:", panel["unit_id"].nunique())
print("Years:", panel["time"].min(), "to", panel["time"].max())
print()
print(panel.head(10))