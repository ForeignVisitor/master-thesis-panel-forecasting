"""
Loader for Penn World Table (PWT) data -- GDP per capita and per-capita
employment, requested at the meeting for the real-data version of Part B
(and later, a real-data alternative for Part A too).

IMPORTANT -- run this ONE LOCALLY, not from the same place the rest of the
pipeline runs from a sandboxed environment. This script downloads a file
from dataverse.nl, which needs a normal internet connection.

There is no maintained official Python package for PWT (there's an R
package, `pwt10`, but nothing equivalent and current for Python), so this
downloads the actual data file PWT publishes and reads it with pandas.

What this does:
1. Downloads the PWT 11.0 Excel file (published Oct 2025) from the
   official PWT/Dataverse distribution.
2. Loads the "Data" sheet.
3. Prints every column name it finds -- PWT's variable names change a
   little between versions and I could not 100% confirm the employment
   column name from here, so the first run of this script is partly a
   "tell me what's actually in the file" step. Compare what prints against
   the PWT 11.0 user guide / codebook (linked on
   https://www.rug.nl/ggdc/productivity/pwt/?lang=en) if a variable below
   isn't found.
4. Builds a simple two-variable panel: GDP per capita and per-capita
   employment, one row per country-year, and saves it to
   data/processed/pwt_gdp_employment.csv in the same unit_id/country/time
   column shape the rest of the pipeline (src/data.py, src/splits.py etc)
   already expects, so it can be dropped straight into the existing
   pipeline scripts.

GDP per capita here is real GDP (output-side, chained PPPs -- "rgdpo" in
PWT's naming) divided by population ("pop"). Per-capita employment is
total employment ("emp") divided by population. If the "emp" column isn't
present under that name, the script prints the full column list so we can
find the right name and I'll adjust it -- see the fallback branch below.

Requires: pandas, openpyxl (`pip install openpyxl` if you don't have it).

Run:
    python load_pwt_gdp_employment.py
"""

from pathlib import Path

import pandas as pd
import requests

PROCESSED_DATA_DIR = Path("data/processed")
RAW_DATA_DIR = Path("data/raw")

# Official PWT 11.0 distribution file (Excel). If this URL has moved, get
# the current one from https://www.rug.nl/ggdc/productivity/pwt/?lang=en
# ("Latest version" -> download PWT 11.0 xlsx) and paste it in here.
PWT_URL = "https://dataverse.nl/api/access/datafile/554105"
RAW_XLSX_PATH = RAW_DATA_DIR / "pwt110.xlsx"

# Candidate column names -- PWT's naming is fairly stable but this makes it
# easy to fix in one place if a name doesn't match what actually downloads.
COL_UNIT_ID = "countrycode"
COL_COUNTRY = "country"
COL_TIME = "year"
COL_GDP = "rgdpo"     # real GDP, output-side, chained PPPs
COL_POP = "pop"       # population, millions
COL_EMP = "emp"       # number of persons engaged, millions -- UNCONFIRMED, verify on first run


def download_pwt_excel(force: bool = False) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_XLSX_PATH.exists() and not force:
        print(f"Already downloaded: {RAW_XLSX_PATH} (delete it, or pass force=True, to re-download)")
        return RAW_XLSX_PATH

    print(f"Downloading PWT data from {PWT_URL} ...")
    response = requests.get(PWT_URL, timeout=120)
    response.raise_for_status()
    RAW_XLSX_PATH.write_bytes(response.content)
    print(f"Saved: {RAW_XLSX_PATH} ({len(response.content) / 1e6:.1f} MB)")
    return RAW_XLSX_PATH


def load_and_inspect(xlsx_path: Path) -> pd.DataFrame:
    print("\nReading the 'Data' sheet (this can take a few seconds)...")
    raw = pd.read_excel(xlsx_path, sheet_name="Data")
    print(f"Loaded {len(raw)} rows, {len(raw.columns)} columns.")
    print("\nFull column list (compare against the PWT 11.0 codebook if something below is missing):")
    for col in raw.columns:
        print(" -", col)
    return raw


def build_gdp_employment_panel(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in (COL_UNIT_ID, COL_COUNTRY, COL_TIME, COL_GDP, COL_POP) if c not in raw.columns]
    if missing:
        raise KeyError(
            f"Expected column(s) not found in the downloaded file: {missing}. "
            "Check the printed column list above and update the COL_* constants "
            "at the top of this script to match."
        )

    if COL_EMP not in raw.columns:
        print(
            f"\nWARNING: employment column '{COL_EMP}' not found in this file. "
            "Building the panel with GDP per capita only -- check the column list "
            "above for the right employment column name (PWT sometimes ships "
            "employment in a separate 'Labor detail' file rather than the main "
            "table) and update COL_EMP once you find it."
        )
        panel = raw[[COL_UNIT_ID, COL_COUNTRY, COL_TIME, COL_GDP, COL_POP]].copy()
        panel["emp_per_capita"] = pd.NA
    else:
        panel = raw[[COL_UNIT_ID, COL_COUNTRY, COL_TIME, COL_GDP, COL_POP, COL_EMP]].copy()
        panel["emp_per_capita"] = panel[COL_EMP] / panel[COL_POP]

    panel = panel.rename(columns={COL_UNIT_ID: "unit_id", COL_COUNTRY: "country", COL_TIME: "time"})
    panel["gdp_per_capita"] = panel[COL_GDP] / panel[COL_POP]
    panel = panel.dropna(subset=["unit_id", "time", "gdp_per_capita"]).copy()
    panel = panel.sort_values(["unit_id", "time"]).reset_index(drop=True)

    # `target` = the single Y variable for Part A (GDP per capita). Kept as
    # a separate column name from `gdp_per_capita` so this file can be fed
    # straight into src/data.py's build_features() / the rest of the
    # existing pipeline without renaming anything.
    panel["target"] = panel["gdp_per_capita"]

    return panel[["unit_id", "country", "time", "gdp_per_capita", "emp_per_capita", "target"]]


def main() -> None:
    xlsx_path = download_pwt_excel()
    raw = load_and_inspect(xlsx_path)
    panel = build_gdp_employment_panel(raw)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "pwt_gdp_employment.csv"
    panel.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"{len(panel)} country-year rows, {panel['unit_id'].nunique()} countries, "
          f"years {int(panel['time'].min())}-{int(panel['time'].max())}")
    print(
        "\nNext step: this file has the same unit_id/country/time/target shape as "
        "data/processed/panel_data.csv, so it can be run through src/data.py's "
        "build_features() and the existing split/model/metric functions in src/ "
        "the same way the GDP-growth panel is -- swap the data source in a copy of "
        "one of the existing driver scripts (e.g. part_a_mean_vs_median.py) to try it."
    )


if __name__ == "__main__":
    main()
