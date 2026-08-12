"""Quick sanity check, not part of the actual pipeline.

Runs all three scripts' logic with just a few replications on the cached
panel data (no download, no waiting for the full 200-rep run) so I can
check nothing's broken after changing src/. Writes to results/_smoke_test/,
kept separate from the real results.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CRISIS_YEARS, PROCESSED_DATA_DIR
from src.metrics import asep
from src.models import ar1_predict, naive_predict, rf_predict
from src.splits import SPLIT_FUNCTIONS, split_new_periods

SMOKE_REPS = 5
OUT_DIR = Path("results/_smoke_test")
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

panel = pd.read_csv(PROCESSED_DATA_DIR / "panel_data.csv")
print(f"Loaded panel: {len(panel)} rows, {panel['unit_id'].nunique()} units")

# 1) predict_performance_pipeline.py logic (random_rows design)
rows = []
for rep in range(SMOKE_REPS):
    train_df, test_df = SPLIT_FUNCTIONS["random_rows"](panel, rep)
    actual = test_df["target"].to_numpy()
    rows.append({"replication": rep, "method": "naive", "asep": asep(actual, naive_predict(test_df))})
    rows.append({"replication": rep, "method": "ar1", "asep": asep(actual, ar1_predict(train_df, test_df))})
    rows.append({"replication": rep, "method": "random_forest", "asep": asep(actual, rf_predict(train_df, test_df))})
pd.DataFrame(rows).to_csv(OUT_DIR / "pipeline_asep_results.csv", index=False)
print("pipeline.py logic: OK ->", OUT_DIR / "pipeline_asep_results.csv")

# 2) predict_performance_by_split_type.py logic (all 3 designs)
rows = []
for design_name, split_fn in SPLIT_FUNCTIONS.items():
    for rep in range(SMOKE_REPS):
        train_df, test_df = split_fn(panel, rep)
        if len(test_df) == 0 or len(train_df) == 0:
            continue
        actual = test_df["target"].to_numpy()
        rows.append({"design": design_name, "replication": rep, "method": "naive",
                     "asep": asep(actual, naive_predict(test_df))})
        rows.append({"design": design_name, "replication": rep, "method": "ar1",
                     "asep": asep(actual, ar1_predict(train_df, test_df))})
        rows.append({"design": design_name, "replication": rep, "method": "random_forest",
                     "asep": asep(actual, rf_predict(train_df, test_df))})
by_design = pd.DataFrame(rows)
by_design.to_csv(OUT_DIR / "by_split_type_asep.csv", index=False)
print("by_split_type.py logic: OK ->", OUT_DIR / "by_split_type_asep.csv")

# Check: on new_units, AR(1) must now differ from naive (the bug is fixed)
nu = by_design[by_design["design"] == "new_units"].pivot(index="replication", columns="method", values="asep")
identical = np.allclose(nu["ar1"].to_numpy(), nu["naive"].to_numpy())
print(f"new_units: AR(1) identical to naive across smoke reps? {identical} "
      f"(expected False -- bug is fixed if False)")
print(nu)

# 3) timeseries_and_new_periods_diagnostics.py logic (new_periods + crisis flag)
rows = []
for rep in range(SMOKE_REPS):
    train_df, test_df, hold_years = split_new_periods(panel, rep)
    if len(test_df) == 0 or len(train_df) == 0:
        continue
    actual = test_df["target"].to_numpy()
    rows.append({
        "replication": rep,
        "contains_crisis_year": bool(hold_years & CRISIS_YEARS),
        "asep_naive": asep(actual, naive_predict(test_df)),
        "asep_ar1": asep(actual, ar1_predict(train_df, test_df)),
        "asep_rf": asep(actual, rf_predict(train_df, test_df)),
    })
diag = pd.DataFrame(rows)
diag.to_csv(OUT_DIR / "diagnostics_new_periods.csv", index=False)
print("diagnostics.py logic: OK ->", OUT_DIR / "diagnostics_new_periods.csv")

print("\nSMOKE TEST PASSED" if not identical else "\nSMOKE TEST FAILED: bug not fixed")
