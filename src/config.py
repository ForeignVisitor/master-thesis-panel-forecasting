"""Shared configuration constants used across all pipeline scripts."""

from pathlib import Path

RANDOM_SEED = 42
N_REPLICATIONS = 200
TEST_FRACTION = 0.10
MIN_TRAIN_OBS_PER_UNIT_AR1 = 5
CRISIS_YEARS = {2008, 2009, 2020}

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("results")
