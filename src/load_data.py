"""Step 1 - Load the UCI Heart Disease (Cleveland) dataset and build a clean baseline.

The raw data uses numeric codes for categories (e.g. sex=1). We convert the
coded categorical columns into readable strings ("male", "asymptomatic", ...)
so that injected typos ("mael", "asymptomtic") look like real data-entry errors.
"""
from pathlib import Path

import pandas as pd

SEED = 42  # one global seed used by every module and the notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Standard column names for the 14 Cleveland attributes.
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Coded categorical columns -> readable string categories.
CATEGORY_MAPS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical_angina", 2: "atypical_angina", 3: "non_anginal", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"},
    "thal": {3: "normal", 6: "fixed_defect", 7: "reversible_defect"},
}

# Column groups used throughout the project.
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]  # continuous measurements
CODED_COLS = ["fbs", "exang", "slope", "ca"]                      # small integer codes, kept numeric
CATEGORICAL_COLS = list(CATEGORY_MAPS)                            # readable string categories
TARGET_COL = "num"                                                # heart-disease label (never corrupted)
FEATURE_COLS = [c for c in COLUMNS if c != TARGET_COL]            # the 13 columns we corrupt / check


def fetch_raw(use_cache=True):
    """Return the raw 303-row Cleveland table (with its original missing values).

    Tries ucimlrepo first, then the direct UCI file. The result is cached in
    data/heart_raw.csv so later runs work offline and are identical.
    """
    cache = DATA_DIR / "heart_raw.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache)

    try:
        from ucimlrepo import fetch_ucirepo

        heart = fetch_ucirepo(id=45)
        raw = pd.concat([heart.data.features, heart.data.targets], axis=1)[COLUMNS]
        print("Loaded Cleveland data via ucimlrepo.")
    except Exception as err:  # network problems, API changes, ...
        print(f"ucimlrepo failed ({err!r}); falling back to {UCI_URL}")
        raw = pd.read_csv(UCI_URL, header=None, names=COLUMNS, na_values="?")

    DATA_DIR.mkdir(exist_ok=True)
    raw.to_csv(cache, index=False)
    return raw


def load_clean(save=True, use_cache=True):
    """Return the clean baseline: complete rows only, readable categories, a record_id."""
    raw = fetch_raw(use_cache=use_cache)

    # Drop the few rows that were already incomplete in the original data, so
    # every error in the corrupted version is one that WE injected and can score.
    df = raw.dropna().reset_index(drop=True)

    # Codes are stored as floats in the source (e.g. 1.0); make them ints first.
    int_cols = ["age", "trestbps", "chol", "thalach", *CODED_COLS, *CATEGORICAL_COLS, TARGET_COL]
    df[int_cols] = df[int_cols].astype(int)
    for col, mapping in CATEGORY_MAPS.items():
        df[col] = df[col].map(mapping)

    # record_id = order in which the record entered the "hospital database".
    # It stays attached to the row after shuffling so ground truth stays aligned.
    df.insert(0, "record_id", range(len(df)))

    if save:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(DATA_DIR / "heart_clean.csv", index=False)
    return df


if __name__ == "__main__":
    clean = load_clean()
    print(clean.shape)
    print(clean.head())
