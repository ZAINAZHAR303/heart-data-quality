"""Step 3 - Rule-based baseline (in the spirit of the Data Washing Machine).

Explicit, hand-written checks. Each rule writes an error label into a
cell-level prediction mask with the same format as the ground truth.

What the rules deliberately do NOT know about:
  - unit errors (there is no "wrong unit" rule; if a converted value falls
    outside the plausible range it is caught, but labelled out_of_range)
  - near-duplicates (only byte-for-byte identical records are found)
"""
import pandas as pd

from src.inject_errors import ALLOWED_CATEGORIES
from src.load_data import FEATURE_COLS, TARGET_COL

# Clinically plausible ranges for an adult cardiology patient, chosen by hand.
PLAUSIBLE_RANGES = {
    "age": (18, 100),        # years
    "trestbps": (80, 220),   # resting systolic blood pressure, mmHg
    "chol": (100, 600),      # serum cholesterol, mg/dL
    "thalach": (60, 220),    # maximum heart rate achieved, beats/min
    "oldpeak": (0, 7),       # ST depression induced by exercise, mm
}


def apply_rules(df):
    """Return a cell-level prediction mask: "" = looks fine, else the rule's label.

    Rules are applied from lowest to highest priority, so a specific cell-level
    finding (e.g. "missing") overwrites a row-level one (exact_duplicate).
    """
    pred = pd.DataFrame("", index=df.index, columns=FEATURE_COLS)

    # Rule 4 - exact duplicates: identical on every column except record_id.
    # Sort by record_id so the FIRST-entered copy is kept and later copies flagged.
    ordered = df.sort_values("record_id")
    is_dup = ordered.duplicated(subset=FEATURE_COLS + [TARGET_COL], keep="first")
    pred.loc[is_dup[is_dup].index, :] = "exact_duplicate"

    # Rule 3 - category check: value must be one of the allowed categories.
    for col, allowed in ALLOWED_CATEGORIES.items():
        bad = df[col].notna() & ~df[col].isin(allowed)
        pred.loc[bad, col] = "category_typo"

    # Rule 2 - range check: value must be inside the plausible range.
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        values = df[col]
        bad = values.notna() & ((values < low) | (values > high))
        pred.loc[bad, col] = "out_of_range"

    # Rule 1 - null check.
    pred[df[FEATURE_COLS].isna()] = "missing"

    pred.insert(0, "record_id", df["record_id"].to_numpy())
    return pred
