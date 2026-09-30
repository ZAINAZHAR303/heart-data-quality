"""Step 2 - Inject realistic, labelled data-quality errors into the clean table.

Because we inject the errors ourselves, we know exactly which cells are wrong
and why. That "ground truth" is what lets us score the detectors.

Six error types (each applied to about `rate` x n_rows cells or rows):
  missing            - a value is blank (NaN)
  out_of_range       - a physically impossible value (age=250, chol=0, ...)
  category_typo      - a misspelled / badly formatted category ("mael", "Male ")
  unit_inconsistency - right value, wrong unit (chol in mmol/L, age in months, BP in kPa)
  exact_duplicate    - a whole record entered twice
  near_duplicate     - a record entered twice with one small difference
"""
import numpy as np
import pandas as pd

from src.load_data import (
    CATEGORICAL_COLS, CATEGORY_MAPS, CODED_COLS, FEATURE_COLS, NUMERIC_COLS, SEED,
)

ERROR_TYPES = [
    "missing", "out_of_range", "category_typo",
    "unit_inconsistency", "exact_duplicate", "near_duplicate",
]
CELL_ERROR_TYPES = ERROR_TYPES[:4]  # errors that live in a single cell
ROW_ERROR_TYPES = ERROR_TYPES[4:]   # errors that concern a whole record

ALLOWED_CATEGORIES = {col: set(m.values()) for col, m in CATEGORY_MAPS.items()}

# Impossible values we plant. Chosen so they do not overlap with what a unit
# error produces (e.g. BP in kPa is ~12-27), keeping the two types distinct.
OUT_OF_RANGE_VALUES = {
    "age": [250, -5, 150, 0],
    "trestbps": [900, 0, 400, 300],
    "chol": [0, 1500, 2000],
    "thalach": [450, 0, 300],
    "oldpeak": [-3.0, 15.0, 25.0],
}

# Wrong-unit conversions: a plausible value recorded in a different unit.
UNIT_CONVERSIONS = {
    "chol": lambda v: round(v / 38.67, 1),    # mg/dL -> mmol/L   (240 -> 6.2)
    "age": lambda v: v * 12,                  # years -> months   (54 -> 648)
    "trestbps": lambda v: round(v / 7.5, 1),  # mmHg -> kPa       (130 -> 17.3)
}

# Numeric columns a near-duplicate may nudge by +-1.
NEAR_DUP_NUMERIC = ["age", "trestbps", "chol", "thalach"]


def make_typo(value, rng):
    """Return a realistic misspelling of a category that is NOT a valid category."""
    allowed = set().union(*ALLOWED_CATEGORIES.values())
    while True:
        kind = rng.choice(["swap", "drop", "case", "whitespace"])
        chars = list(value)
        if kind == "swap":        # male -> mael
            i = rng.integers(0, len(chars) - 1)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            typo = "".join(chars)
        elif kind == "drop":      # asymptomatic -> asymptomtic
            i = rng.integers(0, len(chars))
            typo = "".join(chars[:i] + chars[i + 1:])
        elif kind == "case":      # male -> Male / MALE
            typo = value.capitalize() if rng.random() < 0.5 else value.upper()
        else:                     # male -> "male " / " male"
            typo = value + " " if rng.random() < 0.5 else " " + value
        if typo != value and typo not in allowed:
            return typo


def inject_errors(df, rate=0.05, seed=SEED):
    """Corrupt a clean table and return (corrupted_df, cell_truth, row_truth).

    corrupted_df : shuffled table with errors and extra duplicate rows.
    cell_truth   : same shape as corrupted_df[FEATURE_COLS] (+ record_id); each
                   cell is "" if clean, else the name of the injected error.
                   Duplicate rows carry their duplicate label in every cell,
                   because the whole record is the error.
    row_truth    : one row per record: is_dirty, error_types, duplicate_of.
    """
    rng = np.random.default_rng(seed)
    data = df.copy().reset_index(drop=True)
    data[NUMERIC_COLS + CODED_COLS] = data[NUMERIC_COLS + CODED_COLS].astype(float)
    truth = pd.DataFrame("", index=data.index, columns=FEATURE_COLS)
    n_errors = max(1, round(rate * len(data)))

    def pick_clean_cells(columns, k):
        """Randomly choose k (row, column) cells that are not corrupted yet."""
        candidates = [(i, c) for i in data.index for c in columns if truth.at[i, c] == ""]
        chosen = rng.choice(len(candidates), size=k, replace=False)
        return [candidates[j] for j in chosen]

    # 1. missing: blank out random cells in any column.
    for i, col in pick_clean_cells(FEATURE_COLS, n_errors):
        data.at[i, col] = np.nan
        truth.at[i, col] = "missing"

    # 2. out_of_range: plant impossible values.
    for i, col in pick_clean_cells(list(OUT_OF_RANGE_VALUES), n_errors):
        data.at[i, col] = rng.choice(OUT_OF_RANGE_VALUES[col])
        truth.at[i, col] = "out_of_range"

    # 3. category_typo: misspell or mis-format a category.
    for i, col in pick_clean_cells(CATEGORICAL_COLS, n_errors):
        data.at[i, col] = make_typo(data.at[i, col], rng)
        truth.at[i, col] = "category_typo"

    # 4. unit_inconsistency: record a correct value in the wrong unit.
    for i, col in pick_clean_cells(list(UNIT_CONVERSIONS), n_errors):
        data.at[i, col] = UNIT_CONVERSIONS[col](data.at[i, col])
        truth.at[i, col] = "unit_inconsistency"

    # 5 + 6. duplicates: copy records that have no other error, so every row
    # carries exactly one kind of problem. Copies get NEW, later record_ids
    # (the duplicate is the record that was entered second).
    clean_rows = truth.index[(truth == "").all(axis=1)].to_numpy()
    sources = rng.choice(clean_rows, size=2 * n_errors, replace=False)
    next_id = int(data["record_id"].max()) + 1
    new_rows, new_truth, duplicate_of = [], [], {}

    for k, src in enumerate(sources):
        row = data.loc[src].copy()
        if k < n_errors:
            label = "exact_duplicate"
        else:
            label = "near_duplicate"
            if rng.random() < 0.5:  # nudge one measurement by +-1
                col = rng.choice(NEAR_DUP_NUMERIC)
                row[col] = row[col] + rng.choice([-1, 1])
            else:                   # one typo in one category
                col = rng.choice(CATEGORICAL_COLS)
                row[col] = make_typo(row[col], rng)
        row["record_id"] = next_id
        duplicate_of[next_id] = int(data.at[src, "record_id"])
        next_id += 1
        new_rows.append(row)
        new_truth.append(pd.Series(label, index=FEATURE_COLS))

    data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)
    truth = pd.concat([truth, pd.DataFrame(new_truth)], ignore_index=True)
    # Rows built from Series come back as generic objects; restore the dtypes.
    data[NUMERIC_COLS + CODED_COLS] = data[NUMERIC_COLS + CODED_COLS].astype(float)
    data["num"] = data["num"].astype(int)

    # Shuffle so duplicates are not simply sitting at the bottom of the file.
    order = rng.permutation(len(data))
    data = data.iloc[order].reset_index(drop=True)
    truth = truth.iloc[order].reset_index(drop=True)
    data["record_id"] = data["record_id"].astype(int)
    truth.insert(0, "record_id", data["record_id"].to_numpy())

    # Row-level ground truth.
    cells = truth[FEATURE_COLS]
    row_truth = pd.DataFrame({
        "record_id": truth["record_id"],
        "is_dirty": (cells != "").any(axis=1),
        "error_types": cells.apply(lambda r: ";".join(sorted(set(r) - {""})), axis=1),
        "duplicate_of": truth["record_id"].map(duplicate_of).astype("Int64"),
    })
    return data, truth, row_truth


def error_summary(cell_truth):
    """Count injected errors per type (cells for cell errors, rows for duplicates)."""
    cells = cell_truth[FEATURE_COLS]
    rows = []
    for err in ERROR_TYPES:
        if err in CELL_ERROR_TYPES:
            rows.append({"error_type": err, "unit": "cells", "count": int((cells == err).sum().sum())})
        else:
            rows.append({"error_type": err, "unit": "rows", "count": int((cells == err).all(axis=1).sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.load_data import DATA_DIR, load_clean

    corrupted, cell_truth, row_truth = inject_errors(load_clean())
    corrupted.to_csv(DATA_DIR / "heart_corrupted.csv", index=False)
    cell_truth.to_csv(DATA_DIR / "ground_truth_cells.csv", index=False)
    print(error_summary(cell_truth).to_string(index=False))
