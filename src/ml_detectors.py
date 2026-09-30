"""Step 4 - Machine-learning detectors.

A) Isolation Forest (UNSUPERVISED, row level)
   Needs no labels. Learns what "typical" records look like and flags records
   that are easy to isolate. Answers "is this ROW suspicious?" but not "which
   cell" or "what kind of error".

B) Random Forest error classifier (SUPERVISED, cell level)
   Learns from the injected labels. One training example per CELL, described by
   hand-made features; the target is the error type ("clean" or one of six).
   Honest caveat: it is trained on errors produced by the same generator it is
   tested on, which gives it an advantage over rules and over Isolation Forest.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

from src.inject_errors import ALLOWED_CATEGORIES
from src.load_data import CATEGORICAL_COLS, CODED_COLS, FEATURE_COLS, NUMERIC_COLS, SEED
from src.rules import PLAUSIBLE_RANGES

NUMBER_COLS = NUMERIC_COLS + CODED_COLS  # every column that holds a number


# ---------------------------------------------------------------------------
# A) Isolation Forest
# ---------------------------------------------------------------------------
def isolation_forest_matrix(df):
    """Turn the table into a purely numeric matrix for Isolation Forest.

    Imputation is only for the model (Isolation Forest cannot take NaN). We add
    one is-missing indicator per column so that blanks are still visible.
    RobustScaler (median/IQR) is used because extreme values like age=648
    would squash a mean/std scaler.
    """
    numbers = df[NUMBER_COLS].astype(float)
    numbers = numbers.fillna(numbers.median())
    numbers = RobustScaler().fit_transform(numbers)

    categories = df[CATEGORICAL_COLS].apply(lambda s: s.fillna(s.mode()[0]))
    one_hot = pd.get_dummies(categories).astype(float).to_numpy()

    missing = df[FEATURE_COLS].isna().astype(float).to_numpy()
    return np.hstack([numbers, one_hot, missing])


def isolation_forest_rows(df, review_fraction=0.10, seed=SEED):
    """Return (flags, scores) per row. flags=True means "this row looks anomalous".

    Isolation Forest only produces a ranking (an anomaly score). To turn it into
    yes/no flags we must pick a threshold. We frame it as a curator's review
    budget: "flag the 10% most suspicious records". This is an assumption, NOT
    the true error rate (which the model is never told). The threshold-free
    ROC-AUC reported in evaluate.py measures the ranking itself.
    """
    X = isolation_forest_matrix(df)
    model = IsolationForest(n_estimators=300, contamination=review_fraction, random_state=seed)
    model.fit(X)
    flags = pd.Series(model.predict(X) == -1, index=df.index)
    scores = pd.Series(-model.score_samples(X), index=df.index)  # higher = more anomalous
    return flags, scores


# ---------------------------------------------------------------------------
# B) Random Forest cell-level error classifier
# ---------------------------------------------------------------------------
def levenshtein(a, b):
    """Minimum number of single-character edits to turn string a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def row_features(df):
    """Features describing the whole ROW, shared by all of its cells.

    n_exact_copies        - how many other rows are identical to this one
    n_earlier_copies      - how many identical rows were entered BEFORE this one
                            (so the original is not flagged, only the re-entry)
    nearest_earlier_dist  - distance to the most similar earlier-entered row;
                            ~0 means "this looks like a re-entry of an existing
                            patient" (a simple form of entity resolution)
    """
    cells = df[FEATURE_COLS]
    key = cells.astype(str).fillna("NA").agg("|".join, axis=1)
    n_exact_copies = key.map(key.value_counts()) - 1
    n_earlier_copies = df.groupby(key)["record_id"].rank(method="first") - 1

    # Distance = sum of scaled numeric differences + number of differing categories.
    numbers = cells[NUMBER_COLS].astype(float)
    numbers = numbers.fillna(numbers.median())
    iqr = numbers.quantile(0.75) - numbers.quantile(0.25)
    scale = iqr.where(iqr > 0, numbers.std()).replace(0, 1)
    z = ((numbers - numbers.median()) / scale).to_numpy()
    cats = cells[CATEGORICAL_COLS].astype(str).fillna("NA").to_numpy()

    dist = np.abs(z[:, None, :] - z[None, :, :]).sum(axis=2)
    dist += (cats[:, None, :] != cats[None, :, :]).sum(axis=2)

    ids = df["record_id"].to_numpy()
    earlier = ids[None, :] < ids[:, None]            # earlier[i, j]: row j entered before row i
    dist = np.where(earlier, dist, np.inf)
    nearest = dist.min(axis=1)
    nearest[np.isinf(nearest)] = np.nan               # the very first record has no earlier row
    nearest = np.nan_to_num(nearest, nan=np.nanmax(nearest))

    return pd.DataFrame({
        "n_exact_copies": n_exact_copies.to_numpy(),
        "n_earlier_copies": n_earlier_copies.to_numpy(),
        "nearest_earlier_dist": nearest,
    }, index=df.index)


def column_features(df, col):
    """Features describing each cell of one column."""
    values = df[col]
    f = pd.DataFrame(index=df.index)
    f["is_null"] = values.isna().astype(int)
    f["value_freq"] = values.map(values.value_counts(normalize=True)).fillna(0)

    if col in NUMBER_COLS:
        x = values.astype(float)
        median = x.median()
        mad = (x - median).abs().median()
        spread = 1.4826 * mad if mad > 0 else (x.std() or 1.0)
        f["value"] = x.fillna(0)
        f["robust_z"] = ((x - median) / spread).fillna(0)
        f["ratio_to_median"] = (x / median).fillna(0) if median != 0 else 0.0
        if col in PLAUSIBLE_RANGES:
            low, high = PLAUSIBLE_RANGES[col]
            outside = (low - x).clip(lower=0) + (x - high).clip(lower=0)
            f["dist_outside_range"] = (outside / (high - low)).fillna(0)
        else:
            f["dist_outside_range"] = 0.0
        # Category-only features are "not applicable" here.
        f["in_allowed_set"] = 1
        f["edit_dist_to_allowed"] = 0
        f["has_whitespace"] = 0
        f["case_mismatch"] = 0
    else:
        allowed = ALLOWED_CATEGORIES[col]
        text = values.fillna("")
        f["value"] = 0.0
        f["robust_z"] = 0.0
        f["ratio_to_median"] = 0.0
        f["dist_outside_range"] = 0.0
        f["in_allowed_set"] = values.isin(allowed).astype(int)
        f["edit_dist_to_allowed"] = text.map(
            lambda v: 0 if v == "" else min(levenshtein(v, a) for a in allowed))
        f["has_whitespace"] = (text != text.str.strip()).astype(int)
        f["case_mismatch"] = (text != text.str.lower()).astype(int)
    return f


def build_cell_features(df):
    """One row per (record, column) cell. Returns a long table with keys + features.

    Features only use the (corrupted) data itself, never the labels, so it is
    fine to compute column statistics on the whole table.
    """
    per_row = row_features(df)
    blocks = []
    for col in FEATURE_COLS:
        f = column_features(df, col)
        f = pd.concat([f, per_row], axis=1)
        for c in FEATURE_COLS:                     # which column is this cell in? (one-hot)
            f[f"col_{c}"] = int(c == col)
        f.insert(0, "column", col)
        f.insert(0, "record_id", df["record_id"].to_numpy())
        f.insert(0, "row", df.index.to_numpy())
        blocks.append(f)
    return pd.concat(blocks, ignore_index=True)


KEY_COLS = ["row", "record_id", "column"]


def cell_labels(features, cell_truth):
    """Look up the ground-truth label of every cell in the long feature table."""
    long = cell_truth[FEATURE_COLS].stack()          # index = (row, column)
    labels = long.reindex(pd.MultiIndex.from_arrays([features["row"], features["column"]]))
    return labels.replace("", "clean").to_numpy()


def split_by_record(features, test_size=0.3, seed=SEED):
    """Train/test split where all cells of a record land on the SAME side.

    If cells of one record were split across train and test, the model would
    see that record's row features (e.g. "this row is a copy") during training
    and be graded on them in testing -> leakage and inflated scores.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(features, groups=features["record_id"]))
    return train_idx, test_idx


def train_random_forest(X_train, y_train, seed=SEED):
    """class_weight="balanced": clean cells outnumber each error type ~50:1, so
    without re-weighting the forest would happily predict "clean" everywhere."""
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def predictions_to_mask(features, predicted, index):
    """Convert per-cell predictions back to a wide mask (rows x columns) like the ground truth."""
    long = pd.DataFrame({"row": features["row"].to_numpy(),
                         "column": features["column"].to_numpy(),
                         "pred": np.where(predicted == "clean", "", predicted)})
    wide = long.pivot(index="row", columns="column", values="pred")
    return wide.reindex(index=index, columns=FEATURE_COLS)
