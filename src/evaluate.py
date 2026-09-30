"""Step 5 - Score every detector against the ground truth and draw the chart.

Scoring units:
  - cell errors (missing, out_of_range, category_typo, unit_inconsistency)
    are scored per CELL
  - duplicate errors (exact_duplicate, near_duplicate) are scored per ROW;
    a row counts as predicted duplicate if more than half its cells say so
  - "any_error" is a row-level question ("is this record dirty at all?") and
    is the only one Isolation Forest can answer

Two recall numbers per type:
  recall      - found the error AND gave it the right label
  recall_any  - flagged the error with ANY label (e.g. a unit error caught by a
                range rule as out_of_range still counts here)

Run the whole experiment from the project root with:  python -m src.evaluate
"""
import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.inject_errors import CELL_ERROR_TYPES, ERROR_TYPES, ROW_ERROR_TYPES, error_summary, inject_errors
from src.load_data import DATA_DIR, FEATURE_COLS, RESULTS_DIR, SEED, load_clean
from src.ml_detectors import (
    KEY_COLS, build_cell_features, cell_labels, isolation_forest_rows,
    predictions_to_mask, split_by_record, train_random_forest,
)
from src.rules import apply_rules


def row_majority_label(mask):
    """Label each row with the error that covers more than half of its cells, else ""."""
    cells = mask[FEATURE_COLS]

    def majority(row):
        counts = row[row != ""].value_counts()
        return counts.index[0] if len(counts) and counts.iloc[0] > len(row) / 2 else ""

    return cells.apply(majority, axis=1)


def _scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return p, r, f


def score_by_type(truth, pred, method):
    """Precision / recall / F1 per error type for one detector (cell-level masks)."""
    t_cells, p_cells = truth[FEATURE_COLS].to_numpy(), pred[FEATURE_COLS].to_numpy()
    t_rows, p_rows = row_majority_label(truth), row_majority_label(pred)
    flagged_row = (pred[FEATURE_COLS] != "").any(axis=1).to_numpy()

    results = []
    for err in ERROR_TYPES:
        if err in CELL_ERROR_TYPES:
            y_true, y_pred = (t_cells == err).ravel(), (p_cells == err).ravel()
            flagged = (p_cells != "").ravel()
            level = "cell"
        else:
            y_true, y_pred = (t_rows == err).to_numpy(), (p_rows == err).to_numpy()
            flagged = flagged_row
            level = "row"
        p, r, f = _scores(y_true, y_pred)
        results.append({
            "method": method, "error_type": err, "level": level,
            "support": int(y_true.sum()), "precision": p, "recall": r, "f1": f,
            "recall_any": flagged[y_true].mean() if y_true.any() else np.nan,
        })
    return results


def score_rows(is_dirty, flagged, method, anomaly_score=None):
    """Row-level "is this record dirty?" scores."""
    p, r, f = _scores(is_dirty, flagged)
    return {
        "method": method, "error_type": "any_error", "level": "row",
        "support": int(is_dirty.sum()), "precision": p, "recall": r, "f1": f,
        "recall_any": r,
        "roc_auc": roc_auc_score(is_dirty, anomaly_score) if anomaly_score is not None else np.nan,
    }


def isolation_forest_by_type(truth, flagged):
    """Which kinds of dirty rows does Isolation Forest flag? (recall only:
    it gives no error type, so per-type precision is not defined)."""
    results = []
    for err in ERROR_TYPES:
        has_err = (truth[FEATURE_COLS] == err).any(axis=1).to_numpy()
        results.append({
            "method": "Isolation Forest", "error_type": err, "level": "row",
            "support": int(has_err.sum()), "precision": np.nan, "recall": np.nan,
            "f1": np.nan, "recall_any": flagged[has_err].mean() if has_err.any() else np.nan,
        })
    return results


def run_experiment(rate=0.05, seed=SEED, test_size=0.3, save=True, verbose=True):
    """Run the full pipeline once. Returns a dict with every intermediate result."""
    np.random.seed(seed)

    clean = load_clean(save=save)
    corrupted, cell_truth, row_truth = inject_errors(clean, rate=rate, seed=seed)
    if save:
        corrupted.to_csv(DATA_DIR / "heart_corrupted.csv", index=False)
        cell_truth.to_csv(DATA_DIR / "ground_truth_cells.csv", index=False)

    # Rules need no training: run them on the whole table.
    rule_mask = apply_rules(corrupted)

    # Random Forest: one example per cell, split by record.
    features = build_cell_features(corrupted)
    labels = cell_labels(features, cell_truth)
    X = features.drop(columns=KEY_COLS)
    train_idx, test_idx = split_by_record(features, test_size=test_size, seed=seed)
    rf = train_random_forest(X.iloc[train_idx], labels[train_idx], seed=seed)
    rf_pred = rf.predict(X.iloc[test_idx])
    test_rows = np.sort(features["row"].iloc[test_idx].unique())
    rf_mask = predictions_to_mask(features.iloc[test_idx], rf_pred, index=test_rows)

    # Isolation Forest: unsupervised, so it may look at every row (no labels used).
    iso_flags, iso_scores = isolation_forest_rows(corrupted, seed=seed)

    # ---- Score everything on the SAME held-out test rows ----
    truth_test = cell_truth.loc[test_rows]
    rules_test = rule_mask.loc[test_rows]
    dirty = row_truth.loc[test_rows, "is_dirty"].to_numpy()

    rows = score_by_type(truth_test, rules_test, "Rules")
    rows += score_by_type(truth_test, rf_mask, "Random Forest")
    rows.append(score_rows(dirty, (rules_test[FEATURE_COLS] != "").any(axis=1).to_numpy(), "Rules"))
    rows.append(score_rows(dirty, (rf_mask != "").any(axis=1).to_numpy(), "Random Forest"))
    rows.append(score_rows(dirty, iso_flags.loc[test_rows].to_numpy(), "Isolation Forest",
                           anomaly_score=iso_scores.loc[test_rows].to_numpy()))
    rows += isolation_forest_by_type(truth_test, iso_flags.loc[test_rows].to_numpy())
    table = pd.DataFrame(rows).round(3)

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        table.to_csv(RESULTS_DIR / "results_table.csv", index=False)

    if verbose:
        print("Injected errors:")
        print(error_summary(cell_truth).to_string(index=False))
        print(f"\nTest set: {len(test_rows)} of {len(corrupted)} rows "
              f"({int(dirty.sum())} dirty)\n")
        print(format_table(table))

    return {
        "clean": clean, "corrupted": corrupted, "cell_truth": cell_truth,
        "row_truth": row_truth, "rule_mask": rule_mask, "features": features,
        "labels": labels, "train_idx": train_idx, "test_idx": test_idx,
        "test_rows": test_rows, "rf": rf, "rf_pred": rf_pred, "rf_mask": rf_mask,
        "iso_flags": iso_flags, "iso_scores": iso_scores, "table": table,
    }


def format_table(table):
    """Pretty text version of the results table for the console."""
    cols = ["method", "error_type", "level", "support", "precision", "recall", "f1", "recall_any"]
    by_type = table[(table.error_type != "any_error") & (table.method != "Isolation Forest")]
    wide = by_type.pivot(index="error_type", columns="method",
                         values=["precision", "recall", "f1", "recall_any"]).reindex(ERROR_TYPES)
    wide.columns = [f"{m}:{metric}" for metric, m in wide.columns]
    wide = wide[[f"{m}:{k}" for m in ["Rules", "Random Forest"]
                 for k in ["precision", "recall", "f1", "recall_any"]]]
    wide.insert(0, "support", by_type.drop_duplicates("error_type").set_index("error_type")["support"])
    rows = table[table.error_type == "any_error"][cols[:7] + ["roc_auc"]]
    iso = table[(table.method == "Isolation Forest") & (table.error_type != "any_error")]
    return ("Per error type (test set; cells for cell errors, rows for duplicates):\n"
            + wide.to_string(float_format="%.2f")
            + "\n\nRow level - is the record dirty at all?\n"
            + rows.to_string(index=False, float_format="%.2f", na_rep="-")
            + "\n\nIsolation Forest - share of dirty rows flagged, by error type:\n"
            + iso[["error_type", "support", "recall_any"]].to_string(index=False, float_format="%.2f"))


def repeat_experiment(seeds=range(42, 52), rate=0.05, test_size=0.3):
    """Robustness check: re-run everything with different seeds (new injected
    errors AND a new train/test split each time) and average the scores.
    One test split holds only a handful of errors per type, so a single run's
    numbers can swing a lot; the mean over 10 runs is more trustworthy."""
    tables = []
    for seed in seeds:
        t = run_experiment(rate=rate, seed=seed, test_size=test_size, save=False, verbose=False)["table"]
        tables.append(t.assign(seed=seed))
    runs = pd.concat(tables, ignore_index=True)
    summary = (runs.groupby(["method", "error_type"], sort=False)
               .agg(support_total=("support", "sum"),
                    precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
                    f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                    recall_any_mean=("recall_any", "mean"),
                    roc_auc_mean=("roc_auc", "mean"))
               .round(3).reset_index())
    return summary


def plot_comparison(summary, path):
    """Grouped bars per error type: Rules vs Random Forest, recall (top) and F1 (bottom).

    Uses the 10-seed means from repeat_experiment(): a single test split holds
    only 1-7 errors per type, which is too few to draw a trustworthy chart.
    """
    by_type = summary[summary.error_type.isin(ERROR_TYPES)]
    methods = ["Rules", "Random Forest"]
    colors = {"Rules": "#8c8c8c", "Random Forest": "#2b6cb0"}
    x = np.arange(len(ERROR_TYPES))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, metric in zip(axes, ["recall_mean", "f1_mean"]):
        for k, method in enumerate(methods):
            vals = (by_type[by_type.method == method]
                    .set_index("error_type").loc[ERROR_TYPES, metric].to_numpy())
            bars = ax.bar(x + (k - 0.5) * width, vals, width, label=method, color=colors[method])
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("F1 (mean of 10 runs)" if metric == "f1_mean" else "Recall (mean of 10 runs)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Error detection on held-out test rows: rule-based checks vs. Random Forest\n"
                      "(mean over 10 seeds; each seed = new injected errors + new train/test split)")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[1].set_xticks(x, [e.replace("_", "\n") for e in ERROR_TYPES])
    axes[1].set_xlabel("Injected error type")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment()
    print("\nRobustness check: mean over 10 seeds (42-51)")
    repeated = repeat_experiment()
    repeated.to_csv(RESULTS_DIR / "results_repeated.csv", index=False)
    plot_comparison(repeated, RESULTS_DIR / "detection_comparison.png")
    print(repeated.to_string(index=False, na_rep="-"))
