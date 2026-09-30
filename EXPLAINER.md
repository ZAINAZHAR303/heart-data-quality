# EXPLAINER: how this project works and how to talk about it

This is a study guide. It walks through the code in the order it runs, explains *why* each choice was made, shows how to read the results, and lists questions a professor is likely to ask.

---

## 0. The one-paragraph story

Clinical datasets are full of quality problems: blanks, impossible values, typos, values recorded in the wrong unit, and the same patient entered twice. A rule-based cleaner like the Data Washing Machine (DWM) catches the problems its rules describe. This project asks: **can machine learning detect data-quality errors automatically, including ones the rules miss?** We take a clean public heart-disease dataset, inject errors whose location and type we know, and compare three detectors: hand-written rules, a supervised Random Forest that classifies every cell, and an unsupervised Isolation Forest that scores every row.

---

## 1. File by file

### `src/load_data.py`: get a *truly clean* starting point
1. **Download** the UCI Heart Disease *Cleveland* data (303 patients, 14 columns) with `ucimlrepo`, or fall back to the raw UCI file (`"?"` = missing). The download is cached in `data/heart_raw.csv` so later runs are offline and identical.
2. **Drop the 6 rows that already had missing values** (in `ca` and `thal`) → 297 rows. *Why:* if the "clean" table already had blanks, we could not tell our injected blanks from the original ones, and scores would be wrong.
3. **Turn codes into words** (`sex 1 → "male"`, `cp 4 → "asymptomatic"`, `thal 7 → "reversible_defect"`). *Why:* a typo in a number code makes no sense, but "mael" or "asymptomtic" is exactly what a clerk produces.
4. **Add `record_id`** (0…296), meaning "the order in which records entered the database". It stays glued to each row after shuffling, so predictions and ground truth always line up.
5. `SEED = 42` lives here and is imported everywhere, which is why every run gives identical numbers.

Column groups (used everywhere): continuous `NUMERIC_COLS` (age, trestbps, chol, thalach, oldpeak), small integer codes `CODED_COLS` (fbs, exang, slope, ca), string `CATEGORICAL_COLS` (sex, cp, restecg, thal), and the target `num`. The target is never corrupted, because this is about feature quality, not label noise.

### `src/inject_errors.py`: make dirty data *with an answer key*
`inject_errors(df, rate=0.05, seed=42)` injects `round(0.05 × 297) = 15` errors **of each type**:

| type | how it's made | what it is in a real hospital |
|---|---|---|
| `missing` | random cell → NaN (any column) | field left blank, lost in an interface, not measured |
| `out_of_range` | age 250 / −5 / 150 / 0, BP 900 / 0, chol 0 / 1500 / 2000, max HR 450 / 0, oldpeak −3 / 15 / 25 | typing slip (extra digit), default "0" placeholder, sensor glitch |
| `category_typo` | swap letters ("mael"), drop a letter ("asymptomtic"), case ("Male", "MALE"), stray space ("male ") | free-text entry, different systems' conventions, copy-paste |
| `unit_inconsistency` | chol ÷ 38.67 (mg/dL → mmol/L), age × 12 (years → months), BP ÷ 7.5 (mmHg → kPa) | data merged from sites/countries/devices with different units; pediatric systems recording age in months |
| `exact_duplicate` | copy a whole record | same visit submitted twice, a merge of two exports |
| `near_duplicate` | copy a record and change one thing (±1 on a measurement, or one typo) | same patient re-registered at another desk; the classic **entity-resolution** problem |

Design choices:
- **Each cell gets at most one error**, and duplicates are copied only from rows with no other error. Every cell then has exactly one true label, which keeps scoring unambiguous.
- **Duplicates get new, later `record_id`s.** The *second* entry is the duplicate; the original is clean.
- **Rows are shuffled** after injection, so duplicates are not sitting at the bottom of the file.
- **Out-of-range values were chosen not to overlap with unit-error values** (e.g. BP in kPa is 12–27, so no out-of-range BP value falls there). Otherwise the two types would be impossible to tell apart even in principle.
- Outputs: the corrupted table, a **cell-level mask** (same shape, `""` or the error name), and a **row-level truth** (`is_dirty`, `error_types`, `duplicate_of`). Duplicate rows carry their label in **every** cell, because the whole record is the error.

Result: 297 clean + 30 duplicate rows = **327 rows**, of which **90 are dirty** (about 27%).

### `src/rules.py`: the DWM-style baseline
Four explicit rules, applied from lowest to highest priority, so a specific cell finding overwrites a row-level one:
1. **Exact duplicate:** sort by `record_id`, then `df.duplicated(keep="first")` on every column except `record_id`. The later copy is flagged.
2. **Category check:** the value must be in the allowed set (`{"male", "female"}` …).
3. **Range check:** `PLAUSIBLE_RANGES` at the top of the file: age 18–100, BP 80–220, chol 100–600, max HR 60–220, oldpeak 0–7. These are hand-chosen adult cardiology ranges. None of the 297 clean patients violates them, so the rules raise no false alarms on clean data.
4. **Null check.**

There is deliberately **no unit rule and no near-duplicate rule**. That is realistic: you only write rules for problems you already know about. Unit errors that fall outside a range get caught, but *as* `out_of_range`, and we count that honestly (see `recall_any` below).

### `src/ml_detectors.py`: two ML approaches

**A) Isolation Forest (unsupervised, row level).**
- Build a numeric matrix: numbers median-imputed and scaled with **RobustScaler** (median/IQR; a mean/std scaler would be dominated by values like age = 648), categories one-hot encoded (so a typo becomes its own rare column), plus one **is-missing indicator per column** (so imputing does not hide the blank).
- Isolation Forest isolates points with random splits; points that are isolated quickly are anomalous.
- It gives a *score*, not a yes/no answer. We flag the **top 10%** of rows and call that a curator's review budget. It is an assumption, not the true error rate, which the model is never told. We also report **ROC-AUC**, which judges the ranking with no threshold at all.

**B) Random Forest error classifier (supervised, cell level).**
One training example **per cell** (327 rows × 13 columns = 4,251 cells). Features:

| feature | catches |
|---|---|
| `col_*` one-hot | which column this cell is in (a value of 6 is fine for oldpeak, not for chol) |
| `is_null` | missing |
| `value`, `robust_z` (median/MAD z-score) | extreme values |
| `dist_outside_range` | how far outside the plausible range (reuses rule knowledge) |
| `ratio_to_median` | unit errors: ≈ 1/38.67, ×12 or 1/7.5 of normal |
| `in_allowed_set`, `edit_dist_to_allowed`, `has_whitespace`, `case_mismatch`, `value_freq` | typos: rare, one or two edits from a real category |
| `n_exact_copies`, `n_earlier_copies` | exact duplicates (only the *later* copy has an earlier copy) |
| `nearest_earlier_dist` | near-duplicates: distance to the most similar record entered *before* this one (scaled numeric differences + number of differing categories) |

Target: `clean` or one of the 6 error types (multi-class).

- **Split by `record_id`** with `GroupShuffleSplit` (70% train / 30% test). See §3.
- **`class_weight="balanced"`**. See §3.
- `predictions_to_mask` turns per-cell predictions back into the same wide mask format as the ground truth, so rules and RF are scored by the same code.

### `src/evaluate.py`: scoring
- Cell errors are scored **per cell**. Duplicates are scored **per row**: a row counts as "predicted duplicate" if more than half of its cells say so.
- Every method is scored on the **same held-out test rows** (99 of 327 rows). Rules and Isolation Forest do not train on labels, but restricting them to the same rows keeps the comparison apples-to-apples.
- Two recalls: **recall** (found it *and* named the right type) and **recall_any** (flagged it with any label).
- Row level ("is this record dirty at all?"): all three methods, plus ROC-AUC for Isolation Forest.
- `repeat_experiment()` re-runs everything with seeds 42–51 (new errors *and* a new split each time) and averages. The chart uses these averages.
- `python -m src.evaluate` runs everything and writes all outputs.

### `notebooks/data_quality_demo.ipynb`
Tells the same story step by step using the functions above, then calls `run_experiment()` and asserts that its Random Forest predictions are **identical** to the step-by-step ones. This is a built-in reproducibility check.

---

## 2. Precision, recall, F1, in this setting

- **Precision** = of the cells the detector *flagged* as type X, how many really were X? Low precision means **false alarms**, so a curator wastes time on clean data.
- **Recall** = of the cells that really *were* type X, how many did the detector find? Low recall means **missed errors**, which flow silently into the downstream model.
- **F1** = harmonic mean of the two. It is only high if both are.
- In data curation, recall on dangerous errors (unit errors, duplicate patients) usually matters most. But precision decides whether people trust and keep using the tool.

---

## 3. Two design choices you must be able to defend

**Why split by `record_id` (GroupShuffleSplit), not by random cells? It prevents data leakage.**
All 13 cells of a record share the same row-level features. For example, every cell of a duplicated record has `n_earlier_copies = 1`. If cells were split randomly, 9 cells of a duplicate record could land in training and the other 4 in testing. The model would effectively have seen that record and be graded on it, which inflates the score. Grouping by record guarantees a test record is completely new to the model (the notebook asserts this). In a real deployment you score *new records*, so this is also the honest way to test.

**Why `class_weight="balanced"`? Because of class imbalance.**
Of 4,251 cells, 3,801 are clean and each cell error type has only 15 cells, roughly 250 clean cells per error. Without re-weighting, a forest can score ~89% accuracy by predicting "clean" for everything and never finding a single error. `balanced` weights each class inversely to its frequency, so a missed error costs as much as a lot of misread clean cells.

---

## 4. How to read the results

Numbers below are the **10-seed means** (`results/results_repeated.csv`). The single seed-42 split (`results/results_table.csv`) has only 1–7 test errors per type, which is too few to trust alone. On that split the Random Forest happens to be perfect on every type.

**Where rules win (or tie):**
- `missing` and `exact_duplicate`: rules F1 = **1.00**. These errors have a crisp definition, so a one-line rule is perfect and ML cannot improve on it. Say this openly: it shows you are not overselling ML.
- Rules need no labels, are transparent, and never over-fit.

**Where ML wins, and why:**
- `unit_inconsistency`: rules F1 **0.00** vs RF **0.96**. The rules *do* notice every unit error (recall_any = **1.00**, because 6.2 mmol/L is below the 100 mg/dL floor), but they call it `out_of_range`. The same mix-up drags rule precision for `out_of_range` down to **0.49** (half the "out of range" flags are really unit errors). The RF uses `ratio_to_median` (a value ≈ 1/38.67 of normal is a unit problem, not a typo) and names it correctly. **Why it matters:** the correct *repair* differs. A unit error should be converted and kept; an impossible value should be removed or re-measured.
- `near_duplicate`: rules F1 **0.00** vs RF **0.91** (recall 1.00, precision 0.85). Byte-level duplicate checks cannot see a record that differs by one character. Rules only stumble on the typo half of them, labelling one cell as a `category_typo` (recall_any 0.37). The RF's single most important feature is `nearest_earlier_dist` (importance 0.23 on seed 42): "this record is almost identical to one we already have". That is a baby version of **entity resolution**, which is exactly what DWM does with its blocking/matching steps. The false alarms come from genuinely similar *different* patients, which is the core difficulty of entity resolution.
- `category_typo`: rules 0.87 vs RF 0.99. The rules' extra false positives are near-duplicates carrying a typo: the typo is real, but the underlying problem is a duplicate record.

**The unsupervised result (the realistic setting):**
- Isolation Forest flags the 10% most anomalous rows and catches only **22%** of dirty rows (precision 0.60, F1 0.32). Its ranking is better than chance (ROC-AUC 0.76 on average). Part of the low recall is by design: 27% of rows are dirty but only 10% may be flagged, so recall can never exceed about 0.37.
- By type it mostly finds rows with blanks (0.52) and odd categories (0.38). It **never** flags an exact duplicate (0.00) and rarely a near-duplicate (0.15). **Why:** a duplicate of a normal patient *is* a normal patient. "Anomalous" and "erroneous" are not the same thing. Duplicates need comparison *between* records, not outlier scoring.

**Row level** ("is this record dirty at all?"): Rules recall **0.90** (they miss the near-duplicates), RF **1.00** (precision 0.98), Isolation Forest 0.22.

---

## 5. Likely professor questions, with short answers

1. **"Why not use real errors?"**
   Real error logs with cell-level ground truth are rare. Injected errors give an exact answer key, so precision and recall per error type can be measured. That is the standard first step in the error-detection literature. The cost is realism: real errors are correlated and subtler. The next step is a real dataset (e.g. MIMIC) with a manually audited sample.

2. **"Isn't the Random Forest cheating, since it trained on the injected labels?"**
   Partly, yes, and I say so in the README. It is supervised and tested on errors from the same generator, so it is an optimistic *upper bound*. Leakage is controlled (the split is by record), but the error *distribution* is shared. Isolation Forest is the no-label end of the spectrum. The interesting research space is in between: weak supervision (rules as noisy labels), active learning (a curator labels a few hundred cells), or training on synthetic errors and testing on real ones.

3. **"Why is Isolation Forest so weak?"**
   Two reasons. First, the 10% review budget caps recall at about 0.37 when 27% of rows are dirty. Second, it detects *unusual* rows, and many errors are not unusual. A duplicate record looks perfectly normal. Its ROC-AUC shows the ranking has signal, so it is useful to prioritise review but not as a standalone cleaner.

4. **"Why split by record_id?"**
   To avoid leakage: all cells of a record share row-level features, so splitting cells randomly would let the model see part of a test record during training. See §3.

5. **"Why `class_weight='balanced'`?"**
   Clean cells outnumber each error type about 250:1, so an unweighted model can look accurate while finding nothing. Balancing makes errors count.

6. **"Your single-split numbers are all 1.00. Isn't that suspicious?"**
   Yes, which is why I did not stop there. The test split has only 1–7 errors per type, so one lucky split proves little. Over 10 seeds with fresh errors and fresh splits, the RF drops to 0.91–0.97 F1 on the hard types, with a standard deviation around 0.1. Those are the numbers I report.

7. **"How would this scale?"**
   Cell features are per-column statistics, linear in data size. The Random Forest trains in seconds. The bottleneck is `nearest_earlier_dist`, which compares every pair of records (O(n²)): fine for 327 rows, not for millions. At scale you add **blocking** (only compare records that share a key such as birth year + sex, or use locality-sensitive hashing). That is exactly the blocking stage of an entity-resolution pipeline like DWM.

8. **"How does this relate to the Data Washing Machine / entity resolution?"**
   The rules module *is* a mini DWM-style rule layer, and the results show where it is sufficient (blanks, exact duplicates) and where it is not (unit errors, near-duplicates). The near-duplicate feature is a toy entity-resolution step. The natural combination is to keep DWM's rules and ER pipeline, and add an ML quality scorer that assigns every cell and record an error probability and type, so curators review the riskiest items first.

9. **"Why these plausible ranges? Aren't they arbitrary?"**
   They are hand-chosen adult cardiology bounds, and none of the 297 real patients violates them. They are arbitrary in the sense that every rule-based system is: someone has to choose them. That is one motivation for ML, which can *learn* per-column distributions instead. The RF still uses the ranges as a feature, so it builds on domain knowledge rather than replacing it.

10. **"What would you do next?"**
    (1) Replace the O(n²) nearest-record feature with proper blocking and pairwise matching (entity resolution). (2) Reduce label needs: use the rules as weak labels, or active learning. (3) Test on a real clinical dataset such as MIMIC with a manually audited sample of real errors. (4) Evaluate the *downstream* effect: does a model trained on the cleaned data predict heart disease better?

---

## 6. Three talking points for the meeting

1. **"Rules and ML are complementary, not competitors."** In my experiment, rules were already perfect on blanks and exact duplicates, but scored zero on unit errors and near-duplicates, where a learned model reached an F1 of about 0.9.
2. **"Detecting an error isn't enough; you need its *type* to fix it."** The rules caught every unit error but mislabelled it as out-of-range. The model learned the ratio signature (for example ÷38.67 for mg/dL to mmol/L), so the value can be converted instead of deleted.
3. **"The hard open problem is labels and duplicates."** The supervised model does well but needs labelled errors. The label-free Isolation Forest can't see duplicates at all, because a duplicate looks normal. That points to entity resolution plus weak supervision, which is where I'd like to build on the Data Washing Machine.
