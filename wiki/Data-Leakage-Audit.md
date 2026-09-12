---
title: "Data Leakage Audit"
category: "Data"
sources:
  - "raw/CODE_REVIEW_2026-08-26.md"
code_refs:
  - "src/preprocessing.py"
  - "src/config.py"
  - "tests/test_pipeline.py"
status: current
updated: 2026-08-26
tags: ["data-leakage", "deduplication", "temporal-split", "collinearity", "cicids2017"]
---

# Data Leakage Audit

## 📌 Executive Summary
The same pipeline scored **99.88% accuracy** under a random split and **0.64 accuracy with 0.00085 ATTACK recall** under a day-based split. That three-order-of-magnitude gap is the single most important fact about this dataset, and it is caused by leakage rather than by model quality.

---

## 🧠 The six checks

| Check | Present? | Mechanism |
| :--- | :--- | :--- |
| Train/val/test leakage | **Yes** | No validation set existed at all; thresholds were tuned on the test day |
| Scaling leakage | Partly | `StandardScaler` was clean; imputation and column selection were not |
| Label leakage | Partly | Ports correctly dropped; a stray `custom_sample.csv` was silently ingested |
| Duplicate samples | **Yes** | Global dedup ran, but keyed on `Day`, so cross-day twins survived |
| Incorrect random splitting | **Yes** | Random split of temporally bursty flows |
| Feature engineering errors | **Yes** | Seven exactly-collinear duplicate features retained |

### 1. Random splitting of temporally correlated flows

CIC-IDS2017 attacks arrive in dense bursts — a single `hulk.py` or `nmap` run emits thousands of near-identical flows within seconds. A random split scatters flows from the same burst, often the same tool invocation milliseconds apart, across train and test. The model is not detecting DoS; it is recognising *this particular DoS run*, which it has already seen tens of thousands of times.

### 2. Deduplication keyed on the `Day` column

`drop_duplicates()` with no `subset=` ran **after** `Day` was appended, so two byte-identical flows captured on Tuesday and Wednesday both survived and could land on opposite sides of a split.

```python
feature_cols = [c for c in df.columns if c not in (LABEL_COL, DAY_COL)]
before = len(df)
df = df.drop_duplicates(subset=feature_cols, keep="first")
print(f"Removed {before - len(df):,} duplicate flows")   # REPORT THIS NUMBER
```

Measured on the real `Thursday-WorkingHours-Morning-WebAttacks` capture alone: **14,466 duplicate flows removed, 8.49% of that day.**

### 3. Seven exactly-collinear feature pairs

Verified byte-identical over 400,000 rows of the project's own preprocessed dataset:

| Feature A | Feature B | Mismatching rows / 400k |
| :--- | :--- | ---: |
| `Fwd Header Length` | `Fwd Header Length.1` | 0 |
| `Total Fwd Packets` | `Subflow Fwd Packets` | 0 |
| `Total Backward Packets` | `Subflow Bwd Packets` | 0 |
| `Total Length of Fwd Packets` | `Subflow Fwd Bytes` | 0 |
| `Total Length of Bwd Packets` | `Subflow Bwd Bytes` | 0 |
| `Fwd Packet Length Mean` | `Avg Fwd Segment Size` | 0 |
| `Bwd Packet Length Mean` | `Avg Bwd Segment Size` | 4 |

`Fwd Header Length.1` is pandas auto-renaming a duplicated CSV header; the rest are CICFlowMeter emitting one measurement under two names.

**This makes feature-importance figures uninterpretable.** A published table listing `Subflow Fwd Bytes` at 0.052 and `Total Length of Fwd Packets` at 0.037 as two findings is reporting one measurement whose importance was split arbitrarily by which twin each tree happened to sample. Feature count drops 69 → 62 once removed.

### 4. Fit-on-everything preprocessing

`StandardScaler` was handled correctly — fitted on train, applied to test. But three earlier operations were fitted on the full dataset before any split existed: the >50%-missing column threshold, the imputation median, and the constant-column list. Numerically small after `±inf → NaN`; methodologically fatal, and a reviewer stops reading at that line. All three now live in a `FrameCleaner` with `fit`/`transform`.

### 5. Stray file ingestion

`load_all_datasets()` globbed *every* `*.csv` in `data/`. `custom_sample.csv` — 115 rows derived from the **Friday DDoS capture**, with a `Label` column — matched no day filter, was tagged `Day="Custom"`, and so joined every random-split experiment while vanishing from both sides of every cross-day split. The dataset's composition depended on what happened to be sitting in a directory. Replaced with an explicit `DAY_FILES` manifest that raises on a missing file.

---

## ⚠️ Gotchas & Security Pitfalls

- **Report the deduplication count in any write-up.** Reviewers of CIC-IDS2017 work expect it, and its absence is read as the work not having been done.
- **Near-duplicates remain.** Two DoS Hulk flows differing by one microsecond in `Flow IAT Min` are distinct rows to `drop_duplicates` and indistinguishable to any classifier. Exact dedup is a floor, not a solution.
- **The `closedset` protocol is still optimistic.** A true group-aware split would need flow-burst identifiers that the dataset does not provide once `Timestamp` and `Flow ID` are dropped. Report it as an architecture comparison, never as a detection result.
- Three regression tests pin this: no row hash appears in two splits; the cleaner's medians come from `fit` data only; a flow duplicated across two days is removed rather than kept twice.

---

## 🔗 Related Topics (Wikilinks)
- [[Training-Protocol]] — The day-based split that replaced the random one.
- [[Dataset-CICIDS2017]] — Attack distribution and per-day composition.
- [[Shortcut-Learning-Port-Bias]] — The label-leakage vector that *was* handled correctly.
- [[Metric-Dilution-Traps]] — Why the honest post-fix numbers look worse.
- [[Index]] — Master Knowledge Graph Index.
