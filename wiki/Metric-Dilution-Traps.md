---
title: "Metric Dilution Traps"
category: "ML"
sources:
  - "raw/CODE_REVIEW_2026-08-26.md"
code_refs:
  - "src/evaluation.py"
  - "src/engine.py"
status: current
updated: 2026-08-26
tags: ["metrics", "macro-f1", "evaluation", "sklearn", "model-selection", "debugging"]
---

# Metric Dilution Traps

## 📌 Executive Summary
Three separate failures in this project were misread as model collapse when they were arithmetic. A macro average is a division, and **what you divide by is a choice sklearn makes for you by default**. On a dataset where one class dominates, that default can move a reported score by an order of magnitude — and in one case it inverted the checkpoint-selection ranking, actively steering the model toward predicting fewer classes.

---

## 🧠 Trap 1: macro average over the *union* of true and predicted labels

`sklearn.metrics.f1_score(y_true, y_pred, average="macro")` with no `labels=` argument averages over `unique_labels(y_true, y_pred)` — the **union**. Every class the model predicted but which does not occur in the ground truth enters the average as a `0.0` term.

### Where it hit: the closed-set report

Under the day-based protocol, the closed-set report covers Friday rows whose true class is *known*. Friday's only known class is BENIGN — DDoS, PortScan and Bot are all novel. So the report has one true class and a label list built from the union with the predictions:

```
accuracy                     : 0.9847
n labels in union            : 10
macro_f1  (union)            : 0.0992    <- read as catastrophic
macro_f1_present (true only) : 0.9923    <- actual performance
```

`f1(BENIGN)` = 2·1.0·0.9847 / 1.9847 = 0.9923, divided by 10 labels = 0.0992. **Reproduced exactly from arithmetic alone.** The union average is not *wrong* — predicting a class that never occurs is a real error — it is unreadable when the subset has one true class.

### Where it hit worse: checkpoint selection

The training loop used the same defaulted call to pick the best epoch. Validation is half of Thursday, which carries 5 of 12 known classes, so **the metric fell as the model got bolder**:

| Checkpoint | Union labels | Diluted macro-F1 *(selected on)* | Correct macro-F1 |
| :--- | ---: | ---: | ---: |
| cautious — 45% attack recall, 1 spurious class | 6 | **0.5592 ← picked** | 0.6711 |
| bolder — 80% attack recall, 7 spurious classes | 12 | 0.3778 | **0.9068 ← better** |

**The ranking inverts.** Early stopping on the diluted number prefers whichever checkpoint predicts the fewest distinct classes — which is a drift toward the majority class. The fix is one argument:

```python
VAL_LABELS = np.unique(y_val)                       # computed once, before the loop
macro_f1 = f1_score(y_val, y_pred, average="macro",
                    labels=VAL_LABELS, zero_division=0)
```

Both variants are now reported: `macro_f1` (union) and `macro_f1_present` (ground-truth classes only). **Quote one and say which.**

---

## 🧠 Trap 2: two different ground truths in one report

The original `evaluate_model` computed accuracy against `y_true_open` (unseen classes folded into `Unknown_Attack`) and precision/recall/F1 against `y_true` (raw labels), then built the confusion matrix from `y_true_open` again. In the random-split experiment those arrays are identical, so the bug is invisible; in the cross-day experiment they differ on every novel row. **One printout described three different problems.**

The rule that replaced it: every metric in a report, the per-class table and the confusion matrix all come from exactly two string arrays and one `labels` list, and a class present in the data but absent from `labels` raises rather than being silently dropped.

---

## 🧠 Trap 3: silent NumPy string truncation

```python
>>> enc = LabelEncoder().fit(["BENIGN", "ATTACK"])
>>> p = np.array(enc.inverse_transform([0,1,0])).astype(str); p.dtype
dtype('<U6')
>>> p[0] = "Unknown_Attack"; p
array(['Unknow', 'BENIGN', 'ATTACK'], dtype='<U6')
```

`.astype(str)` produces a **fixed-width** array sized to the longest existing class name. Assigning a longer string truncates it without warning, creating a phantom class that matches nothing in the ground truth. Dormant in the 12-class case (`WebAttack_SQLInjection` is 22 chars); fires immediately on the binary head. Fix: `dtype=object`, which has no width.

---

## ⚠️ Gotchas & Security Pitfalls

- **`balanced_accuracy_score` and `matthews_corrcoef` return `0.0` with a `UserWarning` when only one class is present.** A `0.0` reads as a real score. Return `nan` instead.
- **A single number cannot distinguish "uniformly mediocre" from "perfect on the majority, zero on everything else".** Print a per-class validation table with support, and warn explicitly on any class with zero recall.
- **Track distinct classes predicted per epoch.** A model emitting one or two classes on a 12-class validation set is degenerate whatever its accuracy says. This is now a column in the training log and an automatic warning.
- **Report support alongside every per-class number.** Heartbleed (n≈11) and SQL Injection (n≈21) have per-class F1 confidence intervals roughly ±0.3 wide; an unqualified "Heartbleed F1 = 1.00" reads as a claim rather than as three test samples.
- **Model selection must optimise the metric you report.** This project's tuners scored `f1_weighted` and `accuracy` while its own documentation called weighted metrics a trivial-classifier trap.

---

## 🔗 Related Topics (Wikilinks)
- [[Model-Evaluation-Metrics]] — The metric definitions this page corrects the use of.
- [[Rejection-Scoring]] — Where AUROC below 0.5 was similarly misread as noise.
- [[Training-Protocol]] — Checkpoint selection and the validation day.
- [[Data-Leakage-Audit]] — Why the honest numbers are lower than the originals.
- [[Index]] — Master Knowledge Graph Index.
