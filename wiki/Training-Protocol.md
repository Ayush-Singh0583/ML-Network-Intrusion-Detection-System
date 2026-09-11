---
title: "Training Protocol and Split Design"
category: "Process"
sources:
  - "raw/CODE_REVIEW_2026-08-26.md"
code_refs:
  - "src/config.py"
  - "src/preprocessing.py"
  - "src/engine.py"
  - "src/seeding.py"
status: current
updated: 2026-08-26
tags: ["protocol", "train-val-test", "reproducibility", "early-stopping", "seeding", "amp"]
---

# Training Protocol and Split Design

## 📌 Executive Summary
The primary protocol is temporal: train on Monday–Wednesday plus half of Thursday, validate on the other half of Thursday, and touch Friday exactly once. A second, explicitly optimistic `closedset` protocol exists only to compare architectures. Every run is seeded, logged to its own directory, and never overwrites a previous result.

---

## 🧠 The two protocols

### `crossday` — the real result

```
train : Monday, Tuesday, Wednesday + 50% of Thursday (stratified)
val   : 50% of Thursday
test  : Friday  (DDoS, PortScan, Bot genuinely unseen)
```

**Why Thursday is split rather than held out whole.** A literal Mon–Wed / Thu / Fri split fails, and the failure is silent: Thursday is the *only* day carrying Infiltration, Web Brute Force, XSS and SQL Injection. Hold all of it out and those classes are absent from training, so they are dropped from validation, leaving a validation set whose only class is BENIGN — macro-F1 is then 1.0 by construction and early stopping selects on noise. This was caught because the first training run printed `val_macroF1 = 1.000000` at epoch 2.

Friday is still never touched by training or by threshold calibration. `THURSDAY_VAL_FRAC = 1.0` restores the purist variant and now prints a warning explaining what it costs.

### `closedset` — the architecture comparison

Known classes only, de-duplicated, stratified. Optimistic relative to the temporal protocol because bursty flows stay temporally adjacent ([[Data-Leakage-Audit]]). The run log says so. Report it as a model comparison, never as a detection result.

---

## 📐 Reproducibility

`random_state=42` on sklearn estimators fixes the split and the tree construction and **does nothing for PyTorch**. Nothing seeded weight initialisation, DataLoader shuffle order, or cuDNN algorithm selection, so two runs produced two different networks with no record of which made the checkpoint.

```python
def set_seed(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic matmul
```

`CUBLAS_WORKSPACE_CONFIG` is the line people miss: without it CUDA matmuls stay non-deterministic even with `manual_seed`. Verified bit-identical per-epoch losses across two runs.

---

## 📐 The training loop

| Mechanism | Setting |
| :--- | :--- |
| Optimiser | AdamW, weight decay excluded from 1-D params (norm gains, biases) |
| Schedule | Cosine with 3 warmup epochs |
| Gradient clipping | 1.0, applied **after** `scaler.unscale_()` |
| Mixed precision | fp16 autocast on CUDA |
| Early stopping | macro-F1 over classes present in validation, patience 12 |
| Checkpointing | Saved only on improvement; best state restored at the end |

Two details that fail silently if got wrong:

- **`scaler.unscale_(optimizer)` must precede `clip_grad_norm_`.** Clipping scaled gradients clips the wrong quantity, and nothing errors.
- **Loss must be accumulated weighted by batch size**, not batch count. The original loop printed the *sum* over batches while discarding the computed mean, which is how [[Hypersphere-Collapse]] went unnoticed.

### Where early stopping is deliberately off

Deep SVDD. The only unsupervised signal is validation distance, and a smaller distance *is* the degenerate solution — selecting the minimum picked epoch 1 every time. Fixed epoch budget plus a collapse assertion instead. The reason is printed in the training log so it does not get "fixed" later.

---

## ⚠️ Gotchas & Security Pitfalls

- **Set `--epochs` to what you will actually run.** The cosine schedule is computed over `cfg.epochs`; early-stopping at 30 out of a declared 60 means the LR was still near base when the selected checkpoint was written, and the model never reached the annealing phase.
- **Class-rebalanced sampling is off by default.** On a 12-class ablation at CIC-IDS2017 imbalance it *traded* rather than improved: macro-F1 0.857 → 0.785, rare-class recall 0.665 → 0.751. That ablation was not collapsing, so it cannot say what the sampler does to a model that is. **Turn it on when the per-class validation table shows classes with zero recall** — that is the collapse it exists to fix.
- **Weight decay on norm gains and biases was wrong but negligible.** Measured 0.870 → 0.870 macro-F1. Fixed anyway; recorded here so nobody re-investigates it as a suspected cause.
- **Every run writes to `runs/<git-sha>-<timestamp>-<tag>/`** and nothing is overwritten. The previous pipeline wrote `results/classification_report.csv` at a fixed path, so each experiment destroyed the last one's evidence — the file on disk was from a different experiment than the README described.

---

## 🔗 Related Topics (Wikilinks)
- [[Data-Leakage-Audit]] — Why the temporal split is mandatory.
- [[Metric-Dilution-Traps]] — The validation metric that selects the checkpoint.
- [[Hypersphere-Collapse]] — The one model where early stopping is disabled.
- [[Rejection-Scoring]] — What the validation day is used to calibrate.
- [[Machine-Learning-Models]] — Estimator settings and artifact bundles.
- [[Index]] — Master Knowledge Graph Index.
