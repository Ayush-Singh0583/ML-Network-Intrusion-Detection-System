# ML-NIDS pipeline

Rebuilt training / evaluation stack for CIC-IDS2017. Everything below runs from
the repository root.

## Install

```bash
pip install -r requirements.txt
# CPU-only torch:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick start

```bash
python src/run.py cache                                   # parse the CSVs once
python src/run.py train --model mlp  --protocol crossday  # open-set protocol
python src/run.py train --model rf   --protocol closedset # architecture table
python src/run.py compare --protocol closedset --models rf xgb lgbm mlp cnn lstm
python src/run.py train --model deep_svdd --pretrain-ae
python src/tune.py --model xgb --protocol crossday --n-iter 25
python -m pytest tests/ -v
```

Every run writes to `runs/<git-sha>-<timestamp>-<tag>/`:
`config.json`, `train.log`, `history.json`, `history.png`, `*_metrics.json`,
`*_per_class.csv`, `*_confusion_matrix.csv`, `*_confusion.png`, `results.json`.
Nothing is ever overwritten.

## Protocols

**`crossday`** — the open-set protocol.

```
train : Monday, Tuesday, Wednesday + 50% of Thursday (stratified)
val   : 50% of Thursday
test  : Friday  (DDoS, PortScan, Bot are genuinely unseen)
```

Thursday is split rather than held out whole because it is the *only* day
carrying Infiltration and the Web attacks. Holding all of it out leaves a
validation set whose only class is BENIGN, so macro-F1 is 1.0 by construction
and early stopping selects on noise. Friday is never touched by training or by
threshold calibration. Set `THURSDAY_VAL_FRAC = 1.0` in `src/config.py` for the
purist variant.

**`closedset`** — the architecture comparison table. Known classes only,
de-duplicated, stratified. It is *optimistic* relative to the temporal protocol
because bursty flows stay temporally adjacent; the run log says so. Report it as
a model comparison, never as a detection result.

## What each run reports

| Report | Meaning |
|---|---|
| `closed_set` | Known-class rows only, no rejection. On `crossday` this is BENIGN-only, because Friday has no known attack classes — that is a property of the dataset, not a bug. |
| `open_set` | Every test row; unseen classes folded into `Unknown_Attack`; rejection applied at a validation-calibrated τ. |
| `detection` | AUROC / AUPR / TPR@5%FPR of the rejection score against known-vs-unknown ground truth. **This is the number to lead with** — it is threshold-free. |
| `alt_scorers` | The same detection metrics for max-softmax and Mahalanobis-on-embeddings, so the choice of score is evidence rather than assertion. |

`macro_f1` averages over the union of true and predicted labels;
`macro_f1_present` averages only over classes that occur in the ground truth.
Quote one and say which.

## Open-set scores

Four are implemented in `src/openset.py`, all "higher = more novel":

- `msp_score` — 1 − max softmax. The weak baseline. A closed-set posterior
  cannot express "none of the above"; tree ensembles are *more* confident
  off-manifold because edge leaves are the purest.
- `energy_score` — `−T·logsumexp(logits/T)`. Retains logit magnitude. Default
  for the neural classifiers.
- `MahalanobisScorer` — distance to the nearest class-conditional Gaussian with
  a Ledoit-Wolf shrunk shared covariance. Model-agnostic; the only option for
  Random Forest, which has no logits.
- Reconstruction error (autoencoder) and squared distance to the hypersphere
  centre (Deep SVDD), both trained on BENIGN only.

τ is always `threshold_at_fpr(validation_scores, target_fpr)` — a quantile of
the **validation** scores. Nothing from the test day enters it.

## Deep SVDD

Rebuilt with `bias=False` on every `Linear` and `affine=False` on every
`BatchNorm1d`. A learnable additive term of any kind lets the encoder output a
constant and drive the objective to zero (Ruff et al., ICML 2018, Prop. 2) — the
previous checkpoint had collapsed onto exactly that solution.

Three guards, all enforced in code:

1. `DeepSVDD.__init__` asserts every `Linear.bias is None`.
2. `init_center` applies the ε-guard: any centre component with `|c_i| < 0.1`
   is pushed to `±0.1`.
3. `train_deep_svdd` raises if the validation distance std falls below `1e-4`.
   `tests/test_pipeline.py::test_collapse_guard_fires_on_a_biased_encoder`
   trains the old biased architecture until it collapses and asserts the raise.

Early stopping is deliberately **off** for Deep SVDD: a lower validation
distance is the signature of collapse, not of a better model, so selecting the
minimum picks epoch 1 every time. Fixed budget + collapse assertion instead.

`--pretrain-ae` warm-starts the encoder from an autoencoder (weights only, the
biases are discarded), which is what makes the centre meaningful.

## Files

```
src/
  config.py         every literal: paths, manifest, split days, hyper-parameters
  seeding.py        set_seed covering random/numpy/torch/cuda/cudnn
  preprocessing.py  manifest loader, structural clean, FrameCleaner(fit/transform),
                    SplitBundle, build_splits, Parquet cache
  datasets.py       TabularDataset, DeviceBatcher, build_loaders
  models.py         MLP(residual) · CNN1D · LSTMNet · Autoencoder · DeepSVDD
  losses.py         FocalLoss, tempered class weights
  engine.py         AdamW + cosine/warmup + AMP + grad clip + early stop +
                    best-checkpoint restore
  evaluation.py     single-ground-truth metrics, plots, run directories
  openset.py        scorers, τ calibration, AUROC/AUPR/OSCR
  training.py       sklearn baselines, save_bundle/load_bundle with integrity checks
  run.py            the CLI
  tune.py           hyper-parameter search on f1_macro
  predict.py        inference against a saved bundle
tests/
  test_pipeline.py  31 regression tests, one per fixed bug
```

The old entry points (`main*.py`, `train_svdd.py`, `compare_models.py`,
`tune_xgb*.py`, `evalAlone.py`, …) are now stubs that print why they were
replaced and what to run instead. The originals are preserved in
`_backup/src_pre_refactor_20260826/`.

## Reproducibility

`set_seed(42)` covers `random`, `numpy`, `torch`, `torch.cuda`,
`cudnn.deterministic` and `CUBLAS_WORKSPACE_CONFIG`. DataLoader workers get
per-worker seeds; `DeviceBatcher` draws its permutation from a seeded generator.
Verified: two runs of `--model mlp --protocol closedset --epochs 5` produce
bit-identical per-epoch losses.
