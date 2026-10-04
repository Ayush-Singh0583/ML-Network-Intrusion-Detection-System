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
python src/run.py cache                                   # parse the CSVs once (--rebuild to start over)
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

> **Added 2026-10-04.** `build_splits` has three more protocols, used by the
> study and not exposed through `run.py train`:
>
> | protocol | what it is |
> |---|---|
> | `random` | all five days pooled, 70/15/15 at random — the leaky baseline, there to be compared against |
> | `blocked` | per (capture, class), in time order: earliest 60% train, next 20% validation, last 20% test |
> | `loco` | `blocked` with one attack class (`holdout_class=`) removed from train and validation; all of it goes to test |
>
> `crossday` also accepts `val_mode="tail"`: the earlier flows of each Thursday
> class train and the later ones validate, instead of a random half.
>
> In these three and in `crossday`, only the training split is de-duplicated;
> validation and test are raw flows. `dedup_train=False` skips even that and
> returns the training days as recorded.

## The study (`src/study.py`)

The experiments behind the paper. One command per research question; each
writes `runs/<git-sha>-<timestamp>-study-<name>/` with its tables (`*.csv`),
figures (`*.png`, `*.pdf`), `log.txt`, `config.json` and `results.json`.

```bash
python src/study.py doctor          # describe the CSVs in data/ -- run first, read it
python src/study.py e1              # protocol effect: random / blocked / crossday
python src/study.py e2              # each attack class seen versus held out
python src/study.py e3              # benign-only detectors, deep and classical
python src/study.py e4              # the per-source behaviour layer
python src/study.py e5              # three layers under one false-alarm budget
python src/study.py e6              # does a threshold survive a new day
python src/study.py all             # doctor, e1..e6 (one process each), report;
                                    # stops after doctor if it lists a problem
python src/study.py all --only e3   # re-run one experiment
python src/study.py report          # newest full runs that follow the plan -> paper/tables,
                                    # paper/figures, paper/SOURCES.md
```

Useful flags (all experiments): `--quick` (subsampled smoke run, never a
result; plain `report` ignores it, and `all --quick` writes to `paper/quick/`
only), `--models xgb` (skip the Random Forest), `--seeds N`, `--epochs N`,
`--device cpu|cuda`, `--n-boot N`.

Requirements and behaviour worth knowing:

- **Data.** E4 and the behaviour layer of E5 need source address and timestamp,
  which only the `GeneratedLabelledFlows` CSVs carry (`data/Readme.md`). With
  the other download, `all` skips E4 and E5 fuses two layers.
- **`all` stops when `doctor` lists a problem** (exit code 2, before the first
  experiment): a missing capture, a mixture of the two downloads, timestamps
  that cannot be read (`time=UNREADABLE`), that fall on another date, or that
  parse outside 08:00-18:00. `--despite-problems` runs anyway, and
  `paper/SOURCES.md` then repeats the problems next to the tables.
- **The rule every table follows.** A threshold is fitted on validation benign
  scores for a false-alarm budget (0.1%, 0.5%, 1%); the table gives the share
  of each attack class above it on the test split, a block-bootstrap interval,
  and the false-alarm rate the threshold actually produced on test benign rows.
- **Nothing is tuned.** XGBoost and Random Forest use `config.XGB_PARAMS` and
  `config.RF_PARAMS`. The deep models use the `TrainConfig` defaults except
  that the study caps them at 30 epochs with patience 10 (`study.train_config`).
  The Isolation Forest settings (`config.IFOREST_PARAMS`) were written with
  the study and never adjusted. The choices that could otherwise be made after
  seeing the results are fixed in `paper/PROTOCOL.md`.
- **Three verdicts are computed.** RQ3, RQ5 and RQ6 have a yes/no rule in the
  protocol; `src/rules.py` applies it and the experiment writes
  `e3_rq3_verdict`, `e5_rq5_verdict`, `e6_rq6_verdict`.
- **Seeds.** E1, E3 and E5 refit every model with seeds 42, 43, 44. E2 and E6
  use seed 42 only.
- **Read `n_blocks` before an interval.** It is the number of one-minute blocks
  a class's flows fall in. Below 20 the interval is printed and not interpreted;
  with one block it has zero width. Every detection and paired table says so
  itself in a true/false column, `interval_counts`.
- **`report` takes, for each experiment, the newest complete run that follows
  the plan** (by the time stamp in the folder name; the command line's defaults
  are the plan's values). A newer run with other settings does not replace it
  and is named as passed over; if no run follows the plan the newest is taken
  and named "not run as planned", with the settings that differ. For an
  experiment it has a run for, it replaces the files its previous report wrote
  into `paper/tables` and `paper/figures`. For one whose run folder is gone it
  keeps them and lists them as kept. A file it did not write is never deleted
  and is named at the end of `paper/SOURCES.md`, which lists every file with
  the run it came from and repeats what `doctor` said about the CSVs.
- **Paths can be redirected** with `NIDS_DATA_DIR`, `NIDS_CACHE_DIR`,
  `NIDS_RUNS_DIR`, `NIDS_ARTIFACT_DIR`, `NIDS_PAPER_DIR`. The tests use this to
  run the whole study on synthetic captures in a temporary folder.
  `tests/conftest.py` points all five at empty temporary folders before any
  test imports the code, so no test reads the CSVs in `data/` or writes into
  `cache/`, `runs/` or `paper/`, whatever those hold.

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
  study.py          the paper's experiments: doctor, e1..e6, all, report
  meta.py           identifier side-table (address, port, timestamp) kept out of the features
  layers.py         classifier / novelty / behaviour, each reduced to "a score per flow"
  behaviour.py      per-source window counts; replay through the live ScanTracker
  fusion.py         min-p and Bonferroni fusion; which layer caught what
  stats.py          budget thresholds, block bootstrap, detection and AUC tables
  rules.py          the yes/no rules of paper/PROTOCOL.md (RQ3, RQ5, RQ6) as code
  figures.py        the paper's figures
tests/
  test_pipeline.py        31 regression tests, one per fixed bug
  test_meta.py            timestamps, side-table, cache rebuild, split protocols
  test_behaviour_layer.py window counts, direction repair, replay
  test_stats_fusion.py    thresholds, bootstrap, fusion, split-editing helpers, report
  test_rules.py           the decision rules, on hand-built tables
  test_study_cli.py       the whole study end to end on synthetic captures
  synth_cic.py            writes CIC-IDS2017-shaped CSVs for the tests (not data)
paper/
  PROTOCOL.md       the analysis plan, fixed before the runs
CLAUDE.md           knowledge-base rules; its last section is the Decision Log
                    (each decision taken since 2026-10-04, with the reason)
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
