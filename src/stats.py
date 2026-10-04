"""
Detection tables with honest uncertainty.

WHY A BLOCK BOOTSTRAP
---------------------
Friday's test split holds 158,930 PortScan flows.  A confidence interval that
treats them as 158,930 independent draws is a few hundredths of a percent wide
and means nothing: they are one scanner run, and Engelen et al. and this
project's own leakage audit both show that flows inside a burst are
near-copies of each other.  The effective sample size is closer to the number
of minutes the scan ran than to the number of flows it produced.

So intervals here resample *time blocks* (one minute of one capture by
default), not flows.  All flows in a block enter or leave a replicate together.
Where a split has no timestamps the blocks are runs of consecutive rows in file
order, which is the same idea with a cruder clock.

WHAT THE INTERVAL COVERS -- AND WHAT IT DOES NOT
-----------------------------------------------
Thresholds are fixed before the bootstrap starts, exactly as they would be in
deployment.  The interval therefore describes sampling variability of the test
traffic *given this model and this threshold*.  It does not include the
variability of retraining (that is what the seed repeats are for) and it says
nothing about a different network.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import (
    BENIGN_LABEL,
    BOOTSTRAP_BLOCK_ROWS,
    BOOTSTRAP_BLOCK_SECONDS,
    BOOTSTRAP_REPLICATES,
    IDENTIFIER_COVERAGE,
    META_CAPTURE,
    META_ROW,
    META_TS,
    STUDY_BUDGETS,
)

ANY_ATTACK = "__ANY_ATTACK__"

# An interval resampled from fewer blocks than this is reported but not
# interpreted: with a handful of blocks the percentile interval is unstable,
# and with one block it has zero width.
MIN_BLOCKS_FOR_INTERVAL = 20


def interval_counts(n_blocks) -> bool:
    """Is an interval built on ``n_blocks`` blocks one the study interprets?

    Written into every table that carries an interval (column
    ``interval_counts``), so the reading rule travels with the number instead
    of living only in the protocol file.
    """
    try:
        return int(n_blocks) >= MIN_BLOCKS_FOR_INTERVAL
    except (TypeError, ValueError):
        return False


def difference_supported(lo, hi, n_blocks) -> bool:
    """The study's test for "A and B differ on this class".

    True only when the paired interval of the difference excludes zero AND
    the class spans at least ``MIN_BLOCKS_FOR_INTERVAL`` blocks.  The second
    condition is what stops a class confined to one minute -- whose interval
    has zero width and therefore "excludes zero" whenever the two rates are
    not identical -- from counting as evidence.
    """
    if not interval_counts(n_blocks):
        return False
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    return bool(lo > 0.0 or hi < 0.0)


# =====================================================================
# BLOCKS
# =====================================================================


def block_ids(
    meta: Optional[pd.DataFrame],
    n: int,
    block_seconds: float = BOOTSTRAP_BLOCK_SECONDS,
    fallback_rows: int = BOOTSTRAP_BLOCK_ROWS,
) -> np.ndarray:
    """One integer per row naming its resampling block.

    (capture, minute) when the split has timestamps; (capture, run of
    ``fallback_rows`` consecutive file rows) when it has row indices only; runs
    of consecutive positions when it has neither.
    """
    if meta is None or len(meta) != n:
        return (np.arange(n, dtype=np.int64) // int(fallback_rows))

    cap = (meta[META_CAPTURE].to_numpy(dtype=np.int64) if META_CAPTURE in meta.columns
           else np.zeros(n, dtype=np.int64))

    part = None
    if META_TS in meta.columns:
        ts = meta[META_TS].to_numpy(dtype=np.float64)
        if np.isfinite(ts).mean() >= IDENTIFIER_COVERAGE:
            t = pd.Series(ts).ffill().bfill().to_numpy()
            part = np.floor(t / float(block_seconds)).astype(np.int64)
    if part is None:
        row = (meta[META_ROW].to_numpy(dtype=np.int64) if META_ROW in meta.columns
               else np.arange(n, dtype=np.int64))
        part = row // int(fallback_rows)

    key = pd.MultiIndex.from_arrays([cap, part])
    codes, _ = pd.factorize(key)
    return codes.astype(np.int64)


# =====================================================================
# BOOTSTRAP
# =====================================================================


def bootstrap_rates(
    fired: np.ndarray,
    masks: Dict[str, np.ndarray],
    blocks: Optional[np.ndarray] = None,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Rate of ``fired`` inside each mask, with a block-bootstrap interval.

    ``masks`` maps a name (a class, or "BENIGN" for the false-alarm rate) to a
    boolean row mask.  Returns ``{name: {"rate", "lo", "hi", "n", "n_blocks"}}``.

    ``n_blocks`` is the number of distinct blocks that hold the mask's rows.
    READ IT BEFORE READING THE INTERVAL.  A class whose flows all sit in one
    block has no variation between blocks to resample: every replicate that
    draws the block returns the same rate, and the interval collapses to zero
    width.  That is not certainty, it is the absence of information.  The
    study's decision rules ignore intervals built on fewer than
    ``MIN_BLOCKS_FOR_INTERVAL`` blocks (``interval_counts``,
    ``difference_supported`` and ``rules.py`` apply that).

    The replicates are computed from per-block counts, so the cost does not
    depend on the number of flows: 1,000 replicates over a 700k-flow test day
    is a 1000 x n_blocks matrix product, not 1,000 passes over the data.
    """
    fired = np.asarray(fired, dtype=bool)
    n = fired.size
    out: Dict[str, Dict[str, float]] = {}

    if blocks is None or n_boot <= 0:
        for name, m in masks.items():
            m = np.asarray(m, dtype=bool)
            k = int(m.sum())
            rate = float(fired[m].mean()) if k else float("nan")
            out[name] = {"rate": rate, "lo": float("nan"), "hi": float("nan"), "n": k,
                         "n_blocks": (int(np.unique(np.asarray(blocks)[m]).size)
                                      if blocks is not None and k else 0)}
        return out

    blocks = np.asarray(blocks, dtype=np.int64)
    if blocks.size != n:
        raise ValueError(f"blocks has {blocks.size} entries for {n} rows")
    n_blocks = int(blocks.max()) + 1 if n else 0

    rng = np.random.default_rng(seed)
    # how many times each block is drawn in each replicate
    W = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_boot).astype(np.float64)

    for name, m in masks.items():
        m = np.asarray(m, dtype=bool)
        k = int(m.sum())
        if k == 0:
            out[name] = {"rate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                         "n_blocks": 0}
            continue
        tot = np.bincount(blocks[m], minlength=n_blocks).astype(np.float64)
        det = np.bincount(blocks[m], weights=fired[m].astype(np.float64),
                          minlength=n_blocks)
        num, den = W @ det, W @ tot
        with np.errstate(divide="ignore", invalid="ignore"):
            reps = np.where(den > 0, num / den, np.nan)
        lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[name] = {"rate": float(det.sum() / tot.sum()), "lo": float(lo),
                     "hi": float(hi), "n": k, "n_blocks": int((tot > 0).sum())}
    return out


# =====================================================================
# THRESHOLDS
# =====================================================================


def budget_threshold(reference: np.ndarray, budget: float) -> float:
    """The threshold a false-alarm budget turns into.

    The lowest OBSERVED benign score that leaves at most ``budget`` of the
    reference sample strictly above it.  A flow fires when its score is
    strictly greater, so on the reference sample itself the false-alarm rate
    is at most the budget -- exactly, not approximately.

    WHY NOT ``np.quantile``.  Its default interpolates between two order
    statistics, and the value it returns can leave one flow more than the
    budget allows above it.  One flow in 144,000 is nothing; but the fusion
    rule divides the budget between layers and promises the sum stays inside
    it, and three layers each one flow over is a promise broken (40,000
    benign flows, budget 0.1%, three layers: 42 fire where 40 are allowed).
    A guarantee that holds "up to rounding" is not one the paper can state.

    With ``n * budget < 1`` the reference sample is too small to support the
    budget at all; the threshold is then its maximum and nothing in it fires.
    """
    ref = np.asarray(reference, dtype=np.float64)
    ref = np.sort(ref[np.isfinite(ref)])
    n = ref.size
    if n == 0:
        raise ValueError("no validation benign scores to calibrate on")
    if not 0.0 < float(budget) < 1.0:
        raise ValueError(f"a false-alarm budget must be between 0 and 1, got {budget!r}")
    allowed = int(np.floor(n * float(budget) + 1e-9))      # flows permitted above the threshold
    return float(ref[n - 1 - allowed])


def fired_at_budget(score_test: np.ndarray, score_val_benign: np.ndarray,
                    budget: float) -> np.ndarray:
    """Which test flows a detector fires on at ``budget``: strictly above the
    threshold fitted on validation benign scores.  The same rule
    ``detection_table`` applies, exposed so two detectors' decisions can be
    compared flow by flow."""
    return np.asarray(score_test, dtype=np.float64) > budget_threshold(score_val_benign, budget)


# =====================================================================
# COMPARING TWO DETECTORS
# =====================================================================


def paired_difference(
    fired_a: np.ndarray,
    fired_b: np.ndarray,
    masks: Dict[str, np.ndarray],
    blocks: Optional[np.ndarray] = None,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Rate of A minus rate of B inside each mask, with a PAIRED block-bootstrap
    interval.  Returns ``{name: {"diff", "lo", "hi", "rate_a", "rate_b", "n",
    "n_blocks"}}``; see ``bootstrap_rates`` for why ``n_blocks`` matters.

    WHY PAIRED.  "A detects 31% [27, 36] and B detects 29% [25, 33]" does not
    say whether A is better: the two intervals overlap, yet both were measured
    on the same flows, and most of their spread is the test traffic itself,
    which moves A and B together.  Resampling the same blocks for both and
    taking the difference inside each replicate removes that shared part.  The
    interval of the difference is the one a claim "A beats B" has to clear.

    An interval that excludes zero supports "A and B differ on this test day,
    for these fitted models".  It says nothing about another day or another
    training run.
    """
    a = np.asarray(fired_a, dtype=bool)
    b = np.asarray(fired_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"the two detectors scored different rows: {a.shape} vs {b.shape}")
    n = a.size
    out: Dict[str, Dict[str, float]] = {}
    nan = float("nan")

    use_blocks = blocks is not None and n_boot > 0
    if use_blocks:
        blocks = np.asarray(blocks, dtype=np.int64)
        if blocks.size != n:
            raise ValueError(f"blocks has {blocks.size} entries for {n} rows")
        n_blocks = int(blocks.max()) + 1 if n else 0
        rng = np.random.default_rng(seed)
        W = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks),
                            size=n_boot).astype(np.float64)

    for name, m in masks.items():
        m = np.asarray(m, dtype=bool)
        k = int(m.sum())
        if k == 0:
            out[name] = {"diff": nan, "lo": nan, "hi": nan, "rate_a": nan, "rate_b": nan, "n": 0,
                         "n_blocks": 0}
            continue
        ra, rb = float(a[m].mean()), float(b[m].mean())
        rec = {"diff": ra - rb, "lo": nan, "hi": nan, "rate_a": ra, "rate_b": rb, "n": k,
               "n_blocks": (int(np.unique(np.asarray(blocks)[m]).size)
                            if blocks is not None else 0)}
        if use_blocks:
            tot = np.bincount(blocks[m], minlength=n_blocks).astype(np.float64)
            d = np.bincount(blocks[m], weights=a[m].astype(np.float64) - b[m].astype(np.float64),
                            minlength=n_blocks)
            num, den = W @ d, W @ tot
            with np.errstate(divide="ignore", invalid="ignore"):
                reps = np.where(den > 0, num / den, np.nan)
            lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            rec["lo"], rec["hi"] = float(lo), float(hi)
        out[name] = rec
    return out


def projected_precision(detection_rate: float, false_alarm_rate: float,
                        prevalence: float) -> float:
    """Share of alerts that are real attacks when attacks are ``prevalence`` of
    all flows.

    The test day of this dataset is roughly two-fifths attack traffic; a real
    link is far below one percent.  Precision measured on the test day is
    therefore flattering by a large factor (Axelsson's base-rate fallacy).
    This re-weights the two measured rates to a stated prevalence:

        precision = TPR * p / (TPR * p + FPR * (1 - p))

    It is a projection, not a measurement: it assumes the detection rate and
    the false-alarm rate measured here would hold at that prevalence.
    """
    tp = float(detection_rate) * float(prevalence)
    fp = float(false_alarm_rate) * (1.0 - float(prevalence))
    return float("nan") if tp + fp <= 0 else tp / (tp + fp)


# =====================================================================
# TABLES
# =====================================================================


def _class_masks(y_true: np.ndarray, benign_label: str) -> Dict[str, np.ndarray]:
    masks = {c: (y_true == c) for c in pd.unique(y_true) if c != benign_label}
    masks[ANY_ATTACK] = y_true != benign_label
    return masks


def detection_table(
    y_true_str: Sequence,
    score_test: np.ndarray,
    score_val_benign: np.ndarray,
    budgets: Iterable[float] = STUDY_BUDGETS,
    blocks: Optional[np.ndarray] = None,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
    benign_label: str = BENIGN_LABEL,
    detector: str = "",
) -> pd.DataFrame:
    """Per-class detection at false-alarm budgets fitted on VALIDATION benign.

    One row per (budget, class), plus an any-attack row per budget.  The
    threshold for budget ``b`` is ``budget_threshold(validation benign scores,
    b)`` -- nothing from the test split enters it -- and the false-alarm rate
    actually observed on test benign rows is reported beside the target,
    because the gap between the two is a result in its own right.

    A flow fires when its score is strictly ABOVE the threshold.  With a
    discrete score (the behaviour layer counts ports) many validation flows
    tie at the threshold; strict inequality keeps the validation false-alarm
    rate at or below the budget instead of above it.
    """
    y_true = np.asarray(y_true_str, dtype=object)
    s = np.asarray(score_test, dtype=np.float64)
    ref = np.asarray(score_val_benign, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        raise ValueError("no validation benign scores to calibrate on")
    if s.shape != y_true.shape:
        raise ValueError(f"scores {s.shape} vs labels {y_true.shape}")

    benign = y_true == benign_label
    masks = _class_masks(y_true, benign_label)
    masks_all = {**masks, benign_label: benign}

    rows = []
    for b in budgets:
        tau = budget_threshold(ref, b)
        fired = s > tau
        r = bootstrap_rates(fired, masks_all, blocks, n_boot, seed)
        fpr = r[benign_label]
        for name in masks:
            rows.append({
                "detector": detector, "budget": float(b), "threshold": tau,
                "class": name, "n": r[name]["n"], "n_blocks": r[name]["n_blocks"],
                "interval_counts": interval_counts(r[name]["n_blocks"]),
                "detection_rate": r[name]["rate"],
                "det_lo": r[name]["lo"], "det_hi": r[name]["hi"],
                "observed_fpr": fpr["rate"], "fpr_lo": fpr["lo"], "fpr_hi": fpr["hi"],
                "n_benign": fpr["n"], "n_blocks_benign": fpr["n_blocks"],
            })
    return pd.DataFrame(rows)


def auc_table(
    y_true_str: Sequence,
    score_test: np.ndarray,
    benign_label: str = BENIGN_LABEL,
    max_fpr: float = 0.01,
    detector: str = "",
) -> pd.DataFrame:
    """Threshold-free separability of each attack class from test benign.

    ``auroc`` is the usual area.  ``pauc`` is the standardised partial area up
    to ``max_fpr`` (McClish; 0.5 = chance, 1 = perfect), which looks only at
    the region an analyst could afford to operate in.  A detector can score
    AUROC 0.9 and still be useless at 1% false alarms; the partial area is the
    number that shows it.
    """
    y_true = np.asarray(y_true_str, dtype=object)
    s = np.asarray(score_test, dtype=np.float64)
    benign = y_true == benign_label
    rows = []
    for name, m in _class_masks(y_true, benign_label).items():
        sel = benign | m
        yb = m[sel].astype(int)
        if yb.sum() == 0 or yb.sum() == yb.size:
            auroc = pauc = float("nan")
        else:
            auroc = float(roc_auc_score(yb, s[sel]))
            pauc = float(roc_auc_score(yb, s[sel], max_fpr=max_fpr))
        rows.append({"detector": detector, "class": name, "n": int(m.sum()),
                     "auroc": auroc, f"pauc@{max_fpr:g}": pauc})
    return pd.DataFrame(rows)


def summarise_seeds(tables: Sequence[pd.DataFrame], keys: Sequence[str],
                    values: Sequence[str]) -> pd.DataFrame:
    """Mean and standard deviation of ``values`` across seed repeats.

    ``tables`` are per-seed frames with the same ``keys``.  The standard
    deviation is across retrainings -- a different source of variation from
    the block bootstrap, and reported separately for that reason.
    """
    if not tables:
        return pd.DataFrame()
    cat = pd.concat([t.assign(_seed=i) for i, t in enumerate(tables)], ignore_index=True)
    g = cat.groupby(list(keys), sort=False, dropna=False)
    out = g[list(values)].mean().add_suffix("_mean")
    sd = g[list(values)].std(ddof=1).add_suffix("_sd")
    n = g["_seed"].nunique().rename("n_seeds")
    return pd.concat([out, sd, n], axis=1).reset_index()
