"""
Open-set recognition hooks.

The old pipeline thresholded max-softmax at 0.90 (while ``CLAUDE.md`` declared
the invariant to be 0.55), applied the rejection to the prediction array, and
printed an "Unknown Rate".  Three problems:

1. Max-softmax is a **closed-set posterior**.  It answers "which of my K
   classes is this most like", never "is this like any of them".  Tree
   ensembles make it worse: leaves at the edge of the training hull are the
   purest, so out-of-distribution points get HIGHER confidence.

2. The threshold was never fitted on held-out data, because no validation split
   existed.

3. Unknown-rate alone is maximised by rejecting everything.  Nothing measured
   whether the score actually separates known from unknown.

This module provides four interchangeable scorers (higher = more novel), a
threshold calibrated to a target benign FPR on the validation day, and the
detection metrics that make the result reportable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from config import UNKNOWN_LABEL


# =====================================================================
# SCORERS  (higher = more novel / more likely unknown)
# =====================================================================


def msp_score(logits: np.ndarray) -> np.ndarray:
    """1 - max softmax probability. Kept as the weak baseline to beat."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return 1.0 - p.max(axis=1)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    -T * logsumexp(logits / T).  Retains logit MAGNITUDE, which softmax
    normalises away, and is consistently stronger than max-softmax for OOD
    detection at zero extra training cost.
    """
    z = np.asarray(logits, dtype=np.float64)
    return -T * logsumexp(z / T, axis=1)


def entropy_score(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    logp = z - logsumexp(z, axis=1, keepdims=True)
    p = np.exp(logp)
    return -(p * logp).sum(axis=1)


class MahalanobisScorer:
    """
    Distance to the nearest class-conditional Gaussian with a shared,
    Ledoit-Wolf shrunk covariance.  Model-agnostic: works on raw features or on
    a network's penultimate embedding, and is the only option for Random Forest
    (which has no logits).
    """

    def __init__(self, shrinkage: bool = True):
        self.shrinkage = shrinkage
        self.mu_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_samples: int = 200_000,
        random_state: int = 42,
    ) -> "MahalanobisScorer":
        """Fit class means and one shared shrunk covariance.

        MEMORY.  The previous body did, on the full training matrix:

            X = np.asarray(X, dtype=np.float64)   # copy 1 -- float32 -> float64
            centered.append(Xk - m)               # copy 2 -- every row again
            C = np.concatenate(centered, axis=0)   # copy 3 -- and again
            LedoitWolf().fit(C)                    # copy 4 inside sklearn

        On 1.56M x 62 that is ~3 GB for a covariance estimate, and it was the
        single largest allocation in the whole training run.

        A covariance in p dimensions has p(p+1)/2 free parameters -- 1,953 for
        p=62.  Estimating it from 1.56M rows rather than 200k buys nothing: at
        200k you already have ~100 observations per parameter.  So the fit is
        taken on a class-stratified subsample.  The class MEANS are still
        computed on every row, in chunks, because those are cheap and the rare
        classes (Heartbleed: 11 rows) need all of them.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = (
            np.unique(y[y >= 0]) if np.issubdtype(y.dtype, np.integer) else np.unique(y)
        )

        # --- exact class means, chunked, no full float64 copy ---
        mus = []
        keep_classes = []
        for k in self.classes_:
            idx = np.flatnonzero(y == k)
            if idx.size == 0:
                continue
            acc = np.zeros(X.shape[1], dtype=np.float64)
            for start in range(0, idx.size, 100_000):
                acc += X[idx[start:start + 100_000]].sum(axis=0, dtype=np.float64)
            mus.append(acc / idx.size)
            keep_classes.append(k)
        self.classes_ = np.asarray(keep_classes)
        self.mu_ = np.stack(mus)

        # --- shared covariance on a stratified subsample ---
        rng = np.random.default_rng(random_state)
        n = len(y)
        if n > max_samples:
            take = []
            for k in self.classes_:
                idx = np.flatnonzero(y == k)
                quota = max(1, int(round(max_samples * idx.size / n)))
                take.append(idx if idx.size <= quota
                            else rng.choice(idx, size=quota, replace=False))
            sub = np.concatenate(take)
        else:
            sub = np.arange(n)

        Xs = X[sub].astype(np.float64, copy=True)
        ys = y[sub]
        for i, k in enumerate(self.classes_):
            m = ys == k
            if m.any():
                Xs[m] -= self.mu_[i]          # centre IN PLACE, no second copy

        if self.shrinkage:
            cov = LedoitWolf(assume_centered=True).fit(Xs).covariance_
        else:
            cov = np.cov(Xs, rowvar=False)
        del Xs

        cov = cov + 1e-6 * np.eye(cov.shape[0])
        self.precision_ = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray, chunk: int = 100_000) -> np.ndarray:
        """Squared Mahalanobis distance to the NEAREST class mean.

        Chunked: the previous version allocated a full float64 copy of X per
        class (703k x 62 x 8 B = 349 MB, twelve times over) inside the loop.
        """
        if self.mu_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisScorer.score before fit")
        X = np.asarray(X)
        out = np.empty(X.shape[0], dtype=np.float64)
        P = self.precision_
        for start in range(0, X.shape[0], chunk):
            Xc = X[start:start + chunk].astype(np.float64, copy=False)
            best = None
            for m in self.mu_:
                d = Xc - m
                dist = np.einsum("ij,jk,ik->i", d, P, d)
                best = dist if best is None else np.minimum(best, dist)
            out[start:start + chunk] = best
        return out

    # convenience alias so every scorer has the same call shape
    __call__ = score


def reconstruction_score(errors: np.ndarray) -> np.ndarray:
    """Autoencoder per-sample MSE. Already 'higher = more novel'."""
    return np.asarray(errors, dtype=np.float64)


def svdd_score(distances: np.ndarray) -> np.ndarray:
    """Deep SVDD squared distance to the hypersphere center."""
    return np.asarray(distances, dtype=np.float64)


# =====================================================================
# THRESHOLD CALIBRATION
# =====================================================================


def threshold_at_fpr(scores_val_negative: np.ndarray, target_fpr: float = 0.05) -> float:
    """
    tau such that only ``target_fpr`` of the validation NEGATIVE population
    (in-distribution / benign) is rejected.

    Fitted on the validation day only.  Nothing from the test day may enter.
    """
    s = np.asarray(scores_val_negative, dtype=np.float64)
    if s.size == 0:
        raise ValueError("threshold_at_fpr received an empty score array")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.quantile(s, 1.0 - target_fpr))


def threshold_at_tpr(scores_val_positive: np.ndarray, target_tpr: float = 0.95) -> float:
    s = np.asarray(scores_val_positive, dtype=np.float64)
    return float(np.quantile(s, 1.0 - target_tpr))


# =====================================================================
# DETECTION METRICS
# =====================================================================


@dataclass
class OpenSetResult:
    auroc: float
    aupr: float
    tau: float
    unknown_recall: float          # TPR: unknowns correctly rejected
    known_rejection: float         # FPR: knowns wrongly rejected
    tpr_at_5fpr: float
    n_known: int
    n_unknown: int
    scorer: str

    def to_dict(self) -> Dict[str, float]:
        return {
            "scorer": self.scorer,
            "auroc": self.auroc,
            "aupr": self.aupr,
            "tau": self.tau,
            "unknown_recall@tau": self.unknown_recall,
            "known_rejection@tau": self.known_rejection,
            "tpr@5%fpr": self.tpr_at_5fpr,
            "n_known": self.n_known,
            "n_unknown": self.n_unknown,
        }

    def __str__(self) -> str:
        return (
            f"\n---------- OPEN-SET [{self.scorer}] ----------\n"
            f"AUROC                : {self.auroc:.6f}\n"
            f"AUPR                 : {self.aupr:.6f}\n"
            f"tau (val-calibrated) : {self.tau:.6f}\n"
            f"unknown recall @tau  : {self.unknown_recall:.6f}\n"
            f"known rejection @tau : {self.known_rejection:.6f}\n"
            f"TPR @ 5% FPR         : {self.tpr_at_5fpr:.6f}\n"
            f"n known / unknown    : {self.n_known:,} / {self.n_unknown:,}"
        )


def unknown_mask(y_true_str: Sequence, known_classes: Sequence[str]) -> np.ndarray:
    known = set(known_classes)
    return np.array([lbl not in known for lbl in np.asarray(y_true_str, dtype=object)], dtype=bool)


def open_set_report(
    scores: np.ndarray,
    is_unknown: np.ndarray,
    tau: float,
    scorer: str = "score",
) -> OpenSetResult:
    s = np.asarray(scores, dtype=np.float64)
    u = np.asarray(is_unknown, dtype=bool)

    if s.shape != u.shape:
        raise ValueError(f"scores {s.shape} vs is_unknown {u.shape}")

    n_unknown, n_known = int(u.sum()), int((~u).sum())
    if n_unknown == 0 or n_known == 0:
        # AUROC undefined with a single class present
        auroc = aupr = float("nan")
        tpr5 = float("nan")
    else:
        auroc = float(roc_auc_score(u, s))
        aupr = float(average_precision_score(u, s))
        fpr, tpr, _ = roc_curve(u, s)
        tpr5 = float(np.interp(0.05, fpr, tpr))

    rejected = s > tau
    return OpenSetResult(
        auroc=auroc, aupr=aupr, tau=float(tau),
        unknown_recall=float(rejected[u].mean()) if n_unknown else float("nan"),
        known_rejection=float(rejected[~u].mean()) if n_known else float("nan"),
        tpr_at_5fpr=tpr5, n_known=n_known, n_unknown=n_unknown, scorer=scorer,
    )


def oscr_curve(
    scores: np.ndarray,
    is_unknown: np.ndarray,
    correct_known: np.ndarray,
    n_points: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Open-Set Classification Rate curve: correct-classification rate on knowns
    vs false-positive rate on unknowns, swept over tau.
    """
    s = np.asarray(scores, dtype=np.float64)
    u = np.asarray(is_unknown, dtype=bool)
    c = np.asarray(correct_known, dtype=bool)

    taus = np.quantile(s, np.linspace(0.0, 1.0, n_points))
    ccr, fpr = [], []
    n_known = max(int((~u).sum()), 1)
    n_unknown = max(int(u.sum()), 1)
    for t in taus:
        accepted = s <= t
        ccr.append((accepted & ~u & c).sum() / n_known)
        fpr.append((accepted & u).sum() / n_unknown)
    order = np.argsort(fpr)
    return {"fpr": np.asarray(fpr)[order], "ccr": np.asarray(ccr)[order], "tau": taus[order]}


def rank_scorers(
    scores_by_name: Dict[str, np.ndarray],
    is_unknown: np.ndarray,
    val_scores_by_name: Optional[Dict[str, np.ndarray]] = None,
    target_fpr: float = 0.05,
) -> "list[OpenSetResult]":
    """
    Score every rejector on the same ground truth and sort by AUROC.

    An AUROC below 0.5 is not noise -- it means the score carries real signal
    with the sign reversed, which on this dataset has a specific cause.  Friday
    contains *near*-OOD: DDoS sits inside the Wednesday DoS-Hulk cloud.  A flow
    deep inside a known class's region gets a LARGE max logit, so
    ``-logsumexp`` is very negative, so energy calls it in-distribution.
    Meanwhile genuine BENIGN traffic sits near several decision boundaries and
    gets small margins, so energy calls it novel.  Result: benign scores as
    more novel than the attack, and AUROC inverts.

    Do NOT respond by flipping the sign -- that is fitting to the test labels.
    Use a scorer that measures distance from the training manifold instead of
    classifier confidence (``mahalanobis_embed``), or select the scorer on a
    held-out pseudo-unknown class (``--holdout-class``).
    """
    out = []
    for name, s in scores_by_name.items():
        if val_scores_by_name and name in val_scores_by_name:
            tau = threshold_at_fpr(val_scores_by_name[name], target_fpr)
        else:
            tau = threshold_at_fpr(s[~np.asarray(is_unknown, dtype=bool)], target_fpr)
        out.append(open_set_report(s, is_unknown, tau, scorer=name))
    return sorted(out, key=lambda r: (-r.auroc if r.auroc == r.auroc else 0.0))


def format_scorer_table(results: "list[OpenSetResult]") -> str:
    lines = [
        "",
        "---------- REJECTION SCORER COMPARISON ----------",
        f"{'scorer':<22s} {'AUROC':>8s} {'AUPR':>8s} {'TPR@5%FPR':>11s} "
        f"{'unk recall':>11s} {'known rej':>10s}",
        "-" * 76,
    ]
    for r in results:
        flag = "   <- INVERTED (< 0.5)" if r.auroc == r.auroc and r.auroc < 0.5 else ""
        lines.append(
            f"{r.scorer:<22s} {r.auroc:>8.4f} {r.aupr:>8.4f} {r.tpr_at_5fpr:>11.4f} "
            f"{r.unknown_recall:>11.4f} {r.known_rejection:>10.4f}{flag}"
        )
    best = results[0] if results else None
    if best is not None and best.auroc == best.auroc and best.auroc < 0.5:
        lines += [
            "",
            "[warn] EVERY scorer is inverted. The classifier is more confident on",
            "       unknown traffic than on known BENIGN traffic. Most likely causes:",
            "         1. BatchNorm in the encoder  -> retrain with --norm ln",
            "         2. near-OOD test classes     -> use --scorer mahalanobis_embed",
            "       Do not flip the sign: that fits the test labels.",
        ]
    return "\n".join(lines)


def make_open_set_predictions(
    y_pred_str: np.ndarray,
    scores: np.ndarray,
    tau: float,
    unknown_label: str = UNKNOWN_LABEL,
) -> np.ndarray:
    out = np.asarray(y_pred_str, dtype=object).copy()
    out[np.asarray(scores, dtype=np.float64) > tau] = unknown_label
    return out
