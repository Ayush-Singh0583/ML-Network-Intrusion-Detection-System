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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MahalanobisScorer":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y[y >= 0]) if np.issubdtype(y.dtype, np.integer) else np.unique(y)

        mus, centered = [], []
        for k in self.classes_:
            Xk = X[y == k]
            if len(Xk) == 0:
                continue
            m = Xk.mean(axis=0)
            mus.append(m)
            centered.append(Xk - m)
        self.mu_ = np.stack(mus)
        C = np.concatenate(centered, axis=0)

        if self.shrinkage:
            cov = LedoitWolf(assume_centered=True).fit(C).covariance_
        else:
            cov = np.cov(C, rowvar=False)
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        self.precision_ = np.linalg.pinv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisScorer.score before fit")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], self.mu_.shape[0]), dtype=np.float64)
        for i, m in enumerate(self.mu_):
            d = X - m
            out[:, i] = np.einsum("ij,jk,ik->i", d, self.precision_, d)
        return out.min(axis=1)          # distance to nearest known class

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


def make_open_set_predictions(
    y_pred_str: np.ndarray,
    scores: np.ndarray,
    tau: float,
    unknown_label: str = UNKNOWN_LABEL,
) -> np.ndarray:
    out = np.asarray(y_pred_str, dtype=object).copy()
    out[np.asarray(scores, dtype=np.float64) > tau] = unknown_label
    return out
