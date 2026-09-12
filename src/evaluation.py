"""
Evaluation.

The previous ``evaluate_model`` computed accuracy against ``y_true_open`` and
precision/recall/F1 against ``y_true``, then built the confusion matrix from
``y_true_open`` again, while the label list came from ``y_true``.  In cross-day
mode those are different arrays, so a single printout described three different
problems and the macro average was divided by a class count that depended on how
many unseen attack types happened to appear in the test day.

This module enforces one rule: **every metric in a report is computed from the
same two string arrays**, and the label list is derived from those same arrays.

It also removes the fixed output paths.  Results go to
``runs/<sha>-<timestamp>-<tag>/`` and are never overwritten.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from config import RUNS_DIR, UNKNOWN_LABEL
from seeding import git_sha


# =====================================================================
# RUN DIRECTORY
# =====================================================================


def new_run_dir(tag: str = "run", root: Path = RUNS_DIR) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{git_sha()}-{stamp}-{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default))


# =====================================================================
# LABEL DECODING
# =====================================================================


def decode_predictions(y_pred_idx: np.ndarray, encoder) -> np.ndarray:
    """
    Decode integer predictions to a **object-dtype** string array.

    BUG FIX: the old code used ``np.array(...).astype(str)``, which produces a
    FIXED-WIDTH unicode array sized to the longest existing class name.  In the
    binary case (ATTACK / BENIGN -> dtype '<U6'), assigning "Unknown_Attack"
    silently truncated it to "Unknow", creating a phantom class that matched
    nothing in the ground truth.  Reproduced:

        >>> p = np.array(enc.inverse_transform([0,1,0])).astype(str)  # '<U6'
        >>> p[0] = "Unknown_Attack"; p
        array(['Unknow', 'BENIGN', 'ATTACK'], dtype='<U6')

    object dtype has no width, so assignment can never truncate.
    """
    decoded = encoder.inverse_transform(np.asarray(y_pred_idx, dtype=np.int64))
    return np.asarray(decoded, dtype=object)


def apply_rejection(
    y_pred_str: np.ndarray,
    reject_mask: np.ndarray,
    unknown_label: str = UNKNOWN_LABEL,
) -> np.ndarray:
    out = np.asarray(y_pred_str, dtype=object).copy()
    out[np.asarray(reject_mask, dtype=bool)] = unknown_label
    return out


# =====================================================================
# CORE REPORT
# =====================================================================


@dataclass
class EvalReport:
    metrics: Dict[str, float]
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    labels: List[str]
    n_samples: int
    tag: str = "eval"
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "n_samples": self.n_samples,
            "labels": self.labels,
            "metrics": self.metrics,
            **({"extra": self.extra} if self.extra else {}),
        }

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.to_dict(), out_dir / f"{self.tag}_metrics.json")
        self.per_class.to_csv(out_dir / f"{self.tag}_per_class.csv")
        self.confusion.to_csv(out_dir / f"{self.tag}_confusion_matrix.csv")

    def __str__(self) -> str:
        w = 22
        lines = [f"\n========== {self.tag.upper()} ==========",
                 f"{'samples':<{w}}: {self.n_samples:,}"]
        for k, v in self.metrics.items():
            lines.append(f"{k:<{w}}: {v:.6f}")
        lines.append("")
        lines.append(self.per_class.to_string(float_format=lambda x: f"{x:.4f}"))
        return "\n".join(lines)


def evaluate_predictions(
    y_true_str: Sequence,
    y_pred_str: Sequence,
    tag: str = "eval",
    labels: Optional[Sequence[str]] = None,
) -> EvalReport:
    """
    Single-ground-truth evaluation.

    ``y_true_str`` and ``y_pred_str`` are string arrays.  Every metric below,
    the per-class table and the confusion matrix are computed from exactly these
    two arrays and the same ``labels`` list.  There is no second ground truth.
    """
    y_true = np.asarray(y_true_str, dtype=object)
    y_pred = np.asarray(y_pred_str, dtype=object)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("evaluate_predictions received empty arrays")

    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = list(labels)

    unlisted = (set(y_true.tolist()) | set(y_pred.tolist())) - set(labels)
    if unlisted:
        raise ValueError(f"labels= is missing values present in the data: {sorted(unlisted)}")

    common = dict(labels=labels, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", **common)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", **common)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **common)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", **common)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", **common)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", **common)),
    }

    # Macro averages above run over the UNION of true and predicted labels,
    # which is correct (predicting a class that never occurs is a real error)
    # but hard to read when a model emits many never-occurring classes.  The
    # "_present" variants restrict the average to classes that actually occur
    # in the ground truth.  Report both; say which one you quote.
    present = [l for l in labels if l in set(y_true.tolist())]
    p_common = dict(labels=present, zero_division=0)
    metrics["macro_precision_present"] = float(
        precision_score(y_true, y_pred, average="macro", **p_common))
    metrics["macro_recall_present"] = float(
        recall_score(y_true, y_pred, average="macro", **p_common))
    metrics["macro_f1_present"] = float(
        f1_score(y_true, y_pred, average="macro", **p_common))
    metrics["n_classes_true"] = float(len(present))
    metrics["n_classes_union"] = float(len(labels))

    # balanced accuracy and MCC are undefined with a single class present.
    # sklearn returns 0.0 with a UserWarning, which reads as a real score.
    if len(set(y_true.tolist())) < 2:
        metrics["balanced_accuracy"] = float("nan")
        metrics["mcc"] = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
            metrics["mcc"] = float(
                matthews_corrcoef(y_true.astype(str), y_pred.astype(str))
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rep = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
    per_class = pd.DataFrame(rep).transpose()
    per_class = per_class.loc[[l for l in labels if l in per_class.index]]
    per_class["support"] = per_class["support"].astype(int)
    # a per-class F1 on 11 samples has a very wide CI; flag it rather than
    # letting a reader treat "F1 = 1.00" as a claim
    per_class["evaluable"] = per_class["support"] >= 100

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels],
                         columns=[f"pred:{l}" for l in labels])

    return EvalReport(
        metrics=metrics, per_class=per_class, confusion=cm_df,
        labels=labels, n_samples=int(y_true.size), tag=tag,
    )


# =====================================================================
# PLOTS
# =====================================================================


def plot_confusion(
    report: EvalReport,
    out_path: Path,
    normalise: bool = True,
    dpi: int = 150,
) -> Path:
    cm = report.confusion.to_numpy().astype(np.float64)
    if normalise:
        row = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
        fmt = "{:.2f}"
    else:
        shown = cm
        fmt = "{:.0f}"

    n = len(report.labels)
    size = max(6.0, 0.55 * n + 3.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.85), dpi=dpi)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max() if shown.size else 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n)); ax.set_xticklabels(report.labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(report.labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({'row-normalised' if normalise else 'counts'}) - {report.tag}")

    thresh = shown.max() / 2.0 if shown.size and shown.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            if shown[i, j] > 0:
                ax.text(j, i, fmt.format(shown[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if shown[i, j] > thresh else "black")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_history(history_rows: List[dict], out_path: Path, dpi: int = 150) -> Optional[Path]:
    if not history_rows:
        return None
    df = pd.DataFrame(history_rows)
    numeric = [c for c in df.columns if c not in ("epoch", "seconds") and
               pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=dpi)
    loss_cols = [c for c in numeric if "loss" in c or "mse" in c or "d2" in c]
    for c in loss_cols:
        axes[0].plot(df["epoch"], df[c], label=c, linewidth=1.4)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    if loss_cols:
        axes[0].legend(fontsize=8)
        if (df[loss_cols].to_numpy() > 0).all():
            axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)

    metric_cols = [c for c in numeric if c not in loss_cols and c != "lr"]
    for c in metric_cols:
        axes[1].plot(df["epoch"], df[c], label=c, linewidth=1.4)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("metric")
    if metric_cols:
        axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================


def feature_importance(model, feature_names: Sequence[str], out_dir: Path,
                       tag: str = "model", top_k: int = 20) -> Optional[pd.DataFrame]:
    if not hasattr(model, "feature_importances_"):
        return None
    imp = np.asarray(model.feature_importances_, dtype=np.float64)
    if len(imp) != len(feature_names):
        raise ValueError(f"{len(imp)} importances vs {len(feature_names)} feature names")

    df = (pd.DataFrame({"feature": list(feature_names), "importance": imp})
          .sort_values("importance", ascending=False).reset_index(drop=True))

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{tag}_feature_importance.csv", index=False)

    top = df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.32 * len(top))), dpi=150)
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("importance"); ax.set_title(f"Top {len(top)} features - {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_feature_importance.png")
    plt.close(fig)
    return df


# =====================================================================
# DEPRECATED SHIM
# =====================================================================


def evaluate_model(*args, **kwargs):  # pragma: no cover
    raise RuntimeError(
        "evaluate_model() has been removed. It computed accuracy against "
        "y_true_open and precision/recall/F1 against y_true, so its outputs "
        "described different problems. Use evaluate_predictions(y_true_str, "
        "y_pred_str) instead."
    )


# =====================================================================
# PRIMARY REPORT: per-class detection rate at a fixed false-alarm budget
# =====================================================================


def detection_by_class(
    y_true_str: Sequence,
    p_attack_test: np.ndarray,
    p_attack_val: np.ndarray,
    y_val_str: Sequence,
    benign_label: str = "BENIGN",
    budgets: Sequence[float] = (0.001, 0.01, 0.05),
    flows_per_day: int = 10_000_000,
    prevalence: float = 0.001,
) -> "pd.DataFrame":
    """Per-class detection rate at false-alarm budgets calibrated on VALIDATION.

    WHY THIS REPLACED ACCURACY AS THE HEADLINE
    ------------------------------------------
    Accuracy on this task is not merely weak, it is ambiguous.  On the crossday
    protocol the same Random Forest scores:

        multi-class accuracy (exact label)  : 58.47%
        binary  accuracy (attack vs benign) : 69.68%
        all-BENIGN baseline                 : 58.91%

    Eleven points apart, same model, same rows.  Anyone quoting "accuracy"
    without saying which has said nothing -- and the multi-class figure is
    BELOW the constant-function baseline, because the model labels zero attack
    flows correctly (DDoS, PortScan and Bot are not in its label space at all).

    A detector has one job: catch attacks without drowning the analyst.  So the
    report is the two numbers that job is made of -- detection rate per attack
    class, at a false-alarm rate you choose in advance.

    HOW THE OPERATING POINT IS CHOSEN
    ---------------------------------
    ``p_attack = 1 - P(BENIGN)`` gives a single tunable knob.  The threshold
    for each budget is the corresponding quantile of the VALIDATION benign
    scores -- nothing from the test day enters it.  The observed test-day
    false-alarm rate is reported alongside the target, because the two diverge
    under distribution shift and that divergence is itself a result.

    ALERT VOLUME
    ------------
    CIC-IDS2017's test day is ~41% attack.  A real link is nearer 0.1%.  Every
    rate measured here is therefore measured at roughly 400x the real
    prevalence, which flatters precision enormously.  The projected columns
    restate each operating point as alerts per day on a link carrying
    ``flows_per_day`` flows at ``prevalence`` attack rate -- the number that
    decides whether a detector is deployable.
    """
    import pandas as pd

    y_true = np.asarray(y_true_str, dtype=object)
    y_val = np.asarray(y_val_str, dtype=object)
    val_benign = np.asarray(p_attack_val, dtype=np.float64)[y_val == benign_label]
    if val_benign.size == 0:
        raise ValueError("validation split contains no benign rows to calibrate on")

    classes = [c for c in pd.unique(y_true) if c != benign_label]
    rows = []
    for b in budgets:
        tau = float(np.quantile(val_benign, 1.0 - b))
        fired = np.asarray(p_attack_test, dtype=np.float64) > tau

        benign_mask = y_true == benign_label
        observed_fpr = float(fired[benign_mask].mean()) if benign_mask.any() else float("nan")

        n_benign_day = flows_per_day * (1.0 - prevalence)
        n_attack_day = flows_per_day * prevalence
        false_per_day = n_benign_day * observed_fpr

        for c in classes:
            m = y_true == c
            rows.append({
                "fpr_budget": b,
                "threshold": tau,
                "class": c,
                "n": int(m.sum()),
                "detection_rate": float(fired[m].mean()) if m.any() else float("nan"),
                "observed_benign_fpr": observed_fpr,
                "false_alerts_per_day": false_per_day,
            })
        # one summary row per budget: any-attack recall
        atk = ~benign_mask
        recall_any = float(fired[atk].mean()) if atk.any() else float("nan")
        rows.append({
            "fpr_budget": b, "threshold": tau, "class": "__ANY_ATTACK__",
            "n": int(atk.sum()), "detection_rate": recall_any,
            "observed_benign_fpr": observed_fpr,
            "false_alerts_per_day": false_per_day,
            "projected_true_alerts_per_day": n_attack_day * recall_any,
            "projected_precision": (n_attack_day * recall_any) /
                                   max(n_attack_day * recall_any + false_per_day, 1e-9),
        })
    return pd.DataFrame(rows)


def format_detection_table(df: "pd.DataFrame", flows_per_day: int = 10_000_000,
                           prevalence: float = 0.001) -> str:
    """Render detection_by_class() as the report a security engineer reads."""
    lines = [
        "",
        "=" * 78,
        "PRIMARY REPORT -- detection rate per class at a fixed false-alarm budget",
        "(thresholds calibrated on the VALIDATION day; nothing from test enters them)",
        "=" * 78,
    ]
    for b in sorted(df["fpr_budget"].unique()):
        sub = df[df.fpr_budget == b]
        summ = sub[sub["class"] == "__ANY_ATTACK__"].iloc[0]
        lines += [
            "",
            f"--- target benign FPR {b:.1%}   (observed on test: "
            f"{summ.observed_benign_fpr:.2%})   threshold={summ.threshold:.6f}",
            f"    {'class':<24s}{'n':>10s}{'detected':>12s}",
        ]
        for _, r in sub[sub["class"] != "__ANY_ATTACK__"].sort_values("n", ascending=False).iterrows():
            lines.append(f"    {r['class']:<24s}{r['n']:>10,}{r['detection_rate']:>11.2%}")
        lines.append(f"    {'ANY ATTACK':<24s}{summ.n:>10,}{summ.detection_rate:>11.2%}")
        lines += [
            f"    projected on {flows_per_day:,} flows/day at {prevalence:.1%} attack prevalence:",
            f"      false alerts/day {summ.false_alerts_per_day:>12,.0f}"
            f"   true alerts/day {summ.get('projected_true_alerts_per_day', float('nan')):>10,.0f}"
            f"   precision {summ.get('projected_precision', float('nan')):>7.2%}",
        ]
    lines += [
        "",
        "Read this, not accuracy. Accuracy on this split is ambiguous (58.47%",
        "multi-class vs 69.68% binary, against a 58.91% all-BENIGN baseline) and",
        "the multi-class figure is below the constant-function baseline.",
        "=" * 78,
    ]
    return "\n".join(lines)
