"""
Model service for the FastAPI backend.

Problems in the previous version:

1.  Four artifacts were loaded at MODULE IMPORT with no try/except, so a
    missing pickle killed the uvicorn process at import rather than returning
    a 503.  Loading is now lazy and guarded, and ``/health`` can report the
    model as unavailable instead of the app failing to start.

2.  It read from ``models/``, which had diverged from ``saved_models/``.  The
    RF, scaler, encoder and feature list there were dated 6 July while the
    preprocessing code had changed repeatedly since.  Nothing verified that
    the scaler width, the feature list and the model agreed.  ``load_bundle``
    now cross-checks all of them and refuses to serve a mismatched set.

3.  ``predict_dataframe`` filled any missing training feature with ``0.0``
    BEFORE scaling.  Zero is not neutral: after StandardScaler it becomes
    ``-mean/std``, a confidently wrong value.  Missing features are now a
    400-level error, not a silent guess.

4.  ``df.dropna(inplace=True)`` silently discarded rows, so the returned
    ``total_records`` did not match the input and the caller had no way to
    know which flows were dropped.  Rows are now imputed with the bundle's
    training medians and the count is reported.

5.  ``predict_dataframe`` wrote ``predicted_output.csv`` into the process
    working directory on every request -- a race between concurrent requests
    and an unbounded disk write.  Removed; the caller receives the results.

6.  The returned ``confidence`` was presented as if it meant "how sure are we
    this is an attack".  It is a closed-set posterior over known classes and
    cannot express "none of the above".  The response now carries an explicit
    ``is_known`` flag driven by a calibrated novelty threshold when one is
    present in the bundle.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend.config import BUNDLE_NAME  # noqa: E402

_lock = threading.Lock()
_bundle = None
_load_error: Optional[str] = None


def get_bundle():
    """Lazy, thread-safe, and it never takes the process down."""
    global _bundle, _load_error
    if _bundle is not None:
        return _bundle
    with _lock:
        if _bundle is not None:
            return _bundle
        try:
            from training import load_bundle

            _bundle = load_bundle(BUNDLE_NAME)
            _load_error = None
            print(f"[model_service] loaded bundle {BUNDLE_NAME!r}: "
                  f"{type(_bundle.model).__name__}, "
                  f"{len(_bundle.feature_names)} features, "
                  f"{len(_bundle.encoder.classes_)} classes")
        except Exception as exc:                      # noqa: BLE001
            _load_error = f"{type(exc).__name__}: {exc}"
            print(f"[model_service] MODEL UNAVAILABLE -- {_load_error}")
    return _bundle


def model_status() -> Dict[str, Any]:
    b = get_bundle()
    if b is None:
        return {"available": False, "error": _load_error, "bundle": BUNDLE_NAME}
    return {
        "available": True,
        "bundle": BUNDLE_NAME,
        "model": type(b.model).__name__,
        "n_features": len(b.feature_names),
        "classes": [str(c) for c in b.encoder.classes_],
        "created": b.meta.get("created"),
        "protocol": b.meta.get("protocol"),
    }


class ModelUnavailable(RuntimeError):
    pass


class FeatureMismatch(ValueError):
    pass


def _require_bundle():
    b = get_bundle()
    if b is None:
        raise ModelUnavailable(_load_error or "model bundle could not be loaded")
    return b


def _to_matrix(df: pd.DataFrame, bundle) -> tuple[np.ndarray, int]:
    """Align, impute with TRAINING medians, scale. Returns (X, n_imputed)."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [c for c in bundle.feature_names if c not in df.columns]
    if missing:
        raise FeatureMismatch(
            f"{len(missing)} required features are absent: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
            "Filling them with 0 would produce a z-score of -mean/std, which is "
            "a confidently wrong value rather than a neutral one."
        )

    X = df.loc[:, bundle.feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    n_imputed = int(X.isna().any(axis=1).sum())

    if bundle.cleaner is not None:
        X = bundle.cleaner.transform(X)               # training medians
    else:
        X = X.fillna(0.0)

    A = bundle.scaler.transform(X.to_numpy(dtype=np.float32)).astype(np.float32)
    np.nan_to_num(A, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(A, -10.0, 10.0), n_imputed


def _novelty(bundle, X: np.ndarray) -> Optional[np.ndarray]:
    """Novelty score if the bundle carries a fitted rejector, else None."""
    scorer = bundle.meta.get("_scorer")
    if scorer is None:
        return None
    try:
        return scorer.score(X)
    except Exception:                                 # noqa: BLE001
        return None


def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single flow."""
    bundle = _require_bundle()
    X, _ = _to_matrix(pd.DataFrame([features]), bundle)

    proba = bundle.model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = str(bundle.encoder.inverse_transform([idx])[0])

    out: Dict[str, Any] = {
        "prediction": label,
        "closed_set_confidence": float(proba[idx]),
        "class_probabilities": {
            str(c): float(p) for c, p in zip(bundle.encoder.classes_, proba)
        },
    }

    tau = bundle.meta.get("novelty_tau")
    score = _novelty(bundle, X)
    if tau is not None and score is not None:
        out["novelty_score"] = float(score[0])
        out["is_known"] = bool(score[0] <= float(tau))
        if not out["is_known"]:
            out["prediction"] = "Unknown_Attack"
    else:
        out["is_known"] = None
        out["note"] = (
            "closed_set_confidence is a posterior over known classes only; it "
            "cannot express 'none of the above'. Train with src/run.py to attach "
            "a calibrated novelty threshold."
        )
    return out


def predict_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Score a CSV upload. Returns a summary plus the per-row predictions."""
    bundle = _require_bundle()

    n_in = len(df)
    if n_in == 0:
        return {"total_records": 0, "imputed_rows": 0, "summary": {}, "predictions": []}

    drop = ["Flow ID", "Source IP", "Src IP", "Source Port", "Src Port",
            "Destination IP", "Dst IP", "Destination Port", "Dst Port",
            "Timestamp", "Protocol", "Label"]
    df = df.drop(columns=drop, errors="ignore")

    X, n_imputed = _to_matrix(df, bundle)
    proba = bundle.model.predict_proba(X)
    idx = proba.argmax(axis=1)
    labels = np.asarray(bundle.encoder.inverse_transform(idx), dtype=object)
    conf = proba.max(axis=1)

    tau = bundle.meta.get("novelty_tau")
    score = _novelty(bundle, X)
    if tau is not None and score is not None:
        labels = labels.copy()
        labels[score > float(tau)] = "Unknown_Attack"

    return {
        "total_records": int(n_in),
        "imputed_rows": n_imputed,
        "summary": pd.Series(labels.astype(str)).value_counts().to_dict(),
        "predictions": [
            {"prediction": str(l), "closed_set_confidence": float(c)}
            for l, c in zip(labels, conf)
        ],
    }
