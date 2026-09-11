#!/usr/bin/env python
"""
Inference against a saved artifact bundle.

Replaces evalAlone.py, which crashed at line 150 with
``ValueError: Mix of label input types (string and number)`` because it passed
integer-encoded ``y_test`` and string ``y_pred`` to ``confusion_matrix``, and
which re-derived "the" test split by re-running preprocessing code that had
changed since the model was trained.

Here the bundle carries its own cleaner, scaler, encoder and feature list, and
they are cross-checked before a single row is scored.

    python src/predict.py --bundle random_forest_crossday --csv flows.csv
    python src/predict.py --bundle mlp_crossday --protocol crossday --evaluate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from config import DAY_COL, LABEL_COL, UNKNOWN_LABEL
from evaluation import decode_predictions, evaluate_predictions, new_run_dir, plot_confusion
from preprocessing import build_splits, open_set_truth, structural_clean
from training import load_bundle


def prepare_frame(df: pd.DataFrame, bundle) -> np.ndarray:
    """Clean and scale an arbitrary CSV using the bundle's own fitted objects."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [c for c in bundle.feature_names if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} features required by this model are absent from the "
            f"input: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            "Filling them with 0 would map to a z-score of -mean/std, i.e. a "
            "confidently wrong value, not a neutral one. Refusing."
        )

    X = df.loc[:, bundle.feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)

    if bundle.cleaner is not None:
        X = bundle.cleaner.transform(X)
    else:
        X = X.fillna(0.0)

    A = bundle.scaler.transform(X.to_numpy(dtype=np.float32)).astype(np.float32)
    np.nan_to_num(A, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(A, -10.0, 10.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="predict with a saved bundle")
    ap.add_argument("--bundle", required=True, help="saved_models/<name> or a path")
    ap.add_argument("--csv", help="a CICFlowMeter CSV to score")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--evaluate", action="store_true",
                    help="score the protocol's held-out test split instead")
    ap.add_argument("--protocol", default="crossday", choices=["crossday", "closedset"])
    args = ap.parse_args(argv)

    bundle = load_bundle(args.bundle)          # raises if the artifacts disagree
    print(f"Loaded bundle : {args.bundle}")
    print(f"Model         : {type(bundle.model).__name__}")
    print(f"Features      : {len(bundle.feature_names)}")
    print(f"Classes       : {list(bundle.encoder.classes_)}")

    if args.evaluate:
        split = build_splits(protocol=args.protocol)
        if list(split.feature_names) != list(bundle.feature_names):
            raise ValueError(
                "The current preprocessing produces different features than this "
                "bundle was trained on. Retrain, or evaluate with --csv instead.\n"
                f"  bundle: {len(bundle.feature_names)}  now: {len(split.feature_names)}"
            )
        proba = bundle.model.predict_proba(split.X_test)
        y_pred = decode_predictions(proba.argmax(1), bundle.encoder)
        y_true = open_set_truth(split.y_test_str, bundle.encoder.classes_)

        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        rep = evaluate_predictions(y_true, y_pred, tag="predict_eval", labels=labels)
        print(rep)
        run_dir = new_run_dir(f"predict-{Path(args.bundle).name}")
        rep.save(run_dir)
        plot_confusion(rep, run_dir / "confusion.png")
        print(f"\nSaved: {run_dir}")
        return 0

    if not args.csv:
        ap.error("pass --csv <file> or --evaluate")

    df = pd.read_csv(args.csv, low_memory=False)
    X = prepare_frame(df, bundle)
    proba = bundle.model.predict_proba(X)
    y_pred = decode_predictions(proba.argmax(1), bundle.encoder)

    out = pd.DataFrame({
        "prediction": y_pred.astype(str),
        "confidence": proba.max(1),
    })
    out.to_csv(args.out, index=False)
    print(f"\n{len(out):,} rows scored -> {args.out}")
    print(out["prediction"].value_counts().to_string())
    print(
        "\n[note] 'confidence' is a closed-set posterior. It cannot express "
        "'none of the above'.\n       For unknown-attack detection use "
        "src/run.py, which calibrates a rejection\n       threshold on the "
        "validation day and reports AUROC."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
