#!/usr/bin/env python
"""
Hyper-parameter search.

Replaces tune_xgb.py and tune_xgboost.py -- two different experiments with
near-identical names:

    tune_xgb.py      random split  + GridSearchCV      + scoring="f1_weighted"
    tune_xgboost.py  cross-day     + RandomizedSearchCV + scoring="accuracy"

Both scored on a metric the project's own ``learnings.md`` calls a trivial
classifier trap: on 80%-benign data, ``scoring="accuracy"`` selects the
hyper-parameters that best predict BENIGN.  Both also ran ``n_jobs=1`` over the
full ~2M-row training set with no subsampling and no ``tree_method="hist"``,
which is a multi-hour to multi-day job for an 8-point grid.

This version: one script, ``f1_macro``, an explicit validation split (not CV
over the training day mixture, which leaks temporal structure), and a
stratified search subsample with a full-data refit of the winner.

    python src/tune.py --model xgb --protocol crossday --n-iter 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler, train_test_split

from config import LGBM_PARAMS, RF_PARAMS, SEED, XGB_PARAMS
from evaluation import new_run_dir, save_json
from preprocessing import build_splits
from seeding import set_seed

SPACES = {
    "xgb": {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [6, 8, 10, 12],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [0.5, 1.0, 5.0],
    },
    "lgbm": {
        "n_estimators": [200, 300, 400, 600],
        "num_leaves": [31, 63, 127],
        "max_depth": [8, 12, 16],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.8, 0.9],
        "reg_lambda": [0.5, 1.0, 5.0],
    },
    "rf": {
        "n_estimators": [200, 300, 500],
        "max_depth": [16, 24, 32, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", "log2", 0.3],
    },
}


def build(model: str, params: dict):
    if model == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(**{**XGB_PARAMS, **params})
    if model == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**{**LGBM_PARAMS, **params})
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**{**RF_PARAMS, **params})
    raise ValueError(model)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hyper-parameter search")
    ap.add_argument("--model", default="xgb", choices=list(SPACES))
    ap.add_argument("--protocol", default="crossday", choices=["crossday", "closedset"])
    ap.add_argument("--n-iter", dest="n_iter", type=int, default=25)
    ap.add_argument("--subsample", type=int, default=300_000,
                    help="stratified rows used during the search; 0 = all")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    bundle = build_splits(protocol=args.protocol, seed=args.seed)

    X, y = bundle.X_train, bundle.y_train
    if args.subsample and len(X) > args.subsample:
        counts = pd.Series(y).value_counts()
        strat = y if (counts >= 2).all() else None
        idx, _ = train_test_split(
            np.arange(len(X)), train_size=args.subsample,
            random_state=args.seed, stratify=strat,
        )
        X, y = X[idx], y[idx]
        print(f"search subsample: {len(X):,} of {len(bundle.X_train):,} rows")

    sampler = ParameterSampler(SPACES[args.model], n_iter=args.n_iter, random_state=args.seed)
    rows = []
    best_score, best_params = -np.inf, None

    print(f"\n{'#':>3} {'macro_f1(val)':>14} {'sec':>7}  params")
    print("-" * 96)
    for i, params in enumerate(sampler, 1):
        t0 = time.time()
        model = build(args.model, params)
        # NOTE: scored on the real held-out validation DAY, not on CV folds of
        # the training days. K-fold over a day mixture puts temporally adjacent
        # flows in both folds and reports an optimistic score.
        model.fit(X, y)
        pred = model.predict(bundle.X_val)
        score = float(f1_score(bundle.y_val, pred, average="macro", zero_division=0))
        dt = time.time() - t0
        rows.append({"macro_f1": score, "seconds": dt, **params})
        print(f"{i:>3} {score:>14.6f} {dt:>7.1f}  {params}")
        if score > best_score:
            best_score, best_params = score, params

    run_dir = new_run_dir(f"tune-{args.model}-{args.protocol}")
    pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(
        run_dir / "search_results.csv", index=False
    )

    print(f"\nBest macro_f1 (validation) : {best_score:.6f}")
    print(f"Best params                : {json.dumps(best_params, indent=2, default=str)}")

    print("\nRefitting the winner on the full training split...")
    final = build(args.model, best_params)
    final.fit(bundle.X_train, bundle.y_train)

    from training import save_bundle

    save_bundle(final, bundle.scaler, bundle.encoder, bundle.cleaner,
                bundle.feature_names, f"{args.model}_tuned_{args.protocol}",
                extra_meta={"best_params": best_params, "val_macro_f1": best_score})
    save_json({"best_params": best_params, "val_macro_f1": best_score,
               "n_iter": args.n_iter, "protocol": args.protocol},
              run_dir / "best.json")
    print(f"\nResults: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
