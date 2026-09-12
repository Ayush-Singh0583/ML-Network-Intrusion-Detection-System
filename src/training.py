"""
Classical baselines and artifact persistence.

Fixes applied to the estimators:

RandomForest  - was ``n_estimators=100`` with unbounded depth, no
                ``min_samples_leaf`` and no class weighting.  On ~2M rows that
                grows until leaves are pure (150 MB pickle) and produces
                ``predict_proba`` outputs that are mostly exactly 1.0, which is
                why a 0.90 confidence threshold rejected nothing.

XGBoost       - ``eval_metric="logloss"`` is the BINARY metric on a 12-class
                problem; ``objective`` was unset, so the same helper silently
                produced ``multi:softprob`` for the multiclass caller and
                ``binary:logistic`` for the binary caller; ``tree_method`` was
                left at the default (the tuner set "hist", the trainer did not).

LightGBM      - ``subsample=0.8`` was a NO-OP.  In LightGBM ``subsample`` maps
                to ``bagging_fraction``, which only activates when
                ``bagging_freq > 0``; the sklearn wrapper's ``subsample_freq``
                defaults to 0.  Row subsampling had never run.
                ``num_leaves=255`` with ``max_depth=-1`` and
                ``min_child_samples=20`` also permitted a private leaf per rare
                sample under ``class_weight="balanced"``.

Artifacts     - ``save_model`` wrote four pickles with no integrity check, and
                two directories (``models/`` and ``saved_models/``) had already
                diverged.  ``load_bundle`` now verifies that the scaler width,
                the feature-name list, the cleaner and the encoder all agree
                with the model before returning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from config import ARTIFACT_DIR, LGBM_PARAMS, RF_PARAMS, SEED, XGB_PARAMS


# =====================================================================
# TRAINERS
# =====================================================================


def train_random_forest(X, y, **overrides) -> RandomForestClassifier:
    params = {**RF_PARAMS, **overrides}
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model


def train_xgboost(X, y, X_val=None, y_val=None, **overrides):
    from xgboost import XGBClassifier

    params = {**XGB_PARAMS, **overrides}
    n_classes = int(len(np.unique(y)))
    if n_classes <= 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"

    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    model = XGBClassifier(**params)

    fit_kw: Dict[str, Any] = {"sample_weight": sample_weight}
    if X_val is not None and y_val is not None:
        # early stopping needs a real validation set; there was none before
        model.set_params(early_stopping_rounds=30)
        fit_kw["eval_set"] = [(X_val, y_val)]
        fit_kw["verbose"] = False

    model.fit(X, y, **fit_kw)
    return model


def train_lightgbm(X, y, X_val=None, y_val=None, **overrides):
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    params = {**LGBM_PARAMS, **overrides}
    n_classes = int(len(np.unique(y)))
    if n_classes <= 2:
        params["objective"] = "binary"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes

    model = LGBMClassifier(**params)
    fit_kw: Dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kw["eval_set"] = [(X_val, y_val)]
        fit_kw["callbacks"] = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
    model.fit(X, y, **fit_kw)
    return model


def train_logistic(X, y, **overrides) -> LogisticRegression:
    params = dict(
        max_iter=2000,
        class_weight="balanced",   # was absent on an 80%-benign dataset
        solver="saga",             # the only solver that scales here
        n_jobs=-1,
        random_state=SEED,
    )
    params.update(overrides)
    model = LogisticRegression(**params)
    model.fit(X, y)
    return model


def train_decision_tree(X, y, **overrides) -> DecisionTreeClassifier:
    params = dict(
        max_depth=24,              # was unbounded
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
    )
    params.update(overrides)
    model = DecisionTreeClassifier(**params)
    model.fit(X, y)
    return model


SKLEARN_TRAINERS = {
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "logistic": train_logistic,
    "decision_tree": train_decision_tree,
}


# =====================================================================
# ARTIFACTS
# =====================================================================


@dataclass
class Bundle:
    model: Any
    scaler: Any
    encoder: Any
    cleaner: Any
    feature_names: List[str]
    meta: Dict[str, Any]

    def check(self) -> None:
        n = len(self.feature_names)
        if getattr(self.scaler, "n_features_in_", n) != n:
            raise ValueError(
                f"scaler expects {self.scaler.n_features_in_} features, "
                f"feature_names has {n}"
            )
        if self.cleaner is not None and list(self.cleaner.feature_names_) != list(self.feature_names):
            raise ValueError("cleaner.feature_names_ does not match feature_names")

        # Names, not just arity.  Comparing counts alone let a 70-feature
        # artifact whose feature LIST was disjoint in 8 positions pass every
        # check, load cleanly, and serve confidently wrong predictions.
        manifest_names = self.meta.get("feature_names")
        if manifest_names is not None and list(manifest_names) != list(self.feature_names):
            raise ValueError(
                "manifest feature_names disagree with the bundle's feature_names"
            )

        n_model = getattr(self.model, "n_features_in_", None)
        if n_model is not None and int(n_model) != n:
            raise ValueError(f"model expects {n_model} features, bundle has {n}")

        n_cls_model = getattr(self.model, "n_classes_", None)
        if n_cls_model is not None and int(n_cls_model) != len(self.encoder.classes_):
            raise ValueError(
                f"model has {n_cls_model} classes, encoder has {len(self.encoder.classes_)}"
            )


def save_bundle(
    model,
    scaler,
    encoder,
    cleaner,
    feature_names: Sequence[str],
    name: str,
    out_root: Path = ARTIFACT_DIR,
    extra_meta: Optional[Dict[str, Any]] = None,
    scorer: Any = None,
    novelty_tau: Optional[float] = None,
) -> Path:
    """
    Single artifact store.  Everything a prediction needs lives in one
    directory with a manifest, so ``models/`` vs ``saved_models/`` drift is
    structurally impossible.
    """
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(feature_names)
    bundle = Bundle(model, scaler, encoder, cleaner, feature_names, {})
    bundle.check()

    joblib.dump(model, out_dir / "model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(encoder, out_dir / "label_encoder.pkl")
    joblib.dump(cleaner, out_dir / "cleaner.pkl")
    joblib.dump(feature_names, out_dir / "feature_names.pkl")

    # THE REJECTOR MUST SHIP WITH THE MODEL.
    #
    # Previously the training run fitted a MahalanobisScorer, calibrated tau on
    # the validation day, printed the detection metrics -- and then threw both
    # away.  ``model_service`` looks for them via ``bundle.meta["_scorer"]`` and
    # ``bundle.meta["novelty_tau"]``; meta is loaded from manifest.json, and a
    # fitted scorer cannot survive a JSON round-trip, so ``_scorer`` was ALWAYS
    # None.  The novelty branch in predict() was unreachable code: every served
    # prediction came back with is_known=None and no rejection applied.
    #
    # The open-set work was measured in the research pipeline and disconnected
    # from the product.  A scorer is part of the model, not part of the report.
    if scorer is not None:
        joblib.dump(scorer, out_dir / "scorer.pkl")

    meta = {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_features": len(feature_names),
        "classes": list(map(str, encoder.classes_)),
        "model_class": type(model).__name__,
        "feature_names": feature_names,
        "novelty_tau": float(novelty_tau) if novelty_tau is not None else None,
        "has_scorer": scorer is not None,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"\nArtifact bundle saved: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")
    return out_dir


def load_bundle(name_or_path, out_root: Path = ARTIFACT_DIR) -> Bundle:
    path = Path(name_or_path)
    if not path.exists():
        path = Path(out_root) / str(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"artifact bundle not found: {name_or_path}")

    required = ["model.pkl", "scaler.pkl", "label_encoder.pkl", "feature_names.pkl"]
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is incomplete, missing: {missing}")

    cleaner_path = path / "cleaner.pkl"
    manifest_path = path / "manifest.json"

    bundle = Bundle(
        model=joblib.load(path / "model.pkl"),
        scaler=joblib.load(path / "scaler.pkl"),
        encoder=joblib.load(path / "label_encoder.pkl"),
        cleaner=joblib.load(cleaner_path) if cleaner_path.exists() else None,
        feature_names=list(joblib.load(path / "feature_names.pkl")),
        meta=json.loads(manifest_path.read_text()) if manifest_path.exists() else {},
    )

    # Rehydrate the rejector into meta under the key model_service looks for.
    # It cannot live in manifest.json -- a fitted estimator is not JSON.
    scorer_path = path / "scorer.pkl"
    if scorer_path.exists():
        bundle.meta["_scorer"] = joblib.load(scorer_path)

    bundle.check()
    return bundle


# =====================================================================
# DEPRECATED SHIM
# =====================================================================


def save_model(*args, **kwargs):  # pragma: no cover
    raise RuntimeError(
        "save_model() has been replaced by save_bundle(model, scaler, encoder, "
        "cleaner, feature_names, name). The old function wrote four unchecked "
        "pickles and allowed models/ and saved_models/ to diverge."
    )
