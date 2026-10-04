"""
Every detector in the study, reduced to the same contract.

A *layer* takes a ``SplitBundle`` and returns one score per validation row and
one per test row, higher meaning more suspicious.  That is all the evaluation
needs: a false-alarm budget becomes a quantile of the validation benign scores,
and detection is the share of each attack class scoring above it.

Three kinds of layer produce such a score:

``classifier_scores``  A supervised multi-class model (XGBoost, Random Forest,
                       LightGBM, or the MLP / CNN / LSTM from ``models.py``).
                       The score is ``max P(attack class)``, the same knob
                       ``run.py`` uses (see ``max_attack_probability``).

``novelty_scores``     A model that has seen BENIGN traffic only: autoencoder
                       reconstruction error, Deep SVDD distance, Isolation
                       Forest depth, PCA reconstruction error, Mahalanobis
                       distance.  Every attack class is unseen to it by
                       construction.

``behaviour_scores``   No model at all: distinct ports and hosts per source per
                       window (``behaviour.py``).

Putting the classical one-class baselines (Isolation Forest, PCA, Mahalanobis)
beside the two deep ones is deliberate.  "The deep model adds X" is only a
finding if a linear model of the same bottleneck width does not add X too.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from config import (
    BEHAVIOUR_HOST_THRESHOLD,
    BEHAVIOUR_PORT_THRESHOLD,
    BEHAVIOUR_WINDOW_SECONDS,
    BENIGN_LABEL,
    IFOREST_PARAMS,
    TrainConfig,
    resolve_device,
)
from preprocessing import SplitBundle, benign_index
from seeding import set_seed

SKLEARN_ALIASES = {
    "rf": "random_forest", "random_forest": "random_forest",
    "xgb": "xgboost", "xgboost": "xgboost",
    "lgbm": "lightgbm", "lightgbm": "lightgbm",
    "lr": "logistic", "logistic": "logistic",
    "dt": "decision_tree", "decision_tree": "decision_tree",
}
TORCH_CLASSIFIERS = ("mlp", "cnn", "lstm")
NOVELTY_DETECTORS = ("autoencoder", "deep_svdd", "iforest", "pca", "mahalanobis")
DEEP_NOVELTY = ("autoencoder", "deep_svdd")


@dataclass
class LayerScores:
    name: str
    val: np.ndarray                 # one score per validation row
    test: np.ndarray                # one score per test row
    extra: Dict[str, object] = field(default_factory=dict)

    def val_benign(self, bundle: SplitBundle) -> np.ndarray:
        """The calibration sample: validation scores of BENIGN rows only."""
        return self.val[np.asarray(bundle.y_val_str, dtype=object) == BENIGN_LABEL]


# =====================================================================
# SUPERVISED CLASSIFIERS
# =====================================================================


def max_attack_probability(proba: np.ndarray, benign_idx: int) -> np.ndarray:
    """``max P(attack class)``: the probability of the most likely attack class.

    NOT ``1 - P(BENIGN)``.  With a dozen classes that sum rewards a flow whose
    probability is spread thinly over ten attack classes as much as one a
    single class confidently claims.  Diffuse mass is noise; concentrated mass
    is signal.  ``run.py`` records the comparison that motivated the choice.
    """
    q = np.array(proba, dtype=np.float64, copy=True)
    if q.ndim != 2 or q.shape[1] < 2:
        raise ValueError(f"expected an (n, n_classes>=2) probability matrix, got {q.shape}")
    q[:, benign_idx] = -1.0
    return q.max(axis=1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_sklearn(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    seed: int = 42,
    overrides: Optional[dict] = None,
):
    """Fit one of the classical estimators with the project's own settings.

    ``y`` must be contiguous integers 0..K-1 (XGBoost insists).  The boosted
    models take the validation pair for early stopping; the rest ignore it.
    """
    from training import SKLEARN_TRAINERS

    key = SKLEARN_ALIASES[model.lower()]
    kw = dict(overrides or {})
    kw.setdefault("random_state", int(seed))
    trainer = SKLEARN_TRAINERS[key]
    if key in ("xgboost", "lightgbm") and X_val is not None and y_val is not None and len(y_val):
        return trainer(X, y, X_val, y_val, **kw)
    return trainer(X, y, **kw)


def classifier_scores(
    bundle: SplitBundle,
    model: str = "xgb",
    seed: int = 42,
    cfg: Optional[TrainConfig] = None,
    work_dir: Optional[Path] = None,
    log: Callable[[str], None] = print,
    overrides: Optional[dict] = None,
    keep_proba: bool = False,
) -> LayerScores:
    """Fit one supervised classifier on the bundle's training split and score
    validation and test.  ``extra`` carries the argmax predictions, so the
    caller can also report closed-set accuracy from the same fit.  With
    ``keep_proba`` it also carries the full class-probability matrices
    (``proba_val``, ``proba_test``), which the recalibration experiment needs."""
    cfg = cfg or TrainConfig()
    name = model.lower()
    bi = benign_index(bundle.encoder)
    t0 = time.time()
    set_seed(seed)

    if name in SKLEARN_ALIASES:
        est = fit_sklearn(name, bundle.X_train, bundle.y_train, bundle.X_val, bundle.y_val,
                          seed=seed, overrides=overrides)
        p_val = est.predict_proba(bundle.X_val)
        p_test = est.predict_proba(bundle.X_test)
        extra = {"estimator": est}

    elif name in TORCH_CLASSIFIERS:
        from datasets import build_loaders
        from engine import predict_logits, train_classifier
        from losses import build_loss
        from models import build_model

        if work_dir is None:
            raise ValueError("a neural classifier needs work_dir for its checkpoint")
        device = resolve_device(cfg.device)
        cfg.seed = int(seed)
        net = build_model(name, bundle.n_features, bundle.n_classes, cfg)
        tr = build_loaders(bundle.X_train, bundle.y_train, cfg.batch_size, True, cfg, device,
                           balanced=cfg.sampler_power > 0.0)
        va = build_loaders(bundle.X_val, bundle.y_val, cfg.eval_batch_size, False, cfg, device)
        te = build_loaders(bundle.X_test, None, cfg.eval_batch_size, False, cfg, device)
        train_classifier(net, tr, va, bundle.y_val, build_loss(cfg, bundle.y_train, bundle.n_classes),
                         cfg, device, Path(work_dir) / f"{name}_seed{seed}.pt", log=log,
                         class_names=[str(c) for c in bundle.encoder.classes_])
        p_val = _softmax(predict_logits(net, va, device))
        p_test = _softmax(predict_logits(net, te, device))
        extra = {}
    else:
        raise ValueError(
            f"unknown classifier {model!r}; expected one of "
            f"{sorted(set(SKLEARN_ALIASES))} or {list(TORCH_CLASSIFIERS)}"
        )

    classes = np.asarray(bundle.encoder.classes_, dtype=object)
    extra.update({
        # the other obvious knob, kept so the choice can be measured, not asserted
        "one_minus_benign_val": 1.0 - np.asarray(p_val, dtype=np.float64)[:, bi],
        "one_minus_benign_test": 1.0 - np.asarray(p_test, dtype=np.float64)[:, bi],
        "y_pred_test": classes[p_test.argmax(axis=1)],
        "y_pred_val": classes[p_val.argmax(axis=1)],
        "seconds": time.time() - t0,
        "model": name, "seed": int(seed), "benign_index": int(bi),
    })
    if keep_proba:
        extra["proba_val"] = np.asarray(p_val, dtype=np.float32)
        extra["proba_test"] = np.asarray(p_test, dtype=np.float32)
    return LayerScores(
        name=f"classifier:{name}",
        val=max_attack_probability(p_val, bi),
        test=max_attack_probability(p_test, bi),
        extra=extra,
    )


def isotonic_recalibration(proba_fit: np.ndarray, y_fit: np.ndarray):
    """Per-class isotonic recalibration, then renormalisation.

    This is what ``CalibratedClassifierCV(method="isotonic")`` does to a
    fitted multi-class model: one monotone map per class, fitted one-vs-rest,
    and each row divided by its sum afterwards.  Returns a function that
    applies the fitted maps to a probability matrix.

    It is written out here, not imported, for two reasons.  The scikit-learn
    wrapper for an already-fitted model has changed its interface between
    releases; and the experiment needs the calibration and the threshold to be
    fitted on DIFFERENT validation flows, which is easier to guarantee when
    the two steps are separate calls.

    Note what this is NOT.  After renormalisation the detection score (the
    largest attack probability) is no longer a monotone function of the
    uncalibrated score, so the "a monotone map cannot move a quantile
    threshold" argument does not cover it.  Whether it changes the realised
    false-alarm rate is an empirical question, and experiment E6 asks it.
    """
    from sklearn.isotonic import IsotonicRegression

    P = np.asarray(proba_fit, dtype=np.float64)
    y = np.asarray(y_fit)
    maps = []
    for k in range(P.shape[1]):
        target = (y == k).astype(np.float64)
        if target.min() == target.max():
            # a class that never (or always) occurs in the fitting sample: an
            # isotonic fit to a constant target is that constant
            maps.append(float(target[0]) if target.size else 0.0)
            continue
        maps.append(IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    .fit(P[:, k], target))

    def apply(proba: np.ndarray) -> np.ndarray:
        Q = np.array(proba, dtype=np.float64, copy=True)
        for k, m in enumerate(maps):
            Q[:, k] = m if isinstance(m, float) else m.predict(Q[:, k])
        total = Q.sum(axis=1, keepdims=True)
        zero = total[:, 0] <= 0
        if zero.any():                           # every class mapped to 0: fall back
            Q[zero] = np.asarray(proba, dtype=np.float64)[zero]
            total = Q.sum(axis=1, keepdims=True)
        return Q / total

    return apply


# =====================================================================
# BENIGN-ONLY NOVELTY DETECTORS
# =====================================================================


def _chunked(fn, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    return np.concatenate([np.asarray(fn(X[i:i + chunk]), dtype=np.float64)
                           for i in range(0, len(X), chunk)]) if len(X) else np.empty(0)


def novelty_scores(
    bundle: SplitBundle,
    detector: str = "autoencoder",
    seed: int = 42,
    cfg: Optional[TrainConfig] = None,
    work_dir: Optional[Path] = None,
    log: Callable[[str], None] = print,
    max_fit_rows: int = 500_000,
) -> LayerScores:
    """Fit a detector on BENIGN training rows only; score validation and test.

    The attack rows of the training split are not shown to these models, not
    even as negatives.  Whatever they detect, they detect as "unlike benign".
    """
    cfg = cfg or TrainConfig()
    name = detector.lower()
    if name not in NOVELTY_DETECTORS:
        raise ValueError(f"unknown detector {detector!r}; expected one of {NOVELTY_DETECTORS}")

    bi = benign_index(bundle.encoder)
    Xtr = bundle.X_train[bundle.y_train == bi]
    Xva_b = bundle.X_val[bundle.y_val == bi]
    if len(Xtr) == 0 or len(Xva_b) == 0:
        raise ValueError("no BENIGN rows in the training or validation split")

    t0 = time.time()
    set_seed(seed)
    rng = np.random.default_rng(seed)

    def fit_rows() -> np.ndarray:
        if len(Xtr) <= max_fit_rows:
            return Xtr
        return Xtr[np.sort(rng.choice(len(Xtr), size=max_fit_rows, replace=False))]

    extra: Dict[str, object] = {"n_benign_train": int(len(Xtr))}

    if name == "iforest":
        from sklearn.ensemble import IsolationForest

        params = {**IFOREST_PARAMS, "random_state": int(seed)}
        est = IsolationForest(**params).fit(fit_rows())
        score = lambda X: -est.score_samples(X)               # noqa: E731
        s_val, s_test = _chunked(score, bundle.X_val), _chunked(score, bundle.X_test)

    elif name == "pca":
        from sklearn.decomposition import PCA

        k = int(min(cfg.latent_dim, bundle.n_features - 1))
        est = PCA(n_components=k, random_state=int(seed)).fit(fit_rows())
        extra["explained_variance"] = float(est.explained_variance_ratio_.sum())
        extra["n_components"] = k

        def score(X):
            X = np.asarray(X, dtype=np.float64)
            return ((X - est.inverse_transform(est.transform(X))) ** 2).mean(axis=1)

        s_val, s_test = _chunked(score, bundle.X_val), _chunked(score, bundle.X_test)

    elif name == "mahalanobis":
        from openset import MahalanobisScorer

        sc = MahalanobisScorer().fit(Xtr, np.zeros(len(Xtr), dtype=np.int64),
                                     random_state=int(seed))
        s_val, s_test = sc.score(bundle.X_val), sc.score(bundle.X_test)

    else:
        from datasets import build_loaders
        from engine import (reconstruction_errors, svdd_distances, train_autoencoder,
                            train_deep_svdd)
        from models import Autoencoder, build_model, init_center

        if work_dir is None:
            raise ValueError("a neural detector needs work_dir for its checkpoint")
        work_dir = Path(work_dir)
        device = resolve_device(cfg.device)
        cfg.seed = int(seed)
        tr = build_loaders(Xtr, None, cfg.batch_size, True, cfg, device)
        va_b = build_loaders(Xva_b, None, cfg.eval_batch_size, False, cfg, device)
        va = build_loaders(bundle.X_val, None, cfg.eval_batch_size, False, cfg, device)
        te = build_loaders(bundle.X_test, None, cfg.eval_batch_size, False, cfg, device)

        if name == "autoencoder":
            net = build_model("autoencoder", bundle.n_features, bundle.n_classes, cfg)
            train_autoencoder(net, tr, va_b, cfg, device, work_dir / f"ae_seed{seed}.pt", log=log)
            s_val = reconstruction_errors(net, va, device)
            s_test = reconstruction_errors(net, te, device)
        else:
            net = build_model("deep_svdd", bundle.n_features, bundle.n_classes, cfg).to(device)
            ae = Autoencoder(bundle.n_features, latent_dim=cfg.latent_dim).to(device)
            ae_cfg = TrainConfig(**{**cfg.to_dict(), "epochs": max(cfg.epochs // 2, 5)})
            train_autoencoder(ae, tr, va_b, ae_cfg, device,
                              work_dir / f"svdd_pretrain_seed{seed}.pt", log=log)
            net.load_pretrained_encoder(ae)
            center = init_center(net, tr, device, eps=0.1)
            _hist, center = train_deep_svdd(net, center, tr, va_b, cfg, device,
                                            work_dir / f"svdd_seed{seed}.pt", log=log)
            s_val = svdd_distances(net, center, va, device)
            s_test = svdd_distances(net, center, te, device)

    extra.update({"seconds": time.time() - t0, "detector": name, "seed": int(seed)})
    return LayerScores(name=f"novelty:{name}",
                       val=np.asarray(s_val, dtype=np.float64),
                       test=np.asarray(s_test, dtype=np.float64), extra=extra)


# =====================================================================
# BEHAVIOUR
# =====================================================================


def behaviour_scores(
    bundle: SplitBundle,
    week=None,
    window_seconds: float = BEHAVIOUR_WINDOW_SECONDS,
    canonical: bool = True,
    port_threshold: int = BEHAVIOUR_PORT_THRESHOLD,
    host_threshold: int = BEHAVIOUR_HOST_THRESHOLD,
) -> LayerScores:
    """Per-source window counts for the validation and test splits.

    ``week`` is a ``behaviour.WeekBehaviour`` built over every flow in the
    cache; each split looks its rows up in it.  That is the correct way to
    run this layer and the study always passes it (see ``WeekBehaviour`` for
    what goes wrong otherwise).

    Without ``week`` the counts are taken over the flows of each split alone.
    That is exact only when a split holds every flow of its time range --
    true of a whole test day, not of a random half of a validation day -- and
    exists for hand-built frames in the tests.
    """
    from behaviour import behaviour_layer_scores
    from meta import require_identifiers

    require_identifiers(bundle.meta_val, "The behaviour layer (validation split)")
    require_identifiers(bundle.meta_test, "The behaviour layer (test split)")
    t0 = time.time()
    kw = dict(window_seconds=window_seconds, canonical=canonical,
              port_threshold=port_threshold, host_threshold=host_threshold)
    if week is not None:
        s_val, c_val = week.lookup(bundle.meta_val)
        s_test, c_test = week.lookup(bundle.meta_test)
        kw = dict(week.params)
    else:
        s_val, c_val = behaviour_layer_scores(bundle.meta_val, **kw)
        s_test, c_test = behaviour_layer_scores(bundle.meta_test, **kw)
    return LayerScores(
        name="behaviour", val=np.asarray(s_val, dtype=np.float64),
        test=np.asarray(s_test, dtype=np.float64),
        extra={"counts_val": c_val, "counts_test": c_test, "seconds": time.time() - t0,
               "whole_week_counts": week is not None, **kw},
    )
