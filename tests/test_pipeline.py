"""
Regression tests.  Each one encodes a bug this refactor fixed, so a future
change that reintroduces it fails here instead of in a paper.

Run:  python -m pytest tests/ -v      (or:  python tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from sklearn.preprocessing import LabelEncoder

from config import DUPLICATE_FEATURES, TrainConfig
from evaluation import decode_predictions, evaluate_predictions
from losses import FocalLoss, class_weights
from models import Autoencoder, CNN1D, DeepSVDD, LSTMNet, MLP, init_center
from openset import MahalanobisScorer, energy_score, open_set_report, threshold_at_fpr
from preprocessing import FrameCleaner, benign_index, build_splits, structural_clean


# =====================================================================
# LEAKAGE
# =====================================================================


def _toy_frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    rows = []
    for d in days:
        labels = ["BENIGN"] * 300
        if d == "Tuesday":
            labels += ["FTP-Patator"] * 80
        if d == "Wednesday":
            labels += ["DoS Hulk"] * 90
        if d == "Thursday":
            labels += ["Infiltration"] * 60 + ["WebAttack_XSS"] * 60
        if d == "Friday":
            labels += ["PortScan"] * 120 + ["DDoS"] * 120     # novel
        X = rng.normal(size=(len(labels), 12)) * 50
        f = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
        f["Flow Duration"] = np.abs(rng.normal(1e5, 1e4, len(labels)))
        f["Label"] = labels
        f["Day"] = d
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def test_no_row_overlap_between_splits():
    """No flow may appear in more than one split."""
    b = build_splits(protocol="crossday", df=_toy_frame(), verbose=False)

    def hashes(X):
        return {hash(r.tobytes()) for r in np.ascontiguousarray(X)}

    tr, va, te = hashes(b.X_train), hashes(b.X_val), hashes(b.X_test)
    assert not (tr & va), f"{len(tr & va)} rows shared between train and val"
    assert not (tr & te), f"{len(tr & te)} rows shared between train and test"
    assert not (va & te), f"{len(va & te)} rows shared between val and test"


def test_scaler_fitted_on_train_only():
    """Training features must be standardised; test features must not be."""
    b = build_splits(protocol="crossday", df=_toy_frame(), verbose=False)
    assert abs(float(b.X_train.mean())) < 0.15
    assert b.scaler.n_features_in_ == len(b.feature_names)
    # a scaler fitted on train+test would make the test mean ~0 too
    assert not np.allclose(b.X_test.mean(axis=0), 0.0, atol=1e-6)


def test_cleaner_learns_nothing_from_test():
    """FrameCleaner medians and column choices must come from fit() data only."""
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": np.ones(500)})
    test = pd.DataFrame({"a": rng.normal(100, 1, 500), "b": np.ones(500)})
    c = FrameCleaner().fit(train)
    assert "b" in c.dropped_constant_          # constant in TRAIN
    assert abs(float(c.medians_["a"])) < 0.5   # median from TRAIN, not ~100
    out = c.transform(test)
    assert list(out.columns) == c.feature_names_


def test_duplicate_features_removed():
    df = _toy_frame()
    for col in DUPLICATE_FEATURES:
        df[col] = 1.0
    out = structural_clean(df, verbose=False)
    assert not (set(DUPLICATE_FEATURES) & set(out.columns))


def test_dedup_ignores_day_column():
    """A flow duplicated across two days must be removed, not kept twice."""
    base = _toy_frame(seed=3)
    row = base.iloc[[0]].copy()
    row["Day"] = "Friday"          # same features, different day
    df = pd.concat([base, row], ignore_index=True)
    out = structural_clean(df, verbose=False)
    feat = [c for c in out.columns if c not in ("Label", "Day")]
    assert out.duplicated(subset=feat).sum() == 0


def test_stray_csv_not_loaded():
    """The manifest loader must ignore files that are not in DAY_FILES."""
    import tempfile

    from preprocessing import load_all_datasets

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "custom_sample.csv").write_text("Label\nDDoS\n")
        with pytest.raises(FileNotFoundError):
            load_all_datasets(d, verbose=False)


# =====================================================================
# EVALUATION
# =====================================================================


def test_decode_predictions_cannot_truncate():
    """
    The old code used np.array(...).astype(str), producing a fixed-width array
    ('<U6' for ATTACK/BENIGN) that silently truncated 'Unknown_Attack' to
    'Unknow'.
    """
    enc = LabelEncoder().fit(["BENIGN", "ATTACK"])
    p = decode_predictions(np.array([0, 1, 0]), enc)
    p[0] = "Unknown_Attack"
    assert p[0] == "Unknown_Attack", f"truncated to {p[0]!r}"
    assert p.dtype == object


def test_metrics_share_one_ground_truth():
    """
    Every metric in a report must come from the same y_true.  Constructed so a
    mismatched ground truth would produce a different accuracy than recall.
    """
    y_true = np.array(["BENIGN"] * 8 + ["DDoS"] * 2, dtype=object)
    y_pred = np.array(["BENIGN"] * 9 + ["DDoS"] * 1, dtype=object)
    rep = evaluate_predictions(y_true, y_pred, tag="t")

    assert rep.metrics["accuracy"] == pytest.approx(0.9)
    # recall_BENIGN = 8/8 = 1.0 ; recall_DDoS = 1/2 = 0.5 -> macro 0.75
    assert rep.metrics["macro_recall"] == pytest.approx(0.75)
    assert int(rep.per_class.loc["BENIGN", "support"]) == 8
    assert int(rep.confusion.to_numpy().sum()) == 10


def test_labels_must_cover_the_data():
    y = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError):
        evaluate_predictions(y, y, labels=["a"])


def test_macro_present_excludes_unpredicted_classes():
    y_true = np.array(["a", "a", "b", "b"], dtype=object)
    y_pred = np.array(["a", "a", "b", "c"], dtype=object)
    rep = evaluate_predictions(y_true, y_pred, tag="t")
    assert rep.metrics["n_classes_true"] == 2
    assert rep.metrics["n_classes_union"] == 3
    assert rep.metrics["macro_f1_present"] > rep.metrics["macro_f1"]


# =====================================================================
# MODELS
# =====================================================================


@pytest.mark.parametrize("name", ["mlp", "cnn", "lstm"])
def test_classifier_shapes_and_dtypes(name):
    cfg = TrainConfig(hidden=64, depth=2, dropout=0.1)
    from models import build_model

    m = build_model(name, input_dim=31, n_classes=7, cfg=cfg).train()
    x = torch.randn(16, 31)
    out = m(x)
    assert out.shape == (16, 7)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert any(p.grad is not None for p in m.parameters())

    m.eval()
    feats = m.features(x)
    assert feats.ndim == 2 and feats.shape[0] == 16


def test_deep_svdd_has_no_bias_anywhere():
    """
    Ruff et al. (ICML 2018): a bias term lets the network output a constant and
    drive the objective to zero. The previous implementation used nn.Linear
    defaults (bias=True) and did collapse.
    """
    m = DeepSVDD(input_dim=20, latent_dim=8)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            assert mod.bias is None, "DeepSVDD Linear has a bias term"
        if isinstance(mod, torch.nn.BatchNorm1d):
            assert mod.weight is None and mod.bias is None, "affine BatchNorm in DeepSVDD"


def test_center_epsilon_guard():
    m = DeepSVDD(input_dim=20, latent_dim=8).eval()
    loader = [(torch.zeros(64, 20),)]     # forces every component toward 0
    c = init_center(m, loader, "cpu", eps=0.1)
    assert (c.abs() >= 0.1 - 1e-6).all(), f"eps-guard failed: {c}"


def test_collapse_guard_fires_on_a_biased_encoder():
    """
    The guard must actually detect collapse.  A biased encoder is trained on
    constant-ish data until it degenerates; train_deep_svdd must raise.
    """
    import torch.nn as nn

    from engine import train_deep_svdd

    class BiasedEncoder(nn.Module):          # the OLD architecture
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 4)
            )

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(0)
    X = torch.randn(512, 10)
    loader = [(X[i: i + 128],) for i in range(0, 512, 128)]

    model = BiasedEncoder()
    center = torch.zeros(4)
    cfg = TrainConfig(epochs=300, lr=0.05, weight_decay=0.0, warmup_epochs=0,
                      patience=999, amp=False)

    with pytest.raises(RuntimeError, match="collapse"):
        train_deep_svdd(model, center, loader, loader, cfg, "cpu",
                        Path("/tmp/_collapse_test.pt"), log=lambda *_: None)


def test_deep_svdd_does_not_collapse():
    """The fixed architecture must keep a non-degenerate distance spread."""
    from engine import train_deep_svdd

    torch.manual_seed(0)
    X = torch.randn(1024, 12)
    loader = [(X[i: i + 256],) for i in range(0, 1024, 256)]

    model = DeepSVDD(input_dim=12, latent_dim=8)
    center = init_center(model, loader, "cpu")
    cfg = TrainConfig(epochs=25, lr=1e-3, weight_decay=1e-4, warmup_epochs=2,
                      patience=999, amp=False)

    _, c = train_deep_svdd(model, center, loader, loader, cfg, "cpu",
                           Path("/tmp/_svdd_ok.pt"), log=lambda *_: None)
    with torch.no_grad():
        d = torch.sum((model(X) - c) ** 2, dim=1)
    assert float(d.std()) > 1e-4


def test_autoencoder_roundtrip():
    ae = Autoencoder(input_dim=25, latent_dim=8).eval()
    x = torch.randn(32, 25)
    assert ae(x).shape == x.shape
    err = ae.reconstruction_error(x)
    assert err.shape == (32,) and (err >= 0).all()


def test_svdd_can_warm_start_from_autoencoder():
    ae = Autoencoder(input_dim=25, latent_dim=8, hidden=(128, 64))
    svdd = DeepSVDD(input_dim=25, latent_dim=8, hidden=(128, 64))
    svdd.load_pretrained_encoder(ae)
    src = [m for m in ae.encoder if isinstance(m, torch.nn.Linear)]
    dst = [m for m in svdd.encoder if isinstance(m, torch.nn.Linear)]
    assert torch.allclose(src[0].weight, dst[0].weight)


# =====================================================================
# LOSSES
# =====================================================================


def test_focal_reduces_to_ce_at_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(64, 5)
    y = torch.randint(0, 5, (64,))
    focal = FocalLoss(gamma=0.0)(logits, y)
    ce = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.allclose(focal, ce, atol=1e-5)


def test_class_weights_favour_rare_classes():
    y = np.array([0] * 1000 + [1] * 10)
    w = class_weights(y, 2, power=0.5)
    assert w[1] > w[0]
    assert float(w.sum()) == pytest.approx(2.0, abs=1e-4)


def test_focal_is_finite_under_extreme_logits():
    logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
    y = torch.tensor([1, 0])
    assert torch.isfinite(FocalLoss(gamma=2.0)(logits, y))


# =====================================================================
# OPEN SET
# =====================================================================


def test_energy_score_orders_confident_below_uncertain():
    confident = np.array([[10.0, -10.0, -10.0]])
    uncertain = np.array([[0.1, 0.0, -0.1]])
    assert energy_score(confident)[0] < energy_score(uncertain)[0]


def test_threshold_hits_the_requested_fpr():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200_000)
    tau = threshold_at_fpr(s, 0.05)
    assert (s > tau).mean() == pytest.approx(0.05, abs=2e-3)


def test_threshold_never_calibrated_on_test():
    """tau must be a pure function of the validation scores."""
    rng = np.random.default_rng(1)
    val = rng.normal(size=5000)
    t1 = threshold_at_fpr(val, 0.05)
    t2 = threshold_at_fpr(val, 0.05)
    assert t1 == t2


def test_mahalanobis_separates_in_from_out():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (500, 6)), rng.normal(6, 1, (500, 6))])
    y = np.array([0] * 500 + [1] * 500)
    sc = MahalanobisScorer().fit(X, y)
    ood = rng.normal(40, 1, (200, 6))
    assert sc.score(ood).mean() > sc.score(X).mean() * 5


def test_open_set_report_is_consistent():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 1000), rng.normal(5, 1, 1000)])
    is_unknown = np.array([False] * 1000 + [True] * 1000)
    r = open_set_report(scores, is_unknown, threshold_at_fpr(scores[:1000], 0.05), "t")
    assert r.auroc > 0.95
    assert r.known_rejection == pytest.approx(0.05, abs=0.02)
    assert r.n_known == 1000 and r.n_unknown == 1000


# =====================================================================
# REPRODUCIBILITY
# =====================================================================


def test_seeding_makes_init_reproducible():
    from seeding import set_seed

    set_seed(123)
    a = MLP(20, 5, hidden=32, depth=1)
    set_seed(123)
    b = MLP(20, 5, hidden=32, depth=1)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_device_batcher_is_deterministic():
    from datasets import DeviceBatcher

    X = np.arange(200, dtype=np.float32).reshape(100, 2)
    y = np.arange(100)
    a = DeviceBatcher(X, y, 16, True, "cpu", seed=7)
    b = DeviceBatcher(X, y, 16, True, "cpu", seed=7)
    for (xa, ya), (xb, yb) in zip(a, b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_benign_index_is_looked_up_not_assumed():
    enc = LabelEncoder().fit(["ATTACK", "BENIGN", "ZZZ"])
    assert benign_index(enc) == 1              # not 0
    with pytest.raises(ValueError):
        benign_index(LabelEncoder().fit(["a", "b"]))


def test_splits_are_float32_and_finite():
    b = build_splits(protocol="crossday", df=_toy_frame(), verbose=False)
    b.validate()
    assert b.X_train.dtype == np.float32
    assert (b.y_test < 0).any(), "Friday should contain novel classes"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
