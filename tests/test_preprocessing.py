"""
Preprocessing / leakage regression tests -- TORCH-FREE.

Extracted from tests/test_pipeline.py, which imports torch at module scope.
That meant that in any environment without torch (a CI container, a machine
behind an egress proxy that blocks the PyTorch CDN, a laptop that only needs
to run the API) ALL 31 tests failed at collection and nothing was covered.

A test suite that cannot run is not a test suite.  Split by dependency:
these need pandas and scikit-learn only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.preprocessing import LabelEncoder

from config import DUPLICATE_FEATURES
from preprocessing import FrameCleaner, benign_index, build_splits, structural_clean


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

