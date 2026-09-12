"""
Regression tests for the BACKEND SERVING CONTRACT.

Every test here pins a bug that shipped.  Deliberately free of torch,
xgboost and any trained artifact, so it runs in a bare environment -- the
existing tests/test_pipeline.py needs torch, which meant nothing in the
backend was covered by anything runnable.

THE BUG THIS FILE EXISTS FOR
----------------------------
``model_service.predict()`` was refactored to return a dict.  Its two callers
kept doing ``attack, confidence = predict(features)``.  That raises
``ValueError: too many values to unpack (expected 2)`` on every call, so:

  * POST /predict returned 500 for every request, and
  * the live cleanup worker raised on every expired flow and swallowed it in
    a broad ``except Exception``, so the dashboard stayed empty and the log
    filled with one error per flow.

The refactor was correct.  The blast radius was never checked.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services import model_service as ms  # noqa: E402


# =====================================================================
# A fake bundle: lets us exercise the contract with no trained model.
# =====================================================================


class _Encoder:
    classes_ = np.array(["BENIGN", "DDoS"])

    def inverse_transform(self, idx):
        return self.classes_[np.asarray(idx)]


class _Model:
    n_features_in_ = 3
    n_classes_ = 2

    def predict_proba(self, X):
        return np.tile([0.25, 0.75], (len(X), 1))


class _Scaler:
    n_features_in_ = 3

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class _Bundle:
    model = _Model()
    scaler = _Scaler()
    encoder = _Encoder()
    cleaner = None
    feature_names = ["a", "b", "c"]
    meta: dict = {}


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(ms, "_bundle", _Bundle())
    monkeypatch.setattr(ms, "_load_error", None)
    return _Bundle()


# =====================================================================
# THE CONTRACT
# =====================================================================


def test_predict_returns_a_mapping(served):
    out = ms.predict({"a": 1.0, "b": 2.0, "c": 3.0})
    assert isinstance(out, dict)
    for key in ("prediction", "closed_set_confidence", "class_probabilities", "is_known"):
        assert key in out, f"predict() dropped {key!r} from its contract"


def test_predict_is_not_a_two_tuple(served):
    """The exact failure that shipped."""
    with pytest.raises(ValueError, match="unpack"):
        _a, _b = ms.predict({"a": 1.0, "b": 2.0, "c": 3.0})  # noqa: F841


def test_missing_features_are_an_error_not_a_silent_zero(served):
    """Filling a missing feature with 0 becomes -mean/std after scaling:
    a confidently wrong value, not a neutral one."""
    with pytest.raises(ms.FeatureMismatch):
        ms.predict({"a": 1.0})


def test_unavailable_model_raises_rather_than_killing_the_process(monkeypatch):
    monkeypatch.setattr(ms, "_bundle", None)
    monkeypatch.setattr(ms, "_load_error", "simulated")
    monkeypatch.setattr(ms, "get_bundle", lambda: None)
    with pytest.raises(ms.ModelUnavailable):
        ms.predict({"a": 1.0, "b": 2.0, "c": 3.0})


# =====================================================================
# STATIC GUARD -- catches this bug class at EVERY call site, forever
# =====================================================================


def _predict_calls_unpacked_into_tuples(path: pathlib.Path):
    """Yield (lineno, source) for `x, y = ...predict(...)` assignments."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not node.targets or not isinstance(node.targets[0], (ast.Tuple, ast.List)):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in {"predict", "predict_dataframe"}:
            yield node.lineno, name


def test_no_call_site_unpacks_predict_into_a_tuple():
    """predict() returns a dict. Unpacking it is always a bug.

    A static check rather than a runtime one, because the runtime path only
    fails once a model is actually loaded -- which is exactly why this went
    unnoticed: the bundle failed to load first, and the 503 masked the 500.
    """
    offenders = []
    for py in (ROOT / "backend").rglob("*.py"):
        for lineno, name in _predict_calls_unpacked_into_tuples(py):
            offenders.append(f"{py.relative_to(ROOT)}:{lineno} unpacks {name}()")
    assert not offenders, "predict() returns a dict; these call sites unpack it:\n" + "\n".join(offenders)
