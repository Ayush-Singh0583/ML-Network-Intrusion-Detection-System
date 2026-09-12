"""
Backend configuration.

The previous version pointed at four loose pickles in ``models/``:

    MODEL_PATH   = MODELS_DIR / "random_forest_model.pkl"
    SCALER_PATH  = MODELS_DIR / "scaler.pkl"
    ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
    FEATURE_PATH = MODELS_DIR / "feature_names.pkl"

That directory had diverged from ``saved_models/``: the four artifacts above
were dated 6 July while the preprocessing that defines their 69 features had
changed repeatedly since, and nothing verified that they still agreed with each
other.  The API was serving a model whose feature contract could not be checked.

There is now one artifact store, ``saved_models/<bundle>/``, and a bundle is
loaded as a unit with its cleaner and a manifest.  Set ``NIDS_BUNDLE`` to serve
a different one without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR / "data"

# Name of the bundle under saved_models/ to serve.
# Produced by:  python src/run.py train --model rf --protocol crossday
BUNDLE_NAME = os.environ.get("NIDS_BUNDLE", "random_forest_crossday")

# CORS origins for the React dev server / deployment
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("NIDS_CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]

DATABASE_URL = os.environ.get("NIDS_DB_URL", f"sqlite:///{BASE_DIR / 'network_ids.db'}")

# ---------------------------------------------------------------------
# Deprecated names.  Kept so an un-migrated import fails with an
# explanation instead of a FileNotFoundError deep inside joblib.
# ---------------------------------------------------------------------


def _deprecated(name: str):
    raise ImportError(
        f"backend.config.{name} has been removed. The four loose pickles in "
        "models/ are replaced by a single verified bundle in saved_models/. "
        "Use BUNDLE_NAME with training.load_bundle()."
    )


class _Removed:
    def __init__(self, name):
        self._name = name

    def __fspath__(self):
        _deprecated(self._name)

    def __str__(self):
        _deprecated(self._name)


MODEL_PATH = _Removed("MODEL_PATH")
SCALER_PATH = _Removed("SCALER_PATH")
ENCODER_PATH = _Removed("ENCODER_PATH")
FEATURE_PATH = _Removed("FEATURE_PATH")
MODELS_DIR = _Removed("MODELS_DIR")
