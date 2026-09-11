"""
Compatibility shim. This file was 0 bytes.

Deep SVDD helpers now live in:
    models.py  -> DeepSVDD, init_center (with the epsilon guard)
    engine.py  -> train_deep_svdd, svdd_distances
"""

from engine import svdd_distances, train_deep_svdd  # noqa: F401
from models import DeepSVDD, init_center  # noqa: F401

__all__ = ["DeepSVDD", "init_center", "train_deep_svdd", "svdd_distances"]
