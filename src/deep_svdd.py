"""
Compatibility shim.

The DeepSVDD definition moved to models.py, where the architecture is correct:
every Linear has bias=False and every BatchNorm affine=False. The version that
used to live here used nn.Linear defaults (bias=True), which permits the
constant solution f(x) = c -- and the saved checkpoint had collapsed onto it.
"""

from models import DeepSVDD, init_center  # noqa: F401

__all__ = ["DeepSVDD", "init_center"]
