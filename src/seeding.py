"""
Reproducibility.

The old codebase set ``random_state=42`` on sklearn splits and estimators and
nothing else.  That fixes the split and the tree construction; it does nothing
for PyTorch.  Nothing seeded the encoder's weight initialisation, the
DataLoader's shuffle order, or cuDNN algorithm selection, so two runs of
``train_svdd.py`` produced two different networks and there was no record of
which one produced the checkpoint on disk.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG this project can reach."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS workspace config is required for deterministic matmuls on CUDA
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct, reproducible seed."""
    import torch

    base = torch.initial_seed() % (2 ** 31)
    seed = (base + worker_id) % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int = 42):
    """Deterministic generator for DataLoader shuffling."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def git_sha(default: str = "nogit") -> str:
    """Short git SHA of the working tree, for run directory naming."""
    import subprocess
    from pathlib import Path

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or default
    except Exception:
        pass
    return default
