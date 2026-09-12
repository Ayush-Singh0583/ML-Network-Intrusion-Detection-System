"""
Dataset and DataLoader construction.

The previous SVDD loop wrapped an already-resident tensor in a
``TensorDataset`` + ``DataLoader(num_workers=0)`` and then copied the same
data host->device on every one of 25 epochs.  Two paths are provided here:

* ``make_loader``  -- standard DataLoader with pin_memory / num_workers /
  persistent_workers, correct for data that does not fit on the device.
* ``DeviceBatcher`` -- moves the whole array to the device once and slices it
  with a permutation index.  No collate, no host->device copy per batch.
  Typically 2-4x faster on this shape.  Used automatically when the array fits
  inside a memory budget.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from seeding import make_generator, worker_init_fn


class TabularDataset(Dataset):
    """float32 features, int64 labels. No per-item Python conversion."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        if y is None:
            self.y = None
        else:
            self.y = torch.from_numpy(np.ascontiguousarray(y.astype(np.int64, copy=False)))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.y is None:
            return (self.X[i],)
        return self.X[i], self.y[i]


def make_loader(
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    cfg,
    device: str = "cpu",
) -> DataLoader:
    pin = bool(cfg.pin_memory and device.startswith("cuda"))
    workers = int(cfg.num_workers) if device.startswith("cuda") else 0
    return DataLoader(
        TabularDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=bool(cfg.persistent_workers and workers > 0),
        prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=worker_init_fn if workers > 0 else None,
        generator=make_generator(cfg.seed) if shuffle else None,
    )


class DeviceBatcher:
    """
    Iterable of (x, y) batches sliced from device-resident tensors.

    Deterministic: the permutation is drawn from a seeded generator that is
    advanced once per epoch, so run N is byte-reproducible.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        batch_size: int,
        shuffle: bool,
        device: str,
        seed: int = 42,
    ):
        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
        self.y = (
            None
            if y is None
            else torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long, device=device)
        )
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = device
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)

    def __len__(self) -> int:
        n = self.X.shape[0]
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        n = self.X.shape[0]
        if self.shuffle:
            idx = torch.randperm(n, generator=self._gen).to(self.device)
        else:
            idx = torch.arange(n, device=self.device)
        for i in range(0, n, self.batch_size):
            sel = idx[i: i + self.batch_size]
            if self.y is None:
                yield (self.X.index_select(0, sel),)
            else:
                yield self.X.index_select(0, sel), self.y.index_select(0, sel)


def _fits_on_device(X: np.ndarray, device: str, budget_frac: float = 0.35) -> bool:
    if not device.startswith("cuda"):
        return True                       # CPU: the array is already in RAM
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return False
    return X.nbytes < free * budget_frac


def build_loaders(
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    cfg,
    device: str,
):
    """Pick the fast path when the data fits, otherwise a real DataLoader."""
    if _fits_on_device(X, device):
        return DeviceBatcher(X, y, batch_size, shuffle, device, seed=cfg.seed)
    return make_loader(X, y, batch_size, shuffle, cfg, device)
