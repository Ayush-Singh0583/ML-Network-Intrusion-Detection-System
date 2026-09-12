"""
Imbalance-aware losses.

CIC-IDS2017 after de-duplication is roughly 80% BENIGN with rare classes in the
tens of samples.  Plain cross-entropy on that distribution converges to the
benign predictor, which is exactly the failure mode ``learnings.md`` documents.

Two options, both selectable from config:

* ``weighted_ce``  -- class-weighted cross-entropy.  Weights use a *tempered*
  inverse frequency, w_c = (N / (K * n_c)) ** power with power <= 1.  Full
  inverse frequency (power=1) gives an 11-sample class a weight of ~2e4, which
  makes the model carve out a private decision region per sample.  power=0.5
  is the usual compromise.

* ``focal``        -- focal loss (Lin et al. 2017) with alpha = those same
  tempered weights.  Down-weights easy, already-correct samples, which on this
  dataset means the enormous benign mass stops dominating the gradient after
  the first epoch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(
    y: np.ndarray,
    n_classes: int,
    power: float = 0.5,
    normalise: bool = True,
) -> torch.Tensor:
    """Tempered inverse-frequency weights, index-aligned to encoder classes."""
    counts = np.bincount(y[y >= 0], minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n = counts.sum()
    w = (n / (n_classes * counts)) ** power
    if normalise:
        w = w * (n_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Multiclass focal loss on raw logits.

    loss = -alpha_c * (1 - p_c)^gamma * log p_c

    Computed from log-probabilities rather than probabilities so it is safe
    under autocast/float16.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()                       # stability under AMP
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()

        focal = (1.0 - p_t).clamp_min(1e-8) ** self.gamma
        loss = -focal * logp_t

        if self.label_smoothing > 0:
            smooth = -logp.mean(dim=1)
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * focal * smooth

        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, target)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg, y_train: np.ndarray, n_classes: int) -> nn.Module:
    kind = getattr(cfg, "loss", "focal").lower()

    if kind == "none":
        return nn.CrossEntropyLoss()

    w = class_weights(y_train, n_classes, power=cfg.class_weight_power)

    if kind == "weighted_ce":
        return nn.CrossEntropyLoss(weight=w)
    if kind == "focal":
        return FocalLoss(alpha=w, gamma=cfg.focal_gamma)

    raise ValueError(f"unknown loss {kind!r}; expected focal | weighted_ce | none")
