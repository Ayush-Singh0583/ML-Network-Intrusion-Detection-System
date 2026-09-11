"""
Deep models for tabular flow features.

Design notes
------------
* All classifiers return **raw logits**.  Softmax lives in the loss and in the
  scoring functions, never in the model.  This keeps ``CrossEntropyLoss``
  numerically stable under mixed precision and makes the energy score
  (``-T*logsumexp(logits/T)``) available for free.

* CNN and LSTM operate on a *tabular* vector, so the sequence axis has to be
  manufactured.  Two honest choices are implemented rather than pretending a
  flow record is a time series:
    - ``CNN1D``  treats the 62 features as a 1-D signal of length F with one
      channel and slides kernels over the feature axis.  Feature ORDER is
      arbitrary, so a wide first kernel plus BatchNorm is used rather than a
      deep stack that would bake in adjacency assumptions.
    - ``LSTMNet`` projects the feature vector into ``n_steps`` learned tokens
      and runs a bidirectional LSTM over those.  This gives the recurrence
      something meaningful to consume instead of feeding 62 scalars in a row.

* ``DeepSVDD`` has **no bias terms anywhere** and uses ``BatchNorm1d(affine=
  False)``.  Ruff et al. (ICML 2018, Prop. 2) prove that a bias term lets the
  network realise a constant function f(x)=c, driving the objective to zero
  while learning nothing.  The previous implementation used ``nn.Linear``
  defaults (bias=True) and did collapse: in the saved checkpoint the final
  layer's bias had converged onto the center (max |diff| 4.85e-05, cosine
  0.99999988) and the 99th-percentile benign radius was 5.58e-09.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# INITIALISATION
# =====================================================================


def init_weights(module: nn.Module) -> None:
    """Kaiming for ReLU-family linears/convs, zeros for biases, 1/0 for norms."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # forget-gate bias = 1 (Jozefowicz et al.)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)


# =====================================================================
# MLP with residual blocks
# =====================================================================


class ResidualBlock(nn.Module):
    """Pre-activation residual block: x + W2(drop(act(BN(W1(act(BN(x)))))))."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.gelu(self.bn1(x)))
        h = self.drop(h)
        h = self.fc2(F.gelu(self.bn2(h)))
        return x + h


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate representation, used by the Mahalanobis rejector."""
        return self.blocks(self.stem(x))


# =====================================================================
# 1-D CNN
# =====================================================================


class CNN1D(nn.Module):
    """
    (B, F) -> (B, 1, F) -> conv stack -> global avg+max pool -> logits.

    Feature order in CIC-IDS2017 is arbitrary, so the first kernel is wide
    (k=7) to mix distant features, and pooling is global rather than
    positional.  A deep narrow stack would encode adjacency that does not exist.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        channels: Tuple[int, ...] = (64, 128, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            k = 7 if i == 0 else 3
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(in_ch * 2, in_ch),
            nn.BatchNorm1d(in_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, n_classes),
        )
        self.apply(init_weights)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))                    # (B, C, F)
        avg = h.mean(dim=2)
        mx = h.amax(dim=2)
        return torch.cat([avg, mx], dim=1)               # (B, 2C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._embed(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._embed(x)


# =====================================================================
# LSTM
# =====================================================================


class LSTMNet(nn.Module):
    """
    (B, F) -> Linear -> (B, T, D) -> BiLSTM -> attention pool -> logits.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_steps: int = 8,
        d_model: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.d_model = d_model

        self.project = nn.Sequential(
            nn.Linear(input_dim, n_steps * d_model),
            nn.BatchNorm1d(n_steps * d_model),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )
        self.apply(init_weights)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = self.project(x).view(b, self.n_steps, self.d_model)
        out, _ = self.lstm(h)                              # (B, T, 2H)
        w = torch.softmax(self.attn(out), dim=1)           # (B, T, 1)
        return (out * w).sum(dim=1)                        # (B, 2H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._embed(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._embed(x)


# =====================================================================
# AUTOENCODER (open-set reconstruction score)
# =====================================================================


class Autoencoder(nn.Module):
    """Symmetric MLP autoencoder trained on BENIGN traffic only."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        enc: List[nn.Module] = []
        d = input_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]
            if dropout:
                enc.append(nn.Dropout(dropout))
            d = h
        enc.append(nn.Linear(d, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: List[nn.Module] = []
        d = latent_dim
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU()]
            d = h
        dec.append(nn.Linear(d, input_dim))
        self.decoder = nn.Sequential(*dec)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE. Higher = more novel."""
        return ((self(x) - x) ** 2).mean(dim=1)


# =====================================================================
# DEEP SVDD
# =====================================================================


class DeepSVDD(nn.Module):
    """
    One-class encoder.  bias=False everywhere and BatchNorm1d(affine=False)
    are *requirements*, not style: a learnable additive term of any kind lets
    the network output a constant and collapse the hypersphere.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden: Tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h, bias=False),
                nn.BatchNorm1d(h, affine=False),
                nn.LeakyReLU(0.1),          # unbounded activation
            ]
            d = h
        layers.append(nn.Linear(d, latent_dim, bias=False))
        self.encoder = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                assert m.bias is None, "DeepSVDD layers must not have bias terms"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def load_pretrained_encoder(self, ae: "Autoencoder") -> None:
        """
        Warm-start from an autoencoder of matching shape.  Only weights are
        copied; the AE's bias terms are deliberately discarded.
        """
        src = [m for m in ae.encoder if isinstance(m, nn.Linear)]
        dst = [m for m in self.encoder if isinstance(m, nn.Linear)]
        if len(src) != len(dst):
            raise ValueError(f"AE has {len(src)} linear layers, SVDD has {len(dst)}")
        with torch.no_grad():
            for s, d in zip(src, dst):
                if s.weight.shape != d.weight.shape:
                    raise ValueError(f"shape mismatch {s.weight.shape} vs {d.weight.shape}")
                d.weight.copy_(s.weight)


@torch.no_grad()
def init_center(model: nn.Module, loader, device: str, eps: float = 0.1) -> torch.Tensor:
    """
    Mean embedding of the training data, with the epsilon guard from the
    reference implementation: any component too close to zero is pushed to
    +/-eps, because a near-zero coordinate is a free collapse direction.
    """
    model.eval()
    total: Optional[torch.Tensor] = None
    n = 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        z = model(x)
        total = z.sum(dim=0) if total is None else total + z.sum(dim=0)
        n += z.size(0)
    if total is None or n == 0:
        raise ValueError("init_center received an empty loader")
    c = total / n
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps
    return c.detach()


# =====================================================================
# FACTORY
# =====================================================================


def build_model(name: str, input_dim: int, n_classes: int, cfg) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLP(input_dim, n_classes, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout)
    if name == "cnn":
        return CNN1D(input_dim, n_classes, dropout=cfg.dropout)
    if name == "lstm":
        return LSTMNet(input_dim, n_classes, dropout=cfg.dropout)
    if name == "autoencoder":
        return Autoencoder(input_dim, latent_dim=cfg.latent_dim)
    if name == "deep_svdd":
        return DeepSVDD(input_dim, latent_dim=cfg.latent_dim)
    raise ValueError(f"unknown model {name!r}")


CLASSIFIERS = {"mlp", "cnn", "lstm"}
ONE_CLASS = {"autoencoder", "deep_svdd"}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
