"""
Training engine.

Replaces the old loop, which had:
  * no validation pass
  * no early stopping (``EPOCHS = 25`` fixed)
  * no LR schedule (flat Adam 1e-3)
  * no gradient clipping
  * no mixed precision
  * a single unconditional ``torch.save`` after the last epoch
  * no best-model restoration
  * ``print(f"Epoch [{epoch+1:02d}/25] Loss : {epoch_loss:.10f}")`` -- the
    SUM over batches, while the computed mean was discarded and the epoch
    total was hard-coded in the format string

Everything here is deterministic given ``cfg.seed``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# =====================================================================
# SCHEDULER
# =====================================================================


def decay_param_groups(model: nn.Module, weight_decay: float):
    """
    AdamW parameter groups that exclude 1-D parameters from weight decay.

    ``AdamW(model.parameters(), weight_decay=0.01)`` applies decay to EVERY
    parameter, including BatchNorm/LayerNorm gains and every bias.  Decaying a
    normalisation gain toward zero shrinks that layer's output range, and
    decaying a classifier bias toward zero on an 80%-majority dataset removes
    the very term that lets the head express a non-uniform class prior.  Neither
    is what "weight decay" is meant to regularise.  Every reference
    implementation of AdamW for classification splits these out; this one did
    not.

    Measured effect on the ablation: small (macro-F1 0.870 -> 0.870, near-OOD
    AUROC 0.116 -> 0.126).  It is a correctness fix, not the main cause -- but
    the cost of getting it right is one function.
    """
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_factor: float = 1e-2,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(0, min(int(warmup_epochs), max(total_epochs - 1, 0)))

    def fn(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        denom = max(total_epochs - warmup_epochs, 1)
        prog = (epoch - warmup_epochs) / denom
        prog = min(max(prog, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr_factor + (1.0 - min_lr_factor) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# =====================================================================
# EARLY STOPPING
# =====================================================================


class EarlyStopping:
    def __init__(self, patience: int = 12, mode: str = "max", min_delta: float = 1e-5):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: float = -math.inf if mode == "max" else math.inf
        self.best_epoch: int = -1
        self.bad: int = 0

    def is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> Tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        if self.is_better(value):
            self.best = value
            self.best_epoch = epoch
            self.bad = 0
            return True, False
        self.bad += 1
        return False, self.bad >= self.patience


# =====================================================================
# HISTORY
# =====================================================================


@dataclass
class History:
    rows: List[Dict[str, float]]

    def append(self, **kw) -> None:
        self.rows.append(dict(kw))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.rows, indent=2))

    def best(self, key: str, mode: str = "max") -> Optional[Dict[str, float]]:
        if not self.rows:
            return None
        f = max if mode == "max" else min
        return f(self.rows, key=lambda r: r.get(key, -math.inf if mode == "max" else math.inf))


# =====================================================================
# HELPERS
# =====================================================================


def _amp_enabled(cfg, device: str) -> bool:
    return bool(getattr(cfg, "amp", True)) and device.startswith("cuda")


def _autocast(device: str, enabled: bool):
    if enabled:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:                                   # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader,
    device: str,
    amp: bool = False,
) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(x)
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, 0), dtype=np.float32)


@torch.no_grad()
def embed(model: nn.Module, loader, device: str) -> np.ndarray:
    """Penultimate features, for the Mahalanobis rejector."""
    model.eval()
    fn = getattr(model, "features", model)
    out: List[np.ndarray] = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        out.append(fn(x).float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, 0), dtype=np.float32)


# =====================================================================
# CLASSIFIER TRAINING
# =====================================================================


def train_classifier(
    model: nn.Module,
    train_loader,
    val_loader,
    y_val: np.ndarray,
    criterion: nn.Module,
    cfg,
    device: str,
    ckpt_path: Path,
    log: Callable[[str], None] = print,
    class_names: Optional[List[str]] = None,
) -> History:
    model.to(device)
    amp = _amp_enabled(cfg, device)
    scaler = _make_scaler(amp)

    # Classes actually present in validation. Used for the macro-F1 that
    # selects the checkpoint -- see the note at the metric computation below.
    VAL_LABELS = np.unique(y_val)

    optimizer = torch.optim.AdamW(
        decay_param_groups(model, cfg.weight_decay), lr=cfg.lr
    )
    scheduler = cosine_with_warmup(
        optimizer, cfg.warmup_epochs, cfg.epochs, cfg.min_lr_factor
    )
    stopper = EarlyStopping(cfg.patience, cfg.monitor_mode, cfg.min_delta)
    history = History(rows=[])

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"\nValidation covers {len(VAL_LABELS)} of the model's "
        f"{getattr(model, 'n_classes', '?')} classes; macro-F1 is averaged over "
        f"those {len(VAL_LABELS)}.")
    log(
        f"\n{'epoch':>6} {'train_loss':>11} {'val_loss':>10} "
        f"{'val_macroF1':>12} {'val_bAcc':>9} {'nCls':>5} {'lr':>9} {'sec':>6}"
    )
    log("-" * 78)

    for epoch in range(cfg.epochs):
        t0 = time.time()

        # ---------------- train ----------------
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                logits = model(x)
                loss = criterion(logits, y)

            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            bs = x.size(0)
            total += loss.item() * bs           # weight by batch size, not batch count
            seen += bs

        train_loss = total / max(seen, 1)

        # ---------------- validate ----------------
        model.eval()
        vtotal, vseen = 0.0, 0
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with _autocast(device, amp):
                    logits = model(x)
                    vloss = criterion(logits, y)
                vtotal += vloss.item() * x.size(0)
                vseen += x.size(0)
                preds.append(logits.float().argmax(1).cpu().numpy())

        val_loss = vtotal / max(vseen, 1)
        y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)

        # BUG FIX (checkpoint selection).  This previously read
        #     f1_score(y_val, y_pred, average="macro", zero_division=0)
        # with no labels=, so sklearn averaged over unique_labels(y_val, y_pred)
        # -- the UNION.  The validation day (Thursday) carries only a subset of
        # the known classes, so every class the model predicted but which does
        # not occur in validation entered the average as a 0.0 term.  The metric
        # therefore fell as the model became BOLDER, and early stopping selected
        # whichever checkpoint predicted the FEWEST distinct classes.  Verified
        # to invert the ranking of two checkpoints:
        #     cautious (45% attack recall, 1 spurious class): diluted 0.559 / true 0.671
        #     bolder   (80% attack recall, 7 spurious):       diluted 0.378 / true 0.907
        # Averaging over the classes present in y_val is the correct macro-F1
        # for model selection.
        macro_f1 = float(
            f1_score(y_val, y_pred, average="macro", labels=VAL_LABELS, zero_division=0)
        )
        macro_f1_union = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        bacc = float(balanced_accuracy_score(y_val, y_pred))
        n_pred = int(len(np.unique(y_pred)))

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history.append(
            epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
            val_macro_f1=macro_f1, val_macro_f1_union=macro_f1_union,
            val_balanced_acc=bacc, val_classes_predicted=n_pred,
            lr=lr_now, seconds=dt,
        )

        # collapse alarm: a model emitting one or two classes on a multi-class
        # validation set is degenerate, whatever its accuracy says.
        if n_pred <= 2 and len(VAL_LABELS) > 2:
            log(f"       [warn] epoch {epoch + 1}: model predicts only {n_pred} "
                f"distinct class(es) on a {len(VAL_LABELS)}-class validation set")

        monitor_value = {
            "macro_f1": macro_f1,
            "balanced_accuracy": bacc,
            "loss": val_loss,
        }[cfg.monitor]

        improved, stop = stopper.step(monitor_value, epoch)
        mark = " *" if improved else ""
        log(
            f"{epoch + 1:>6} {train_loss:>11.6f} {val_loss:>10.6f} "
            f"{macro_f1:>12.6f} {bacc:>9.6f} {n_pred:>5} {lr_now:>9.2e} {dt:>6.1f}{mark}"
        )

        if improved:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "monitor": cfg.monitor,
                    "value": monitor_value,
                },
                ckpt_path,
            )
        if stop:
            log(f"\nEarly stop at epoch {epoch + 1} "
                f"(no improvement for {cfg.patience} epochs)")
            break

    # ---------------- restore best ----------------
    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        log(
            f"Restored best checkpoint: epoch {blob['epoch'] + 1}, "
            f"{blob['monitor']} = {blob['value']:.6f}"
        )
    else:
        log("[warn] no checkpoint written; model is the final-epoch state")

    # ---------------- per-class validation breakdown ----------------
    # Without this, "macro-F1 = 0.234" is a single number that cannot
    # distinguish "uniformly mediocre" from "perfect on the majority, zero on
    # everything else". The second is the failure mode that matters here.
    preds = []
    with torch.no_grad():
        model.eval()
        for x, _y in val_loader:
            x = x.to(device, non_blocking=True)
            preds.append(model(x).float().argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)

    rec = recall_score(y_val, y_pred, average=None, labels=VAL_LABELS, zero_division=0)
    prec = precision_score(y_val, y_pred, average=None, labels=VAL_LABELS, zero_division=0)
    f1s = f1_score(y_val, y_pred, average=None, labels=VAL_LABELS, zero_division=0)
    sup = np.array([(y_val == c).sum() for c in VAL_LABELS])
    names = class_names if class_names is not None else [str(c) for c in VAL_LABELS]

    log(f"\n{'validation class':<26s} {'support':>9s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
    log("-" * 60)
    for i, c in enumerate(VAL_LABELS):
        nm = names[int(c)] if class_names is not None else str(c)
        log(f"{nm:<26s} {sup[i]:>9,} {prec[i]:>7.3f} {rec[i]:>7.3f} {f1s[i]:>7.3f}")
    dead = [names[int(c)] if class_names is not None else str(c)
            for i, c in enumerate(VAL_LABELS) if rec[i] == 0.0]
    if dead:
        log(f"\n[warn] {len(dead)} validation class(es) with ZERO recall: {dead}")

    return history


# =====================================================================
# ONE-CLASS TRAINING (autoencoder / Deep SVDD)
# =====================================================================


def train_autoencoder(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg,
    device: str,
    ckpt_path: Path,
    log: Callable[[str], None] = print,
) -> History:
    """Reconstruction training on BENIGN traffic only."""
    model.to(device)
    amp = _amp_enabled(cfg, device)
    scaler = _make_scaler(amp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine_with_warmup(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.min_lr_factor)
    stopper = EarlyStopping(cfg.patience, "min", cfg.min_delta)
    history = History(rows=[])
    mse = nn.MSELoss()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"\n{'epoch':>6} {'train_mse':>12} {'val_mse':>12} {'lr':>9} {'sec':>6}")
    log("-" * 52)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss = mse(model(x), x)
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            total += loss.item() * x.size(0)
            seen += x.size(0)
        train_mse = total / max(seen, 1)

        model.eval()
        vt, vs = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device, non_blocking=True)
                with _autocast(device, amp):
                    vloss = mse(model(x), x)
                vt += vloss.item() * x.size(0)
                vs += x.size(0)
        val_mse = vt / max(vs, 1)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        history.append(epoch=epoch + 1, train_mse=train_mse, val_mse=val_mse, lr=lr_now, seconds=dt)

        improved, stop = stopper.step(val_mse, epoch)
        log(f"{epoch+1:>6} {train_mse:>12.6f} {val_mse:>12.6f} {lr_now:>9.2e} {dt:>6.1f}"
            f"{' *' if improved else ''}")

        if improved:
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "monitor": "val_mse", "value": val_mse}, ckpt_path)
        if stop:
            log(f"\nEarly stop at epoch {epoch + 1}")
            break

    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        log(f"Restored best checkpoint: epoch {blob['epoch'] + 1}, val_mse = {blob['value']:.6f}")
    return history


def train_deep_svdd(
    model: nn.Module,
    center: torch.Tensor,
    train_loader,
    val_loader,
    cfg,
    device: str,
    ckpt_path: Path,
    log: Callable[[str], None] = print,
    collapse_tol: float = 1e-4,
) -> Tuple[History, torch.Tensor]:
    """
    One-class Deep SVDD (nu -> 0):  L = mean_i ||f(x_i) - c||^2

    Early stopping is deliberately NOT used here, and that is not an omission.
    The only unsupervised signal available is the validation distance, and a
    smaller validation distance is exactly what collapse produces -- selecting
    the checkpoint that minimises it selects *toward* the degenerate solution.
    (Observed: min-val-d2 selection picks epoch 1 every time.)  Deep SVDD is
    therefore trained on a fixed budget with a cosine schedule, the final state
    is kept, and correctness is enforced by the collapse assertion below rather
    than by a selection metric.
    """
    model.to(device)
    center = center.to(device)
    amp = _amp_enabled(cfg, device)
    scaler = _make_scaler(amp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine_with_warmup(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.min_lr_factor)
    history = History(rows=[])

    log("\n[note] early stopping disabled for Deep SVDD: lower validation "
        "distance\n       is the signature of hypersphere collapse, not of a "
        "better model.\n       Fixed epoch budget + collapse assertion instead.")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"\n{'epoch':>6} {'train_d2':>13} {'val_d2':>13} {'val_d2_std':>13} {'lr':>9} {'sec':>6}")
    log("-" * 70)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                z = model(x)
                loss = torch.sum((z - center) ** 2, dim=1).mean()
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            total += loss.item() * x.size(0)
            seen += x.size(0)
        train_d2 = total / max(seen, 1)

        model.eval()
        dists: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device, non_blocking=True)
                with _autocast(device, amp):
                    z = model(x)
                dists.append(torch.sum((z.float() - center) ** 2, dim=1).cpu())
        d = torch.cat(dists) if dists else torch.zeros(1)
        val_d2 = float(d.mean())
        val_std = float(d.std())

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        history.append(epoch=epoch + 1, train_d2=train_d2, val_d2=val_d2,
                       val_d2_std=val_std, lr=lr_now, seconds=dt)

        log(f"{epoch+1:>6} {train_d2:>13.6e} {val_d2:>13.6e} {val_std:>13.6e} "
            f"{lr_now:>9.2e} {dt:>6.1f}")

        if val_std < collapse_tol:
            log(f"\n[FATAL] hypersphere collapse detected at epoch {epoch + 1}: "
                f"distance std = {val_std:.3e} < {collapse_tol:.1e}")
            raise RuntimeError(
                "Deep SVDD collapsed to a constant function. Every Linear must "
                "have bias=False and every BatchNorm affine=False; check also "
                "that weight_decay is not driving the weights to zero."
            )

        # checkpoint every epoch so a crash does not lose the run
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                    "monitor": "val_d2(not used for selection)", "value": val_d2,
                    "val_d2_std": val_std, "center": center.cpu()}, ckpt_path)

    log(f"\nKept final epoch (fixed budget of {cfg.epochs}); "
        f"final validation distance std = {val_std:.6e}")
    return history, center


@torch.no_grad()
def svdd_distances(model: nn.Module, center: torch.Tensor, loader, device: str) -> np.ndarray:
    """Batched squared distances. The old evaluate_svdd.py allocated the whole
    test set on the device in one tensor and OOMed on small GPUs."""
    model.eval()
    center = center.to(device)
    out: List[np.ndarray] = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        z = model(x)
        out.append(torch.sum((z - center) ** 2, dim=1).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


@torch.no_grad()
def reconstruction_errors(model: nn.Module, loader, device: str) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        err = ((model(x) - x) ** 2).mean(dim=1)
        out.append(err.float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)
