---
title: "Hypersphere Collapse in Deep SVDD"
category: "ML"
sources:
  - "raw/CODE_REVIEW_2026-08-26.md"
code_refs:
  - "src/models.py"
  - "src/engine.py"
  - "tests/test_pipeline.py"
status: current
updated: 2026-08-26
tags: ["deep-svdd", "one-class", "anomaly-detection", "collapse", "pytorch", "open-set"]
---

# Hypersphere Collapse in Deep SVDD

## 📌 Executive Summary
Deep SVDD minimises the squared distance from every embedded sample to a fixed centre **c**. If the encoder can represent a *constant function*, it will: mapping every input to **c** drives the objective to exactly zero while learning nothing. This project's original one-class detector had collapsed onto that solution and was a constant map — its AUROC was 0.5 by construction. The cause was a single default argument.

---

## 🧠 Core Concepts & Mechanics

The one-class objective (ν → 0) is:

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\lVert f_\theta(x_i) - \mathbf{c} \rVert^2$$

Ruff et al. (ICML 2018, Prop. 2) prove that this objective admits a trivial minimiser whenever the network contains **any learnable additive term**. A bias vector in the final layer can simply *become* **c**, after which the weights are free to shrink to zero and the loss is exactly 0 for all inputs.

The published requirements are therefore:

1. **No bias term in any layer.**
2. **Unbounded activations only** (a bounded activation lets the network saturate to a constant).
3. **c ≠ 0**, with components guarded away from zero.

### The failure, read out of the saved checkpoint

The original `deep_svdd.py` used `nn.Linear(...)` with its default `bias=True` on all three layers. The evidence was recoverable directly from the trained artifact without re-running anything:

```
enc0.weight (64,69)  norm=1.062629
enc2.weight (32,64)  norm=0.993793
enc4.weight (16,32)  norm=0.529684   <- shrinking toward 0
enc4.bias   (16,)    norm=0.384880
center      (16,)    norm=0.384870   <- identical

max |enc4.bias − center| = 4.85e-05
cosine(enc4.bias, center) = 0.99999988
radius.pt (99th pct of benign d²) = 5.58e-09
```

The final bias **had become the centre**. Every input — benign flow, DDoS flood, port scan — landed on the same point.

### Why the training loop did not notice

A falling loss is also what collapse looks like. The original loop printed `epoch_loss` (the **sum** over batches) while computing and discarding the mean, so a per-batch loss of 2e-9 across 400 batches displayed as ~1e-6 and read as convergence. See [[Training-Protocol]].

Worse, **early stopping on validation distance selects *toward* collapse** — a smaller distance is the degenerate solution. Selecting the minimum picked epoch 1 on every run. Deep SVDD is therefore trained on a fixed epoch budget here, with correctness enforced by assertion rather than by a selection metric.

---

## 📐 Architecture, Equations & Code Patterns

```python
class DeepSVDD(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden=(128, 64)):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h, bias=False),      # no additive term
                       nn.BatchNorm1d(h, affine=False),  # no learnable shift
                       nn.LeakyReLU(0.1)]                # unbounded
            d = h
        layers.append(nn.Linear(d, latent_dim, bias=False))
        self.encoder = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                assert m.bias is None, "DeepSVDD layers must not have bias terms"
```

The ε-guard on the centre (any component with `|c_i| < 0.1` pushed to `±0.1`) removes the near-zero coordinates that are free collapse directions:

```python
c[(c.abs() < eps) & (c <  0)] = -eps
c[(c.abs() < eps) & (c >= 0)] =  eps
```

And the training loop refuses to save a degenerate model:

```python
if val_std < collapse_tol:            # 1e-4
    raise RuntimeError("Deep SVDD collapsed to a constant function.")
```

---

## ⚠️ Gotchas & Security Pitfalls

- **`affine=False` on BatchNorm is not optional.** An affine norm reintroduces a learnable shift, which is a bias by another name.
- **Weight decay accelerates collapse** when biases are free: it shrinks the weights toward zero, which is precisely the degenerate direction.
- **Autoencoder pretraining matters.** Without it the centre is computed from a randomly-initialised network and is an arbitrary point unrelated to the data manifold. `--pretrain-ae` warm-starts the encoder from a trained AE, copying weights only and discarding the AE's biases.
- **Guard the guard.** `tests/test_pipeline.py` trains the *old* biased architecture until it degenerates and asserts the collapse detector raises. A safety check that has never been observed to fire is not known to work.
- After the fix, validation distance std reached `3.40e+00` where the old checkpoint's 99th-percentile radius was `5.58e-09`.

---

## 🔗 Related Topics (Wikilinks)
- [[Rejection-Scoring]] — Where the SVDD distance sits among the novelty scores.
- [[Open-Set-Recognition]] — The problem this detector exists to solve.
- [[Training-Protocol]] — Seeding, checkpoint selection, and why early stopping is disabled here.
- [[Machine-Learning-Models]] — The wider model portfolio.
- [[Index]] — Master Knowledge Graph Index.
