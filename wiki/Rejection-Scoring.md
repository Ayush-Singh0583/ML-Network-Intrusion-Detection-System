---
title: "Rejection Scoring for Unknown Attacks"
category: "Security"
sources:
  - "raw/CODE_REVIEW_2026-08-26.md"
code_refs:
  - "src/openset.py"
  - "src/run.py"
  - "src/config.py"
status: current
updated: 2026-08-26
tags: ["open-set", "ood-detection", "mahalanobis", "energy-score", "auroc", "near-ood"]
---

# Rejection Scoring for Unknown Attacks

## 📌 Executive Summary
Detecting an unknown attack requires a **score** that says how far a flow is from everything the model was trained on, and a **threshold** calibrated without touching the test day. The obvious score — maximum softmax posterior — is the wrong one, and on this dataset it does not merely underperform: it **inverts**, producing AUROC below 0.5. The scorer that survives is Mahalanobis distance in the penultimate embedding.

---

## 🧠 Core Concepts & Mechanics

### Why max-softmax cannot work

The softmax is normalised **over known classes**. It answers "which of my K classes is this most like", never "is this like any of them". A point infinitely far from all training data still yields a probability vector summing to 1, and usually a confident one.

Gradient-boosted trees make this strictly worse: an out-of-hull point is decided by whichever leaves it falls into, and leaves at the edge of the training hull are the *purest*. A Random Forest with unbounded depth returns `predict_proba` values of exactly 1.0 for most inputs, so no threshold below 1.0 rejects anything.

### Why AUROC came out *below* 0.5

AUROC 0.5 is noise. A score landing at 0.295 carries real signal **with the sign reversed**, and the cause here is specific:

- **Friday's DDoS is near-OOD.** It sits inside the Wednesday DoS-Hulk cloud. A flow deep inside a known class's region gets a large max logit, so `−logsumexp` is very negative, so energy calls it *in-distribution*.
- **BENIGN is the least confident class.** Benign traffic sits near several decision boundaries and gets small margins, so energy calls genuine BENIGN *more novel* than the attack.
- **BatchNorm compounds it.** At inference BN normalises with running statistics estimated from ~80%-BENIGN batches, rescaling an OOD flow *toward* that benign reference before it reaches the head — destroying the distance-from-manifold signal the score depends on.

**Do not respond by flipping the sign.** You only know to flip it *because* you looked at the test labels; that is selection on the test set.

---

## 📐 Measured Results

Identical models and data. 12 classes at CIC-IDS2017 imbalance (majority 86.7%, rarest n=11); near-OOD drawn inside a known class's cloud to mimic DDoS-vs-DoS-Hulk.

| Scorer | BatchNorm near | BatchNorm far | LayerNorm near | LayerNorm far |
| :--- | ---: | ---: | ---: | ---: |
| `msp` | **0.112** | 0.547 | 0.578 | 0.894 |
| `energy` | **0.120** | 0.739 | 0.630 | 0.989 |
| **`mahalanobis_embed`** | **0.914** | **1.000** | **0.930** | 0.991 |
| `mahalanobis_input` | 0.630 | 1.000 | 0.630 | 1.000 |

Both logit-based scorers are inverted under BatchNorm and recover under LayerNorm. Mahalanobis on the penultimate embedding stays above 0.9 in every configuration, because it measures distance from the training manifold rather than classifier confidence — and confidence is exactly what near-OOD corrupts.

### The scorers

```python
def energy_score(logits, T=1.0):
    """-T·logsumexp(logits/T). Retains logit MAGNITUDE, which softmax normalises away."""
    return -T * logsumexp(np.asarray(logits, dtype=np.float64) / T, axis=1)

class MahalanobisScorer:
    """Distance to the nearest class-conditional Gaussian with a shared
    Ledoit-Wolf shrunk covariance. Model-agnostic — the only option for
    Random Forest, which has no logits."""
```

Plus autoencoder reconstruction error and Deep SVDD distance, both trained on BENIGN only ([[Hypersphere-Collapse]]).

### Threshold calibration

τ is always a quantile of the **validation** scores. Nothing from the test day enters it.

```python
def threshold_at_fpr(scores_val_negative, target_fpr=0.05):
    return float(np.quantile(scores_val_negative, 1.0 - target_fpr))
```

### Reporting

Unknown-rate alone is maximised by rejecting everything. The metrics that mean something:

- **AUROC** — threshold-free; says whether the score separates known from unknown *at all*. **Lead with this.**
- **AUPR** — better behaved when unknowns are a small fraction.
- **TPR @ 5% FPR** — the operating point a SOC would actually run.
- **OSCR curve** — correct-classification rate on knowns vs false-positive rate on unknowns, swept over τ.

---

## ⚠️ Gotchas & Security Pitfalls

- **Scorer ranking is not stable across runs.** Mahalanobis won decisively in two experiments and was the inverted one in a third. The pipeline prints a comparison table and **refuses to auto-select**, because picking the winner from a table computed on the test day is test-set selection.
- **To choose a scorer honestly**, hold one known attack class out of training and use its validation rows as a pseudo-unknown. Not yet implemented — the right next addition if the open-set result becomes the headline.
- **`"Unknown_Attack"` must not be written into a fixed-width NumPy array.** `np.array(...).astype(str)` sizes to the longest existing class name; in the binary case (`<U6`) the label silently truncates to `"Unknow"`. Use `dtype=object`. See [[Metric-Dilution-Traps]].
- **A documented threshold that the code does not use is worse than none.** This project carried `θ = 0.55` in its schema and `0.90` in `main_xgb.py` simultaneously — possible only because neither was ever fitted to anything.

---

## 🔗 Related Topics (Wikilinks)
- [[Open-Set-Recognition]] — The original confidence-thresholding page this supersedes in method.
- [[Hypersphere-Collapse]] — The one-class detector that supplies the SVDD distance score.
- [[Metric-Dilution-Traps]] — How the reported AUROC and macro-F1 were misread.
- [[Training-Protocol]] — Where the validation day that calibrates τ comes from.
- [[Dataset-CICIDS2017]] — Why Friday's DDoS is near-OOD rather than far-OOD.
- [[Index]] — Master Knowledge Graph Index.
