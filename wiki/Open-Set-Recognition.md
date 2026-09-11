---
title: "Open-Set Recognition and Unknown Attack Gating"
category: "Security"
sources:
  - "raw/IMPLEMENTATION_REPORT.md"
code_refs:
  - "src/openset.py"
  - "src/run.py"
tags: ["open-set", "zero-day", "unknown-attack", "confidence-thresholding", "security"]
status: superseded
superseded_by: "wiki/Rejection-Scoring.md"
updated: 2026-08-26
---

# Open-Set Recognition & Unknown Attack Gating

> **⚠️ CORRECTION (2026-08-26) — the method on this page does not work on this dataset.**
>
> This page presents **maximum posterior confidence thresholding at θ = 0.55** as the
> open-set mechanism. Measurement has refuted it. On a 12-class ablation at CIC-IDS2017
> imbalance with near-OOD test classes, max-softmax scored **AUROC 0.112** and energy
> **0.120** — i.e. *inverted*, worse than random — while Mahalanobis distance in the
> penultimate embedding scored **0.914** on the identical model. A real cross-day run
> reported AUROC 0.295 and 3.3% unknown recall using this method.
>
> Two causes, both documented in [[Rejection-Scoring]]: a softmax is normalised over
> known classes and cannot express "none of the above"; and Friday's DDoS is *near*-OOD,
> sitting inside the Wednesday DoS-Hulk cloud, so it receives a **larger** max logit than
> genuine BENIGN traffic.
>
> Also note: this project simultaneously carried θ = 0.55 in its schema and 0.90 in
> `main_xgb.py`. That was possible only because neither number was ever fitted to
> held-out data — there was no validation set. See [[Training-Protocol]].
>
> **Current method:** `mahalanobis_embed` scoring with τ calibrated on the validation day
> at a stated FPR. See [[Rejection-Scoring]].
>
> The original text is retained below: the closed-world framing and the SOC rationale are
> still correct, and the refuted mechanism is worth knowing as a refuted mechanism.


## 📌 Executive Summary
Standard supervised classifiers operate under a **closed-world assumption**: they are forced to assign every test instance to one of their predefined training categories. In real-world network security, zero-day attacks and out-of-distribution traffic regularly appear. **Open-set recognition** introduces posterior probability confidence thresholding to classify uncertain, novel inputs as `"Unknown_Attack"`.

---

## 🧠 Mechanics of Confidence Thresholding

Let a trained model output posterior probability vector $\mathbf{p} = [p_1, p_2, \dots, p_K]$ for an input flow $x$ across $K$ known classes.

1. Compute maximum posterior confidence:
   $$C(x) = \max_{k \in \{1, \dots, K\}} p_k$$
2. Apply decision boundary with rejection threshold $\theta$ (default $\theta = 0.55$):
   $$\hat{y}(x) = \begin{cases} \text{encoder.inverse\_transform}(\operatorname{argmax} \mathbf{p}) & \text{if } C(x) \ge \theta \\ \text{"Unknown\_Attack"} & \text{if } C(x) < \theta \end{cases}$$

```python
# Implementation in src/evaluation.py
probabilities = model.predict_proba(X_test)
confidence = probabilities.max(axis=1)
y_pred = probabilities.argmax(axis=1)
y_pred_labels = np.array(encoder.inverse_transform(y_pred)).astype(str)

if unknown_threshold is not None:
    y_pred_labels[confidence < unknown_threshold] = "Unknown_Attack"
```

---

## 🛡️ Security Rationale & Cross-Day Validation

In cross-day evaluation (e.g. training on Monday–Thursday and testing on Friday):
* Friday contains novel attack classes (`Botnet`, `DDoS`, `PortScan`) not present in Monday–Thursday training.
* Without thresholding, the model either falsely classifies them as `BENIGN` (catastrophic evasion) or randomly forces them into unrelated attack buckets (false alarms).
* With $\theta = 0.55$, novel traffic produces diffuse posterior probabilities across classes, triggering the `Unknown_Attack` flag.
* **SOC Analyst Action**: Flags novel threats for payload inspection, PCAP capture, and signature creation without blind evasion.

---

## 🔗 Related Topics (Wikilinks)
- [[Model-Evaluation-Metrics]] — Measuring Unknown Detection Rate.
- [[Dataset-CICIDS2017]] — Cross-day attack distribution differences.
- [[Machine-Learning-Models]] — Probability outputs from RF and XGBoost.
- [[System-Architecture]] — Routing Unknown_Attack alerts to the React SOC dashboard.
- [[Index]] — Master Knowledge Graph Index.
