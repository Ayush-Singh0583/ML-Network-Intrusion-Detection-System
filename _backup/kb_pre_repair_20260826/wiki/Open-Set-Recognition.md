---
title: "Open-Set Recognition and Unknown Attack Gating"
category: "Security"
sources:
  - "raw/IMPLEMENTATION_REPORT.md"
tags: ["open-set", "zero-day", "unknown-attack", "confidence-thresholding", "security"]
updated: 2026-08-26
---

# Open-Set Recognition & Unknown Attack Gating

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
