---
title: "Model Evaluation Metrics in Intrusion Detection"
category: "ML"
sources:
  - "raw/IMPLEMENTATION_REPORT.md"
tags: ["metrics", "evaluation", "macro-f1", "precision", "recall", "confusion-matrix"]
updated: 2026-08-26
---

# Model Evaluation Metrics in Intrusion Detection

## 📌 Executive Summary
In network intrusion detection, reporting overall **Accuracy** or **Weighted F1** is fundamentally misleading due to extreme class imbalance (>80% Benign traffic). Robust security evaluation requires unweighted **Macro Precision**, **Macro Recall**, **Macro F1**, per-class detection matrices, and out-of-distribution unknown attack detection rates.

---

## 🧠 Why Standard Accuracy & Weighted F1 Fail

Consider a typical CICIDS2017 test slice with 100,000 packets:
* `BENIGN`: 99,000 samples (99%)
* `Heartbleed`: 10 samples (0.01%)
* `SQL Injection`: 20 samples (0.02%)

A naive or collapsed model that outputs `BENIGN` for 100% of inputs achieves:
$$\text{Accuracy} = \frac{99000}{100000} = 99.0\%$$
$$\text{Weighted F1} = 0.99 \times 0.995 + 0.0001 \times 0 + 0.0002 \times 0 \approx 0.985$$
Despite near-perfect scores, the IDS failed completely at security enforcement: **Recall on all cyber attacks was 0.0%**.

---

## 📐 Security-Aligned Metrics (`src/evaluation.py`)

The evaluation pipeline computes unweighted macro averages across all $K$ classes:

| Metric | Mathematical Formula | Security Engineering Significance |
| :--- | :--- | :--- |
| **Macro Precision** | $\frac{1}{K}\sum_{i=1}^K \text{Precision}_i$ | Penalizes false alarms equally across all attack classes regardless of volume. |
| **Macro Recall** | $\frac{1}{K}\sum_{i=1}^K \text{Recall}_i$ | Directly measures whether rare high-severity intrusions are being detected. |
| **Macro F1** | $\frac{2 \times \text{Macro Prec} \times \text{Macro Rec}}{\text{Macro Prec} + \text{Macro Rec}}$ | Primary objective benchmark metric for multi-class NIDS research. |
| **Unknown Detection Rate** | $\frac{N_{\text{Unknown}}}{N_{\text{Total}}} \times 100\%$ | Proportion of test traffic routed to `Unknown_Attack` via [[Open-Set-Recognition]]. |

---

## 📊 Evaluation Artifacts Exported

When running `evaluate_model(model, X_test, y_test, encoder, unknown_threshold=0.55)`:
1. **Console Output**: Macro Precision, Macro Recall, Macro F1, Weighted F1, and Unknown Attack Count.
2. **Classification Report**: Exported to `results/classification_report.csv` containing support, precision, recall, and f1-score per individual attack class.
3. **Confusion Matrix Heatmap**: Exported to `results/confusion_matrix.png` using Seaborn showing cross-class misclassification patterns.

---

## 🔗 Related Topics (Wikilinks)
- [[Dataset-CICIDS2017]] — Imbalance properties of the benchmark dataset.
- [[Machine-Learning-Models]] — Optimizing models for balanced macro performance.
- [[Open-Set-Recognition]] — Evaluating open-world unknown class detection.
- [[Index]] — Master Knowledge Graph Index.
