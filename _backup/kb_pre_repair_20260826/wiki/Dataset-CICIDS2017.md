---
title: "Dataset CICIDS2017 Benchmark"
category: "Data"
sources:
  - "raw/IMPLEMENTATION_REPORT.md"
  - "raw/PROJECT_DOCUMENTATION.md"
tags: ["cicids2017", "dataset", "features", "attacks", "benchmark"]
updated: 2026-08-26
---

# Dataset CICIDS2017 Benchmark

## 📌 Executive Summary
The **CICIDS2017** dataset is the primary training and evaluation benchmark for this project. Developed by the Canadian Institute for Cybersecurity, it contains realistic benign background traffic alongside common cyber-attacks collected over five days (Monday through Friday). Features were originally computed using `CICFlowMeter` in bidirectional 78+ dimensional statistical summaries.

---

## 🧠 Traffic Composition & Attack Taxonomy

The dataset captures diverse multi-class intrusion profiles across weekday captures:

| Capture Day | Traffic Profiles & Attacks Included | Class Balance |
| :--- | :--- | :--- |
| **Monday** | Pure Benign Normal Activity | 100% Benign |
| **Tuesday** | FTP-Patator (Brute Force), SSH-Patator | Benign + Brute Force |
| **Wednesday** | DoS (GoldenEye, Hulk, SlowHTTPTest, Slowloris), Heartbleed | Benign + DoS Attacks |
| **Thursday** | Web Attacks (Brute Force, XSS, Sql Injection), Infiltration | Benign + Web/Minority Intrusions |
| **Friday** | Botnet, PortScan, DDoS | Benign + Volumetric Attacks |

---

## 📐 Splitting Strategies: Random Split vs. Cross-Day Split

The codebase supports two distinct experimental evaluation paradigms in `src/preprocessing.py`:

### 1. Standard Stratified Random Split (`split_dataset`)
- Shuffles Monday–Friday data uniformly into 80% train and 20% test splits.
- Preserves relative class proportions across all known attack types.
- **Limitation**: Can artificially inflate performance if temporal leakage occurs across identical IP session streams.

### 2. Cross-Day Split (`split_dataset_by_day`)
- Trains exclusively on **Monday–Thursday** captures and tests against unseen **Friday** traffic (or vice versa).
- Mimics real-world deployment where tomorrow's attack vectors (e.g. novel Botnets or DDoS floods) have never been seen by the classifier.
- Interacts directly with [[Open-Set-Recognition]] to test zero-day rejection.

---

## ⚠️ Security Vulnerabilities & Dataset Biases
1. **Port-to-Attack Correlation**: CICIDS2017 attack scripts targeted static predefined ports (e.g. DoS on 80, Patator on 21/22). Failing to drop target ports results in [[Shortcut-Learning-Port-Bias]].
2. **Extreme Class Imbalance**: Benign traffic constitutes >80% of samples, while severe threats like `Heartbleed` (~11 samples) and `SQL Injection` (~21 samples) are heavily under-represented, necessitating balanced loss functions in [[Machine-Learning-Models]].
3. **Temporal Unit Conventions**: All duration and inter-arrival features are reported in **microseconds ($\mu s$)**, which must be strictly adhered to during live capture ([[Feature-Engineering-Timing]]).

---

## 🔗 Related Topics (Wikilinks)
- [[Shortcut-Learning-Port-Bias]] — The necessity of removing `Destination Port`.
- [[Feature-Engineering-Timing]] — Feature definitions and microsecond scaling.
- [[Open-Set-Recognition]] — Handling novel Friday attack classes during cross-day evaluation.
- [[Model-Evaluation-Metrics]] — Evaluating on imbalanced attack distributions.
- [[Index]] — Master Knowledge Graph Index.
