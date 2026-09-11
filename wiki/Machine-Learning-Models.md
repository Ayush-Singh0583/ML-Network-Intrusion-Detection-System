---
title: "Machine Learning Models and Serialization"
category: "ML"
sources:
  - "raw/IMPLEMENTATION_REPORT.md"
  - "raw/PROJECT_DOCUMENTATION.md"
code_refs:
  - "src/training.py"
  - "src/models.py"
  - "src/config.py"
tags: ["models", "random-forest", "xgboost", "lightgbm", "decision-tree", "logistic-regression", "training"]
status: stale
updated: 2026-08-26
---

# Machine Learning Models & Serialization

> **⚠️ CORRECTION (2026-08-26) — hyperparameters and the artifact format have changed.**
>
> Every estimator setting quoted below was replaced during the 2026-08-26 refactor:
>
> | Claim on this page | Now |
> | :--- | :--- |
> | RF `n_estimators=100`, unbounded depth | `n_estimators=300`, `max_depth=24`, `min_samples_leaf=5`, `class_weight="balanced_subsample"` |
> | XGB `eval_metric="logloss"` | `"mlogloss"`; `objective` set explicitly; `tree_method="hist"` |
> | LightGBM `subsample=0.8` | `subsample_freq=1` added — **`subsample` was a silent no-op** without it |
> | 4-file bundle, `feature_names.pkl` with 78 columns | Single verified bundle incl. `cleaner.pkl` + `manifest.json`; **62 features** after dropping 7 collinear twins |
>
> The unbounded RF also produced a 150 MB pickle whose `predict_proba` was mostly exactly
> 1.0, which is why confidence thresholding rejected nothing — see [[Rejection-Scoring]].
>
> Deep SVDD is mentioned below as "one-class unsupervised anomaly detection". The
> implementation that existed at the time had **collapsed to a constant function**; see
> [[Hypersphere-Collapse]].
>
> Four deep classifiers (MLP, 1D CNN, LSTM, Autoencoder) now exist and are not described
> below. See [[Training-Protocol]].


## 📌 Executive Summary
The ML-NIDS leverages a portfolio of supervised machine learning algorithms trained on [[Dataset-CICIDS2017]] to classify network flows into benign or multi-class cyber attacks. Key models include **Random Forest**, **XGBoost** (with cost-sensitive sample weighting), **LightGBM**, **Decision Trees**, and **Logistic Regression**, with one-class **Deep SVDD** used for unsupervised anomaly detection.

---

## 🤖 Supported Model Implementations (`src/training.py`)

### 1. Random Forest (`train_random_forest`)
- **Default Production Classifier**: Robust against noisy tabular data and outliers.
- **Parameters**: `n_estimators=100`, `random_state=42`, `n_jobs=-1`.

### 2. XGBoost (`train_xgboost`)
- **Cost-Sensitive Class Balancing**: In heavily imbalanced network traffic (>80% Benign), standard cross-entropy minimizes loss by ignoring rare minority attacks. XGBoost uses balanced sample weights:
  ```python
  sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
  model = XGBClassifier(
      n_estimators=200,
      max_depth=8,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      eval_metric="logloss"
  )
  model.fit(X_train, y_train, sample_weight=sample_weights)
  ```

### 3. LightGBM (`train_lightgbm`)
- **Fast Multiclass GBDT**: Configured with dynamic `num_class=len(np.unique(y_train))` and `class_weight="balanced"` to prevent hardcoded class mismatches across binary and multi-class dataset splits.

### 4. Decision Trees & Logistic Regression (`train_decision_tree`, `train_logistic`)
- Baseline models used for feature interpretability, latency benchmarking, and lightweight edge device profiling.

---

## 📦 4-File Serialization Bundle

When persisting or serving any trained model in production (`saved_models/<model_name>/`), **four separate artifacts must be saved and loaded synchronously**:

```
saved_models/<model_name>/
├── <model_name>_model.pkl    # Serialized scikit-learn / XGBoost model
├── scaler.pkl                 # StandardScaler fitted on training features
├── label_encoder.pkl          # LabelEncoder mapping integers <-> string classes
└── feature_names.pkl          # List of 78 feature column names in exact training order
```

Inside `backend/services/model_service.py`, incoming feature dictionaries are mapped to DataFrame columns matching `feature_names.pkl`, transformed via `scaler.pkl`, and predicted using `model.pkl`.

---

## 🔗 Related Topics (Wikilinks)
- [[Feature-Engineering-Timing]] — Input feature vector scaling.
- [[Shortcut-Learning-Port-Bias]] — Feature pruning to prevent port memorization.
- [[Open-Set-Recognition]] — Post-processing model posterior probabilities.
- [[Model-Evaluation-Metrics]] — Benchmark scores and comparative metrics.
- [[Index]] — Master Knowledge Graph Index.
