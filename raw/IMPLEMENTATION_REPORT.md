# ML NIDS Refactor Report

## 1. Executive Summary

This report documents the architectural, security, and algorithmic refactoring of the **Machine Learning Network Intrusion Detection System (ML-NIDS)** based on the CICIDS2017 benchmark dataset.

The refactor directly addresses critical failure points identified during the security engineering audit:
1. **Shortcut Learning Elimination:** Removed `"Destination Port"` from the feature space, eliminating synthetic port-to-attack memorization and forcing tree and boosting models to learn behavioral network flow dynamics.
2. **Cost-Sensitive Class Balancing:** Integrated `compute_sample_weight("balanced")` into XGBoost and dynamic class sizing into LightGBM to improve detection rates on rare minority intrusions (e.g., SQL Injection, Infiltration, Heartbleed).
3. **Open-Set Unknown Attack Detection:** Implemented maximum posterior probability thresholding (`unknown_threshold=0.55`) in `evaluate_model` to flag novel/zero-day attack vectors as `"Unknown_Attack"`.
4. **Comprehensive Macro & Security Evaluation Metrics:** Expanded model evaluation to compute unweighted Macro Precision, Macro Recall, Macro F1, and Weighted F1, while saving full classification reports and high-resolution confusion matrix heatmaps.
5. **Live Feature Scale Synchronization:** Converted all live packet timing and inter-arrival statistics from seconds ($s$) to microseconds ($\mu s$) to ensure exact statistical compatibility with `CICFlowMeter` training features.
6. **API Consistency & Bug Fixes:** Standardized function signatures across all training, standalone evaluation, and API service files.

---

## 2. Files Modified

| File | Functions Changed | Reason |
| :--- | :--- | :--- |
| `src/preprocessing.py` | `remove_identifier_columns` | Added `"Destination Port"` to dropped identifiers to prevent shortcut learning. |
| `src/training.py` | `train_xgboost`, `train_lightgbm` | Added balanced sample weights for XGBoost and dynamic `num_class` for LightGBM. |
| `src/evaluation.py` | `evaluate_model`, `feature_importance` | Added Macro metrics, Unknown Attack detection via confidence thresholding, robust label decoding, and flexible argument defaults. |
| `src/main.py` | Top-level execution pipeline | Updated calls to `evaluate_model`, `feature_importance`, and `save_model` to supply all required arguments. |
| `src/main_rf.py` | Top-level execution pipeline | Updated `evaluate_model` to pass `encoder` and `model_name`. |
| `src/main_xgb.py` | Top-level execution pipeline | Updated `evaluate_model` to pass `encoder`, `unknown_threshold=0.55`, and `model_name`. |
| `src/main_lgbm.py` | Top-level execution pipeline | Fixed `evaluate_model` and `feature_importance` argument signatures. |
| `src/compare_models.py` | Top-level model benchmark loop | Added Macro Precision, Macro Recall, and Macro F1 to comparison table export. |
| `src/evalAlone.py` | `evaluate_alone` | Supplied `encoder` to `evaluate_model` call. |
| `backend/live/statistics.py` | `calculate_iat` | Multiplied inter-arrival timestamp deltas by $10^6$ to output microseconds. |
| `backend/live/flow_manager.py` | `process_packet` | Converted active and idle interval gaps to microseconds before appending to flow buffers. |
| `backend/live/extractor.py` | `extract_features` | Assigned `Flow Duration` in microseconds ($10^6\times\text{seconds}$). |
| `backend/services/model_service.py` | `predict_dataframe` | Added `"Source Port"` and `"Destination Port"` to dropped identifier columns during CSV prediction. |

---

## 3. Detailed Changes

### 3.1 `src/preprocessing.py`
* **Before:** `identifier_columns = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]`
* **After:** `identifier_columns = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp"]`
* **Why it was changed:** In CICIDS2017, static target ports (Port 80 for DoS/Web attacks, Port 21 for FTP-Patator, Port 22 for SSH-Patator) allowed tree-based models to memorize port-to-attack correlations instead of learning actual packet sequence distributions.
* **Impact on IDS:** Forces classifiers to learn invariant statistical flow characteristics (packet sizes, flow IATs, TCP flag distributions, active/idle periods). Prevents catastrophic failure when legitimate traffic uses port 80/443 or attack traffic shifts to non-standard ports.

### 3.2 `src/training.py`
* **Before:** `train_xgboost` called `model.fit(X_train, y_train)` with no sample weighting. `train_lightgbm` hardcoded `num_class=15`.
* **After:** `train_xgboost` calculates `sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)` and passes it to `model.fit()`. `train_lightgbm` sets `num_class = len(np.unique(y_train))`.
* **Why it was changed:** Standard Cross-Entropy/Logloss optimization on heavily imbalanced datasets (>80% Benign) causes XGBoost to optimize almost exclusively for Benign traffic, producing high False Negative rates on low-frequency attacks.
* **Impact on IDS:** Penalizes misclassifications on minority attack classes proportionally to their scarcity, improving recall on high-severity attacks.

### 3.3 `src/evaluation.py`
* **Before:** `evaluate_model` only printed Accuracy and weighted F1, did not support unknown attack thresholding, and threw `TypeError` when called with fewer arguments.
* **After:**
  - Added `unknown_threshold=None` parameter.
  - Computes and displays Accuracy, Macro Precision, Macro Recall, Macro F1, and Weighted F1.
  - Automatically handles string labels (Cross-Day) and integer labels (Random Split).
  - Saves `results/classification_report.csv` and `results/confusion_matrix.png`.
* **Why it was changed:** Weighted metrics mask minority class failures. Closed-set classifiers need an out-of-distribution rejection mechanism when tested against novel day/traffic distributions.
* **Impact on IDS:** Provides security teams with uninflated detection metrics and flags uncertain predictions for SOC analyst triage.

### 3.4 `backend/live/statistics.py`, `extractor.py`, & `flow_manager.py`
* **Before:** Timestamps were subtracted using raw `time.time()` seconds (e.g. `0.005` s).
* **After:** Inter-arrival times, flow duration, active times, and idle times are converted to microseconds (e.g. `5000.0` $\mu s$) via `* 1_000_000.0`.
* **Why it was changed:** The model's `StandardScaler` was trained on `CICFlowMeter` outputs in microseconds. Feeding values $1,000,000\times$ smaller produced massive negative z-score distortion during live inference.
* **Impact on IDS:** Live packet captures now produce feature distributions consistent with the model's training artifacts, enabling accurate real-time inference.

---

## 4. Metrics Added

| Metric | Formula / Definition | IDS Engineering Significance |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall classification correctness. Kept for baseline comparisons, though misleading on imbalanced data. |
| **Macro Precision** | $\frac{1}{K}\sum_{i=1}^K \text{Precision}_i$ | Unweighted average precision across all $K$ classes. Penalizes false alarms equally across all attack types. |
| **Macro Recall** | $\frac{1}{K}\sum_{i=1}^K \text{Recall}_i$ | Unweighted average detection rate across all $K$ classes. Directly measures whether minority intrusions are being detected. |
| **Macro F1** | $\frac{1}{K}\sum_{i=1}^K \text{F1}_i$ | Harmonic mean of Macro Precision and Macro Recall. Primary benchmark metric for multi-class NIDS research. |
| **Weighted F1** | $\sum_{i=1}^K \left(\frac{N_i}{N}\right) \text{F1}_i$ | Support-weighted F1 score. Reflects overall traffic volume performance. |
| **Unknown Detection Rate** | $\frac{N_{\text{Unknown}}}{N_{\text{Total}}} \times 100\%$ | Proportion of flows flagged as out-of-distribution / novel attacks below the confidence threshold. |

---

## 5. Unknown Attack Detection

### Mechanics
1. During evaluation or live inference, `predict_proba(X_test)` produces a probability vector $\mathbf{p} = [p_1, p_2, \dots, p_K]$ across all known training classes.
2. The model's prediction confidence $C$ is computed as the maximum posterior probability:
   $$C = \max_{k \in \{1, \dots, K\}} p_k$$
3. If $C < \theta$ (where default $\theta = 0.55$), the classifier assigns:
   $$\hat{y} = \text{"Unknown\_Attack"}$$
   Otherwise, it assigns the standard argmax class label:
   $$\hat{y} = \text{encoder.inverse\_transform}(\operatorname{argmax} \mathbf{p})$$

### Security Rationale (Open-Set Recognition)
In standard closed-world machine learning, classifiers are forced to assign an instance to one of their predefined training classes, even when facing completely novel zero-day attacks (such as Friday's DDoS or Botnet attacks when trained only on Monday–Thursday).

By introducing an uncertainty threshold:
* Zero-day attacks that produce low, dispersed probabilities across known classes are routed to `Unknown_Attack` instead of being falsely classified as `BENIGN` (evasion) or misidentified as an unrelated attack.
* SOC analysts receive an immediate indicator of potential novel attack vectors or zero-day threats requiring manual payload inspection.

---

## 6. Live Feature Compatibility

The following table summarizes the timing features that were synchronized from **seconds $\to$ microseconds** ($1\text{ s} = 10^6\ \mu\text{s}$):

| Feature Name | Source | Conversion Applied |
| :--- | :--- | :--- |
| `Flow Duration` | `extractor.py` | `duration * 1_000_000.0` |
| `Flow IAT Mean` | `statistics.py` | `calculate_iat()` computes $\Delta t \times 10^6$ |
| `Flow IAT Std` | `statistics.py` | `calculate_iat()` computes $\Delta t \times 10^6$ |
| `Flow IAT Max` | `statistics.py` | `calculate_iat()` computes $\Delta t \times 10^6$ |
| `Flow IAT Min` | `statistics.py` | `calculate_iat()` computes $\Delta t \times 10^6$ |
| `Fwd IAT Total/Mean/Std/Max/Min` | `statistics.py` | Forward timestamps converted via `calculate_iat()` |
| `Bwd IAT Total/Mean/Std/Max/Min` | `statistics.py` | Backward timestamps converted via `calculate_iat()` |
| `Active Mean/Std/Max/Min` | `flow_manager.py` | `gap * 1_000_000.0` appended to `active_times` |
| `Idle Mean/Std/Max/Min` | `flow_manager.py` | `gap * 1_000_000.0` appended to `idle_times` |

---

## 7. Breaking Changes

| Scope | Function / API | Change Description | Migration Action |
| :--- | :--- | :--- | :--- |
| Feature Dimension | `remove_identifier_columns` | Feature count reduced by 1 (Destination Port dropped). | Re-train models and regenerate `feature_names.pkl` and `scaler.pkl`. |
| Evaluation Signature | `evaluate_model` | Now accepts `(model, X_test, y_test, encoder=None, unknown_threshold=None, model_name="model")`. | Callers can optionally supply `unknown_threshold` (e.g. `0.55`). Defaults provided for all optional parameters. |
| Evaluation Signature | `feature_importance` | Default parameter `model_name="model"` added. | Callers passing 2 arguments (`model`, `feature_names`) or 3 arguments (`model`, `feature_names`, `model_name`) are both fully supported. |
