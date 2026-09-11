# Project Learnings & Retrospectives

This log tracks architectural decisions, optimization experiments, what works, what fails, and recurring security gotchas across development and research sessions.

---

## [2026-08-26] - NIDS Security Refactor & Microsecond Feature Synchronization

### ✅ What Worked
- **Removing `Destination Port`**: Successfully broke the shortcut learning artifact where tree models memorized port-to-attack mappings (e.g. Port 80 = DoS, Port 21 = FTP-Patator). Forces classifiers to learn invariant traffic geometry (packet lengths, inter-arrival time variance, TCP flag sequences).
- **Balanced Sample Weighting for XGBoost & LightGBM**: Using `compute_sample_weight("balanced", y_train)` directly eliminated zero-recall failure modes on rare minority attacks (Infiltration, Heartbleed, SQL Injection).
- **Posterior Probability Thresholding for Open-Set Zero-Day Detection**: Establishing a confidence cutoff ($\theta = 0.55$) routes novel Friday attack distributions and out-of-distribution traffic to `Unknown_Attack` instead of falsely misclassifying them as `BENIGN`.
- **Microsecond Timing Conversion in Live Sniffer**: Multiplying `Flow Duration`, `Flow IAT`, `Active`, and `Idle` deltas by $10^6$ resolved the $1,000,000\times$ feature scale divergence between Scapy live captures and `CICFlowMeter` training features.
- **Two-Stage Mutex in Flow Manager / Garbage Collector**: Acquiring `flows_lock` only during the key-eviction phase and running model inference outside the lock prevents Scapy's packet capture thread from dropping incoming packets.

### ❌ What Failed / Gotchas
- **Using raw `time.time()` differences**: Passing seconds ($s$) to `StandardScaler` trained on microseconds ($\mu s$) generated massive negative z-score distortion, driving prediction confidence to near zero or defaulting to random classes.
- **Weighted F1 as a Primary Success Metric**: In CICIDS2017 (>80% Benign traffic), a trivial classifier predicting `BENIGN` for all packets yields >99% Accuracy and ~0.99 Weighted F1 while having 0.00 Recall on minority cyber attacks.
- **Closed-World Evaluation on Cross-Day Splits**: Standard `LabelEncoder.inverse_transform()` threw index errors when encountering Friday attacks unseen during Monday-Thursday training. Ground-truth mapping must remap unseen labels to `Unknown_Attack` prior to metric calculation.
- **Sorting by non-existent `F1 Score` column in `compare_models.py`**: Model comparison script crashed when sorting by `"F1 Score"` after renaming output metrics to `"Macro F1"` and `"Weighted F1"`.

### 💡 Actionable Rules for Future Sessions
1. **Always Synchronize Feature Dimensions & Scales**: Any timing feature added to live capture must strictly output microseconds ($\mu s$).
2. **Never Re-introduce Port Identifiers into the Feature Set**: Keep `remove_identifier_columns()` clean of `Destination Port`, `Source Port`, `Source IP`, `Destination IP`, and `Flow ID`.
3. **Always Save the 4-File Artifact Bundle**: Model updates must serialize `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and `feature_names.pkl` into `saved_models/<model_name>/` in the same transaction.
4. **Mandatory Unweighted Macro Reporting**: Every evaluation must print and export Macro Precision, Macro Recall, Macro F1, and confusion matrix heatmaps.
