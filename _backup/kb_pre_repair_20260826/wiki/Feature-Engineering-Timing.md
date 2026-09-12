---
title: "Feature Engineering and Timing Scale Synchronization"
category: "ML"
sources:
  - "raw/IMPLEMENTATION_REPORT.md"
  - "raw/PROJECT_DOCUMENTATION.md"
tags: ["features", "statistics", "timing", "scaling", "microseconds"]
updated: 2026-08-26
---

# Feature Engineering & Timing Scale Synchronization

## 📌 Executive Summary
The ML-NIDS extracts a 78-dimensional statistical feature representation from bidirectional network sessions. A critical engineering requirement discovered during security audits is that all timing and duration statistics must be computed in **microseconds ($\mu s$)** ($1\text{ s} = 10^6\ \mu\text{s}$) to match the `StandardScaler` trained on `CICFlowMeter` datasets.

---

## 🧠 The Timing Scale Divergence Bug

When offline training on [[Dataset-CICIDS2017]], features like `Flow Duration`, `Flow IAT Mean`, and `Idle Max` are measured in microseconds ($\mu s$). 

If live packet sniffers (Scapy) compute durations in seconds using raw `time.time()`:
$$\text{Live Duration} = 0.05\text{ s} \quad \text{vs.} \quad \text{Training Scaler Mean} \approx 50,000.0\ \mu\text{s}$$
Applying `scaler.transform()` yields:
$$z = \frac{0.05 - 50000}{\sigma} \approx -100.0$$
This massive negative z-score distortion corrupts distance metrics and tree split thresholds, causing live predictions to fail catastrophically.

---

## 📐 Synchronized Microsecond Features

All live extraction helpers in `backend/live/statistics.py`, `extractor.py`, and `flow_manager.py` apply explicit $10^6$ multiplier scaling:

```python
# 1. Flow Duration in extractor.py
flow_duration = (flow.end_time - flow.start_time) * 1_000_000.0

# 2. Inter-Arrival Times in statistics.py
def calculate_iat(timestamps):
    if len(timestamps) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = np.diff(timestamps) * 1_000_000.0  # Convert seconds -> microseconds
    return float(np.sum(diffs)), float(np.mean(diffs)), float(np.std(diffs)), float(np.max(diffs)), float(np.min(diffs))

# 3. Active & Idle Time Tracking in flow_manager.py
if gap >= IDLE_THRESHOLD:
    flow.idle_times.append(gap * 1_000_000.0)
else:
    flow.active_times.append(gap * 1_000_000.0)
```

### Complete Summary of Synchronized Features

| Feature Name | Extractor Source | Conversion Applied |
| :--- | :--- | :--- |
| `Flow Duration` | `backend/live/extractor.py` | `duration * 1_000_000.0` |
| `Flow IAT Mean / Std / Max / Min` | `backend/live/statistics.py` | `calculate_iat()` with $\Delta t \times 10^6$ |
| `Fwd IAT Total / Mean / Std / Max / Min` | `backend/live/statistics.py` | Forward direction $\Delta t \times 10^6$ |
| `Bwd IAT Total / Mean / Std / Max / Min` | `backend/live/statistics.py` | Backward direction $\Delta t \times 10^6$ |
| `Active Mean / Std / Max / Min` | `backend/live/flow_manager.py` | `gap * 1_000_000.0` |
| `Idle Mean / Std / Max / Min` | `backend/live/flow_manager.py` | `gap * 1_000_000.0` |

---

## 🔬 Non-Temporal Feature Categories
Beyond timing, the feature vector extracts:
1. **Packet Length Statistics**: `Total Length of Fwd/Bwd Packets`, `Fwd/Bwd Packet Length Max/Min/Mean/Std`, `Average Packet Size`.
2. **Flow Rates**: `Flow Packets/s`, `Flow Bytes/s`, `Down/Up Ratio`.
3. **TCP Flag Telemetry**: `FIN`, `SYN`, `RST`, `PSH`, `ACK`, `URG`, `CWE`, `ECE` flag counts and header lengths.
4. **Subflow Quantities**: `Subflow Fwd/Bwd Packets`, `Subflow Fwd/Bwd Bytes`.

---

## 🔗 Related Topics (Wikilinks)
- [[Dataset-CICIDS2017]] — Origin of the 78-feature specification.
- [[Live-Packet-Flow-Pipeline]] — Packet aggregation and buffer management in Scapy.
- [[Shortcut-Learning-Port-Bias]] — Exclusion of port identifiers from the feature vector.
- [[Machine-Learning-Models]] — Consuming scaled feature arrays.
- [[Index]] — Master Knowledge Graph Index.
