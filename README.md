# 🛡️ ML Network Intrusion Detection System (NIDS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Scapy](https://img.shields.io/badge/Scapy-Packet%20Capture-red)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A Real-Time Machine Learning Powered Network Intrusion Detection System built using FastAPI, React, Scapy, and Random Forest.**

</div>

---

# 📖 Overview

This project is a **real-time Network Intrusion Detection System (NIDS)** that captures live network traffic, converts packets into bidirectional network flows, extracts features compatible with the **CICIDS2017** dataset, predicts malicious activity using a trained **Random Forest** model, stores the results in SQLite, and visualizes everything through an interactive React dashboard.

Unlike traditional offline IDS implementations, this project performs **live packet capture and real-time inference**, making it suitable as a learning platform for network security, machine learning deployment, and full-stack software engineering.

---

# 📊 Measured Results

**Protocol.** Train on Monday–Wednesday plus half of Thursday, validate on the
other half of Thursday, test on **Friday**. `DDoS`, `PortScan` and `Bot` appear
only on Friday, so they are genuinely unseen at training time. Friday is never
touched during training or threshold calibration.

**Friday test set** — 703,198 flows:

| class | count | share |
|---|---:|---:|
| BENIGN | 414,275 | 58.9% |
| PortScan | 158,930 | 22.6% |
| DDoS | 128,027 | 18.2% |
| Bot | 1,966 | 0.3% |

An all-BENIGN classifier scores **58.91%** accuracy here. That is the number any
result must beat.

## The headline metric is detection rate at a fixed false-alarm budget

Not accuracy. Accuracy on this split is ambiguous — the same model scores
**58.47%** multi-class and **69.68%** binary — and the multi-class figure is
*below* the all-BENIGN baseline, because the model labels **zero** attack flows
with their correct class (`DDoS`, `PortScan` and `Bot` are not in its label
space at all). A detector's job is to catch attacks without drowning the
analyst, so the report is those two quantities.

Thresholds are quantiles of the **validation** day's benign scores. Nothing from
the test day enters them.

**XGBoost** — 400 rounds, depth 8, balanced sample weights:

| benign FPR budget | observed FPR | DDoS | PortScan | Bot | any attack |
|---|---:|---:|---:|---:|---:|
| 0.1% | 0.14% | **61.93%** | 0.23% | 0.00% | 27.57% |
| 1.0% | 4.38% | **63.62%** | 0.62% | 0.05% | 28.53% |
| 5.0% | 9.48% | **67.00%** | 5.41% | 9.41% | 32.73% |

Reproduce with `python src/run.py train --model xgb --protocol crossday`; the
table is written to `runs/<sha>-<timestamp>/detection_by_class.csv`.

### Random Forest, and why it may be the better *deployed* model

A Random Forest (25 trees, depth 14 — deliberately small) on the identical
split:

| benign FPR budget | observed FPR | DDoS | PortScan | any attack |
|---|---:|---:|---:|---:|
| 0.1% | **0.11%** | 58.11% | 0.08% | 25.79% |
| 1.0% | **1.02%** | 62.22% | 0.11% | 27.63% |
| 5.0% | **5.55%** | 63.76% | 37.76% | 49.02% |

XGBoost ranks slightly better. **Its thresholds do not transfer.** A budget of
1% lands at 4.38% on the test day for XGBoost and 1.02% for the Random Forest.
Measured on benign rows, XGBoost's 99th percentile moves 23× between
validation and test (0.00487 → 0.11339); the Random Forest's moves 0.2%
(0.32252 → 0.32320).

Boosting optimises a loss, not calibration, and `compute_sample_weight
("balanced")` makes it worse — benign probabilities are crushed toward zero, so
the threshold quantile sits on a near-vertical stretch of the CDF where a small
day-to-day shift moves the alert *rate* enormously. A Random Forest probability
is a vote fraction averaged over trees: granular and stable. **A threshold that
misses its target by 4× is not an operating point.** Fix with isotonic
calibration fitted on the validation day, or threshold by rank rather than by
value.

## What the numbers say

**It detects volumetric HTTP floods, and essentially nothing else.** 58–64% of
`DDoS` — a class it has never seen — is caught, and those flows are classified
as `DoS Hulk`. That is not a coincidence: HULK (Wednesday, in training) and
LOIC (Friday, unseen) are both high-rate HTTP floods with near-identical flow
geometry. The model learned the geometry rather than the label, which is real
generalisation across attack families.

**Loosening the threshold buys almost nothing.** Going from 0.1% to 5% false
alarms raises DDoS detection by 5.6 points and multiplies false alerts by ~50.
The tight operating point is strictly better.

**Alert volume is the binding constraint, not detection rate.** CIC-IDS2017's
test day is ~41% attack; a real link is nearer 0.1%. Projected onto 10M
flows/day at 0.1% prevalence:

| FPR budget | false alerts/day | true alerts/day | precision |
|---|---:|---:|---:|
| 0.1% | 14,252 | 2,757 | **16.2%** |
| 1.0% | 437,532 | 2,853 | 0.7% |
| 5.0% | 946,997 | 3,273 | 0.3% |

Every rate on this dataset is measured at roughly **400× the real attack
prevalence**, which flatters precision enormously. No operating point here is
deployable as-is.

---

# ⚠️ Known Limitations

**Port scans are not detectable from a single flow, by construction.** A port
scan is the same flow repeated against different ports: identical duration,
packet counts, lengths and flags. Measured on the raw capture — 158,930
PortScan flows reduce to **1,958 unique feature vectors** once `Destination
Port` is removed, versus 90,819 with it. Scanning is a property of a *set* of
flows (one source touching many ports in a window), not of any one flow, so a
per-flow classifier cannot see it. Detection is 0.23% at the usable threshold. The fix is a windowed per-source
aggregation stage, not a better model — see Roadmap.

**Botnet C2 detection is 0.00% at every threshold tested.** Beaconing has no
per-flow signature either.

**Model capacity is not the constraint.** Three independent experiments agree:
a 25-tree Random Forest, a 400-round class-balanced XGBoost, and every
threshold between 0.1% and 5% false alarms all return near-zero on PortScan and
Bot. A 16× larger model buys 3.8 points of DDoS and nothing else. The signal is
not in the data.

**`Destination Port` is deliberately excluded from the feature set** to prevent
shortcut learning (port 21 → FTP-Patator). This is why scan flows become
indistinguishable. The port belongs in the *detection logic* as an aggregate,
never as a per-flow feature.

**Thresholds calibrated on XGBoost do not transfer across days.** A 1% benign
false-alarm budget lands at 4.38% on the test day, because the model's
probabilities are uncalibrated (see Results). Use the Random Forest bundle, or
calibrate, before trusting an operating point.

**The open-set rejector is weak.** Mahalanobis distance on scaled features
reaches AUROC 0.657 and TPR@5%FPR of 1.7% against known-vs-unknown ground
truth. Its threshold also transfers poorly across days: calibrated for 5%
rejection on Thursday, it rejects 10.1% on Friday. (The classifier's own
threshold transfers well — 0.1% → 0.11%, 1% → 1.02% — so the drift is specific
to the novelty score.)

---

# ✨ Features

## 🔍 Live Packet Capture

- Capture packets using **Scapy**
- Supports TCP, UDP and IP traffic
- Bidirectional flow creation
- Automatic flow expiration

---

## 🤖 Machine Learning Detection

- Random Forest / XGBoost / LightGBM classifiers over ~60 flow features
- Trained on CIC-IDS2017 with a **temporal** train/test split (see Measured Results)
- Real-time inference on completed flows
- Closed-set posterior **plus** an explicit novelty score, so the API can say
  "this resembles nothing I was trained on" rather than silently guessing
- Known limits are documented rather than hidden — see Known Limitations

---

## 📊 Live Dashboard

- Live Traffic Statistics
- Total Flows
- Packets Captured
- Bytes Processed
- Threat Counter
- Recent Flow History
- Protocol Statistics
- Charts & Graphs
- CSV Download

---

## 🗄 Database Logging

Every completed flow is stored inside SQLite with

- Source IP
- Destination IP
- Ports
- Protocol
- Duration
- Packets
- Bytes
- Prediction
- Confidence

---

## ⚙ Backend

Built using **FastAPI**

Features include

- REST APIs
- Background Cleanup Worker
- Thread-safe Flow Manager
- Feature Extraction
- Model Prediction Service
- SQLite Integration

---

## 🎨 Frontend

Built using

- React
- Axios
- Chart.js
- CSS

Provides

- Modern Dark UI
- Live Statistics
- Flow History
- Download Reports
- Interactive Charts

---

# 🏗 Project Architecture

```
                Internet Traffic
                       │
                       ▼
               Scapy Packet Capture
                       │
                       ▼
                Flow Manager
                       │
                       ▼
             Feature Extraction
                       │
                       ▼
          Random Forest Prediction
                       │
         ┌─────────────┴──────────────┐
         ▼                            ▼
   SQLite Database            Live Dashboard
         │                            │
         └─────────────┬──────────────┘
                       ▼
                 React Frontend
```

---

# 📂 Project Structure

```
ML-Network-Intrusion-Detection-System
│
├── backend
│   ├── live
│   │   ├── capture.py
│   │   ├── cleanup.py
│   │   ├── extractor.py
│   │   ├── flow_manager.py
│   │   ├── models.py
│   │   ├── statistics.py
│   │   └── features
│   │
│   ├── routers
│   ├── services
│   ├── trained_models
│   ├── database.py
│   └── app.py
│
├── frontend
│   ├── src
│   │   ├── components
│   │   ├── pages
│   │   └── App.jsx
│   │
│   └── package.json
│
├── data
├── models
├── README.md
└── requirements.txt
```

---

# 🧠 Machine Learning Pipeline

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Random Forest Training
    │
    ▼
Model Serialization (.pkl)
    │
    ▼
FastAPI Prediction Service
    │
    ▼
Live Detection
```

---

# 📡 Live Detection Pipeline

```
Incoming Packet
        │
        ▼
Packet Capture
        │
        ▼
Flow Aggregation
        │
        ▼
Feature Extraction
        │
        ▼
Random Forest
        │
        ▼
Prediction
        │
        ▼
SQLite Logging
        │
        ▼
Dashboard Update
```

---

# 📊 Extracted Features

The system extracts network flow features including

- Flow Duration
- Packet Counts
- Byte Counts
- Packet Length Statistics
- Flow Rate
- Header Length
- TCP Flags
- Active Time
- Idle Time
- Inter Arrival Time (IAT)
- Window Size
- Segment Size
- Bulk Statistics
- Packet Ratios

These features closely follow the CICIDS2017 feature set.

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/ML-Network-Intrusion-Detection-System.git

cd ML-Network-Intrusion-Detection-System
```

---

## Backend

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

uvicorn backend.app:app --reload
```

Backend runs on

```
http://127.0.0.1:8000
```

Swagger

```
http://127.0.0.1:8000/docs
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

Runs on

```
http://localhost:5173
```

---

# 📈 Dashboard

The dashboard provides

- Live Statistics
- Threat Counter
- Total Packets
- Total Flows
- Total Bytes
- Average Packets/sec
- Recent Flow History
- Charts
- Download CSV

---

# 📷 Screenshots

> Add screenshots here

Example

```
screenshots/

dashboard.png

capture.png

prediction.png

history.png
```

---

# 📚 Dataset

This project uses

**CICIDS2017**

for model training.

---

# 🛠 Tech Stack

### Backend

- Python
- FastAPI
- Scapy
- SQLite
- Scikit-Learn

### Frontend

- React
- Axios
- Chart.js
- CSS

### Machine Learning

- Random Forest
- Pandas
- NumPy

---

# 📌 Future Improvements

- WebSocket Support
- PostgreSQL
- SQLAlchemy ORM
- Docker Deployment
- Authentication
- User Management
- **Windowed per-source aggregation for scan detection** (the documented gap)
- Multi-Model Detection
- SIEM Integration
- Email Alerts
- Kubernetes Deployment

---

# 👨‍💻 Author

**Ayush Singh**

Information Science Engineering Student

RV Institute of Technology and Management

GitHub:
https://github.com/<YOUR_USERNAME>

LinkedIn:
(Add your LinkedIn)

Portfolio:
(Add your Portfolio)

---

# ⭐ If you like this project

Please consider giving it a **Star ⭐**

---

# 📄 License

This project is licensed under the MIT License.