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

### Random Forest, and what the comparison actually shows

The same protocol, a full Random Forest (300 trees, depth 24,
`class_weight="balanced_subsample"`):

| benign FPR budget | observed FPR | DDoS | PortScan | any attack |
|---|---:|---:|---:|---:|
| 0.1% | **0.09%** | 26.68% | 0.09% | 11.87% |
| 1.0% | **0.97%** | 63.40% | 0.10% | 28.15% |
| 5.0% | **5.36%** | 63.83% | 0.26% | 28.43% |

**Two findings, and they point in opposite directions.**

**The Random Forest is calibrated and XGBoost is not.** Every RF budget lands
where it was aimed; XGBoost's 1% budget arrives at 4.38% and its 5% at 9.48%.
Measured on benign rows, XGBoost's 99th percentile moves **23×** between
validation and test (0.00487 → 0.11339); the Random Forest's moves 0.2%
(0.32252 → 0.32320). Boosting optimises a loss, not calibration, and
`compute_sample_weight("balanced")` makes it worse — benign probabilities are
crushed toward zero, so the threshold quantile sits on a near-vertical stretch
of the CDF where a small day-to-day shift moves the alert *rate* enormously. A
Random Forest probability is a vote fraction averaged over trees: granular and
stable.

**But XGBoost dominates at the operating point you would actually use.**
Compare at matched *observed* false-alarm rate rather than at matched budget:

| model | observed FPR | DDoS | any attack | precision |
|---|---:|---:|---:|---:|
| Random Forest | 0.09% | 26.68% | 11.87% | 11.83% |
| **XGBoost** | 0.14% | **61.93%** | **27.57%** | **16.21%** |

Higher recall *and* higher precision at a comparable false-alarm rate. The
Random Forest has to loosen all the way to 0.97% FPR — eleven times the false
alarms — merely to match what XGBoost achieves at 0.14%.

**So the answer is neither "use RF" nor "use XGBoost as-is".** Switching to the
Random Forest to obtain trustworthy thresholds would cost roughly 35 points of
DDoS detection. Wrapping XGBoost in `CalibratedClassifierCV(method="isotonic",
cv="prefit")` fitted on the validation day costs nothing and buys the same
threshold reliability. **You do not pick the worse model to get a working knob;
you fix the knob.**

> Worth noting and not yet explained: a deliberately small Random Forest (25
> trees, depth 14) scored **58.11%** on DDoS at 0.11% FPR — more than twice the
> 300-tree model at the same operating point. More capacity made the tight
> threshold worse. Recorded as an observation, not a theory.

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

# 🔎 Scan Detection

The flow classifier detects **0.23%** of port scans, and no amount of model
capacity changes that — see Known Limitations. Scanning is a property of a
*set* of flows, so it is detected by counting rather than by classifying, in a
component that runs **alongside** the model rather than inside it.

`backend/live/scan_tracker.py` maintains a 60-second sliding window per source
address and detects both scan shapes:

| shape | pattern | example |
|---|---|---|
| **Vertical** | one source → many **ports** on few hosts | `nmap -p-` |
| **Horizontal** | one source → many **hosts** on few ports | subnet sweep for 445 |

Covering only the first is the common mistake; CIC-IDS2017's PortScan class is
mostly vertical, which makes it easy to forget the other exists.

**Design decisions worth knowing:**

- **Observed at flow creation, not flow expiry.** A SYN scan against closed
  ports produces flows that receive one packet and never another, so they sit
  in the flow table until `IDLE_TIMEOUT` (5s). Feeding the tracker from the
  cleanup worker would report a scan ~6s after it ended, in one lump.
- **`deque` + `Counter`, not a `set`.** A set cannot expire correctly: when an
  observation ages out you cannot tell whether its port should leave, because
  the set has forgotten how many times it was seen.
- **Bounded source table.** Without a cap this *is* a memory-exhaustion vector:
  a scanner using spoofed or decoy sources (`nmap -D`) allocates one window per
  fake address. Eviction is least-recently-observed — refusing new sources when
  full would let an attacker fill the table with decoys and then scan from an
  address the detector has stopped accepting.
- **Alert suppression per source per kind.** Without it, every packet past the
  threshold emits another alert — thousands per second during the scan you are
  trying to report once.
- **Alerts are a separate resource.** A scan alert is a statement about a
  *source over a window*: no flow_id, no duration, no packet count. It lives in
  its own `scan_alerts` table behind its own endpoints, not jammed into the
  per-flow predictions.

```
GET /live/scans          persisted alerts + summary by kind and source
GET /live/scans/active   the live window, including sources below threshold
```

Covered by 22 tests in `tests/test_scan_tracker.py`, driven entirely by
synthetic `(src_ip, dst_ip, dst_port, timestamp)` tuples — no packets, no
sockets, no sleeping. Each guard was verified by removing it and confirming the
matching test fails.

---

# ⚠️ Known Limitations

**Port scans are not detectable from a single flow, by construction.** A port
scan is the same flow repeated against different ports: identical duration,
packet counts, lengths and flags. Measured on the raw capture — 158,930
PortScan flows reduce to **1,958 unique feature vectors** once `Destination
Port` is removed, versus 90,819 with it. Scanning is a property of a *set* of
flows (one source touching many ports in a window), not of any one flow, so a
per-flow classifier cannot see it. Detection is 0.23% at the usable threshold.

**This is now handled outside the model.** A windowed per-source aggregation
stage detects both scan shapes — see Scan Detection above. The limitation on
the *classifier* stands and is not fixable by training; the *system* covers it.

**Botnet C2 detection is 0.00% at every threshold tested.** Beaconing has no
per-flow signature either.

**Model capacity is not the constraint.** Four independent experiments agree: a
25-tree Random Forest, a 300-tree Random Forest, a 400-round class-balanced
XGBoost, and every threshold between 0.1% and 5% false alarms all return
0.09–0.26% on PortScan and 0.00% on Bot. Twelve times the trees buys nothing.
The signal is not in the data.

**`Destination Port` is deliberately excluded from the feature set** to prevent
shortcut learning (port 21 → FTP-Patator). This is why scan flows become
indistinguishable. The port belongs in the *detection logic* as an aggregate,
never as a per-flow feature.

**Thresholds calibrated on XGBoost do not transfer across days.** A 1% benign
false-alarm budget lands at 4.38% on the test day, because the model's
probabilities are uncalibrated (see Results). Calibrate before trusting an
operating point — do not switch to the Random Forest, which is well calibrated
but detects 35 points less DDoS at the same false-alarm rate.

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
- **A second detection stage for what the model structurally cannot see**:
  windowed per-source aggregation catching vertical and horizontal scans

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

> **Shipped:** windowed per-source aggregation for scan detection — see Scan
> Detection. It was the documented gap in this section.


- WebSocket Support
- PostgreSQL
- SQLAlchemy ORM
- Docker Deployment
- Authentication
- User Management
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