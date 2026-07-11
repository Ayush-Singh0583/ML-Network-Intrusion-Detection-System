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

# ✨ Features

## 🔍 Live Packet Capture

- Capture packets using **Scapy**
- Supports TCP, UDP and IP traffic
- Bidirectional flow creation
- Automatic flow expiration

---

## 🤖 Machine Learning Detection

- Random Forest Classifier
- Trained on CICIDS2017 Dataset
- Real-time prediction
- Confidence Score
- Benign / Attack Classification

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