# 🛡️ ML Network Intrusion Detection System

An end-to-end Machine Learning based Network Intrusion Detection System capable of detecting multiple cyber attacks from network traffic using the **CIC-IDS2017** dataset. The project includes a **FastAPI backend**, **React dashboard**, **Random Forest classifier**, CSV upload & prediction, interactive visualization, and downloadable prediction reports.

---

## 📌 Project Overview

This project analyzes network traffic and classifies each network flow into one of multiple attack categories or as normal (BENIGN).

The application consists of:

- **Machine Learning Pipeline**
- **FastAPI REST Backend**
- **React Frontend Dashboard**
- **CSV Upload & Prediction**
- **Interactive Charts**
- **Prediction Report Download**

---

# 🚀 Features

### Machine Learning

- Multi-class Intrusion Detection
- Random Forest Classifier
- Logistic Regression (Implemented)
- Decision Tree (Implemented)
- Data Cleaning & Preprocessing
- Feature Scaling
- Label Encoding
- Feature Importance Analysis
- Model Evaluation

---

### Backend (FastAPI)

- Predict single network flow
- Upload CSV for bulk prediction
- Download prediction report
- Interactive Swagger Documentation
- REST API

---

### Frontend (React)

- Modern Dashboard
- Upload CSV Files
- Prediction Summary
- Attack Distribution Pie Chart
- Attack Count Bar Chart
- Model Information
- Download Prediction Report
- Responsive Dark UI

---

# 🧠 Attack Classes

The model predicts:

- BENIGN
- Bot
- DDoS
- DoS Hulk
- DoS GoldenEye
- DoS Slowloris
- DoS Slowhttptest
- FTP-Patator
- SSH-Patator
- PortScan
- Infiltration
- WebAttack - Brute Force
- WebAttack - SQL Injection
- WebAttack - XSS
- Heartbleed

---

# 📊 Dataset

Dataset Used:

**CIC IDS 2017**

The dataset is **not included** in this repository because of its large size.

Download:

https://www.unb.ca/cic/datasets/ids-2017.html

Place all CSV files inside:

```text
data/
```

---

# 📁 Project Structure

```text
ML-Network-Intrusion-Detection-System
│
├── backend/
│   ├── routers/
│   ├── services/
│   ├── app.py
│   ├── config.py
│   └── schemas.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── data/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── generate_sample.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Tech Stack

### Programming Language

- Python
- JavaScript

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Joblib

### Backend

- FastAPI
- Uvicorn

### Frontend

- React
- Axios
- Chart.js

### Visualization

- Matplotlib
- Chart.js

---

# 📈 Model Performance

| Model | Accuracy |
|---------|----------|
| Logistic Regression | ~98% |
| Decision Tree | Implemented |
| Random Forest | **99.82%** |

Current Production Model:

✅ Random Forest

---

# 🛠 Installation

## Clone Repository

```bash
git clone https://github.com/YourUsername/ML-Network-Intrusion-Detection-System.git

cd ML-Network-Intrusion-Detection-System
```

---

## Create Virtual Environment

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Python Requirements

```bash
pip install -r requirements.txt
```

---

## Install Frontend

```bash
cd frontend

npm install
```

---

# 🧠 Train the Model

Run

```bash
python src/main.py
```

This generates

```text
models/

random_forest_model.pkl

scaler.pkl

label_encoder.pkl

feature_names.pkl
```

---

# 🚀 Run Backend

```bash
uvicorn backend.app:app --reload
```

Backend

```
http://127.0.0.1:8000
```

Swagger

```
http://127.0.0.1:8000/docs
```

---

# 🌐 Run Frontend

```bash
cd frontend

npm run dev
```

Frontend

```
http://localhost:5173
```

---

# 📡 API Endpoints

## Health

```
GET /

GET /health
```

---

## Predict Single Flow

```
POST /predict
```

---

## Predict CSV

```
POST /predict_csv
```

Upload a CIC IDS CSV file and receive prediction summary.

---

## Download Prediction

```
GET /download
```

Downloads the predicted CSV report.

---

# 📷 Screenshots

## Dashboard

> *(Add Screenshot Here)*

---

## Prediction Summary

> *(Add Screenshot Here)*

---

## Swagger Documentation

> *(Add Screenshot Here)*

---

# 📊 Feature Importance

The project also generates:

- Feature Importance CSV
- Feature Importance Graph

Saved inside

```text
models/
```

---

# 🔮 Future Enhancements

- Live Packet Capture using Scapy
- Real-Time Intrusion Detection
- Network Interface Monitoring
- Threat Severity Dashboard
- User Authentication
- Docker Deployment
- Cloud Deployment
- SIEM Integration
- Model Comparison Dashboard

---

# 👨‍💻 Author

**Ayush Singh**

Information Science & Engineering

RV Institute of Technology and Management

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.