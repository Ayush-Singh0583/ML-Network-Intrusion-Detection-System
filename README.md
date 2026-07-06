# 🛡️ ML Network Intrusion Detection System (NIDS)

A Machine Learning based **Network Intrusion Detection System (NIDS)** capable of detecting and classifying multiple types of cyber attacks using the **CIC-IDS2017** dataset.

The project implements a complete Machine Learning pipeline including data preprocessing, feature engineering, model training, evaluation, and model persistence. Future versions will include a FastAPI backend, React dashboard, and real-time packet monitoring.

---

# 📌 Features

- Multi-class Intrusion Detection
- Automatic loading of multiple datasets
- Data cleaning and preprocessing pipeline
- Feature scaling and label encoding
- Multiple Machine Learning models
- Model evaluation and comparison
- Feature importance analysis
- Model serialization for deployment

---

# 🚀 Current Capabilities

✔ Automatic loading of all CIC-IDS2017 CSV files

✔ Automatic dataset merging

✔ Data cleaning

- Remove invalid Flow Duration
- Handle missing values
- Replace Infinity values
- Remove duplicate records

✔ Multi-class classification

✔ Logistic Regression

✔ Decision Tree

✔ Random Forest

✔ Model comparison

✔ Save trained model

---

# 📂 Dataset

Dataset Used:

**CIC-IDS2017 (Canadian Institute for Cybersecurity Intrusion Detection Dataset)**

The project automatically loads every CSV present inside the `data/` folder.

Supported attack classes include:

- BENIGN
- Bot
- DDoS
- PortScan
- DoS Hulk
- DoS GoldenEye
- DoS Slowloris
- DoS SlowHTTPTest
- FTP-Patator
- SSH-Patator
- Heartbleed
- Infiltration
- Web Attack - Brute Force
- Web Attack - SQL Injection
- Web Attack - XSS

Current Dataset Statistics

- **Rows:** 2,522,255
- **Features:** 78
- **Classes:** 15

---

# 🧠 Machine Learning Models

The project currently implements

- Logistic Regression
- Decision Tree
- Random Forest

Future versions will also compare

- XGBoost
- LightGBM
- Support Vector Machine

---

# 📊 Current Performance

Random Forest

| Metric | Value |
|---------|-------|
| Accuracy | **99.83%** |
| Classes | 15 |
| Dataset Size | 2.5 Million Flows |

> Note: Accuracy alone is not sufficient due to severe class imbalance. Precision, Recall, and F1-score are also evaluated for every attack category.

---

# 🏗 Project Structure

```text
ML-Network-Intrusion-Detection-System
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
│   └── main.py
│
├── frontend/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Ayush-Singh0583/ML-Network-Intrusion-Detection-System.git

cd ML-Network-Intrusion-Detection-System
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Place the CIC-IDS2017 CSV files inside

```text
data/
```

Run

```bash
python src/main.py
```

---

# 🔄 Machine Learning Pipeline

```text
Load CIC-IDS2017 CSV Files
            │
            ▼
Merge Datasets
            │
            ▼
Data Cleaning
            │
            ▼
Feature Engineering
            │
            ▼
Label Encoding
            │
            ▼
Train/Test Split
            │
            ▼
Feature Scaling
            │
            ▼
Model Training
            │
            ▼
Model Evaluation
            │
            ▼
Save Trained Model
```

---

# 📈 Results

The system evaluates each model using

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Feature importance is also exported for further analysis.

---

# 🚧 Roadmap

## Version 2.0 ✅

- [x] Multi-class classification
- [x] Automatic dataset loading
- [x] Complete preprocessing pipeline
- [x] Logistic Regression
- [x] Decision Tree
- [x] Random Forest

---

## Version 3.0

- [ ] FastAPI REST API
- [ ] Swagger Documentation
- [ ] CSV Prediction Endpoint
- [ ] Batch Prediction

---

## Version 4.0

- [ ] React Dashboard
- [ ] Upload CSV
- [ ] Attack Visualization
- [ ] Prediction History
- [ ] Interactive Charts

---

## Version 5.0

- [ ] Live Packet Capture
- [ ] Real-time Intrusion Detection
- [ ] Network Monitoring Dashboard
- [ ] Live Alerts
- [ ] Docker Deployment

---

# 🛠 Tech Stack

Programming

- Python

Machine Learning

- Scikit-learn
- Pandas
- NumPy

Visualization

- Matplotlib

Version Control

- Git
- GitHub

Dataset

- CIC-IDS2017

---

# 👨‍💻 Author

**Ayush Singh**

Information Science & Engineering

RV Institute of Technology and Management

---

# ⭐ Future Scope

- Real-time Network Intrusion Detection
- Docker Deployment
- Cloud Deployment
- Model Optimization
- Deep Learning Models
- Explainable AI (SHAP/LIME)
- Live Packet Monitoring using Scapy/PyShark

---

If you find this project useful, consider giving it a ⭐ on GitHub.