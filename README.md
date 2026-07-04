# 🛡️ ML-Based Network Intrusion Detection System (NIDS)

A Machine Learning based Network Intrusion Detection System capable of detecting malicious network traffic using supervised learning techniques.

This project is being developed incrementally. The current version focuses on **binary classification (DDoS vs BENIGN)** using the **CIC-IDS2017** dataset. Future versions will support multiple attack types, real-time packet capture, REST APIs, and a React dashboard.

---

# 📌 Current Status

**Version:** v1.0

## ✅ Implemented

- Data preprocessing
- Data cleaning
- Missing value handling
- Infinite value handling
- Duplicate removal
- Identifier feature removal
- Label Encoding
- Train-Test Split
- Feature Scaling
- Logistic Regression Classifier
- Model Evaluation
- Feature Importance Analysis
- Model Saving (.pkl)

---

# 📊 Current Results

Dataset:

**CIC-IDS2017 (Friday Working Hours Afternoon - DDoS)**

Classification:

```
BENIGN
DDoS
```

Model:

```
Logistic Regression
```

Accuracy

```
99.856%
```

Confusion Matrix

```
[[19494    43]
 [   22 25583]]
```

---

# 🏗 Current Architecture

```
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Train-Test Split
   │
   ▼
Standard Scaling
   │
   ▼
Logistic Regression
   │
   ▼
Evaluation
   │
   ▼
Feature Importance
   │
   ▼
Save Model
```

---

# 📂 Project Structure

```
ML_Intrusion_Detection/
│
├── data/
├── models/
├── notebooks/
├── frontend/
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   ├── prediction.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠 Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy

### Model Persistence

- Joblib

### Visualization

- Matplotlib

---

# 🚀 Future Roadmap

## v2.0

- Random Forest
- Model Comparison
- Hyperparameter Tuning

## v3.0

- Multi-class Attack Detection
- Support for Multiple CIC-IDS2017 Attack Types

## v4.0

- FastAPI REST API

## v5.0

- React Dashboard

## v6.0

- Live Packet Capture
- Real-Time Intrusion Detection

---

# 📥 Dataset

This project uses the **CIC-IDS2017 (Canadian Institute for Cybersecurity Intrusion Detection System 2017)** dataset.

The dataset is **not included** in this repository due to GitHub file size limits.

Download it from:

https://www.unb.ca/cic/datasets/ids-2017.html

Place

```
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

inside the `data` directory.

---

# ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/ML_Intrusion_Detection.git

cd ML_Intrusion_Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python src/main.py
```

---

# 📈 Planned Features

- Random Forest
- XGBoost
- Feature Selection
- Explainable AI (SHAP)
- REST API
- React Dashboard
- Live Packet Capture
- Attack Logs
- Database Support
- Docker Deployment

---

# 👨‍💻 Author

**Ayush Singh**

B.E. Information Science and Engineering

RV Institute of Technology and Management