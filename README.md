# 🚀 Real-Time ML System with Drift Detection and Weighted Retraining

A production-inspired MLOps project that simulates a real-time machine learning pipeline capable of detecting **data drift**, monitoring model performance, and automatically retraining the model when significant drift is detected.

The system generates synthetic streaming data, performs real-time inference using an **XGBoost** model, monitors statistical drift using **KL Divergence** and **Entropy**, and retrains the model with **weighted updates** that prioritize recent data. An interactive **Streamlit** dashboard provides live visualization of predictions, drift metrics, and model performance.

---

## ✨ Features

- 📊 Real-time synthetic data generation
- 🤖 XGBoost-based machine learning model
- 📈 Data drift detection using:
  - KL Divergence
  - Entropy
- 🔄 Automatic model retraining when drift exceeds a threshold
- ⚖️ Weighted retraining to prioritize recent observations
- 📉 Interactive Streamlit dashboard
- 📋 Real-time monitoring of model accuracy and drift metrics
- 🚀 End-to-end ML pipeline simulation

---

## 🏗️ System Workflow

```
Synthetic Data Generation
            │
            ▼
     Data Preprocessing
            │
            ▼
      Model Inference
            │
      ┌─────┴─────┐
      ▼           ▼
 Predictions   Drift Detection
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   No Significant Drift   Drift Detected
        │                     │
        ▼                     ▼
 Continue Monitoring   Automatic Retraining
                              │
                              ▼
                     Updated XGBoost Model
                              │
                              ▼
                    Performance Dashboard
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| ML Model | XGBoost |
| Dashboard | Streamlit |
| Data Processing | NumPy, Pandas |
| Visualization | Plotly, Matplotlib |
| Statistics | SciPy |
| Machine Learning | Scikit-learn |

---

## 📊 Drift Detection

The system continuously compares incoming streaming data with the reference training distribution using statistical metrics.

### Metrics Used

- **KL Divergence** – Measures the difference between two probability distributions.
- **Entropy** – Quantifies uncertainty in the incoming data distribution.

If the drift score exceeds a predefined threshold, the retraining pipeline is automatically triggered.

---

## 🔄 Automated Retraining

When drift is detected:

1. Recent streaming samples are collected.
2. Higher weights are assigned to recent observations.
3. The XGBoost model is retrained.
4. The updated model replaces the previous one.
5. Dashboard metrics are refreshed automatically.

---

## 📈 Dashboard

The Streamlit dashboard provides:

- Real-time predictions
- KL Divergence trend
- Entropy monitoring
- Drift alerts
- Model accuracy
- Retraining status
- Live performance metrics

---

## 🎲 Synthetic Data

This project does **not** rely on an external dataset.

Instead, synthetic streaming data is generated using Python's random number generation utilities to simulate changing data distributions and concept drift. This enables testing of real-world monitoring and retraining workflows without requiring a production dataset.

---

## 📂 Project Structure

```
Real-Time-ML-System-with-Drift-Detection-and-Weighted-Retraining/
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Real-Time-ML-System-with-Drift-Detection-and-Weighted-Retraining.git

cd Real-Time-ML-System-with-Drift-Detection-and-Weighted-Retraining
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- Docker containerization
- Kafka-based real-time streaming
- MLflow experiment tracking
- Prometheus & Grafana monitoring
- Airflow workflow orchestration
- Model versioning
- Cloud deployment (AWS/GCP/Azure)

---

## 👨‍💻 Author

**Deepanshu Sharma**

If you found this project useful, consider giving it a ⭐ on GitHub.
