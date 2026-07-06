# Real-Time Machine Learning Data Drift Detection & Automated Retraining System

A complete end-to-end MLOps pipeline that continuously monitors incoming data for distribution drift, detects performance degradation using statistical techniques, and automatically retrains the model using recent data.

The system simulates a production ML environment with real-time data streaming, model inference, drift monitoring, automated retraining, and an interactive dashboard.

---

## 📌 Features

- 📊 Real-time data drift detection
- 📈 Statistical drift analysis using:
  - KL Divergence
  - Entropy
- 🤖 Automated model retraining when drift exceeds threshold
- ⚖️ Weighted retraining giving higher importance to recent data
- ⚡ Real-time model inference
- 📉 Live monitoring dashboard built with Streamlit
- 📋 Model performance tracking over time
- 📦 End-to-end ML pipeline from data ingestion to deployment

---

# System Architecture

```
                  Incoming Data Stream
                           │
                           ▼
                  Data Preprocessing
                           │
                           ▼
                    Model Inference
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
 Prediction Output                Drift Detection
                                          │
                     ┌────────────────────┴──────────────────┐
                     ▼                                       ▼
               No Drift                           Drift Detected
                     │                                       │
                     ▼                                       ▼
             Continue Serving                Automatic Retraining
                                                     │
                                                     ▼
                                            Updated ML Model
                                                     │
                                                     ▼
                                             Performance Dashboard
```

---

# Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Machine Learning | XGBoost |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Statistics | SciPy |
| Model Monitoring | Custom Drift Detection |
| Version Control | Git |

---

# Drift Detection

The system continuously compares the incoming production data with the training distribution.

## Metrics Used

### 1. KL Divergence

Measures how one probability distribution differs from another.

Higher KL divergence indicates greater drift.

\[
D_{KL}(P||Q)=\sum P(x)\log\frac{P(x)}{Q(x)}
\]

---

### 2. Entropy

Measures uncertainty in the incoming data distribution.

Higher entropy often indicates increased randomness or changes in feature distributions.

---

# Automated Retraining

Whenever drift exceeds a predefined threshold:

1. New data is collected.
2. Recent samples receive larger weights.
3. XGBoost model is retrained.
4. Updated model replaces the previous version.
5. Dashboard reflects new model metrics.

Weighted retraining helps the model adapt faster to changing environments while retaining historical knowledge.

---

# Dashboard

The Streamlit dashboard provides:

- Real-time predictions
- Drift score visualization
- KL Divergence trends
- Entropy monitoring
- Model accuracy
- Retraining history
- Performance metrics
- Incoming data statistics

---

# Project Structure

```
├── data/
│   ├── train.csv
│   ├── incoming_stream.csv
│
├── models/
│   ├── model.pkl
│
├── drift/
│   ├── drift_detector.py
│   ├── metrics.py
│
├── retraining/
│   ├── retrain.py
│
├── streaming/
│   ├── stream_data.py
│
├── dashboard/
│   ├── app.py
│
├── utils/
│
├── requirements.txt
│
├── README.md
│
└── main.py
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Pheonix-1002/Real-Time-ML-System-with-Drift-Detection-and-Weighted-Retraining.git
cd Real-Time-ML-System-with-Drift-Detection-and-Weighted-Retraining
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

Start the pipeline

```bash
python main.py
```

Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

# Workflow

1. Train initial model.
2. Stream incoming data.
3. Perform predictions.
4. Compute KL divergence and entropy.
5. Detect drift.
6. Trigger retraining if threshold is crossed.
7. Update deployed model.
8. Display metrics on dashboard.

---

# Example Use Cases

- Fraud Detection
- Predictive Maintenance
- Customer Churn Prediction
- Credit Risk Monitoring
- Healthcare Diagnosis Models
- Recommendation Systems
- IoT Sensor Monitoring
- Financial Forecasting

---

# Future Improvements

- Docker deployment
- Kubernetes support
- MLflow integration
- Prometheus & Grafana monitoring
- Kafka data streaming
- Airflow pipeline orchestration
- SHAP explainability
- CI/CD integration using GitHub Actions

---

# Results

- Continuous monitoring of production data
- Automatic drift detection
- Adaptive model retraining
- Reduced model degradation over time
- Interactive visualization of model health and drift statistics

---

# License

This project is licensed under the MIT License.

---

# Author

**Deepanshu Sharma**

If you found this project useful, feel free to ⭐ the repository!
