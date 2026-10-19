# ML Model Evaluation and Monitoring

An enterprise machine learning evaluation and post-deployment monitoring platform built with **Python**, **Scikit-learn**, **MLflow**, and **Plotly**.

---

## Features

- **Experiment Tracking with MLflow**:
  - Model candidate training (Random Forest, Gradient Boosting, Logistic Regression) on tabular risk assessment datasets.
  - Automatic logging of hyperparameters, validation metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC), and model tags to local/remote MLflow servers.
- **Statistical Covariate Drift Detection**:
  - **Population Stability Index (PSI)**: Multi-quantile binning calculating distribution deviation:
    - $\text{PSI} < 0.1$: Distribution Stable.
    - $0.1 \le \text{PSI} < 0.2$: Moderate Drift Warning.
    - $\text{PSI} \ge 0.2$: Critical Drift requiring intervention.
  - **Kolmogorov-Smirnov (KS) Test**: Two-sample nonparametric hypothesis testing determining feature distribution divergence ($p < 0.05$).
- **Performance Regression & Retraining Alerts**:
  - Automated detection of production F1 degradation compared to baseline thresholds.
  - Visual density distribution overlays comparing training baselines against live production batches.
- **Interactive Streamlit Dashboard**: Seamless interface for model training runs, drift scorecards, and distribution density charts.

---

## Project Structure

```text
ml-model-monitoring/
├── data/
│   ├── data_generator.py         # Synthetic baseline & drifted dataset generator
│   ├── training_data.csv         # Baseline training distribution (1,000 samples)
│   ├── production_normal.csv     # In-distribution production data (500 samples)
│   └── production_drifted.csv    # Covariate-shifted production data (500 samples)
├── src/
│   ├── drift_detector.py         # PSI, KS-test, and performance regression logic
│   └── trainer.py                # Scikit-learn model training & MLflow tracker
├── tests/
│   ├── test_drift_detector.py    # Unit tests for PSI, KS-tests, and alert thresholds
│   └── test_trainer.py           # Model training and metric computation tests
├── app.py                        # Streamlit monitoring dashboard
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+

### 2. Installation

```bash
cd ml-model-monitoring
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

python -m pip install -r requirements.txt
```

### 3. Running the Dashboard

Launch the Streamlit app:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

To view the MLflow UI directly:
```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000).

---

## Automated Tests

Run the PyTest test suite:

```bash
python -m pytest tests -v
```

Tests cover:
- Data splitting and stratified validation
- Model training, metric logging, and score validation
- PSI calculation on stable vs drifted distributions
- Kolmogorov-Smirnov two-sample test divergence detection
- Performance drop calculation and automated retraining alerts

---

## Security Considerations

1. **Model Serialization**: When storing and loading serialized model artifacts (`.pkl` / `.joblib`), verify artifact signatures and integrity to avoid arbitrary code execution vulnerabilities.
2. **Data Privacy**: Ensure feature sets do not include prohibited attributes or unhashed Personally Identifiable Information (PII).

---

## Known Limitations

- **Ground Truth Delay**: Model performance regression alerts rely on post-event target label availability; for delayed feedback loops, feature drift (PSI/KS) serves as the primary leading indicator.
- **Multivariate Drift**: Features are currently analyzed univariately; joint high-dimensional drift detection (e.g. Mahalanobis distance or autoencoder reconstruction error) can be added for complex feature interactions.
