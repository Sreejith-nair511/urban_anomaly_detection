# Smart Urban Anomaly Detection System
### Bangalore Urban Intelligence Platform

---

## Overview

The Smart Urban Anomaly Detection System is a production-grade, full-stack machine learning platform designed to monitor, analyze, and detect anomalies across urban sensor networks in Bangalore. The system ingests multi-dimensional environmental and traffic data, applies a layered unsupervised and supervised ML pipeline, and surfaces actionable insights through a modern web dashboard and a REST API.

The platform is built to simulate real-world deployment scenarios where sensor data streams from multiple city zones are continuously evaluated for pollution spikes, traffic surges, noise pollution events, and combined environmental irregularities.

---

## Architecture

The system is organized into three independent layers that communicate through well-defined interfaces.

```
urban_anomaly_detection/
|
|-- backend/                        FastAPI ML service
|   |-- app/
|   |   |-- main.py                 Application entry point, lifespan management
|   |   |-- routes/
|   |   |   |-- predict.py          REST API route handlers
|   |   |-- ml/
|   |   |   |-- generator.py        Synthetic data generation and anomaly injection
|   |   |   |-- preprocessing.py    Feature engineering and normalization
|   |   |   |-- pca_module.py       PCA dimensionality reduction
|   |   |   |-- clustering.py       K-Means and DBSCAN clustering
|   |   |   |-- labeling.py         Label engineering from cluster outputs
|   |   |   |-- models.py           Random Forest and XGBoost training
|   |   |   |-- evaluation.py       Classification metrics computation
|   |   |   |-- pipeline.py         Full pipeline orchestrator and in-memory cache
|   |   |-- schemas/
|   |       |-- input_schema.py     Pydantic request/response models
|   |-- requirements.txt
|
|-- frontend/                       Next.js dashboard
|   |-- app/
|   |   |-- layout.tsx              Root layout with global styles
|   |   |-- page.tsx                Root redirect to dashboard
|   |   |-- dashboard/
|   |   |   |-- layout.tsx          Dashboard shell: sidebar + topbar
|   |   |   |-- page.tsx            Overview page
|   |   |   |-- analytics/
|   |   |   |   |-- page.tsx        Analytics and feature insights page
|   |   |   |-- clustering/
|   |   |   |   |-- page.tsx        PCA scatter and cluster analysis page
|   |   |   |-- predictions/
|   |   |       |-- page.tsx        Live anomaly prediction page
|   |   |-- components/
|   |   |   |-- Sidebar.tsx         Fixed navigation sidebar
|   |   |   |-- Card.tsx            KPI cards, section cards, utility components
|   |   |   |-- Charts.tsx          All Recharts chart components
|   |   |   |-- PredictionForm.tsx  Interactive prediction form with sliders
|   |   |-- lib/
|   |       |-- api.ts              Axios API client and TypeScript types
|   |-- tailwind.config.js
|   |-- tsconfig.json
|   |-- package.json
|   |-- .env.local
|
|-- src/                            Standalone ML modules (Streamlit pipeline)
|   |-- generator.py
|   |-- preprocessing.py
|   |-- pca_module.py
|   |-- clustering.py
|   |-- labeling.py
|   |-- models.py
|   |-- evaluation.py
|   |-- visualization.py
|   |-- data_loader.py
|
|-- app/
|   |-- dashboard.py                Streamlit dashboard (standalone UI)
|
|-- data/
|   |-- raw/                        Auto-generated synthetic CSV
|   |-- processed/                  Labeled dataset, scaler artifact, plots
|
|-- notebooks/
|   |-- eda.ipynb                   Exploratory data analysis notebook
|
|-- main.py                         CLI pipeline runner
|-- README.md
```

---

## Machine Learning Pipeline

The ML pipeline executes once at backend startup and caches all results in memory. No model retraining occurs per API request.

### 1. Data Generation

Synthetic sensor data is generated to simulate 10 Bangalore zones: Koramangala, Whitefield, Indiranagar, Hebbal, Electronic City, Jayanagar, Marathahalli, Yelahanka, BTM Layout, and HSR Layout.

Base dataset: 5,000 records at 30-minute intervals from January 2024.
Augmentation: 300 additional extreme anomaly records appended.

Features generated per record:

| Feature | Range | Description |
|---|---|---|
| timestamp | 2024-01-01 onwards | 30-minute intervals |
| location_id | 10 zones | Bangalore urban zones |
| AQI | 50 - 500 | Air Quality Index |
| temperature | 18 - 40 C | Ambient temperature |
| humidity | 35 - 90 % | Relative humidity |
| traffic_density | 0 - 100 | Normalized traffic load |
| noise_level | 35 - 120 dB | Ambient noise level |

Anomaly injection strategy (13% of base data):

- AQI spikes: AQI raised to 310-500, humidity elevated to 75-90%
- Traffic surges: traffic_density raised to 82-100, noise_level to 85-120 dB
- Combined events: AQI 280-450, traffic 78-100, noise 82-115 dB, temperature 36-40 C

### 2. Preprocessing

- Missing value imputation using column medians
- Temporal feature engineering: hour of day (0-23), day of week (0-6)
- StandardScaler normalization fitted on all 7 feature columns
- Scaler artifact persisted to disk for inference reuse

### 3. Dimensionality Reduction (PCA)

Principal Component Analysis reduces the 7-feature normalized space to 2 components for visualization. Explained variance is approximately 31% for PC1 and 14% for PC2, totaling 45% of total variance captured in 2 dimensions.

### 4. Clustering

Two clustering algorithms run in parallel on the full normalized feature space:

**K-Means (k=3)**
- Partitions data into 3 clusters using Euclidean distance
- Silhouette score computed to measure cluster cohesion
- Cluster 1 is designated as the Moderate anomaly class

**DBSCAN (eps=1.5, min_samples=15)**
- Density-based spatial clustering
- Points labeled -1 (noise) are treated as High Anomaly detections
- Parameters tuned for normalized 7-dimensional urban data

### 5. Label Engineering

Final labels are derived from the combination of DBSCAN and K-Means outputs using a priority rule:

| Condition | Label | Encoded Value |
|---|---|---|
| DBSCAN label == -1 | High Anomaly | 2 |
| K-Means cluster == 1 | Moderate | 1 |
| Otherwise | Normal | 0 |

DBSCAN anomaly designation takes priority over K-Means cluster assignment.

### 6. Supervised Classification

Two classifiers are trained on the engineered labels using an 80/20 stratified train/test split:

**Random Forest**
- 200 estimators, max depth 10
- Parallel training with all available CPU cores
- Feature importances extracted post-training

**XGBoost**
- 200 estimators, max depth 6, learning rate 0.1
- Multi-class log loss evaluation metric
- Feature importances extracted post-training

### 7. Evaluation

Both models are evaluated on the held-out test set:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Confusion Matrix (3x3)
- K-Means Silhouette Score
- PCA Explained Variance Ratio

Typical results on this dataset:

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | ~0.989 | ~0.988 |
| F1 Score | ~0.988 | ~0.988 |

---

## API Reference

The FastAPI backend exposes four endpoints. All ML endpoints are prefixed with `/api`.

### GET /health

Returns service health status.

Response:
```json
{
  "status": "ok",
  "service": "Urban Anomaly Detection API"
}
```

### POST /api/predict

Accepts sensor readings and returns anomaly classification from both models.

Request body:
```json
{
  "AQI": 320,
  "temperature": 34,
  "humidity": 72,
  "traffic_density": 85,
  "noise_level": 95,
  "hour": 9,
  "day_of_week": 1
}
```

Response:
```json
{
  "rf_prediction": "High Anomaly",
  "xgb_prediction": "High Anomaly",
  "rf_confidence": {
    "Normal": 0.02,
    "Moderate": 0.05,
    "High Anomaly": 0.93
  },
  "xgb_confidence": {
    "Normal": 0.01,
    "Moderate": 0.04,
    "High Anomaly": 0.95
  }
}
```

### GET /api/metrics

Returns model evaluation metrics, feature importances, and clustering quality scores.

Response fields: `random_forest`, `xgboost`, `kmeans_silhouette`, `pca_variance`, `rf_importance`, `xgb_importance`

### GET /api/data-summary

Returns all data required to populate the dashboard: label distribution, zone anomaly rates, hourly trends, PCA scatter coordinates (1,500 point sample), and AQI histogram buckets.

---

## Dashboard Pages

The Next.js frontend connects to the FastAPI backend and renders four pages.

### Overview

Displays five KPI cards: total samples, high anomaly count, moderate event count, RF accuracy, and XGB accuracy. Includes a label distribution breakdown with percentage bars and a zone-level anomaly rate comparison.

### Analytics

Feature distribution charts, hourly sensor trend lines across all zones, Random Forest and XGBoost feature importance bar charts, PCA explained variance visualization, and K-Means silhouette score display.

### Clustering

Interactive PCA 2D scatter plot with label-based coloring. Filter controls allow isolating Normal, Moderate, or High Anomaly points. Includes cluster statistics and zone anomaly rate chart.

### Predictions

Live anomaly prediction panel. Users adjust sliders for AQI, temperature, humidity, traffic density, noise level, hour of day, and day of week. On submission, both RF and XGBoost predictions are returned with confidence probability bars and a radar chart comparing model confidence distributions.

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- npm 9 or higher

### Backend Setup

```bash
cd urban_anomaly_detection/backend
pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn app.main:app --reload --port 8000
```

The ML pipeline trains automatically on first startup. Subsequent requests use the cached in-memory state. The API will be available at http://localhost:8000.

Interactive API documentation is available at http://localhost:8000/docs.

If multiple Python versions are installed, ensure uvicorn and all dependencies use the same interpreter:

```bash
/path/to/python -m pip install -r requirements.txt
/path/to/python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd urban_anomaly_detection/frontend
npm install
npm run dev
```

The dashboard will be available at http://localhost:3000.

The API base URL is configured in `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Standalone Streamlit Dashboard (Optional)

A self-contained Streamlit dashboard is also available and does not require the FastAPI backend:

```bash
cd urban_anomaly_detection
pip install streamlit plotly
streamlit run app/dashboard.py
```

### CLI Pipeline Runner

To run the full ML pipeline from the command line and generate static plots:

```bash
cd urban_anomaly_detection
python main.py
```

Output artifacts are saved to `data/processed/`.

---

## Technology Stack

### Backend

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Data Processing | pandas, numpy |
| ML Pipeline | scikit-learn |
| Gradient Boosting | XGBoost |
| Serialization | Pydantic v2 |

### Frontend

| Component | Technology |
|---|---|
| Framework | Next.js 15 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Charts | Recharts |
| HTTP Client | Axios |
| Icons | Lucide React |

---

## Configuration

### Backend

The backend requires no external configuration beyond the Python environment. Data is generated synthetically on first run and cached to `data/raw/urban_data.csv`. The scaler artifact is saved to `data/processed/scaler.pkl`.

CORS is configured to allow requests from `http://localhost:3000`. To deploy to a different origin, update the `allow_origins` list in `backend/app/main.py`.

### Frontend

All environment variables are prefixed with `NEXT_PUBLIC_` and defined in `.env.local`.

| Variable | Default | Description |
|---|---|---|
| NEXT_PUBLIC_API_URL | http://localhost:8000 | FastAPI backend base URL |

---

## Data Notes

All data used in this system is synthetically generated. No real sensor data from Bangalore or any other city is used. The synthetic generation is designed to produce statistically realistic distributions that reflect known urban patterns:

- AQI follows a bimodal distribution with a normal operating range of 50-200 and injected spikes above 300
- Traffic density follows diurnal patterns with peaks during morning and evening hours
- Noise levels correlate with traffic density with added variance for construction and event noise
- Temperature and humidity are drawn from Bangalore's typical climate range

The anomaly injection rate is set at 13% of the base dataset, consistent with real-world urban anomaly prevalence estimates.

---

## Project Status

This project is a complete, runnable demonstration of a production-style ML system. It is intended for educational, portfolio, and prototyping purposes. Extending it to use real sensor data would require replacing the `generator.py` module with a live data ingestion layer while keeping all downstream pipeline components unchanged.
