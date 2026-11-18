# WATTWISE — Intelligent Energy Consumption Prediction System for Smart Buildings

> **Academic Capstone Project — CSE276**  
> **Institution:** Lovely Professional University  
> **Student Developers:**  
> - **Itesh Choudhary** (Registration No. `12517042`)  
> - **Vivak Mishra** (Registration No. `12516612`)  

---

## 1. Executive Summary & Overview

Modern commercial buildings account for over 35% of global electrical energy consumption and are responsible for significant greenhouse gas emissions. In smart facilities, heating, ventilation, and air-conditioning (HVAC), server infrastructure, lighting, and tenant plug loads exhibit complex, non-linear dependencies on ambient weather conditions, occupancy schedules, and historical consumption momentum.

**WATTWISE** is an end-to-end intelligent energy consumption prediction and load management web application. It integrates time-series feature engineering, strict chronological cross-validation, and multi-model machine learning regression to forecast electrical demand in kilowatt-hours (kWh) and estimate commercial energy costs in Indian Rupees (₹).

### Key Features
- **Modern Glassmorphic Dark UI:** Premium deep-navy background (`#090b12`) with cyan (`#5ee5df`) and violet (`#9e8bff`) accents, Space Grotesk typography, and fluid micro-animations.
- **Dynamic Smart Building Art:** Animated building wireframe with floor-level power metrics, scan-beam telemetry, and real-time wave bars.
- **Strict Chronological Validation:** Chronological 80/20 train-test splitting to eliminate future data leakage.
- **Multi-Model Benchmark:** Simultaneous training and automated selection across **Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**.
- **Transparent Evaluation Metrics:** Computes real **MAE**, **RMSE**, and **R² Score** without fabricated accuracy metrics.
- **Explainable AI Breakdown:** Decomposes predicted power into Base Building Load, HVAC Thermal Load, and Occupancy Plug Load.
- **Custom CSV Upload & Cleaning:** Drag-and-drop CSV ingestion with timestamp validation, sensor gap interpolation, and retrain pipeline.
- **Interactive Chart.js Analytics:** Hourly (48h), Daily, Weekly Day-of-Week profiles, Test-Set Ground Truth vs Prediction overlay, and Consumption Frequency Distribution.

---

## 2. System Architecture

```
+---------------------------------------------------------------------------------+
|                                 CLIENT BROWSER                                  |
|   HTML5 / CSS3 (Navy & Cyan Glassmorphism) / Vanilla JS (ES6+) / Chart.js v4    |
|   - Real-time Dashboards       - Interactive Prediction Workspace               |
|   - CSV Upload & Preview       - Model Benchmarking & Telemetry Charts          |
+---------------------------------------------------------------------------------+
                                       │ ▲
                        HTTP REST JSON │ │ Responses
                                       ▼ │
+---------------------------------------------------------------------------------+
|                                FLASK BACKEND                                    |
|   app.py (REST API Router & File Upload Controller)                             |
|   - GET  /                     - POST /api/predict     - POST /api/upload       |
|   - GET  /api/health           - GET  /api/metrics     - POST /api/train        |
|   - GET  /api/analytics        - GET  /api/model-info  - POST /api/reset-demo   |
+---------------------------------------------------------------------------------+
          │                                                    │
          ▼                                                    ▼
+------------------------------------+   +------------------------------------+
|       PREPROCESSING PIPELINE       |   |       MACHINE LEARNING ENGINE      |
|       (preprocessing.py)           |   |       (train_model.py)             |
|   - Chronological Sorting          |   |   - Linear Regression              |
|   - Missing Value Imputation       |   |   - Random Forest Regressor        |
|   - Lag Features (t-1, t-2, t-24)  |   |   - Gradient Boosting Regressor    |
|   - Rolling Means (6h, 24h)        |   |   - Chronological Evaluation       |
|   - Cooling & Heating Degrees      |   |   - Automated Model Selection      |
+------------------------------------+   +------------------------------------+
                                           │
                                           ▼
                                 +-------------------+
                                 |  JOBLIB ARTIFACT  |
                                 |  models/bundle    |
                                 +-------------------+
```

---

## 3. Mathematical Foundations & Metrics

Model performance is evaluated on the chronological out-of-sample test partition using three standard regression metrics:

### 1. Mean Absolute Error (MAE)
Quantifies the average absolute deviation between predicted ($\hat{y}_i$) and actual ($y_i$) energy consumption in kWh:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

### 2. Root Mean Squared Error (RMSE)
Penalizes larger forecast errors more heavily, reflecting operational disruption during unexpected peak electrical demand:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

### 3. Coefficient of Determination ($R^2$ Score)
Measures the proportion of total variance in electrical consumption explained by the engineered meteorological, operational, and lag features:
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
Where $\bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$ is the empirical mean of measured consumption.

---

## 4. Machine Learning Pipeline & Feature Engineering

### Feature Dictionary
| Feature | Type | Description |
|---|---|---|
| `hour` | Integer (0–23) | Diurnal solar and business hour cycle |
| `dayofweek` | Integer (0–6) | Weekly operational schedule (0 = Monday) |
| `month` | Integer (1–12) | Seasonal climatic variation |
| `is_weekend` | Binary (0 or 1) | Weekend occupancy setback flag |
| `is_business_hour` | Binary (0 or 1) | Commercial operating hours (08:00–18:00 Mon–Fri) |
| `temperature` | Float (°C) | Ambient outdoor temperature |
| `humidity` | Float (%) | Relative humidity percentage |
| `occupancy` | Integer | Total active occupants in the smart facility |
| `lag_1` | Float (kWh) | Measured energy consumption at $t-1$ hour |
| `lag_2` | Float (kWh) | Measured energy consumption at $t-2$ hours |
| `lag_24` | Float (kWh) | Same-hour energy consumption yesterday ($t-24$) |
| `rolling_mean_6` | Float (kWh) | 6-hour backward moving average (excluding current) |
| `rolling_mean_24` | Float (kWh) | 24-hour backward moving average (excluding current) |
| `cooling_degree` | Float (°C) | $\max(0, \text{temperature} - 22.0)$ HVAC cooling demand |
| `heating_degree` | Float (°C) | $\max(0, 18.0 - \text{temperature})$ HVAC heating demand |

### Chronological Train-Test Split (Preventing Data Leakage)
In time-series modeling, random k-fold shuffling is strictly invalid because future information leaks into the training partition. WattWise partitions data chronologically:
- **Training Set (Earliest 80%):** Fits model parameters and standardizes scalers.
- **Testing Set (Latest 20%):** Validates real-world out-of-sample generalization.

---

## 5. Project Directory Structure

```
intelligent-energy-prediction/
├── app.py                      # Flask REST API server & web controller
├── train_model.py              # ML pipeline: train, evaluate & serialize
├── prediction.py               # Inference engine & explainability logic
├── preprocessing.py            # Data cleaning, lags, rolling averages & demo generator
├── requirements.txt            # Python dependencies (Flask, pandas, scikit-learn, etc.)
├── Procfile                    # Render/Heroku process configuration (Gunicorn)
├── render.yaml                 # Render Blueprint deployment definition
├── .gitignore                  # Git exclusions (virtualenvs, temporary uploads)
├── README.md                   # Comprehensive academic documentation
├── setup_windows.bat           # Automated Windows setup script
├── run_windows.bat             # Automated Windows run script
│
├── models/
│   ├── .gitkeep
│   └── energy_model_bundle.joblib  # Serialized model, scaler & metrics bundle
│
├── data/
│   ├── .gitkeep
│   ├── active_dataset.json     # Configuration tracking active dataset
│   └── demo_smart_building_energy.csv # 90-day hourly synthetic benchmark
│
├── uploads/
│   └── .gitkeep                # Staged custom uploaded CSVs
│
├── templates/
│   └── index.html              # Modern glassmorphic dashboard template
│
└── static/
    ├── css/
    │   └── style.css           # Premium navy, cyan & violet stylesheet
    └── js/
        └── app.js              # Chart.js controller & dynamic UI logic
```

---

## 6. REST API Reference

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/` | `GET` | — | Serves the interactive WattWise web application |
| `/api/health` | `GET` | — | System health, service status, and model readiness |
| `/api/metrics` | `GET` | — | Returns MAE, RMSE, R² scores for all 3 benchmarked models |
| `/api/analytics` | `GET` | — | Returns 48h hourly, daily, weekly, actual-vs-predicted, and distribution telemetry |
| `/api/predict` | `POST` | JSON Body | Computes next-hour kWh prediction, cost (₹), and component breakdown |
| `/api/upload` | `POST` | `multipart/form-data` | Ingests, validates, cleans, and previews user CSV datasets |
| `/api/train` | `POST` | — | Triggers chronological retraining of all 3 models on the active dataset |
| `/api/reset-demo` | `POST` | — | Restores the default 90-day synthetic building simulation dataset |
| `/api/model-info` | `GET` | — | Model metadata, feature importance, and training timestamps |

### Example Prediction Request:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 32.5,
    "humidity": 60,
    "occupancy": 150,
    "hour": 14,
    "day_of_week": 2,
    "prev_energy_kwh": 65.2
  }'
```

---

## 7. Local Setup and Installation

### Prerequisites
- Python 3.11+
- Git

### Quick Setup (Windows)
1. Double-click `setup_windows.bat` to create `.venv`, install packages, and train the baseline model.
2. Double-click `run_windows.bat` to launch the server at `http://127.0.0.1:5000`.

### Manual Setup (Cross-Platform / Linux / macOS)
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/intelligent-energy-prediction.git
cd intelligent-energy-prediction

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Train initial model on benchmark dataset
python train_model.py

# 6. Launch web server
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

---

## 8. GitHub and Render Deployment Guide

### A. Pushing to GitHub
```bash
# Initialize git repository
git init

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/intelligent-energy-prediction.git

# Stage files (verified by .gitignore to exclude virtualenvs and temporary uploads)
git add .

# Commit
git commit -m "Initial release: WattWise Smart Energy AI Prediction System"

# Push to main branch
git branch -M main
git push -u origin main
```

### B. Deploying to Render
1. Create a free account on [Render](https://render.com/).
2. On your Render Dashboard, click **New +** and select **Web Service**.
3. Connect your GitHub repository: `intelligent-energy-prediction`.
4. Configure service settings:
   - **Name:** `wattwise-energy-prediction`
   - **Environment:** `Python`
   - **Region:** Any (e.g. `Oregon (US West)`)
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type:** `Free`
5. Click **Create Web Service**.
6. Render will automatically build the environment, execute `train_model.py` to create the baseline model bundle, and bind Gunicorn to the dynamic `$PORT`.

---

## 9. Academic Integrity & Ethics Statement

1. **Synthetic Demo Labeling:** The built-in 90-day dataset is generated through physical smart building equations and is transparently labeled as **"Synthetic Simulation Demo"**. It is never misrepresented as actual empirical hardware measurements.
2. **True Validation:** All displayed R², MAE, and RMSE figures are computed directly from out-of-sample testing via `scikit-learn` algorithms.
3. **Realistic Utility Framing:** The system is designed for **forecasting, peak-shaving insight, and facility planning**, and does not claim to magically reduce physical energy usage on its own.

---

## 10. Acknowledgements

- **Course:** CSE276 Capstone Project
- **University:** Lovely Professional University
- **Supervising Faculty & Department of Computer Science & Engineering**
