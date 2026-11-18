"""
WattWise - Smart Energy AI
Flask Application Server & REST API.

Provides endpoints for:
- Serving modern glassmorphic dashboard UI
- CSV dataset upload, preview, and validation
- Machine learning model training & chronological evaluation
- Real-time energy consumption inference
- Analytics telemetry for Chart.js visualizations
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from prediction import EnergyPredictor, TARIFF_INR_PER_KWH
from preprocessing import (
    generate_synthetic_building_dataset,
    parse_and_clean_data,
)
from train_model import (
    DEFAULT_DEMO_PATH,
    MODEL_BUNDLE_PATH,
    ensure_demo_dataset,
    train_pipeline,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
ACTIVE_DATASET_CONFIG = DATA_DIR / "active_dataset.json"
ALLOWED_EXTENSIONS = {".csv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# Ensure essential directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize predictor
predictor = EnergyPredictor(bundle_path=MODEL_BUNDLE_PATH)


def get_active_dataset_path() -> Path:
    """Returns the path to the currently active CSV dataset."""
    if ACTIVE_DATASET_CONFIG.exists():
        try:
            with open(ACTIVE_DATASET_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
                active_path = Path(data.get("path", ""))
                if active_path.exists():
                    return active_path
        except Exception:
            pass
    return ensure_demo_dataset()


def set_active_dataset_path(path: Path, is_synthetic: bool = False, original_name: str = ""):
    """Updates the record of the currently active dataset."""
    with open(ACTIVE_DATASET_CONFIG, "w", encoding="utf-8") as f:
        json.dump(
            {
                "path": str(path.resolve()),
                "is_synthetic": is_synthetic,
                "original_name": original_name or path.name,
            },
            f,
            indent=2,
        )


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Page Routes
# --------------------------------------------------------------------------

@app.get("/")
def index():
    active_path = get_active_dataset_path()
    is_synthetic = active_path.name == DEFAULT_DEMO_PATH.name
    return render_template(
        "index.html",
        model_ready=predictor.is_ready,
        is_synthetic=is_synthetic,
        tariff=TARIFF_INR_PER_KWH,
    )


# --------------------------------------------------------------------------
# REST API Endpoints
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    active_path = get_active_dataset_path()
    return jsonify({
        "status": "healthy",
        "service": "WattWise Smart Energy AI",
        "version": "1.0.0",
        "model_ready": predictor.is_ready,
        "active_dataset": active_path.name,
        "is_synthetic": active_path.name == DEFAULT_DEMO_PATH.name,
        "tariff_per_kwh_inr": TARIFF_INR_PER_KWH,
    })


@app.get("/api/model-info")
def model_info():
    if not predictor.is_ready:
        return jsonify({
            "status": "not_ready",
            "message": "Machine learning model has not been trained yet. Call POST /api/train or trigger training in the dashboard.",
        }), 503
    return jsonify(predictor.get_info())


@app.get("/api/metrics")
def get_metrics():
    if not predictor.is_ready:
        return jsonify({"error": "Model bundle not loaded. Please train the model."}), 503

    bundle = predictor.bundle
    return jsonify({
        "best_model_name": bundle["best_model_name"],
        "all_metrics": bundle["all_metrics"],
        "dataset_info": bundle["dataset_info"],
        "feature_importance": bundle.get("feature_importance", {}),
        "trained_at": bundle.get("trained_at"),
    })


@app.post("/api/predict")
def predict_energy():
    if not predictor.is_ready:
        return jsonify({
            "error": "The AI model is not ready. Train the model first.",
            "code": "MODEL_NOT_READY",
        }), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid request: JSON body required."}), 400

    required_fields = ["temperature", "humidity", "occupancy", "hour", "day_of_week", "prev_energy_kwh"]
    missing = [f for f in required_fields if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required input fields: {', '.join(missing)}"}), 400

    try:
        prediction_result = predictor.predict(
            temperature=payload["temperature"],
            humidity=payload["humidity"],
            occupancy=payload["occupancy"],
            hour=payload["hour"],
            day_of_week=payload["day_of_week"],
            prev_energy_kwh=payload["prev_energy_kwh"],
        )
        return jsonify(prediction_result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Internal prediction error: {str(exc)}"}), 500


@app.post("/api/upload")
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Please upload a CSV file."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file was selected for upload."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Please upload a CSV file (.csv)."}), 400

    orig_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{orig_name}"
    saved_path = UPLOAD_DIR / unique_name

    try:
        file.save(saved_path)

        # Inspect and validate CSV
        try:
            df = pd.read_csv(saved_path)
        except Exception as e:
            return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 422

        # Validate headers & clean
        try:
            cleaned_df = parse_and_clean_data(df)
        except ValueError as ve:
            return jsonify({
                "error": str(ve),
                "details": "Ensure your CSV has 'timestamp' and 'energy_kwh' columns and at least 50 valid chronological records."
            }), 422

        # Record as active dataset
        set_active_dataset_path(saved_path, is_synthetic=False, original_name=orig_name)

        # Build preview of first 5 rows and summary statistics
        preview_rows = cleaned_df.head(5).to_dict(orient="records")
        for row in preview_rows:
            if "timestamp" in row and hasattr(row["timestamp"], "strftime"):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        summary = {
            "total_records": len(cleaned_df),
            "start_time": str(cleaned_df["timestamp"].min()),
            "end_time": str(cleaned_df["timestamp"].max()),
            "mean_energy_kwh": round(float(cleaned_df["energy_kwh"].mean()), 2),
            "min_energy_kwh": round(float(cleaned_df["energy_kwh"].min()), 2),
            "max_energy_kwh": round(float(cleaned_df["energy_kwh"].max()), 2),
            "has_temperature": "temperature" in df.columns,
            "has_humidity": "humidity" in df.columns,
            "has_occupancy": "occupancy" in df.columns,
        }

        return jsonify({
            "message": "Dataset successfully uploaded and validated.",
            "filename": orig_name,
            "summary": summary,
            "preview": preview_rows,
            "columns": list(cleaned_df.columns),
        })

    except Exception as exc:
        app.logger.exception("Upload handling error")
        return jsonify({"error": f"Unexpected error processing upload: {str(exc)}"}), 500


@app.post("/api/train")
def train_models():
    """Triggers model training on the currently active dataset."""
    active_path = get_active_dataset_path()
    try:
        bundle = train_pipeline(csv_path=active_path)
        predictor.reload_model = predictor.load_model()

        return jsonify({
            "message": "Models successfully trained and evaluated.",
            "best_model_name": bundle["best_model_name"],
            "all_metrics": bundle["all_metrics"],
            "dataset_info": bundle["dataset_info"],
            "trained_at": bundle["trained_at"],
            "training_time_seconds": bundle["total_pipeline_time_seconds"],
        })
    except Exception as exc:
        app.logger.exception("Training pipeline failed")
        return jsonify({"error": f"Model training failed: {str(exc)}"}), 500


@app.post("/api/reset-demo")
def reset_demo():
    """Resets the active dataset to the default synthetic smart building demo dataset and retrains."""
    demo_path = ensure_demo_dataset()
    set_active_dataset_path(demo_path, is_synthetic=True, original_name="demo_smart_building_energy.csv")

    try:
        bundle = train_pipeline(csv_path=demo_path)
        predictor.load_model()
        return jsonify({
            "message": "Reset to synthetic smart building demo dataset completed.",
            "best_model_name": bundle["best_model_name"],
            "all_metrics": bundle["all_metrics"],
            "dataset_info": bundle["dataset_info"],
        })
    except Exception as exc:
        app.logger.exception("Reset demo failed")
        return jsonify({"error": f"Reset failed: {str(exc)}"}), 500


@app.get("/api/analytics")
def get_analytics():
    """
    Returns aggregated metrics and timeseries data for Chart.js:
    - 4 Top Metric Cards (Current kWh, Next-hour predicted kWh, Daily Average kWh, Estimated Cost ₹)
    - Hourly consumption (last 48h)
    - Daily consumption totals
    - Weekly averages (Mon-Sun)
    - Actual vs Predicted (from test evaluation)
    - Distribution histogram bins
    """
    active_path = get_active_dataset_path()
    try:
        raw_df = pd.read_csv(active_path)
        df = parse_and_clean_data(raw_df)
    except Exception as exc:
        return jsonify({"error": f"Could not read active dataset: {str(exc)}"}), 500

    # 1. Metric Cards calculations
    current_kwh = round(float(df["energy_kwh"].iloc[-1]), 2)
    last_row = df.iloc[-1]

    # Predict next hour if model is ready
    if predictor.is_ready:
        next_hour = (int(last_row["timestamp"].hour) + 1) % 24
        next_day = int(last_row["timestamp"].weekday())
        if next_hour == 0:
            next_day = (next_day + 1) % 7
        try:
            pred_res = predictor.predict(
                temperature=float(last_row.get("temperature", 28.0)),
                humidity=float(last_row.get("humidity", 55.0)),
                occupancy=int(last_row.get("occupancy", 50)),
                hour=next_hour,
                day_of_week=next_day,
                prev_energy_kwh=current_kwh,
            )
            predicted_next_kwh = pred_res["predicted_kwh"]
        except Exception:
            predicted_next_kwh = current_kwh
    else:
        predicted_next_kwh = current_kwh

    # Daily aggregation
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    daily_grp = df.groupby("date_str")["energy_kwh"].sum()
    avg_daily_kwh = round(float(daily_grp.mean()), 2)
    estimated_daily_cost_inr = round(avg_daily_kwh * TARIFF_INR_PER_KWH, 2)

    # 2. Hourly Timeseries (last 48 hours for crisp visualization)
    recent_48 = df.tail(48).copy()
    hourly_data = {
        "labels": [dt.strftime("%d %b %H:%M") for dt in recent_48["timestamp"]],
        "values": [round(float(v), 2) for v in recent_48["energy_kwh"]],
        "temperatures": [round(float(v), 1) for v in recent_48["temperature"]],
        "occupancies": [int(v) for v in recent_48["occupancy"]],
    }

    # 3. Daily Totals (last 14 days)
    recent_daily = daily_grp.tail(14)
    daily_data = {
        "labels": [datetime.strptime(d, "%Y-%m-%d").strftime("%d %b") for d in recent_daily.index],
        "values": [round(float(v), 2) for v in recent_daily.values],
    }

    # 4. Weekly Average by Day of Week (0=Monday ... 6=Sunday)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["dow"] = df["timestamp"].dt.day_of_week
    dow_means = df.groupby("dow")["energy_kwh"].mean()
    weekly_data = {
        "labels": day_names,
        "values": [round(float(dow_means.get(i, 0.0)), 2) for i in range(7)],
    }

    # 5. Actual vs Predicted (from test split in model bundle)
    actual_vs_predicted = {"timestamps": [], "actual": [], "predicted": []}
    if predictor.is_ready and "test_chart_data" in predictor.bundle:
        actual_vs_predicted = predictor.bundle["test_chart_data"]

    # 6. Distribution Histogram (10 bins)
    hist_counts, bin_edges = np.histogram(df["energy_kwh"], bins=10)
    distribution_data = {
        "labels": [f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}" for i in range(len(hist_counts))],
        "counts": [int(c) for c in hist_counts],
    }

    return jsonify({
        "metrics": {
            "current_energy_kwh": current_kwh,
            "predicted_next_kwh": predicted_next_kwh,
            "average_daily_kwh": avg_daily_kwh,
            "estimated_daily_cost_inr": estimated_daily_cost_inr,
            "is_synthetic": active_path.name == DEFAULT_DEMO_PATH.name,
            "dataset_name": active_path.name,
            "total_records": len(df),
        },
        "hourly": hourly_data,
        "daily": daily_data,
        "weekly": weekly_data,
        "actual_vs_predicted": actual_vs_predicted,
        "distribution": distribution_data,
    })


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "File size exceeds the 25 MB upload limit."}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    # Ensure demo data and trained model exist before starting
    ensure_demo_dataset()
    if not predictor.is_ready:
        print("[WattWise] Initializing baseline model...")
        train_pipeline()
        predictor.load_model()

    port = int(os.environ.get("PORT", 5000))
    # Debug is False by default for security & production compatibility
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ["true", "1"]
    print(f"[WattWise] Server starting on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
