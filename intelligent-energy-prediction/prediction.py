"""
Inference and Prediction Engine for WattWise Smart Energy AI.
Loads the serialized model bundle, handles feature mapping, conducts validation,
and generates predictions with explainability breakdown.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from preprocessing import build_prediction_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_BUNDLE_PATH = BASE_DIR / "models" / "energy_model_bundle.joblib"
TARIFF_INR_PER_KWH = 7.50  # Commercial smart-building electricity tariff in INR (₹)


class EnergyPredictor:
    def __init__(self, bundle_path: str | Path = MODEL_BUNDLE_PATH):
        self.bundle_path = Path(bundle_path)
        self.bundle: Optional[Dict[str, Any]] = None
        self.load_model()

    def load_model(self) -> bool:
        """Loads or reloads the serialized model bundle from disk."""
        if self.bundle_path.exists():
            try:
                self.bundle = joblib.load(self.bundle_path)
                return True
            except Exception as e:
                print(f"[WattWise] Error loading model bundle: {e}")
                self.bundle = None
                return False
        self.bundle = None
        return False

    @property
    def is_ready(self) -> bool:
        return self.bundle is not None and "best_model" in self.bundle

    def get_info(self) -> Dict[str, Any]:
        if not self.is_ready:
            return {"status": "not_ready", "message": "ML model bundle has not been trained yet."}
        return {
            "status": "ready",
            "best_model_name": self.bundle["best_model_name"],
            "dataset_info": self.bundle.get("dataset_info", {}),
            "all_metrics": self.bundle.get("all_metrics", {}),
            "feature_importance": self.bundle.get("feature_importance", {}),
            "trained_at": self.bundle.get("trained_at", "Unknown"),
            "tariff_per_kwh_inr": TARIFF_INR_PER_KWH,
        }

    def predict(
        self,
        temperature: float,
        humidity: float,
        occupancy: int,
        hour: int,
        day_of_week: int,
        prev_energy_kwh: float,
    ) -> Dict[str, Any]:
        """
        Executes prediction using the active machine learning model.
        Returns predicted consumption (kWh), cost estimate (₹), and explainable breakdown.
        """
        if not self.is_ready:
            raise RuntimeError("Model is not loaded or trained. Please train the model first.")

        # Validate inputs
        try:
            temperature = float(temperature)
            humidity = float(humidity)
            occupancy = max(0, int(occupancy))
            hour = int(hour)
            day_of_week = int(day_of_week)
            prev_energy_kwh = float(prev_energy_kwh)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid input parameter format: {exc}")

        if not (0 <= hour <= 23):
            raise ValueError("Hour must be between 0 and 23.")
        if not (0 <= day_of_week <= 6):
            raise ValueError("Day of week must be between 0 (Monday) and 6 (Sunday).")
        if not (-20.0 <= temperature <= 60.0):
            raise ValueError("Temperature must be between -20°C and 60°C.")
        if not (0.0 <= humidity <= 100.0):
            raise ValueError("Humidity must be between 0% and 100%.")
        if prev_energy_kwh < 0:
            raise ValueError("Previous energy consumption must be non-negative.")

        # Construct feature vector
        features_df = build_prediction_features(
            temperature=temperature,
            humidity=humidity,
            occupancy=occupancy,
            hour=hour,
            day_of_week=day_of_week,
            prev_energy_kwh=prev_energy_kwh,
        )

        model = self.bundle["best_model"]
        scaler = self.bundle.get("scaler")
        best_model_name = self.bundle.get("best_model_name", "Random Forest Regressor")
        best_metrics = self.bundle.get("all_metrics", {}).get(best_model_name, {})

        # Apply scaler if linear model
        use_scaled = best_metrics.get("use_scaled", False)
        X_input = scaler.transform(features_df) if (use_scaled and scaler) else features_df

        raw_pred = model.predict(X_input)[0]
        # Physics guardrail: building power cannot be zero or negative
        predicted_kwh = round(max(5.0, float(raw_pred)), 2)

        # Delta comparisons
        diff_kwh = round(predicted_kwh - prev_energy_kwh, 2)
        pct_change = round((diff_kwh / prev_energy_kwh) * 100, 1) if prev_energy_kwh > 0 else 0.0

        # Cost calculation in INR (₹)
        estimated_cost_inr = round(predicted_kwh * TARIFF_INR_PER_KWH, 2)

        # Categorize load state
        if predicted_kwh < 35.0:
            load_category = "Off-Peak / Low Demand"
            load_class = "low"
        elif predicted_kwh < 75.0:
            load_category = "Normal Operational Demand"
            load_class = "normal"
        else:
            load_category = "High Peak Load"
            load_class = "peak"

        # Explainability & component decomposition
        cooling_deg = max(0.0, temperature - 22.0)
        heating_deg = max(0.0, 18.0 - temperature)
        estimated_hvac = round((cooling_deg * 2.2) + (heating_deg * 1.5) + (occupancy * 0.08), 2)
        estimated_plug = round(occupancy * 0.16, 2)
        base_load_est = round(max(15.0, predicted_kwh - estimated_hvac - estimated_plug), 2)

        days_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        return {
            "predicted_kwh": predicted_kwh,
            "previous_kwh": prev_energy_kwh,
            "difference_kwh": diff_kwh,
            "percentage_change": pct_change,
            "estimated_cost_inr": estimated_cost_inr,
            "load_category": load_category,
            "load_class": load_class,
            "selected_model": best_model_name,
            "model_metrics": {
                "mae": best_metrics.get("mae", 0.0),
                "rmse": best_metrics.get("rmse", 0.0),
                "r2_score": best_metrics.get("r2_score", 0.0),
            },
            "breakdown": {
                "base_building_load": base_load_est,
                "hvac_thermal_load": estimated_hvac,
                "occupancy_plug_load": estimated_plug,
            },
            "input_summary": {
                "temperature": temperature,
                "humidity": humidity,
                "occupancy": occupancy,
                "hour": hour,
                "day_name": days_map[day_of_week],
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
