"""
Automated Test Suite for WattWise Smart Energy AI.
Validates all Flask REST API endpoints, model predictions, file uploads,
and analytics telemetry using standard Python unittest.
"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from app import app, predictor

class WattWiseApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("WATTWISE", html)
        self.assertIn("Intelligent Energy Consumption", html)
        self.assertIn("chart.js", html.lower())

    def test_02_api_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_ready"])
        self.assertIn("tariff_per_kwh_inr", data)

    def test_03_api_metrics(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("best_model_name", data)
        self.assertIn("all_metrics", data)
        self.assertIn("Linear Regression", data["all_metrics"])
        self.assertIn("Random Forest Regressor", data["all_metrics"])
        self.assertIn("Gradient Boosting Regressor", data["all_metrics"])

        rf_metrics = data["all_metrics"]["Random Forest Regressor"]
        self.assertGreater(rf_metrics["mae"], 0)
        self.assertGreater(rf_metrics["rmse"], 0)
        self.assertIn("r2_score", rf_metrics)

    def test_04_api_analytics(self):
        response = self.client.get("/api/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("metrics", data)
        self.assertIn("current_energy_kwh", data["metrics"])
        self.assertIn("predicted_next_kwh", data["metrics"])
        self.assertIn("average_daily_kwh", data["metrics"])
        self.assertIn("estimated_daily_cost_inr", data["metrics"])
        self.assertIn("hourly", data)
        self.assertGreater(len(data["hourly"]["values"]), 0)
        self.assertIn("daily", data)
        self.assertIn("weekly", data)
        self.assertIn("actual_vs_predicted", data)
        self.assertIn("distribution", data)

    def test_05_api_model_info(self):
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ready")
        self.assertIn("best_model_name", data)

    def test_06_api_predict_success(self):
        payload = {
            "temperature": 30.5,
            "humidity": 65.0,
            "occupancy": 140,
            "hour": 14,
            "day_of_week": 2,
            "prev_energy_kwh": 62.4,
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("predicted_kwh", data)
        self.assertGreater(data["predicted_kwh"], 10.0)
        self.assertIn("estimated_cost_inr", data)
        self.assertIn("load_category", data)
        self.assertIn("breakdown", data)
        self.assertIn("base_building_load", data["breakdown"])
        self.assertIn("hvac_thermal_load", data["breakdown"])
        self.assertIn("occupancy_plug_load", data["breakdown"])

    def test_07_api_predict_validation_errors(self):
        # Missing required field
        response = self.client.post("/api/predict", json={"temperature": 25.0})
        self.assertEqual(response.status_code, 400)

        # Invalid hour
        payload = {
            "temperature": 25.0,
            "humidity": 50.0,
            "occupancy": 50,
            "hour": 29,  # invalid!
            "day_of_week": 1,
            "prev_energy_kwh": 40.0,
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_08_api_upload_and_train_and_reset(self):
        # 1. Create a mini valid CSV with 60 hourly rows
        from datetime import datetime, timedelta
        base = datetime(2026, 1, 1, 0, 0, 0)
        csv_content = "timestamp,energy_kwh,temperature,humidity,occupancy\n"
        for i in range(60):
            dt = base + timedelta(hours=i)
            csv_content += f"{dt.strftime('%Y-%m-%d %H:%M:%S')},{50.0 + (i % 15)},{25.0 + (i % 5)},55.0,{20 + i}\n"

        data = {
            "file": (io.BytesIO(csv_content.encode("utf-8")), "test_telemetry.csv")
        }
        upload_res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(upload_res.status_code, 200)
        upload_json = upload_res.get_json()
        self.assertEqual(upload_json["filename"], "test_telemetry.csv")
        self.assertEqual(upload_json["summary"]["total_records"], 60)
        self.assertEqual(len(upload_json["preview"]), 5)

        # 2. Retrain models on this uploaded data
        train_res = self.client.post("/api/train")
        self.assertEqual(train_res.status_code, 200)
        train_json = train_res.get_json()
        self.assertIn("best_model_name", train_json)
        self.assertIn("all_metrics", train_json)

        # 3. Reset back to demo dataset
        reset_res = self.client.post("/api/reset-demo")
        self.assertEqual(reset_res.status_code, 200)
        reset_json = reset_res.get_json()
        self.assertTrue(reset_json["dataset_info"]["is_synthetic"])

if __name__ == "__main__":
    unittest.main()
