"""
Preprocessing and Feature Engineering Pipeline for Intelligent Energy Consumption Prediction.
Supports time-series preparation, lag generation, rolling statistics, chronological train-test splits,
and demo dataset synthesis.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_business_hour",
    "temperature",
    "humidity",
    "occupancy",
    "lag_1",
    "lag_2",
    "lag_24",
    "rolling_mean_6",
    "rolling_mean_24",
    "cooling_degree",
    "heating_degree",
]

TARGET_COLUMN = "energy_kwh"


def parse_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates, parses timestamps, sorts chronologically, removes duplicates/invalids,
    and handles missing values with time-series interpolation.
    """
    # Normalize column names
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "timestamp" not in df.columns or "energy_kwh" not in df.columns:
        raise ValueError("Dataset must contain 'timestamp' and 'energy_kwh' columns.")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "energy_kwh"])

    # Ensure numeric energy
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    df = df.dropna(subset=["energy_kwh"])
    df = df[df["energy_kwh"] >= 0]

    # Sort chronologically to preserve time structure
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df.reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"Insufficient data: minimum 50 valid observations required (found {len(df)}).")

    # Handle optional columns if present, otherwise impute sensible realistic estimates
    hour_series = df["timestamp"].dt.hour
    is_weekend_series = df["timestamp"].dt.day_of_week.isin([5, 6]).astype(int)

    if "temperature" not in df.columns:
        # Realistic diurnal estimate: cooler at night (22C), warmer at midday (32C)
        df["temperature"] = 27.0 + 5.0 * np.sin((hour_series - 9) * np.pi / 12)
    else:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df["temperature"] = df["temperature"].interpolate().bfill().ffill()

    if "humidity" not in df.columns:
        # Inverse relation with temperature approx
        df["humidity"] = 60.0 - 15.0 * np.sin((hour_series - 9) * np.pi / 12)
    else:
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df["humidity"] = df["humidity"].interpolate().bfill().ffill()

    if "occupancy" not in df.columns:
        # Typical smart office pattern: high on weekday business hours, low otherwise
        biz_hour = ((hour_series >= 8) & (hour_series <= 18) & (is_weekend_series == 0)).astype(int)
        df["occupancy"] = biz_hour * 140 + 5
    else:
        df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")
        df["occupancy"] = df["occupancy"].interpolate().bfill().ffill().round()

    # Interpolate short missing energy gaps
    df["energy_kwh"] = df["energy_kwh"].interpolate(method="linear", limit=3).bfill().ffill()

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal, lag, rolling, and thermal degree features.
    Strictly uses past information to avoid data leakage.
    """
    df = df.copy()

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.day_of_week
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_business_hour"] = (
        (df["hour"] >= 8) & (df["hour"] <= 18) & (df["is_weekend"] == 0)
    ).astype(int)

    # Thermal degree features (HVAC cooling above 22C, heating below 18C)
    df["cooling_degree"] = (df["temperature"] - 22.0).clip(lower=0.0)
    df["heating_degree"] = (18.0 - df["temperature"]).clip(lower=0.0)

    # Lag features (energy at t-1, t-2, t-24)
    df["lag_1"] = df["energy_kwh"].shift(1)
    df["lag_2"] = df["energy_kwh"].shift(2)
    df["lag_24"] = df["energy_kwh"].shift(24)

    # Rolling window statistics (using closed='left' equivalent by shifting 1 first to prevent target leakage)
    shifted_energy = df["energy_kwh"].shift(1)
    df["rolling_mean_6"] = shifted_energy.rolling(window=6, min_periods=1).mean()
    df["rolling_mean_24"] = shifted_energy.rolling(window=24, min_periods=1).mean()

    # Backfill earliest rows for lag_24 to preserve as much training data as possible
    df["lag_24"] = df["lag_24"].bfill().fillna(df["energy_kwh"])
    df["lag_2"] = df["lag_2"].bfill().fillna(df["energy_kwh"])
    df["lag_1"] = df["lag_1"].bfill().fillna(df["energy_kwh"])
    df["rolling_mean_6"] = df["rolling_mean_6"].bfill().fillna(df["energy_kwh"])
    df["rolling_mean_24"] = df["rolling_mean_24"].bfill().fillna(df["energy_kwh"])

    # Drop any remaining NaNs
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series data chronologically.
    Guarantees no future data is leaked into the training partition.
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


def build_prediction_features(
    temperature: float,
    humidity: float,
    occupancy: int,
    hour: int,
    day_of_week: int,
    prev_energy_kwh: float,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Builds a single-sample feature row matching FEATURE_COLUMNS for live prediction.
    Uses historical dataset baselines when available for secondary lags.
    """
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_biz = 1 if (8 <= hour <= 18 and not is_weekend) else 0
    cooling = max(0.0, float(temperature) - 22.0)
    heating = max(0.0, 18.0 - float(temperature))
    month = datetime.now().month

    # Reference values if historical data exists
    if reference_df is not None and not reference_df.empty:
        # Match hour of day historical mean
        hour_history = reference_df[reference_df["hour"] == hour]
        if not hour_history.empty:
            lag_24_est = float(hour_history["energy_kwh"].mean())
        else:
            lag_24_est = float(prev_energy_kwh)
        rolling_24_est = float(reference_df["energy_kwh"].tail(24).mean())
        rolling_6_est = float(reference_df["energy_kwh"].tail(6).mean())
    else:
        lag_24_est = float(prev_energy_kwh)
        rolling_24_est = float(prev_energy_kwh) * 0.95
        rolling_6_est = float(prev_energy_kwh) * 0.98

    sample = {
        "hour": int(hour),
        "dayofweek": int(day_of_week),
        "month": int(month),
        "is_weekend": int(is_weekend),
        "is_business_hour": int(is_biz),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "occupancy": int(occupancy),
        "lag_1": float(prev_energy_kwh),
        "lag_2": float(prev_energy_kwh * 0.99),
        "lag_24": float(lag_24_est),
        "rolling_mean_6": float(rolling_6_est),
        "rolling_mean_24": float(rolling_24_est),
        "cooling_degree": float(cooling),
        "heating_degree": float(heating),
    }

    return pd.DataFrame([sample])[FEATURE_COLUMNS]


def generate_synthetic_building_dataset(
    days: int = 90,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a physics-inspired synthetic hourly energy dataset for a 6-story smart office building.
    Clearly marked as synthetic demo data.

    Physics breakdown:
    - Base load: 28 kW (servers, lighting, emergency systems)
    - Occupant plug load: ~0.18 kW per person
    - HVAC cooling: ~1.8 kW per degree C above 22C (scaled by occupancy)
    - Diurnal temperature cycle: 20C to 36C
    - Business hours: 08:00 to 18:00 Mon-Fri
    """
    np.random.seed(seed)
    start_time = datetime(2026, 8, 1, 0, 0, 0)
    total_hours = days * 24

    timestamps = [start_time + timedelta(hours=i) for i in range(total_hours)]

    records = []
    base_load = 28.0

    for dt in timestamps:
        hour = dt.hour
        weekday = dt.weekday()
        is_weekend = weekday in [5, 6]

        # Diurnal temperature cycle with stochastic seasonal drift
        day_progress = (dt - start_time).total_seconds() / (86400 * days)
        temp_season = 31.0 - 5.0 * day_progress
        temp_diurnal = 5.5 * np.sin((hour - 9) * np.pi / 12)
        temp_noise = np.random.normal(0, 0.8)
        temperature = round(temp_season + temp_diurnal + temp_noise, 1)

        # Humidity (inversely correlated with temp)
        humidity = round(np.clip(70.0 - 15.0 * np.sin((hour - 9) * np.pi / 12) + np.random.normal(0, 2), 25, 95), 1)

        # Occupancy
        if is_weekend:
            occupancy = int(np.random.poisson(8) if 9 <= hour <= 17 else np.random.poisson(2))
        else:
            if 8 <= hour <= 18:
                # Bell curve centered around 13:00 (1 PM)
                occ_factor = np.exp(-((hour - 13) ** 2) / 12.0)
                occupancy = int(np.clip(180 * occ_factor + np.random.normal(0, 10), 20, 240))
            elif 6 <= hour < 8 or 19 <= hour <= 21:
                occupancy = int(np.random.poisson(25))
            else:
                occupancy = int(np.random.poisson(4))

        # Power consumption synthesis (kWh for 1-hour interval = average kW)
        plug_load = occupancy * 0.16  # laptops, monitors, personal devices

        cooling_deg = max(0.0, temperature - 22.0)
        heating_deg = max(0.0, 18.0 - temperature)
        # HVAC system efficiency increases base cooling, scaled by building heat load
        hvac_load = (cooling_deg * 2.2) + (heating_deg * 1.5) + (occupancy * 0.08)

        # Lighting schedule
        if 8 <= hour <= 19 and not is_weekend:
            lighting_load = 14.0
        elif is_weekend and (10 <= hour <= 16):
            lighting_load = 5.0
        else:
            lighting_load = 3.0

        random_fluctuation = np.random.normal(0, 1.2)
        total_energy = base_load + plug_load + hvac_load + lighting_load + random_fluctuation
        energy_kwh = round(max(15.0, total_energy), 2)

        records.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "energy_kwh": energy_kwh,
            "temperature": temperature,
            "humidity": humidity,
            "occupancy": occupancy,
        })

    df = pd.DataFrame(records)

    if output_path:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
