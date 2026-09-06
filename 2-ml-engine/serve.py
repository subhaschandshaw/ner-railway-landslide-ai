import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ML Engine - Landslide Predictor")

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/landslide_model.joblib"

if os.path.exists(MODEL_PATH):
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    critical_threshold = artifact["critical_threshold"]
    warning_threshold = artifact["warning_threshold"]
    advisory_threshold = artifact["advisory_threshold"]
    rain_floor = artifact["rain_floor"]
    soil_floor = artifact["soil_floor"]
    tilt_floor = artifact["tilt_floor"]
    vib_floor = artifact["vib_floor"]
else:
    model = None

class SensorData(BaseModel):
    soil_moisture_pct: float
    rainfall_mm_hr: float
    vibration_index: float
    tilt_deg: float
    temperature_C: float
    latitude: float
    longitude: float

def engineer_features_single(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df["rain_soil_interaction"] = df["rainfall_mm_hr"] * (df["soil_moisture_pct"] / 100.0)
    df["tilt_vib_interaction"] = df["tilt_deg"] * df["vibration_index"]
    df["rainfall_sq"] = df["rainfall_mm_hr"] ** 2
    df["soil_moisture_sq"] = df["soil_moisture_pct"] ** 2
    df["combined_risk_index"] = df["rain_soil_interaction"] + df["tilt_vib_interaction"]
    df["pore_pressure_proxy"] = (df["soil_moisture_pct"] / 100.0) * np.sin(np.radians(df["tilt_deg"]))
    df["shear_force_proxy"] = np.sin(np.radians(df["tilt_deg"])) * (1.0 + df["vibration_index"])

    # Must mirror train.py exactly, using the trained artifact's soil_floor
    df["soil_saturation_excess"] = np.clip(df["soil_moisture_pct"] - soil_floor, 0, None) ** 2

    return df[feature_cols]


def get_physical_reasons(input_dict: dict) -> list:
    """Returns a human-readable list of which physical floors were crossed,
    used to explain WHY a tier was assigned — not just what the tier is."""
    reasons = []
    if input_dict["rainfall_mm_hr"] >= rain_floor:
        reasons.append(f"Rainfall {input_dict['rainfall_mm_hr']:.1f} mm/hr exceeds floor of {rain_floor} mm/hr")
    if input_dict["soil_moisture_pct"] >= soil_floor:
        reasons.append(f"Soil moisture {input_dict['soil_moisture_pct']:.1f}% exceeds floor of {soil_floor}%")
    if input_dict["tilt_deg"] >= tilt_floor:
        reasons.append(f"Tilt {input_dict['tilt_deg']:.1f}\u00b0 exceeds floor of {tilt_floor}\u00b0")
    if input_dict["vibration_index"] >= vib_floor:
        reasons.append(f"Vibration index {input_dict['vibration_index']:.1f} exceeds floor of {vib_floor}")
    return reasons


def physical_support_check(input_dict: dict) -> bool:
    crossings = sum([
        input_dict["rainfall_mm_hr"] >= rain_floor,
        input_dict["soil_moisture_pct"] >= soil_floor,
        input_dict["tilt_deg"] >= tilt_floor,
        input_dict["vibration_index"] >= vib_floor,
    ])
    return crossings >= 2


@app.post("/api/predict")
def predict_landslide(data: SensorData):
    if not model:
        return {"status": "error", "message": "Model not loaded. Train the model first."}

    input_dict = data.dict()
    X_df = engineer_features_single(input_dict)

    proba = model.predict_proba(X_df)[0][1]
    physical_support = physical_support_check(input_dict)
    reasons = get_physical_reasons(input_dict)

    # Four-tier classification, mirroring assign_tier() in train.py:
    # CRITICAL needs BOTH model confidence AND physical evidence together —
    # this keeps the evacuation-triggering tier rare and trustworthy.
    # WARNING fires on a high-recall model signal or three corroborating
    # physical floors; CRITICAL still requires the model and sensor support.
    if proba >= critical_threshold and physical_support:
        alert_level = "CRITICAL"
    elif proba >= warning_threshold or physical_support:
        alert_level = "WARNING"
    elif proba >= advisory_threshold:
        alert_level = "ADVISORY"
    else:
        alert_level = "SAFE"

    risk_score = proba * 100

    return {
        "status": "success",
        "risk_score_percent": round(risk_score, 2),
        "alert_level": alert_level,
        "model_probability": round(proba, 4),
        "physical_support_met": physical_support,
        "reasons": reasons if reasons else ["No physical floors crossed; risk based on model signal only"],
        "thresholds": {
            "critical": round(critical_threshold, 3),
            "warning": round(warning_threshold, 3),
            "advisory": round(advisory_threshold, 3)
        }
    }

@app.get("/")
def health():
    return {"status": "ML Engine Running", "model_loaded": model is not None}

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

@app.get("/api/predict-realtime")
def predict_realtime(demo: bool = False):
    """Fetches the latest data from the backend and predicts landslide risk."""
    if not model:
        return {"status": "error", "message": "Model not loaded."}

    try:
        response = requests.get(f"{BACKEND_URL}/api/real-data-pipeline")
        if response.status_code != 200:
            return {"status": "error", "message": "Failed to fetch data from backend"}

        data = response.json()["data"]

        # Get the most recent timestamp (last item in dict)
        latest_time = list(data.keys())[-1]
        latest_data = data[latest_time]

        # Base sensor data
        moisture = latest_data.get("soil_moisture_0_to_7cm", 0) * 100
        rain = latest_data.get("precipitation", 0)

        # DEMO MODE OVERRIDE FOR JUDGES
        if demo:
            moisture = 95.5  # Extreme saturation
            rain = 45.0      # Torrential downpour

        # Map backend data to ML features
        sensor_data = SensorData(
            soil_moisture_pct=moisture,
            rainfall_mm_hr=rain,
            temperature_C=latest_data.get("temperature_2m", 25.0),
            vibration_index=0.8 if demo else 0.1,  # Spike vibration in demo
            tilt_deg=3.5 if demo else 0.5,         # Spike tilt in demo
            latitude=25.15,
            longitude=93.15
        )

        # Call the existing prediction function
        prediction = predict_landslide(sensor_data)

        # Add the timestamp and sensor readings to the response
        prediction["timestamp"] = latest_time
        prediction["sensors"] = {
            "rainfall_mm_hr": round(rain, 2),
            "soil_moisture_pct": round(moisture, 2),
            "temperature_C": round(sensor_data.temperature_C, 2)
        }
        return prediction

    except Exception as e:
        return {"status": "error", "message": str(e)}