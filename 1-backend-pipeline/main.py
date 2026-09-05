import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Target Location: Lumding-Badarpur Railway Hill Section, Assam (NER)
LAT = 25.15
LON = 93.15

class AlertTranslationRequest(BaseModel):
    level: str
    location: str

@app.post("/api/translate-alert")
def translate_alert(request: AlertTranslationRequest):
    """Translate the local warning with an LLM when configured, otherwise keep alerts usable offline."""
    english = (
        f"Landslide risk is {request.level.lower()} near {request.location}. "
        "Please stay away from the railway slope and follow official instructions."
    )
    is_critical = request.level.upper() == "CRITICAL"
    fallback = {
        "assamese": (
            "লুমডিং-বদৰপুৰ ৰে'লৱে ছেকচন, অসমত ভূমিস্খলনৰ গুৰুতৰ আশংকা আছে। "
            "ৰেলপথৰ ঢালৰ পৰা আঁতৰি থাকক আৰু চৰকাৰী নিৰ্দেশনা মানক।"
            if is_critical else
            "লুমডিং-বদৰপুৰ ৰে'লৱে ছেকচন, অসমত ভূমিস্খলনৰ আশংকা আছে। "
            "ৰেলপথৰ ঢালৰ পৰা আঁতৰি থাকক আৰু চৰকাৰী নিৰ্দেশনা মানক।"
        ),
        "bengali": (
            "আসামের লুমডিং-বদরপুর রেলওয়ে সেকশনে গুরুতর ভূমিধসের ঝুঁকি রয়েছে। "
            "রেলপথের ঢাল থেকে দূরে থাকুন এবং সরকারি নির্দেশনা মেনে চলুন।"
            if is_critical else
            "আসামের লুমডিং-বদরপুর রেলওয়ে সেকশনে ভূমিধসের ঝুঁকি রয়েছে। "
            "রেলপথের ঢাল থেকে দূরে থাকুন এবং সরকারি নির্দেশনা মেনে চলুন।"
        ),
        "hindi": (
            "असम के लुमडिंग-बदरपुर रेलवे सेक्शन में भूस्खलन का गंभीर खतरा है। "
            "रेलवे ढलान से दूर रहें और आधिकारिक निर्देशों का पालन करें।"
            if is_critical else
            "असम के लुमडिंग-बदरपुर रेलवे सेक्शन में भूस्खलन का खतरा है। "
            "रेलवे ढलान से दूर रहें और आधिकारिक निर्देशों का पालन करें।"
        ),
    }
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "fallback", "translations": fallback}

    prompt = (
        "Translate this public safety alert into Assamese, Bengali, and Hindi. "
        "You must preserve the exact location meaning in every translation; do not omit the location. "
        "If the alert is CRITICAL, explicitly preserve its critical and immediate urgency in every language; do not soften it to a generic risk or danger. "
        "Return only a JSON object with keys assamese, bengali, hindi. Alert: " + english
    )
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini"), "temperature": 0.2,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=15,
        )
        response.raise_for_status()
        translations = response.json()["choices"][0]["message"]["content"]
        import json
        return {"status": "success", "translations": json.loads(translations)}
    except (requests.RequestException, KeyError, ValueError):
        return {"status": "fallback", "translations": fallback}

@app.get("/")
def read_root():
    return {"message": "NER Railway Landslide API is running. Access /api/real-data-pipeline for data."}

@app.get("/api/real-data-pipeline")
def get_pipeline_data():
    """
    Fetches real environmental data, cleans it, and returns a fused DataFrame.
    """
    # 1. Fetch real satellite and radar data for the past 7 days
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=precipitation,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,temperature_2m"
        f"&past_days=7"
    )
    
    response = requests.get(url)
    raw_json = response.json()
    
    # 2. Extract into Pandas DataFrame
    df = pd.DataFrame(raw_json['hourly'])
    
    # 3. Time-Series Alignment
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # 4. Feature Engineering (The Pipeline Magic)
    # Antecedent Rainfall: Total rain in the previous 24 hours
    df['rolling_24h_rain'] = df['precipitation'].rolling(window=24).sum()
    
    # Moisture Delta: How much did the topsoil moisture change in the last 3 hours?
    df['moisture_delta_3h'] = df['soil_moisture_0_to_7cm'] - df['soil_moisture_0_to_7cm'].shift(3)
    
    # 5. Clean up NaN values caused by the rolling calculations
    df.dropna(inplace=True)
    
    # Return the fully engineered data as a clean JSON dictionary
    return {
        "status": "success",
        "location": "Lumding-Badarpur, Assam",
        "total_rows": len(df),
        "data": df.to_dict(orient="index")
    }

@app.get("/api/historical-data")
def get_historical_data(start_date: str = "2020-01-01", end_date: str = "2023-12-31"):
    """
    Fetches historical environmental data for ML training.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=precipitation,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        return {"status": "error", "message": "Failed to fetch data from Open-Meteo"}
        
    raw_json = response.json()
    
    # Extract into Pandas DataFrame
    df = pd.DataFrame(raw_json['hourly'])
    
    # Time-Series Alignment
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Feature Engineering
    df['rolling_24h_rain'] = df['precipitation'].rolling(window=24).sum()
    df['moisture_delta_3h'] = df['soil_moisture_0_to_7cm'] - df['soil_moisture_0_to_7cm'].shift(3)
    
    # Clean up NaN values
    df.dropna(inplace=True)
    
    return {
        "status": "success",
        "location": "Lumding-Badarpur, Assam",
        "start_date": start_date,
        "end_date": end_date,
        "total_rows": len(df),
        # For historical data, returning it as a list of records might be safer for large datasets, 
        # but to keep it consistent with the real-time API we'll use orient="index"
        "data": df.to_dict(orient="index")
    }
