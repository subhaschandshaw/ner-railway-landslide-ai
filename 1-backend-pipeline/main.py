from fastapi import FastAPI
import requests
import pandas as pd

app = FastAPI()

# Target Location: Lumding-Badarpur Railway Hill Section, Assam (NER)
LAT = 25.15
LON = 93.15

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
        f"&hourly=precipitation,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm"
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