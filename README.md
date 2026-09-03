# ⛰️ NER Railway Landslide AI (SIH26001)

An IoT-driven, predictive machine learning pipeline designed to protect railway infrastructure in India's North Eastern Region (NER) from rainfall-induced landslides. 

While traditional disaster management systems rely entirely on static historical data or delayed satellite imagery, this system introduces a **hybrid data-fusion approach**. By combining real-time edge hardware (soil moisture and vibration) with antecedent rainfall forecasts (via Open-Meteo), our ML engine dynamically calculates the exact saturation threshold where slope failure occurs.

### 🏗️ System Architecture
This monorepo contains a 4-part microservices architecture, built for resilience in low-connectivity zones:
1. **IoT Ingestion Gateway:** Catches live telemetry from edge hardware and polls external weather/soil APIs.
2. **Data Engineering Pipeline:** Aligns irregular time-series data and calculates 1-hour moisture deltas.
3. **ML Prediction Engine:** An XGBoost model that evaluates the fused data to generate a real-time Risk Score.
4. **GIS Dashboard & Alerting:** A geospatial interface that visualizes track safety and automatically dispatches SMS warnings to railway authorities.
