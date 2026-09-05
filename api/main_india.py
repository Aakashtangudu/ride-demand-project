from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Indore Ride Demand Prediction API")

model = joblib.load("../model/demand_model_india.pkl")
zone_encoder = joblib.load("../model/zone_encoder_india.pkl")

class DemandRequest(BaseModel):
    zone_name: str
    hour: int
    day_of_week: int
    lag_1: float
    lag_2: float
    rolling_mean_4: float

@app.get("/")
def root():
    return {"status": "Indore Ride Demand Prediction API is running"}

@app.get("/zones")
def list_zones():
    return {"zones": sorted(zone_encoder.classes_.tolist())}

@app.post("/predict")
def predict_demand(request: DemandRequest):
    if request.zone_name not in zone_encoder.classes_:
        raise HTTPException(status_code=400, detail=f"Unknown zone: {request.zone_name}")

    zone_id = int(zone_encoder.transform([request.zone_name])[0])

    features = pd.DataFrame([{
        "zone_id": zone_id,
        "hour": request.hour,
        "day_of_week": request.day_of_week,
        "lag_1": request.lag_1,
        "lag_2": request.lag_2,
        "rolling_mean_4": request.rolling_mean_4,
    }])
    prediction = model.predict(features)[0]
    return {"zone_name": request.zone_name, "predicted_booking_count": float(prediction)}
