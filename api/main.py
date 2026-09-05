from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Ride Demand Prediction API")

model = joblib.load("../model/demand_model.pkl")

class DemandRequest(BaseModel):
    zone_id: int
    hour: int
    day_of_week: int
    lag_1: float
    lag_2: float
    rolling_mean_4: float

@app.get("/")
def root():
    return {"status": "Ride Demand Prediction API is running"}

@app.post("/predict")
def predict_demand(request: DemandRequest):
    features = pd.DataFrame([{
        "zone_id": request.zone_id,
        "hour": request.hour,
        "day_of_week": request.day_of_week,
        "is_weekend": 1 if request.day_of_week >= 5 else 0,
        "lag_1": request.lag_1,
        "lag_2": request.lag_2,
        "rolling_mean_4": request.rolling_mean_4,
    }])
    prediction = model.predict(features)[0]
    return {"predicted_trip_count": float(prediction)}
