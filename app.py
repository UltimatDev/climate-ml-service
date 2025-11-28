from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

model = joblib.load("model.joblib")

app = FastAPI()

class ClimateInput(BaseModel):
    rainfall_mm: float
    temperature_c: float
    ndvi: float

@app.post("/predict")
def predict_yield(data: ClimateInput):
    x = [[data.rainfall_mm, data.temperature_c, data.ndvi]]
    pred = model.predict(x)[0]
    return {"predicted_yield": float(pred)}

@app.get("/")
def root():
    return {"status": "ML model up"}
