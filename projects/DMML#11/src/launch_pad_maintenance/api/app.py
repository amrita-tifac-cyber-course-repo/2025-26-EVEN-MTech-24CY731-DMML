from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_system

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API Running"}

@app.post("/predict")
def predict(data: InputData):
    prob, decision = predict_system(data.features)

    return {
        "failure_probability": float(prob),
        "recommended_action": decision
    }