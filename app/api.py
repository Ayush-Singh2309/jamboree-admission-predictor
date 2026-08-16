from fastapi import FastAPI, Depends
import pandas as pd

from app.schemas import (
    AdmissionRequest,
    AdmissionResponse
)

from app.dependencies import get_model
from src.model.inference import predict
import src.config


app = FastAPI(
    title="Jamboree Admission Predictor"
)

@app.get("/")
def home():
    return {
        "message": "Jamboree Admission Predictor API"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.get("/model-info")
def model_info():
    return {
        "name": src.config.MODEL_NAME,
        "version": "1.0.0",
    }

@app.post(
    "/predict",
    response_model=AdmissionResponse
)
def predict_admission(
    request: AdmissionRequest,
    model = Depends(get_model)
):

    df = pd.DataFrame([request.model_dump()])

    prediction = predict(model, df)

    return AdmissionResponse(
        chance_of_admit=float(round(prediction[0], 4))
    )