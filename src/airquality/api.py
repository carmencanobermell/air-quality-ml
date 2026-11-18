"""HTTP API for the trained model.

    uvicorn airquality.api:app --port 8000
    curl -X POST localhost:8000/predict -H 'content-type: application/json' \
         -d '{"country":"ES","pollutant":"NO2","lat":40.4,"lon":-3.7}'

The model path comes from the MODEL_PATH environment variable (default models/model.joblib).
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import model as model_io


class Reading(BaseModel):
    country: str = Field(description="ISO country code, e.g. ES", examples=["ES"])
    pollutant: str = Field(description="e.g. PM2.5, PM10, NO2, O3, CO, SO2", examples=["NO2"])
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Prediction(BaseModel):
    high_pollution: bool
    probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = model_io.load(os.environ.get("MODEL_PATH", model_io.DEFAULT_MODEL))
    yield


app = FastAPI(title="Air quality: high pollution classifier", lifespan=lifespan)


@app.get("/health")
def health():
    b = app.state.bundle
    return {"status": "ok", "n_train": b["n_train"], "sklearn_version": b["sklearn_version"]}


@app.post("/predict", response_model=Prediction)
def predict(reading: Reading):
    return model_io.predict(app.state.bundle, [reading.model_dump()])[0]
