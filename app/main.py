from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rossmann_mlops.config import load_config
from rossmann_mlops.predict import PredictionInputError, Predictor


class PredictionRow(BaseModel):
    Store: int = Field(..., ge=1)
    DayOfWeek: int = Field(..., ge=1, le=7)
    Date: str
    Open: int = Field(..., ge=0, le=1)
    Promo: int = Field(..., ge=0, le=1)
    StateHoliday: str
    SchoolHoliday: int = Field(..., ge=0, le=1)


class PredictionRequest(BaseModel):
    records: list[PredictionRow]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.predictor = Predictor(
        model_path=config["paths"]["model_file"],
        store_data_path=config["paths"]["store_data"],
    )
    yield


app = FastAPI(title="Rossmann Sales Forecast API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "message": "Invalid request payload"})


@app.exception_handler(PredictionInputError)
async def prediction_input_exception_handler(_: Request, exc: PredictionInputError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc), "message": "Invalid prediction input"})


@app.exception_handler(ValueError)
async def value_error_exception_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc), "message": "Request could not be processed"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    records = [row.model_dump() for row in payload.records]
    predictions = app.state.predictor.predict(records)
    return {"predictions": predictions, "count": len(predictions)}
