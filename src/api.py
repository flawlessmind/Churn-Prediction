"""FastAPI-сервис предсказания оттока.   uvicorn src.api:app --port 8000"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import structlog
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src import config as cfg
from src.predict import ChurnPredictor
from src.schemas import CustomerFeatures, Prediction

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(os.getenv("LOG_LEVEL", "INFO"))),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger("churn_api")

MAX_BATCH = 10_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Модель грузится ОДИН раз при старте, а не на каждый запрос."""
    model_path = Path(os.getenv("MODEL_PATH", str(cfg.MODEL_PATH)))
    app.state.predictor = None
    try:
        app.state.predictor = ChurnPredictor.load(model_path)
        log.info("model_loaded", path=str(model_path), version=app.state.predictor.metadata["model_version"])
    except Exception as exc:  # сервис поднимается, но /predict отдаёт 503 с понятным сообщением
        log.error("model_load_failed", path=str(model_path), error=str(exc))
    yield


app = FastAPI(title="Telco Churn API", version=cfg.MODEL_VERSION, lifespan=lifespan,
              description="Вероятность ухода клиента и сегмент риска для отдела удержания.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # непредвиденная ошибка -> 500 с понятным сообщением (детали только в логе)
        log.exception("unhandled_error", method=request.method, path=request.url.path)
        return JSONResponse(status_code=500, content={
            "detail": "Внутренняя ошибка сервиса при обработке запроса.", "request_id": request_id,
            "error_type": type(exc).__name__})
    log.info("request", method=request.method, path=request.url.path, status=response.status_code,
             duration_ms=round((time.perf_counter() - start) * 1000, 2))
    response.headers["X-Request-ID"] = request_id
    return response


def _predictor(request: Request) -> ChurnPredictor:
    p = getattr(request.app.state, "predictor", None)
    if p is None:
        raise HTTPException(503, "Модель не загружена. Обучите её (python -m src.train) и перезапустите сервис.")
    return p


def _to_frame(items: list[CustomerFeatures]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in items])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model_info")
def model_info(request: Request) -> dict:
    return _predictor(request).info()


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
def predict(customer: CustomerFeatures, request: Request):
    res = _predictor(request).predict_frame(_to_frame([customer]))[0]
    return {"customerID": customer.customerID, **res}


@app.post("/predict_batch", response_model=list[Prediction], response_model_exclude_none=True)
def predict_batch(request: Request, customers: list[CustomerFeatures] = Body(..., min_length=1, max_length=MAX_BATCH)):
    results = _predictor(request).predict_frame(_to_frame(customers))
    return [{"customerID": c.customerID, **r} for c, r in zip(customers, results)]
