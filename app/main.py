from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import JSONResponse
import os

from starlette.middleware.cors import CORSMiddleware

from .config import load_config, setup_logging, validate_config
from .schemas import (
    GeminiHealthAnalysis,
    HealthProfile,
    NutritionResponse,
    UserInput,
)
from .services import (
    build_health_response,
    predict_macros,
    recommend_meal_greedy,
    user_input_to_profile,
)

app = FastAPI(title="health-ai-service", version="0.3.0")
log = logging.getLogger("health_ai")

# Allow cross-origin calls (frontend/server integration + Swagger).
# NOTE: Some browsers/proxies dislike allow_origins=["*"] in certain setups.
_cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if _cors_origins:
    allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
    allow_origin_regex = None
else:
    allow_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000",
    ]
    # Swagger is often opened via 0.0.0.0 or a different localhost port.
    # This regex keeps dev convenient without opening to arbitrary origins.
    allow_origin_regex = r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ResponseValidationError)
async def _response_validation_error_handler(
    request: Request, exc: ResponseValidationError
) -> JSONResponse:
    # When response_model validation fails, FastAPI raises after the endpoint returns.
    # Logging this makes 500s diagnosable in container logs.
    log.exception("ResponseValidationError on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "ResponseValidationError", "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.on_event("startup")
def _startup() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root=repo_root)
    setup_logging(cfg.log_level)
    validate_config(cfg, logger=log)


@app.get("/")
def root():
    return {"message": "AI Nutrition Service running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=NutritionResponse)
def predict(data: UserInput) -> NutritionResponse:
    profile = user_input_to_profile(data)
    t = predict_macros(profile)
    return NutritionResponse(
        calories=float(t.calories),
        protein=float(t.protein_g),
        carb=float(t.carbs_g),
        fat=float(t.fat_g),
    )


@app.post("/recommend", response_model=GeminiHealthAnalysis)
def recommend(profile: HealthProfile):
    """
    Full 7-field body (matches Nest HealthProfileDto + Next HealthProfile).
    Response matches GeminiHealthAnalysis / GeminiHealthAnalysisDto.
    """
    return build_health_response(profile)


@app.post("/demo/recommend")
def demo_recommend(data: UserInput):
    """Lab-style: 4 fields -> nutrition dict + greedy meal list."""
    profile = user_input_to_profile(data)
    t = predict_macros(profile)
    meals = recommend_meal_greedy(t.calories)
    return {
        "nutrition": {
            "calories": t.calories,
            "protein": t.protein_g,
            "carb": t.carbs_g,
            "fat": t.fat_g,
        },
        "meals": meals,
    }
