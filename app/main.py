from __future__ import annotations

from fastapi import FastAPI

from .schemas import (
    GeminiHealthAnalysis,
    HealthProfile,
    NutritionResponse,
    UserInput,
)
from .services import (
    build_gemini_style_response,
    predict_macros,
    recommend_meal_greedy,
    user_input_to_profile,
)

app = FastAPI(title="health-ai-service", version="0.3.0")


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
    return build_gemini_style_response(profile)


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
