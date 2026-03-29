"""
Core logic: load ML model, predict daily calories/macros, meal suggestions (greedy + weekly split).

Default paths (match repo layout):
  model/model.pkl
  data/food.csv

Request/response shapes mirror:
  food-delivery-app:  gemini-health-ai.service.ts (GeminiHealthAnalysis)
  food-delivery-server: healthGemini.service.ts (GeminiHealthAnalysisDto)

POST /recommend body (7 fields): age, gender, height, weight, activityLevel, healthGoal, dietaryRestrictions
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from .schemas import HealthProfile, UserInput

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "model" / "model.pkl"
DEFAULT_FOOD_CSV = ROOT / "data" / "food.csv"

ACTIVITY_ORDINAL_TO_LEVEL: Sequence[str] = (
    "sedentary",
    "light",
    "moderate",
    "active",
    "very-active",
)


def user_input_to_profile(data: UserInput) -> HealthProfile:
    """Map minimal 4-field demo input to HealthProfile (shared 6-feature pipeline)."""
    idx = int(round(max(0.0, min(4.0, float(data.activity)))))
    return HealthProfile(
        age=data.age,
        gender="male",
        height=data.height,
        weight=data.weight,
        activityLevel=ACTIVITY_ORDINAL_TO_LEVEL[idx],
        healthGoal="maintenance",
        dietaryRestrictions="",
    )


def _model_path() -> Path:
    p = os.getenv("MODEL_PATH", "").strip()
    return Path(p) if p else DEFAULT_MODEL_PATH


def _food_csv_path() -> Path:
    p = os.getenv("FOODS_CSV_PATH", "").strip()
    return Path(p) if p else DEFAULT_FOOD_CSV


# --- BMI / BMR / TDEE (baseline when no model; BMR/TDEE always computed for transparency) ---


def _bmi(height_cm: float, weight_kg: float) -> float:
    h = height_cm / 100.0
    return float(weight_kg / (h * h))


def _bmi_category(v: float) -> str:
    if v < 18.5:
        return "Underweight"
    if v < 25:
        return "Normal weight"
    if v < 30:
        return "Overweight"
    return "Obese"


def _bmr_mifflin(profile: HealthProfile) -> float:
    base = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age
    if profile.gender == "male":
        return float(base + 5)
    if profile.gender == "female":
        return float(base - 161)
    return float(base - 78)


def _activity_mult(level: str) -> float:
    return {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very-active": 1.9,
    }[level]


def _goal_mult(goal: str) -> float:
    return {
        "weight-loss": 0.8,
        "weight-gain": 1.2,
        "muscle-gain": 1.1,
        "maintenance": 1.0,
        "health-improvement": 1.0,
    }[goal]


def _macro_ratio(goal: str) -> Dict[str, float]:
    if goal == "weight-loss":
        return {"protein": 0.30, "carbs": 0.35, "fat": 0.35}
    if goal == "muscle-gain":
        return {"protein": 0.30, "carbs": 0.40, "fat": 0.30}
    if goal == "weight-gain":
        return {"protein": 0.20, "carbs": 0.50, "fat": 0.30}
    return {"protein": 0.25, "carbs": 0.45, "fat": 0.30}


def baseline_targets(
    profile: HealthProfile,
) -> Tuple[int, int, int, int, int, int]:
    bmr = _bmr_mifflin(profile)
    tdee = bmr * _activity_mult(profile.activityLevel)
    calories = int(round(tdee * _goal_mult(profile.healthGoal)))
    ratio = _macro_ratio(profile.healthGoal)
    protein_g = int(round((calories * ratio["protein"]) / 4))
    carbs_g = int(round((calories * ratio["carbs"]) / 4))
    fat_g = int(round((calories * ratio["fat"]) / 9))
    return (
        int(round(bmr)),
        int(round(tdee)),
        calories,
        protein_g,
        carbs_g,
        fat_g,
    )


@dataclass(frozen=True)
class MacroTargets:
    bmi: float
    bmi_category: str
    bmr: int
    tdee: int
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int


_ml_model: Any = None


def _load_ml_model() -> Any:
    global _ml_model
    if _ml_model is not None:
        return _ml_model
    path = _model_path()
    if path.exists():
        _ml_model = joblib.load(path)
    else:
        _ml_model = None
    return _ml_model


def predict_macros(profile: HealthProfile) -> MacroTargets:
    """
    Predict daily calories + macros (g/day for P/C/F).
    If model.pkl is missing, uses deterministic BMR/TDEE + macro split (same as train labels).
    """
    v_bmi = _bmi(profile.height, profile.weight)
    bmi_cat = _bmi_category(v_bmi)
    model = _load_ml_model()

    if model is None:
        bmr, tdee, cal, p, c, f = baseline_targets(profile)
        return MacroTargets(
            bmi=round(v_bmi, 2),
            bmi_category=bmi_cat,
            bmr=bmr,
            tdee=tdee,
            calories=cal,
            protein_g=p,
            carbs_g=c,
            fat_g=f,
        )

    gender_map = {"male": 0, "female": 1, "other": 2}
    activity_map = {
        "sedentary": 0,
        "light": 1,
        "moderate": 2,
        "active": 3,
        "very-active": 4,
    }
    goal_map = {
        "weight-loss": 0,
        "weight-gain": 1,
        "muscle-gain": 2,
        "maintenance": 3,
        "health-improvement": 4,
    }

    x = np.array(
        [
            [
                float(profile.age),
                float(profile.weight),
                float(profile.height),
                float(gender_map[profile.gender]),
                float(activity_map[profile.activityLevel]),
                float(goal_map[profile.healthGoal]),
            ]
        ],
        dtype=np.float32,
    )
    y = model.predict(x)
    y = np.asarray(y).reshape(-1)
    calories = int(round(float(y[0])))
    protein_g = int(round(float(y[1])))
    carbs_g = int(round(float(y[2])))
    fat_g = int(round(float(y[3])))

    bmr = int(round(_bmr_mifflin(profile)))
    tdee = int(round(bmr * _activity_mult(profile.activityLevel)))

    return MacroTargets(
        bmi=round(v_bmi, 2),
        bmi_category=bmi_cat,
        bmr=bmr,
        tdee=tdee,
        calories=calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
    )


# --- Rule-based copy for analysis.* text fields (no LLM; matches Gemini JSON shape) ---


def _analysis_text_bundle(profile: HealthProfile, bmi: float, bmi_cat: str) -> Dict[str, Any]:
    """Fill healthStatus, risks, insights, recommendations in English (UI-safe strings)."""
    goal = profile.healthGoal
    risks: List[str] = []
    if bmi >= 30:
        risks.extend(
            [
                "Higher risk of type 2 diabetes and cardiovascular disease at this BMI range.",
                "Discuss a gradual weight plan with a clinician if you have medical conditions.",
            ]
        )
    elif bmi >= 25:
        risks.append("Elevated BMI may increase long-term cardiometabolic risk if trends persist.")
    elif bmi < 18.5:
        risks.append("Underweight can affect energy, immunity, and recovery; seek guidance if unintended.")

    status = f"BMI indicates {bmi_cat.lower()}. This is informational, not a diagnosis."
    insights = [
        f"Estimated maintenance context: goal is {goal.replace('-', ' ')}; adjust calories gradually.",
        "Prioritize protein distribution across meals to support satiety and recovery.",
        "Whole foods and consistent meal timing usually outperform extreme restriction.",
    ]
    recs = [
        "Build most meals from lean protein, vegetables, whole grains, and healthy fats.",
        "Hydrate regularly; limit sugar-sweetened drinks.",
        "Track weekly averages (weight/energy) rather than single-day swings.",
        "If you have allergies or conditions, validate any plan with a qualified professional.",
    ]
    if profile.dietaryRestrictions:
        recs.append(
            f"Respect stated dietary constraints: {profile.dietaryRestrictions.strip()}"
        )

    return {
        "healthStatus": status,
        "healthRisks": risks,
        "healthInsights": insights,
        "recommendations": recs,
    }


# --- Food data + simple restriction filter ---


def _restriction_tokens(dietary_restrictions: str) -> List[str]:
    if not dietary_restrictions or not str(dietary_restrictions).strip():
        return []
    parts = re.split(r"[,;\n]+", dietary_restrictions.lower())
    return [p.strip() for p in parts if len(p.strip()) >= 2]


def _food_matches_restriction(food_name: str, tokens: List[str]) -> bool:
    name = food_name.lower()
    for t in tokens:
        if len(t) >= 3 and t in name:
            return True
    return False


@dataclass(frozen=True)
class FoodItem:
    food_id: str
    name: str
    calories: int
    protein: float
    carbs: float
    fat: float
    meal_type: str


def load_foods() -> List[FoodItem]:
    path = _food_csv_path()
    if not path.exists():
        return []
    df = pd.read_csv(path)
    foods: List[FoodItem] = []
    for _, r in df.iterrows():
        foods.append(
            FoodItem(
                food_id=str(r["food_id"]),
                name=str(r["name"]),
                calories=int(r["calories"]),
                protein=float(r.get("protein", 0.0)),
                carbs=float(r.get("carbs", 0.0)),
                fat=float(r.get("fat", 0.0)),
                meal_type=str(r.get("meal_type", "any")),
            )
        )
    return foods


def filter_foods_by_restrictions(
    foods: List[FoodItem], dietary_restrictions: str
) -> List[FoodItem]:
    """Drop foods whose name contains a restriction token (naive; good enough for demo CSV)."""
    tokens = _restriction_tokens(dietary_restrictions)
    if not tokens:
        return foods
    return [f for f in foods if not _food_matches_restriction(f.name, tokens)]


def recommend_meal_greedy(target_calories: int) -> List[Dict[str, Any]]:
    """Simple knapsack-style greedy pick by calories (ascending) until near target."""
    foods = load_foods()
    if not foods:
        return []
    rows = sorted(
        [
            {
                "food_id": f.food_id,
                "name": f.name,
                "calories": f.calories,
                "protein": f.protein,
                "carbs": f.carbs,
                "fat": f.fat,
            }
            for f in foods
        ],
        key=lambda x: x["calories"],
    )
    result: List[Dict[str, Any]] = []
    total = 0
    for food in rows:
        if total + food["calories"] <= target_calories:
            result.append(food)
            total += food["calories"]
        if total >= target_calories * 0.98:
            break
    return result


def _filter_by_meal_type(foods: List[FoodItem], meal_type: str) -> List[FoodItem]:
    meal_type = meal_type.lower().strip()
    out = [f for f in foods if f.meal_type in (meal_type, "any")]
    return out if out else foods


def _greedy_select(
    target_cal: int, foods: List[FoodItem]
) -> Tuple[List[FoodItem], int]:
    selected: List[FoodItem] = []
    total = 0
    for food in sorted(foods, key=lambda x: x.calories):
        if total + food.calories <= target_cal:
            selected.append(food)
            total += food.calories
        if total >= target_cal * 0.98:
            break
    return selected, total


def _slot_from_foods(title: str, selected: List[FoodItem], calories: int) -> Dict[str, Any]:
    if not selected:
        return {
            "meal": title,
            "calories": calories,
            "description": "No matching foods found (check food.csv or dietary filters).",
        }
    names = ", ".join([f.name for f in selected[:8]])
    if len(selected) > 8:
        names += ", ..."
    return {"meal": title, "calories": calories, "description": names}


def generate_weekly_meal_plan(
    target_daily_calories: int,
    dietary_restrictions: str = "",
    days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Split daily calories across meals (30/40/25/5), greedy pick per slot.
    Applies dietaryRestrictions as a naive name filter when possible.
    """
    foods = filter_foods_by_restrictions(load_foods(), dietary_restrictions)
    dist = {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.25, "snack": 0.05}
    plan: List[Dict[str, Any]] = []
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    for i in range(days):
        b_target = int(round(target_daily_calories * dist["breakfast"]))
        l_target = int(round(target_daily_calories * dist["lunch"]))
        d_target = int(round(target_daily_calories * dist["dinner"]))
        s_target = max(0, target_daily_calories - (b_target + l_target + d_target))

        b_sel, b_cal = _greedy_select(b_target, _filter_by_meal_type(foods, "breakfast"))
        l_sel, l_cal = _greedy_select(l_target, _filter_by_meal_type(foods, "lunch"))
        d_sel, d_cal = _greedy_select(d_target, _filter_by_meal_type(foods, "dinner"))
        s_sel, s_cal = _greedy_select(s_target, _filter_by_meal_type(foods, "snack"))

        daily_total = b_cal + l_cal + d_cal + s_cal
        plan.append(
            {
                "day": day_names[i % len(day_names)],
                "breakfast": _slot_from_foods("Breakfast", b_sel, b_cal),
                "lunch": _slot_from_foods("Lunch", l_sel, l_cal),
                "dinner": _slot_from_foods("Dinner", d_sel, d_cal),
                "snack": _slot_from_foods("Snack", s_sel, s_cal),
                "dailyTotalCalories": int(daily_total),
            }
        )
    return plan


def build_gemini_style_response(profile: HealthProfile) -> Dict[str, Any]:
    """
    Full response matching GeminiHealthAnalysis / GeminiHealthAnalysisDto (Nest + Next).
    """
    targets = predict_macros(profile)
    text = _analysis_text_bundle(profile, targets.bmi, targets.bmi_category)
    weekly = generate_weekly_meal_plan(
        target_daily_calories=targets.calories,
        dietary_restrictions=profile.dietaryRestrictions or "",
    )

    return {
        "analysis": {
            "bmi": targets.bmi,
            "bmiCategory": targets.bmi_category,
            "bmr": targets.bmr,
            "tdee": targets.tdee,
            "recommendedCalories": targets.calories,
            "macronutrients": {
                "protein": targets.protein_g,
                "carbs": targets.carbs_g,
                "fat": targets.fat_g,
            },
            "healthStatus": text["healthStatus"],
            "healthRisks": text["healthRisks"],
            "healthInsights": text["healthInsights"],
            "recommendations": text["recommendations"],
        },
        "exerciseRecommendations": [],
        "foodRecommendations": [],
        "weeklyMealPlan": weekly,
        "aiInsights": [],
        "aiRecommendations": [],
        "personalityProfile": {
            "eatingStyle": "Structured, whole-food forward",
            "motivation": profile.healthGoal.replace("-", " "),
            "challenges": ["Consistency under busy schedules", "Portion awareness"],
            "strengths": ["You are actively reviewing nutrition targets"],
            "preferences": (
                [profile.dietaryRestrictions.strip()]
                if profile.dietaryRestrictions
                else ["Balanced meals", "Flexible planning"]
            ),
        },
    }
