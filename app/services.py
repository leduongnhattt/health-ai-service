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

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from .schemas import HealthProfile, UserInput
from .food_catalog_v2 import filter_foods_v2, load_foods_v2
from .mealplan_optimizer import MealPlanConstraints, build_daily_plan
from .restrictions import normalize_restrictions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "model" / "model.pkl"
DEFAULT_FOOD_CSV = ROOT / "data" / "food.csv"
DEFAULT_FOOD_V2_CSV = ROOT / "data" / "food.v2.csv"
log = logging.getLogger("health_ai.services")

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


def _food_v2_csv_path() -> Path:
    p = os.getenv("FOODS_V2_CSV_PATH", "").strip()
    return Path(p) if p else DEFAULT_FOOD_V2_CSV


def _use_food_v2() -> bool:
    return os.getenv("USE_FOOD_V2", "0").strip().lower() in ("1", "true", "yes", "on")


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
        t0 = time.perf_counter()
        _ml_model = joblib.load(path)
        log.info("Loaded model.pkl in %.1fms (%s)", (time.perf_counter() - t0) * 1000.0, path)
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
    try:
        y = model.predict(x)
        y = np.asarray(y).reshape(-1)
        calories = int(round(float(y[0])))
        protein_g = int(round(float(y[1])))
        carbs_g = int(round(float(y[2])))
        fat_g = int(round(float(y[3])))
    except Exception as e:
        # Model files can be incompatible across sklearn versions or corrupted; never 500 the API.
        log.warning("model_predict_failed: %r; falling back to baseline targets", e)
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


def _profile_plan_seed(profile: HealthProfile) -> int:
    """
    Stable 64-bit fingerprint of the full profile so different inputs yield different
    meal rotations and copy variants (same profile -> same seed).
    """
    dr = (profile.dietaryRestrictions or "").strip()
    payload = {
        "activityLevel": profile.activityLevel,
        "age": profile.age,
        "dietaryRestrictions": dr,
        "gender": profile.gender,
        "healthGoal": profile.healthGoal,
        "height": round(float(profile.height), 2),
        "weight": round(float(profile.weight), 2),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _analysis_text_bundle(profile: HealthProfile, bmi: float, bmi_cat: str) -> Dict[str, Any]:
    """Fill healthStatus, risks, insights, recommendations in English (UI-safe strings)."""
    goal = profile.healthGoal
    seed = _profile_plan_seed(profile)
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
    act = profile.activityLevel.replace("-", " ")
    insights = [
        (
            f"Profile context: age {profile.age}, {profile.gender}, activity {act}; "
            f"height {int(profile.height)} cm, weight {int(profile.weight)} kg — targets below reflect this snapshot."
        ),
        f"Estimated maintenance context: goal is {goal.replace('-', ' ')}; adjust calories gradually.",
        "Prioritize protein distribution across meals to support satiety and recovery.",
        "Whole foods and consistent meal timing usually outperform extreme restriction.",
    ]
    hydrate_variants = [
        "Hydrate regularly; limit sugar-sweetened drinks.",
        "Aim for steady hydration across the day; treat sweet drinks as occasional.",
        "Prefer water or unsweetened tea; keep liquid calories intentional.",
    ]
    recs = [
        "Build most meals from lean protein, vegetables, whole grains, and healthy fats.",
        hydrate_variants[seed % len(hydrate_variants)],
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


_foods_cache: Dict[str, Any] = {"mtime": None, "path": None, "foods": None}


def _validate_foods_df(df: pd.DataFrame) -> None:
    required = {"food_id", "name", "calories"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"food.csv is missing columns: {missing} (required: {sorted(required)})")


def load_foods() -> List[FoodItem]:
    path = _food_csv_path()
    if not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = None

    cached_path = _foods_cache.get("path")
    cached_mtime = _foods_cache.get("mtime")
    cached_foods = _foods_cache.get("foods")
    if cached_foods is not None and cached_path == str(path) and cached_mtime == mtime:
        return cached_foods

    t0 = time.perf_counter()
    df = pd.read_csv(path)
    _validate_foods_df(df)
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
    _foods_cache["path"] = str(path)
    _foods_cache["mtime"] = mtime
    _foods_cache["foods"] = foods
    log.debug("Loaded foods CSV in %.1fms (%s) rows=%d", (time.perf_counter() - t0) * 1000.0, path, len(foods))
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
    # IMPORTANT: keep incoming order for diversity.
    # The caller may rotate candidate ordering per-day; sorting here would erase that.
    for food in foods:
        if total + food.calories <= target_cal:
            selected.append(food)
            total += food.calories
        if total >= target_cal * 0.98:
            break
    return selected, total


def _deterministic_shuffle_seq(items: Sequence[Any], mix_key: int) -> List[Any]:
    """
    Same mix_key -> same order; different profile seed / day / meal -> different greedy paths.
    (Plain rotation often leaves greedy picks unchanged when one dominant item fits first.)
    """
    seq = list(items)
    if len(seq) <= 1:
        return seq
    rng = random.Random((mix_key % (2**61 - 2)) + 1)
    rng.shuffle(seq)
    return seq


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
    *,
    profile: HealthProfile | None = None,
) -> List[Dict[str, Any]]:
    """
    Split daily calories across meals (30/40/25/5), greedy pick per slot.
    Applies dietaryRestrictions as a naive name filter when possible.
    Candidate order is shuffled deterministically per profile/day/meal so greedy picks diverge.
    """
    foods = filter_foods_by_restrictions(load_foods(), dietary_restrictions)
    seed = _profile_plan_seed(profile) if profile is not None else 0
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

    def _slot_mix_key(day_idx: int, meal_code: int) -> int:
        # Include daily calorie target so nearby profiles still diverge if seeds were close.
        return (
            seed
            + day_idx * 1_000_003
            + meal_code * 97_679
            + int(target_daily_calories) * 31_337
        )

    for i in range(days):
        b_target = int(round(target_daily_calories * dist["breakfast"]))
        l_target = int(round(target_daily_calories * dist["lunch"]))
        d_target = int(round(target_daily_calories * dist["dinner"]))
        s_target = max(0, target_daily_calories - (b_target + l_target + d_target))

        b_list = _filter_by_meal_type(foods, "breakfast")
        l_list = _filter_by_meal_type(foods, "lunch")
        d_list = _filter_by_meal_type(foods, "dinner")
        s_list = _filter_by_meal_type(foods, "snack")
        b_candidates = _deterministic_shuffle_seq(b_list, _slot_mix_key(i, 1))
        l_candidates = _deterministic_shuffle_seq(l_list, _slot_mix_key(i, 2))
        d_candidates = _deterministic_shuffle_seq(d_list, _slot_mix_key(i, 3))
        s_candidates = _deterministic_shuffle_seq(s_list, _slot_mix_key(i, 4))

        b_sel, b_cal = _greedy_select(b_target, b_candidates)
        l_sel, l_cal = _greedy_select(l_target, l_candidates)
        d_sel, d_cal = _greedy_select(d_target, d_candidates)
        s_sel, s_cal = _greedy_select(s_target, s_candidates)

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


def _personality_from_seed(profile: HealthProfile, seed: int) -> Dict[str, Any]:
    eating_styles = [
        "Structured, whole-food forward",
        "Balanced plates with steady meal rhythm",
        "Simple ingredients, repeatable meal templates",
    ]
    challenges_pool = [
        ["Consistency under busy schedules", "Portion awareness"],
        ["Snacking triggers", "Weekend overeating"],
        ["Time to cook", "Hitting protein targets daily"],
    ]
    strengths_pool = [
        ["You are actively reviewing nutrition targets"],
        ["You are building sustainable habits step by step"],
        ["You care about measurable progress"],
    ]
    # Decorrelate indices so small seed differences don't reuse the same tuple everywhere.
    return {
        "eatingStyle": eating_styles[(seed >> 0) % len(eating_styles)],
        "motivation": profile.healthGoal.replace("-", " "),
        "challenges": challenges_pool[(seed >> 9) % len(challenges_pool)],
        "strengths": strengths_pool[(seed >> 18) % len(strengths_pool)],
        "preferences": (
            [profile.dietaryRestrictions.strip()]
            if profile.dietaryRestrictions
            else ["Balanced meals", "Flexible planning"]
        ),
    }


def build_gemini_style_response(profile: HealthProfile) -> Dict[str, Any]:
    """
    Full response matching GeminiHealthAnalysis / GeminiHealthAnalysisDto (Nest + Next).
    """
    seed = _profile_plan_seed(profile)
    targets = predict_macros(profile)
    text = _analysis_text_bundle(profile, targets.bmi, targets.bmi_category)
    weekly = generate_weekly_meal_plan(
        target_daily_calories=targets.calories,
        dietary_restrictions=profile.dietaryRestrictions or "",
        profile=profile,
    )

    # Optional v2 optimizer path (keeps output schema compatible).
    if _use_food_v2():
        v2_path = _food_v2_csv_path()
        foods_v2 = load_foods_v2(v2_path) if v2_path.exists() else []
        if foods_v2:
            use_llm = os.getenv("USE_LOCAL_LLM", "0").strip() in ("1", "true", "yes", "on")
            chat = None
            if use_llm:
                try:
                    from .llama_client import chat_completions as _chat

                    chat = _chat
                except Exception:
                    chat = None

            constraints = normalize_restrictions(
                profile.dietaryRestrictions or "",
                use_llm=use_llm,
                chat_completions=chat,
                locale=os.getenv("LOCALE", "vi"),
            )
            mc = MealPlanConstraints(
                avoid_allergens=set(constraints.avoid_allergens),
                require_tags=set(constraints.required_tags),
                avoid_tags=set(constraints.avoid_tags),
            )
            # Build weekly plan with rotation by excluding previous day picks (simple diversity).
            dist_meals = {"breakfast", "lunch", "dinner", "snack"}
            day_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            meal_ord = {"breakfast": 1, "lunch": 2, "dinner": 3, "snack": 4}
            used_food_ids: set[str] = set()
            weekly_v2: List[Dict[str, Any]] = []
            for i in range(7):
                foods_by_meal: Dict[str, List[Any]] = {}
                for meal in dist_meals:
                    filtered = filter_foods_v2(
                        foods_v2,
                        allowed_meal_types={meal},
                        avoid_allergens=mc.avoid_allergens,
                        require_tags=mc.require_tags,
                        avoid_tags=mc.avoid_tags,
                    )
                    # Diversity: prefer not used foods, but fallback if empty.
                    fresh = [f for f in filtered if f.food_id not in used_food_ids]
                    base = fresh or filtered
                    if base:
                        m = meal_ord.get(meal, 0)
                        mk = (
                            seed
                            + i * 1_000_003
                            + m * 97_679
                            + int(targets.calories) * 31_337
                        )
                        base = _deterministic_shuffle_seq(base, mk)
                    foods_by_meal[meal] = base

                daily = build_daily_plan(
                    foods_by_meal=foods_by_meal,
                    target_daily_calories=targets.calories,
                    dietary_constraints=mc,
                    macro_targets_daily=(targets.protein_g, targets.carbs_g, targets.fat_g),
                )
                # Diversity: track IDs chosen by solver so next day prefers fresh foods.
                for slot_key in ("breakfast", "lunch", "dinner", "snack"):
                    slot_obj = daily.get(slot_key) if isinstance(daily, dict) else None
                    if isinstance(slot_obj, dict):
                        chosen_ids = slot_obj.get("_foodIds")
                        if isinstance(chosen_ids, list):
                            used_food_ids.update([str(x) for x in chosen_ids if str(x).strip()])

                def _strip_meta(slot: Any) -> Any:
                    if not isinstance(slot, dict):
                        return slot
                    return {k: v for k, v in slot.items() if k != "_foodIds"}

                weekly_v2.append(
                    {
                        "day": day_names[i % len(day_names)],
                        "breakfast": _strip_meta(daily["breakfast"]),
                        "lunch": _strip_meta(daily["lunch"]),
                        "dinner": _strip_meta(daily["dinner"]),
                        "snack": _strip_meta(daily["snack"]),
                        "dailyTotalCalories": daily["dailyTotalCalories"],
                    }
                )
            weekly = weekly_v2

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
        # Engine mode: no invented exercise/food lists; narrative is rule-based + model.pkl.
        "exerciseRecommendations": [],
        "foodRecommendations": [],
        "weeklyMealPlan": weekly,
        "personalityProfile": _personality_from_seed(profile, seed),
    }


def build_health_response(profile: HealthProfile) -> Dict[str, Any]:
    """
    Entry point for `/recommend`: calories/macros from trained `model.pkl`, weekly plan from
    `food.csv` (or v2 optimizer when `USE_FOOD_V2=1`). No cloud LLM.
    """
    t0 = time.perf_counter()
    out = build_gemini_style_response(profile)
    log.info("build_health_response mode=engine_only ms=%.1f", (time.perf_counter() - t0) * 1000.0)
    return out
