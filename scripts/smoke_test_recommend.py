from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import httpx


def main() -> int:
    """
    Smoke test for /recommend.

    Usage:
      python scripts/smoke_test_recommend.py http://127.0.0.1:8000

    Notes:
      - If MODEL_PATH points to an existing model.pkl, the API should use ML predictions.
      - If USE_LOCAL_LLM=1 and llama.cpp is reachable, narrative fields may be enriched.
    """
    base = (sys.argv[1] if len(sys.argv) >= 2 else "http://127.0.0.1:8000").rstrip("/")
    url = f"{base}/recommend"

    payload: Dict[str, Any] = {
        "age": 28,
        "gender": "female",
        "height": 165,
        "weight": 60,
        "activityLevel": "moderate",
        "healthGoal": "maintenance",
        "dietaryRestrictions": "peanut",
    }

    model_path = os.getenv("MODEL_PATH", "").strip()
    print(f"POST {url}")
    if model_path:
        print(f"MODEL_PATH={model_path}")
    print(f"USE_LOCAL_LLM={os.getenv('USE_LOCAL_LLM', '0')}")

    with httpx.Client(timeout=60) as client:
        res = client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()

    # Print just the most important fields.
    analysis = data.get("analysis") or {}
    out = {
        "analysis": {
            "bmi": analysis.get("bmi"),
            "bmr": analysis.get("bmr"),
            "tdee": analysis.get("tdee"),
            "recommendedCalories": analysis.get("recommendedCalories"),
            "macronutrients": analysis.get("macronutrients"),
        },
        "weeklyMealPlanDays": len(data.get("weeklyMealPlan") or []),
        "healthInsightsCount": len((analysis.get("healthInsights") or [])),
        "aiInsightsCount": len((data.get("aiInsights") or [])),
        "aiRecommendationsCount": len((data.get("aiRecommendations") or [])),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

