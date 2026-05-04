from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.food_catalog_v2 import filter_foods_v2, load_foods_v2
from app.mealplan_optimizer import MealPlanConstraints, build_daily_plan


@dataclass(frozen=True)
class ProfileCase:
    name: str
    calories: int
    protein: int
    carbs: int
    fat: int
    restrictions: MealPlanConstraints


def main() -> int:
    """
    Benchmark meal-plan accuracy against targets.

    Usage:
      python scripts/benchmark_mealplan.py data/food.v2.sample.csv
    """
    import sys

    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/food.v2.sample.csv")
    foods = load_foods_v2(path)
    if not foods:
        print("No foods found. Provide a valid food.v2.csv")
        return 1

    cases = [
        ProfileCase(
            name="maintenance_moderate",
            calories=2000,
            protein=125,
            carbs=225,
            fat=67,
            restrictions=MealPlanConstraints(set(), set(), set()),
        ),
        ProfileCase(
            name="lowcarb_no_peanut",
            calories=1800,
            protein=140,
            carbs=120,
            fat=80,
            restrictions=MealPlanConstraints({"peanut"}, {"low_carb"}, set()),
        ),
    ]

    results: List[Dict[str, Any]] = []
    for c in cases:
        foods_by_meal = {}
        for meal in ("breakfast", "lunch", "dinner", "snack"):
            filtered = filter_foods_v2(
                foods,
                allowed_meal_types={meal},
                avoid_allergens=c.restrictions.avoid_allergens,
                require_tags=c.restrictions.require_tags,
                avoid_tags=c.restrictions.avoid_tags,
            )
            foods_by_meal[meal] = filtered

        daily = build_daily_plan(
            foods_by_meal=foods_by_meal,
            target_daily_calories=c.calories,
            dietary_constraints=c.restrictions,
            macro_targets_daily=(c.protein, c.carbs, c.fat),
        )
        err = abs(int(daily["dailyTotalCalories"]) - c.calories)
        results.append({"case": c.name, "target": c.calories, "actual": daily["dailyTotalCalories"], "abs_error": err})

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

