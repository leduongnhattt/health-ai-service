from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ortools.sat.python import cp_model

from .food_catalog_v2 import FoodItemV2

log = logging.getLogger("health_ai.optimizer")


@dataclass(frozen=True)
class MealPlanConstraints:
    avoid_allergens: Set[str]
    require_tags: Set[str]
    avoid_tags: Set[str]


@dataclass(frozen=True)
class SlotResult:
    title: str
    calories: int
    description: str
    selected_food_ids: List[str]


def _scale_int(v: float, scale: int = 10) -> int:
    return int(round(float(v) * scale))


def solve_meal_slot(
    *,
    foods: Sequence[FoodItemV2],
    target_calories: int,
    title: str,
    max_items: int = 3,
    max_servings_per_item: int = 2,
    calorie_tolerance: float = 0.10,
    macro_targets: Optional[Tuple[int, int, int]] = None,  # (protein, carb, fat) grams
) -> SlotResult:
    """
    CP-SAT optimizer:
    - Select up to max_items items, each with 0..max_servings_per_item servings.
    - Minimize deviation from target calories (+ optional macro deviation).
    """
    if not foods:
        return SlotResult(
            title=title,
            calories=0,
            description="No matching foods found.",
            selected_food_ids=[],
        )

    model = cp_model.CpModel()

    servings: List[cp_model.IntVar] = []
    chosen: List[cp_model.BoolVar] = []
    for i in range(len(foods)):
        s = model.NewIntVar(0, max_servings_per_item, f"s_{i}")
        c = model.NewBoolVar(f"c_{i}")
        # chosen implies servings >= 1
        model.Add(s >= 1).OnlyEnforceIf(c)
        model.Add(s == 0).OnlyEnforceIf(c.Not())
        servings.append(s)
        chosen.append(c)

    model.Add(sum(chosen) <= max_items)

    cal_expr = sum(servings[i] * int(foods[i].calories_per_serving) for i in range(len(foods)))
    cal_min = int(round(target_calories * (1.0 - calorie_tolerance)))
    cal_max = int(round(target_calories * (1.0 + calorie_tolerance)))
    model.Add(cal_expr >= cal_min)
    model.Add(cal_expr <= cal_max)

    # Absolute deviation for calories
    cal_dev = model.NewIntVar(0, max(10000, cal_max + target_calories), "cal_dev")
    model.Add(cal_dev >= cal_expr - target_calories)
    model.Add(cal_dev >= target_calories - cal_expr)

    obj_terms: List[cp_model.IntVar] = [cal_dev]

    # Optional macro objective (soft)
    if macro_targets:
        p_t, c_t, f_t = macro_targets
        scale = 10
        p_expr = sum(servings[i] * _scale_int(foods[i].protein_g_per_serving, scale) for i in range(len(foods)))
        c_expr = sum(servings[i] * _scale_int(foods[i].carb_g_per_serving, scale) for i in range(len(foods)))
        f_expr = sum(servings[i] * _scale_int(foods[i].fat_g_per_serving, scale) for i in range(len(foods)))

        p_dev = model.NewIntVar(0, 50000, "p_dev")
        c_dev = model.NewIntVar(0, 50000, "c_dev")
        f_dev = model.NewIntVar(0, 50000, "f_dev")
        model.Add(p_dev >= p_expr - _scale_int(p_t, scale))
        model.Add(p_dev >= _scale_int(p_t, scale) - p_expr)
        model.Add(c_dev >= c_expr - _scale_int(c_t, scale))
        model.Add(c_dev >= _scale_int(c_t, scale) - c_expr)
        model.Add(f_dev >= f_expr - _scale_int(f_t, scale))
        model.Add(f_dev >= _scale_int(f_t, scale) - f_expr)

        # Weight macros lower than calories (scaled already).
        obj_terms.extend([p_dev, c_dev, f_dev])

    # Minimize weighted sum: calories dominant.
    model.Minimize(obj_terms[0] * 1000 + sum(obj_terms[1:]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.25
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SlotResult(
            title=title,
            calories=0,
            description="No feasible meal found (tight constraints).",
        )

    selected: List[str] = []
    selected_food_ids: List[str] = []
    total_cal = 0
    for i, food in enumerate(foods):
        s = int(solver.Value(servings[i]))
        if s <= 0:
            continue
        total_cal += s * int(food.calories_per_serving)
        selected_food_ids.append(food.food_id)
        name = food.name
        if s > 1:
            name = f"{name} x{s}"
        selected.append(name)

    desc = ", ".join(selected[:10])
    if len(selected) > 10:
        desc += ", ..."
    return SlotResult(
        title=title,
        calories=int(total_cal),
        description=desc or "No foods selected.",
        selected_food_ids=selected_food_ids,
    )


def build_daily_plan(
    *,
    foods_by_meal: Dict[str, List[FoodItemV2]],
    target_daily_calories: int,
    dietary_constraints: MealPlanConstraints,
    macro_targets_daily: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Build a single-day plan with 4 slots using a fixed distribution.
    Returns dict compatible with existing MealPlan schema (MealSlot fields).
    """
    dist = {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.25, "snack": 0.05}
    targets = {
        k: int(round(target_daily_calories * v)) for k, v in dist.items()
    }
    # Ensure totals sum correctly
    targets["snack"] = max(0, target_daily_calories - (targets["breakfast"] + targets["lunch"] + targets["dinner"]))

    # Macro targets per slot (optional): proportional to calories.
    macro_targets_slot: Dict[str, Optional[Tuple[int, int, int]]] = {k: None for k in targets}
    if macro_targets_daily:
        p_d, c_d, f_d = macro_targets_daily
        for k in targets:
            frac = targets[k] / max(1, target_daily_calories)
            macro_targets_slot[k] = (
                int(round(p_d * frac)),
                int(round(c_d * frac)),
                int(round(f_d * frac)),
            )

    slots = {}
    for meal_key, title in [
        ("breakfast", "Breakfast"),
        ("lunch", "Lunch"),
        ("dinner", "Dinner"),
        ("snack", "Snack"),
    ]:
        slot_foods = foods_by_meal.get(meal_key, [])
        # Filter by constraints (already pre-filtered upstream; keep as-is here).
        res = solve_meal_slot(
            foods=slot_foods,
            target_calories=targets[meal_key],
            title=title,
            macro_targets=macro_targets_slot[meal_key],
        )
        slots[meal_key] = {
            "meal": res.title,
            "calories": res.calories,
            "description": res.description,
            "_foodIds": res.selected_food_ids,
        }

    daily_total = int(sum(int(slots[k]["calories"]) for k in slots))
    return {**slots, "dailyTotalCalories": daily_total}

