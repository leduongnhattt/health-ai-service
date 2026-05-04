from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

log = logging.getLogger("health_ai.food_v2")


MealType = str  # breakfast|lunch|dinner|snack|any


@dataclass(frozen=True)
class FoodItemV2:
    food_id: str
    name: str
    serving_g: int
    calories_per_serving: int
    protein_g_per_serving: float
    carb_g_per_serving: float
    fat_g_per_serving: float
    meal_type: MealType
    tags: Set[str]
    allergens: Set[str]


_cache: Dict[str, Any] = {"path": None, "mtime": None, "foods": None}


def _parse_csv_set(value: Any) -> Set[str]:
    if value is None:
        return set()
    s = str(value).strip()
    if not s:
        return set()
    parts = [p.strip().lower() for p in s.split(",")]
    return {p for p in parts if p}


REQUIRED_COLS = {
    "food_id",
    "name",
    "serving_g",
    "calories_per_serving",
    "protein_g_per_serving",
    "carb_g_per_serving",
    "fat_g_per_serving",
    "meal_type",
    "tags",
    "allergens",
}


def validate_food_v2_df(df: pd.DataFrame) -> None:
    missing = [c for c in sorted(REQUIRED_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"food.v2.csv is missing columns: {missing}")

    # Basic sanity checks (non-negative numeric columns)
    numeric_cols = [
        "serving_g",
        "calories_per_serving",
        "protein_g_per_serving",
        "carb_g_per_serving",
        "fat_g_per_serving",
    ]
    for c in numeric_cols:
        if (df[c].astype(float) < 0).any():
            raise ValueError(f"food.v2.csv has negative values in column: {c}")


def load_foods_v2(path: Path) -> List[FoodItemV2]:
    if not path.exists():
        return []

    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = None

    if _cache["foods"] is not None and _cache["path"] == str(path) and _cache["mtime"] == mtime:
        return _cache["foods"]

    t0 = time.perf_counter()
    df = pd.read_csv(path)
    validate_food_v2_df(df)

    foods: List[FoodItemV2] = []
    for _, r in df.iterrows():
        foods.append(
            FoodItemV2(
                food_id=str(r["food_id"]),
                name=str(r["name"]),
                serving_g=int(r["serving_g"]),
                calories_per_serving=int(r["calories_per_serving"]),
                protein_g_per_serving=float(r["protein_g_per_serving"]),
                carb_g_per_serving=float(r["carb_g_per_serving"]),
                fat_g_per_serving=float(r["fat_g_per_serving"]),
                meal_type=str(r.get("meal_type", "any")).strip().lower() or "any",
                tags=_parse_csv_set(r.get("tags", "")),
                allergens=_parse_csv_set(r.get("allergens", "")),
            )
        )

    _cache["path"] = str(path)
    _cache["mtime"] = mtime
    _cache["foods"] = foods
    log.debug(
        "Loaded food v2 CSV in %.1fms (%s) rows=%d",
        (time.perf_counter() - t0) * 1000.0,
        path,
        len(foods),
    )
    return foods


def filter_foods_v2(
    foods: Iterable[FoodItemV2],
    *,
    allowed_meal_types: Optional[Set[str]] = None,
    avoid_allergens: Optional[Set[str]] = None,
    require_tags: Optional[Set[str]] = None,
    avoid_tags: Optional[Set[str]] = None,
) -> List[FoodItemV2]:
    allowed_meal_types = {m.lower() for m in (allowed_meal_types or set())}
    avoid_allergens = {a.lower() for a in (avoid_allergens or set())}
    require_tags = {t.lower() for t in (require_tags or set())}
    avoid_tags = {t.lower() for t in (avoid_tags or set())}

    out: List[FoodItemV2] = []
    for f in foods:
        if allowed_meal_types and f.meal_type not in allowed_meal_types and f.meal_type != "any":
            continue
        if avoid_allergens and (f.allergens & avoid_allergens):
            continue
        if require_tags and not require_tags.issubset(f.tags):
            continue
        if avoid_tags and (f.tags & avoid_tags):
            continue
        out.append(f)
    return out

