from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field

log = logging.getLogger("health_ai.restrictions")


class RestrictionConstraints(BaseModel):
    avoid_allergens: list[str] = Field(default_factory=list)
    required_tags: list[str] = Field(default_factory=list)
    avoid_tags: list[str] = Field(default_factory=list)


def constraints_json_schema() -> Dict[str, Any]:
    schema = RestrictionConstraints.model_json_schema()
    return {"name": "RestrictionConstraints", "schema": schema, "strict": True}


def _tokenize(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[,;\n]+", str(text).lower())
    return [p.strip() for p in parts if len(p.strip()) >= 2]


# Minimal mapping (extend as your food catalog grows)
ALLERGEN_SYNONYMS = {
    "peanut": {"peanut", "đậu phộng", "lac", "lạc"},
    "milk": {"milk", "sữa", "dairy"},
    "egg": {"egg", "trứng"},
    "soy": {"soy", "đậu nành"},
    "fish": {"fish", "cá"},
    "shellfish": {"shellfish", "tôm", "cua", "mực"},
    "tree_nut": {"almond", "hạt", "tree nut", "nuts", "hạnh nhân"},
    "gluten": {"gluten", "wheat", "bánh mì", "mì", "lúa mì"},
}

TAG_SYNONYMS = {
    "vegetarian": {"vegetarian", "ăn chay", "chay"},
    "low_carb": {"low carb", "ít carb", "ít tinh bột"},
    "high_protein": {"high protein", "nhiều đạm", "giàu đạm"},
    "no_spicy": {"no spicy", "không cay", "ít cay"},
}


def parse_constraints_rule_based(dietary_restrictions: str) -> RestrictionConstraints:
    tokens = _tokenize(dietary_restrictions)
    avoid_allergens: Set[str] = set()
    required_tags: Set[str] = set()
    avoid_tags: Set[str] = set()

    for t in tokens:
        for k, syns in ALLERGEN_SYNONYMS.items():
            if any(s in t for s in syns):
                avoid_allergens.add(k)
        for k, syns in TAG_SYNONYMS.items():
            if any(s in t for s in syns):
                # heuristics: if token contains 'no'/'không' -> avoid, else require
                if "no" in t or "không" in t:
                    avoid_tags.add(k)
                else:
                    required_tags.add(k)

    return RestrictionConstraints(
        avoid_allergens=sorted(avoid_allergens),
        required_tags=sorted(required_tags),
        avoid_tags=sorted(avoid_tags),
    )


def parse_constraints_llm(
    *,
    dietary_restrictions: str,
    chat_completions,
    locale: str = "vi",
) -> Optional[RestrictionConstraints]:
    """
    Ask LLM to convert free-text restrictions into structured constraints.
    Expects JSON matching RestrictionConstraints schema.
    """
    if not dietary_restrictions or not str(dietary_restrictions).strip():
        return RestrictionConstraints()

    system = (
        "Bạn là trợ lý chuyển đổi hạn chế ăn uống thành JSON."
        if locale.lower().startswith("vi")
        else "You convert dietary restrictions into JSON."
    )
    user = (
        "Trả về JSON theo schema RestrictionConstraints.\n"
        "Chỉ JSON, không thêm chữ.\n"
        f"Input: {dietary_restrictions.strip()}"
        if locale.lower().startswith("vi")
        else f"Return JSON only.\nInput: {dietary_restrictions.strip()}"
    )
    try:
        text = chat_completions(system=system, user=user, json_schema=constraints_json_schema())
        obj = json.loads(text.strip().strip("```").replace("json", "", 1).strip())
        return RestrictionConstraints.model_validate(obj)
    except Exception as e:
        log.warning("LLM restrictions parse failed; falling back. err=%s", str(e))
        return None


def normalize_restrictions(
    dietary_restrictions: str,
    *,
    use_llm: bool,
    chat_completions=None,
    locale: str = "vi",
) -> RestrictionConstraints:
    if use_llm and chat_completions is not None:
        parsed = parse_constraints_llm(
            dietary_restrictions=dietary_restrictions,
            chat_completions=chat_completions,
            locale=locale,
        )
        if parsed is not None:
            return parsed
    return parse_constraints_rule_based(dietary_restrictions)

