from __future__ import annotations

"""
Generate a synthetic SFT dataset (JSONL) for fine-tuning a small instruction LLM to output
GeminiHealthAnalysis JSON. This generator anchors all numeric fields to the engine outputs
and produces diverse narrative text via templates.

This is intentionally lightweight and deterministic-ish so you can iterate quickly.

Usage (from health-ai-service):
  python scripts/generate_sft_dataset.py --out data/health_sft_train.jsonl --n 2000
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from app.schemas import HealthProfile
from app.services import generate_weekly_meal_plan, predict_macros


def _rand_profile() -> HealthProfile:
    genders = ["male", "female", "other"]
    activities = ["sedentary", "light", "moderate", "active", "very-active"]
    goals = [
        "weight-loss",
        "weight-gain",
        "muscle-gain",
        "maintenance",
        "health-improvement",
    ]
    dietary = ["", "no peanuts", "lactose-free", "low sugar", "no pork", "vegetarian"]
    return HealthProfile(
        age=random.randint(18, 65),
        gender=random.choice(genders),  # type: ignore[arg-type]
        height=float(random.randint(150, 190)),
        weight=float(random.randint(45, 115)),
        activityLevel=random.choice(activities),  # type: ignore[arg-type]
        healthGoal=random.choice(goals),  # type: ignore[arg-type]
        dietaryRestrictions=random.choice(dietary),
    )


def _templates(profile: HealthProfile, computed: Dict[str, Any]) -> Dict[str, Any]:
    goal = profile.healthGoal.replace("-", " ")
    dr = (profile.dietaryRestrictions or "").strip()
    bmi_cat = computed["bmiCategory"]

    risks: List[str] = []
    if bmi_cat in ("Overweight", "Obese"):
        risks.append("Thừa cân/béo phì có thể làm tăng nguy cơ tim mạch và rối loạn chuyển hóa nếu kéo dài.")
    if bmi_cat == "Underweight":
        risks.append("Thiếu cân kéo dài có thể ảnh hưởng năng lượng, miễn dịch và phục hồi.")
    if not risks:
        risks.append("Không thấy nguy cơ nổi bật từ BMI; vẫn cần theo dõi giấc ngủ, vận động và stress.")

    insights = [
        f"Mục tiêu hiện tại là {goal}; ưu tiên điều chỉnh bền vững thay vì cực đoan.",
        "Chia protein đều trong ngày giúp no lâu và hỗ trợ phục hồi.",
        "Theo dõi xu hướng theo tuần (không chỉ 1 ngày) để ra quyết định chính xác.",
    ]
    if dr:
        insights.append(f"Cần tuân thủ hạn chế ăn uống: {dr}.")

    recs = [
        "Ưu tiên thực phẩm ít chế biến: đạm nạc, rau, ngũ cốc nguyên hạt, chất béo tốt.",
        "Uống đủ nước, hạn chế đồ uống có đường.",
        "Nếu bạn tập tạ, hãy giữ protein cao và ngủ đủ để phục hồi.",
    ]

    exercises = [
        {
            "name": "Đi bộ nhanh (Zone 2)",
            "description": "Cardio cường độ vừa, dễ duy trì để tăng tiêu hao năng lượng.",
            "duration": "30–45 phút",
            "frequency": "4–5 buổi/tuần",
            "difficultyLevel": "Dễ",
            "benefits": ["Cải thiện tim mạch", "Tăng tiêu hao calo", "Giảm stress"],
            "tutorialLink": "https://www.youtube.com/results?search_query=zone+2+walking",
            "equipment": "Giày thể thao",
            "instructions": [
                "Khởi động 5 phút.",
                "Tăng tốc đến mức nói chuyện được nhưng hơi thở nhanh.",
                "Giữ 30–45 phút.",
                "Hạ nhiệt 5 phút.",
            ],
        }
    ]

    food_recs = [
        {
            "category": "Ưu tiên",
            "eat": ["Đạm nạc (gà, cá, trứng)", "Rau xanh và trái cây nguyên quả", "Ngũ cốc nguyên hạt"],
            "avoid": ["Nước ngọt", "Đồ chiên rán", "Bánh kẹo nhiều đường"],
            "benefits": "Giúp kiểm soát năng lượng, hỗ trợ mục tiêu và ổn định đường huyết.",
        }
    ]
    if dr:
        food_recs[0]["avoid"].append(dr)

    ai_insights = [
        {
            "category": "Nutrition",
            "priority": "high",
            "insight": "Tập trung vào protein và chất xơ sẽ giúp bạn dễ duy trì mục tiêu hơn.",
            "reasoning": "Protein + fiber tăng cảm giác no và giúp kiểm soát tổng năng lượng.",
            "actionable": "Mỗi bữa chính có 1 khẩu phần đạm rõ ràng + 2 nắm rau.",
            "confidence": round(random.uniform(0.62, 0.82), 2),
        }
    ]

    ai_recs = [
        {
            "type": "habit",
            "title": "Theo dõi trung bình tuần",
            "description": "Cân 3–4 lần/tuần và lấy trung bình để đánh giá tiến độ.",
            "reasoning": "Dao động nước khiến cân nặng ngày lẻ không phản ánh chính xác.",
            "priority": 1,
            "timeframe": "2 tuần",
            "difficulty": "Dễ",
            "expectedOutcome": "Ra quyết định điều chỉnh hợp lý và tránh nản.",
        }
    ]

    personality = {
        "eatingStyle": "Thực tế, ưu tiên bữa ăn dễ duy trì",
        "motivation": goal,
        "challenges": ["Lịch bận dễ ăn vặt", "Khó kiểm soát khẩu phần khi ăn ngoài"],
        "strengths": ["Có mục tiêu rõ ràng", "Sẵn sàng theo dõi"],
        "preferences": [dr] if dr else ["Bữa ăn đơn giản, giàu protein"],
    }

    status = (
        f"BMI đang ở mức {bmi_cat.lower()}. Đây là chỉ số tham khảo, không phải chẩn đoán."
    )

    return {
        "healthStatus": status,
        "healthRisks": risks,
        "healthInsights": insights,
        "recommendations": recs,
        "exerciseRecommendations": exercises,
        "foodRecommendations": food_recs,
        "aiInsights": ai_insights,
        "aiRecommendations": ai_recs,
        "personalityProfile": personality,
    }


def make_example(profile: HealthProfile) -> Dict[str, Any]:
    t = predict_macros(profile)
    weekly = generate_weekly_meal_plan(
        target_daily_calories=t.calories,
        dietary_restrictions=profile.dietaryRestrictions or "",
        days=7,
    )
    computed = {
        "bmi": t.bmi,
        "bmiCategory": t.bmi_category,
        "bmr": t.bmr,
        "tdee": t.tdee,
        "recommendedCalories": t.calories,
        "macronutrients": {"protein": t.protein_g, "carbs": t.carbs_g, "fat": t.fat_g},
    }
    narrative = _templates(profile, computed)

    response = {
        "analysis": {
            **computed,
            **{
                "healthStatus": narrative["healthStatus"],
                "healthRisks": narrative["healthRisks"],
                "healthInsights": narrative["healthInsights"],
                "recommendations": narrative["recommendations"],
            },
        },
        "exerciseRecommendations": narrative["exerciseRecommendations"],
        "foodRecommendations": narrative["foodRecommendations"],
        "weeklyMealPlan": weekly,
        "aiInsights": narrative["aiInsights"],
        "aiRecommendations": narrative["aiRecommendations"],
        "personalityProfile": narrative["personalityProfile"],
    }

    system = (
        "Bạn là một chuyên gia dinh dưỡng và huấn luyện viên. "
        "Bạn PHẢI trả về JSON hợp lệ theo đúng schema. Không được thêm giải thích bên ngoài JSON."
    )
    user = (
        "Nhiệm vụ: tạo JSON theo schema `GeminiHealthAnalysis`.\n"
        "RÀNG BUỘC:\n"
        "- Giữ nguyên các số trong `computed` khi điền vào `analysis`.\n"
        "- `weeklyMealPlan` phải đúng y hệt dữ liệu đã cho.\n"
        "- Output: CHỈ JSON.\n\n"
        + "INPUT_CONTEXT_JSON:\n"
        + json.dumps(
            {"profile": profile.model_dump(), "computed": computed, "weeklyMealPlan": weekly},
            ensure_ascii=False,
        )
    )

    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "response": response}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/health_sft_train.jsonl")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for _ in range(int(args.n)):
            ex = make_example(_rand_profile())
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} examples -> {out}")


if __name__ == "__main__":
    main()

