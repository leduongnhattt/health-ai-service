from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


ActivityLevel = Literal["sedentary", "light", "moderate", "active", "very-active"]
HealthGoal = Literal[
    "weight-loss",
    "weight-gain",
    "muscle-gain",
    "maintenance",
    "health-improvement",
]
Gender = Literal["male", "female", "other"]


class UserInput(BaseModel):
    """
    Minimal 4-field demo for /predict and /demo/recommend.
    Maps activity 0..4 to activityLevel (sedentary .. very-active).
    For production, use HealthProfile (matches Nest + Next body).
    """

    age: int = Field(ge=1, le=120)
    weight: float = Field(ge=20, le=300)
    height: float = Field(ge=50, le=250)
    activity: float = Field(
        ge=0,
        le=4,
        description="Ordinal 0=sedentary … 4=very-active",
    )


class NutritionResponse(BaseModel):
    calories: float
    protein: float
    carb: float
    fat: float


class HealthProfile(BaseModel):
    """Same fields as food-delivery-server HealthProfileDto / UI HealthProfile."""

    age: int = Field(ge=1, le=120)
    gender: Gender
    height: float = Field(ge=50, le=250, description="cm")
    weight: float = Field(ge=20, le=300, description="kg")
    activityLevel: ActivityLevel
    healthGoal: HealthGoal
    dietaryRestrictions: Optional[str] = ""


class Macronutrients(BaseModel):
    protein: int
    carbs: int
    fat: int


class HealthAnalysis(BaseModel):
    bmi: float
    bmiCategory: str
    bmr: int
    tdee: int
    recommendedCalories: int
    macronutrients: Macronutrients
    healthStatus: str
    healthRisks: List[str] = Field(default_factory=list)
    healthInsights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ExerciseRecommendation(BaseModel):
    name: str
    description: str
    duration: str
    frequency: str
    difficultyLevel: str
    benefits: List[str] = Field(default_factory=list)
    tutorialLink: str
    equipment: str
    instructions: List[str] = Field(default_factory=list)


class FoodRecommendation(BaseModel):
    category: str
    eat: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    benefits: Optional[str] = None


class MealSlot(BaseModel):
    meal: str
    calories: int
    description: str


class MealPlan(BaseModel):
    day: str
    breakfast: MealSlot
    lunch: MealSlot
    dinner: MealSlot
    snack: MealSlot
    dailyTotalCalories: int


class PersonalityProfile(BaseModel):
    eatingStyle: str
    motivation: str
    challenges: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)


class GeminiHealthAnalysis(BaseModel):
    """Same nested shape as gemini-health-ai.service.ts GeminiHealthAnalysis."""

    analysis: HealthAnalysis
    exerciseRecommendations: List[ExerciseRecommendation] = Field(default_factory=list)
    foodRecommendations: List[FoodRecommendation] = Field(default_factory=list)
    weeklyMealPlan: List[MealPlan] = Field(default_factory=list)
    personalityProfile: PersonalityProfile
