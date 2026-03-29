"""
Generate synthetic labels (optional) and train a multi-output RandomForestRegressor.

Outputs:
  model/model.pkl  (joblib; sklearn MultiOutputRegressor)

Training CSV schema (required columns):
  age, weight, height, gender, activity, goal, calories, protein, carb, fat

Where:
  gender:   male | female | other
  activity: sedentary | light | moderate | active | very-active  (same strings as Nest HealthProfileDto.activityLevel)
  goal:     weight-loss | weight-gain | muscle-gain | maintenance | health-improvement  (same as healthGoal)

Aliases accepted (exported from spreadsheets / DB):
  activityLevel -> activity
  healthGoal    -> goal
  carbs         -> carb

Environment:
  NUTRITION_DATASET_PATH  override CSV path (default: data/nutrition_dataset.csv)

Usage (from health-ai-service directory):
  python train.py
  python train.py --csv path/to/real_data.csv
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = ROOT / "data" / "nutrition_dataset.csv"
MODEL_OUT = ROOT / "model" / "model.pkl"

ActivityLevel = Literal["sedentary", "light", "moderate", "active", "very-active"]
HealthGoal = Literal[
    "weight-loss",
    "weight-gain",
    "muscle-gain",
    "maintenance",
    "health-improvement",
]
Gender = Literal["male", "female", "other"]


def bmr_mifflin(age: int, gender: Gender, height: float, weight: float) -> float:
    base = 10 * weight + 6.25 * height - 5 * age
    if gender == "male":
        return base + 5
    if gender == "female":
        return base - 161
    return base - 78


def activity_mult(level: ActivityLevel) -> float:
    return {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very-active": 1.9,
    }[level]


def goal_mult(goal: HealthGoal) -> float:
    return {
        "weight-loss": 0.8,
        "weight-gain": 1.2,
        "muscle-gain": 1.1,
        "maintenance": 1.0,
        "health-improvement": 1.0,
    }[goal]


def macro_ratio(goal: HealthGoal) -> Tuple[float, float, float]:
    if goal == "weight-loss":
        return 0.30, 0.35, 0.35
    if goal == "muscle-gain":
        return 0.30, 0.40, 0.30
    if goal == "weight-gain":
        return 0.20, 0.50, 0.30
    return 0.25, 0.45, 0.30


@dataclass(frozen=True)
class Row:
    age: int
    weight: float
    height: float
    gender: Gender
    activity: ActivityLevel
    goal: HealthGoal
    calories: int
    protein: int
    carb: int
    fat: int


def synth_row() -> Row:
    age = random.randint(15, 70)
    height = random.randint(145, 195)
    weight = random.randint(40, 120)
    gender: Gender = random.choice(["male", "female", "other"])
    activity: ActivityLevel = random.choice(
        ["sedentary", "light", "moderate", "active", "very-active"]
    )
    goal: HealthGoal = random.choice(
        [
            "weight-loss",
            "weight-gain",
            "muscle-gain",
            "maintenance",
            "health-improvement",
        ]
    )

    bmr = bmr_mifflin(age, gender, float(height), float(weight))
    tdee = bmr * activity_mult(activity)
    calories = int(round(tdee * goal_mult(goal)))

    pr, cr, fr = macro_ratio(goal)
    protein = int(round((calories * pr) / 4))
    carb = int(round((calories * cr) / 4))
    fat = int(round((calories * fr) / 9))

    return Row(
        age=age,
        weight=float(weight),
        height=float(height),
        gender=gender,
        activity=activity,
        goal=goal,
        calories=calories,
        protein=protein,
        carb=carb,
        fat=fat,
    )


def generate_dataset(n: int = 50000, out_path: Path | None = None) -> Path:
    out = out_path or DEFAULT_DATA_CSV
    rows: List[Row] = [synth_row() for _ in range(n)]
    df = pd.DataFrame([asdict(r) for r in rows])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    return out


def normalize_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Align spreadsheet/DB export column names with the trainer."""
    if "activity" not in df.columns and "activityLevel" in df.columns:
        df = df.rename(columns={"activityLevel": "activity"})
    if "goal" not in df.columns and "healthGoal" in df.columns:
        df = df.rename(columns={"healthGoal": "goal"})
    if "carb" not in df.columns and "carbs" in df.columns:
        df = df.rename(columns={"carbs": "carb"})
    return df


REQUIRED = [
    "age",
    "weight",
    "height",
    "gender",
    "activity",
    "goal",
    "calories",
    "protein",
    "carb",
    "fat",
]


def validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Required: {REQUIRED}. "
            f"Tip: export activityLevel/healthGoal as activity/goal, or carbs as carb."
        )


def encode(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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

    x = np.stack(
        [
            df["age"].astype(float).to_numpy(),
            df["weight"].astype(float).to_numpy(),
            df["height"].astype(float).to_numpy(),
            df["gender"].map(gender_map).astype(float).to_numpy(),
            df["activity"].map(activity_map).astype(float).to_numpy(),
            df["goal"].map(goal_map).astype(float).to_numpy(),
        ],
        axis=1,
    )
    y = np.stack(
        [
            df["calories"].astype(float).to_numpy(),
            df["protein"].astype(float).to_numpy(),
            df["carb"].astype(float).to_numpy(),
            df["fat"].astype(float).to_numpy(),
        ],
        axis=1,
    )
    return x, y


def train(csv_path: Path | None = None) -> None:
    data_csv = csv_path or Path(os.getenv("NUTRITION_DATASET_PATH", str(DEFAULT_DATA_CSV)))
    if not data_csv.exists():
        generate_dataset(50000, out_path=data_csv)

    df = pd.read_csv(data_csv)
    df = normalize_training_columns(df)
    validate_df(df)
    x, y = encode(df)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    base = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base)
    model.fit(x_train, y_train)

    pred = model.predict(x_val)
    mae = mean_absolute_error(y_val, pred, multioutput="raw_values")
    print("MAE [calories, protein, carb, fat] =", mae.tolist())

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Saved model -> {MODEL_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nutrition macro regressor.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to training CSV (default: NUTRITION_DATASET_PATH or data/nutrition_dataset.csv)",
    )
    args = parser.parse_args()
    path = Path(args.csv) if args.csv else None
    train(csv_path=path)


if __name__ == "__main__":
    main()
