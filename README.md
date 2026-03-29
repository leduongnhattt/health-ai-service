# Health AI Service (FastAPI)

Structure:

```text
health-ai-service/
├── model/
│   └── model.pkl              # produced by train.py (gitignored)
├── data/
│   ├── food.csv               # foods for meal optimizer
│   ├── nutrition_dataset.csv  # generated or your real training CSV
│   └── nutrition_dataset.example.csv
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── services.py
├── train.py
└── requirements.txt
```

## Contract (same as Gemini integration)

**Request** — `POST /recommend` body (matches `food-delivery-server` `HealthProfileDto` and `food-delivery-app` `HealthProfile`):

- `age`, `gender`, `height`, `weight`, `activityLevel`, `healthGoal`, `dietaryRestrictions`

**Response** — `GeminiHealthAnalysis` shape (see `app/schemas.py` and `gemini-health-ai.service.ts`).

## Install

```bash
pip install -r requirements.txt
```

## Train

Synthetic data is generated automatically if `data/nutrition_dataset.csv` is missing.

Use your own CSV (real labels):

```bash
set NUTRITION_DATASET_PATH=data/my_real.csv
python train.py
```

Or:

```bash
python train.py --csv data/my_real.csv
```

Required columns after normalization:

`age, weight, height, gender, activity, goal, calories, protein, carb, fat`

Accepted aliases: `activityLevel`→`activity`, `healthGoal`→`goal`, `carbs`→`car`.

`gender`, `activity`, `goal` must use the **same string enums** as the Nest DTO (see `train.py`).

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service message |
| GET | `/health` | Liveness |
| POST | `/predict` | 4-field demo → `NutritionResponse` |
| POST | `/recommend` | **7-field production** → `GeminiHealthAnalysis` |
| POST | `/demo/recommend` | 4-field → nutrition + greedy meals |

## Environment

- `MODEL_PATH` — model file (default `model/model.pkl`)
- `FOODS_CSV_PATH` — foods CSV (default `data/food.csv`)
- `NUTRITION_DATASET_PATH` — training CSV for `train.py` (default `data/nutrition_dataset.csv`)
