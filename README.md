# Health AI Service (FastAPI)

Structure:

```text
health-ai-service/
├── model/
│   └── model.pkl              # produced by train.py (gitignored)
├── models/
│   └── model.gguf             # llama.cpp GGUF (optional; used by docker-compose.local-llm.yml)
├── data/
│   ├── food.csv               # foods for meal optimizer
│   ├── food.v2.sample.csv      # sample food catalog v2 (optimizer/constraints)
│   ├── nutrition_dataset.csv  # generated or your real training CSV
│   └── nutrition_dataset.example.csv
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── services.py
├── train.py
└── requirements.txt
```

## API contract (Nest / Next health profile)

**Request** — `POST /recommend` body (matches `food-delivery-server` `HealthProfileDto` and `food-delivery-app` `HealthProfile`):

- `age`, `gender`, `height`, `weight`, `activityLevel`, `healthGoal`, `dietaryRestrictions`

**Response** — `GeminiHealthAnalysis` shape (see `app/schemas.py` and `gemini-health-ai.service.ts`).

## Local LLM (llama.cpp), optional

`/recommend` is **always** served from the trained regressor (`model.pkl`) plus the food catalog (`food.csv`, or v2 when `USE_FOOD_V2=1`). No cloud LLM.

If you enable `USE_FOOD_V2=1`, you may set `USE_LOCAL_LLM=1` so `restrictions.py` can optionally call a local llama.cpp server (`POST /v1/chat/completions`) to parse free-text dietary notes into tags/allergens. Rule-based parsing still applies when the server is down.

- `USE_LOCAL_LLM` — default `0`
- `LLAMA_BASE_URL` — default `http://127.0.0.1:8080`
- `LLAMA_API_KEY`, `LLAMA_MODEL`, `LLAMA_TIMEOUT_S`, `LLAMA_MAX_TOKENS`, `LLAMA_TEMPERATURE`

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

> Fix: alias đúng là `carbs`→`carb`.

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run with Docker Compose (public FastAPI, private model server)

1) Put your GGUF model at:

- `health-ai-service/models/model.gguf`

2) Start:

```bash
docker compose -f docker-compose.local-llm.yml up --build
```

This publishes only:

- `health-ai` on port `8000`

The `llama` service is private to the compose network (no host port published).

## Kaggle train (export `model.pkl`)

Xem hướng dẫn chi tiết:

- `KAGGLE_TRAIN.md`

## Smoke test

Sau khi chạy service, bạn có thể smoke test `/recommend`:

```bash
python scripts/smoke_test_recommend.py http://127.0.0.1:8000
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
- `USE_FOOD_V2` — `1` to enable v2 optimizer path (default off)
- `FOODS_V2_CSV_PATH` — food v2 CSV path (default `data/food.v2.csv`)
- `NUTRITION_DATASET_PATH` — training CSV for `train.py` (default `data/nutrition_dataset.csv`)

## Sample data

- `data/food.sample.csv` — a larger example food catalog (copy to `data/food.csv` if desired)
- `data/health_sft_train.sample.jsonl` — one example SFT record (instruction → JSON response)
- `scripts/generate_sft_dataset.py` — generate synthetic SFT JSONL for fine-tuning
