# GGUF + llama.cpp guide (health-ai-service)

Mục tiêu: bật **LLM enrichment** cho `/recommend` bằng **llama.cpp server** và một file **GGUF**.

Trong repo, Docker Compose đã có sẵn:

- `health-ai-service/docker-compose.local-llm.yml`

## 1) Bạn cần tải file GGUF (LLM pretrained)

GGUF là file model để llama.cpp chạy inference. Repo này **không train GGUF**.

### Khuyến nghị cho máy 16GB

- Ưu tiên **3B/4B Instruct** + quant **`Q4_K_M`** (nhẹ, ít lag hơn 7B).
- Nếu bạn dùng **7B**: vẫn chọn **`Q4_K_M`**, và nên giảm `LLAMA_MAX_TOKENS` xuống 400–800 để đỡ chậm.

### Repo GGUF gợi ý (đúng loại Instruct)

- Qwen2.5-7B-Instruct (GGUF): `bartowski/Qwen2.5-7B-Instruct-GGUF`
  - File khuyến nghị: `Qwen2.5-7B-Instruct-Q4_K_M.gguf`

### Đặt đúng vị trí và tên file

Tạo thư mục:

- `health-ai-service/models/`

Sau khi tải xong, đặt file tại:

- `health-ai-service/models/model.gguf`

> Compose mount `./models:/models:ro` và llama.cpp đọc `/models/model.gguf`.

## 2) Bật llama.cpp bằng Docker Compose

Trong `health-ai-service`:

```bash
docker compose -f docker-compose.local-llm.yml up --build
```

Compose này sẽ chạy:

- `llama` (private, không publish ra host)
- `health-ai` (publish `8000:8000`)

## 3) Env liên quan (đã set sẵn trong compose)

Trong `docker-compose.local-llm.yml`, service `health-ai` đã set:

- `USE_LOCAL_LLM=1`
- `LLAMA_BASE_URL=http://llama:8080`
- `LLAMA_MODEL=local-gguf`
- `LLAMA_TIMEOUT_S`, `LLAMA_MAX_TOKENS`, `LLAMA_TEMPERATURE`
- `LOCALE=vi`

Tune để đỡ nặng:

- `LLAMA_MAX_TOKENS`: 400–800 (khuyến nghị)

## 4) Smoke test nhanh

1) Chạy compose.\n+2) Gọi smoke test:

```bash
python scripts/smoke_test_recommend.py http://127.0.0.1:8000
```

Nếu llama.cpp lỗi/timeout/không parse được JSON, API sẽ fallback về rule-based (không crash).

