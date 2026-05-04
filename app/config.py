from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    foods_csv_path: Path

    use_local_llm: bool
    locale: str
    log_level: str

    llama_base_url: str
    llama_api_key: str
    llama_model: str
    llama_timeout_s: float
    llama_max_tokens: int
    llama_temperature: float


def load_config(*, repo_root: Path) -> AppConfig:
    # Best-effort: load health-ai-service/.env without requiring python-dotenv.
    # Repo root .gitignore already ignores `.env` files.
    env_path = repo_root / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            pass

    model_path = Path(os.getenv("MODEL_PATH", "").strip() or (repo_root / "model" / "model.pkl"))
    foods_csv_path = Path(
        os.getenv("FOODS_CSV_PATH", "").strip() or (repo_root / "data" / "food.csv")
    )

    use_local_llm = _truthy(os.getenv("USE_LOCAL_LLM", "0"))
    locale = (os.getenv("LOCALE", "vi") or "vi").strip()
    log_level = (os.getenv("LOG_LEVEL", "INFO") or "INFO").strip().upper()

    llama_base_url = (os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080") or "").strip().rstrip(
        "/"
    )
    llama_api_key = (os.getenv("LLAMA_API_KEY", "") or "").strip()
    llama_model = (os.getenv("LLAMA_MODEL", "local-gguf") or "local-gguf").strip()
    llama_timeout_s = float(os.getenv("LLAMA_TIMEOUT_S", "60"))
    llama_max_tokens = int(os.getenv("LLAMA_MAX_TOKENS", "1400"))
    llama_temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.4"))

    return AppConfig(
        model_path=model_path,
        foods_csv_path=foods_csv_path,
        use_local_llm=use_local_llm,
        locale=locale,
        log_level=log_level,
        llama_base_url=llama_base_url,
        llama_api_key=llama_api_key,
        llama_model=llama_model,
        llama_timeout_s=llama_timeout_s,
        llama_max_tokens=llama_max_tokens,
        llama_temperature=llama_temperature,
    )


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def validate_config(cfg: AppConfig, *, logger: Optional[logging.Logger] = None) -> None:
    log = logger or logging.getLogger("health_ai.config")

    if cfg.model_path.exists():
        log.info("ML model enabled (model.pkl found): %s", str(cfg.model_path))
    else:
        log.info("ML model missing -> fallback baseline: %s", str(cfg.model_path))

    if cfg.foods_csv_path.exists():
        log.info("Foods CSV: %s", str(cfg.foods_csv_path))
    else:
        log.warning("Foods CSV missing; meal plan may be empty: %s", str(cfg.foods_csv_path))

    if cfg.use_local_llm:
        if not cfg.llama_base_url:
            raise ValueError("USE_LOCAL_LLM=1 but LLAMA_BASE_URL is empty")
        log.info("Local LLM enabled (llama.cpp): %s", cfg.llama_base_url)
    else:
        log.info("Local LLM disabled (USE_LOCAL_LLM=0)")

