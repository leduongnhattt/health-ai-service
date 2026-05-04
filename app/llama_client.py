from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

log = logging.getLogger("health_ai.llama")


@dataclass(frozen=True)
class LlamaConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float
    max_tokens: int
    temperature: float


def get_llama_config() -> LlamaConfig:
    base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    api_key = os.getenv("LLAMA_API_KEY", "").strip()
    model = os.getenv("LLAMA_MODEL", "local-gguf").strip() or "local-gguf"
    timeout_s = float(os.getenv("LLAMA_TIMEOUT_S", "60"))
    max_tokens = int(os.getenv("LLAMA_MAX_TOKENS", "1400"))
    temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.4"))
    return LlamaConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _auth_headers(api_key: str) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _try_resolve_model_id(*, client: httpx.Client, base_url: str) -> Optional[str]:
    """
    llama.cpp OpenAI server may require model id to match /v1/models.
    If user configured a mismatching LLAMA_MODEL, /v1/chat/completions can return errors.
    """
    try:
        res = client.get(f"{base_url}/v1/models")
        if not res.is_success:
            return None
        data = res.json()
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, dict) and isinstance(first.get("id"), str) and first["id"].strip():
                return first["id"].strip()
    except Exception:
        return None
    return None


def chat_completions(
    *,
    system: str,
    user: str,
    json_schema: Optional[Dict[str, Any]] = None,
    config: Optional[LlamaConfig] = None,
) -> str:
    """
    Call an OpenAI-compatible /v1/chat/completions endpoint exposed by llama.cpp server.
    Returns raw assistant content (string).
    """
    cfg = config or get_llama_config()
    url = f"{cfg.base_url}/v1/chat/completions"
    # Retry 503: llama.cpp returns 503 when model is still loading or server is temporarily unavailable.
    max_retries = int(os.getenv("LLAMA_RETRY_MAX", "6"))
    backoff_s = float(os.getenv("LLAMA_RETRY_BACKOFF_S", "1.0"))

    def build_payload(*, include_response_format: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if include_response_format and json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        return payload

    with httpx.Client(timeout=cfg.timeout_s, headers=_auth_headers(cfg.api_key)) as client:
        resolved_model_id: Optional[str] = None
        payload = build_payload(include_response_format=True)
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                res = client.post(url, json=payload)
                if not res.is_success and json_schema:
                    # Some OpenAI-compatible servers don't support response_format=json_schema.
                    body = res.text or ""
                    if res.status_code in (400, 404, 422) and (
                        "response_format" in body
                        or "json_schema" in body
                        or "unsupported" in body.lower()
                    ):
                        log.warning(
                            "llama.cpp endpoint rejected response_format; retrying without it (status=%s)",
                            res.status_code,
                        )
                        payload = build_payload(include_response_format=False)
                        res = client.post(url, json=payload)

                # If model id mismatches what llama.cpp exposes, attempt to auto-resolve once.
                if res.status_code in (400, 404, 503) and resolved_model_id is None:
                    body = (res.text or "").lower()
                    if "model" in body and ("not found" in body or "unknown" in body or "available" in body):
                        mid = _try_resolve_model_id(client=client, base_url=cfg.base_url)
                        if mid and mid != cfg.model:
                            resolved_model_id = mid
                            log.warning("Resolved llama model id via /v1/models: %s (was %s)", mid, cfg.model)
                            cfg = LlamaConfig(
                                base_url=cfg.base_url,
                                api_key=cfg.api_key,
                                model=mid,
                                timeout_s=cfg.timeout_s,
                                max_tokens=cfg.max_tokens,
                                temperature=cfg.temperature,
                            )
                            payload = build_payload(include_response_format=True)
                            continue

                if res.status_code == 503 and attempt < max_retries:
                    wait = backoff_s * (2**attempt)
                    log.warning(
                        "llama.cpp 503 Service Unavailable (attempt %d/%d); retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                res.raise_for_status()
                data = res.json()
                break
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    wait = backoff_s * (2**attempt)
                    log.warning(
                        "llama.cpp request failed (attempt %d/%d): %r; retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise
        else:
            # Shouldn't happen, but keep mypy happy.
            raise last_exc or RuntimeError("llama.cpp request failed")

    try:
        return str(data["choices"][0]["message"]["content"] or "")
    except Exception:
        return json.dumps(data)

