"""Small direct Gemini API client for the fixed pilot protocol."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Generation:
    text: str
    model_version: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    finish_reason: str = ""
    thoughts_tokens: int = 0
    total_tokens: int = 0


_last_request_at = 0.0


def generate(prompt: str, system_prompt: str, model: str, *, judge: bool = False,
             seed: int | None = None) -> Generation:
    """Generate content, retrying transient Gemini 5xx responses with backoff."""
    global _last_request_at
    if model not in config.ALLOWED_MODELS:
        raise ValueError(f"Model is not allowed by the fixed protocol: {model}")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    max_tokens = config.JUDGE_MAX_TOKENS if judge else config.WORKER_MAX_TOKENS
    generation_config = {
        "temperature": config.JUDGE_TEMPERATURE if judge else config.WORKER_TEMPERATURE,
        "topP": 1.0,
        "maxOutputTokens": max_tokens,
        "seed": seed,
    }
    if judge:
        generation_config["thinkingConfig"] = {
            "thinkingLevel": config.JUDGE_THINKING_LEVEL,
        }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        config.GEMINI_API_URL.format(model=model),
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    for attempt in range(config.API_MAX_RETRIES + 1):
        now = time.monotonic()
        wait = config.MIN_REQUEST_INTERVAL_SEC - (now - _last_request_at)
        if _last_request_at and wait > 0:
            time.sleep(wait)

        _last_request_at = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=config.API_TIMEOUT_SEC) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            if (str(exc.code).startswith("5") or exc.code == 429) and attempt < config.API_MAX_RETRIES:
                print(f"\nRetrying request after HTTP {exc.code}: {detail}\nAttempt {attempt + 1} of {config.API_MAX_RETRIES} for model {model}. Waiting {config.API_RETRY_BASE_SEC * (2 ** attempt):.1f}s...\n")
                time.sleep(config.API_RETRY_BASE_SEC * (2 ** attempt))
                continue
            raise RuntimeError(f"Gemini returned HTTP {exc.code}: {detail}") from None
        except TimeoutError:
            if attempt < config.API_MAX_RETRIES:
                print(f"\nRetrying request after timeout. Attempt {attempt + 1} of {config.API_MAX_RETRIES} for model {model}. Waiting {config.API_RETRY_BASE_SEC * (2 ** attempt):.1f}s...\n")
                time.sleep(config.API_RETRY_BASE_SEC * (2 ** attempt))
                continue
            raise RuntimeError("Gemini request timed out") from None
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Gemini request failed: {type(exc).__name__}: {exc}") from None
    else:
        raise RuntimeError("Gemini request exhausted retry attempts")

    candidates = body.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    parts = (candidate.get("content") or {}).get("parts", []) if candidate else []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    finish_reason = str(candidate.get("finishReason", "")) if candidate else ""
    if not text and not judge:
        raise RuntimeError("Gemini response did not contain generated text")
    model_version = str(body.get("modelVersion", ""))
    if model_version and model_version != model and not model_version.startswith(model + "-"):
        raise RuntimeError(f"Requested {model}, but Gemini reported {model_version}")
    usage = body.get("usageMetadata") or {}
    return Generation(
        text=text,
        model_version=model_version,
        prompt_tokens=int(usage.get("promptTokenCount", 0)),
        completion_tokens=int(usage.get("candidatesTokenCount", 0)),
        latency_seconds=time.perf_counter() - started,
        finish_reason=finish_reason,
        thoughts_tokens=int(usage.get("thoughtsTokenCount", 0)),
        total_tokens=int(usage.get("totalTokenCount", 0)),
    )
