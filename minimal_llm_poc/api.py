"""Direct Gemini API client with exponential backoff retry logic.

Preserves the retry logic for 5xx, 429, and timeouts, and uses the model
endpoints specified in config.py.
"""
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


_last_request_at = 0.0


def generate(
    prompt: str,
    system_prompt: str,
    model: str,
    *,
    judge: bool = False,
    mock: bool = False,
) -> Generation:
    """Generate content from Gemini models with rate pacing and exponential backoff."""
    global _last_request_at

    if mock:
        return Generation(
            text="MOCK_OUTPUT: SUCCESS",
            model_version=f"{model}-mock",
            prompt_tokens=len(prompt) // 4,
            completion_tokens=10,
            latency_seconds=0.01,
            finish_reason="STOP",
        )

    if model not in config.AVAILABLE_MODELS:
        raise ValueError(f"Model is not allowed: {model}")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is missing")

    max_tokens = config.JUDGE_MAX_TOKENS if judge else config.WORKER_MAX_TOKENS
    temperature = config.JUDGE_TEMPERATURE if judge else config.WORKER_TEMPERATURE

    generation_config = {
        "temperature": temperature,
        "topP": config.WORKER_TOP_P,
        "maxOutputTokens": max_tokens,
    }

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": generation_config,
    }

    url = config.GEMINI_API_URL.format(model=model)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()

    for attempt in range(config.API_MAX_RETRIES + 1):
        # Rate pacing
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
                delay = config.API_RETRY_BASE_SEC * (2 ** attempt)
                print(
                    f"\n[API retry] HTTP {exc.code} for {model} (attempt {attempt + 1}/{config.API_MAX_RETRIES}): "
                    f"waiting {delay:.1f}s... {detail[:80]}"
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini returned HTTP {exc.code}: {detail}") from None
        except TimeoutError:
            if attempt < config.API_MAX_RETRIES:
                delay = config.API_RETRY_BASE_SEC * (2 ** attempt)
                print(
                    f"\n[API retry] Timeout for {model} (attempt {attempt + 1}/{config.API_MAX_RETRIES}): "
                    f"waiting {delay:.1f}s..."
                )
                time.sleep(delay)
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
    model_version = str(body.get("modelVersion", model))
    usage = body.get("usageMetadata") or {}

    return Generation(
        text=text,
        model_version=model_version,
        prompt_tokens=int(usage.get("promptTokenCount", 0)),
        completion_tokens=int(usage.get("candidatesTokenCount", 0)),
        latency_seconds=time.perf_counter() - started,
        finish_reason=finish_reason,
    )
