"""Layer 3: LLM workers (OpenRouter live or deterministic mock) + rubric judge.

Worker protocol per (task, slot):
- the task prompt is filled from its template with the episode input and, for
  dependency tasks, the upstream task's recorded answer (same-slot co-execution
  is what the energy term rewards);
- one API call (retries per design), usage + latency captured;
- invalid/failed output scores 0 on the rubric (a worker metric, not silently
  dropped).

Judge: rubric-anchored LLM scoring (0..len(rubric)-1), normalized to 0..1.
When judge_enabled is False (or worker_mode is mock with --no-judge), a
deterministic rubric proxy is used instead so that the whole pipeline can be
tested end-to-end without API access. The proxy is clearly flagged in outputs
as 'scorer: rubric_proxy'.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import random
import email.utils
from datetime import datetime, timezone
from pathlib import Path
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import torch

from . import config
from .tasks import AgentSlot, EpisodeTasks, TaskSpec

# ---------------------------------------------------------------------------
# API plumbing (plain urllib, matching the repo's existing convention)
# ---------------------------------------------------------------------------
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""
    endpoint_model: str = ""
    routing_attempt: int = 0
    router_strategy: str = ""
    router_region: str = ""
    provider_is_byok: bool = False
    generation_id: str = ""
    finish_reason: str = ""


@dataclass
class WorkerOutput:
    task_id: int
    agent_id: int
    ok: bool
    text: str
    error: str | None
    usage: Usage
    latency_first_token_sec: float
    latency_total_sec: float
    retries: int


_LAST_REQUEST_AT = 0.0
_PROGRESS_LOG_PATH = None
_PROGRESS_CALL_COUNT = 0
_PROGRESS_CONTEXT: dict = {}
_REQUEST_ATTEMPT_COUNTS: dict[str, int] = {}
_COMPLETED_REQUESTS: dict[str, dict] = {}
_REQUEST_LAST_EVENTS: dict[str, dict] = {}
_CONTEXT_REQUEST_KEYS: dict[str, set[str]] = {}
_MAX_PROGRESS_ATTEMPTS = config.MAX_LIVE_API_CALLS
_RESUME_LIVE_RUN = False


def set_progress_log(path, resume: bool = False,
                     max_attempts: int = config.MAX_LIVE_API_CALLS) -> None:
    """Enable/reload a journal, recover completed tasks, and enforce a total cap."""
    global _PROGRESS_LOG_PATH, _PROGRESS_CALL_COUNT, _RESUME_LIVE_RUN
    global _REQUEST_ATTEMPT_COUNTS, _COMPLETED_REQUESTS, _MAX_PROGRESS_ATTEMPTS
    global _REQUEST_LAST_EVENTS, _CONTEXT_REQUEST_KEYS
    if not 1 <= max_attempts <= config.MAX_LIVE_API_CALLS:
        raise ValueError(f"max_attempts must be between 1 and {config.MAX_LIVE_API_CALLS}")
    _RESUME_LIVE_RUN = bool(resume)
    set_progress_context()
    _MAX_PROGRESS_ATTEMPTS = max_attempts
    _PROGRESS_LOG_PATH = path
    _PROGRESS_CALL_COUNT = 0
    _REQUEST_ATTEMPT_COUNTS = {}
    _COMPLETED_REQUESTS = {}
    _REQUEST_LAST_EVENTS = {}
    _CONTEXT_REQUEST_KEYS = {}
    if path is None:
        _MAX_PROGRESS_ATTEMPTS = config.MAX_LIVE_API_CALLS
        _RESUME_LIVE_RUN = False
        return
    journal_path = Path(path)
    if not journal_path.exists():
        return
    with journal_path.open("r+b") as journal:
        journal.seek(0, 2)
        end = journal.tell()
        journal.seek(0)
        content = journal.read(end)
        valid_end = content.rfind(b"\n") + 1
        if valid_end < len(content):
            # A crash can leave a partial trailing JSONL write; it never began a request.
            journal.seek(valid_end)
            journal.truncate()
            journal.flush()
        lines = content[:valid_end].decode("utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid progress journal at line {line_number}") from exc
            request_key = record.get("request_key")
            event = record.get("event")
            if event == "request_started":
                _PROGRESS_CALL_COUNT += 1
                if request_key:
                    _REQUEST_ATTEMPT_COUNTS[request_key] = (
                        _REQUEST_ATTEMPT_COUNTS.get(request_key, 0) + 1)
                    context_key = json.dumps(record.get("context", {}), sort_keys=True)
                    _CONTEXT_REQUEST_KEYS.setdefault(context_key, set()).add(request_key)
                    _REQUEST_LAST_EVENTS[request_key] = {"event": event}
            elif event == "request_failed" and request_key:
                _REQUEST_LAST_EVENTS[request_key] = record
            elif event == "retry_aborted" and request_key:
                _REQUEST_LAST_EVENTS[request_key] = {
                    "event": "retry_aborted",
                    "retryable": False,
                    "will_retry": False,
                }
            elif event == "retry_scheduled" and request_key:
                _REQUEST_LAST_EVENTS[request_key] = {
                    "event": "request_failed",
                    "retryable": True,
                    "will_retry": True,
                    "attempt": record.get("failed_attempt"),
                }
            elif event == "request_completed" and request_key:
                _COMPLETED_REQUESTS[request_key] = record
                _REQUEST_LAST_EVENTS[request_key] = record
        if resume:
            _write_progress("run_resumed", prior_http_attempts=_PROGRESS_CALL_COUNT,
                            total_http_attempt_cap=_MAX_PROGRESS_ATTEMPTS,
                            timeout_sec=config.API_TIMEOUT_SEC,
                            extra_attempts_for_exhausted_transient_request=(
                                config.MAX_RESUME_EXTRA_ATTEMPTS))


def set_progress_context(**context) -> None:
    """Attach experiment coordinates to the next synchronous live request."""
    global _PROGRESS_CONTEXT
    _PROGRESS_CONTEXT = context


def _write_progress(event: str, **fields) -> None:
    if _PROGRESS_LOG_PATH is None:
        return
    record = {
        "event": event,
        "request_number": _PROGRESS_CALL_COUNT,
        "context": _PROGRESS_CONTEXT,
        **fields,
    }
    with open(_PROGRESS_LOG_PATH, "a", encoding="utf-8") as journal:
        journal.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        journal.flush()


def _request_fingerprint(model: str, provider: str, payload: dict) -> str:
    identity = {
        "model": model,
        "provider": provider,
        "payload": payload,
        # Matched prompts across conditions are still distinct experimental calls.
        "context": _PROGRESS_CONTEXT,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _backoff_delay(failed_attempt: int) -> float:
    base = min(config.RETRY_MAX_SEC,
               config.RETRY_BASE_SEC * (2 ** max(0, failed_attempt - 1)))
    jitter = random.uniform(1.0 - config.RETRY_JITTER_RATIO,
                            1.0 + config.RETRY_JITTER_RATIO)
    return min(config.RETRY_MAX_SEC, base * jitter)


class LiveWorkerFailure(RuntimeError):
    """A live request failed; halt immediately to avoid further quota use."""


def call_gemini(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: float = config.API_TIMEOUT_SEC,
    max_retries: int = config.MAX_RETRIES,
    seed: int | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    provider: str | None = None,
) -> tuple[str | None, Usage, float, float, int, str | None]:
    """Call Gemini directly, retry transient failures, and journal each attempt."""
    if model != config.DEFAULT_MODEL:
        return None, Usage(), 0.0, 0.0, 0, f"model_mismatch: only {config.DEFAULT_MODEL} is allowed"
    if provider not in (None, config.MODEL_PROVIDER):
        return None, Usage(), 0.0, 0.0, 0, f"provider_mismatch: only {config.MODEL_PROVIDER} is allowed"
    retry_limit = config.MAX_RETRIES
    if not 0 <= max_retries <= retry_limit:
        return None, Usage(), 0.0, 0.0, 0, f"Gemini pilot allows at most {retry_limit} retries"

    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": temperature},
    }
    generation = payload["generationConfig"]
    if top_p is not None:
        generation["topP"] = top_p
    if max_tokens is not None:
        generation["maxOutputTokens"] = max_tokens
    if seed is not None:
        generation["seed"] = seed

    url = config.GEMINI_API_URL.format(model=model)
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    request_key = _request_fingerprint(model, provider or config.MODEL_PROVIDER, payload)
    context_key = json.dumps(_PROGRESS_CONTEXT, sort_keys=True)
    if (_RESUME_LIVE_RUN and context_key in _CONTEXT_REQUEST_KEYS and
            request_key not in _CONTEXT_REQUEST_KEYS[context_key]):
        error = "resume_request_mismatch: inputs or request parameters changed for an attempted task"
        _write_progress("resume_request_rejected", request_key=request_key, error=error)
        return None, Usage(), 0.0, 0.0, 0, error
    cached = (_COMPLETED_REQUESTS.get(request_key)
             if _PROGRESS_LOG_PATH is not None else None)
    if cached is not None:
        _write_progress("request_reused", request_key=request_key,
                        prior_attempts=_REQUEST_ATTEMPT_COUNTS.get(request_key, 0))
        usage = Usage(
            prompt_tokens=int(cached.get("prompt_tokens", 0)),
            completion_tokens=int(cached.get("completion_tokens", 0)),
            total_tokens=int(cached.get("total_tokens", 0)),
            model=str(cached.get("model_version", "")),
            provider=config.MODEL_PROVIDER,
            endpoint_model=model,
            finish_reason=str(cached.get("finish_reason", "")),
        )
        request_latency = float(cached.get("latency_sec", 0.0))
        total_latency = float(cached.get("total_elapsed_sec", request_latency))
        return str(cached["response_text"]), usage, total_latency, request_latency, max(
            0, _REQUEST_ATTEMPT_COUNTS.get(request_key, 1) - 1), None

    attempts_already_made = (_REQUEST_ATTEMPT_COUNTS.get(request_key, 0)
                             if _PROGRESS_LOG_PATH is not None else 0)
    if (attempts_already_made and not _RESUME_LIVE_RUN and
            _REQUEST_LAST_EVENTS.get(request_key, {}).get("event") in
            {"request_failed", "retry_aborted"}):
        return (None, Usage(), 0.0, 0.0, attempts_already_made - 1,
                "prior_failure: explicit --resume is required before reattempting this task")

    global _LAST_REQUEST_AT, _PROGRESS_CALL_COUNT
    started_at = time.perf_counter()
    last_event = _REQUEST_LAST_EVENTS.get(request_key, {})
    if (_RESUME_LIVE_RUN and attempts_already_made and
            last_event.get("event") == "request_failed" and
            last_event.get("retryable") is not True):
        return (None, Usage(), 0.0, 0.0, attempts_already_made - 1,
                "resume_stopped: last failure was non-retryable")
    if (_RESUME_LIVE_RUN and attempts_already_made and
            last_event.get("event") == "request_failed" and
            last_event.get("retryable") is True and
            last_event.get("will_retry") is False and
            attempts_already_made < config.MAX_LIVE_REQUEST_ATTEMPTS):
        return (None, Usage(), 0.0, 0.0, attempts_already_made - 1,
                "resume_stopped: retry ended before the configured transient retry budget")
    if _RESUME_LIVE_RUN and last_event.get("event") == "retry_aborted":
        return (None, Usage(), 0.0, 0.0, max(0, attempts_already_made - 1),
                "resume_stopped: retry was aborted by its safety bound")
    max_total_attempts = max_retries + 1
    resume_extension_eligible = (
        last_event.get("event") == "request_started" or
        (last_event.get("event") == "request_failed" and
         last_event.get("retryable") is True and
         (last_event.get("will_retry") is True or
          attempts_already_made >= config.MAX_LIVE_REQUEST_ATTEMPTS))
    )
    if (_RESUME_LIVE_RUN and
            config.MAX_LIVE_REQUEST_ATTEMPTS <= attempts_already_made <
            config.MAX_RESUME_REQUEST_ATTEMPTS and
            resume_extension_eligible):
        max_total_attempts = config.MAX_RESUME_REQUEST_ATTEMPTS
    if max_total_attempts < 1:
        return None, Usage(), 0.0, 0.0, 0, "max_attempts must be positive"
    if attempts_already_made >= max_total_attempts:
        return (None, Usage(), 0.0, 0.0, attempts_already_made - 1,
                "retry_exhausted: this exact request already used its retry budget")
    attempt = attempts_already_made
    delay_before_next = _backoff_delay(attempt) if attempt > 0 else None
    resumed = attempt > 0
    while attempt < max_total_attempts:
        if (_PROGRESS_LOG_PATH is not None and
                _PROGRESS_CALL_COUNT >= _MAX_PROGRESS_ATTEMPTS):
            _write_progress("attempt_cap_reached", request_key=request_key,
                            hard_cap=_MAX_PROGRESS_ATTEMPTS)
            return (None, Usage(), time.perf_counter() - started_at, 0.0, attempt,
                    f"live HTTP attempt cap reached ({_MAX_PROGRESS_ATTEMPTS})")
        if delay_before_next is not None:
            _write_progress("retry_scheduled", request_key=request_key,
                            failed_attempt=attempt, delay_sec=delay_before_next,
                            resumed=resumed)
            time.sleep(delay_before_next)
            delay_before_next = None
            resumed = False
        wait = config.MIN_REQUEST_INTERVAL_SEC - (time.monotonic() - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()
        request_started = time.perf_counter()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        if _PROGRESS_LOG_PATH is not None:
            _PROGRESS_CALL_COUNT += 1
            _REQUEST_ATTEMPT_COUNTS[request_key] = attempt + 1
            _REQUEST_LAST_EVENTS[request_key] = {"event": "request_started"}
            context_key = json.dumps(_PROGRESS_CONTEXT, sort_keys=True)
            _CONTEXT_REQUEST_KEYS.setdefault(context_key, set()).add(request_key)
            _write_progress("request_started", model=model, attempt=attempt + 1,
                            request_key=request_key,
                            run_request_attempt=attempt + 1)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            request_latency = time.perf_counter() - request_started
            candidate = (data.get("candidates") or [{}])[0]
            parts = (candidate.get("content") or {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            if not text:
                raise ValueError("API response did not contain generated text")
            model_version = str(data.get("modelVersion", ""))
            usage_data = data.get("usageMetadata", {}) or {}
            usage = Usage(
                prompt_tokens=int(usage_data.get("promptTokenCount", 0)),
                completion_tokens=int(usage_data.get("candidatesTokenCount", 0)),
                total_tokens=int(usage_data.get("totalTokenCount", 0)),
                model=model_version,
                provider=config.MODEL_PROVIDER,
                endpoint_model=model,
                finish_reason=str(candidate.get("finishReason", "")),
            )
            if model_version != model and not model_version.startswith(f"{model}-"):
                error = f"model_mismatch: requested {model}, got modelVersion={model_version!r}"
                _write_progress("request_failed", error=error, model_version=model_version,
                                attempt=attempt + 1, retryable=False, will_retry=False)
                return None, usage, time.perf_counter() - started_at, request_latency, attempt, error
            total_elapsed = time.perf_counter() - started_at
            completion = {
                "model_version": model_version,
                "response_text": text,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "finish_reason": usage.finish_reason,
                "latency_sec": request_latency,
                "total_elapsed_sec": total_elapsed,
            }
            if _PROGRESS_LOG_PATH is not None:
                _COMPLETED_REQUESTS[request_key] = completion
            _write_progress(
                "request_completed", request_key=request_key, **completion,
                attempt=attempt + 1,
            )
            return text, usage, time.perf_counter() - started_at, request_latency, attempt, None
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                detail = ""
            error = f"http_{exc.code}: {exc.reason} {detail}"
            # 429 may mean a quota is exhausted; do not issue another request.
            retryable = exc.code in {408, 500, 502, 503, 504}
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After")
                                               if exc.headers else None)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            retryable = True
            retry_after = None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            retryable = False
            retry_after = None

        if api_key:
            error = error.replace(api_key, "[REDACTED]")
        elapsed = time.perf_counter() - request_started
        attempts_made = attempt + 1
        retry_after_too_long = (retry_after is not None and
                                retry_after > config.MAX_RETRY_AFTER_SEC)
        will_retry = (retryable and attempts_made < max_total_attempts and
                      not retry_after_too_long)
        failure_record = {
            "event": "request_failed",
            "retryable": retryable,
            "will_retry": will_retry,
            "attempt": attempts_made,
        }
        _REQUEST_LAST_EVENTS[request_key] = failure_record
        _write_progress("request_failed", request_key=request_key, error=error,
                        latency_sec=elapsed, attempt=attempts_made,
                        retryable=retryable, will_retry=will_retry,
                        retry_after_sec=retry_after)
        if not will_retry:
            if retry_after_too_long:
                error += f"; Retry-After {retry_after:.1f}s exceeds configured bound"
            return None, Usage(), time.perf_counter() - started_at, elapsed, attempt, error
        backoff = _backoff_delay(attempts_made)
        if retry_after is not None:
            backoff = max(backoff, retry_after)
        if backoff > config.MAX_RETRY_AFTER_SEC:
            error += f"; computed retry wait {backoff:.1f}s exceeds configured bound"
            _REQUEST_LAST_EVENTS[request_key] = {"event": "retry_aborted"}
            _write_progress("retry_aborted", request_key=request_key,
                            failed_attempt=attempts_made, delay_sec=backoff,
                            reason="retry wait exceeds configured bound")
            return None, Usage(), time.perf_counter() - started_at, elapsed, attempt, error
        failure_record["will_retry"] = True
        _REQUEST_LAST_EVENTS[request_key] = failure_record
        attempt += 1
        delay_before_next = backoff
    raise AssertionError("unreachable retry loop")


def probe_gemini(api_key: str) -> tuple[str | None, Usage, str | None]:
    """Minimal single generation used only to verify model/key reachability."""
    text, usage, _, _, _, error = call_gemini(
        config.PROBE_PROMPT, config.PROBE_SYSTEM_PROMPT, config.DEFAULT_MODEL,
        api_key, temperature=0.0, max_retries=0, seed=0, max_tokens=8, top_p=1.0,
    )
    return text, usage, error


def _call_model(*args, **kwargs):
    """Compatibility name for the shared worker/judge path and test seam."""
    return call_openrouter(*args, **kwargs)


# Preserve the old import name for existing downstream PoC scripts.
call_openrouter = call_gemini


# ---------------------------------------------------------------------------
# Deterministic mock workers (pipeline validation; NOT evidence about LLMs)
# ---------------------------------------------------------------------------
_MOCK_QUALITY = {  # slot_id -> base quality for on-specialization tasks
    0: 0.80, 1: 0.78, 2: 0.74,
}


def mock_worker(task: TaskSpec, slot: AgentSlot, prompt: str, rng,
                dep_available: bool = True) -> WorkerOutput:
    """Deterministic pseudo-worker.

    Quality model (design-time, fixed):
    - affinity between task embedding and slot capability (0..1, mean over dims)
      scales the base quality, so cross-functional dependency tasks are genuinely
      competitive on more than one slot;
    - a missing upstream dependency costs one rubric level for dependent tasks
      (coordination failure is visible, as in real pipelines);
    - difficulty reduces quality; a deterministic hash jitter keeps runs
      reproducible without being degenerate.
    """
    cap = torch.tensor(slot.capability).float()
    tmb = torch.tensor(task.embedding).float()
    affinity = float((cap * tmb).sum())  # both normalized to sum 1 -> cosine-like overlap
    base = 0.30 + 0.55 * affinity       # 0.30 (no overlap) .. 0.85 (perfect match)
    quality = base - 0.35 * task.difficulty
    if task.depends_on is not None and not dep_available:
        quality -= 0.45                 # coordination failure: lose ~1 rubric level
    quality = max(0.0, min(1.0, quality))
    digest = hashlib.sha256(f"{task.id}|{slot.id}|{prompt}".encode("utf-8")).digest()
    h = (int.from_bytes(digest[:4], "big") % 1000) / 1000.0
    q = quality + (h - 0.5) * 0.12
    score_level = 2 if q > 0.62 else (1 if q > 0.30 else 0)
    if score_level == 2:
        text = task.reference_answer
    elif score_level == 1:
        text = f"(approximate) {task.reference_answer}"
    else:
        text = "unable to determine"
    return WorkerOutput(
        task_id=task.id, agent_id=slot.id, ok=True, text=text, error=None,
        usage=Usage(prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4,
                    total_tokens=(len(prompt) + len(text)) // 4),
        latency_first_token_sec=0.0, latency_total_sec=0.0, retries=0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class ScoredTask:
    task_id: int
    agent_id: int
    rubric_score: float          # normalized 0..1
    scorer: str                  # "judge:<model>" | "rubric_proxy"
    judge_raw: int | None
    judge_error: str | None = None
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_model: str = ""
    judge_provider: str = ""
    spotcheck_score: float | None = None
    spotcheck_scorer: str | None = None


def build_task_prompt(task: TaskSpec, episode_tasks: EpisodeTasks,
                      dep_answer: str | None) -> str:
    text = task.prompt_template
    if task.depends_on is not None:
        text = text.replace("{dep}", (dep_answer or "[upstream output unavailable]").strip()[:1200])
    return text.replace("{input}", episode_tasks.inputs[task.id] if task.id < len(
        episode_tasks.inputs) else "")


JUDGE_SYSTEM = (
    "You are a strict, consistent grader. Compare the candidate answer against the "
    "rubric levels and output ONLY a single integer: the index (0-based) of the "
    "highest rubric level the answer fully satisfies. Output nothing else."
)


def judge_score(task: TaskSpec, candidate: str, model: str, api_key: str,
                temperature: float, seed: int | None = None,
                max_tokens: int = config.MAX_JUDGE_TOKENS,
                provider: str = config.MODEL_PROVIDER
                ) -> tuple[int | None, str | None, Usage]:
    candidate = candidate[:config.MAX_WORKER_OUTPUT_CHARS]
    rubric_text = "\n".join(f"{i}: {anchor}" for i, anchor in enumerate(task.rubric))
    prompt = (
        f"Task: {task.prompt_template.split('{')[0].strip()}\n\n"
        f"Reference-correct behaviour: {task.reference_answer}\n\n"
        f"Rubric levels:\n{rubric_text}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        "Which rubric level (integer index) does the candidate fully satisfy?"
    )
    text, usage, _, _, _, err = _call_model(
        prompt, JUDGE_SYSTEM, model, api_key, temperature=temperature,
        seed=seed, max_tokens=max_tokens, top_p=config.WORKER_TOP_P,
        provider=provider)
    if text is None:
        return None, err, usage
    text = text[:config.MAX_JUDGE_OUTPUT_CHARS]
    match = re.fullmatch(r"\s*(\d+)\s*", text)
    if not match:
        return None, f"unparseable judge output: {text[:120]}", usage
    level = int(match.group(1))
    return (level if 0 <= level < len(task.rubric) else None,
            None if 0 <= level < len(task.rubric) else f"judge level out of range: {level}",
            usage)


def rubric_proxy_score(task: TaskSpec, candidate: str, worker_ok: bool) -> int:
    """Deterministic stand-in for the judge used ONLY in mock/no-judge mode."""
    if not worker_ok or not candidate:
        return 0
    if candidate.strip() == task.reference_answer.strip():
        return len(task.rubric) - 1
    if task.reference_answer.lower() in candidate.lower():
        return max(1, len(task.rubric) - 2)
    return 0


# ---------------------------------------------------------------------------
# Episode execution (Layer 3)
# ---------------------------------------------------------------------------
@dataclass
class EpisodeExecution:
    outputs: list[WorkerOutput]
    scores: list[ScoredTask]
    judge_usage: Usage = field(default_factory=Usage)
    judge_calls: int = 0
    judge_failures: int = 0


def derive_request_seed(base_seed: int, block: str, episode: int,
                        repetition: int, task_id: int, role: str) -> int:
    """Stable matched seed independent of condition and Python hash randomization."""
    key = (f"{base_seed}|{block}|{episode}|{repetition}|{task_id}|{role}")
    raw = hashlib.sha256(key.encode("utf-8")).digest()[:4]
    return int.from_bytes(raw, "big") & 0x7FFFFFFF


def execute_episode(
    episode_tasks: EpisodeTasks,
    slots: list[AgentSlot],
    X,                      # (N, M) assignment tensor from Layer 1/2
    cfg: config.PoCConfig,
    repetition: int,
    episode_seed: int,
    condition: str = "",
    block: str | None = None,
) -> EpisodeExecution:
    """Execute all assigned tasks with Layer-3 workers and score the outputs.

    Dependency tasks receive the upstream answer ONLY if both tasks were assigned
    to the same agent (co-execution is the mechanism the interaction term rewards);
    otherwise the downstream task runs with the dependency marked unavailable --
    mirroring the coordination hypothesis under test.
    """
    import numpy as np
    import torch

    Xf = X.float()
    assigned_agent = {t: int(torch.argmax(Xf[:, t]).item()) for t in range(Xf.shape[1])}
    outputs: list[WorkerOutput] = []
    raw_by_task: dict[int, str] = {}
    # First pass: non-dependent tasks (dependencies can only consume completed output).
    # Dependent tasks run after their upstream task within the same episode pass.
    ordered = sorted(episode_tasks.specs, key=lambda t: (t.depends_on is not None, t.id))
    for task in ordered:
        slot = slots[assigned_agent[task.id]]
        dep_answer = None
        if task.depends_on is not None:
            if assigned_agent[task.depends_on] == assigned_agent[task.id] \
                    and task.depends_on in raw_by_task:
                dep_answer = raw_by_task[task.depends_on]
        dep_available = dep_answer is not None
        prompt = build_task_prompt(task, episode_tasks, dep_answer)
        if cfg.worker_mode == "mock":
            rng = np.random.default_rng(derive_request_seed(
                cfg.seed, block or episode_tasks.block, episode_tasks.episode,
                repetition, task.id, "worker"))
            out = mock_worker(task, slot, prompt, rng, dep_available=dep_available)
        else:
            set_progress_context(condition=condition, block=block or episode_tasks.block,
                                 episode=episode_tasks.episode, repetition=repetition,
                                 task_id=task.id)
            text, usage, t_first, t_total, retries, err = _call_model(
                prompt, slot.system_prompt, cfg.model, cfg.api_key,
                temperature=cfg.worker_temperature, max_retries=cfg.max_retries,
                seed=(derive_request_seed(cfg.seed, block or episode_tasks.block,
                                         episode_tasks.episode, repetition, task.id, "worker")
                      if cfg.worker_seed else None),
                max_tokens=cfg.max_worker_tokens, top_p=cfg.worker_top_p,
                provider=cfg.provider)
            if text is not None:
                text = text[:cfg.max_worker_output_chars]
            out = WorkerOutput(
                task_id=task.id, agent_id=slot.id, ok=text is not None,
                text=text or "", error=err, usage=usage,
                latency_first_token_sec=t_first, latency_total_sec=t_total,
                retries=retries)
            if not out.ok:
                raise LiveWorkerFailure(
                    f"live request failed; stopping immediately after task {task.id}: {out.error}"
                )
        if out.ok and out.text:
            raw_by_task[task.id] = out.text
        outputs.append(out)

    # Execution order is changed to satisfy dependencies; restore task-id order
    # before pairing outputs with the canonical task list for scoring and logging.
    outputs.sort(key=lambda out: out.task_id)

    # Scoring
    scores: list[ScoredTask] = []
    judge_usage_total = Usage()
    judge_calls = 0
    judge_failures = 0
    scorer = "rubric_proxy" if not cfg.judge_enabled else f"judge:{cfg.judge_model}"
    spotcheck_rng = np.random.default_rng(derive_request_seed(
        cfg.seed, block or episode_tasks.block, episode_tasks.episode,
        repetition, 0, "spotcheck"))
    for task, out in zip(episode_tasks.specs, outputs):
        if scorer == "rubric_proxy":
            level = rubric_proxy_score(task, out.text, out.ok)
            raw_level: int | None = None
            judge_error: str | None = None
            judge_prompt_tokens = 0
            judge_completion_tokens = 0
            judge_model = ""
            judge_provider = ""
        else:
            level, jerr, jusage = judge_score(
                task, out.text, cfg.judge_model, cfg.api_key, cfg.judge_temperature,
                seed=(derive_request_seed(cfg.seed, block or episode_tasks.block,
                                         episode_tasks.episode, repetition, task.id, "judge")
                      if cfg.judge_seed else None),
                max_tokens=cfg.max_judge_tokens, provider=cfg.provider)
            judge_usage_total.prompt_tokens += jusage.prompt_tokens
            judge_usage_total.completion_tokens += jusage.completion_tokens
            judge_usage_total.total_tokens += jusage.total_tokens
            judge_prompt_tokens = jusage.prompt_tokens
            judge_completion_tokens = jusage.completion_tokens
            judge_model = jusage.model
            judge_provider = jusage.provider
            judge_calls += 1
            judge_failures += level is None
            judge_error = jerr
            if level is None:  # conservative zero is explicit and logged as a judge failure
                level = 0
            raw_level = level
        norm = level / max(1, len(task.rubric) - 1)
        spot: float | None = None
        spot_scorer: str | None = None
        if scorer != "rubric_proxy" and cfg.spotcheck_judge_model \
                and float(spotcheck_rng.random()) < config.RUBRIC_SPOTCHECK_RATE:
            lvl_b, _, usage_b = judge_score(
                task, out.text, cfg.spotcheck_judge_model, cfg.api_key,
                cfg.judge_temperature,
                seed=(derive_request_seed(cfg.seed, block or episode_tasks.block,
                                         episode_tasks.episode, repetition, task.id, "judge_b")
                      if cfg.judge_seed else None),
                max_tokens=cfg.max_judge_tokens, provider=cfg.provider)
            judge_usage_total.prompt_tokens += usage_b.prompt_tokens
            judge_usage_total.completion_tokens += usage_b.completion_tokens
            judge_usage_total.total_tokens += usage_b.total_tokens
            judge_calls += 1
            judge_failures += lvl_b is None
            if lvl_b is not None:
                spot = lvl_b / max(1, len(task.rubric) - 1)
                spot_scorer = f"judge:{cfg.spotcheck_judge_model}"
        scores.append(ScoredTask(
            task_id=task.id, agent_id=out.agent_id, rubric_score=norm, scorer=scorer,
            judge_raw=raw_level, judge_error=judge_error,
            judge_prompt_tokens=judge_prompt_tokens,
            judge_completion_tokens=judge_completion_tokens,
            judge_model=judge_model, judge_provider=judge_provider,
            spotcheck_score=spot, spotcheck_scorer=spot_scorer))
    return EpisodeExecution(outputs=outputs, scores=scores, judge_usage=judge_usage_total,
                            judge_calls=judge_calls, judge_failures=judge_failures)
