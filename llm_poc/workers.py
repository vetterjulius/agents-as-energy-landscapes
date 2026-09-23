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

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

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


def call_openrouter(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: float = config.API_TIMEOUT_SEC,
    max_retries: int = config.MAX_RETRIES,
) -> tuple[str | None, Usage, float, float, int, str | None]:
    """Single chat completion. Returns (text|None, usage, t_first, t_total, retries, error)."""
    url = config.OPENROUTER_URL
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    usage = Usage()
    retries = 0
    err: str | None = None
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            total = time.perf_counter() - t0
            choice = data["choices"][0]["message"]["content"]
            u = data.get("usage", {}) or {}
            usage = Usage(
                prompt_tokens=int(u.get("prompt_tokens", 0)),
                completion_tokens=int(u.get("completion_tokens", 0)),
                total_tokens=int(u.get("total_tokens",
                                       u.get("prompt_tokens", 0) + u.get("completion_tokens", 0))),
            )
            # OpenRouter may return error payloads with HTTP 200.
            if data.get("error"):
                err = f"api_error: {json.dumps(data['error'])[:200]}"
            else:
                return choice, usage, total, total, retries, None
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001
                pass
            err = f"http_{exc.code}: {exc.reason} {detail}"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        retries = attempt
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    return None, usage, time.perf_counter() - t0, time.perf_counter() - t0, retries, err


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
    h = (hash((task.id, slot.id, prompt)) % 1000) / 1000.0
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
                temperature: float) -> tuple[int | None, str | None, Usage]:
    rubric_text = "\n".join(f"{i}: {anchor}" for i, anchor in enumerate(task.rubric))
    prompt = (
        f"Task: {task.prompt_template.split('{')[0].strip()}\n\n"
        f"Reference-correct behaviour: {task.reference_answer}\n\n"
        f"Rubric levels:\n{rubric_text}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        "Which rubric level (integer index) does the candidate fully satisfy?"
    )
    text, usage, _, _, _, err = call_openrouter(
        prompt, JUDGE_SYSTEM, model, api_key, temperature=temperature)
    if text is None:
        return None, err, usage
    match = re.search(r"\d+", text)
    if not match:
        return None, f"unparseable judge output: {text[:120]}", usage
    level = int(match.group())
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


def _jittered_default_rng():
    import numpy as np
    return np.random.default_rng()


def execute_episode(
    episode_tasks: EpisodeTasks,
    slots: list[AgentSlot],
    X,                      # (N, M) assignment tensor from Layer 1/2
    cfg: config.PoCConfig,
    repetition: int,
    episode_seed: int,
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
            rng = np.random.default_rng(episode_seed + task.id)
            out = mock_worker(task, slot, prompt, rng, dep_available=dep_available)
        else:
            text, usage, t_first, t_total, retries, err = call_openrouter(
                prompt, slot.system_prompt, cfg.model, cfg.api_key,
                max_retries=cfg.max_retries)
            out = WorkerOutput(
                task_id=task.id, agent_id=slot.id, ok=text is not None,
                text=text or "", error=err, usage=usage,
                latency_first_token_sec=t_first, latency_total_sec=t_total,
                retries=retries)
        if out.ok and out.text:
            raw_by_task[task.id] = out.text
        outputs.append(out)

    # Scoring
    scores: list[ScoredTask] = []
    judge_usage_total = Usage()
    scorer = "rubric_proxy" if (cfg.worker_mode == "mock" or not cfg.judge_enabled) \
        else f"judge:{cfg.judge_model}"
    spotcheck_rng = np.random.default_rng(episode_seed + 777)
    for task, out in zip(episode_tasks.specs, outputs):
        if scorer == "rubric_proxy":
            level = rubric_proxy_score(task, out.text, out.ok)
            raw_level: int | None = None
        else:
            level, jerr, jusage = judge_score(
                task, out.text, cfg.judge_model, cfg.api_key, cfg.judge_temperature)
            judge_usage_total.prompt_tokens += jusage.prompt_tokens
            judge_usage_total.completion_tokens += jusage.completion_tokens
            judge_usage_total.total_tokens += jusage.total_tokens
            if level is None:  # judge failure -> conservative 0 with error logged downstream
                level = 0
            raw_level = level
        norm = level / max(1, len(task.rubric) - 1)
        spot: float | None = None
        spot_scorer: str | None = None
        if scorer != "rubric_proxy" and cfg.spotcheck_judge_model \
                and float(spotcheck_rng.random()) < config.RUBRIC_SPOTCHECK_RATE:
            lvl_b, _, _ = judge_score(
                task, out.text, cfg.spotcheck_judge_model, cfg.api_key,
                cfg.judge_temperature)
            if lvl_b is not None:
                spot = lvl_b / max(1, len(task.rubric) - 1)
                spot_scorer = f"judge:{cfg.spotcheck_judge_model}"
        scores.append(ScoredTask(
            task_id=task.id, agent_id=out.agent_id, rubric_score=norm, scorer=scorer,
            judge_raw=raw_level, spotcheck_score=spot, spotcheck_scorer=spot_scorer))
    return EpisodeExecution(outputs=outputs, scores=scores, judge_usage=judge_usage_total)
