"""Runner for the single live Gemini transfer pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from .api import generate
from .judge import score
from .orchestration import make_initial_state, orchestrate
from .tasks import build_agent_slots, build_episode, declared_dependency_pairs


def planned_call_counts() -> tuple[int, int]:
    """Return the fixed worker and judge request counts before any API request."""
    worker_calls = (
        len(config.CONDITIONS) * len(config.BLOCKS) * config.N_EPISODES
        * config.N_REPETITIONS * config.N_TASKS
    )
    return worker_calls, worker_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed live Gemini LLM transfer pilot.")
    parser.add_argument("--output-dir", default="results/llm_poc_rewrite")
    parser.add_argument(
        "--plan-only", action="store_true",
        help="show the fixed design and call counts without making API requests",
    )
    return parser.parse_args()


def _request_seed(base: int, block: str, episode: int, repetition: int,
                  task: int, role: str) -> int:
    payload = f"{base}|{block}|{episode}|{repetition}|{task}|{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    args = parse_args()
    workers, judges = planned_call_counts()
    if (workers, judges) != (config.EXPECTED_WORKER_CALLS, config.EXPECTED_JUDGE_CALLS):
        raise RuntimeError("Fixed run design does not match the protocol call totals")
    print(
        f"Fixed design: {workers} worker calls + {judges} judge calls "
        f"= {workers + judges} Gemini API calls; agents={config.AGENT_MODELS}; "
        f"judge={config.JUDGE_MODEL}"
    )
    print(f"conditions={config.CONDITIONS}; Blocks={config.BLOCKS};")
    if args.plan_only:
        return 0
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is missing; no requests were made")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    slots = build_agent_slots()
    base_seed = 20260923
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    for condition in config.CONDITIONS:
        for block in config.BLOCKS:
            print(f"Running block {block} under condition {condition}...")
            state = None
            for episode in range(config.N_EPISODES):
                tasks = build_episode(block, episode)
                dependencies = declared_dependency_pairs(block, episode)
                if state is None:
                    state = make_initial_state(tasks, dependencies)
                assignment, next_state = orchestrate(
                    condition, slots, tasks, dependencies, state,
                    base_seed * 100000 + episode * 1000,
                )
                assignments = torch.argmax(assignment.X.float(), dim=0).tolist()
                ordered_tasks = sorted(tasks.specs, key=lambda task: (task.depends_on is not None, task.id))
                for repetition in range(config.N_REPETITIONS):
                    outputs: dict[int, str] = {}
                    # Dependencies are evaluated first, then consumers receive their actual response.
                    for task in ordered_tasks:
                        print(f"Evaluating task {task.id} (depends on {task.depends_on}) under repetition {repetition} of episode {episode}...")
                        agent = slots[int(assignments[task.id])]
                        prompt = task.prompt_template
                        if task.depends_on is not None:
                            upstream = outputs.get(task.depends_on, "[upstream output unavailable]")
                            prompt = prompt.replace("{dep}", upstream[:1200])
                        prompt = prompt.replace("{input}", tasks.inputs[task.id])
                        worker = generate(
                            prompt, agent.system_prompt, agent.model,
                            seed=_request_seed(base_seed, block, episode, repetition, task.id, "worker"),
                        )
                        outputs[task.id] = worker.text[:1200]
                        level, judge, judge_error = score(
                            task, worker.text, task_prompt=prompt,
                            seed=_request_seed(base_seed, block, episode, repetition, task.id, "judge"),
                        )
                        rows.append({
                            "condition": condition,
                            "block": block,
                            "episode": episode,
                            "repetition": repetition,
                            "task_id": task.id,
                            "agent_id": agent.id,
                            "agent": agent.name,
                            "worker_model": agent.model,
                            "worker_model_version": worker.model_version,
                            "worker_text": worker.text,
                            "worker_prompt_tokens": worker.prompt_tokens,
                            "worker_completion_tokens": worker.completion_tokens,
                            "worker_latency_seconds": worker.latency_seconds,
                            "judge_model": config.JUDGE_MODEL,
                            "judge_model_version": judge.model_version,
                            "judge_level": level,
                            "judge_normalized_score": level / max(1, len(task.rubric) - 1),
                            "judge_raw_text": judge.text,
                            "judge_error": judge_error or "",
                            "judge_prompt_tokens": judge.prompt_tokens,
                            "judge_completion_tokens": judge.completion_tokens,
                            "judge_latency_seconds": judge.latency_seconds,
                        })
                state = next_state

    elapsed = time.perf_counter() - started
    if len(rows) != workers:
        raise RuntimeError(f"Expected {workers} task rows, got {len(rows)}")
    with (output_dir / "llm_poc_rewrite.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    judge_errors = sum(bool(row["judge_error"]) for row in rows)
    summary = {
        "status": "hypothesis-generating transfer pilot; not a confirmatory benchmark",
        "judge_parse_errors": judge_errors,
        "judge_parse_error_policy": "retain raw judge output and use conservative score 0",
        "worker_calls": workers,
        "judge_calls": judges,
        "total_gemini_api_calls": workers + judges,
        "agent_models": config.AGENT_MODELS,
        "judge_model": config.JUDGE_MODEL,
        "conditions": config.CONDITIONS,
        "blocks": config.BLOCKS,
        "episodes_per_block": config.N_EPISODES,
        "repetitions": config.N_REPETITIONS,
        "tasks_per_episode": config.N_TASKS,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "llm_poc_rewrite_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Completed {workers} worker and {judges} judge calls in {elapsed:.1f}s; output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
