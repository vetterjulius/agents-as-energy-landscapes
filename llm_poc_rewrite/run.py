"""Runner for the single live Gemini transfer pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
import tempfile

import numpy as np
import torch

from . import config
from .api import Generation, generate
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
    parser.add_argument(
        "--resume", action="store_true",
        help="continue an interrupted run from its checkpoint in --output-dir",
    )
    return parser.parse_args()


def _request_seed(base: int, block: str, episode: int, repetition: int,
                  task: int, role: str) -> int:
    payload = f"{base}|{block}|{episode}|{repetition}|{task}|{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _task_key(condition: str, block: str, episode: int, repetition: int,
              task_id: int) -> tuple[str, str, int, int, int]:
    return condition, block, episode, repetition, task_id


def _row_key(row: dict) -> tuple[str, str, int, int, int]:
    return _task_key(row["condition"], row["block"], row["episode"],
                     row["repetition"], row["task_id"])


def _checkpoint_design() -> dict:
    return {
        "agent_models": config.AGENT_MODELS,
        "judge_model": config.JUDGE_MODEL,
        "conditions": list(config.CONDITIONS),
        "blocks": list(config.BLOCKS),
        "episodes": config.N_EPISODES,
        "repetitions": config.N_REPETITIONS,
        "tasks": config.N_TASKS,
    }


def _write_checkpoint(path: Path, checkpoint: dict) -> None:
    """Atomically replace the checkpoint so interruption cannot leave partial JSON."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f"{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(checkpoint, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read resume checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise SystemExit("Invalid resume checkpoint: expected a JSON object")
    if checkpoint.get("schema_version") != 1:
        raise SystemExit("Unsupported or invalid resume checkpoint schema")
    if checkpoint.get("status") == "complete":
        raise SystemExit("This run is already complete; use a new --output-dir for a fresh run")
    if checkpoint.get("status") != "running":
        raise SystemExit("Invalid resume checkpoint status")
    if checkpoint.get("design") != _checkpoint_design():
        raise SystemExit("Checkpoint design does not match current models or experiment dimensions")
    if not isinstance(checkpoint.get("rows"), list):
        raise SystemExit("Checkpoint is missing its completed result rows")
    return checkpoint


def _new_checkpoint() -> dict:
    return {
        "schema_version": 1,
        "status": "running",
        "design": _checkpoint_design(),
        "started_at": time.time(),
        "rows": [],
        "pending": None,
    }


def _generation_from_checkpoint(data: dict) -> Generation:
    return Generation(**data)


def _validate_checkpoint_rows(checkpoint: dict, expected_keys: set[tuple]) -> dict:
    completed = {}
    for row in checkpoint["rows"]:
        try:
            key = _row_key(row)
        except (KeyError, TypeError) as exc:
            raise SystemExit("Invalid task entry in checkpoint") from exc
        if key not in expected_keys or key in completed:
            raise SystemExit(f"Invalid or duplicate task entry in checkpoint: {key}")
        completed[key] = row
    pending = checkpoint.get("pending")
    if pending is not None:
        if not isinstance(pending, dict):
            raise SystemExit("Invalid pending task in checkpoint")
        key = tuple(pending.get("key", ()))
        if key not in expected_keys or key in completed:
            raise SystemExit(f"Invalid pending task in checkpoint: {key}")
        try:
            _generation_from_checkpoint(pending["worker"])
        except (KeyError, TypeError) as exc:
            raise SystemExit("Invalid pending worker result in checkpoint") from exc
        pending["key"] = list(key)
    return completed


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
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is missing; no requests were made")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "llm_poc_rewrite_checkpoint.json"
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"No checkpoint found to resume: {checkpoint_path}")
        checkpoint = _load_checkpoint(checkpoint_path)
    else:
        if checkpoint_path.exists():
            raise SystemExit(
                f"Checkpoint already exists at {checkpoint_path}; use --resume or choose a new --output-dir"
            )
        checkpoint = _new_checkpoint()
        _write_checkpoint(checkpoint_path, checkpoint)
    rows: list[dict] = checkpoint["rows"]
    expected_keys = {
        _task_key(condition, block, episode, repetition, task_id)
        for condition in config.CONDITIONS
        for block in config.BLOCKS
        for episode in range(config.N_EPISODES)
        for repetition in range(config.N_REPETITIONS)
        for task_id in range(config.N_TASKS)
    }
    completed_by_key = _validate_checkpoint_rows(checkpoint, expected_keys)
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
                        key = _task_key(condition, block, episode, repetition, task.id)
                        existing_row = completed_by_key.get(key)
                        if existing_row is not None:
                            outputs[task.id] = existing_row["worker_text"][:1200]
                            continue

                        print(f"Evaluating task {task.id} (depends on {task.depends_on}) under repetition {repetition} of episode {episode}...")
                        agent = slots[int(assignments[task.id])]
                        prompt = task.prompt_template
                        if task.depends_on is not None:
                            upstream = outputs.get(task.depends_on, "[upstream output unavailable]")
                            prompt = prompt.replace("{dep}", upstream[:1200])
                        prompt = prompt.replace("{input}", tasks.inputs[task.id])
                        pending = checkpoint.get("pending")
                        if pending is not None:
                            if tuple(pending["key"]) != key:
                                raise RuntimeError(
                                    f"Checkpoint pending task {pending['key']} does not match next task {key}"
                                )
                            worker = _generation_from_checkpoint(pending["worker"])
                        else:
                            worker = generate(
                                prompt, agent.system_prompt, agent.model,
                                seed=_request_seed(base_seed, block, episode, repetition, task.id, "worker"),
                            )
                            checkpoint["pending"] = {
                                "key": list(key),
                                "worker": asdict(worker),
                            }
                            _write_checkpoint(checkpoint_path, checkpoint)
                        outputs[task.id] = worker.text[:1200]
                        level, judge, judge_error = score(
                            task, worker.text, task_prompt=prompt,
                            seed=_request_seed(base_seed, block, episode, repetition, task.id, "judge"),
                        )
                        row = {
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
                        }
                        rows.append(row)
                        completed_by_key[key] = row
                        checkpoint["rows"] = rows
                        checkpoint["pending"] = None
                        _write_checkpoint(checkpoint_path, checkpoint)
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
        "resumed": args.resume,
    }
    (output_dir / "llm_poc_rewrite_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    checkpoint["status"] = "complete"
    checkpoint["pending"] = None
    _write_checkpoint(checkpoint_path, checkpoint)
    print(f"Completed {workers} worker and {judges} judge calls in {elapsed:.1f}s; output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
