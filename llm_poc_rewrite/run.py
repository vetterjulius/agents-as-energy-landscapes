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
from . import judge
from .judge import score
from .scoring import deterministic_score, is_deterministic_task
from .orchestration import (
    make_initial_state,
    make_real_issue_initial_state,
    orchestrate,
    orchestrate_real_issue,
    orchestration_diagnostic,
)
from .tasks import build_agent_slots, build_episode, declared_dependency_pairs


def planned_call_counts() -> tuple[int, int]:
    """Return the fixed worker and judge request counts before any API request."""
    episodes = len(config.CONDITIONS) * len(config.BLOCKS) * config.N_EPISODES * config.N_REPETITIONS
    worker_calls = episodes * config.N_TASKS
    judge_calls = episodes * len(config.LLM_JUDGE_TASK_IDS)
    return worker_calls, judge_calls


def planned_real_issue_patch_calls() -> int:
    """Return the real-issue pilot's proposed patch-generation count, excluding retries."""
    return (
        len(config.CONDITIONS) * config.N_REPETITIONS
        * len(config.REAL_ISSUE_TASK_IDS) * config.REAL_ISSUE_EPISODES
    )


def real_issue_episode_diagnostics(condition: str, seed: int) -> list[dict]:
    """Run a clearly synthetic, offline two-episode mechanism diagnostic only."""
    from .tasks import EpisodeTasks, TaskSpec

    specs = [
        TaskSpec(i, "patch", (0.0,) * config.D, 0.0, "", "", (), None)
        for i in range(len(config.REAL_ISSUE_TASK_IDS))
    ]
    slots = build_agent_slots()
    state = make_real_issue_initial_state(EpisodeTasks(specs=specs))
    records = []
    for episode_index in range(config.REAL_ISSUE_EPISODES):
        episode = EpisodeTasks(specs=specs, episode=episode_index, block="real_issue")
        result, next_state = orchestrate_real_issue(
            condition, slots, episode, state, rng_seed=seed + episode_index,
        )
        record = orchestration_diagnostic(
            result, evaluator_theta=torch.tensor(config.REAL_ISSUE_G1),
        )
        record.update({
            "episode": episode_index,
            "task_to_slot": torch.argmax(result.X.float(), dim=0).tolist(),
            "next_theta": next_state.Theta.tolist(),
            "graph_update_source": "assignment_history_only",
            "co_assignment_prior": "heuristic_not_empirically_validated",
            "positive_edge_means_shared_agent": True,
            "assignment_synergy_validated": False,
        })
        records.append(record)
        state = next_state
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed live Gemini LLM transfer pilot.")
    parser.add_argument("--output-dir", default="results/llm_poc_rewrite")
    parser.add_argument(
        "--plan-only", action="store_true",
        help="show the fixed design and call counts without making API requests",
    )
    parser.add_argument(
        "--real-issue-plan-only", action="store_true",
        help="show the gated two-issue pilot design and request count without API calls",
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
        "judge_protocol_version": config.JUDGE_PROTOCOL_VERSION,
        "judge_max_tokens": config.JUDGE_MAX_TOKENS,
        "judge_thinking_level": config.JUDGE_THINKING_LEVEL,
        "llm_judge_task_ids": list(config.LLM_JUDGE_TASK_IDS),
        "conditions": list(config.CONDITIONS),
        "blocks": list(config.BLOCKS),
        "episodes": config.N_EPISODES,
        "repetitions": config.N_REPETITIONS,
        "tasks": config.N_TASKS,
        "scoring_version": config.SCORING_PROTOCOL_VERSION,
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
        "judge_preflight_calls": 0,
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
    if getattr(args, "real_issue_plan_only", False):
        planned = planned_real_issue_patch_calls()
        if planned != config.REAL_ISSUE_EXPECTED_PATCH_CALLS:
            raise RuntimeError("Real-issue pilot request count does not match its fixed design")
        print(
            "Real-issue pilot design only: "
            f"tasks={config.REAL_ISSUE_TASK_IDS}; G0={config.REAL_ISSUE_G0}; "
            f"G1={config.REAL_ISSUE_G1}; episodes={config.REAL_ISSUE_EPISODES}; "
            f"conditions={config.CONDITIONS}; repetitions={config.N_REPETITIONS}; "
            f"adaptive_mode={config.REAL_ISSUE_ADAPTATION_MODE}; "
            "edge_semantics=unvalidated_same-agent heuristic; "
            f"planned_patch_generations={planned}; judge_calls=0; preflight_calls=0; "
            "API requests=0; live run remains gated on validated vectors and historical tests; "
            "the following trace uses synthetic role vectors and is mechanism-only."
        )
        for condition in config.CONDITIONS:
            for record in real_issue_episode_diagnostics(condition, seed=912):
                print(json.dumps(record))
        return 0
    workers, judges = planned_call_counts()
    if (workers, judges) != (config.EXPECTED_WORKER_CALLS, config.EXPECTED_JUDGE_CALLS):
        raise RuntimeError("Fixed run design does not match the protocol call totals")
    print(
        f"Fixed design: {workers} worker calls + {judges} judge calls "
        f"+ 1 judge preflight = {workers + judges + 1} planned Gemini API calls on a fresh run; "
        f"agents={config.AGENT_MODELS}; judge={config.JUDGE_MODEL}"
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
    base_seed = 20260923
    try:
        preflight_generation = judge.preflight(
            seed=_request_seed(base_seed, "stationary", 0, 0, 0, "judge-preflight"),
        )
    except RuntimeError as exc:
        checkpoint["judge_preflight_calls"] = checkpoint.get("judge_preflight_calls", 0) + 1
        checkpoint["judge_preflight_error"] = str(exc)
        if not args.resume:
            _write_checkpoint(checkpoint_path, checkpoint)
        elif checkpoint["judge_preflight_calls"] > 1 or checkpoint.get("rows") or checkpoint.get("pending"):
            _write_checkpoint(checkpoint_path, checkpoint)
        raise
    checkpoint["judge_preflight_calls"] = checkpoint.get("judge_preflight_calls", 0) + 1
    checkpoint.pop("judge_preflight_error", None)
    _write_checkpoint(checkpoint_path, checkpoint)
    print(
        f"Judge preflight passed: score output validated; "
        f"finishReason={preflight_generation.finish_reason}; "
        f"tokens={preflight_generation.completion_tokens}"
    )
    slots = build_agent_slots()
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
                    state = make_initial_state(tasks)
                assignment, next_state = orchestrate(
                    condition, slots, tasks, state,
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
                        judge_generation = None
                        if is_deterministic_task(task.id):
                            level, judge_error = deterministic_score(task, worker.text)
                            if judge_error is not None:
                                checkpoint["pending"]["scoring_failure"] = {"error": judge_error}
                                _write_checkpoint(checkpoint_path, checkpoint)
                                raise RuntimeError(f"Deterministic scoring failed for task {key}: {judge_error}")
                        else:
                            try:
                                level, judge_generation, judge_error = score(
                                    task, worker.text, task_prompt=prompt,
                                    seed=_request_seed(base_seed, block, episode, repetition, task.id, "judge"),
                                )
                            except Exception as exc:
                                checkpoint["pending"]["judge_failure"] = {
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                                _write_checkpoint(checkpoint_path, checkpoint)
                                raise
                            if judge_error is not None:
                                checkpoint["pending"]["judge_failure"] = {
                                    "generation": asdict(judge_generation),
                                    "error": judge_error,
                                }
                                _write_checkpoint(checkpoint_path, checkpoint)
                                raise RuntimeError(
                                    f"Judge validation failed for task {key}; "
                                    f"finishReason={judge_generation.finish_reason or 'missing'}, "
                                    f"thoughtsTokens={judge_generation.thoughts_tokens}, "
                                    f"completionTokens={judge_generation.completion_tokens}, "
                                    f"output={judge_generation.text!r}: {judge_error}"
                                )
                        row = {
                            "condition": condition,
                            "block": block,
                            "episode": episode,
                            "repetition": repetition,
                            "task_id": task.id,
                            "dependency_upstream": task.depends_on,
                            "agent_id": agent.id,
                            "agent": agent.name,
                            "worker_model": agent.model,
                            "worker_model_version": worker.model_version,
                            "worker_text": worker.text,
                            "task_reference_answer": task.reference_answer,
                            "task_dependencies": ",".join(
                                f"{upstream}->{downstream}" for upstream, downstream in dependencies
                                if task.id in (upstream, downstream)
                            ),
                            "worker_prompt_tokens": worker.prompt_tokens,
                            "worker_completion_tokens": worker.completion_tokens,
                            "worker_latency_seconds": worker.latency_seconds,
                            "scoring_method": "llm_judge" if judge_generation else "deterministic",
                            "judge_model": config.JUDGE_MODEL if judge_generation else "",
                            "judge_model_version": judge_generation.model_version if judge_generation else "",
                            "judge_finish_reason": judge_generation.finish_reason if judge_generation else "",
                            "judge_thoughts_tokens": judge_generation.thoughts_tokens if judge_generation else 0,
                            "judge_total_tokens": judge_generation.total_tokens if judge_generation else 0,
                            "judge_level": level,
                            "judge_normalized_score": level / max(1, len(task.rubric) - 1),
                            "judge_raw_text": judge_generation.text if judge_generation else "",
                            "judge_error": "",
                            "judge_prompt_tokens": judge_generation.prompt_tokens if judge_generation else 0,
                            "judge_completion_tokens": judge_generation.completion_tokens if judge_generation else 0,
                            "judge_latency_seconds": judge_generation.latency_seconds if judge_generation else 0.0,
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
    if judge_errors:
        raise RuntimeError(f"Refusing to finalize results with {judge_errors} invalid judge rows")
    judge_rows = sum(row["scoring_method"] == "llm_judge" for row in rows)
    if judge_rows != judges:
        raise RuntimeError(f"Expected {judges} semantic judge scores, got {judge_rows}")

    summary = {
        "status": "hypothesis-generating transfer pilot; not a confirmatory benchmark",
        "judge_parse_errors": judge_errors,
        "scoring_errors": judge_errors,
        "judge_parse_error_policy": "abort before recording a task row if a semantic judge response is invalid",
        "scoring_protocol_version": config.SCORING_PROTOCOL_VERSION,
        "scoring_protocol": "exact deterministic checks for objective tasks; LLM judge only for synthesis tasks 7 and 9",
        "judge_protocol_version": config.JUDGE_PROTOCOL_VERSION,
        "judge_preflight": {
            "passed": True,
            "finish_reason": preflight_generation.finish_reason,
            "prompt_tokens": preflight_generation.prompt_tokens,
            "completion_tokens": preflight_generation.completion_tokens,
            "thoughts_tokens": preflight_generation.thoughts_tokens,
            "total_tokens": preflight_generation.total_tokens,
        },
        "worker_calls": workers,
        "judge_calls": judges,
        "deterministic_task_scores": workers - judges,
        "judge_preflight_calls": checkpoint["judge_preflight_calls"],
        "total_task_api_calls": workers + judges,
        "total_gemini_api_calls": workers + judges + checkpoint["judge_preflight_calls"],
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
