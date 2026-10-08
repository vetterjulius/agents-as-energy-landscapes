"""LLM PoC runner: orchestrate -> execute -> score -> log."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config, metrics, workers
from .orchestration import orchestrate
from .tasks import build_agent_slots, build_episode, declared_dependency_pairs


_RESUME_IDENTITY_FIELDS = (
    "model", "provider", "blocks", "episodes", "repetitions", "seed",
    "task_call_cap", "protocol_version", "worker_temperature", "worker_top_p",
    "max_worker_tokens", "max_worker_output_chars",
)


def prepare_live_run(output_path: Path, manifest: dict, *, resume: bool = False) -> dict:
    """Create a live run directory or validate a matching safe resume."""
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "poc_live_run.json"
    journal_path = output_path / "poc_live_progress.jsonl"
    existing = {path.name for path in output_path.iterdir()}
    allowed = {manifest_path.name, journal_path.name}
    unexpected = existing - allowed
    if unexpected:
        raise SystemExit(
            f"[llm_poc] live output directory contains non-resumable artifacts: "
            f"{', '.join(sorted(unexpected))}; choose a new directory"
        )
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"[llm_poc] cannot read live run manifest: {exc}") from exc
        if resume:
            mismatched = [key for key in _RESUME_IDENTITY_FIELDS
                          if previous.get(key) != manifest.get(key)]
            # Legacy journals predating task-bank fingerprints remain resumable;
            # per-coordinate request fingerprints still reject prompt changes.
            if ("task_bank_sha256" in previous and
                    previous.get("task_bank_sha256") != manifest.get("task_bank_sha256")):
                mismatched.append("task_bank_sha256")
            if ("request_impl_sha256" in previous and
                    previous.get("request_impl_sha256") != manifest.get("request_impl_sha256")):
                mismatched.append("request_impl_sha256")
            old_cap = previous.get("attempt_cap", 0)
            if old_cap < 1 or manifest.get("attempt_cap", 0) > old_cap:
                mismatched.append("attempt_cap (cannot increase an existing run's cap)")
            if mismatched:
                raise SystemExit(
                    "[llm_poc] live run cannot be resumed with changed identity fields: "
                    + ", ".join(mismatched)
                )
        elif previous != manifest:
            raise SystemExit(
                "[llm_poc] live output directory belongs to a different run "
                "configuration; choose a new directory"
            )
        if not journal_path.exists():
            raise SystemExit("[llm_poc] resumable run manifest has no progress journal")
    elif journal_path.exists():
        raise SystemExit(
            "[llm_poc] progress journal has no run manifest and cannot be resumed safely"
        )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        previous = manifest
    return previous


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run the LLM orchestration PoC.")
    parser.add_argument("--worker-mode", choices=("mock", "gemini"), default="mock")
    parser.add_argument("--probe-model", action="store_true",
                        help="make exactly one tiny Gemma API generation to verify key/model reachability")
    parser.add_argument("--no-judge", action="store_true",
                        help="retained for backwards compatibility; pilot scoring is always rubric_proxy")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--block", choices=("all", *config.BLOCKS), default=None)
    parser.add_argument("--output-dir", default="results/llm_poc")
    parser.add_argument("--seed", type=int, default=config.SEED_BASE)
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--limit-calls", type=int, default=None,
                        help="optional task-call cap; run is rejected if the design exceeds it")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted live run from its matching journal")
    parser.add_argument("--attempt-cap", type=int, default=config.MAX_LIVE_API_CALLS,
                        help="absolute total HTTP attempt cap for live run/resume")
    return parser.parse_args()


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    args = parse_args()
    if args.probe_model:
        api_key = config.PoCConfig(worker_mode="gemini").api_key
        if not api_key:
            raise SystemExit("[llm_poc] missing GEMINI_API_KEY; add it to .env")
        text, usage, error = workers.probe_gemini(api_key)
        if error:
            print(f"[llm_poc] reachability probe failed: {error}")
            return 1
        print(f"[llm_poc] reachable: requested={config.DEFAULT_MODEL} "
              f"modelVersion={usage.model or 'not reported'} provider={usage.provider}; "
              "one minimal generation call made")
        return 0
    live = args.worker_mode == "gemini"
    if live and args.model != config.DEFAULT_MODEL:
        raise SystemExit(
            f"[llm_poc] live runs are locked to {config.DEFAULT_MODEL}; "
            "changing models requires a new protocol version"
        )
    args.episodes = args.episodes if args.episodes is not None else (1 if live else config.N_EPISODES)
    args.repetitions = (args.repetitions if args.repetitions is not None
                        else (1 if live else config.N_REPETITIONS))
    args.block = args.block if args.block is not None else ("stationary" if live else "all")
    if not 1 <= args.episodes <= config.N_EPISODES:
        raise SystemExit(f"[llm_poc] --episodes must be between 1 and {config.N_EPISODES}")
    if args.repetitions < 1:
        raise SystemExit("[llm_poc] --repetitions must be positive")
    if args.limit_calls is not None and args.limit_calls < 1:
        raise SystemExit("[llm_poc] --limit-calls must be positive")
    if not 1 <= args.attempt_cap <= config.MAX_LIVE_API_CALLS:
        raise SystemExit(
            f"[llm_poc] --attempt-cap must be between 1 and {config.MAX_LIVE_API_CALLS}"
        )
    if args.resume and not live:
        raise SystemExit("[llm_poc] --resume is only valid with --worker-mode gemini")

    cfg = config.PoCConfig(
        worker_mode=args.worker_mode,
        judge_enabled=False,
        n_repetitions=args.repetitions,
        n_episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        model=args.model,
    )
    cfg.require_api()
    provenance = config.collect_environment()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    slots = build_agent_slots()
    blocks = config.BLOCKS if args.block == "all" else (args.block,)
    planned_worker_calls = (len(metrics.CONDITIONS) * len(blocks) * args.episodes *
                            cfg.n_repetitions * config.N_TASKS)
    if args.worker_mode == "gemini" and planned_worker_calls > config.MAX_LIVE_TASK_CALLS:
        raise SystemExit(
            f"[llm_poc] requested run needs {planned_worker_calls} task calls, "
            f"exceeding the pilot design cap of {config.MAX_LIVE_TASK_CALLS}; "
            "reduce blocks/episodes/repetitions or use the offline mock"
        )
    if args.limit_calls is not None and planned_worker_calls > args.limit_calls:
        raise SystemExit(
            f"[llm_poc] requested design needs {planned_worker_calls} calls, "
            f"exceeding --limit-calls={args.limit_calls}; no partial run started"
        )
    if live:
        output_path = Path(cfg.output_dir)
        live_manifest = {
            "model": cfg.model,
            "provider": cfg.provider,
            "blocks": list(blocks),
            "episodes": args.episodes,
            "repetitions": cfg.n_repetitions,
            "seed": cfg.seed,
            "max_retries": cfg.max_retries,
            "task_call_cap": config.MAX_LIVE_TASK_CALLS,
            "attempt_cap": args.attempt_cap,
            "resume_extra_attempts": config.MAX_RESUME_EXTRA_ATTEMPTS,
            "timeout_sec": config.API_TIMEOUT_SEC,
            "retry_jitter_ratio": config.RETRY_JITTER_RATIO,
            "max_retry_after_sec": config.MAX_RETRY_AFTER_SEC,
            "protocol_version": cfg.protocol_version,
            "worker_temperature": cfg.worker_temperature,
            "worker_top_p": cfg.worker_top_p,
            "max_worker_tokens": cfg.max_worker_tokens,
            "max_worker_output_chars": cfg.max_worker_output_chars,
            "source_sha256": provenance["source_sha256"],
            "task_bank_sha256": hashlib.sha256(
                Path(__file__).with_name("tasks.py").read_bytes()
            ).hexdigest(),
            "request_impl_sha256": hashlib.sha256(
                Path(__file__).with_name("workers.py").read_bytes()
            ).hexdigest(),
            "retry_base_sec": config.RETRY_BASE_SEC,
            "retry_max_sec": config.RETRY_MAX_SEC,
        }
        journal_path = output_path / "poc_live_progress.jsonl"
        if args.resume:
            if not (output_path / "poc_live_run.json").exists():
                raise SystemExit("[llm_poc] --resume requires an existing live run directory")
            if not journal_path.exists():
                raise SystemExit("[llm_poc] --resume requires an existing progress journal")
        elif output_path.exists() and any(output_path.iterdir()):
            raise SystemExit(
                f"[llm_poc] live output directory is not empty: {output_path}; "
                "pass --resume to continue a matching run, or choose a fresh directory"
            )
        prior_manifest = prepare_live_run(
            output_path, live_manifest, resume=args.resume)
        workers.set_progress_log(
            journal_path, resume=args.resume, max_attempts=args.attempt_cap)
        if args.resume:
            workers._write_progress(
                "resume_configuration",
                prior_source_sha256=prior_manifest.get("source_sha256"),
                current_source_sha256=live_manifest.get("source_sha256"),
                previous_timeout_sec=prior_manifest.get("timeout_sec"),
                current_timeout_sec=config.API_TIMEOUT_SEC,
                previous_attempt_cap=prior_manifest.get("attempt_cap"),
                current_attempt_cap=args.attempt_cap,
            )

    rows: list[dict] = []
    judge_rows: list[dict] = []
    call_count = judge_call_total = judge_failure_total = 0
    judge_prompt_tokens_total = judge_completion_tokens_total = 0
    resolved_model_ids: set[str] = set()
    resolved_providers: set[str] = set()
    start = time.perf_counter()
    print(f"[llm_poc] mode={cfg.worker_mode} judge={cfg.judge_enabled} "
          f"model={cfg.model} provider={cfg.provider} repetitions={cfg.n_repetitions} "
          f"planned_calls={planned_worker_calls} limit={config.MAX_LIVE_TASK_CALLS if live else 'n/a'} "
          f"-> {cfg.output_dir}", flush=True)

    for condition in metrics.CONDITIONS:
        for block in blocks:
            state = None
            for episode in range(args.episodes):
                episode_tasks = build_episode(block, episode)
                dependencies = declared_dependency_pairs(block, episode)
                if state is None:
                    from .orchestration import make_initial_state
                    state = make_initial_state(episode_tasks, dependencies)
                solver_seed = cfg.seed * 100000 + episode * 1000
                orchestration, state = orchestrate(
                    condition, slots, episode_tasks, dependencies, state, solver_seed)

                for repetition in range(cfg.n_repetitions):
                    episode_seed = cfg.seed + repetition * 977 + episode * 31
                    try:
                        execution = workers.execute_episode(
                            episode_tasks, slots, orchestration.X, cfg, repetition,
                            episode_seed, condition=condition, block=block)
                    except workers.LiveWorkerFailure as exc:
                        print(f"[llm_poc] {exc}; progress journal: "
                              f"{Path(cfg.output_dir) / 'poc_live_progress.jsonl'}", flush=True)
                        return 1
                    call_count += len(execution.outputs)
                    judge_call_total += execution.judge_calls
                    judge_failure_total += execution.judge_failures
                    judge_prompt_tokens_total += execution.judge_usage.prompt_tokens
                    judge_completion_tokens_total += execution.judge_usage.completion_tokens
                    resolved_model_ids.update(out.usage.model for out in execution.outputs
                                              if out.usage.model)
                    resolved_model_ids.update(score.judge_model for score in execution.scores
                                              if score.judge_model)
                    resolved_providers.update(out.usage.provider for out in execution.outputs
                                              if out.usage.provider)
                    resolved_providers.update(score.judge_provider for score in execution.scores
                                              if score.judge_provider)

                    external = metrics.external_score(
                        episode_tasks, orchestration.X, execution.scores, dependencies)
                    assignments = torch.argmax(orchestration.X.float(), dim=0).tolist()
                    dependent_tasks = [task for task in episode_tasks.specs
                                       if task.depends_on is not None]
                    dependency_rate = (
                        sum(assignments[task.depends_on] == assignments[task.id]
                            for task in dependent_tasks) / len(dependent_tasks)
                        if dependent_tasks else float("nan")
                    )

                    for task, output, score in zip(
                            episode_tasks.specs, execution.outputs, execution.scores):
                        worker_seed = workers.derive_request_seed(
                            cfg.seed, block, episode, repetition, task.id, "worker")
                        judge_seed = workers.derive_request_seed(
                            cfg.seed, block, episode, repetition, task.id, "judge")
                        judge_rows.append({
                            "condition": condition,
                            "block": block,
                            "episode": episode,
                            "repetition": repetition,
                            "task_id": task.id,
                            "judge_seed": judge_seed if cfg.judge_enabled else "",
                            "judge_prompt_tokens": score.judge_prompt_tokens,
                            "judge_completion_tokens": score.judge_completion_tokens,
                            "judge_model": score.judge_model,
                            "judge_provider": score.judge_provider,
                            "judge_error": score.judge_error or "",
                            "spotcheck_score": (score.spotcheck_score
                                                if score.spotcheck_score is not None else ""),
                            "spotcheck_scorer": score.spotcheck_scorer or "",
                        })
                        agent_id = int(assignments[task.id])
                        rows.append({
                            "condition": condition,
                            "condition_label": metrics.CONDITION_OF[condition],
                            "block": block,
                            "episode": episode,
                            "repetition": repetition,
                            "task_id": task.id,
                            "task_kind": task.kind,
                            "depends_on": task.depends_on,
                            "assigned_agent": agent_id,
                            "agent_name": slots[agent_id].name,
                            "on_specialization": task.kind == slots[agent_id].specialization,
                            "dependency_coexecuted": bool(task.depends_on is not None and
                                                           assignments[task.depends_on] == agent_id),
                            "worker_ok": output.ok,
                            "worker_error": output.error or "",
                            "worker_text": output.text,
                            "worker_seed": worker_seed,
                            "judge_seed": judge_seed if cfg.judge_enabled else "",
                            "model": output.usage.model,
                            "provider": output.usage.provider,
                            "endpoint_model": output.usage.endpoint_model,
                            "routing_attempt": output.usage.routing_attempt,
                            "router_strategy": output.usage.router_strategy,
                            "router_region": output.usage.router_region,
                            "provider_is_byok": output.usage.provider_is_byok,
                            "generation_id": output.usage.generation_id,
                            "retries": output.retries,
                            "latency_total_sec": output.latency_total_sec,
                            "prompt_tokens": output.usage.prompt_tokens,
                            "completion_tokens": output.usage.completion_tokens,
                            "finish_reason": output.usage.finish_reason,
                            "rubric_score": score.rubric_score,
                            "judge_raw": score.judge_raw,
                            "judge_error": score.judge_error or "",
                            "judge_prompt_tokens": score.judge_prompt_tokens,
                            "judge_completion_tokens": score.judge_completion_tokens,
                            "judge_model": score.judge_model,
                            "judge_provider": score.judge_provider,
                            "scorer": score.scorer,
                            "spotcheck_score": (score.spotcheck_score
                                                if score.spotcheck_score is not None else ""),
                            "spotcheck_scorer": score.spotcheck_scorer or "",
                            "mean_quality_episode": external["mean_quality"],
                            "dependency_satisfaction_episode": dependency_rate,
                            "realized_external_score_episode": external["realized_external_score"],
                            "dep_available": bool(task.depends_on is None or
                                                   assignments[task.depends_on] == agent_id),
                            "solver_energy": orchestration.solver_energy,
                            "solver_energy_evaluations": orchestration.energy_evaluations,
                            "solver_iterations": orchestration.iterations,
                            "solver_accepted_moves": orchestration.accepted_moves,
                            "solver_termination": orchestration.termination_reason,
                            "theta_norm": float(state.Theta.norm()),
                            "theta_diff_from_gt": float(
                                (state.Theta - _gt_graph(len(episode_tasks.specs), dependencies)).norm()),
                            "kappa_norm": float(state.kappa.norm()),
                        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("[llm_poc] no tasks were completed; increase the free-model quota or retry later")
    elapsed = time.perf_counter() - start
    summary = build_summary(frame, cfg, call_count, elapsed)
    summary["environment"] = provenance
    summary["protocol"] = {
        "version": cfg.protocol_version,
        "model_requested": cfg.model,
        "provider_required": cfg.provider,
        "api_endpoint": config.GEMINI_API_URL.format(model=cfg.model),
        "api_key_env_var": cfg.api_key_env,
        "model_catalog_snapshot": cfg.model_catalog_snapshot,
        "provider_routing": "direct API; no router or provider fallback",
        "worker_temperature": cfg.worker_temperature,
        "worker_top_p": cfg.worker_top_p,
        "judge_temperature": cfg.judge_temperature,
        "seed_base": cfg.seed,
        "seed_scheme": config.RNG_SCHEME,
        "worker_max_tokens": cfg.max_worker_tokens,
        "judge_max_tokens": cfg.max_judge_tokens,
        "max_retries": cfg.max_retries,
        "max_resume_extra_attempts": config.MAX_RESUME_EXTRA_ATTEMPTS if live else 0,
        "max_live_http_attempts": args.attempt_cap if live else None,
        "task_bank_sha256": live_manifest.get("task_bank_sha256") if live else None,
        "request_impl_sha256": live_manifest.get("request_impl_sha256") if live else None,
        "retry_policy": (f"up to {cfg.max_retries} retries for 408/500/502/503/504 or "
                         "network timeout; resumed exhausted requests may have up to "
                         f"{config.MAX_RESUME_EXTRA_ATTEMPTS} extra attempts; "
                         "exponential backoff from "
                         f"{config.RETRY_BASE_SEC}s, capped at {config.RETRY_MAX_SEC}s, "
                         "with jitter; honor bounded Retry-After; never retry 429, "
                         "other HTTP errors, or malformed/config errors"),
        "retry_base_sec": config.RETRY_BASE_SEC,
        "retry_max_sec": config.RETRY_MAX_SEC,
        "retry_jitter_ratio": config.RETRY_JITTER_RATIO,
        "max_retry_after_sec": config.MAX_RETRY_AFTER_SEC,
        "progress_journal": "poc_live_progress.jsonl" if live else None,
        "live_run_resumable": live,
        "api_timeout_sec": config.API_TIMEOUT_SEC,
        "minimum_request_interval_sec": config.MIN_REQUEST_INTERVAL_SEC,
        "max_worker_output_chars": cfg.max_worker_output_chars,
        "max_judge_output_chars": cfg.max_judge_output_chars,
        "requested_blocks": list(blocks),
        "episodes_requested": args.episodes,
        "repetitions_requested": cfg.n_repetitions,
        "planned_worker_calls": planned_worker_calls,
        "actual_worker_calls": call_count,
        "live_worker_task_call_cap": config.MAX_LIVE_TASK_CALLS if live else None,
        "live_http_attempt_hard_cap": args.attempt_cap if live else None,
        "resumed_live_run": args.resume,
        "judge_enabled": False,
        "scoring_note": "rubric_proxy is a deterministic proxy, not an LLM or independent quality judge",
        "worker_seed_supported": cfg.worker_seed,
        "judge_seed_supported": cfg.judge_seed,
        "spotcheck_judge_model": cfg.spotcheck_judge_model or None,
        "worker_system_prompts": {slot.name: slot.system_prompt for slot in slots},
        "api_seeds_are_hard_deterministic": False,
    }
    summary["total_judge_calls"] = judge_call_total
    summary["total_judge_failures"] = judge_failure_total
    summary["total_judge_prompt_tokens"] = judge_prompt_tokens_total
    summary["total_judge_completion_tokens"] = judge_completion_tokens_total
    summary["total_judge_tokens"] = judge_prompt_tokens_total + judge_completion_tokens_total
    summary["resolved_model_ids"] = sorted(resolved_model_ids)
    summary["resolved_providers"] = sorted(resolved_providers)
    artifact_paths = metrics.run_artifacts(rows, summary, cfg.output_dir)
    output_dir = Path(cfg.output_dir)
    pd.DataFrame(judge_rows).to_csv(output_dir / "poc_judges.csv", index=False)
    report = metrics.write_report(summary, cfg.output_dir)

    print(f"[llm_poc] done: {len(frame)} task rows, {call_count} worker calls, {elapsed:.1f}s")
    for condition in metrics.CONDITIONS:
        values = summary["per_condition"][condition]
        print(f"  {metrics.CONDITION_OF[condition]}: quality={values['mean_quality']:.3f} "
              f"dep={values['dependency_satisfaction']:.2f} "
              f"realized={values['realized_external_score']:.3f}")
    print(f"[llm_poc] artifacts: {artifact_paths['episodes_csv']}")
    print(f"[llm_poc] report:   {report}")
    return 0


def _gt_graph(size: int, dependencies: list[tuple[int, int]]) -> torch.Tensor:
    graph = torch.zeros(size, size)
    for upstream, downstream in dependencies:
        graph[upstream, downstream] = 1.0
        graph[downstream, upstream] = 1.0
    return graph


def build_summary(df: pd.DataFrame, cfg: config.PoCConfig,
                  calls: int, runtime: float) -> dict:
    per_condition: dict[str, dict] = {}
    for condition in metrics.CONDITIONS:
        subset = df[df.condition == condition]
        episodes = subset.groupby(["block", "episode", "repetition"]).agg(
            quality=("rubric_score", "mean"),
            dependency=("dependency_satisfaction_episode", "first"),
            realized=("realized_external_score_episode", "first"),
        ).reset_index()
        unique_episodes = subset.drop_duplicates(["block", "episode", "repetition"])
        entry = {
            "mean_quality": float(subset.rubric_score.mean()),
            "dependency_satisfaction": float(unique_episodes.dependency_satisfaction_episode.mean()),
            "realized_external_score": float(unique_episodes.realized_external_score_episode.mean()),
            "mean_latency_sec": float(subset.latency_total_sec.mean()),
            "mean_tokens_per_task": float((subset.prompt_tokens + subset.completion_tokens).mean()),
            "worker_failure_rate": float((~subset.worker_ok).mean()),
            "on_specialization_rate": float(subset.on_specialization.mean()),
            "per_block": {},
        }
        for block, group in episodes.groupby("block"):
            entry["per_block"][block] = {
                "mean_quality": float(group.quality.mean()),
                "dependency_satisfaction": float(group.dependency.mean()),
                "realized_external_score": float(group.realized.mean()),
            }
        per_condition[condition] = entry

    episode_rows = df.drop_duplicates(["condition", "block", "episode", "repetition"])
    if (len(episode_rows) < 2 or episode_rows.mean_quality_episode.nunique() < 2 or
            episode_rows.realized_external_score_episode.nunique() < 2):
        correlation = None
    else:
        correlation_value = episode_rows.realized_external_score_episode.corr(
            episode_rows.mean_quality_episode)
        correlation = None if pd.isna(correlation_value) else float(correlation_value)

    return {
        "worker_mode": cfg.worker_mode,
        "scorer": "rubric_proxy" if not cfg.judge_enabled else f"judge:{cfg.judge_model}",
        "model": cfg.model,
        "resolved_models": sorted({value for value in df.model.astype(str) if value}),
        "resolved_providers": sorted({value for value in df.provider.astype(str) if value}),
        "n_conditions": int(df.condition.nunique()),
        "n_blocks": int(df.block.nunique()),
        "n_episodes_per_block": int(df.groupby(["condition", "block"]).episode.nunique().max()),
        "n_tasks": int(df.task_id.nunique()),
        "n_repetitions": int(df.repetition.nunique()),
        "total_worker_calls": calls,
        "total_worker_failures": int((~df.worker_ok).sum()),
        "total_prompt_tokens": int(df.prompt_tokens.sum()),
        "total_completion_tokens": int(df.completion_tokens.sum()),
        "total_judge_tokens": 0,
        "total_judge_calls": 0,
        "total_judge_failures": 0,
        "total_judge_prompt_tokens": 0,
        "total_judge_completion_tokens": 0,
        "total_runtime_sec": runtime,
        "seed": cfg.seed,
        "per_condition": per_condition,
        "transferability_correlation": correlation,
        "status": "hypothesis-generating PoC; not a confirmatory benchmark",
    }


if __name__ == "__main__":
    raise SystemExit(main())
