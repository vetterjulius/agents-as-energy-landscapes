"""LLM PoC runner: orchestrate -> execute -> score -> log, for 3 conditions x
2 blocks x 3 episodes x 3 repetitions.

Usage:
    python -m llm_poc.run                      # dry-run validation (mock workers, no judge calls)
    python -m llm_poc.run --worker-mode openrouter   # live OpenRouter execution
    python -m llm_poc.run --worker-mode openrouter --no-judge   # live workers, rubric-proxy scoring
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from . import config, metrics, workers
from .orchestration import orchestrate
from .tasks import build_agent_slots, build_episode, declared_dependency_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the LLM orchestration PoC.")
    p.add_argument("--worker-mode", choices=("mock", "openrouter"), default="mock")
    p.add_argument("--no-judge", action="store_true",
                   help="disable LLM judging; use deterministic rubric proxy")
    p.add_argument("--repetitions", type=int, default=config.N_REPETITIONS)
    p.add_argument("--episodes", type=int, default=config.N_EPISODES)
    p.add_argument("--output-dir", default="results/llm_poc")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--limit-calls", type=int, default=None,
                   help="safety valve: stop after N worker calls (live mode)")
    return p.parse_args()


def main() -> int:
    try:  # optional: pick up OPENROUTER_API_KEY from .env like the legacy PoC
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    args = parse_args()
    cfg = config.PoCConfig(
        worker_mode=args.worker_mode,
        judge_enabled=False if args.no_judge else None,  # None => auto by worker mode
        n_repetitions=args.repetitions,
        seed=args.seed,
        output_dir=args.output_dir,
        model=args.model,
    )
    cfg.require_api()
    slots = build_agent_slots()
    conditions = metrics.CONDITIONS

    rows: list[dict] = []
    call_count = 0
    t_start = time.perf_counter()
    print(f"[llm_poc] mode={cfg.worker_mode} judge={cfg.judge_enabled} "
          f"repetitions={cfg.n_repetitions} -> {cfg.output_dir}")

    for condition in conditions:
        # Independent adaptation state per (condition, block) — adaptation only
        # within a block's trajectory, mirroring the benchmark's per-trajectory runs.
        for block in config.BLOCKS:
            state = None
            for episode in range(args.episodes):
                episode_tasks = build_episode(block, episode)
                deps = declared_dependency_pairs(block, episode)
                if state is None:
                    from .orchestration import make_initial_state
                    state = make_initial_state(episode_tasks, deps)
                rng_seed = cfg.seed * 100000 + episode * 1000
                orch, next_state = orchestrate(
                    condition, slots, episode_tasks, deps, state, rng_seed)
                state = next_state

                for repetition in range(cfg.n_repetitions):
                    episode_seed = cfg.seed + repetition * 977 + episode * 31
                    execution = workers.execute_episode(
                        episode_tasks, slots, orch.X, cfg, repetition, episode_seed)
                    call_count += len(execution.outputs)
                    if args.limit_calls and call_count > args.limit_calls:
                        print(f"[llm_poc] --limit-calls reached ({args.limit_calls}); stopping.")
                        break

                    ext = metrics.external_score(episode_tasks, orch.X, execution.scores, deps)
                    assignments = {t: int(a) for t, a in enumerate(
                        torch.argmax(orch.X.float(), dim=0).tolist())}
                    # dependency rows only, for the per-episode dep-satisfaction rate
                    dep_rows = [(t, o, s) for t, o, s in
                                zip(episode_tasks.specs, execution.outputs, execution.scores)
                                if t.depends_on is not None]
                    dep_rate_episode = (
                        sum(1 for t, _, _ in dep_rows
                            if assignments[t.depends_on] == assignments[t.id]) / len(dep_rows)
                        if dep_rows else float("nan"))
                    for task, out, sc in zip(episode_tasks.specs, execution.outputs, execution.scores):
                        rows.append({
                            "condition": condition, "condition_label": metrics.CONDITION_OF[condition],
                            "block": block, "episode": episode, "repetition": repetition,
                            "task_id": task.id, "task_kind": task.kind,
                            "depends_on": task.depends_on,
                            "assigned_agent": assignments[task.id],
                            "agent_name": slots[assignments[task.id]].name,
                            "on_specialization": task.kind == slots[assignments[task.id]].specialization,
                            "dependency_coexecuted": (
                                task.depends_on is not None and
                                assignments[task.depends_on] == assignments[task.id]),
                            "worker_ok": out.ok,
                            "worker_error": out.error or "",
                            "retries": out.retries,
                            "latency_total_sec": out.latency_total_sec,
                            "prompt_tokens": out.usage.prompt_tokens,
                            "completion_tokens": out.usage.completion_tokens,
                            "rubric_score": sc.rubric_score,
                            "judge_raw": sc.judge_raw,
                            "scorer": sc.scorer,
                            "spotcheck_score": sc.spotcheck_score if sc.spotcheck_score is not None else "",
                            "spotcheck_scorer": sc.spotcheck_scorer or "",
                            "mean_quality_episode": ext["mean_quality"],
                            "dependency_satisfaction_episode": dep_rate_episode,
                            "realized_external_score_episode": ext["realized_external_score"],
                            "dep_available": (task.depends_on is None) or (
                                assignments[task.depends_on] == assignments[task.id]),
                            "solver_energy": orch.solver_energy,
                            "solver_energy_evaluations": orch.energy_evaluations,
                            "solver_iterations": orch.iterations,
                            "solver_accepted_moves": orch.accepted_moves,
                            "solver_termination": orch.termination_reason,
                            "theta_norm": float(state.Theta.norm()),
                            "theta_diff_from_gt": float(
                                (state.Theta - _gt_graph(len(episode_tasks.specs), deps)).norm()),
                            "kappa_norm": float(state.kappa.norm()),
                        })
                if args.limit_calls and call_count > args.limit_calls:
                    break
            if args.limit_calls and call_count > args.limit_calls:
                break
        if args.limit_calls and call_count > args.limit_calls:
            break

    df = pd.DataFrame(rows)
    summary = build_summary(df, cfg, call_count, time.perf_counter() - t_start)
    paths = metrics.run_artifacts(rows, summary, cfg.output_dir)
    report = metrics.write_report(summary, cfg.output_dir)

    print(f"[llm_poc] done: {len(df)} task rows, {call_count} worker calls, "
          f"{summary['total_runtime_sec']:.1f}s")
    for cond in metrics.CONDITIONS:
        c = summary["per_condition"][cond]
        print(f"  {metrics.CONDITION_OF[cond]}: quality={c['mean_quality']:.3f} "
              f"dep={c['dependency_satisfaction']:.2f} realized={c['realized_external_score']:.3f}")
    print(f"[llm_poc] artifacts: {paths['episodes_csv']}")
    print(f"[llm_poc] report:   {report}")
    return 0


def _gt_graph(m: int, deps: list[tuple[int, int]]) -> torch.Tensor:
    g = torch.zeros(m, m)
    for up, down in deps:
        g[up, down] = 1.0
        g[down, up] = 1.0
    return g


def build_summary(df: pd.DataFrame, cfg: config.PoCConfig, calls: int,
                  runtime: float) -> dict:
    per_condition: dict[str, dict] = {}
    for cond in metrics.CONDITIONS:
        sub = df[df.condition == cond]
        eps = sub.groupby(["block", "episode", "repetition"]).agg(
            quality=("rubric_score", "mean"),
            dep=("dependency_satisfaction_episode", "first"),
            realized=("realized_external_score_episode", "first"),
            latency=("latency_total_sec", "mean"),
            tokens=("completion_tokens", "mean"),
        ).reset_index()
        entry = {
            "mean_quality": float(sub.rubric_score.mean()),
            "dependency_satisfaction": float(
                sub.drop_duplicates(["block", "episode", "repetition"])
                .dependency_satisfaction_episode.mean()),
            "realized_external_score": float(
                sub.drop_duplicates(["block", "episode", "repetition"])
                .realized_external_score_episode.mean()),
            "mean_latency_sec": float(sub.latency_total_sec.mean()),
            "mean_tokens_per_task": float(
                (sub.prompt_tokens + sub.completion_tokens).mean()),
            "worker_failure_rate": float((~sub.worker_ok).mean()),
            "on_specialization_rate": float(sub.on_specialization.mean()),
            "per_block": {},
        }
        for block, g in eps.groupby("block"):
            entry["per_block"][block] = {
                "mean_quality": float(g.quality.mean()),
                "dependency_satisfaction": float(g.dep.mean()),
                "realized_external_score": float(g.realized.mean()),
            }
        per_condition[cond] = entry

    # Transferability: realized external score vs mean quality across episode-runs
    eps_agg = df.drop_duplicates(["condition", "block", "episode", "repetition"])
    corr = float(eps_agg.realized_external_score_episode.corr(
        eps_agg.mean_quality_episode))
    if pd.isna(corr):
        corr = None
    summary = {
        "worker_mode": cfg.worker_mode,
        "scorer": "rubric_proxy" if (cfg.worker_mode == "mock" or not cfg.judge_enabled)
                  else f"judge:{cfg.judge_model}",
        "model": cfg.model,
        "n_conditions": len(metrics.CONDITIONS),
        "n_blocks": len(config.BLOCKS),
        "n_episodes_per_block": int(df.groupby(["condition", "block"]).episode.nunique().max()),
        "n_tasks": int(df.task_id.nunique()),
        "n_repetitions": cfg.n_repetitions,
        "total_worker_calls": calls,
        "total_worker_failures": int((~df.worker_ok).sum()),
        "total_prompt_tokens": int(df.prompt_tokens.sum()),
        "total_completion_tokens": int(df.completion_tokens.sum()),
        "total_judge_tokens": 0,  # judge usage not per-row; see note in report
        "total_runtime_sec": runtime,
        "seed": cfg.seed,
        "per_condition": per_condition,
        "transferability_correlation": corr,
        "status": "hypothesis-generating PoC; not a confirmatory benchmark",
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
