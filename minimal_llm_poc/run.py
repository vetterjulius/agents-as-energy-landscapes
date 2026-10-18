"""Main runner for the Minimal LLM PoC.

Executes the 3 orchestrators (baseline, static_energy_sa, dynamic_energy_sa)
across the 6 evolutionary episodes of the Dependency Change scenario.

Usage:
  python -m minimal_llm_poc.run --mock               # Fast local verification without API
  python -m minimal_llm_poc.run                      # Live run using Gemini API
  python -m minimal_llm_poc.run --condition baseline # Run single condition
"""
from __future__ import annotations

import argparse
import json
import os
from dotenv import load_dotenv
import time
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import torch

from . import api, config, evaluation, scenario
from .orchestrators import (
    BaselineOrchestrator,
    DynamicEnergySAOrchestrator,
    StaticEnergySAOrchestrator,
    make_problem_context,
)


def get_orchestrator(condition: str):
    if condition == "baseline":
        return BaselineOrchestrator()
    elif condition == "static_energy_sa":
        return StaticEnergySAOrchestrator()
    elif condition == "dynamic_energy_sa":
        return DynamicEnergySAOrchestrator()
    else:
        raise ValueError(f"Unknown condition: {condition}")


def run_experiment(
    conditions: List[str],
    *,
    mock: bool = False,
    output_dir: str = config.OUTPUT_DIR,
    seed: int = 42,
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Minimal LLM PoC ===")
    print(f"Mode: {'MOCK (no API calls)' if mock else 'LIVE (Gemini API)'}")
    print(f"Conditions: {conditions}")
    print(f"Episodes: {config.NUM_EPISODES} (Phases: {config.NUM_PHASES})")
    print(f"Tasks per episode: {config.NUM_TASKS_PER_EPISODE}")
    print(f"Agents: {[a.id for a in scenario.AGENTS]}")
    print(f"Judge Model: {config.JUDGE_MODEL} (disjoint: {config.JUDGE_MODEL not in set(config.AGENT_MODELS.values())})")
    print("-" * 50)

    task_records: List[Dict[str, Any]] = []
    episode_records: List[Dict[str, Any]] = []

    total_start = time.perf_counter()

    for condition in conditions:
        print(f"\n>>> Running condition: {condition}")
        orch = get_orchestrator(condition)

        for episode in range(config.NUM_EPISODES):
            phase = scenario.get_phase(episode)
            tasks = scenario.get_tasks_for_episode(episode)
            context = make_problem_context(tasks)

            print(f"  [Ep {episode} | Phase {phase}] Assigning tasks...")
            orch_res = orch.assign(episode, tasks, seed=seed)

            # Step 1: Execute all 8 tasks with their assigned agent
            raw_evals: List[evaluation.TaskEvaluation] = []
            for t_idx, task in enumerate(tasks):
                agent_id = orch_res.assignment_map[t_idx]
                agent_def = next(a for a in scenario.AGENTS if a.id == agent_id)

                print(f"    Task {task.slot_index} ({task.id}) -> {agent_id} ({agent_def.model})...", end="", flush=True)

                # Generate output from assigned agent
                gen = api.generate(
                    prompt=task.prompt,
                    system_prompt=agent_def.system_prompt,
                    model=agent_def.model,
                    mock=mock,
                )

                # Evaluate single task
                ev = evaluation.evaluate_task(
                    task=task,
                    output=gen.text,
                    assigned_agent=agent_id,
                    mock=mock,
                )
                raw_evals.append(ev)
                print(f" [{ev.eval_method}] base={ev.base_score}")

            # Step 2: Compute co-assignment synergy bonuses
            final_evals = evaluation.compute_episode_evaluations(
                episode=episode,
                evaluations=raw_evals,
                assignment_map=orch_res.assignment_map,
            )

            # Step 3: Trigger post-episode online adaptation (for dynamic SA)
            orch.adapt(episode, orch_res.assignment_matrix, context)

            # Record metrics
            ep_base_score = sum(e.base_score for e in final_evals) / len(final_evals)
            ep_synergy_bonus = sum(e.synergy_bonus for e in final_evals)
            ep_total_score = sum(e.total_score for e in final_evals)

            ep_record = {
                "condition": condition,
                "episode": episode,
                "phase": phase,
                "internal_energy": orch_res.internal_energy,
                "ground_truth_energy": orch_res.ground_truth_energy,
                "mean_base_score": ep_base_score,
                "total_synergy_bonus": ep_synergy_bonus,
                "total_score": ep_total_score,
                "evaluations_used": orch_res.evaluations_used,
                "solver_status": orch_res.solver_status,
                "assignments": json.dumps(orch_res.assignment_map),
            }
            episode_records.append(ep_record)

            print(
                f"  --> Episode {episode} Summary: GT_Energy={orch_res.ground_truth_energy:.4f} | "
                f"BaseScore={ep_base_score:.2f} | SynergyBonus={ep_synergy_bonus:.1f} | TotalScore={ep_total_score:.2f}"
            )

            for ev in final_evals:
                task_records.append({
                    "condition": condition,
                    "episode": episode,
                    "phase": phase,
                    "slot_index": ev.slot_index,
                    "task_id": ev.task_id,
                    "assigned_agent": ev.assigned_agent,
                    "eval_method": ev.eval_method,
                    "base_score": ev.base_score,
                    "synergy_bonus": ev.synergy_bonus,
                    "total_score": ev.total_score,
                    "details": ev.details,
                    "raw_output": ev.raw_output[:200] + ("..." if len(ev.raw_output) > 200 else ""),
                })

    elapsed_total = time.perf_counter() - total_start

    # Save output artifacts
    episodes_df = pd.DataFrame(episode_records)
    episodes_df.to_csv(out_path / "episodes.csv", index=False)

    tasks_df = pd.DataFrame(task_records)
    tasks_df.to_csv(out_path / "task_traces.csv", index=False)

    # Compute high-level summary
    summary_by_condition = {}
    for cond in conditions:
        cdf = episodes_df[episodes_df["condition"] == cond]
        summary_by_condition[cond] = {
            "mean_ground_truth_energy": float(cdf["ground_truth_energy"].mean()),
            "mean_base_score": float(cdf["mean_base_score"].mean()),
            "total_synergy_bonus": float(cdf["total_synergy_bonus"].sum()),
            "total_score": float(cdf["total_score"].sum()),
            "p1_mean_energy": float(cdf[cdf["phase"] == 0]["ground_truth_energy"].mean()),
            "p2_mean_energy": float(cdf[cdf["phase"] == 1]["ground_truth_energy"].mean()),
            "p3_mean_energy": float(cdf[cdf["phase"] == 2]["ground_truth_energy"].mean()),
        }

    summary_payload = {
        "status": "completed",
        "mock": mock,
        "elapsed_seconds": elapsed_total,
        "num_episodes": config.NUM_EPISODES,
        "num_tasks_per_episode": config.NUM_TASKS_PER_EPISODE,
        "models_used": {
            "agents": config.AGENT_MODELS,
            "judge": config.JUDGE_MODEL,
        },
        "conditions": summary_by_condition,
    }

    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Minimal LLM PoC Completed in {elapsed_total:.2f}s")
    print(f"Results written to: {out_path.resolve()}")
    print("=" * 50)
    for cond, stats in summary_by_condition.items():
        print(f"  Condition: {cond}")
        print(f"    Mean GT Energy:     {stats['mean_ground_truth_energy']:.4f} (lower is better)")
        print(f"    Total Synergy Bonus: {stats['total_synergy_bonus']:.1f}")
        print(f"    Total Task Score:   {stats['total_score']:.2f}")

    return summary_payload


def main():
    load_dotenv()  # Load environment variables from .env file if present
    parser = argparse.ArgumentParser(description="Run Minimal LLM PoC")
    parser.add_argument("--mock", action="store_true", help="Run with mock LLM outputs")
    parser.add_argument(
        "--condition",
        choices=["all", "baseline", "static_energy_sa", "dynamic_energy_sa"],
        default="all",
        help="Orchestration condition to run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR, help="Results directory")
    args = parser.parse_args()

    conditions = (
        list(config.CONDITIONS)
        if args.condition == "all"
        else [args.condition]
    )

    run_experiment(
        conditions=conditions,
        mock=args.mock,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
