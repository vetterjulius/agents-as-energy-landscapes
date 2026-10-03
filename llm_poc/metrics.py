"""Metrics and artifact writing for the LLM PoC.

Writes per-episode CSV rows (assignments, outcomes, usage) plus a run-level
JSON summary and a markdown report, mirroring the benchmark's artifact style.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from . import config
from .tasks import EpisodeTasks, declared_dependency_pairs

CONDITIONS = ("baseline", "static_energy", "adaptive_energy")

CONDITION_LABEL = {
    "baseline": "B0' Conventional Greedy (no energy)",
    "static_energy": "B1' Static Energy (fixed Theta_0)",
    "adaptive_energy": "B3' Adaptive Energy (kappa/Theta EMA)",
}

CONDITION_ORDER = ["B0'", "B1'", "B3'"]
CONDITION_OF = {"baseline": "B0'", "static_energy": "B1'", "adaptive_energy": "B3'"}


def external_score(
    episode_tasks: EpisodeTasks,
    X: torch.Tensor,
    scores: list,
    dependency_pairs: list[tuple[int, int]],
) -> dict:
    """Post-hoc 'realized external energy' analogue: quality + dependency
    satisfaction + load balance, mirroring the benchmark's ground-truth terms.

    Higher is better; used for the transferability correlation and for
    condition comparisons (mean quality is the primary LLM-level metric).
    """
    import numpy as np

    quality = float(sum(s.rubric_score for s in scores) / max(1, len(scores)))
    Xf = X.float()
    assigned = {t: int(torch.argmax(Xf[:, t]).item()) for t in range(Xf.shape[1])}
    dep_total = max(1, len(dependency_pairs))
    dep_satisfied = sum(
        1 for up, down in dependency_pairs if assigned[up] == assigned[down])
    workload = Xf.sum(dim=1).tolist()
    load_balance = float(torch.tensor(workload, dtype=torch.float32).std())
    realized = quality + 0.1 * (dep_satisfied / dep_total) - 0.05 * load_balance
    return {
        "mean_quality": quality,
        "dependency_satisfaction": dep_satisfied / dep_total,
        "dependency_satisfied_count": dep_satisfied,
        "dependency_total": dep_total,
        "load_balance_std": load_balance,
        "realized_external_score": realized,
    }


def mean_quality_of(scores: list) -> float:
    return float(sum(s.rubric_score for s in scores) / max(1, len(scores)))


def run_artifacts(rows: list[dict], summary: dict, out_dir: str) -> dict[str, Path]:
    out = Path(out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "poc_episodes.csv", index=False)
    (out / "poc_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"episodes_csv": out / "poc_episodes.csv", "summary_json": out / "poc_summary.json"}


def write_report(summary: dict, out_dir: str) -> Path:
    """Human-readable markdown report with the pre-registered comparisons."""
    out = Path(out_dir)
    per_cond = summary["per_condition"]
    lines = [
        "# LLM PoC Results",
        "",
        f"- worker mode: `{summary['worker_mode']}` · scorer: `{summary['scorer']}` · "
        f"model: `{summary['model']}`",
        f"- scale: {summary['n_conditions']} conditions × {summary['n_blocks']} blocks × "
        f"{summary['n_episodes_per_block']} episodes × {summary['n_tasks']} tasks × "
        f"{summary['n_repetitions']} repetitions",
        f"- total worker calls: {summary['total_worker_calls']} "
        f"(failures: {summary['total_worker_failures']})",
        f"- tokens: prompt {summary['total_prompt_tokens']:,} / completion "
        f"{summary['total_completion_tokens']:,} (workers) + {summary['total_judge_tokens']:,} (judges)",
        "",
        "> Status: hypothesis-generating external validation. Not a confirmatory benchmark;",
        "> no multiplicity-corrected claims are drawn from this experiment.",
        "",
        "## Condition-level results",
        "",
        "| Condition | mean quality | dep. satisfaction | realized score | mean latency (s) | mean tokens/task |",
        "|---|---|---|---|---|---|",
    ]
    for cond in CONDITIONS:
        c = per_cond[cond]
        lines.append(
            f"| {CONDITION_LABEL[cond]} | {c['mean_quality']:.3f} | "
            f"{c['dependency_satisfaction']:.2f} | {c['realized_external_score']:.3f} | "
            f"{c['mean_latency_sec']:.2f} | {c['mean_tokens_per_task']:.0f} |")
    lines += [
        "",
        "## Pre-registered expectations (from the frozen benchmark)",
        "",
    ]
    b, s, a = (per_cond[c] for c in CONDITIONS)
    e1 = s["mean_quality"] >= b["mean_quality"]
    e2 = a["mean_quality"] <= s["mean_quality"] + 1e-9 and \
        a["dependency_satisfaction"] <= s["dependency_satisfaction"] + 1e-9
    lines += [
        f"1. Static energy ≥ baseline on quality: "
        f"{'OBSERVED' if e1 else 'NOT observed'} "
        f"({s['mean_quality']:.3f} vs {b['mean_quality']:.3f}).",
        f"2. Adaptive shows no benefit under the stationary task distribution "
        f"(quality {a['mean_quality']:.3f} vs static {s['mean_quality']:.3f}; "
        f"dep. satisfaction {a['dependency_satisfaction']:.2f} vs {s['dependency_satisfaction']:.2f}) "
        f"— consistent with the benchmark's stationary-drift warning: "
        f"{'yes' if e2 else 'no'}.",
        "3. Realized external score vs quality correlation: see "
        "`poc_summary.json` -> transferability_correlation.",
        "",
        "## Per-block detail",
        "",
        "| Condition | block | mean quality | dep. satisfaction | realized |",
        "|---|---|---|---|---|",
    ]
    for cond in CONDITIONS:
        for block, v in per_cond[cond]["per_block"].items():
            lines.append(
                f"| {CONDITION_OF[cond]} | {block} | {v['mean_quality']:.3f} | "
                f"{v['dependency_satisfaction']:.2f} | {v['realized_external_score']:.3f} |")
    lines += [
        "",
        f"Report generated {time.strftime('%Y-%m-%dT%H:%M:%S')}.",
    ]
    path = out / "poc_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
