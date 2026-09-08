from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch

from landscape import Landscape


@dataclass
class EpisodeRecord:
    """Detailed observation recorded at each episode."""

    episode: int
    internal_energy: float
    external_energy: float
    energy_evaluations: int
    accepted_moves: int
    iterations: int
    runtime_sec: float
    termination_reason: str
    solver_status: str
    is_optimal: Optional[bool]
    mip_gap: Optional[float]
    timeout: bool
    fallback_used: bool
    reconfig_cost: float
    constraint_violations: float
    coordination_score: float
    load_balance: float
    kappa_norm: float
    theta_diff_norm: float


@dataclass
class TrajectorySummary:
    """Summary metrics over a multi-episode scenario trajectory."""

    # Primary Metric
    recovery_time: float

    # Secondary Metrics
    perf_drop: float
    cumulative_regret: float
    pre_base_energy: float
    post_base_energy: float
    final_energy: float
    mean_external_energy: float
    mean_internal_energy: float
    convergence: float
    stability: float
    adaptation_magnitude_kappa: float
    adaptation_magnitude_theta: float
    mean_constraint_violations: float
    total_energy_evaluations: int
    total_runtime_sec: float
    termination_reasons_summary: str
    all_episodes_optimal: bool
    fallback_used: bool
    evaluation_landscape_id: str = "external_ground_truth"


def compute_reconfig_cost(prev_X: Optional[torch.Tensor], curr_X: torch.Tensor) -> float:
    """Fraction of tasks reassigned to different agents."""
    if prev_X is None or prev_X.shape != curr_X.shape:
        return 0.0
    return float((prev_X - curr_X).abs().sum().item() / 2.0)


def compute_constraint_violations(problem_C: torch.Tensor, X: torch.Tensor) -> float:
    """Sum of co-assignment costs incurred for conflicting tasks assigned to the same agent."""
    co = X.T @ X
    conflicts = (problem_C > 0) * co
    return float(conflicts.sum().item())


def compute_coordination_score(interaction_graph: torch.Tensor, X: torch.Tensor) -> float:
    """Sum of synergistic interaction weights co-assigned to the same agent."""
    co = X.T @ X
    synergies = (interaction_graph > 0) * co
    return float(synergies.sum().item())


def compute_load_balance(X: torch.Tensor) -> float:
    """Workload standard deviation across agents."""
    workload = X.sum(dim=1)
    return float(torch.std(workload).item())


def compute_trajectory_summary(
    records: List[EpisodeRecord],
    scenario_id: str,
    perturb_episode: int,
    pre_window: int = 10,
    post_window: int = 10,
) -> TrajectorySummary:
    """
    Compute rigorous trajectory-level metrics.

    FORMAL PRIMARY METRIC: recovery_time
    ------------------------------------
    Let E(t) be external ground truth energy at episode t in [0, T-1].
    Pre-perturbation baseline E_pre is the mean over [max(0, perturb_episode - pre_window), perturb_episode - 1].
    Post-perturbation baseline E_post is the mean over [T - post_window, T - 1].

    Target recovery threshold tau:
    - If scenario == 'Task Shift' (permanent shift): tau = 1.1 * E_post.
    - Otherwise (Capability Drift, Dependency Change, Stationary): tau = 1.1 * E_pre.

    recovery_time is the first episode offset k in [0, T - perturb_episode - 1] such that:
        E(perturb_episode + k) <= tau.
    If E never reaches <= tau within the remaining episodes, recovery_time defaults to
    the full remaining horizon (T - perturb_episode) [censored recovery].
    """
    total_episodes = len(records)
    external_energies = np.array([r.external_energy for r in records])
    internal_energies = np.array([r.internal_energy for r in records])

    # Pre-perturbation baseline window
    pre_start = max(0, perturb_episode - pre_window)
    pre_end = max(1, perturb_episode)
    pre_base = float(np.mean(external_energies[pre_start:pre_end]))

    # Post-perturbation late window
    post_start = max(perturb_episode, total_episodes - post_window)
    post_base = float(np.mean(external_energies[post_start:]))

    # Immediate performance drop at perturbation
    ep_pre_idx = max(0, perturb_episode - 1)
    ep_post_idx = min(total_episodes - 1, perturb_episode)
    perf_drop = float(max(0.0, external_energies[ep_post_idx] - external_energies[ep_pre_idx]))

    # Primary Metric: recovery_time
    is_permanent = (scenario_id == "Task Shift")
    target = 1.1 * post_base if is_permanent else 1.1 * pre_base

    recovery_time = float(total_episodes - perturb_episode)
    for offset, ep in enumerate(range(perturb_episode, total_episodes)):
        if external_energies[ep] <= target:
            recovery_time = float(offset)
            break

    # Secondary Metric: cumulative regret over post-perturbation horizon
    cum_regret = float(np.sum(np.maximum(0.0, external_energies[perturb_episode:] - pre_base)))

    # Secondary Metric: late-window convergence (energy std in late episodes)
    convergence = float(np.std(external_energies[post_start:]))

    # Secondary Metric: late-window stability (mean reconfiguration cost)
    reconfigs = [r.reconfig_cost for r in records[post_start:]]
    stability = float(np.mean(reconfigs)) if reconfigs else 0.0

    # Final magnitudes
    last_record = records[-1]
    adaptation_magnitude_kappa = last_record.kappa_norm
    adaptation_magnitude_theta = last_record.theta_diff_norm

    # Aggregate solver statistics
    total_evals = sum(r.energy_evaluations for r in records)
    total_runtime = sum(r.runtime_sec for r in records)
    mean_conflicts = float(np.mean([r.constraint_violations for r in records]))
    reasons = set(r.termination_reason for r in records)
    termination_summary = "; ".join(sorted(reasons))

    all_optimal = all(bool(r.is_optimal) for r in records)
    any_fallback = any(bool(r.fallback_used) for r in records)

    return TrajectorySummary(
        recovery_time=recovery_time,
        perf_drop=perf_drop,
        cumulative_regret=cum_regret,
        pre_base_energy=pre_base,
        post_base_energy=post_base,
        final_energy=float(external_energies[-1]),
        mean_external_energy=float(np.mean(external_energies)),
        mean_internal_energy=float(np.mean(internal_energies)),
        convergence=convergence,
        stability=stability,
        adaptation_magnitude_kappa=adaptation_magnitude_kappa,
        adaptation_magnitude_theta=adaptation_magnitude_theta,
        mean_constraint_violations=mean_conflicts,
        total_energy_evaluations=total_evals,
        total_runtime_sec=total_runtime,
        termination_reasons_summary=termination_summary,
        all_episodes_optimal=all_optimal,
        fallback_used=any_fallback,
        evaluation_landscape_id="external_ground_truth",
    )
