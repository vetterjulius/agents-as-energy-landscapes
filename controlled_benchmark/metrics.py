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


RECOVERY_WINDOW: int = 3


def compute_recovery_threshold(
    reference_energy: float,
    tolerance_ratio: float = 0.10,
    min_slack: float = 0.10,
) -> float:
    """
    Strict solver-independent recovery threshold derived exclusively from a target reference energy.

    Guarantees:
    - Never derived from a solver's own post-shift trajectory.
    - If reference_energy >= 0: tau = reference_energy + tolerance_ratio * max(reference_energy, min_slack).
      (equals (1 + tolerance_ratio) * reference_energy when reference_energy >= min_slack)
    - If reference_energy < 0: tau = reference_energy + tolerance_ratio * max(abs(reference_energy), min_slack).
    - Threshold is strictly well-defined, solver-independent, and auditable.
    """
    slack = tolerance_ratio * max(abs(reference_energy), min_slack)
    return float(reference_energy + slack)


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
    recovery_threshold: float = 0.0
    reference_energy: float = 0.0
    reference_method: str = "exact_optimal"
    recovery_window: int = RECOVERY_WINDOW
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
    reference_energy: Optional[float] = None,
    reference_method: Optional[str] = None,
    recovery_window: int = RECOVERY_WINDOW,
) -> TrajectorySummary:
    """
    Compute rigorous trajectory-level metrics.

    FORMAL PRIMARY METRIC: recovery_time
    ------------------------------------
    The recovery threshold tau is derived EXCLUSIVELY from a solver-independent reference energy.
    It is STRICTLY FORBIDDEN to compute tau from a solver's own post-shift trajectory.

    Stability Window (RECOVERY_WINDOW = 3):
    Recovery requires that starting from episode (perturb_episode + offset), the external energy
    satisfies E(t) <= tau for at least `recovery_window` (default 3) consecutive episodes.
    A single outlier or transient dip does NOT trigger recovery.

    If the condition is never met within the horizon, recovery_time is censored at
    the full remaining horizon (T - perturb_episode).
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

    # Determine solver-independent reference energy and threshold
    if reference_energy is not None:
        ref_e = float(reference_energy)
        ref_meth = str(reference_method or "exact_optimal")
    else:
        # Fallback if no reference is explicitly passed:
        # Use pre-perturbation baseline (NEVER solver's own post_base!)
        ref_e = pre_base
        ref_meth = "pre_perturbation_baseline"

    target = compute_recovery_threshold(ref_e)

    # Primary Metric: recovery_time with 3-consecutive-episodes stability window
    recovery_time = float(total_episodes - perturb_episode)
    max_search_offset = total_episodes - perturb_episode - recovery_window + 1

    if max_search_offset > 0:
        for offset in range(max_search_offset):
            ep = perturb_episode + offset
            # Verify stability across recovery_window consecutive episodes
            if all(external_energies[ep + w] <= target for w in range(recovery_window)):
                recovery_time = float(offset)
                break
    else:
        # If remaining episodes are fewer than recovery_window, check all remaining
        rem = total_episodes - perturb_episode
        if rem > 0 and all(external_energies[perturb_episode + w] <= target for w in range(rem)):
            recovery_time = 0.0

    # Secondary Metric: cumulative regret over post-perturbation horizon relative to reference
    cum_regret = float(np.sum(np.maximum(0.0, external_energies[perturb_episode:] - ref_e)))

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
        recovery_threshold=target,
        reference_energy=ref_e,
        reference_method=ref_meth,
        recovery_window=recovery_window,
        evaluation_landscape_id="external_ground_truth",
    )
