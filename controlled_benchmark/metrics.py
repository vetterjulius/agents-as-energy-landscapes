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
    theta_norm: float = 0.0
    # Per-episode adaptation delta norms (0.0 at episode 0 — no prior state)
    delta_kappa_norm: float = 0.0
    delta_theta_norm: float = 0.0


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


def compute_ppee(
    records: List[EpisodeRecord],
    perturb_episode: int,
    post_ppee_window: int,
    reference_energy: float,
) -> float:
    """
    Post-Perturbation Excess Energy (PPEE) over the first `post_ppee_window` post-perturbation episodes.

    PPEE = mean( max(0, external_E_t - reference_energy)
                 for t in [perturb_episode, perturb_episode + post_ppee_window) )

    Uses external_energy exclusively.  max(0, ...) guards against floating-point
    artefacts that could yield trivially negative excess values.

    The reference_energy must be the ground-truth exact optimum of the post-perturbation
    regime — it is NEVER derived from the adaptive solver's own landscape.
    """
    external_energies = np.array([r.external_energy for r in records])
    total = len(external_energies)
    start = perturb_episode
    end = min(start + post_ppee_window, total)
    if start >= total or start >= end:
        return 0.0
    window = external_energies[start:end]
    excess = np.maximum(0.0, window - reference_energy)
    return float(np.mean(excess))


def compute_cumulative_excess_energy(
    records: List[EpisodeRecord],
    perturb_episode: int,
    reference_energy: float,
) -> float:
    """
    Cumulative Excess Energy (CEE) over the entire post-perturbation horizon.

    CEE = sum( max(0, external_E_t - reference_energy)
               for t in [perturb_episode, total_episodes) )

    Named 'cumulative_excess_energy' rather than 'cumulative_regret' to avoid
    conflation with online-learning regret (which requires a changing optimal policy).
    Here the post-perturbation regime is stationary, so the reference is constant.
    """
    external_energies = np.array([r.external_energy for r in records])
    if perturb_episode >= len(external_energies):
        return 0.0
    post = external_energies[perturb_episode:]
    return float(np.sum(np.maximum(0.0, post - reference_energy)))


@dataclass
class TrajectorySummary:
    """Summary metrics over a multi-episode scenario trajectory."""

    # PRIMARY metric — Post-Perturbation Excess Energy over first post_ppee_window episodes
    ppee_10: float

    # SECONDARY metrics
    cumulative_excess_energy: float  # CEE over full post-perturbation horizon
    perf_drop: float
    recovery_time: float
    pre_base_energy: float
    post_base_energy: float
    final_energy: float
    mean_external_energy: float
    mean_internal_energy: float
    convergence: float
    stability: float
    adaptation_magnitude_kappa: float
    adaptation_magnitude_theta: float
    # Mechanism metrics: Δ‖κ‖ and Δ‖Θ‖ from perturb_episode to perturb_episode+10
    kappa_change_post: float
    theta_change_post: float
    mean_co_assignment_conflicts: float  # renamed from mean_constraint_violations
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

    # Legacy alias so that existing callers that read cumulative_regret still work
    @property
    def cumulative_regret(self) -> float:
        """Legacy alias for cumulative_excess_energy. New code should use cumulative_excess_energy."""
        return self.cumulative_excess_energy

    # Legacy alias for mean_co_assignment_conflicts
    @property
    def mean_constraint_violations(self) -> float:
        """Legacy alias. New code should use mean_co_assignment_conflicts."""
        return self.mean_co_assignment_conflicts


def compute_reconfig_cost(prev_X: Optional[torch.Tensor], curr_X: torch.Tensor) -> float:
    """Fraction of tasks reassigned to different agents."""
    if prev_X is None or prev_X.shape != curr_X.shape:
        return 0.0
    return float((prev_X - curr_X).abs().sum().item() / 2.0)


def compute_constraint_violations(problem_C: torch.Tensor, X: torch.Tensor) -> float:
    """
    Sum of co-assignment weights incurred for task-pairs assigned to the same agent
    where co_assignment_costs > 0.

    NOTE: this is *not* a classical hard-constraint violation count.  It measures
    co-assignment conflict exposure induced by the cost matrix C.  The preferred name
    in new outputs is 'co_assignment_conflicts'.  This function is kept for backward
    compatibility with existing tests and callers.
    """
    co = X.T @ X
    conflicts = (problem_C > 0) * co
    return float(conflicts.sum().item())


# Preferred alias for new code — identical computation, clearer semantics
compute_co_assignment_conflicts = compute_constraint_violations


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
    post_ppee_window: int = 10,
    reference_energy: Optional[float] = None,
    reference_method: Optional[str] = None,
    recovery_window: int = RECOVERY_WINDOW,
    kappa_change_post: Optional[float] = None,
    theta_change_post: Optional[float] = None,
) -> TrajectorySummary:
    """
    Compute rigorous trajectory-level metrics.

    PRIMARY METRIC: ppee_10
    -----------------------
    Post-Perturbation Excess Energy over the first post_ppee_window episodes after perturbation.
    Uses external_energy and the ground-truth reference exclusively.  max(0,...) guards
    against floating-point artefacts.

    SECONDARY METRIC: recovery_time
    --------------------------------
    Retained for exploratory analysis.  Recovery threshold tau is derived EXCLUSIVELY from
    a solver-independent reference energy.  It is STRICTLY FORBIDDEN to compute tau from a
    solver's own post-shift trajectory.

    Stability Window (RECOVERY_WINDOW = 3):
    Recovery requires that starting from episode (perturb_episode + offset), the external energy
    satisfies E(t) <= tau for at least `recovery_window` (default 3) consecutive episodes.
    A single outlier or transient dip does NOT trigger recovery.

    If the condition is never met within the horizon, recovery_time is censored at
    the full remaining horizon (T - perturb_episode).

    NOTE: The current problem instance is directly observed by the solver each episode.
    Adaptation therefore does not provide hidden change detection — it provides historical
    landscape information that may or may not improve optimization quality.

    Each episode starts from the same deterministic initialization policy.
    Previous episode solutions are NOT used as warm starts.
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

    # PRIMARY Metric: PPEE over first post_ppee_window post-perturbation episodes
    ppee = compute_ppee(records, perturb_episode, post_ppee_window, ref_e)

    # Secondary Metric: cumulative excess energy over entire post-perturbation horizon
    cee = compute_cumulative_excess_energy(records, perturb_episode, ref_e)

    # Secondary Metric: recovery_time with 3-consecutive-episodes stability window
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

    # Secondary Metric: late-window convergence (energy std in late episodes)
    convergence = float(np.std(external_energies[post_start:]))

    # Secondary Metric: late-window stability (mean reconfiguration cost)
    reconfigs = [r.reconfig_cost for r in records[post_start:]]
    stability = float(np.mean(reconfigs)) if reconfigs else 0.0

    # Final magnitudes (from last record)
    last_record = records[-1]
    adaptation_magnitude_kappa = last_record.kappa_norm
    adaptation_magnitude_theta = last_record.theta_norm

    # Mechanism metrics: use exact state deltas supplied by the runner.  The fallback
    # keeps direct callers compatible while measuring changes in the logged state norms.
    if kappa_change_post is None or theta_change_post is None:
        if perturb_episode < total_episodes:
            later_ep = min(perturb_episode + post_ppee_window, total_episodes - 1)
            kappa_change_post = abs(records[later_ep].kappa_norm - records[perturb_episode].kappa_norm)
            theta_change_post = abs(records[later_ep].theta_norm - records[perturb_episode].theta_norm)
        else:
            kappa_change_post = 0.0
            theta_change_post = 0.0
    kappa_change_post = float(kappa_change_post)
    theta_change_post = float(theta_change_post)

    # Aggregate solver statistics
    total_evals = sum(r.energy_evaluations for r in records)
    total_runtime = sum(r.runtime_sec for r in records)
    mean_conflicts = float(np.mean([r.constraint_violations for r in records]))
    reasons = set(r.termination_reason for r in records)
    termination_summary = "; ".join(sorted(reasons))

    all_optimal = all(bool(r.is_optimal) for r in records)
    any_fallback = any(bool(r.fallback_used) for r in records)

    return TrajectorySummary(
        ppee_10=ppee,
        cumulative_excess_energy=cee,
        recovery_time=recovery_time,
        perf_drop=perf_drop,
        pre_base_energy=pre_base,
        post_base_energy=post_base,
        final_energy=float(external_energies[-1]),
        mean_external_energy=float(np.mean(external_energies)),
        mean_internal_energy=float(np.mean(internal_energies)),
        convergence=convergence,
        stability=stability,
        adaptation_magnitude_kappa=adaptation_magnitude_kappa,
        adaptation_magnitude_theta=adaptation_magnitude_theta,
        kappa_change_post=kappa_change_post,
        theta_change_post=theta_change_post,
        mean_co_assignment_conflicts=mean_conflicts,
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
