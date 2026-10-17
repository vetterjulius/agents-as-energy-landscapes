"""Layer 1 (energy formulation) + Layer 2 (solver) for the PoC.

Reuses the frozen benchmark components unchanged:
- landscape.Landscape / LandscapeState / ProblemContext  (Layer 1)
- controlled_benchmark.solvers (Energy Greedy, SA, Conventional Greedy)  (Layer 2)
- controlled_benchmark.adaptation.EpisodeAdaptationManager (kappa/Theta EMA)

Builds solver-visible ProblemContexts from design-time task/agent embeddings.
The ordinary synthetic graph is not passed through the context; the optional
real-issue pilot initializes Theta from its declared G0 prior explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from controlled_benchmark.solvers import (
    BudgetedLandscape,
    ConventionalGreedySolver,
    EnergyAwareSimulatedAnnealingSolver,
    SolverResult,
)
from landscape import Landscape, LandscapeState, ProblemContext

from . import config
from .tasks import AgentSlot, EpisodeTasks


@dataclass
class OrchestrationResult:
    """Assignment decision for one episode, plus orchestration diagnostics."""
    condition: str
    X: torch.Tensor                 # (N, M) binary assignment
    solver_energy: float            # internal (landscape) energy of the chosen X
    energy_evaluations: int
    iterations: int
    accepted_moves: int
    runtime_sec: float
    termination_reason: str
    initial_X: torch.Tensor
    adaptation_mode: str
    theta_before: torch.Tensor
    theta_after: torch.Tensor
    theta_observation: torch.Tensor


def _problem_context(
    slots: list[AgentSlot],
    episode_tasks: EpisodeTasks,
) -> ProblemContext:
    """Build the solver-visible context from task and agent features only.

    No graph is passed through this context. The pilot's G0 prior, if any, is
    provided separately as LandscapeState.Theta; evaluator-only G1 is never
    exposed to the solver.
    """
    n, m, d = len(slots), len(episode_tasks.specs), config.D
    s = torch.tensor([slot.capability for slot in slots], dtype=torch.float32)
    c = torch.tensor([t.embedding for t in episode_tasks.specs], dtype=torch.float32)
    # No co-assignment penalties and no risk channel in the PoC task family.
    C = torch.zeros(m, m)
    W_risk = torch.zeros(3 * d, 1)
    return ProblemContext(
        s=s, c=c, C=C, W_risk=W_risk, N=n, M=m, d=d,
        lambda_align=config.LAMBDA_ALIGN, lambda_memory=config.LAMBDA_MEMORY,
        interaction_weight=config.INTERACTION_WEIGHT, cost_weight=config.COST_WEIGHT,
        risk_weight=config.RISK_WEIGHT,
    )


def make_initial_assignment(n: int, m: int) -> torch.Tensor:
    """Deterministic round-robin, identical to the benchmark's initial policy."""
    X = torch.zeros(n, m)
    for task in range(m):
        X[task % n, task] = 1.0
    return X


def make_initial_state(
    episode_tasks: EpisodeTasks,
    initial_theta: torch.Tensor | None = None,
) -> LandscapeState:
    """Start from zero memory and either zero interaction state or a supplied prior."""
    m = len(episode_tasks.specs)
    theta = torch.zeros(m, m) if initial_theta is None else torch.as_tensor(
        initial_theta, dtype=torch.float32,
    ).clone()
    if theta.shape != (m, m):
        raise ValueError(f"initial_theta must have shape ({m}, {m}), got {tuple(theta.shape)}")
    if not torch.isfinite(theta).all():
        raise ValueError("initial_theta must contain only finite values")
    if not torch.allclose(theta, theta.T) or torch.any(torch.diagonal(theta) != 0):
        raise ValueError("initial_theta must be symmetric with a zero diagonal")
    return LandscapeState(
        kappa=torch.zeros(config.N_AGENTS, config.D),
        Theta=theta,
    )


def _observed_theta(X: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Return the normalized, symmetric, zero-diagonal assignment observation."""
    co = X.T @ X
    total = co.sum().item()
    if total < epsilon:
        return torch.zeros_like(co)
    observation = co / (total + epsilon)
    observation = (observation + observation.T) / 2.0
    observation.fill_diagonal_(0.0)
    return observation


def orchestrate(
    condition: str,
    slots: list[AgentSlot],
    episode_tasks: EpisodeTasks,
    state: LandscapeState,
    rng_seed: int,
    adaptation_mode: str = "full",
) -> tuple[OrchestrationResult, LandscapeState]:
    """Run one orchestration episode for a condition.

    conditions:
      baseline          conventional greedy (no landscape; no adaptation)
      static_energy     energy landscape with zero, fixed interaction state
      adaptive_energy   same solver, with episode-boundary kappa/Theta EMA updates
    Returns the decision and the (possibly adapted) next state.

    Energy conditions share the same solver, seed, initialization and budget so
    static-vs-adaptive differences isolate state adaptation. State updates occur
    after solving the episode, and the evaluator-only post-change graph is not
    passed here. A supplied Theta prior is controller input, not evidence of
    assignment synergy.
    """
    ctx = _problem_context(slots, episode_tasks)
    initial_X = make_initial_assignment(config.N_AGENTS, len(episode_tasks.specs))
    effective_adaptation_mode = adaptation_mode if condition == "adaptive_energy" else "static"
    theta_before = state.Theta.clone()
    adaptation_diagnostics: dict = {}

    if condition == "baseline":
        result: SolverResult = ConventionalGreedySolver().solve(ctx)
        next_state = state.clone()          # no landscape state exists for B0
    elif condition in ("static_energy", "adaptive_energy"):
        landscape = Landscape(ctx, state)
        budgeted = BudgetedLandscape(landscape, max_evaluations=config.MAX_ENERGY_EVALUATIONS)
        result = EnergyAwareSimulatedAnnealingSolver(
            temperature_init=config.SA_TEMPERATURE_INIT,
            min_temperature=config.SA_MIN_TEMPERATURE,
            cooling_rate=config.SA_COOLING_RATE,
        ).solve(budgeted, initial_X, seed=rng_seed)
        if condition == "static_energy":
            next_state = state.clone()
        else:
            manager = EpisodeAdaptationManager(
                adaptation_mode=adaptation_mode, eta_memory=config.ETA_MEMORY,
                eta_theta=config.ETA_THETA)
            next_state = manager.step(state, result.X, ctx, diagnostics=adaptation_diagnostics)
    else:
        raise ValueError(f"unknown condition {condition!r}")

    return (
        OrchestrationResult(
            condition=condition,
            X=result.X.clone(),
            solver_energy=float(result.energy),
            energy_evaluations=int(result.energy_evaluations),
            iterations=int(result.iterations),
            accepted_moves=int(result.accepted_moves),
            runtime_sec=float(result.runtime_sec),
            termination_reason=result.termination_reason,
            initial_X=initial_X,
            adaptation_mode=effective_adaptation_mode,
            theta_before=theta_before,
            theta_after=next_state.Theta.clone(),
            theta_observation=adaptation_diagnostics.get(
                "theta_observation", _observed_theta(result.X),
            ).clone(),
        ),
        next_state,
    )


def make_real_issue_initial_state(episode_tasks: EpisodeTasks) -> LandscapeState:
    """Initialize the fixed two-node Gensim/Flair pilot at G0."""
    if len(episode_tasks.specs) != len(config.REAL_ISSUE_TASK_IDS):
        raise ValueError(
            f"real-issue pilot expects {len(config.REAL_ISSUE_TASK_IDS)} tasks, "
            f"got {len(episode_tasks.specs)}"
        )
    return make_initial_state(
        episode_tasks,
        initial_theta=torch.tensor(config.REAL_ISSUE_G0, dtype=torch.float32),
    )


def orchestrate_real_issue(
    condition: str,
    slots: list[AgentSlot],
    episode_tasks: EpisodeTasks,
    state: LandscapeState,
    rng_seed: int,
) -> tuple[OrchestrationResult, LandscapeState]:
    """Run the two-node pilot with theta-only adaptation, without a post-change graph input."""
    if len(episode_tasks.specs) != len(config.REAL_ISSUE_TASK_IDS):
        raise ValueError(
            f"real-issue pilot expects {len(config.REAL_ISSUE_TASK_IDS)} tasks, "
            f"got {len(episode_tasks.specs)}"
        )
    expected_shape = (len(config.REAL_ISSUE_TASK_IDS),) * 2
    if state.Theta.shape != expected_shape:
        raise ValueError(f"real-issue pilot state must have shape {expected_shape}")
    return orchestrate(
        condition,
        slots,
        episode_tasks,
        state,
        rng_seed,
        adaptation_mode=config.REAL_ISSUE_ADAPTATION_MODE,
    )


def orchestration_diagnostic(
    result: OrchestrationResult,
    evaluator_theta: torch.Tensor | None = None,
) -> dict:
    """Serialize controller output, then optionally score graph distance post hoc."""
    evaluator = None if evaluator_theta is None else torch.as_tensor(
        evaluator_theta, dtype=torch.float32,
    )
    if evaluator is not None and evaluator.shape != result.theta_before.shape:
        raise ValueError("evaluator_theta must have the same shape as the recorded graph")
    diagnostic = {
        "condition": result.condition,
        "adaptation_mode": result.adaptation_mode,
        "assignment_matrix": result.X.tolist(),
        "theta_before": result.theta_before.tolist(),
        "theta_after": result.theta_after.tolist(),
        "co_assignment": (result.X.T @ result.X).tolist(),
        "theta_observation": result.theta_observation.tolist(),
        "solver_energy": result.solver_energy,
    }
    if evaluator is not None:
        diagnostic["theta_distance_to_evaluator_before"] = float(
            torch.linalg.vector_norm(result.theta_before - evaluator)
        )
        diagnostic["theta_distance_to_evaluator_after"] = float(
            torch.linalg.vector_norm(result.theta_after - evaluator)
        )
    return diagnostic
