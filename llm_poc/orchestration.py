"""Layer 1 (energy formulation) + Layer 2 (solver) for the PoC.

Reuses the frozen benchmark components unchanged:
- landscape.Landscape / LandscapeState / ProblemContext  (Layer 1)
- controlled_benchmark.solvers (Energy Greedy, SA, Conventional Greedy)  (Layer 2)
- controlled_benchmark.adaptation.EpisodeAdaptationManager (kappa/Theta EMA)

Builds ProblemContexts from the design-time task/agent embeddings and declared
dependency structure, exactly mirroring how the benchmark constructs problems.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from controlled_benchmark.solvers import (
    BudgetedLandscape,
    ConventionalGreedySolver,
    EnergyAwareGreedySolver,
    EnergyAwareSimulatedAnnealingSolver,
    SolverResult,
)
from landscape import Landscape, LandscapeState, ProblemContext

from . import config
from .tasks import AgentSlot, EpisodeTasks, declared_dependency_pairs


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


def _problem_context(
    slots: list[AgentSlot],
    episode_tasks: EpisodeTasks,
    dependency_pairs: list[tuple[int, int]],
) -> ProblemContext:
    """Build the ProblemContext: s = agent capabilities, c = task embeddings,
    C = co-assignment penalties (0 for PoC), interaction graph = declared deps."""
    n, m, d = len(slots), len(episode_tasks.specs), config.D
    s = torch.tensor([slot.capability for slot in slots], dtype=torch.float32)
    c = torch.tensor([t.embedding for t in episode_tasks.specs], dtype=torch.float32)
    interaction = torch.zeros(m, m)
    for up, down in dependency_pairs:
        interaction[up, down] = 1.0
        interaction[down, up] = 1.0
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


def make_initial_state(episode_tasks: EpisodeTasks,
                       dependency_pairs: list[tuple[int, int]]) -> LandscapeState:
    """kappa_0 = 0, Theta_0 = declared dependency graph (pre-change)."""
    m = len(episode_tasks.specs)
    theta0 = torch.zeros(m, m)
    for up, down in dependency_pairs:
        theta0[up, down] = 1.0
        theta0[down, up] = 1.0
    return LandscapeState(kappa=torch.zeros(config.N_AGENTS, config.D), Theta=theta0)


def orchestrate(
    condition: str,
    slots: list[AgentSlot],
    episode_tasks: EpisodeTasks,
    dependency_pairs: list[tuple[int, int]],
    state: LandscapeState,
    rng_seed: int,
) -> tuple[OrchestrationResult, LandscapeState]:
    """Run one orchestration episode for a condition.

    conditions:
      baseline          conventional greedy (no landscape; no adaptation)
      static_energy     energy landscape with fixed Theta_0 (no adaptation)
      adaptive_energy   energy landscape with kappa_t/Theta_t EMA adaptation
    Returns the decision and the (possibly adapted) next state.

    Semantics mirror the frozen benchmark exactly: current episode inputs only,
    episode-boundary adaptation, no warm starts, common evaluation budget.
    """
    ctx = _problem_context(slots, episode_tasks, dependency_pairs)
    initial_X = make_initial_assignment(config.N_AGENTS, len(episode_tasks.specs))

    if condition == "baseline":
        result: SolverResult = ConventionalGreedySolver().solve(ctx)
        next_state = state.clone()          # no landscape state exists for B0
    elif condition in ("static_energy", "adaptive_energy"):
        landscape = Landscape(ctx, state)
        budgeted = BudgetedLandscape(landscape, max_evaluations=config.MAX_ENERGY_EVALUATIONS)
        if condition == "static_energy":
            result = EnergyAwareGreedySolver().solve(budgeted, initial_X)
        else:
            result = EnergyAwareSimulatedAnnealingSolver(
                temperature_init=config.SA_TEMPERATURE_INIT,
                min_temperature=config.SA_MIN_TEMPERATURE,
                cooling_rate=config.SA_COOLING_RATE,
            ).solve(budgeted, initial_X, seed=rng_seed)
        if condition == "static_energy":
            next_state = state.clone()
        else:
            manager = EpisodeAdaptationManager(
                adaptation_mode="full", eta_memory=config.ETA_MEMORY,
                eta_theta=config.ETA_THETA)
            next_state = manager.step(state, result.X, ctx)
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
        ),
        next_state,
    )
