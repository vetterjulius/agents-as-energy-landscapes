"""Orchestrators for minimal_llm_poc.

Implements the three specified orchestrators from the controlled benchmark:
1. Baseline: Conventional myopic greedy capability assignment.
2. Static Energy SA: Energy-aware Simulated Annealing with fixed Theta_0.
3. Dynamic Energy SA: Adaptive Energy-aware Simulated Annealing with online Theta and kappa updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from controlled_benchmark.solvers import (
    BudgetedLandscape,
    ConventionalGreedySolver,
    EnergyAwareSimulatedAnnealingSolver,
)
from landscape import Landscape, LandscapeState, ProblemContext
from . import config
from .scenario import AGENTS, NUM_AGENTS, NUM_TASKS_PER_EPISODE, TaskDef, get_interaction_graph


@dataclass
class OrchestrationResult:
    condition: str
    episode: int
    assignment_matrix: torch.Tensor          # Shape (N, M)
    assignment_map: Dict[int, str]           # slot_index -> agent_id
    internal_energy: float
    ground_truth_energy: float
    evaluations_used: int
    solver_status: str


def make_problem_context(
    tasks: List[TaskDef],
    *,
    interaction_weight: float = config.INTERACTION_WEIGHT,
    cost_weight: float = config.COST_WEIGHT,
    risk_weight: float = config.RISK_WEIGHT,
) -> ProblemContext:
    """Build immutable ProblemContext from agents and current tasks."""
    s = torch.tensor([a.capability for a in AGENTS], dtype=torch.float32)
    c = torch.tensor([t.task_affinity for t in tasks], dtype=torch.float32)
    N = len(AGENTS)
    M = len(tasks)
    d = s.shape[1]

    # Mild co-assignment cost matrix to discourage clustering all tasks on one agent
    C = torch.full((M, M), 0.05, dtype=torch.float32)
    C.fill_diagonal_(0.0)

    # Risk weights
    W_risk = torch.zeros(3 * d, 1, dtype=torch.float32)

    return ProblemContext(
        s=s,
        c=c,
        C=C,
        W_risk=W_risk,
        N=N,
        M=M,
        d=d,
        lambda_align=config.LAMBDA_ALIGN,
        lambda_memory=config.LAMBDA_MEMORY,
        interaction_weight=interaction_weight,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
    )


class BaseOrchestrator:
    def __init__(self, condition_name: str):
        self.condition_name = condition_name

    def assign(
        self,
        episode: int,
        tasks: List[TaskDef],
        seed: int = 42,
    ) -> OrchestrationResult:
        raise NotImplementedError

    def adapt(
        self,
        episode: int,
        X: torch.Tensor,
        context: ProblemContext,
    ) -> None:
        """Hook for post-episode online adaptation."""
        pass


class BaselineOrchestrator(BaseOrchestrator):
    """Myopic capability greedy solver: assigns each task to the closest agent."""

    def __init__(self):
        super().__init__("baseline")
        self.solver = ConventionalGreedySolver()

    def assign(
        self,
        episode: int,
        tasks: List[TaskDef],
        seed: int = 42,
    ) -> OrchestrationResult:
        context = make_problem_context(tasks)
        solver_res = self.solver.solve(context)
        X = solver_res.X

        # Evaluate against ground truth
        gt_theta = torch.tensor(get_interaction_graph(episode), dtype=torch.float32)
        gt_state = LandscapeState(kappa=torch.zeros(context.N, context.d), Theta=gt_theta)
        gt_landscape = Landscape(context, gt_state)
        gt_energy = gt_landscape.evaluate(X)

        assignment_map = {}
        for m in range(context.M):
            agent_idx = int(torch.argmax(X[:, m]).item())
            assignment_map[m] = AGENTS[agent_idx].id

        return OrchestrationResult(
            condition=self.condition_name,
            episode=episode,
            assignment_matrix=X.clone(),
            assignment_map=assignment_map,
            internal_energy=0.0,
            ground_truth_energy=gt_energy,
            evaluations_used=0,
            solver_status=solver_res.status,
        )


class StaticEnergySAOrchestrator(BaseOrchestrator):
    """Energy-aware SA solver using static initial Theta_0 (Phase 1 synergies)."""

    def __init__(self):
        super().__init__("static_energy_sa")
        self.sa_solver = EnergyAwareSimulatedAnnealingSolver(
            temperature_init=config.SA_TEMPERATURE_INIT,
            min_temperature=config.SA_MIN_TEMPERATURE,
            cooling_rate=config.SA_COOLING_RATE,
        )
        # Fixed initial belief: Phase 1 dependencies
        self.theta_0 = torch.tensor(get_interaction_graph(0), dtype=torch.float32)

    def assign(
        self,
        episode: int,
        tasks: List[TaskDef],
        seed: int = 42,
    ) -> OrchestrationResult:
        context = make_problem_context(tasks)
        state = LandscapeState(
            kappa=torch.zeros(context.N, context.d),
            Theta=self.theta_0.clone(),
        )
        landscape = Landscape(context, state)
        budgeted = BudgetedLandscape(landscape, max_evaluations=config.MAX_ENERGY_EVALUATIONS)

        # Initial round-robin assignment
        initial_X = torch.zeros(context.N, context.M)
        for m in range(context.M):
            initial_X[m % context.N, m] = 1.0

        solver_res = self.sa_solver.solve(
            budgeted,
            initial_X,
            seed=seed * 1000 + episode,
        )
        X = solver_res.X

        # Evaluate against actual ground truth for this episode
        gt_theta = torch.tensor(get_interaction_graph(episode), dtype=torch.float32)
        gt_state = LandscapeState(kappa=torch.zeros(context.N, context.d), Theta=gt_theta)
        gt_landscape = Landscape(context, gt_state)
        gt_energy = gt_landscape.evaluate(X)

        assignment_map = {}
        for m in range(context.M):
            agent_idx = int(torch.argmax(X[:, m]).item())
            assignment_map[m] = AGENTS[agent_idx].id

        return OrchestrationResult(
            condition=self.condition_name,
            episode=episode,
            assignment_matrix=X.clone(),
            assignment_map=assignment_map,
            internal_energy=float(solver_res.energy),
            ground_truth_energy=gt_energy,
            evaluations_used=solver_res.energy_evaluations,
            solver_status=solver_res.status,
        )


class DynamicEnergySAOrchestrator(BaseOrchestrator):
    """Adaptive Energy-aware SA solver updating Theta and kappa online."""

    def __init__(self):
        super().__init__("dynamic_energy_sa")
        self.sa_solver = EnergyAwareSimulatedAnnealingSolver(
            temperature_init=config.SA_TEMPERATURE_INIT,
            min_temperature=config.SA_MIN_TEMPERATURE,
            cooling_rate=config.SA_COOLING_RATE,
        )
        self.manager = EpisodeAdaptationManager(
            adaptation_mode="full",
            eta_memory=config.ETA_MEMORY,
            eta_theta=config.ETA_THETA,
        )
        # Starts with Theta_0
        self.current_state = LandscapeState(
            kappa=torch.zeros(NUM_AGENTS, config.DIM),
            Theta=torch.tensor(get_interaction_graph(0), dtype=torch.float32),
        )

    def assign(
        self,
        episode: int,
        tasks: List[TaskDef],
        seed: int = 42,
    ) -> OrchestrationResult:
        context = make_problem_context(tasks)
        landscape = Landscape(context, self.current_state)
        budgeted = BudgetedLandscape(landscape, max_evaluations=config.MAX_ENERGY_EVALUATIONS)

        initial_X = torch.zeros(context.N, context.M)
        for m in range(context.M):
            initial_X[m % context.N, m] = 1.0

        solver_res = self.sa_solver.solve(
            budgeted,
            initial_X,
            seed=seed * 1000 + episode,
        )
        X = solver_res.X

        # Evaluate on ground truth
        gt_theta = torch.tensor(get_interaction_graph(episode), dtype=torch.float32)
        gt_state = LandscapeState(kappa=torch.zeros(context.N, context.d), Theta=gt_theta)
        gt_landscape = Landscape(context, gt_state)
        gt_energy = gt_landscape.evaluate(X)

        assignment_map = {}
        for m in range(context.M):
            agent_idx = int(torch.argmax(X[:, m]).item())
            assignment_map[m] = AGENTS[agent_idx].id

        return OrchestrationResult(
            condition=self.condition_name,
            episode=episode,
            assignment_matrix=X.clone(),
            assignment_map=assignment_map,
            internal_energy=float(solver_res.energy),
            ground_truth_energy=gt_energy,
            evaluations_used=solver_res.energy_evaluations,
            solver_status=solver_res.status,
        )

    def adapt(
        self,
        episode: int,
        X: torch.Tensor,
        context: ProblemContext,
    ) -> None:
        """Online boundary update: updates internal belief for episode t+1."""
        self.current_state = self.manager.step(
            self.current_state,
            X,
            context,
        )
