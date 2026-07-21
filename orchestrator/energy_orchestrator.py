from __future__ import annotations

import random
from typing import List

import torch

from dynamics.memory_update import MemoryUpdater
from dynamics.theta_update import ThetaUpdater
from energy.assignment import AssignmentEnergy
from energy.cost import CostEnergy
from energy.interaction import InteractionEnergy
from energy.registry import EnergyRegistry
from energy.risk import RiskEnergy, RiskPredictor
from orchestrator.base import Agent, Assignment, BaseOrchestrator, Task
from state.orchestration_state import OrchestrationState


class EnergyOrchestrator(BaseOrchestrator):
    """Energy-based orchestrator that builds a real orchestration state from benchmark tasks."""

    def __init__(self, cfg, initial_state=None, W_risk=None):
        super().__init__(cfg)
        self.cfg = cfg
        m = cfg.get("model", {})

        self.N = m.get("num_agents", 2)
        self.M = m.get("num_tasks", 2)
        self.d = m.get("dim", 8)

        self.lambda_align = m.get("lambda_align", 0.5)
        self.eta_theta = m.get("eta_theta", 0.1)
        self.eta_memory = m.get("eta_memory", 0.05)
        self.w_risk = m.get("risk_weight", 1.0)
        self.w_int = m.get("interaction_weight", 1.0)
        self.w_cost = m.get("cost_weight", 1.0)

        self.theta_mode = m.get("theta_mode", "static")  # "static", "dynamic", "hybrid"

        self.state = initial_state.clone() if initial_state is not None else self._build_state_from_config()
        if self.theta_mode == "dynamic" and initial_state is None:
            self.state.Theta = torch.zeros_like(self.state.Theta)
        self.risk_predictor = RiskPredictor(self.d, W_risk=W_risk.clone() if W_risk is not None else None, scale=m.get("risk_scale", 1.0))

        self.energy_registry = EnergyRegistry()
        self.energy_registry.add(AssignmentEnergy(self.lambda_align, weight=1.0))
        self.energy_registry.add(InteractionEnergy(weight=self.w_int))
        self.energy_registry.add(CostEnergy(weight=self.w_cost))
        self.energy_registry.add(RiskEnergy(self.risk_predictor, weight=self.w_risk))

        self.theta_updater = ThetaUpdater(self.eta_theta)
        self.memory_updater = MemoryUpdater(self.eta_memory)

    def solve(self, tasks: List[Task], agents: List[Agent]) -> Assignment:
        assignment = Assignment()
        if not tasks or not agents:
            return assignment

        self._initialize_from_tasks(tasks, agents)
        self._apply_energy_optimization(tasks, agents)

        for task_idx, task in enumerate(tasks):
            agent_idx = int(torch.argmax(self.state.X[:, task_idx]).item())
            agent = self._coerce_agent(agents[agent_idx])
            assignment[task.id] = agent.id
        return assignment

    def total_energy(self):
        total, _ = self.energy_registry.compute(self.state)
        return total

    def _initialize_from_tasks(self, tasks: List[Task], agents: List[Agent]) -> None:
        task_embeddings = torch.stack([task.embedding for task in tasks], dim=0)
        agent_embeddings = torch.stack([self._coerce_agent(agent).capability_embedding for agent in agents], dim=0)
        self.state.c = task_embeddings
        self.state.s = agent_embeddings
        self.state.M = len(tasks)
        self.state.N = len(agents)
        self.state.d = self.state.c.shape[1]
        self.state.kappa = torch.zeros(self.state.N, self.state.d)
        self.state.Theta = torch.zeros(self.state.M, self.state.M)
        self.state.C = torch.rand(self.state.M, self.state.M)
        self.state.C.fill_diagonal_(0)

        self.state.X = torch.zeros(self.state.N, self.state.M)
        for task_idx, task in enumerate(tasks):
            if self.theta_mode != "dynamic":
                if task.dependencies:
                    for dep in task.dependencies:
                        dep_idx = next((idx for idx, candidate in enumerate(tasks) if candidate.id == dep), None)
                        if dep_idx is not None:
                            self.state.Theta[task_idx, dep_idx] = 1.0
            best_agent_idx = 0
            best_score = float("inf")
            for agent_idx, agent in enumerate(agents):
                agent_embedding = self._coerce_agent(agent).capability_embedding
                score = float(torch.sum((task.embedding - agent_embedding) ** 2).item())
                if score < best_score:
                    best_score = score
                    best_agent_idx = agent_idx
            self.state.X[best_agent_idx, task_idx] = 1.0

    def _apply_energy_optimization(self, tasks: List[Task], agents: List[Agent]) -> None:
        if self.theta_mode != "static":
            self.theta_updater.apply(self.state)
        self.memory_updater.apply(self.state, self.risk_predictor)
        for _ in range(2):
            best_X, _, improved = self._find_best_reassignment()
            if improved:
                self.state.X = best_X
            if self.theta_mode != "static":
                self.theta_updater.apply(self.state)
            self.memory_updater.apply(self.state, self.risk_predictor)

    def _find_best_reassignment(self):
        X_orig = self.state.X.clone()
        best_X = X_orig.clone()
        best_E, _ = self.energy_registry.compute(self.state)
        best_E = best_E.item()
        improved = False

        for t in range(self.state.M):
            a_curr = torch.argmax(X_orig[:, t]).item()
            for a in range(self.state.N):
                if a == a_curr:
                    continue
                X_prop = X_orig.clone()
                X_prop[a_curr, t] = 0.0
                X_prop[a, t] = 1.0
                self.state.X = X_prop
                E, _ = self.energy_registry.compute(self.state)
                E_val = E.item()
                if E_val < best_E - 1e-6:
                    best_E = E_val
                    best_X = X_prop.clone()
                    improved = True

        self.state.X = X_orig
        return best_X, best_E, improved

    def _build_state_from_config(self) -> OrchestrationState:
        n = self.N
        m = self.M
        d = self.d
        s = torch.randn(n, d)
        c = torch.randn(m, d)
        kappa = torch.zeros(n, d)
        Theta = torch.zeros(m, m)
        C = torch.rand(m, m)
        C.fill_diagonal_(0)
        X = torch.zeros(n, m)
        for t in range(m):
            a = random.randint(0, n - 1)
            X[a, t] = 1.0
        return OrchestrationState(X=X, s=s, c=c, kappa=kappa, Theta=Theta, C=C, N=n, M=m, d=d)

    @staticmethod
    def _coerce_agent(agent: Agent | dict):
        if isinstance(agent, dict):
            return type("AgentShim", (), {
                "id": agent.get("id", "unknown"),
                "role": agent.get("role", "unknown"),
                "capability_embedding": torch.as_tensor(agent.get("capability_embedding", []), dtype=torch.float32),
                "memory_state": agent.get("memory_state", {}),
            })()
        return agent
