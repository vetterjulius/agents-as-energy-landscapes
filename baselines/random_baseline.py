import torch
import random
from state.orchestration_state import OrchestrationState
from energy.registry import EnergyRegistry
from energy.assignment import AssignmentEnergy
from energy.interaction import InteractionEnergy
from energy.cost import CostEnergy
from energy.risk import RiskEnergy, RiskPredictor
from dynamics.theta_update import ThetaUpdater
from dynamics.memory_update import MemoryUpdater
from orchestrator.base import Agent, Assignment, BaseOrchestrator, Task

class RandomOrchestrator(BaseOrchestrator):
    def __init__(self, cfg, initial_state=None, W_risk=None):
        """
        Random Orchestrator Baseline.
        At each step, assigns each task to a random agent.
        """
        self.cfg = cfg
        m = cfg["model"]
        self.N = m["num_agents"]
        self.M = m["num_tasks"]
        self.d = m["dim"]

        self.lambda_align = m["lambda_align"]
        self.eta_theta = m["eta_theta"]
        self.eta_memory = m["eta_memory"]
        self.w_risk = m["risk_weight"]
        self.w_int = m["interaction_weight"]
        self.w_cost = m["cost_weight"]

        s_gen = torch.randn(self.N, self.d)
        c_gen = torch.randn(self.M, self.d)
        kappa_gen = torch.zeros(self.N, self.d)
        Theta_gen = torch.zeros(self.M, self.M)
        C_gen = torch.rand(self.M, self.M)
        C_gen.fill_diagonal_(0)

        X_gen = torch.zeros(self.N, self.M)
        for t in range(self.M):
            a = random.randint(0, self.N - 1)
            X_gen[a, t] = 1.0

        risk_scale = m.get("risk_scale", 1.0)
        W_risk_gen = torch.randn(3 * self.d, 1) * risk_scale

        if initial_state is not None:
            self.state = initial_state.clone()
        else:
            self.state = OrchestrationState(
                X=X_gen, s=s_gen, c=c_gen, kappa=kappa_gen, Theta=Theta_gen, C=C_gen, N=self.N, M=self.M, d=self.d
            )

        if W_risk is not None:
            self.risk_predictor = RiskPredictor(self.d, W_risk=W_risk.clone())
        else:
            self.risk_predictor = RiskPredictor(self.d, W_risk=W_risk_gen)

        self.energy_registry = EnergyRegistry()
        self.energy_registry.add(AssignmentEnergy(self.lambda_align, weight=1.0))
        self.energy_registry.add(InteractionEnergy(weight=self.w_int))
        self.energy_registry.add(CostEnergy(weight=self.w_cost))
        self.energy_registry.add(RiskEnergy(self.risk_predictor, weight=self.w_risk))

        self.theta_updater = ThetaUpdater(self.eta_theta)
        self.memory_updater = MemoryUpdater(self.eta_memory)

    def total_energy(self):
        total, _ = self.energy_registry.compute(self.state)
        return total

    def solve(self, tasks: list[Task], agents: list[Agent]) -> Assignment:
        assignment = Assignment()
        if not tasks:
            return assignment
        for task in tasks:
            agent_data = agents[random.randrange(len(agents))]
            if hasattr(agent_data, "id"):
                agent_id = agent_data.id
            else:
                agent_id = agent_data["id"]
            assignment[task.id] = agent_id
        return assignment

    def step(self):
        X_new = torch.zeros_like(self.state.X)
        for t in range(self.M):
            a = random.randint(0, self.N - 1)
            X_new[a, t] = 1.0
        self.state.X = X_new

        # Update dynamics
        self.theta_updater.apply(self.state)
        self.memory_updater.apply(self.state, self.risk_predictor)
