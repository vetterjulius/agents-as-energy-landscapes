import random
import torch

from state.orchestration_state import OrchestrationState
from energy.registry import EnergyRegistry
from energy.assignment import AssignmentEnergy
from energy.interaction import InteractionEnergy
from energy.cost import CostEnergy
from energy.risk import RiskEnergy, RiskPredictor
from dynamics.proposal import AssignmentProposal
from dynamics.sampler import SimulatedAnnealingSampler
from dynamics.theta_update import ThetaUpdater
from dynamics.memory_update import MemoryUpdater
from dynamics.temperature import TemperatureController

class Orchestrator:
    def __init__(self, cfg):
        self.cfg = cfg
        m = cfg["model"]

        # dimensions
        self.N = m["num_agents"]
        self.M = m["num_tasks"]
        self.d = m["dim"]

        # hyperparameters
        self.lambda_align = m["lambda_align"]
        self.eta_theta = m["eta_theta"]
        self.eta_memory = m["eta_memory"]

        self.w_risk = m["risk_weight"]
        self.w_int = m["interaction_weight"]
        self.w_cost = m["cost_weight"]

        # temperature hyperparameters
        self.T_init = m["temperature_init"]
        self.T_min = m["min_temperature"]
        self.T_max = m["max_temperature"]
        self.target_accept = m["target_accept_rate"]

        # Initialize tensors in the exact sequence as the original implementation to preserve random seeds
        s = torch.randn(self.N, self.d)
        c = torch.randn(self.M, self.d)
        kappa = torch.zeros(self.N, self.d)

        Theta = torch.zeros(self.M, self.M)
        C = torch.rand(self.M, self.M)
        C.fill_diagonal_(0)

        X = torch.zeros(self.N, self.M)
        for t in range(self.M):
            a = random.randint(0, self.N - 1)
            X[a, t] = 1.0

        risk_scale = m.get("risk_scale", 1.0)
        W_risk = torch.randn(3 * self.d, 1) * risk_scale

        # State object
        self.state = OrchestrationState(
            X=X, s=s, c=c, kappa=kappa, Theta=Theta, C=C, N=self.N, M=self.M, d=self.d
        )

        # Risk predictor & energy registry
        self.risk_predictor = RiskPredictor(self.d, W_risk=W_risk)

        self.energy_registry = EnergyRegistry()
        self.energy_registry.add(AssignmentEnergy(self.lambda_align, weight=1.0))
        self.energy_registry.add(InteractionEnergy(weight=self.w_int))
        self.energy_registry.add(CostEnergy(weight=self.w_cost))
        self.energy_registry.add(RiskEnergy(self.risk_predictor, weight=self.w_risk))

        # Dynamics components
        self.proposal_mechanism = AssignmentProposal()
        self.sampler = SimulatedAnnealingSampler(
            self.proposal_mechanism,
            self.energy_registry,
            T_init=self.T_init,
            target_accept=self.target_accept
        )
        self.theta_updater = ThetaUpdater(self.eta_theta)
        self.memory_updater = MemoryUpdater(self.eta_memory)
        self.temperature_controller = TemperatureController(
            self.target_accept,
            self.T_min,
            self.T_max
        )

    def step(self):
        self.sampler.step(self.state)
        self.theta_updater.apply(self.state)
        self.memory_updater.apply(self.state, self.risk_predictor)
        self.temperature_controller.apply(self.sampler)

    def total_energy(self):
        total, _ = self.energy_registry.compute(self.state)
        return total

    @property
    def T(self):
        return self.sampler.T

    @property
    def acc_rate(self):
        return self.sampler.acc_rate

    @property
    def X(self):
        return self.state.X

    @property
    def s(self):
        return self.state.s

    @property
    def c(self):
        return self.state.c

    @property
    def kappa(self):
        return self.state.kappa

    @property
    def Theta(self):
        return self.state.Theta

    @property
    def C(self):
        return self.state.C

    @property
    def W_risk(self):
        return self.risk_predictor.W_risk

    def log(self, step):
        E, _ = self.energy_registry.compute(self.state)
        print(
            f"Step {step} | "
            f"E={E.item():.4f} | "
            f"T={self.sampler.T:.3f} | "
            f"acc={self.sampler.acc_rate:.3f}"
        )
