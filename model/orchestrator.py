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
    def __init__(self, cfg, initial_state=None, W_risk=None):
        self.cfg = cfg
        m = cfg["model"]

        # dimensions
        if initial_state is not None:
            self.N = initial_state.N
            self.M = initial_state.M
            self.d = initial_state.d
        else:
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

        # modes
        self.theta_mode = m.get("theta_mode", "static")  # "static", "dynamic", "hybrid"
        self.search_mode = m.get("search_mode", "hybrid")  # "hybrid", "pure_sa", "pure_greedy"

        # guided proposal config
        self.proposal_candidates = m.get("proposal_candidates", 12)
        self.proposal_task_sample = m.get("proposal_task_sample", 8)
        self.agent_sample_size = m.get("agent_sample_size", 6)
        self.block_move_size = m.get("block_move_size", 4)

        if self.search_mode == "pure_sa":
            self.warm_start_steps = 0
            self.warm_start_type = "random"
            self.hybrid_cleanup_prob = 0.0
            self.local_refine_steps = 0
        elif self.search_mode == "pure_greedy":
            self.warm_start_steps = m.get("warm_start_steps", 6)
            self.warm_start_type = "greedy"
            self.hybrid_cleanup_prob = 0.0
            self.local_refine_steps = m.get("local_refine_steps", 2)
        else:
            self.warm_start_steps = m.get("warm_start_steps", 6)
            self.warm_start_type = m.get("warm_start_type", "greedy")
            self.hybrid_cleanup_prob = m.get("hybrid_cleanup_prob", 0.25)
            self.local_refine_steps = m.get("local_refine_steps", 2)

        # temperature hyperparameters
        self.T_init = m["temperature_init"]
        self.T_min = m["min_temperature"]
        self.T_max = m["max_temperature"]
        self.target_accept = m["target_accept_rate"]

        # Initialize tensors in the exact sequence to preserve random seeds
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
        W_risk_gen = torch.randn(3 * self.d, 1) * risk_scale

        # State object
        if initial_state is not None:
            self.state = initial_state.clone()
        else:
            self.state = OrchestrationState(
                X=X, s=s, c=c, kappa=kappa, Theta=Theta, C=C, N=self.N, M=self.M, d=self.d
            )

        # Clear Theta if pure learning mode is active and we are starting fresh
        if self.theta_mode == "dynamic" and initial_state is None:
            self.state.Theta = torch.zeros_like(self.state.Theta)

        # Risk predictor & energy registry
        if W_risk is not None:
            self.risk_predictor = RiskPredictor(self.d, W_risk=W_risk.clone())
        else:
            self.risk_predictor = RiskPredictor(self.d, W_risk=W_risk_gen)

        self.energy_registry = EnergyRegistry()
        self.energy_registry.add(AssignmentEnergy(self.lambda_align, weight=1.0))
        self.energy_registry.add(InteractionEnergy(weight=self.w_int))
        self.energy_registry.add(CostEnergy(weight=self.w_cost))
        self.energy_registry.add(RiskEnergy(self.risk_predictor, weight=self.w_risk))

        # Dynamics components
        self.proposal_mechanism = AssignmentProposal(
            self.energy_registry,
            lambda_align=self.lambda_align,
            num_tasks=self.proposal_task_sample,
            block_size=self.block_move_size,
            agent_sample_size=self.agent_sample_size
        )
        self.sampler = SimulatedAnnealingSampler(
            self.proposal_mechanism,
            self.energy_registry,
            T_init=self.T_init,
            target_accept=self.target_accept,
            num_candidates=self.proposal_candidates
        )
        self.theta_updater = ThetaUpdater(self.eta_theta)
        self.memory_updater = MemoryUpdater(self.eta_memory)
        self.temperature_controller = TemperatureController(
            self.target_accept,
            self.T_min,
            self.T_max
        )

        self.converged = False

        if self.warm_start_steps > 0:
            self._warm_start()

    def step(self):
        if self.converged:
            return

        if self.search_mode == "pure_greedy":
            # Pure Greedy search mode
            best_X, _, improved = self._find_best_reassignment()
            if improved:
                self.state.X = best_X
            else:
                self.converged = True
        else:
            # SA or Hybrid modes
            accepted = self.sampler.step(self.state)

            if self.search_mode == "hybrid":
                self._local_refine(self.local_refine_steps)

                if not accepted and random.random() < self.hybrid_cleanup_prob:
                    best_X, _, improved = self._find_best_reassignment()
                    if improved:
                        self.state.X = best_X

                # Additional greedy cleanup on every step for stronger exploitation.
                if self.local_refine_steps > 0 and random.random() < 0.35:
                    best_X, _, improved = self._find_best_reassignment()
                    if improved:
                        self.state.X = best_X

        if self.theta_mode != "static":
            self.theta_updater.apply(self.state)
        self.memory_updater.apply(self.state, self.risk_predictor)
        if self.search_mode != "pure_greedy":
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

    def _find_best_reassignment(self):
        X_orig = self.state.X.clone()
        best_X = X_orig.clone()
        best_E, _ = self.energy_registry.compute(self.state)
        best_E = best_E.item()
        improved = False

        for t in range(self.M):
            a_curr = torch.argmax(X_orig[:, t]).item()
            for a in range(self.N):
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

    def _warm_start(self):
        if self.warm_start_type == "greedy":
            self._greedy_construction_init()

        for i in range(self.warm_start_steps):
            best_X, _, improved = self._find_best_reassignment()
            if not improved:
                break
            self.state.X = best_X
            self.theta_updater.apply(self.state)
            self.memory_updater.apply(self.state, self.risk_predictor)

    def _local_refine(self, max_iters=2):
        for _ in range(max_iters):
            best_X, _, improved = self._find_best_reassignment()
            if not improved:
                break
            self.state.X = best_X

    def _greedy_construction_init(self):
        X_init = torch.zeros_like(self.state.X)
        original_X = self.state.X.clone()

        for t in range(self.M):
            best_E = float("inf")
            best_agent = 0
            for a in range(self.N):
                X_init[:, t] = 0.0
                X_init[a, t] = 1.0
                self.state.X = X_init
                E, _ = self.energy_registry.compute(self.state)
                E_val = E.item()
                if E_val < best_E:
                    best_E = E_val
                    best_agent = a
            X_init[:, t] = 0.0
            X_init[best_agent, t] = 1.0

        self.state.X = X_init
        if not torch.equal(self.state.X, original_X):
            self.theta_updater.apply(self.state)
            self.memory_updater.apply(self.state, self.risk_predictor)

    def log(self, step):
        E, _ = self.energy_registry.compute(self.state)
        print(
            f"Step {step} | "
            f"E={E.item():.4f} | "
            f"T={self.sampler.T:.3f} | "
            f"acc={self.sampler.acc_rate:.3f}"
        )
