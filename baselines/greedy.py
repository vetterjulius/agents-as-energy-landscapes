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

class GreedyOrchestrator(BaseOrchestrator):
    def __init__(self, cfg, initial_state=None, W_risk=None, mode="local_improvement"):
        """
        Greedy Orchestrator Baseline.
        
        Args:
            cfg: Configuration dictionary.
            initial_state: Optional initial OrchestrationState.
            W_risk: Optional risk weights.
            mode: Greedy strategy. Supported values:
                - "local_improvement": At each step, evaluates all single-task reassignments (M * (N - 1))
                                       and makes the one that minimizes the total energy.
                - "local_search": At each step, performs local reassignments until no single reassignment
                                  further improves the energy (local minimum).
                - "construction": Starts by greedily assigning each task one-by-one to the best agent,
                                  and in subsequent steps, does local improvement.
        """
        self.cfg = cfg
        m = cfg["model"]
        self.N = m["num_agents"]
        self.M = m["num_tasks"]
        self.d = m["dim"]
        self.mode = mode

        self.lambda_align = m["lambda_align"]
        self.eta_theta = m["eta_theta"]
        self.eta_memory = m["eta_memory"]
        self.w_risk = m["risk_weight"]
        self.w_int = m["interaction_weight"]
        self.w_cost = m["cost_weight"]

        # Handle fallback initialization
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
        if not tasks or not agents:
            return assignment
        for task in tasks:
            best_agent = None
            best_score = float("inf")
            for agent in agents:
                if hasattr(agent, "capability_embedding"):
                    agent_embedding = agent.capability_embedding
                    agent_id = agent.id
                else:
                    agent_embedding = torch.as_tensor(agent["capability_embedding"], dtype=torch.float32)
                    agent_id = agent["id"]
                score = float(torch.sum((task.embedding - agent_embedding) ** 2).item())
                if score < best_score:
                    best_score = score
                    best_agent = agent_id
            assignment[task.id] = best_agent if best_agent is not None else (agents[0].id if hasattr(agents[0], "id") else agents[0]["id"])
        return assignment

    def _find_best_reassignment(self):
        """
        Evaluates all possible single-task reassignments (M * (N - 1))
        and returns the best assignment X, its energy, and whether it improved.
        """
        X_orig = self.state.X.clone()
        best_X = X_orig.clone()
        best_E, _ = self.energy_registry.compute(self.state)
        best_E = best_E.item()
        improved = False

        # Iterate over all tasks
        for t in range(self.M):
            a_curr = torch.argmax(X_orig[:, t]).item()
            # Try other agents
            for a in range(self.N):
                if a == a_curr:
                    continue
                
                # Propose moving task t to agent a
                X_prop = X_orig.clone()
                X_prop[a_curr, t] = 0.0
                X_prop[a, t] = 1.0
                
                # Temporarily update state X
                self.state.X = X_prop
                
                # Compute energy
                E, _ = self.energy_registry.compute(self.state)
                E_val = E.item()
                
                if E_val < best_E - 1e-6:  # Small epsilon to ensure real improvement
                    best_E = E_val
                    best_X = X_prop.clone()
                    improved = True

        # Restore original state
        self.state.X = X_orig
        return best_X, best_E, improved

    def step(self):
        if self.mode == "local_improvement":
            best_X, _, _ = self._find_best_reassignment()
            self.state.X = best_X
        elif self.mode == "local_search":
            # Run local search to convergence in a single step
            while True:
                best_X, _, improved = self._find_best_reassignment()
                if not improved:
                    break
                self.state.X = best_X
        elif self.mode == "construction":
            if not hasattr(self, "_constructed"):
                self._constructed = True
                # Reset X to all zeros
                self.state.X = torch.zeros_like(self.state.X)
                # Greedily assign each task sequentially
                for t in range(self.M):
                    best_a = 0
                    best_E = float('inf')
                    for a in range(self.N):
                        self.state.X[a, t] = 1.0
                        E, _ = self.energy_registry.compute(self.state)
                        E_val = E.item()
                        if E_val < best_E:
                            best_E = E_val
                            best_a = a
                        self.state.X[a, t] = 0.0
                    self.state.X[best_a, t] = 1.0
            else:
                # Subsequent steps do local improvement
                best_X, _, _ = self._find_best_reassignment()
                self.state.X = best_X

        # Update dynamics (Theta, Memory)
        self.theta_updater.apply(self.state)
        self.memory_updater.apply(self.state, self.risk_predictor)
