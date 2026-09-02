import torch

from .base import Orchestrator
from ..benchmark.scenarios.base import ProblemInstance
from state.orchestration_state import OrchestrationState
from model.orchestrator import Orchestrator as SystemOrchestrator


class EnergyBasedOrchestrator(Orchestrator):
    def __init__(self, cfg, search_mode="hybrid", theta_mode="static"):
        self.cfg = cfg
        self.search_mode = search_mode
        self.theta_mode = theta_mode

        self.energy_history = []
        self.temp_history = []

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)
        d = problem.agents[0].capability_embedding.shape[0]

        state = OrchestrationState(
            X=self._initial_assignment(N, M),
            s=torch.stack([
                agent.capability_embedding
                for agent in problem.agents
            ]),
            c=torch.stack([
                task.embedding
                for task in problem.tasks
            ]),
            kappa=torch.zeros(N, d),
            Theta=problem.interaction_graph.clone(),
            C=problem.co_assignment_costs.clone(),
            N=N,
            M=M,
            d=d,
        )

        model_cfg = self._build_model_config(N, M, d)

        orchestrator = SystemOrchestrator(
            model_cfg,
            initial_state=state,
            W_risk=problem.risk_weights,
        )

        self.energy_history = [
            orchestrator.total_energy().item()
        ]
        self.temp_history = [orchestrator.T]

        iterations = self.cfg["solver"].get("iterations", 100)

        for _ in range(iterations):
            orchestrator.step()

            self.energy_history.append(
                orchestrator.total_energy().item()
            )
            self.temp_history.append(orchestrator.T)

        return orchestrator.state.X

    def _initial_assignment(self, N, M):
        X = torch.zeros(N, M)

        for t in range(M):
            X[t % N, t] = 1.0

        return X

    def _build_model_config(self, N, M, d):
        model_cfg = dict(self.cfg)

        model_cfg["model"] = dict(
            self.cfg.get("model", {})
        )

        model_cfg["model"].update({
            "num_agents": N,
            "num_tasks": M,
            "dim": d,
            "search_mode": self.search_mode,
            "theta_mode": self.theta_mode,
        })

        return model_cfg


class EnergyPureSAOrchestrator(EnergyBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(
            cfg,
            search_mode="pure_sa",
            theta_mode=theta_mode,
        )


class EnergyHybridOrchestrator(EnergyBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(
            cfg,
            search_mode="hybrid",
            theta_mode=theta_mode,
        )


class EnergyPureGreedyOrchestrator(EnergyBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(
            cfg,
            search_mode="pure_greedy",
            theta_mode=theta_mode,
        )