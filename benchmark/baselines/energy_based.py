import torch
from .base import Orchestrator
from ..scenarios.base import ProblemInstance
from state.orchestration_state import OrchestrationState
from model.orchestrator import Orchestrator as SystemOrchestrator

class EnergyBasedOrchestrator(Orchestrator):
    def __init__(self, cfg, search_mode="hybrid", theta_mode="static"):
        self.cfg = cfg
        self.search_mode = search_mode
        self.theta_mode = theta_mode

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        # Map ProblemInstance to OrchestrationState
        N = len(problem.agents)
        M = len(problem.tasks)
        d = problem.agents[0].capability_embedding.shape[0]

        s = torch.stack([a.capability_embedding for a in problem.agents])
        c = torch.stack([t.embedding for t in problem.tasks])
        kappa = torch.zeros(N, d)

        # Initial random assignment for pure SA, or default modulo assignment
        X_init = torch.zeros(N, M)
        for t in range(M):
            X_init[t % N, t] = 1.0

        state = OrchestrationState(
            X=X_init, s=s, c=c, kappa=kappa,
            Theta=problem.interaction_graph,
            C=problem.co_assignment_costs,
            N=N, M=M, d=d
        )

        # We adapt our config to match what model.Orchestrator expects
        model_cfg = {
            "model": {
                "num_agents": N,
                "num_tasks": M,
                "dim": d,
                "lambda_align": self.cfg["model"].get("lambda_align", 0.5),
                "eta_theta": 0.1,
                "eta_memory": 0.05,
                "risk_weight": self.cfg["model"].get("risk_weight", 1.0),
                "interaction_weight": self.cfg["model"].get("interaction_weight", 1.0),
                "cost_weight": self.cfg["model"].get("cost_weight", 1.0),
                "temperature_init": self.cfg["solver"].get("temperature_init", 1.0),
                "min_temperature": self.cfg["solver"].get("min_temperature", 0.01),
                "max_temperature": self.cfg["solver"].get("max_temperature", 2.0),
                "target_accept_rate": self.cfg["solver"].get("target_accept_rate", 0.25),
                "proposal_candidates": 12,
                "proposal_task_sample": 8,
                "agent_sample_size": 6,
                "block_move_size": 4,
                "warm_start_steps": 6 if self.search_mode != "pure_sa" else 0,
                "warm_start_type": "greedy" if self.search_mode != "pure_sa" else "random",
                "hybrid_cleanup_prob": 0.25 if self.search_mode == "hybrid" else 0.0,
                "local_refine_steps": 2 if self.search_mode != "pure_sa" else 0,
                "theta_mode": self.theta_mode,
                "search_mode": self.search_mode
            }
        }

        orchestrator = SystemOrchestrator(model_cfg, initial_state=state, W_risk=problem.risk_weights)

        for _ in range(self.cfg["solver"].get("iterations", 100)):
            orchestrator.step()

        return orchestrator.state.X

class EnergyPureSAOrchestrator(EnergyBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(cfg, search_mode="pure_sa", theta_mode=theta_mode)

class EnergyHybridOrchestrator(EnergyBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(cfg, search_mode="hybrid", theta_mode=theta_mode)

class EnergyPureGreedyOrchestrator(EnergyBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(cfg, search_mode="pure_greedy", theta_mode=theta_mode)
