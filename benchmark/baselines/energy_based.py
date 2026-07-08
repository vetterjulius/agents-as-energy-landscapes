import torch
from .base import Orchestrator
from ..scenarios.base import ProblemInstance
from state.orchestration_state import OrchestrationState
from model.orchestrator import Orchestrator as SystemOrchestrator

class EnergyBasedOrchestrator(Orchestrator):
    def __init__(self, cfg):
        self.cfg = cfg

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        # Map ProblemInstance to OrchestrationState
        N = len(problem.agents)
        M = len(problem.tasks)
        d = problem.agents[0].capability_embedding.shape[0]

        s = torch.stack([a.capability_embedding for a in problem.agents])
        c = torch.stack([t.embedding for t in problem.tasks])
        kappa = torch.zeros(N, d)

        # Initial random assignment
        X_init = torch.zeros(N, M)
        for t in range(M):
            X_init[t % N, t] = 1.0

        state = OrchestrationState(
            X=X_init, s=s, c=c, kappa=kappa,
            Theta=problem.interaction_graph,
            C=problem.co_assignment_costs,
            N=N, M=M, d=d
        )

        # Use the system's orchestrator with its internal SA solver
        # We need to adapt our config to match what model.Orchestrator expects
        model_cfg = {
            "model": {
                "num_agents": N,
                "num_tasks": M,
                "dim": d,
                "lambda_align": self.cfg.get("lambda_align", 0.5),
                "eta_theta": 0.1,
                "eta_memory": 0.05,
                "risk_weight": 1.0,
                "interaction_weight": 1.0,
                "cost_weight": 1.0,
                "temperature_init": self.cfg["solver"]["temperature_init"],
                "min_temperature": self.cfg["solver"]["min_temperature"],
                "max_temperature": self.cfg["solver"]["max_temperature"],
                "target_accept_rate": self.cfg["solver"]["target_accept_rate"],
                "proposal_candidates": 12,
                "proposal_task_sample": 8,
                "agent_sample_size": 6,
                "block_move_size": 4,
                "warm_start_steps": 6,
                "warm_start_type": "greedy",
                "hybrid_cleanup_prob": 0.25,
                "local_refine_steps": 2
            }
        }

        orchestrator = SystemOrchestrator(model_cfg, initial_state=state, W_risk=problem.risk_weights)

        for _ in range(self.cfg["solver"]["iterations"]):
            orchestrator.step()

        return orchestrator.state.X
