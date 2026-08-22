import torch

from .base import Orchestrator
from ..scenarios.base import ProblemInstance
from state.orchestration_state import OrchestrationState
from model.ebmao_orchestrator import EBMAOOrchestrator as EBMAOSystemOrchestrator


class EBMAOBasedOrchestrator(Orchestrator):
    def __init__(self, cfg, search_mode="hybrid", theta_mode="static"):
        self.cfg = cfg
        self.search_mode = search_mode
        self.theta_mode = theta_mode

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        # Map ProblemInstance to OrchestrationState.
        N = len(problem.agents)
        M = len(problem.tasks)
        d = problem.agents[0].capability_embedding.shape[0]

        s = torch.stack([agent.capability_embedding for agent in problem.agents])
        c = torch.stack([task.embedding for task in problem.tasks])
        kappa = torch.zeros(N, d, dtype=s.dtype)

        # Deterministic initial assignment for the benchmark adapter.
        X_init = torch.zeros(N, M, dtype=s.dtype)
        for task_idx in range(M):
            X_init[task_idx % N, task_idx] = 1.0

        state = OrchestrationState(
            X=X_init,
            s=s,
            c=c,
            kappa=kappa,
            Theta=problem.interaction_graph.clone(),
            C=problem.co_assignment_costs.clone(),
            N=N,
            M=M,
            d=d,
        )

        # EBMAO expects its configuration under cfg["model"].
        source_model_cfg = self.cfg.get("model", {})

        model_cfg = {
            "model": {
                "num_agents": N,
                "num_tasks": M,
                "dim": d,

                "lambda_align": source_model_cfg.get("lambda_align", 0.5),
                "lambda_memory": source_model_cfg.get(
                    "lambda_memory",
                    source_model_cfg.get("lambda_align", 0.5),
                ),

                "eta_theta": source_model_cfg.get("eta_theta", 0.1),
                "eta_memory": source_model_cfg.get("eta_memory", 0.05),

                "risk_weight": source_model_cfg.get("risk_weight", 1.0),
                "risk_scale": source_model_cfg.get("risk_scale", 1.0),
                "interaction_weight": source_model_cfg.get("interaction_weight", 1.0),
                "cost_weight": source_model_cfg.get("cost_weight", 1.0),

                "temperature_init": source_model_cfg.get("temperature_init", 4.0),
                "min_temperature": source_model_cfg.get("min_temperature", 1.0),
                "max_temperature": source_model_cfg.get("max_temperature", 6.0),
                "target_accept_rate": source_model_cfg.get(
                    "target_accept_rate",
                    0.3,
                ),

                "proposal_candidates": source_model_cfg.get(
                    "proposal_candidates",
                    12,
                ),
                "proposal_task_sample": source_model_cfg.get(
                    "proposal_task_sample",
                    8,
                ),
                "agent_sample_size": source_model_cfg.get(
                    "agent_sample_size",
                    6,
                ),
                "block_move_size": source_model_cfg.get(
                    "block_move_size",
                    4,
                ),

                "warm_start_steps": source_model_cfg.get(
                    "warm_start_steps",
                    6,
                ),
                "warm_start_type": source_model_cfg.get(
                    "warm_start_type",
                    "greedy",
                ),
                "hybrid_cleanup_prob": source_model_cfg.get(
                    "hybrid_cleanup_prob",
                    0.25,
                ),
                "local_refine_steps": source_model_cfg.get(
                    "local_refine_steps",
                    2,
                ),

                "theta_mode": self.theta_mode,
                "search_mode": self.search_mode,
            }
        }

        # Training iterations belong to the training section in the
        # current benchmark configuration.
        iterations = self.cfg.get("training", {}).get("iterations", 100)

        orchestrator = EBMAOSystemOrchestrator(
            model_cfg,
            initial_state=state,
            W_risk=problem.risk_weights,
        )

        for _ in range(iterations):
            orchestrator.step()
            if orchestrator.converged:
                break

        return orchestrator.state.X


class EBMAOPureSAOrchestrator(EBMAOBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(
            cfg,
            search_mode="pure_sa",
            theta_mode=theta_mode,
        )


class EBMAOHybridOrchestrator(EBMAOBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(
            cfg,
            search_mode="hybrid",
            theta_mode=theta_mode,
        )


class EBMAOPureGreedyOrchestrator(EBMAOBasedOrchestrator):
    def __init__(self, cfg, theta_mode="static"):
        super().__init__(
            cfg,
            search_mode="pure_greedy",
            theta_mode=theta_mode,
        )