import torch

from .base import Orchestrator
from ..scenarios.base import ProblemInstance
from state.orchestration_state import OrchestrationState
from model.ebmao_orchestrator import (
    EBMAOOrchestrator as EBMAOSystemOrchestrator,
)


class EBMAOBasedOrchestrator(Orchestrator):
    """
    Benchmark adapter for the core EBMAO orchestrator.

    The runner is responsible for selecting the experiment and
    constructing the solver-specific configuration.

    This adapter only consumes:

        cfg["energy"]
        cfg["ebmao"]
        cfg["solver"]
        cfg["iterations"]

    No dependency on cfg["experiment_1"] or cfg["experiment_2"].
    """

    def __init__(self, cfg, search_mode="hybrid", theta_mode="static"):
        self.cfg = cfg
        self.search_mode = search_mode
        self.theta_mode = theta_mode

    def _build_state(self, problem: ProblemInstance):
        N = len(problem.agents)
        M = len(problem.tasks)
        d = problem.agents[0].capability_embedding.shape[0]

        s = torch.stack(
            [agent.capability_embedding for agent in problem.agents]
        )

        c = torch.stack(
            [task.embedding for task in problem.tasks]
        )

        kappa = torch.zeros(
            N,
            d,
            dtype=s.dtype,
            device=s.device,
        )

        # Deterministic initial assignment.
        X_init = torch.zeros(
            N,
            M,
            dtype=s.dtype,
            device=s.device,
        )

        for task_idx in range(M):
            X_init[task_idx % N, task_idx] = 1.0

        return OrchestrationState(
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

    def _build_model_config(self, problem: ProblemInstance):
        N = len(problem.agents)
        M = len(problem.tasks)
        d = problem.agents[0].capability_embedding.shape[0]

        energy_cfg = self.cfg.get("energy", {})
        ebmao_cfg = self.cfg.get("ebmao", {})
        solver_cfg = self.cfg.get("solver", {})

        # ------------------------------------------------------------
        # Energy configuration
        # ------------------------------------------------------------

        lambda_align = energy_cfg.get(
            "lambda_align",
            0.5,
        )

        lambda_memory = energy_cfg.get(
            "lambda_memory",
            ebmao_cfg.get(
                "lambda_memory",
                0.5,
            ),
        )

        interaction_weight = energy_cfg.get(
            "interaction_weight",
            1.0,
        )

        risk_weight = energy_cfg.get(
            "risk_weight",
            1.0,
        )

        risk_scale = energy_cfg.get(
            "risk_scale",
            1.0,
        )

        cost_weight = energy_cfg.get(
            "cost_weight",
            1.0,
        )

        # ------------------------------------------------------------
        # EBMAO dynamics
        # ------------------------------------------------------------

        eta_theta = ebmao_cfg.get(
            "eta_theta",
            0.1,
        )

        eta_memory = ebmao_cfg.get(
            "eta_memory",
            0.05,
        )

        # ------------------------------------------------------------
        # Solver configuration
        #
        # The runner has already selected the experiment and solver.
        # The adapter only reads the resulting solver configuration.
        #
        # Explicit solver values override EBMAO defaults.
        # ------------------------------------------------------------

        temperature_init = solver_cfg.get(
            "temperature_init",
            ebmao_cfg.get(
                "temperature_init",
                4.0,
            ),
        )

        min_temperature = solver_cfg.get(
            "min_temperature",
            ebmao_cfg.get(
                "min_temperature",
                1.0,
            ),
        )

        max_temperature = solver_cfg.get(
            "max_temperature",
            ebmao_cfg.get(
                "max_temperature",
                6.0,
            ),
        )

        target_accept_rate = solver_cfg.get(
            "target_accept_rate",
            ebmao_cfg.get(
                "target_accept_rate",
                0.3,
            ),
        )

        proposal_candidates = solver_cfg.get(
            "proposal_candidates",
            ebmao_cfg.get(
                "proposal_candidates",
                12,
            ),
        )

        proposal_task_sample = solver_cfg.get(
            "proposal_task_sample",
            ebmao_cfg.get(
                "proposal_task_sample",
                8,
            ),
        )

        agent_sample_size = solver_cfg.get(
            "agent_sample_size",
            ebmao_cfg.get(
                "agent_sample_size",
                6,
            ),
        )

        block_move_size = solver_cfg.get(
            "block_move_size",
            ebmao_cfg.get(
                "block_move_size",
                4,
            ),
        )

        # ------------------------------------------------------------
        # Search-mode behavior
        # ------------------------------------------------------------

        if self.search_mode == "pure_sa":
            warm_start_steps = 0
            warm_start_type = "random"
            hybrid_cleanup_prob = 0.0
            local_refine_steps = 0

        elif self.search_mode == "pure_greedy":
            warm_start_steps = solver_cfg.get(
                "warm_start_steps",
                ebmao_cfg.get(
                    "warm_start_steps",
                    6,
                ),
            )

            warm_start_type = solver_cfg.get(
                "warm_start_type",
                ebmao_cfg.get(
                    "warm_start_type",
                    "greedy",
                ),
            )

            hybrid_cleanup_prob = 0.0

            local_refine_steps = solver_cfg.get(
                "local_refine_steps",
                ebmao_cfg.get(
                    "local_refine_steps",
                    2,
                ),
            )

        else:
            warm_start_steps = solver_cfg.get(
                "warm_start_steps",
                ebmao_cfg.get(
                    "warm_start_steps",
                    6,
                ),
            )

            warm_start_type = solver_cfg.get(
                "warm_start_type",
                ebmao_cfg.get(
                    "warm_start_type",
                    "greedy",
                ),
            )

            hybrid_cleanup_prob = solver_cfg.get(
                "hybrid_cleanup_prob",
                ebmao_cfg.get(
                    "hybrid_cleanup_prob",
                    0.25,
                ),
            )

            local_refine_steps = solver_cfg.get(
                "local_refine_steps",
                ebmao_cfg.get(
                    "local_refine_steps",
                    2,
                ),
            )

        # ------------------------------------------------------------
        # Final config expected by core EBMAO orchestrator
        # ------------------------------------------------------------

        return {
            "model": {
                "num_agents": N,
                "num_tasks": M,
                "dim": d,

                # Energy
                "lambda_align": lambda_align,
                "lambda_memory": lambda_memory,
                "interaction_weight": interaction_weight,
                "risk_weight": risk_weight,
                "risk_scale": risk_scale,
                "cost_weight": cost_weight,

                # EBMAO dynamics
                "eta_theta": eta_theta,
                "eta_memory": eta_memory,

                # Temperature
                "temperature_init": temperature_init,
                "min_temperature": min_temperature,
                "max_temperature": max_temperature,
                "target_accept_rate": target_accept_rate,

                # Proposal mechanism
                "proposal_candidates": proposal_candidates,
                "proposal_task_sample": proposal_task_sample,
                "agent_sample_size": agent_sample_size,
                "block_move_size": block_move_size,

                # Warm start / refinement
                "warm_start_steps": warm_start_steps,
                "warm_start_type": warm_start_type,
                "hybrid_cleanup_prob": hybrid_cleanup_prob,
                "local_refine_steps": local_refine_steps,

                # Modes
                "theta_mode": self.theta_mode,
                "search_mode": self.search_mode,
            }
        }

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        state = self._build_state(problem)

        model_cfg = self._build_model_config(problem)

        orchestrator = EBMAOSystemOrchestrator(
            model_cfg,
            initial_state=state,
            W_risk=problem.risk_weights,
        )

        # Iteration count is selected by the benchmark runner.
        iterations = self.cfg.get(
            "iterations",
            100,
        )

        for _ in range(iterations):
            orchestrator.step()

            if getattr(
                orchestrator,
                "converged",
                False,
            ):
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