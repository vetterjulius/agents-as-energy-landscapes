from __future__ import annotations

import torch

from benchmark.scenarios.base import ProblemInstance
from model.ebmao_orchestrator import EBMAOOrchestrator as CoreEBMAOOrchestrator
from state.orchestration_state import OrchestrationState


class EBMAOOrchestrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.core = None

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        state = self._build_state(problem)

        self.core = CoreEBMAOOrchestrator(
            self.cfg,
            initial_state=state,
            W_risk=problem.risk_weights,
        )

        num_steps = self.cfg.get("solver", {}).get("iterations", 100)

        for _ in range(num_steps):
            self.core.step()

        return self.core.X

    def total_energy(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )

        return self.core.total_energy()

    @property
    def state(self):
        if self.core is None:
            return None
        return self.core.state

    @property
    def X(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )
        return self.core.X

    @property
    def s(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )
        return self.core.s

    @property
    def c(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )
        return self.core.c

    @property
    def kappa(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )
        return self.core.kappa

    @property
    def Theta(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )
        return self.core.Theta

    @property
    def C(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )
        return self.core.C

    @property
    def W_risk(self):
        if self.core is None:
            return None
        return self.core.W_risk

    @property
    def T(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )
        return self.core.T

    @property
    def acc_rate(self):
        if self.core is None:
            raise RuntimeError(
                "EBMAOOrchestrator.solve() must be called first."
            )
        return self.core.acc_rate

    def _build_state(
        self,
        problem: ProblemInstance,
    ) -> OrchestrationState:

        task_embeddings = torch.stack(
            [task.embedding for task in problem.tasks],
            dim=0,
        )

        agent_embeddings = torch.stack(
            [agent.capability_embedding for agent in problem.agents],
            dim=0,
        )

        N = len(problem.agents)
        M = len(problem.tasks)
        d = task_embeddings.shape[1]

        X = torch.zeros(N, M)

        for task_idx, task in enumerate(problem.tasks):
            distances = torch.sum(
                (
                    agent_embeddings
                    - task.embedding.unsqueeze(0)
                ) ** 2,
                dim=1,
            )

            agent_idx = int(torch.argmin(distances).item())
            X[agent_idx, task_idx] = 1.0

        kappa = torch.zeros(N, d)

        return OrchestrationState(
            X=X,
            s=agent_embeddings,
            c=task_embeddings,
            kappa=kappa,
            Theta=problem.interaction_graph.clone(),
            C=problem.co_assignment_costs.clone(),
            N=N,
            M=M,
            d=d,
        )