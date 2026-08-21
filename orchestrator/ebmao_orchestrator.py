from __future__ import annotations

from typing import List

import torch

from model.ebmao_orchestrator import EBMAOOrchestrator as CoreEBMAOOrchestrator
from orchestrator.base import Agent, Assignment, BaseOrchestrator, Task
from state.orchestration_state import OrchestrationState


class EBMAOOrchestrator(BaseOrchestrator):
    """Benchmark adapter around the canonical EBMAO implementation."""

    def __init__(self, cfg, initial_state=None, W_risk=None):
        super().__init__(cfg)

        self.cfg = cfg
        self.initial_state = initial_state
        self._W_risk = W_risk

        # The actual EBMAO implementation lives in model/.
        self.core = None

    def solve(self, tasks: List[Task], agents: List[Agent]) -> Assignment:
        assignment = Assignment()

        if not tasks or not agents:
            return assignment

        state = self._build_state(tasks, agents)

        self.core = CoreEBMAOOrchestrator(
            self.cfg,
            initial_state=state,
            W_risk=self._W_risk,
        )

        # Run the canonical EBMAO dynamics.
        num_steps = self.cfg.get("model", {}).get("benchmark_steps", 10)

        for _ in range(num_steps):
            self.core.step()

        for task_idx, task in enumerate(tasks):
            agent_idx = int(torch.argmax(self.core.X[:, task_idx]).item())
            assignment[task.id] = agents[agent_idx].id

        return assignment

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
            return self._W_risk
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
        tasks: List[Task],
        agents: List[Agent],
    ) -> OrchestrationState:
        task_embeddings = torch.stack(
            [task.embedding for task in tasks],
            dim=0,
        )

        agent_embeddings = torch.stack(
            [agent.capability_embedding for agent in agents],
            dim=0,
        )

        M = len(tasks)
        N = len(agents)
        d = task_embeddings.shape[1]

        # Initial assignment based on capability distance.
        X = torch.zeros(N, M)

        for task_idx, task in enumerate(tasks):
            distances = torch.sum(
                (agent_embeddings - task.embedding.unsqueeze(0)) ** 2,
                dim=1,
            )

            agent_idx = int(torch.argmin(distances).item())
            X[agent_idx, task_idx] = 1.0

        # Encode task dependencies in Theta.
        Theta = torch.zeros(M, M)

        for task_idx, task in enumerate(tasks):
            for dependency in task.dependencies:
                dep_idx = next(
                    (
                        idx
                        for idx, candidate in enumerate(tasks)
                        if candidate.id == dependency
                    ),
                    None,
                )

                if dep_idx is not None:
                    Theta[task_idx, dep_idx] = 1.0

        # Cost/coupling matrix is part of the EBMAO state.
        C = torch.rand(M, M)
        C.fill_diagonal_(0)

        kappa = torch.zeros(N, d)

        return OrchestrationState(
            X=X,
            s=agent_embeddings,
            c=task_embeddings,
            kappa=kappa,
            Theta=Theta,
            C=C,
            N=N,
            M=M,
            d=d,
        )