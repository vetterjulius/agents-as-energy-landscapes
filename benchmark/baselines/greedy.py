import torch
from .base import Orchestrator
from ..scenarios.base import ProblemInstance

class GreedyOrchestrator(Orchestrator):
    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)
        X = torch.zeros(N, M)

        s = torch.stack([a.capability_embedding for a in problem.agents])
        c = torch.stack([t.embedding for t in problem.tasks])

        # Distance matrix (N, M)
        dist = torch.cdist(s, c)

        for t in range(M):
            best_agent = torch.argmin(dist[:, t]).item()
            X[best_agent, t] = 1.0

        return X
