import torch
from .base import Orchestrator
from ..scenarios.base import ProblemInstance

class GreedyLoadBalancingOrchestrator(Orchestrator):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)
        X = torch.zeros(N, M)

        s = torch.stack([a.capability_embedding for a in problem.agents])
        c = torch.stack([t.embedding for t in problem.tasks])

        workload = torch.zeros(N)

        for t in range(M):
            # score = distance + alpha * current_workload
            dist = torch.norm(s - c[t], dim=1)
            score = dist + self.alpha * workload

            best_agent = torch.argmin(score).item()
            X[best_agent, t] = 1.0
            workload[best_agent] += 1.0

        return X
