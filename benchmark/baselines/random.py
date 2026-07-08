import torch
import random
from .base import Orchestrator
from ..scenarios.base import ProblemInstance

class RandomOrchestrator(Orchestrator):
    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)
        X = torch.zeros(N, M)
        for t in range(M):
            a = random.randint(0, N - 1)
            X[a, t] = 1.0
        return X
