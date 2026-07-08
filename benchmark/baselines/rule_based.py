import torch
from .base import Orchestrator
from ..scenarios.base import ProblemInstance

class RuleBasedOrchestrator(Orchestrator):
    """
    Simulates a rule-based system.
    Matches tags if present, otherwise falls back to a simple rule.
    """
    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)
        X = torch.zeros(N, M)

        for t_idx, task in enumerate(problem.tasks):
            assigned = False
            # Rule 1: Tag matching
            if task.tags:
                for a_idx, agent in enumerate(problem.agents):
                    if any(tag in agent.tags for tag in task.tags):
                        X[a_idx, t_idx] = 1.0
                        assigned = True
                        break

            # Rule 2: Simple modulo if no tag match
            if not assigned:
                X[t_idx % N, t_idx] = 1.0

        return X
