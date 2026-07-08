import torch
import random
from .base import Scenario, ProblemInstance, Task, Agent

class InteractionScenario(Scenario):
    def __init__(self, num_agents=5, num_tasks=10, dim=8):
        self.N = num_agents
        self.M = num_tasks
        self.d = dim

    def generate(self, seed: int) -> ProblemInstance:
        torch.manual_seed(seed)
        random.seed(seed)

        agents = []
        for i in range(self.N):
            agents.append(Agent(
                id=f"agent_{i}",
                role="specialist",
                capability_embedding=torch.randn(self.d)
            ))

        tasks = []
        for j in range(self.M):
            tasks.append(Task(
                id=f"task_{j}",
                embedding=torch.randn(self.d)
            ))

        # Synergy: Task A and B should be together (Positive Theta)
        interaction_graph = torch.zeros(self.M, self.M)
        for _ in range(self.M // 2):
            i, j = random.sample(range(self.M), 2)
            interaction_graph[i, j] = 1.0
            interaction_graph[j, i] = 1.0

        # Conflict: Task C and D should NOT be together (Positive C)
        co_assignment_costs = torch.zeros(self.M, self.M)
        for _ in range(self.M // 4):
            i, j = random.sample(range(self.M), 2)
            if interaction_graph[i, j] == 0:
                co_assignment_costs[i, j] = 1.5
                co_assignment_costs[j, i] = 1.5

        risk_weights = torch.randn(3 * self.d, 1)

        return ProblemInstance(
            agents=agents,
            tasks=tasks,
            interaction_graph=interaction_graph,
            co_assignment_costs=co_assignment_costs,
            risk_weights=risk_weights
        )
