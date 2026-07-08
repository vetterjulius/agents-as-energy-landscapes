import torch
import random
from .base import Scenario, ProblemInstance, Task, Agent

class IndependentScenario(Scenario):
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
                role="generalist",
                capability_embedding=torch.randn(self.d)
            ))

        tasks = []
        for j in range(self.M):
            tasks.append(Task(
                id=f"task_{j}",
                embedding=torch.randn(self.d),
                estimated_cost=random.uniform(0.5, 1.5)
            ))

        interaction_graph = torch.zeros(self.M, self.M)
        co_assignment_costs = torch.zeros(self.M, self.M)
        risk_weights = torch.randn(3 * self.d, 1)

        return ProblemInstance(
            agents=agents,
            tasks=tasks,
            interaction_graph=interaction_graph,
            co_assignment_costs=co_assignment_costs,
            risk_weights=risk_weights
        )
