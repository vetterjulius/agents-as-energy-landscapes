import torch
import random
from .base import Scenario, ProblemInstance, Task, Agent

class FrustratedScenario(Scenario):
    """
    Scenario where local greedy optimization fails.
    Concept: Tasks have high individual alignment with certain agents,
    BUT those agents have high co-assignment costs or negative interactions (conflicts).
    Alternatively, a 'ring' of dependencies where any local move looks bad but a global
    rearrangement is better.
    """
    def __init__(self, num_agents=3, num_tasks=6, dim=8):
        self.N = num_agents
        self.M = num_tasks
        self.d = dim

    def generate(self, seed: int) -> ProblemInstance:
        torch.manual_seed(seed)
        random.seed(seed)

        # Create agents with distinct specializations
        agents = []
        for i in range(self.N):
            capability = torch.zeros(self.d)
            capability[i % self.d] = 1.0
            agents.append(Agent(
                id=f"agent_{i}",
                role="specialist",
                capability_embedding=capability
            ))

        # Create tasks that 'locally' fit agents well
        tasks = []
        for j in range(self.M):
            # Task j fits agent j % N
            target_agent = j % self.N
            embedding = torch.zeros(self.d)
            embedding[target_agent % self.d] = 1.0
            tasks.append(Task(
                id=f"task_{j}",
                embedding=embedding
            ))

        # Add FRUSTRATION:
        # If task j and task k are assigned to the same agent, it's VERY costly
        # even if they both fit that agent well.
        co_assignment_costs = torch.zeros(self.M, self.M)
        for j in range(self.M):
            for k in range(j + 1, self.M):
                if j % self.N == k % self.N:
                    # They want to go to the same agent locally
                    co_assignment_costs[j, k] = 10.0
                    co_assignment_costs[k, j] = 10.0

        # Positive interactions (synergies) are between tasks that do NOT fit the same agent
        interaction_graph = torch.zeros(self.M, self.M)
        for j in range(self.M):
            neighbor = (j + 1) % self.M
            if j % self.N != neighbor % self.N:
                interaction_graph[j, neighbor] = 2.0
                interaction_graph[neighbor, j] = 2.0

        risk_weights = torch.zeros(3 * self.d, 1)

        return ProblemInstance(
            agents=agents,
            tasks=tasks,
            interaction_graph=interaction_graph,
            co_assignment_costs=co_assignment_costs,
            risk_weights=risk_weights
        )
