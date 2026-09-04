import torch
import random
from .base import Scenario, ProblemInstance, Task, Agent

class FrustratedScenario(Scenario):
    """
    Frustrated Energy Landscape with Local Minima & Overload Barriers.
    Tasks have local capability fit with specific agents (creating a strong attraction basin for Greedy),
    but co-locating all local tasks creates severe overload conflict.
    Furthermore, coupled task pairs have inter-task synergies, forcing 1-step
    local search (Greedy, Hill-Climbing, Tabu Search) into local minima while Energy-Based Orchestration
    (EBMAO / SA dynamics) escapes barriers to reach the global energy minimum.
    """
    def __init__(self, num_agents=4, num_tasks=8, dim=8):
        self.N = num_agents
        self.M = num_tasks
        self.d = dim

    def generate(self, seed: int) -> ProblemInstance:
        torch.manual_seed(seed)
        random.seed(seed)

        half_M = max(1, self.M // 2)

        # Create agents with specialized capability embeddings
        agents = []
        for i in range(self.N):
            cap = torch.randn(self.d) * 0.05
            cap[i % self.d] += 2.0
            agents.append(Agent(
                id=f"agent_{i}",
                role="specialist",
                capability_embedding=cap
            ))

        # Create tasks: Tasks 0..half_M-1 target Agent 0; Tasks half_M..M-1 target Agent 1
        tasks = []
        for j in range(self.M):
            target_agent = 0 if j < half_M else min(1, self.N - 1)
            emb = torch.randn(self.d) * 0.05
            emb[target_agent % self.d] += 2.0
            tasks.append(Task(
                id=f"task_{j}",
                embedding=emb
            ))

        co_assignment_costs = torch.zeros(self.M, self.M)
        interaction_graph = torch.zeros(self.M, self.M)

        # Overload conflict: Tasks in Set 1 (0..half_M-1) and Set 2 (half_M..M-1) conflict if co-located
        for j in range(half_M):
            for k in range(j + 1, half_M):
                co_assignment_costs[j, k] = 50.0
                co_assignment_costs[k, j] = 50.0

        for j in range(half_M, self.M):
            for k in range(j + 1, self.M):
                co_assignment_costs[j, k] = 50.0
                co_assignment_costs[k, j] = 50.0

        # Synergies between Task j (from Set 1) and Task j + half_M (from Set 2) when co-located
        for j in range(half_M):
            partner = j + half_M
            if partner < self.M:
                interaction_graph[j, partner] = 120.0
                interaction_graph[partner, j] = 120.0

        risk_weights = torch.zeros(3 * self.d, 1)

        return ProblemInstance(
            agents=agents,
            tasks=tasks,
            interaction_graph=interaction_graph,
            co_assignment_costs=co_assignment_costs,
            risk_weights=risk_weights
        )


