import torch
import random
import copy
from .base import Scenario, ProblemInstance, Task, Agent

class DynamicScenario(Scenario):
    """
    Simulates changes in the environment.
    Actually, we generate a ProblemInstance that represents the 'next' state
    or a sequence. For simplicity of the benchmark, we'll provide a 'before'
    and 'after' or just an instance with high reconfiguration needs.
    """
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
                role="adaptive",
                capability_embedding=torch.randn(self.d)
            ))

        tasks = []
        for j in range(self.M):
            tasks.append(Task(
                id=f"task_{j}",
                embedding=torch.randn(self.d)
            ))

        interaction_graph = torch.randn(self.M, self.M) * 0.1
        co_assignment_costs = torch.rand(self.M, self.M) * 0.5
        risk_weights = torch.randn(3 * self.d, 1)

        # To simulate dynamics, we could store the initial 'state' in constraints
        # but for now, we just follow the requested structure.
        return ProblemInstance(
            agents=agents,
            tasks=tasks,
            interaction_graph=interaction_graph,
            co_assignment_costs=co_assignment_costs,
            risk_weights=risk_weights,
            constraints={"dynamic": True}
        )
