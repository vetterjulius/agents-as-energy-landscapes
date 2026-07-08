"""Whole system acts as if it was a single agent"""

from orchestrator.energy_orchestrator import EnergyOrchestrator
from orchestrator.executor import Executor
from orchestrator.task_decomposer import TaskDecomposer


class SingleAgentBase:
    def __init__(self, agents):
        self.agents = agents
    def invoke(self, question):
        raise NotImplementedError()

class EnergySingleAgent(SingleAgentBase):
    def invoke(self, question):
        tasks = TaskDecomposer().decompose(question)
        assignment = EnergyOrchestrator().solve(tasks, self.agents)
        answer = Executor().execute(assignment, tasks, self.agents)
        return answer
