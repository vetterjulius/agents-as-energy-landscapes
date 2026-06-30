from __future__ import annotations

from typing import Dict, List

from benchmark.agentbench.adapter import AgentBenchAdapter, BenchmarkTask
from orchestrator.base import Agent, Assignment, Task
from orchestrator.executor import Executor
from orchestrator.llm_planner import LLMPlannerOrchestrator
from orchestrator.energy_orchestrator import EnergyOrchestrator
from orchestrator.task_decomposer import TaskDecomposer


class PublicBenchmarkRunner:
    def __init__(self, adapter: AgentBenchAdapter | None = None):
        self.adapter = adapter or AgentBenchAdapter()
        self.decomposer = TaskDecomposer()

    def run(self, orchestrator_name: str = "energy") -> List[Dict[str, object]]:
        results = []
        tasks = self.adapter.load_tasks()
        for benchmark_task in tasks:
            subtasks = self.decomposer.decompose(benchmark_task)
            agents = self._build_agents()
            orchestrator = self._build_orchestrator(orchestrator_name)
            assignment = orchestrator.solve(subtasks, agents)
            executor = Executor(benchmark_task)
            outputs, metadata = executor.execute(assignment, subtasks, agents)
            results.append({
                "benchmark_task": benchmark_task.id,
                "assignment": assignment,
                "outputs": outputs,
                "metadata": metadata,
            })
        return results

    def _build_orchestrator(self, orchestrator_name: str):
        if orchestrator_name == "llm":
            return LLMPlannerOrchestrator(cfg={})
        if orchestrator_name == "energy":
            return EnergyOrchestrator(cfg={})
        return EnergyOrchestrator(cfg={})

    def _build_agents(self) -> List[Agent]:
        return [
            Agent(id="planner", role="Planner", capability_embedding=[1.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Agent(id="sql-expert", role="SQL Expert", capability_embedding=[0.8, 0.9, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0]),
            Agent(id="verifier", role="Verifier", capability_embedding=[0.6, 0.2, 0.8, 0.7, 0.1, 0.0, 0.0, 0.0]),
            Agent(id="debugger", role="Debugger", capability_embedding=[0.7, 0.4, 0.3, 0.9, 0.2, 0.0, 0.0, 0.0]),
        ]
