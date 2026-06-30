from __future__ import annotations

from typing import Dict, List, Tuple

from benchmark.agentbench.adapter import BenchmarkTask
from orchestrator.base import Agent, Assignment, Task


class Executor:
    """Execute a decomposition and return a benchmark-compatible result."""

    def __init__(self, benchmark_task: BenchmarkTask):
        self.benchmark_task = benchmark_task

    def execute(self, assignment: Assignment, tasks: List[Task], agents: List[Agent]) -> Tuple[Dict[str, str], Dict[str, object]]:
        outputs = {}
        for task in tasks:
            agent_id = assignment.get(task.id)
            agent = next((agent for agent in agents if agent.id == agent_id), None)
            outputs[task.id] = {
                "agent": agent_id,
                "role": agent.role if agent else "unknown",
                "instruction": self.benchmark_task.instruction,
                "task": task.id,
            }

        return outputs, {
            "benchmark_task": self.benchmark_task.id,
            "submission": self.benchmark_task.ground_truth,
            "num_tasks": len(tasks),
            "num_agents": len(agents),
        }
