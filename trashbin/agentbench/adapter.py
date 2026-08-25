from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkTask:
    id: str
    instruction: str
    ground_truth: str


class AgentBenchAdapter:
    def __init__(self, path: str | None = None):
        self.path = path

    def load_tasks(self) -> List[BenchmarkTask]:
        from benchmark.agentbench.loader import load_agentbench_tasks

        return load_agentbench_tasks(self.path)
