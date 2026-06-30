from __future__ import annotations

from typing import List

import torch

from benchmark.agentbench.adapter import BenchmarkTask
from orchestrator.base import Task


class TaskDecomposer:
    """Decompose a benchmark instruction into a small task graph."""

    def decompose(self, task: BenchmarkTask) -> List[Task]:
        instruction = task.instruction.lower()
        template = [
            "schema analysis",
            "entity identification",
            "query planning",
            "sql generation",
            "verification",
        ]
        if "sql" in instruction:
            template = [
                "schema analysis",
                "query planning",
                "sql generation",
                "verification",
            ]

        embeddings = []
        for label in template:
            emb = torch.zeros(8, dtype=torch.float32)
            for idx, char in enumerate(label):
                emb[idx % 8] += (ord(char) % 7) / 10.0
            embeddings.append(emb)

        return [
            Task(
                id=f"{task.id}:{idx}",
                embedding=embedding,
                dependencies=[] if idx == 0 else [f"{task.id}:{idx - 1}"],
                estimated_cost=1.0 + 0.2 * idx,
                estimated_risk=0.05 * (idx + 1),
            )
            for idx, embedding in enumerate(embeddings)
        ]
