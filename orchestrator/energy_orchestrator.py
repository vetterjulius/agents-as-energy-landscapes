from __future__ import annotations

from typing import List

import torch

from orchestrator.base import Agent, Assignment, BaseOrchestrator, Task


class EnergyOrchestrator(BaseOrchestrator):
    """A lightweight energy-based orchestrator that uses the same scoring heuristic as the benchmark baseline."""

    def solve(self, tasks: List[Task], agents: List[Agent]) -> Assignment:
        assignment = Assignment()
        if not tasks or not agents:
            return assignment

        agent_embeddings = torch.stack([agent.capability_embedding for agent in agents])
        task_embeddings = torch.stack([task.embedding for task in tasks])
        similarity = task_embeddings @ agent_embeddings.T

        for idx, task in enumerate(tasks):
            best_idx = int(torch.argmax(similarity[idx]).item())
            assignment[task.id] = agents[best_idx].id
        return assignment
