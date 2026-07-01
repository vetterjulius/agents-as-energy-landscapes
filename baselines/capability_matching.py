from __future__ import annotations

from typing import List

import torch

from orchestrator.base import Agent, Assignment, BaseOrchestrator, Task


class CapabilityMatchingOrchestrator(BaseOrchestrator):
    """Baseline that assigns each task to the most similar agent by cosine similarity."""

    def solve(self, tasks: List[Task], agents: List[Agent]) -> Assignment:
        assignment = Assignment()
        if not tasks or not agents:
            return assignment

        agent_embeddings = [self._normalize_embedding(agent.capability_embedding) for agent in agents]
        for task in tasks:
            task_embedding = self._normalize_embedding(task.embedding)
            best_agent = None
            best_score = float("-inf")
            for agent, embedding in zip(agents, agent_embeddings):
                score = float(torch.dot(task_embedding, embedding).item())
                if score > best_score:
                    best_score = score
                    best_agent = agent
            assignment[task.id] = best_agent.id if best_agent is not None else agents[0].id
        return assignment

    @staticmethod
    def _normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(embedding, dtype=torch.float32).reshape(-1)
        norm = tensor.norm()
        if norm.item() == 0.0:
            return tensor
        return tensor / norm
