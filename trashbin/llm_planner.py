from __future__ import annotations

from typing import List

from orchestrator.base import Agent, Assignment, BaseOrchestrator, Task


class LLMPlannerOrchestrator(BaseOrchestrator):
    """A simple local-planning baseline that assigns each task to the best-matching agent by role."""

    def solve(self, tasks: List[Task], agents: List[Agent]) -> Assignment:
        assignment = Assignment()
        for task in tasks:
            best_agent = None
            best_score = float("-inf")
            for agent in agents:
                score = float(task.embedding @ agent.capability_embedding)
                if score > best_score:
                    best_score = score
                    best_agent = agent
            assignment[task.id] = best_agent.id if best_agent is not None else "unknown"
        return assignment
