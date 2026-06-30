from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class Task:
    def __init__(self, id: str, embedding: torch.Tensor, dependencies: List[str] | None = None,
                 estimated_cost: float = 1.0, estimated_risk: float = 0.0):
        self.id = id
        self.embedding = torch.as_tensor(embedding, dtype=torch.float32)
        self.dependencies = dependencies or []
        self.estimated_cost = estimated_cost
        self.estimated_risk = estimated_risk


class Agent:
    def __init__(self, id: str, role: str, capability_embedding: torch.Tensor, memory_state: Dict | None = None):
        self.id = id
        self.role = role
        self.capability_embedding = torch.as_tensor(capability_embedding, dtype=torch.float32)
        self.memory_state = memory_state or {}


class Assignment(dict):
    """Mapping from task id to agent id."""


class BaseOrchestrator(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def solve(self, tasks: List[Task], agents: List[Agent]) -> Assignment:
        raise NotImplementedError
