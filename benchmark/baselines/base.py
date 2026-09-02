import torch
from typing import Dict
from ..scenarios.base import ProblemInstance

class Orchestrator:
    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        raise NotImplementedError
