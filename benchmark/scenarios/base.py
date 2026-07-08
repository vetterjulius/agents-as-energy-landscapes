from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch

@dataclass
class Task:
    id: str
    embedding: torch.Tensor
    dependencies: List[str] = field(default_factory=list)
    estimated_cost: float = 1.0
    estimated_risk: float = 0.0
    tags: List[str] = field(default_factory=list)

@dataclass
class Agent:
    id: str
    role: str
    capability_embedding: torch.Tensor
    memory_state: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class ProblemInstance:
    agents: List[Agent]
    tasks: List[Task]
    interaction_graph: torch.Tensor  # Theta matrix
    co_assignment_costs: torch.Tensor # C matrix
    risk_weights: torch.Tensor       # W_risk
    constraints: Dict = field(default_factory=dict)
    ground_truth: Optional[torch.Tensor] = None # Best known X

class Scenario:
    def generate(self, seed: int) -> ProblemInstance:
        raise NotImplementedError
