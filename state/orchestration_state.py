from dataclasses import dataclass
import torch

@dataclass
class OrchestrationState:
    X: torch.Tensor
    s: torch.Tensor
    c: torch.Tensor
    kappa: torch.Tensor
    Theta: torch.Tensor
    C: torch.Tensor
    N: int
    M: int
    d: int
