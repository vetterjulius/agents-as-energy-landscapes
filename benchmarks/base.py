from abc import ABC, abstractmethod
from typing import List
import torch
from state.orchestration_state import OrchestrationState

class BenchmarkScenario:
    def __init__(self, name: str, N: int, M: int, d: int,
                 s: torch.Tensor, c: torch.Tensor, C: torch.Tensor,
                 W_risk: torch.Tensor, X_init: torch.Tensor):
        """
        Represents a specific task assignment problem instance.
        
        Args:
            name: Description/name of this scenario.
            N: Number of agents.
            M: Number of tasks.
            d: Embedding dimensions.
            s: Agent embedding matrix of shape (N, d).
            c: Task embedding matrix of shape (M, d).
            C: Co-assignment cost matrix of shape (M, M).
            W_risk: Risk predictor weights of shape (3*d, 1).
            X_init: Initial assignment matrix of shape (N, M).
        """
        self.name = name
        self.N = N
        self.M = M
        self.d = d
        self.s = s
        self.c = c
        self.C = C
        self.W_risk = W_risk
        self.X_init = X_init

    def to_state(self) -> OrchestrationState:
        """Converts this scenario's parameters into a fresh OrchestrationState."""
        kappa = torch.zeros(self.N, self.d)
        Theta = torch.zeros(self.M, self.M)
        return OrchestrationState(
            X=self.X_init.clone(),
            s=self.s.clone(),
            c=self.c.clone(),
            kappa=kappa,
            Theta=Theta,
            C=self.C.clone(),
            N=self.N,
            M=self.M,
            d=self.d
        )

class BaseBenchmark(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The display name of the benchmark suite."""
        pass

    @abstractmethod
    def load_scenarios(self) -> List[BenchmarkScenario]:
        """Loads or generates the list of scenarios for this benchmark."""
        pass
