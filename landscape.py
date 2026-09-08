from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from energy.assignment import AssignmentEnergy
from energy.cost import CostEnergy
from energy.interaction import InteractionEnergy
from energy.registry import EnergyRegistry
from energy.risk import RiskEnergy, RiskPredictor


@dataclass
class ProblemContext:
    """Immutable problem-level inputs used to evaluate a candidate assignment.

    This intentionally excludes solver state, proposal state, and adaptation state.
    """

    s: torch.Tensor
    c: torch.Tensor
    C: torch.Tensor
    W_risk: torch.Tensor
    N: int
    M: int
    d: int
    lambda_align: float = 0.5
    lambda_memory: float | None = None
    interaction_weight: float = 1.0
    cost_weight: float = 1.0
    risk_weight: float = 1.0

    def __post_init__(self):
        if self.lambda_memory is None:
            self.lambda_memory = self.lambda_align
        self.s = torch.as_tensor(self.s, dtype=torch.float32)
        self.c = torch.as_tensor(self.c, dtype=torch.float32)
        self.C = torch.as_tensor(self.C, dtype=torch.float32)
        self.W_risk = torch.as_tensor(self.W_risk, dtype=torch.float32)

    def clone(self) -> "ProblemContext":
        return ProblemContext(
            s=self.s.clone(),
            c=self.c.clone(),
            C=self.C.clone(),
            W_risk=self.W_risk.clone(),
            N=self.N,
            M=self.M,
            d=self.d,
            lambda_align=self.lambda_align,
            lambda_memory=self.lambda_memory,
            interaction_weight=self.interaction_weight,
            cost_weight=self.cost_weight,
            risk_weight=self.risk_weight,
        )


@dataclass
class LandscapeState:
    """Mutable landscape parameters that affect the energy but are not X itself."""

    kappa: torch.Tensor
    Theta: torch.Tensor

    def __post_init__(self):
        self.kappa = torch.as_tensor(self.kappa, dtype=torch.float32)
        self.Theta = torch.as_tensor(self.Theta, dtype=torch.float32)

    def clone(self) -> "LandscapeState":
        return LandscapeState(
            kappa=self.kappa.clone(),
            Theta=self.Theta.clone(),
        )


class Landscape:
    """Solver-independent evaluation object for fixed problem context and state."""

    def __init__(
        self,
        problem: ProblemContext,
        state: LandscapeState,
        energy_registry: EnergyRegistry | None = None,
    ):
        self.problem = problem.clone()
        self.state = state.clone()
        self.energy_registry = energy_registry or self._build_default_registry()

    def _build_default_registry(self) -> EnergyRegistry:
        registry = EnergyRegistry()
        registry.add(AssignmentEnergy(self.problem.lambda_align, self.problem.lambda_memory, weight=1.0))
        registry.add(InteractionEnergy(weight=self.problem.interaction_weight))
        registry.add(CostEnergy(weight=self.problem.cost_weight))
        registry.add(RiskEnergy(RiskPredictor(self.problem.d, W_risk=self.problem.W_risk.clone()), weight=self.problem.risk_weight))
        return registry

    def _evaluation_state(self, X: torch.Tensor):
        X_t = torch.as_tensor(X, dtype=torch.float32).clone()
        return type("_EvalState", (), {
            "X": X_t,
            "s": self.problem.s.clone(),
            "c": self.problem.c.clone(),
            "kappa": self.state.kappa.clone(),
            "Theta": self.state.Theta.clone(),
            "C": self.problem.C.clone(),
            "N": self.problem.N,
            "M": self.problem.M,
            "d": self.problem.d,
        })()

    def evaluate(self, X: torch.Tensor) -> float:
        """Pure, deterministic energy evaluation for a candidate assignment X."""

        eval_state = self._evaluation_state(X)
        total, _ = self.energy_registry.compute(eval_state)
        return float(total)

    def breakdown(self, X: torch.Tensor) -> Dict[str, float]:
        eval_state = self._evaluation_state(X)
        total, breakdown = self.energy_registry.compute(eval_state)
        result = dict(breakdown)
        result["total"] = float(total)
        return result

    def validate(self, X: torch.Tensor) -> bool:
        X_t = torch.as_tensor(X, dtype=torch.float32)
        if X_t.shape != (self.problem.N, self.problem.M):
            return False
        if not torch.all((X_t == 0) | (X_t == 1)):
            return False
        if not torch.allclose(X_t.sum(dim=0), torch.ones(self.problem.M), atol=1e-8, rtol=1e-8):
            return False
        return True

    def clone(self) -> "Landscape":
        return Landscape(
            problem=self.problem.clone(),
            state=self.state.clone(),
            energy_registry=self._build_default_registry(),
        )

    def __deepcopy__(self, memo):
        return self.clone()


__all__ = ["ProblemContext", "LandscapeState", "Landscape"]
