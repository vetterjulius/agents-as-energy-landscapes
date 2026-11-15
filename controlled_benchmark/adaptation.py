from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import torch

from energy.risk import RiskPredictor
from landscape import LandscapeState, ProblemContext


def compute_cooccurrence_observation(X: torch.Tensor) -> torch.Tensor:
    """Compute the unnormalized, symmetric off-diagonal co-assignment observation matrix C.

    C = offdiag((X.T @ X + (X.T @ X).T) / 2)

    Preserves input dtype and device, has a zero diagonal, and returns all-zeros
    for an empty or all-zero assignment input without modifying X in place.
    """
    if X.numel() == 0:
        M = X.shape[1] if X.ndim >= 2 else 0
        return torch.zeros((M, M), dtype=X.dtype, device=X.device)
    co = X.T @ X
    if co.is_floating_point():
        C = (co + co.T) / 2.0
    else:
        C = torch.div(co + co.T, 2, rounding_mode="floor")
    C = C.clone()
    C.fill_diagonal_(0)
    return C


class EpisodeAdaptationManager:
    """
    Explicit episode-boundary adaptation manager.

    Strictly isolated from solvers:
    - Never mutated during an episode's optimization.
    - Updated ONLY at the explicit episode boundary:
        (episode t, assignment X_t) -> state_{t+1}.
    - Preserves exact existing adaptation formulas without modification.
    """

    def __init__(
        self,
        adaptation_mode: str = "full",  # 'static', 'kappa-only', 'theta-only', 'full'
        eta_memory: float = 0.05,
        eta_theta: float = 0.10,
        epsilon: float = 1e-8,
    ):
        norm_mode = adaptation_mode.strip().lower().replace("_", "-").replace("adaptive-", "")
        if norm_mode in ("static", "none"):
            self.mode = "static"
        elif norm_mode in ("kappa-only", "kappa"):
            self.mode = "kappa-only"
        elif norm_mode in ("theta-only", "theta"):
            self.mode = "theta-only"
        elif norm_mode in ("full", "adaptive-full", "adaptive"):
            self.mode = "full"
        else:
            raise ValueError(f"Unknown adaptation mode: {adaptation_mode}")

        self.eta_memory = eta_memory
        self.eta_theta = eta_theta
        self.epsilon = epsilon

    def reset(self) -> None:
        """Reset internal running statistics between independent runs."""
        pass

    @staticmethod
    def _offdiag(A: torch.Tensor) -> torch.Tensor:
        """Return A with all diagonal elements set to zero."""
        result = A.clone()
        result.fill_diagonal_(0.0)
        return result

    def step(
        self,
        current_state: LandscapeState,
        X: torch.Tensor,
        problem: ProblemContext,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> LandscapeState:
        """
        Compute explicit episode boundary adaptation from current and past observations only.

        Args:
            current_state: LandscapeState at episode t.
            X: Binary assignment matrix (N, M) chosen at episode t.
            problem: ProblemContext of episode t.

        Returns:
            New LandscapeState for episode t+1.
            Strictly contains zero information from episode t+1 or beyond.
        """
        N = problem.N
        M = problem.M
        d = problem.d

        update_kappa = self.mode in ("kappa-only", "full")
        update_theta = self.mode in ("theta-only", "full")

        # 1. Memory (kappa) adaptation update
        risk_probabilities = torch.zeros(N, M, dtype=current_state.kappa.dtype)
        kappa_target = torch.zeros_like(current_state.kappa)
        if update_kappa:
            new_kappa = current_state.kappa.clone()
            risk_pred = RiskPredictor(d, W_risk=problem.W_risk)

            # Predict risk probabilities p: shape (N, M)
            s_exp = problem.s.unsqueeze(1).expand(-1, M, -1)
            c_exp = problem.c.unsqueeze(0).expand(N, -1, -1)
            k_exp = current_state.kappa.unsqueeze(1).expand(-1, M, -1)
            x_feat = torch.cat([s_exp, c_exp, k_exp], dim=-1)
            logits = torch.matmul(x_feat, problem.W_risk).squeeze(-1) / math.sqrt(max(d, 1))
            p = torch.sigmoid(logits)
            risk_probabilities = p.clone()

            for a in range(N):
                assigned_tasks = (X[a] > 0).nonzero(as_tuple=True)[0]
                if len(assigned_tasks) > 0:
                    task_emb = problem.c[assigned_tasks]
                    success_probs = p[a, assigned_tasks].unsqueeze(1)
                    weighted_update = (task_emb * success_probs).mean(dim=0)
                    kappa_target[a] = weighted_update
                    new_kappa[a] = (
                        (1.0 - self.eta_memory) * current_state.kappa[a]
                        + self.eta_memory * weighted_update
                    )
                else:
                    new_kappa[a] = (1.0 - self.eta_memory * 0.1) * current_state.kappa[a]
        else:
            # Static and theta-only modes: kappa strictly retains its current state (starts at kappa_0)
            new_kappa = current_state.kappa.clone()
            if diagnostics is not None:
                s_exp = problem.s.unsqueeze(1).expand(-1, M, -1)
                c_exp = problem.c.unsqueeze(0).expand(N, -1, -1)
                k_exp = current_state.kappa.unsqueeze(1).expand(-1, M, -1)
                x_feat = torch.cat([s_exp, c_exp, k_exp], dim=-1)
                logits = torch.matmul(x_feat, problem.W_risk).squeeze(-1) / math.sqrt(max(d, 1))
                risk_probabilities = torch.sigmoid(logits)

        # 2. Structural Dependency (Theta) adaptation update
        #
        # Observation semantics: unnormalized, symmetric off-diagonal co-assignment indicators.
        #
        #   C = offdiag((X_t.T @ X_t + (X_t.T @ X_t).T) / 2)
        #   Theta_new = (1 - eta_theta) * Theta_old + eta_theta * C
        #   Theta_new = offdiag((Theta_new + Theta_new.T) / 2)
        #
        # Theta is updated exclusively from the observed assignment X_t.
        # No running_co baseline. No additive residual/integrator semantics.
        theta_observation = compute_cooccurrence_observation(X)
        if update_theta:
            if X.numel() > 0 and (X.abs().sum().item() >= self.epsilon):
                # EMA update toward C_t
                Theta_new = (1.0 - self.eta_theta) * current_state.Theta + self.eta_theta * theta_observation

                # Re-symmetrize and zero diagonal
                new_Theta = self._offdiag((Theta_new + Theta_new.T) / 2.0)
            else:
                # No assignments observed: Theta unchanged
                new_Theta = current_state.Theta.clone()
        else:
            # Static and kappa-only modes: Theta strictly retains Theta_0 across the entire trajectory
            new_Theta = current_state.Theta.clone()

        if diagnostics is not None:
            diagnostics.update(
                {
                    "kappa_update_active": update_kappa,
                    "theta_update_active": update_theta,
                    "risk_probabilities": risk_probabilities.clone(),
                    "kappa_target": kappa_target.clone(),
                    "theta_observation": theta_observation.clone(),
                }
            )

        return LandscapeState(
            kappa=new_kappa,
            Theta=new_Theta,
        )
