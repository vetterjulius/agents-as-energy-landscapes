from __future__ import annotations

import math
from typing import Tuple
import torch

from energy.risk import RiskPredictor
from landscape import LandscapeState, ProblemContext


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
        self.running_co: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset internal running statistics between independent runs."""
        self.running_co = None

    def step(
        self,
        current_state: LandscapeState,
        X: torch.Tensor,
        problem: ProblemContext,
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

            for a in range(N):
                assigned_tasks = (X[a] > 0).nonzero(as_tuple=True)[0]
                if len(assigned_tasks) > 0:
                    task_emb = problem.c[assigned_tasks]
                    success_probs = p[a, assigned_tasks].unsqueeze(1)
                    weighted_update = (task_emb * success_probs).mean(dim=0)
                    new_kappa[a] = (
                        (1.0 - self.eta_memory) * current_state.kappa[a]
                        + self.eta_memory * weighted_update
                    )
                else:
                    new_kappa[a] = (1.0 - self.eta_memory * 0.1) * current_state.kappa[a]
        else:
            # Static and theta-only modes: kappa strictly retains its current state (starts at kappa_0)
            new_kappa = current_state.kappa.clone()

        # 2. Structural Dependency (Theta) adaptation update
        if update_theta:
            new_Theta = current_state.Theta.clone()
            co = X.T @ X
            co_sum = co.sum().item()
            if co_sum >= self.epsilon:
                co_norm = co / (co_sum + self.epsilon)
                if self.running_co is None or self.running_co.shape != (M, M):
                    self.running_co = co_norm.clone()
                    old_running = co_norm.clone()
                else:
                    old_running = self.running_co.clone()
                    self.running_co = (
                        (1.0 - self.eta_theta) * self.running_co
                        + self.eta_theta * co_norm
                    )

                new_Theta = (
                    (1.0 - self.eta_theta) * current_state.Theta
                    + self.eta_theta * (co_norm - old_running)
                )
        else:
            # Static and kappa-only modes: Theta strictly retains Theta_0 across the entire trajectory
            new_Theta = current_state.Theta.clone()

        return LandscapeState(
            kappa=new_kappa,
            Theta=new_Theta,
        )
