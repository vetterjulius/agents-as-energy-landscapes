from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from typing import Any

import torch

from energy.risk import RiskPredictor
from landscape import Landscape, ProblemContext


def _as_problem(problem_or_landscape: Any) -> ProblemContext:
    if isinstance(problem_or_landscape, Landscape):
        return problem_or_landscape.problem
    if isinstance(problem_or_landscape, ProblemContext):
        return problem_or_landscape
    if hasattr(problem_or_landscape, "problem"):
        return problem_or_landscape.problem
    raise TypeError("Expected a ProblemContext or Landscape-like object.")


def _as_landscape(landscape_or_problem: Any) -> Landscape:
    if isinstance(landscape_or_problem, Landscape):
        return landscape_or_problem
    if hasattr(landscape_or_problem, "problem") and hasattr(landscape_or_problem, "state"):
        return landscape_or_problem
    raise TypeError("Expected a Landscape object.")


def enumerate_valid_assignments(problem_or_landscape: Any) -> Iterator[torch.Tensor]:
    """Yield every valid one-hot assignment matrix for a tiny problem instance."""
    problem = _as_problem(problem_or_landscape)
    for choices in itertools.product(range(problem.N), repeat=problem.M):
        X = torch.zeros(problem.N, problem.M, dtype=torch.float32)
        for task_idx, agent_idx in enumerate(choices):
            X[agent_idx, task_idx] = 1.0
        yield X


def _risk_matrix(landscape: Landscape) -> torch.Tensor:
    problem = landscape.problem
    state = landscape.state
    s_exp = problem.s.unsqueeze(1).expand(-1, problem.M, -1)
    c_exp = problem.c.unsqueeze(0).expand(problem.N, -1, -1)
    k_exp = state.kappa.unsqueeze(1).expand(-1, problem.M, -1)
    x_feat = torch.cat([s_exp, c_exp, k_exp], dim=-1)
    logits = torch.matmul(x_feat, problem.W_risk).squeeze(-1)
    logits = logits / math.sqrt(max(problem.d, 1))
    return torch.sigmoid(logits)


def compiled_ilp_objective(X: torch.Tensor, landscape: Landscape) -> float:
    """Compute the exact scalar objective used by the fixed landscape for a candidate X."""
    landscape_obj = _as_landscape(landscape)
    problem = landscape_obj.problem
    state = landscape_obj.state

    X_t = torch.as_tensor(X, dtype=torch.float32, device=problem.s.device).clone()
    if X_t.shape != (problem.N, problem.M):
        raise ValueError(f"Expected X with shape {(problem.N, problem.M)}, got {tuple(X_t.shape)}")
    if not torch.all((X_t == 0) | (X_t == 1)):
        raise ValueError("ILP objective requires a binary assignment matrix X.")
    if not torch.allclose(X_t.sum(dim=0), torch.ones(problem.M), atol=1e-8, rtol=1e-8):
        raise ValueError("X violates the one-agent-per-task assignment constraint.")

    dist = torch.cdist(problem.s, problem.c) ** 2
    align_sc = problem.s @ problem.c.T
    align_mem = (problem.s * state.kappa).sum(dim=1, keepdim=True)
    assignment_score = dist - problem.lambda_align * align_sc - problem.lambda_memory * align_mem
    assignment_term = (X_t * assignment_score).sum() / (problem.N * problem.M)

    risk_probs = _risk_matrix(landscape_obj)
    risk_term = -(X_t * torch.log(risk_probs + 1e-8)).sum() / (problem.N * problem.M)

    co = X_t.T @ X_t
    interaction_matrix = state.Theta * co
    upper_mask = torch.triu(torch.ones_like(interaction_matrix), diagonal=1)
    interaction_term = -(interaction_matrix * upper_mask).sum() / (problem.N * problem.M)

    cost_term = (problem.C * co).sum() / (problem.N * problem.M)

    total = (
        assignment_term
        + problem.interaction_weight * interaction_term
        + problem.cost_weight * cost_term
        + problem.risk_weight * risk_term
    )
    return float(total)


def compile_landscape_to_ilp(landscape: Landscape) -> dict[str, Any]:
    """Compile the current fixed landscape into an exact ILP-style representation.

    The compilation does not alter any energy formulas or solver state semantics. It
    stores the exact coefficients derived from the fixed ProblemContext and LandscapeState.
    """
    landscape_obj = _as_landscape(landscape)
    problem = landscape_obj.problem
    state = landscape_obj.state

    dist = torch.cdist(problem.s, problem.c) ** 2
    align_sc = problem.s @ problem.c.T
    align_mem = (problem.s * state.kappa).sum(dim=1, keepdim=True)
    assignment_coefficients = (
        dist
        - problem.lambda_align * align_sc
        - problem.lambda_memory * align_mem
    ) / (problem.N * problem.M)

    risk_coefficients = -torch.log(_risk_matrix(landscape_obj) + 1e-8) / (problem.N * problem.M)

    model = {
        "kind": "exact_landscape_ilp",
        "problem": {
            "N": problem.N,
            "M": problem.M,
            "d": problem.d,
            "lambda_align": problem.lambda_align,
            "lambda_memory": problem.lambda_memory,
            "interaction_weight": problem.interaction_weight,
            "cost_weight": problem.cost_weight,
            "risk_weight": problem.risk_weight,
        },
        "state": {
            "kappa": state.kappa.clone(),
            "Theta": state.Theta.clone(),
            "C": problem.C.clone(),
        },
        "landscape": landscape_obj,
        "objective": {
            "assignment_coefficients": assignment_coefficients,
            "risk_coefficients": risk_coefficients,
            "interaction_weight": problem.interaction_weight,
            "cost_weight": problem.cost_weight,
            "risk_weight": problem.risk_weight,
        },
        "constraints": {
            "assignment": "sum_a X[a, t] = 1 for all tasks t",
            "binary": "X[a, t] in {0, 1}",
            "solver_state_isolation": "LandscapeState is fixed and solver-independent",
        },
    }
    return model


def solve(landscape_or_model: Any) -> torch.Tensor:
    """Return the minimal-energy valid assignment for a tiny fixed landscape instance."""
    if isinstance(landscape_or_model, dict) and "landscape" in landscape_or_model:
        landscape = landscape_or_model["landscape"]
    else:
        landscape = _as_landscape(landscape_or_model)

    problem = landscape.problem
    best_X = None
    best_E = float("inf")
    for X in enumerate_valid_assignments(problem):
        E = compiled_ilp_objective(X, landscape)
        if E < best_E:
            best_E = E
            best_X = X.clone()
    if best_X is None:
        raise ValueError("No valid assignments found for the given problem.")
    return best_X


__all__ = [
    "ProblemContext",
    "Landscape",
    "compile_landscape_to_ilp",
    "compiled_ilp_objective",
    "enumerate_valid_assignments",
    "solve",
]
