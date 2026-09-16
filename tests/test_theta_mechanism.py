from __future__ import annotations

import torch

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from landscape import LandscapeState, ProblemContext


N = M = 4
D = 2
ETA = 0.2


def _problem(ground_truth: torch.Tensor | None = None) -> ProblemContext:
    """Create a deterministic context; C is varied only as a ground-truth proxy."""
    if ground_truth is None:
        ground_truth = torch.zeros(M, M)
    return ProblemContext(
        s=torch.zeros(N, D),
        c=torch.zeros(M, D),
        C=ground_truth.clone(),
        W_risk=torch.zeros(3 * D, 1),
        N=N,
        M=M,
        d=D,
    )


def _state() -> LandscapeState:
    return LandscapeState(
        kappa=torch.zeros(N, D),
        Theta=torch.zeros(M, M),
    )


def _assignment(pair_a: tuple[int, int], pair_b: tuple[int, int]) -> torch.Tensor:
    """Assign each pair of task indices to one agent; all other agents are idle."""
    X = torch.zeros(N, M)
    X[0, list(pair_a)] = 1.0
    X[1, list(pair_b)] = 1.0
    return X


def _observed_C(X: torch.Tensor) -> torch.Tensor:
    """Reference C calculation matching EpisodeAdaptationManager.step."""
    co = X.T @ X
    co_norm = co / (co.sum() + 1e-8)
    C = (co_norm + co_norm.T) / 2.0
    C.fill_diagonal_(0.0)
    return C


def _run(manager: EpisodeAdaptationManager, state: LandscapeState,
        problem: ProblemContext, X: torch.Tensor, steps: int) -> LandscapeState:
    for _ in range(steps):
        state = manager.step(state, X, problem)
    return state


def test_theta_learns_known_observed_co_assignment_structure():
    """Theta converges to observed pairs (0,1) and (2,3), not arbitrary edges."""
    problem = _problem()
    X_A = _assignment((0, 1), (2, 3))
    manager = EpisodeAdaptationManager("theta-only", eta_theta=ETA)

    state = _run(manager, _state(), problem, X_A, steps=60)
    theta = state.Theta

    assert theta[0, 1] == theta[1, 0]
    assert theta[2, 3] == theta[3, 2]
    assert theta[0, 1] > theta[0, 2]
    assert theta[2, 3] > theta[2, 1]
    assert torch.allclose(theta, theta.T, atol=1e-7)
    assert torch.allclose(theta.diagonal(), torch.zeros(M), atol=1e-7)
    assert torch.allclose(theta, _observed_C(X_A), atol=1e-4)


def test_theta_moves_toward_new_observed_structure_without_ema_jump():
    """After A→B, Theta moves toward B gradually and eventually tracks B."""
    problem = _problem()
    X_A = _assignment((0, 1), (2, 3))
    X_B = _assignment((0, 2), (1, 3))
    C_B = _observed_C(X_B)
    manager = EpisodeAdaptationManager("theta-only", eta_theta=ETA)

    state = _run(manager, _state(), problem, X_A, steps=60)
    distance_before = torch.dist(state.Theta, C_B).item()

    after_one = manager.step(state, X_B, problem)
    distance_after_one = torch.dist(after_one.Theta, C_B).item()

    assert distance_after_one < distance_before
    assert not torch.allclose(after_one.Theta, C_B, atol=1e-6)

    state = _run(manager, after_one, problem, X_B, steps=60)
    assert torch.dist(state.Theta, C_B).item() < distance_after_one
    assert state.Theta[0, 2] > state.Theta[0, 1]
    assert state.Theta[1, 3] > state.Theta[1, 0]
    assert torch.allclose(state.Theta, C_B, atol=1e-4)
    assert torch.allclose(state.Theta, state.Theta.T, atol=1e-7)
    assert torch.allclose(state.Theta.diagonal(), torch.zeros(M), atol=1e-7)


def test_theta_follows_observation_and_ignores_ground_truth_changes():
    """With identical observed X, changing a ground-truth proxy cannot change Theta."""
    X_A = _assignment((0, 1), (2, 3))
    X_B = _assignment((0, 2), (1, 3))
    ground_truth_A = _observed_C(X_A)
    ground_truth_B = _observed_C(X_B)

    manager_a = EpisodeAdaptationManager("theta-only", eta_theta=ETA)
    manager_b = EpisodeAdaptationManager("theta-only", eta_theta=ETA)
    state_a = _run(manager_a, _state(), _problem(ground_truth_A), X_B, steps=60)
    state_b = _run(manager_b, _state(), _problem(ground_truth_B), X_B, steps=60)

    expected_B = _observed_C(X_B)
    assert torch.allclose(state_a.Theta, expected_B, atol=1e-4)
    assert torch.allclose(state_b.Theta, expected_B, atol=1e-4)
    assert torch.allclose(state_a.Theta, state_b.Theta, atol=1e-7)
    assert not torch.allclose(state_a.Theta, ground_truth_A, atol=1e-4)
