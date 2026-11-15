from __future__ import annotations

import pytest
import torch

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from landscape import LandscapeState, ProblemContext


def make_problem_and_state(M: int = 3, theta_val: float = 1.0):
    N, d = 2, 2
    problem = ProblemContext(
        s=torch.zeros(N, d),
        c=torch.zeros(M, d),
        C=torch.zeros(M, M),
        W_risk=torch.zeros(3 * d, 1),
        N=N,
        M=M,
        d=d,
        interaction_weight=1.0,
        cost_weight=1.0,
    )
    initial_state = LandscapeState(
        kappa=torch.zeros(N, d),
        Theta=torch.full((M, M), theta_val),
    )
    return problem, initial_state


def _compute_C(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Independent oracle: compute unnormalized, symmetric, off-diagonal co-assignment matrix C from X."""
    if X.numel() == 0 or X.abs().sum().item() < eps:
        M = X.shape[1] if X.ndim >= 2 else 0
        return torch.zeros((M, M), dtype=X.dtype, device=X.device)
    co = X.T @ X
    C = (co + co.T) / 2.0
    C = C.clone()
    C.fill_diagonal_(0.0)
    return C


def test_stationary_ema_converges_to_C():
    """1. Stationärer Fall: Repeated EMA updates with constant X converge Theta toward C."""
    problem, state = make_problem_and_state(M=3, theta_val=1.0)
    eta = 0.5
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=eta)

    X = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C = _compute_C(X)

    curr_state = state
    for _ in range(60):
        curr_state = manager.step(curr_state, X, problem)

    # After many EMA steps Theta must be very close to C (the EMA fixed point)
    assert torch.allclose(curr_state.Theta, C, atol=1e-4)


def test_constant_X_ema_fixed_point():
    """2. Konstantes X: After many steps, Theta converges to C (the EMA fixed point)."""
    problem, state = make_problem_and_state(M=3, theta_val=0.5)
    eta = 0.3
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=eta)

    X = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C = _compute_C(X)

    curr_state = state
    for _ in range(80):
        curr_state = manager.step(curr_state, X, problem)

    assert torch.allclose(curr_state.Theta, C, atol=1e-4)


def test_positive_signal_directional_movement():
    """3. Positives Signal: After assigning tasks 0 and 1 together, Theta[0,1] > 0."""
    problem, state = make_problem_and_state(M=3, theta_val=0.0)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.5)

    # X0: agents each assigned one task - tasks 0 and 1 NOT co-assigned
    X0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    state1 = manager.step(state, X0, problem)

    # X1: agent 0 assigned tasks 0 and 1 together -> C[0,1] > 0
    X1 = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    state2 = manager.step(state1, X1, problem)

    assert state2.Theta[0, 1].item() > 0.0


def test_numerical_stability():
    """4. Stabilität: No NaNs/Infs for normal inputs and edge cases (e.g. zero assignments)."""
    problem, state = make_problem_and_state(M=3, theta_val=1.0)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.1)

    X_zero = torch.zeros(2, 3)
    res_state = manager.step(state, X_zero, problem)

    assert not torch.isnan(res_state.Theta).any()
    assert not torch.isinf(res_state.Theta).any()


def test_symmetry_preservation():
    """5. Symmetrie: Theta symmetry is preserved after updates."""
    problem, state = make_problem_and_state(M=3, theta_val=0.5)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.1)

    X0 = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    state1 = manager.step(state, X0, problem)
    state2 = manager.step(state1, X0, problem)

    assert torch.allclose(state2.Theta, state2.Theta.T, atol=1e-7)


def test_diagonal_is_zero():
    """6. Diagonale: Theta diagonal must remain 0 after updates."""
    problem, state = make_problem_and_state(M=3, theta_val=0.0)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.3)

    X = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    curr_state = state
    for _ in range(10):
        curr_state = manager.step(curr_state, X, problem)

    diag = curr_state.Theta.diagonal()
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-7)


def test_ema_update_formula_one_step():
    """7. EMA-Formel: Verify one-step update matches the exact formula."""
    M = 3
    eta = 0.4
    problem, state = make_problem_and_state(M=M, theta_val=0.0)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=eta)

    X = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C = _compute_C(X)

    # One step: Theta_new = offdiag((1-eta)*Theta_old + eta*C + T) / 2)
    Theta_old = state.Theta.clone()
    Theta_expected_raw = (1.0 - eta) * Theta_old + eta * C
    Theta_expected = (Theta_expected_raw + Theta_expected_raw.T) / 2.0
    Theta_expected.fill_diagonal_(0.0)

    state1 = manager.step(state, X, problem)
    assert torch.allclose(state1.Theta, Theta_expected, atol=1e-6)


def test_zero_assignment_theta_unchanged():
    """8. Null-Zuweisung: With zero-assignment X, Theta must not change."""
    problem, state = make_problem_and_state(M=3, theta_val=2.0)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.1)

    X_zero = torch.zeros(2, 3)
    state1 = manager.step(state, X_zero, problem)

    assert torch.allclose(state1.Theta, state.Theta, atol=1e-7)


def test_cooccurrence_observation_properties():
    """Test canonical observation properties: binary indicators, symmetry, dtype, zero-diagonal."""
    from controlled_benchmark.adaptation import compute_cooccurrence_observation

    # Valid one-hot assignment (each task to exactly one agent):
    # Agent 0: tasks 0, 1
    # Agent 1: task 2
    X = torch.tensor([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    X_clone = X.clone()

    C = compute_cooccurrence_observation(X)

    # 1. Preserves dtype and device
    assert C.dtype == X.dtype
    assert C.device == X.device

    # 2. Input tensor not modified in place
    assert torch.equal(X, X_clone)

    # 3. Exact symmetry
    assert torch.equal(C, C.T)

    # 4. Zero diagonal
    assert torch.equal(C.diagonal(), torch.zeros(3, dtype=torch.float64))

    # 5. Correct binary off-diagonal co-assignment indicators
    expected = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=torch.float64)
    assert torch.equal(C, expected)

    # 6. All-zero assignment input
    X_all_zero = torch.zeros((2, 3), dtype=torch.float32)
    C_zero = compute_cooccurrence_observation(X_all_zero)
    assert torch.equal(C_zero, torch.zeros((3, 3), dtype=torch.float32))

    # 7. Empty assignment input
    X_empty = torch.zeros((0, 3), dtype=torch.float32)
    C_empty = compute_cooccurrence_observation(X_empty)
    assert torch.equal(C_empty, torch.zeros((3, 3), dtype=torch.float32))


def test_diagnostics_theta_observation_agreement():
    """The observation used for adaptation must match diagnostics['theta_observation']."""
    problem, state = make_problem_and_state(M=3, theta_val=0.5)
    eta = 0.2
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=eta)
    X = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    diag: dict = {}
    new_state = manager.step(state, X, problem, diagnostics=diag)

    C_oracle = _compute_C(X)
    assert "theta_observation" in diag
    assert torch.equal(diag["theta_observation"], C_oracle)

    expected_theta = (1.0 - eta) * state.Theta + eta * diag["theta_observation"]
    expected_theta.fill_diagonal_(0.0)
    assert torch.allclose(new_state.Theta, expected_theta, atol=1e-7)
