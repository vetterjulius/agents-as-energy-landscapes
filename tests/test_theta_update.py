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


def test_stationary_case_no_decay():
    """1. Stationärer Fall: Repeated updates with constant co-assignment should not decay Theta to 0."""
    problem, state = make_problem_and_state(M=3, theta_val=1.0)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.1)

    X = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    initial_norm = state.Theta.norm().item()
    curr_state = state
    for _ in range(50):
        curr_state = manager.step(curr_state, X, problem)

    final_norm = curr_state.Theta.norm().item()
    assert final_norm == pytest.approx(initial_norm, abs=1e-6)


def test_no_signal_no_drift():
    """2. Kein Signal: When co_norm matches running_co exactly, Theta experiences zero artificial drift."""
    problem, state = make_problem_and_state(M=3, theta_val=0.5)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.2)

    X = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    state1 = manager.step(state, X, problem)
    state2 = manager.step(state1, X, problem)

    assert torch.allclose(state1.Theta, state2.Theta, atol=1e-7)


def test_positive_signal_directional_movement():
    """3. Positives Signal: When C_t deviates systematically from R, Theta moves in expected direction."""
    N, M, d = 2, 3, 2
    problem, state = make_problem_and_state(M=3, theta_val=0.0)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.5)

    X0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    state1 = manager.step(state, X0, problem)

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
    """5. Symmetrie: Theta symmetry is preserved after updates if input is symmetric."""
    problem, state = make_problem_and_state(M=3, theta_val=0.5)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.1)

    X0 = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    state1 = manager.step(state, X0, problem)
    state2 = manager.step(state1, X0, problem)

    assert torch.allclose(state2.Theta, state2.Theta.T, atol=1e-7)


def test_initialization_t0_no_spurious_update():
    """6. Initialisierung: At t=0, no unintended artificial update step occurs on Theta."""
    problem, state = make_problem_and_state(M=3, theta_val=2.0)
    manager = EpisodeAdaptationManager(adaptation_mode="theta-only", eta_theta=0.1)

    X0 = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    state1 = manager.step(state, X0, problem)

    assert torch.allclose(state1.Theta, state.Theta, atol=1e-7)