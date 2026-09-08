import copy

import pytest
import torch

from landscape import Landscape, LandscapeState, ProblemContext


@pytest.fixture
def sample_problem():
    torch.manual_seed(7)
    N, M, d = 3, 5, 4
    s = torch.randn(N, d)
    c = torch.randn(M, d)
    kappa = torch.randn(N, d)

    X = torch.zeros(N, M)
    assignments = [0, 1, 1, 2, 0]
    for task_idx, agent_idx in enumerate(assignments):
        X[agent_idx, task_idx] = 1.0

    Theta = torch.randn(M, M)
    Theta = (Theta + Theta.T) / 2.0
    Theta.fill_diagonal_(0.0)

    C = torch.rand(M, M)
    C = (C + C.T) / 2.0
    C.fill_diagonal_(0.0)

    W_risk = torch.randn(3 * d, 1)

    problem = ProblemContext(
        s=s,
        c=c,
        C=C,
        W_risk=W_risk,
        N=N,
        M=M,
        d=d,
        lambda_align=0.5,
        lambda_memory=0.5,
        interaction_weight=1.0,
        cost_weight=1.0,
        risk_weight=1.0,
    )

    state = LandscapeState(kappa=kappa, Theta=Theta)
    return problem, state, X


def test_landscape_evaluate_is_deterministic(sample_problem):
    problem, state, X = sample_problem
    landscape = Landscape(problem=problem, state=state)

    e1 = landscape.evaluate(X)
    e2 = landscape.evaluate(X.clone())

    assert torch.isclose(torch.tensor(e1), torch.tensor(e2), atol=1e-8, rtol=1e-8)


def test_landscape_evaluate_is_pure_and_clone_equivalent(sample_problem):
    problem, state, X = sample_problem
    landscape = Landscape(problem=problem, state=state)
    clone = landscape.clone()

    before = landscape.state.kappa.clone()
    before_theta = landscape.state.Theta.clone()

    e_original = landscape.evaluate(X)
    e_clone = clone.evaluate(X)

    assert torch.equal(landscape.state.kappa, before)
    assert torch.equal(landscape.state.Theta, before_theta)
    assert torch.isclose(torch.tensor(e_original), torch.tensor(e_clone), atol=1e-8, rtol=1e-8)


def test_landscape_breakdown_matches_total(sample_problem):
    problem, state, X = sample_problem
    landscape = Landscape(problem=problem, state=state)
    breakdown = landscape.breakdown(X)

    total = float(sum(v for k, v in breakdown.items() if k != "total" and isinstance(v, (int, float))))
    assert pytest.approx(total, rel=1e-6, abs=1e-6) == float(breakdown["total"])


def test_landscape_validate_requires_single_assignment_per_task(sample_problem):
    problem, state, X = sample_problem
    landscape = Landscape(problem=problem, state=state)

    invalid = X.clone()
    invalid[:, 0] = 0.0
    invalid[0, 0] = 1.0
    invalid[1, 0] = 1.0

    assert landscape.validate(X)
    assert not landscape.validate(invalid)


def test_landscape_state_and_problem_context_are_distinct(sample_problem):
    problem, state, X = sample_problem
    landscape = Landscape(problem=problem, state=state)

    assert hasattr(problem, "s")
    assert hasattr(state, "kappa")
    assert hasattr(state, "Theta")
    assert landscape.evaluate(X) == landscape.evaluate(X)


def test_landscape_clone_is_independent(sample_problem):
    problem, state, X = sample_problem
    landscape = Landscape(problem=problem, state=state)
    clone = landscape.clone()

    original_eval = landscape.evaluate(X)
    clone.state.kappa[0, 0] += 1.0
    clone.state.Theta[0, 1] += 1.0

    assert not torch.equal(landscape.state.kappa, clone.state.kappa)
    assert not torch.equal(landscape.state.Theta, clone.state.Theta)
    assert torch.isclose(torch.tensor(original_eval), torch.tensor(landscape.evaluate(X)), atol=1e-6, rtol=1e-6)
    assert not torch.isclose(torch.tensor(original_eval), torch.tensor(clone.evaluate(X)), atol=1e-6, rtol=1e-6)
