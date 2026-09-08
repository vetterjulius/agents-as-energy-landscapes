import torch

from dynamics.proposal import AssignmentProposal
from dynamics.sampler import SimulatedAnnealingSampler
from ilp import (
    compiled_ilp_objective,
    compile_landscape_to_ilp,
    enumerate_valid_assignments,
)
from landscape import Landscape, LandscapeState, ProblemContext


def make_small_problem():
    torch.manual_seed(0)
    N, M, d = 2, 3, 2
    s = torch.tensor(
        [[1.0, 0.2], [0.4, 1.1]],
        dtype=torch.float32,
    )
    c = torch.tensor(
        [[0.8, 1.5], [1.2, 0.3], [0.5, 0.9]],
        dtype=torch.float32,
    )
    kappa = torch.tensor(
        [[0.3, -0.1], [-0.2, 0.7]],
        dtype=torch.float32,
    )

    Theta = torch.tensor(
        [[0.0, 0.5, -0.2], [0.5, 0.0, 0.8], [-0.2, 0.8, 0.0]],
        dtype=torch.float32,
    )
    C = torch.tensor(
        [[0.0, 0.7, 0.3], [0.7, 0.0, 0.4], [0.3, 0.4, 0.0]],
        dtype=torch.float32,
    )
    W_risk = torch.randn(3 * d, 1, dtype=torch.float32)

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
    landscape = Landscape(problem=problem, state=state)
    return problem, state, landscape


def test_enumerate_valid_assignments_is_complete():
    problem, _, _ = make_small_problem()
    assignments = list(enumerate_valid_assignments(problem))
    assert len(assignments) == problem.N ** problem.M
    for X in assignments:
        assert X.shape == (problem.N, problem.M)
        assert torch.all((X == 0) | (X == 1))
        assert torch.allclose(X.sum(dim=0), torch.ones(problem.M))


def test_compiled_ilp_objective_matches_landscape_for_all_valid_assignments():
    problem, _, landscape = make_small_problem()
    for X in enumerate_valid_assignments(problem):
        e_land = landscape.evaluate(X)
        e_ilp = compiled_ilp_objective(X, landscape)
        assert abs(e_land - e_ilp) < 1e-6, (e_land, e_ilp)


def test_compiled_model_uses_fixed_landscape_state():
    problem, state, landscape = make_small_problem()
    model = compile_landscape_to_ilp(landscape)

    assert model["problem"]["N"] == problem.N
    assert model["state"]["Theta"].shape == state.Theta.shape
    assert torch.allclose(model["state"]["Theta"], state.Theta)
    assert torch.allclose(model["state"]["kappa"], state.kappa)

    changed = landscape.state.clone()
    changed.kappa[0, 0] += 1.0
    assert not torch.allclose(model["state"]["kappa"], changed.kappa)


def test_sampler_supports_landscape_evaluation_adapter():
    _, _, landscape = make_small_problem()
    X = next(iter(enumerate_valid_assignments(landscape.problem)))
    proposal = AssignmentProposal(energy_registry=None, mode="random")
    sampler = SimulatedAnnealingSampler(
        proposal_mechanism=proposal,
        energy_registry=None,
        landscape=landscape,
    )

    assert abs(sampler.evaluate_assignment(X) - landscape.evaluate(X)) < 1e-6
