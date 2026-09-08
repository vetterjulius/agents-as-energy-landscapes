import torch

from dynamics.memory_update import MemoryUpdater
from dynamics.proposal import AssignmentProposal
from dynamics.sampler import SimulatedAnnealingSampler
from dynamics.theta_update import ThetaUpdater
from landscape import Landscape, LandscapeState, ProblemContext
from model.orchestrator import Orchestrator
from state.orchestration_state import OrchestrationState


def make_fixture():
    torch.manual_seed(21)
    N, M, d = 3, 4, 2
    X = torch.zeros(N, M)
    for task in range(M):
        X[task % N, task] = 1.0
    problem = ProblemContext(
        s=torch.randn(N, d),
        c=torch.randn(M, d),
        C=torch.rand(M, M),
        W_risk=torch.randn(3 * d, 1),
        N=N,
        M=M,
        d=d,
    )
    theta = torch.randn(M, M)
    theta = (theta + theta.T) / 2.0
    theta.fill_diagonal_(0.0)
    landscape = Landscape(
        problem=problem,
        state=LandscapeState(kappa=torch.randn(N, d), Theta=theta),
    )
    state = OrchestrationState(
        X=X,
        s=problem.s.clone(),
        c=problem.c.clone(),
        kappa=landscape.state.kappa.clone(),
        Theta=landscape.state.Theta.clone(),
        C=problem.C.clone(),
        N=N,
        M=M,
        d=d,
    )
    return problem, landscape, state


def make_config(search_mode="pure_sa", theta_mode="static", memory_mode="static"):
    return {
        "model": {
            "num_agents": 3,
            "num_tasks": 4,
            "dim": 2,
            "lambda_align": 0.5,
            "lambda_memory": 0.5,
            "eta_theta": 0.1,
            "eta_memory": 0.05,
            "temperature_init": 2.0,
            "min_temperature": 0.1,
            "max_temperature": 5.0,
            "target_accept_rate": 0.3,
            "proposal_candidates": 2,
            "proposal_task_sample": 2,
            "agent_sample_size": 3,
            "block_move_size": 2,
            "warm_start_steps": 0,
            "local_refine_steps": 0,
            "risk_weight": 1.0,
            "risk_scale": 1.0,
            "interaction_weight": 1.0,
            "cost_weight": 1.0,
            "search_mode": search_mode,
            "theta_mode": theta_mode,
            "memory_mode": memory_mode,
        }
    }


def test_identical_adaptation_runs_and_energy_are_reproducible():
    _, landscape, state = make_fixture()
    first = state.clone()
    second = state.clone()
    theta_first = ThetaUpdater(0.1)
    theta_second = ThetaUpdater(0.1)
    risk_first = landscape.problem.W_risk.clone()
    risk_second = landscape.problem.W_risk.clone()
    from energy.risk import RiskPredictor

    memory_first = MemoryUpdater(0.05)
    memory_second = MemoryUpdater(0.05)
    X = state.X.clone()
    trajectory_first = []
    trajectory_second = []
    for _ in range(4):
        theta_first.apply(first)
        memory_first.apply(first, RiskPredictor(state.d, W_risk=risk_first))
        theta_second.apply(second)
        memory_second.apply(second, RiskPredictor(state.d, W_risk=risk_second))
        trajectory_first.append((first.kappa.clone(), first.Theta.clone()))
        trajectory_second.append((second.kappa.clone(), second.Theta.clone()))

    for (kappa_a, theta_a), (kappa_b, theta_b) in zip(trajectory_first, trajectory_second):
        assert torch.allclose(kappa_a, kappa_b, atol=1e-7, rtol=1e-7)
        assert torch.allclose(theta_a, theta_b, atol=1e-7, rtol=1e-7)
    assert torch.allclose(first.kappa, second.kappa, atol=1e-7, rtol=1e-7)
    assert torch.allclose(first.Theta, second.Theta, atol=1e-7, rtol=1e-7)
    first_landscape = Landscape(landscape.problem, LandscapeState(first.kappa, first.Theta))
    second_landscape = Landscape(landscape.problem, LandscapeState(second.kappa, second.Theta))
    assert abs(first_landscape.evaluate(X) - second_landscape.evaluate(X)) < 1e-7


def test_solver_state_isolation_and_temperature_isolation():
    _, landscape, state = make_fixture()
    before_kappa = landscape.state.kappa.clone()
    before_theta = landscape.state.Theta.clone()
    cfg = make_config()
    solver_a = Orchestrator(cfg, initial_state=state, landscape=landscape)
    solver_b = Orchestrator(cfg, initial_state=state, landscape=landscape)
    solver_a.step()
    solver_b.step()
    assert torch.equal(landscape.state.kappa, before_kappa)
    assert torch.equal(landscape.state.Theta, before_theta)

    X = state.X.clone()
    proposal = AssignmentProposal(energy_registry=None, mode="random")
    sampler_a = SimulatedAnnealingSampler(proposal, landscape=landscape, T_init=0.1)
    sampler_b = SimulatedAnnealingSampler(proposal, landscape=landscape, T_init=10.0)
    assert sampler_a.evaluate_assignment(X) == sampler_b.evaluate_assignment(X)


def test_adaptation_boundary_and_static_mode():
    _, landscape, state = make_fixture()
    before_kappa = landscape.state.kappa.clone()
    before_theta = landscape.state.Theta.clone()
    fixed_X = state.X.clone()
    energy = landscape.evaluate(fixed_X)
    assert torch.equal(landscape.state.kappa, before_kappa)
    assert torch.equal(landscape.state.Theta, before_theta)

    solver = Orchestrator(make_config(), initial_state=state, landscape=landscape)
    solver.step()
    assert torch.equal(landscape.state.kappa, before_kappa)
    assert torch.equal(landscape.state.Theta, before_theta)
    assert landscape.evaluate(fixed_X) == energy

    update_state = state.clone()
    ThetaUpdater(0.1).apply(update_state)
    MemoryUpdater(0.05).apply(update_state, solver.risk_predictor)
    assert not torch.equal(update_state.kappa, before_kappa) or not torch.equal(update_state.Theta, before_theta)

    static_solver = Orchestrator(make_config(search_mode="pure_sa"), initial_state=state, landscape=landscape)
    static_energies = []
    for _ in range(3):
        static_energies.append(landscape.evaluate(fixed_X))
        static_solver.step()
    assert torch.equal(landscape.state.kappa, before_kappa)
    assert torch.equal(landscape.state.Theta, before_theta)
    assert static_energies == [energy, energy, energy]