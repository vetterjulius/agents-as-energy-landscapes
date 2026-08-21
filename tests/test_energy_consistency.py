import torch

from state.orchestration_state import OrchestrationState

from energy.assignment import AssignmentEnergy
from energy.interaction import InteractionEnergy
from energy.cost import CostEnergy
from energy.risk import RiskEnergy, RiskPredictor

from energy.ebmao_assignment import EBMAOAssignmentEnergy
from energy.ebmao_interaction import EBMAOInteractionEnergy
from energy.ebmao_cost import EBMAOCostEnergy
from energy.ebmao_risk import EBMAORiskEnergy


def make_test_state():
    torch.manual_seed(42)

    N = 3
    M = 5
    d = 4

    s = torch.randn(N, d)
    c = torch.randn(M, d)
    kappa = torch.randn(N, d)

    # Jede Aufgabe genau einem Agenten zugewiesen
    X = torch.zeros(N, M)
    assignments = [0, 1, 1, 2, 0]

    for task, agent in enumerate(assignments):
        X[agent, task] = 1.0

    # Symmetrische Theta-Matrix
    Theta = torch.randn(M, M)
    Theta = (Theta + Theta.T) / 2
    Theta.fill_diagonal_(0)

    # Symmetrische Kostenmatrix
    C = torch.rand(M, M)
    C = (C + C.T) / 2
    C.fill_diagonal_(0)

    W_risk = torch.randn(3 * d, 1)

    state = OrchestrationState(
        X=X,
        s=s,
        c=c,
        kappa=kappa,
        Theta=Theta,
        C=C,
        N=N,
        M=M,
        d=d,
    )

    return state, W_risk


def test_assignment_energy_consistency():
    state, _ = make_test_state()

    normal = AssignmentEnergy(
        lambda_align=0.5,
        lambda_memory=0.3,
    )

    ebmao = EBMAOAssignmentEnergy(
        lambda_align=0.5,
        lambda_memory=0.3,
    )

    e_normal = normal.compute(state)
    e_ebmao = ebmao.compute(state)

    print("\nAssignment:")
    print("normal:", e_normal.item())
    print("EBMAO: ", e_ebmao.item())

    assert torch.allclose(
        e_normal,
        e_ebmao,
        atol=1e-6,
    )


def test_interaction_energy_consistency():
    state, _ = make_test_state()

    normal = InteractionEnergy()
    ebmao = EBMAOInteractionEnergy()

    e_normal = normal.compute(state)
    e_ebmao = ebmao.compute(state)

    print("\nInteraction:")
    print("normal:", e_normal.item())
    print("EBMAO: ", e_ebmao.item())

    assert torch.allclose(
        e_normal,
        e_ebmao,
        atol=1e-6,
    )


def test_cost_energy_consistency():
    state, _ = make_test_state()

    normal = CostEnergy()
    ebmao = EBMAOCostEnergy()

    e_normal = normal.compute(state)
    e_ebmao = ebmao.compute(state)

    print("\nCost:")
    print("normal:", e_normal.item())
    print("EBMAO: ", e_ebmao.item())

    assert torch.allclose(
        e_normal,
        e_ebmao,
        atol=1e-6,
    )


def test_risk_energy_consistency():
    state, W_risk = make_test_state()

    predictor_1 = RiskPredictor(
        state.d,
        W_risk=W_risk,
    )

    predictor_2 = RiskPredictor(
        state.d,
        W_risk=W_risk,
    )

    normal = RiskEnergy(predictor_1)
    ebmao = EBMAORiskEnergy(predictor_2)

    e_normal = normal.compute(state)
    e_ebmao = ebmao.compute(state)

    print("\nRisk:")
    print("normal:", e_normal.item())
    print("EBMAO: ", e_ebmao.item())

    assert torch.allclose(
        e_normal,
        e_ebmao,
        atol=1e-6,
    )