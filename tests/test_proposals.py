import torch

from state.orchestration_state import OrchestrationState
from energy.registry import EnergyRegistry
from energy.ebmao_assignment import EBMAOAssignmentEnergy
from dynamics.ebmao_proposal import EBMAOAssignmentProposal


def make_state(N=4, M=6, d=3):
    torch.manual_seed(0)

    X = torch.zeros(N, M)
    for t in range(M):
        X[t % N, t] = 1.0

    return OrchestrationState(
        X=X,
        s=torch.randn(N, d),
        c=torch.randn(M, d),
        kappa=torch.randn(N, d),
        Theta=torch.zeros(M, M),
        C=torch.zeros(M, M),
        N=N,
        M=M,
        d=d,
    )


def test_proposal_preserves_one_assignment_per_task():
    state = make_state()

    registry = EnergyRegistry()
    registry.add(EBMAOAssignmentEnergy(
        lambda_align=0.5,
        lambda_memory=0.5,
    ))

    proposal = EBMAOAssignmentProposal(
        registry,
        lambda_align=0.5,
        lambda_memory=0.5,
        num_tasks=3,
        block_size=2,
    )

    X_old = state.X.clone()
    X_new = proposal.propose(state)

    # Shape bleibt gleich
    assert X_new.shape == X_old.shape

    # Jede Aufgabe hat weiterhin genau einen Agenten
    assert torch.allclose(
        X_new.sum(dim=0),
        torch.ones(state.M),
    )

    # Nur 0/1-Zuweisungen
    assert torch.all((X_new == 0) | (X_new == 1))


def test_proposal_does_not_modify_state():
    state = make_state()

    registry = EnergyRegistry()
    registry.add(EBMAOAssignmentEnergy(
        lambda_align=0.5,
        lambda_memory=0.5,
    ))

    proposal = EBMAOAssignmentProposal(registry)

    X_old = state.X.clone()

    proposal.propose(state)

    assert torch.equal(state.X, X_old)