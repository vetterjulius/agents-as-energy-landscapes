import torch

from state.orchestration_state import OrchestrationState
from energy.registry import EnergyRegistry
from energy.ebmao_assignment import EBMAOAssignmentEnergy
from dynamics.ebmao_proposal import EBMAOAssignmentProposal
from dynamics.sampler import SimulatedAnnealingSampler


def make_state():
    N, M, d = 3, 5, 2

    X = torch.zeros(N, M)
    for t in range(M):
        X[t % N, t] = 1.

    return OrchestrationState(
        X=X,
        s=torch.randn(N, d),
        c=torch.randn(M, d),
        kappa=torch.zeros(N, d),
        Theta=torch.zeros(M, M),
        C=torch.zeros(M, M),
        N=N,
        M=M,
        d=d,
    )


def test_sampler_preserves_valid_assignment():
    torch.manual_seed(0)

    state = make_state()

    registry = EnergyRegistry()
    registry.add(
        EBMAOAssignmentEnergy(
            lambda_align=0.5,
            lambda_memory=0.5,
        )
    )

    proposal = EBMAOAssignmentProposal(registry)

    sampler = SimulatedAnnealingSampler(
        proposal,
        registry,
        T_init=2.0,
        num_candidates=5,
    )

    sampler.step(state)

    assert torch.allclose(
        state.X.sum(dim=0),
        torch.ones(state.M),
    )