import torch

from state.orchestration_state import OrchestrationState
from dynamics.ebmao_memory_update import EBMAOMemoryUpdater


class DummyRiskPredictor:
    def predict(self, state):
        # deterministische Erfolgswahrscheinlichkeiten
        return torch.full(
            (state.N, state.M),
            0.5,
            dtype=state.s.dtype,
        )


def test_memory_update():
    N, M, d = 2, 3, 2

    X = torch.tensor([
        [1., 1., 0.],
        [0., 0., 1.],
    ])

    c = torch.tensor([
        [1., 0.],
        [0., 1.],
        [2., 2.],
    ])

    state = OrchestrationState(
        X=X,
        s=torch.zeros(N, d),
        c=c,
        kappa=torch.zeros(N, d),
        Theta=torch.zeros(M, M),
        C=torch.zeros(M, M),
        N=N,
        M=M,
        d=d,
    )

    updater = EBMAOMemoryUpdater(eta_memory=1.0)

    updater.apply(state, DummyRiskPredictor())

    # Agent 0 hat Task 0 und 1:
    # mean([1,0], [0,1]) * 0.5 = [0.25, 0.25]
    assert torch.allclose(
        state.kappa[0],
        torch.tensor([0.25, 0.25]),
    )

    # Agent 1 hat Task 2:
    # [2,2] * 0.5
    assert torch.allclose(
        state.kappa[1],
        torch.tensor([1., 1.]),
    )