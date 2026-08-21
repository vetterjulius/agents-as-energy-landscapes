import torch

from state.orchestration_state import OrchestrationState
from dynamics.theta_update import ThetaUpdater


def make_state():
    N, M, d = 2, 3, 2

    X = torch.tensor([
        [1., 1., 0.],
        [0., 0., 1.],
    ])

    return OrchestrationState(
        X=X,
        s=torch.zeros(N, d),
        c=torch.zeros(M, d),
        kappa=torch.zeros(N, d),
        Theta=torch.zeros(M, M),
        C=torch.zeros(M, M),
        N=N,
        M=M,
        d=d,
    )


def test_theta_update_initializes_running_co():
    state = make_state()

    updater = ThetaUpdater(eta_theta=0.5)
    updater.apply(state)

    assert updater.running_co is not None
    assert updater.running_co.shape == (state.M, state.M)


def test_theta_update_preserves_shape():
    state = make_state()

    updater = ThetaUpdater(eta_theta=0.5)
    updater.apply(state)

    assert state.Theta.shape == (state.M, state.M)