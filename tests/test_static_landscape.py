import copy
import random

import torch

from model.ebmao_orchestrator import EBMAOOrchestrator
from model.orchestrator import Orchestrator
from state.orchestration_state import OrchestrationState


def make_config():
    return {
        "model": {
            "num_agents": 3,
            "num_tasks": 5,
            "dim": 4,
            "lambda_align": 0.5,
            "lambda_memory": 0.5,
            "eta_theta": 0.1,
            "eta_memory": 0.05,
            "temperature_init": 2.0,
            "min_temperature": 0.1,
            "max_temperature": 5.0,
            "target_accept_rate": 0.3,
            "proposal_candidates": 3,
            "proposal_task_sample": 2,
            "agent_sample_size": 3,
            "block_move_size": 2,
            "warm_start_steps": 0,
            "local_refine_steps": 0,
            "risk_weight": 1.0,
            "risk_scale": 1.0,
            "interaction_weight": 1.0,
            "cost_weight": 1.0,
            "search_mode": "pure_sa",
            "theta_mode": "static",
        }
    }


def make_state():
    N, M, d = 3, 5, 4
    X = torch.zeros(N, M)
    for task_idx in range(M):
        X[task_idx % N, task_idx] = 1.0

    return OrchestrationState(
        X=X,
        s=torch.randn(N, d),
        c=torch.randn(M, d),
        kappa=torch.randn(N, d),
        Theta=torch.rand(M, M),
        C=torch.rand(M, M),
        N=N,
        M=M,
        d=d,
    )


def test_static_landscape_is_unchanged_in_core_orchestrators():
    for orchestrator_class in (Orchestrator, EBMAOOrchestrator):
        torch.manual_seed(0)
        random.seed(0)
        state = make_state()
        initial_kappa = state.kappa.clone()
        initial_theta = state.Theta.clone()

        orchestrator = orchestrator_class(
            make_config(),
            initial_state=state,
            W_risk=torch.zeros(3 * state.d, 1),
        )

        for _ in range(3):
            orchestrator.step()

        assert torch.equal(orchestrator.kappa, initial_kappa)
        assert torch.equal(orchestrator.Theta, initial_theta)


def test_kappa_only_updates_memory_without_theta():
    torch.manual_seed(1)
    random.seed(1)
    state = make_state()
    initial_theta = state.Theta.clone()

    config = make_config()
    config["model"]["theta_mode"] = "static"
    config["model"]["memory_mode"] = "dynamic"
    orchestrator = EBMAOOrchestrator(
        config,
        initial_state=state,
        W_risk=torch.zeros(3 * state.d, 1),
    )

    initial_kappa = orchestrator.kappa.clone()
    orchestrator._update_adaptive_state()

    assert torch.equal(orchestrator.Theta, initial_theta)
    assert not torch.equal(orchestrator.kappa, initial_kappa)


def test_theta_only_updates_theta_without_memory():
    torch.manual_seed(2)
    random.seed(2)
    state = make_state()
    initial_kappa = state.kappa.clone()

    config = copy.deepcopy(make_config())
    config["model"]["theta_mode"] = "dynamic"
    config["model"]["memory_mode"] = "static"
    orchestrator = EBMAOOrchestrator(
        config,
        initial_state=state,
        W_risk=torch.zeros(3 * state.d, 1),
    )

    orchestrator.theta_updater.running_co = torch.zeros_like(orchestrator.Theta)
    initial_theta = orchestrator.Theta.clone()
    orchestrator._update_adaptive_state()

    assert not torch.equal(orchestrator.Theta, initial_theta)
    assert torch.equal(orchestrator.kappa, initial_kappa)
