import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.orchestration_state import OrchestrationState
from energy.ebmao_assignment import EBMAOAssignmentEnergy
from energy.ebmao_interaction import EBMAOInteractionEnergy
from energy.ebmao_cost import EBMAOCostEnergy
from energy.ebmao_risk import EBMAORiskEnergy
from energy.risk import RiskPredictor, RiskEnergy
from dynamics.ebmao_memory_update import EBMAOMemoryUpdater
from model.ebmao_orchestrator import EBMAOOrchestrator
from orchestrator.ebmao_orchestrator import EBMAOOrchestrator as BenchmarkEBMAOOrchestrator
from orchestrator.base import Task, Agent


def test_ebmao_assignment_energy():
    N, M, d = 2, 3, 4
    s = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    c = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    kappa = torch.tensor([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]], dtype=torch.float32)
    Theta = torch.zeros(M, M)
    C = torch.zeros(M, M)
    X = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

    state = OrchestrationState(X=X, s=s, c=c, kappa=kappa, Theta=Theta, C=C, N=N, M=M, d=d)

    # Calculate expected assign energy
    # E_assign = \sum_{a,i} X_{a,i} ( ||s_a - c_i||^2 - \lambda s_a^\top \kappa_a )
    # For a=0, i=0: X_{0,0}=1. ||s_0 - c_0||^2 = ||[1,0,0,0] - [1,0,0,0]||^2 = 0.
    #              s_0^\top \kappa_0 = 1.0 * 0.5 = 0.5.
    #              Value = 0 - lambda * 0.5 = -0.25 (with lambda=0.5).
    # For a=0, i=2: X_{0,2}=1. ||s_0 - c_2||^2 = ||[1,0,0,0] - [0,0,1,0]||^2 = 2.
    #              s_0^\top \kappa_0 = 0.5.
    #              Value = 2 - lambda * 0.5 = 1.75 (with lambda=0.5).
    # For a=1, i=1: X_{1,1}=1. ||s_1 - c_1||^2 = ||[0,1,0,0] - [0,1,0,0]||^2 = 0.
    #              s_1^\top \kappa_1 = 1.0 * 0.5 = 0.5.
    #              Value = 0 - lambda * 0.5 = -0.25 (with lambda=0.5).
    # Total sum: -0.25 + 1.75 - 0.25 = 1.25.
    # Normalized by N * M (6): 1.25 / 6.0 = 0.20833333

    energy_term = EBMAOAssignmentEnergy(lambda_align=0.5, weight=1.0)
    result = energy_term.compute(state)
    assert abs(result.item() - 0.25 / 6.0) < 1e-5


def test_ebmao_interaction_energy():
    N, M, d = 2, 3, 4
    s = torch.randn(N, d)
    c = torch.randn(M, d)
    kappa = torch.randn(N, d)
    Theta = torch.tensor([[0.0, 1.5, 0.5],
                          [1.5, 0.0, 2.0],
                          [0.5, 2.0, 0.0]], dtype=torch.float32)
    C = torch.zeros(M, M)
    X = torch.tensor([[1.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    state = OrchestrationState(X=X, s=s, c=c, kappa=kappa, Theta=Theta, C=C, N=N, M=M, d=d)

    # co = X.T @ X = [[1, 1, 0],
    #                 [1, 1, 0],
    #                 [0, 0, 1]]
    # Upper triangle for i < j:
    # (i, j) = (0, 1): Theta_{0,1} = 1.5. co_{0,1} = 1.0. Prod = 1.5
    # (i, j) = (0, 2): Theta_{0,2} = 0.5. co_{0,2} = 0. Prod = 0
    # (i, j) = (1, 2): Theta_{1,2} = 2.0. co_{1,2} = 0. Prod = 0
    # Sum: 1.5.
    # Negative sum normalized: -1.5 / 6.0 = -0.25

    energy_term = EBMAOInteractionEnergy(weight=1.0)
    result = energy_term.compute(state)
    assert abs(result.item() - (-0.25)) < 1e-5


def test_ebmao_cost_energy():
    N, M, d = 2, 3, 4
    s = torch.randn(N, d)
    c = torch.randn(M, d)
    kappa = torch.randn(N, d)
    Theta = torch.zeros(M, M)
    C = torch.tensor([[0.0, 1.0, 2.0],
                      [1.0, 0.0, 3.0],
                      [2.0, 3.0, 0.0]], dtype=torch.float32)
    X = torch.tensor([[1.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    state = OrchestrationState(X=X, s=s, c=c, kappa=kappa, Theta=Theta, C=C, N=N, M=M, d=d)

    # co = X.T @ X = [[1, 1, 0],
    #                 [1, 1, 0],
    #                 [0, 0, 1]]
    # (state.C * co).sum():
    # C * co has elements (0,1) -> 1.0, (1,0) -> 1.0. Other elements are 0.
    # Sum: 2.0.
    # Normalized: 2.0 / 6.0 = 0.33333333

    energy_term = EBMAOCostEnergy(weight=1.0)
    result = energy_term.compute(state)
    assert abs(result.item() - 2.0 / 6.0) < 1e-5


def test_ebmao_memory_updater():
    class DummyPredictor:
        def predict(self, state):
            # Return p_a,i such that:
            # p_0,0 = 0.8, p_0,1 = 0.6
            return torch.tensor([[0.8, 0.6], [0.0, 0.0]], dtype=torch.float32)

    N, M, d = 2, 2, 3
    s = torch.randn(N, d)
    c = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    kappa = torch.zeros(N, d)
    Theta = torch.zeros(M, M)
    C = torch.zeros(M, M)
    X = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    state = OrchestrationState(X=X, s=s, c=c, kappa=kappa, Theta=Theta, C=C, N=N, M=M, d=d)

    # Tasks assigned to agent 0: 0, 1
    # Context embeddings: [1,0,0], [0,1,0]
    # Success probabilities: 0.8, 0.6
    # p_{0, i} * c_i:
    # task 0: 0.8 * [1,0,0] = [0.8, 0, 0]
    # task 1: 0.6 * [0,1,0] = [0, 0.6, 0]
    # Mean weighted update: [0.4, 0.3, 0]
    # kappa_0 updated with decay: (1 - 0.1) * [0,0,0] + 0.1 * [0.4, 0.3, 0] = [0.04, 0.03, 0]

    updater = EBMAOMemoryUpdater(eta_memory=0.1)
    updater.apply(state, DummyPredictor())

    assert torch.allclose(state.kappa[0], torch.tensor([0.04, 0.03, 0.0], dtype=torch.float32))
    assert torch.allclose(state.kappa[1], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))


def test_ebmao_model_orchestrator_compiles_and_steps():
    cfg = {
        "model": {
            "num_agents": 3,
            "num_tasks": 5,
            "dim": 8,
            "lambda_align": 0.5,
            "eta_theta": 0.1,
            "eta_memory": 0.05,
            "temperature_init": 4.0,
            "min_temperature": 1.0,
            "max_temperature": 6.0,
            "proposal_candidates": 4,
            "proposal_task_sample": 2,
            "agent_sample_size": 2,
            "block_move_size": 2,
            "warm_start_steps": 2,
            "warm_start_type": "greedy",
            "hybrid_cleanup_prob": 0.25,
            "local_refine_steps": 1,
            "target_accept_rate": 0.3,
            "risk_weight": 1.0,
            "risk_scale": 1.0,
            "interaction_weight": 1.0,
            "cost_weight": 1.0,
        }
    }

    orchestrator = EBMAOOrchestrator(cfg)
    assert orchestrator.state is not None

    # Step the orchestrator
    init_energy = orchestrator.total_energy().item()
    orchestrator.step()
    final_energy = orchestrator.total_energy().item()

    # Make sure we didn't crash and energy can be computed
    assert isinstance(final_energy, float)


def test_benchmark_ebmao_orchestrator():
    cfg = {
        "model": {
            "num_agents": 2,
            "num_tasks": 2,
            "dim": 6,
            "lambda_align": 0.5,
            "eta_theta": 0.1,
            "eta_memory": 0.05,
            "temperature_init": 3.0,
            "min_temperature": 1.0,
            "max_temperature": 5.0,
            "proposal_candidates": 2,
            "proposal_task_sample": 2,
            "agent_sample_size": 2,
            "block_move_size": 2,
            "warm_start_steps": 0,
            "warm_start_type": "greedy",
            "hybrid_cleanup_prob": 0.0,
            "local_refine_steps": 0,
            "target_accept_rate": 0.3,
            "risk_weight": 1.0,
            "risk_scale": 1.0,
            "interaction_weight": 1.0,
            "cost_weight": 1.0,
        }
    }

    orchestrator = BenchmarkEBMAOOrchestrator(cfg)
    tasks = [
        Task(id="t1", embedding=torch.randn(6), dependencies=[], estimated_cost=1.0, estimated_risk=0.1),
        Task(id="t2", embedding=torch.randn(6), dependencies=["t1"], estimated_cost=1.2, estimated_risk=0.2),
    ]
    agents = [
        {"id": "planner", "role": "Planner", "capability_embedding": torch.randn(6), "memory_state": {}},
        {"id": "sql", "role": "SQL Expert", "capability_embedding": torch.randn(6), "memory_state": {}},
    ]

    assignment = orchestrator.solve(tasks, agents)
    assert assignment is not None
    assert len(assignment) == len(tasks)
    assert all(task_id in assignment for task_id in [t.id for t in tasks])
    assert hasattr(orchestrator, "state")
    assert orchestrator.state.X.shape == (2, 2)
