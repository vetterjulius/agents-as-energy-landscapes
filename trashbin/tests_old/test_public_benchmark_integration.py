import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.agentbench.adapter import BenchmarkTask
from orchestrator.task_decomposer import Task, TaskDecomposer
from orchestrator.base import BaseOrchestrator
from baselines.random_baseline import RandomOrchestrator
from orchestrator.energy_orchestrator import EnergyOrchestrator
from orchestrator.benchmark_runner import PublicBenchmarkRunner


def test_task_decomposer_and_adapter_shapes():
    task = BenchmarkTask(
        id="task-1",
        instruction="Generate SQL query",
        ground_truth="SELECT * FROM users",
    )
    decomposer = TaskDecomposer()
    tasks = decomposer.decompose(task)

    assert len(tasks) >= 2
    assert all(isinstance(item, Task) for item in tasks)
    assert tasks[0].id.startswith("task-1")
    assert tasks[0].embedding.shape[0] > 0


def test_orchestrator_interface_works_with_baseline():
    cfg = {
        "model": {
            "num_agents": 3,
            "num_tasks": 2,
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

    orchestrator = RandomOrchestrator(cfg)
    tasks = [
        Task(id="t1", embedding=torch.randn(8), dependencies=[], estimated_cost=1.0, estimated_risk=0.1),
        Task(id="t2", embedding=torch.randn(8), dependencies=[], estimated_cost=1.0, estimated_risk=0.2),
    ]
    agents = [
        {"id": "planner", "role": "Planner", "capability_embedding": torch.randn(8), "memory_state": {}},
        {"id": "sql", "role": "SQL Expert", "capability_embedding": torch.randn(8), "memory_state": {}},
        {"id": "verifier", "role": "Verifier", "capability_embedding": torch.randn(8), "memory_state": {}},
    ]

    assignment = orchestrator.solve(tasks, agents)
    assert assignment is not None
    assert len(assignment) == len(tasks)
    assert all(task_id in assignment for task_id in [t.id for t in tasks])


def test_energy_orchestrator_builds_state_and_runs_benchmark_suite():
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

    orchestrator = EnergyOrchestrator(cfg)
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


def test_public_benchmark_runner_supports_multiple_orchestrators_and_export(tmp_path):
    runner = PublicBenchmarkRunner(adapter=None)
    results = runner.run(orchestrator_names=["random", "energy"], export_dir=str(tmp_path))

    assert len(results) >= 1
    assert any(item["metrics"]["orchestrator_name"] == "energy" for item in results)
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark_results.csv").exists()

    payload = json.loads((tmp_path / "benchmark_results.json").read_text(encoding="utf-8"))
    assert isinstance(payload, list)
