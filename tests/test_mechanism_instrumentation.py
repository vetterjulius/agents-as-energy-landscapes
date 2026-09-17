from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.runner import ControlledBenchmarkRunner, _observed_cooccurrence
from controlled_benchmark.scenarios import (
    generate_scenario_trajectory,
    make_initial_landscape_state,
    problem_instance_to_problem_context,
)


def test_episode_mechanism_diagnostics_have_expected_shapes_and_are_finite():
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42], scenarios=["Stationary"], num_episodes=4, perturb_episode=2,
        verbose_diagnostics=True,
    )
    runner = ControlledBenchmarkRunner(cfg)
    trajectory = generate_scenario_trajectory(
        "Stationary", 42, cfg.num_episodes, cfg.perturb_episode,
        cfg.num_agents, cfg.num_tasks, cfg.dim
    )

    runner.run_trajectory(
        "Stationary", 42, "Simulated Annealing", "Theta-only", "theta-only", trajectory
    )
    diagnostics = runner.last_mechanism_diagnostics

    assert len(diagnostics) == cfg.num_episodes
    for episode, item in enumerate(diagnostics):
        assert item["episode"] == episode
        assert item["assignment_matrix"].shape == (cfg.num_agents, cfg.num_tasks)
        assert item["cooccurrence_matrix"].shape == (cfg.num_tasks, cfg.num_tasks)
        assert item["theta_before"].shape == (cfg.num_tasks, cfg.num_tasks)
        assert item["theta_after"].shape == (cfg.num_tasks, cfg.num_tasks)
        assert item["ground_truth_dependency"].shape == (cfg.num_tasks, cfg.num_tasks)
        for value in item.values():
            if isinstance(value, np.ndarray):
                assert np.isfinite(value).all()


def test_saved_cooccurrence_is_exact_implementation_from_saved_assignment():
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42], scenarios=["Stationary"], num_episodes=3, perturb_episode=1,
        verbose_diagnostics=True,
    )
    runner = ControlledBenchmarkRunner(cfg)
    trajectory = generate_scenario_trajectory(
        "Stationary", 42, cfg.num_episodes, cfg.perturb_episode,
        cfg.num_agents, cfg.num_tasks, cfg.dim
    )
    runner.run_trajectory(
        "Stationary", 42, "Energy Greedy", "Theta-only", "theta-only", trajectory
    )

    for item in runner.last_mechanism_diagnostics:
        X = torch.from_numpy(item["assignment_matrix"])
        expected = _observed_cooccurrence(X).numpy()
        np.testing.assert_array_equal(item["cooccurrence_matrix"], expected)


def test_theta_after_is_actual_boundary_update_and_next_episode_theta_before():
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42], scenarios=["Stationary"], num_episodes=4, perturb_episode=2,
        verbose_diagnostics=True,
    )
    runner = ControlledBenchmarkRunner(cfg)
    trajectory = generate_scenario_trajectory(
        "Stationary", 42, cfg.num_episodes, cfg.perturb_episode,
        cfg.num_agents, cfg.num_tasks, cfg.dim
    )
    runner.run_trajectory(
        "Stationary", 42, "Energy Greedy", "Theta-only", "theta-only", trajectory
    )
    diagnostics = runner.last_mechanism_diagnostics
    manager = EpisodeAdaptationManager("theta-only", eta_theta=cfg.eta_theta)
    state = make_initial_landscape_state(trajectory[0])

    for episode, item in enumerate(diagnostics):
        np.testing.assert_array_equal(item["theta_before"], state.Theta.numpy())
        X = torch.from_numpy(item["assignment_matrix"])
        context = problem_instance_to_problem_context(
            trajectory[episode],
            lambda_align=cfg.lambda_align,
            lambda_memory=cfg.lambda_memory,
            interaction_weight=cfg.interaction_weight,
            cost_weight=cfg.cost_weight,
            risk_weight=cfg.risk_weight,
        )
        state = manager.step(state, X, context)
        np.testing.assert_array_equal(item["theta_after"], state.Theta.numpy())
        if episode + 1 < len(diagnostics):
            np.testing.assert_array_equal(
                diagnostics[episode + 1]["theta_before"], item["theta_after"]
            )


def test_verbose_trace_does_not_change_solver_results():
    trajectory = generate_scenario_trajectory(
        "Capability Drift", 42, 4, 2, 3, 6, 4
    )
    plain = ControlledBenchmarkRunner(
        BenchmarkConfig.quick_mode(
            seeds=[42], scenarios=["Capability Drift"], num_episodes=4,
            perturb_episode=2, max_energy_evaluations=20,
            verbose_diagnostics=False,
        )
    )
    verbose = ControlledBenchmarkRunner(
        BenchmarkConfig.quick_mode(
            seeds=[42], scenarios=["Capability Drift"], num_episodes=4,
            perturb_episode=2, max_energy_evaluations=20,
            verbose_diagnostics=True,
        )
    )
    _, plain_records = plain.run_trajectory(
        "Capability Drift", 42, "Simulated Annealing", "Full", "full", trajectory
    )
    _, verbose_records = verbose.run_trajectory(
        "Capability Drift", 42, "Simulated Annealing", "Full", "full", trajectory
    )
    assert [record.external_energy for record in plain_records] == pytest.approx(
        [record.external_energy for record in verbose_records]
    )
    assert [record.internal_energy for record in plain_records] == pytest.approx(
        [record.internal_energy for record in verbose_records]
    )
    assert len(verbose.last_mechanism_diagnostics[0]["solver_trace"]) > 0


def test_large_verbose_artifact_uses_incremental_chunks(tmp_path):
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42], scenarios=["Stationary"], num_episodes=1, perturb_episode=0,
        num_agents=1, num_tasks=6, max_energy_evaluations=100000,
        output_dir=str(tmp_path), verbose_diagnostics=True,
    )
    result = ControlledBenchmarkRunner(cfg).run_benchmark()

    chunk_dir = tmp_path / "mechanism_diagnostics"
    manifest = json.loads((tmp_path / "mechanism_diagnostics.json").read_text(encoding="utf-8"))
    assert result["integrity"]["all_checks_pass"] is True
    assert manifest["format_version"] == "2.0-chunked"
    assert manifest["completed_runs"] == 8
    assert len(list(chunk_dir.glob("chunk_*.npz"))) == 8
    assert not (tmp_path / "mechanism_diagnostics.npz").exists()
    with np.load(chunk_dir / "chunk_000000.npz") as artifact:
        assert artifact["assignment_matrix"].shape == (1, 1, 6)
        assert artifact["cooccurrence_matrix"].shape == (1, 6, 6)
        assert np.isfinite(artifact["assignment_matrix"]).all()


def test_mechanism_diagnostic_artifact_has_manifest_and_run_episode_mapping(tmp_path):
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42], scenarios=["Dependency Change"], num_episodes=3,
        perturb_episode=1, max_energy_evaluations=20, output_dir=str(tmp_path),
        verbose_diagnostics=True,
    )
    result = ControlledBenchmarkRunner(cfg).run_benchmark()

    tensor_path = tmp_path / "mechanism_diagnostics.npz"
    manifest_path = tmp_path / "mechanism_diagnostics.json"
    assert tensor_path.exists()
    assert manifest_path.exists()

    with np.load(tensor_path) as artifact:
        assert artifact["assignment_matrix"].shape == (8, 3, 3, 6)
        assert artifact["cooccurrence_matrix"].shape == (8, 3, 6, 6)
        assert artifact["theta_before"].shape == (8, 3, 6, 6)
        assert artifact["theta_after"].shape == (8, 3, 6, 6)
        assert artifact["ground_truth_dependency"].shape == (8, 3, 6, 6)
        assert artifact["initial_assignment"].shape == (8, 3, 3, 6)
        assert artifact["agent_capabilities"].shape == (8, 3, 3, 4)
        assert artifact["task_embeddings"].shape == (8, 3, 6, 4)
        assert artifact["co_assignment_costs"].shape == (8, 3, 6, 6)
        assert artifact["risk_weights"].shape == (8, 3, 12, 1)
        assert artifact["kappa_before"].shape == (8, 3, 3, 4)
        assert artifact["kappa_after"].shape == (8, 3, 3, 4)
        assert artifact["adaptation_risk_probabilities"].shape == (8, 3, 3, 6)
        assert artifact["adaptation_kappa_target"].shape == (8, 3, 3, 4)
        assert artifact["adaptation_theta_observation"].shape == (8, 3, 6, 6)
        assert "trace_assignment_matrix" in artifact.files
        assert artifact["trace_assignment_matrix"].shape[1:] == (3, 6)
        assert artifact["trace_assignment_matrix"].shape[0] > 0
        assert "internal_energy_AssignmentEnergy" in artifact.files
        assert "external_energy_RiskEnergy" in artifact.files
        assert np.array_equal(artifact["episode"], np.tile(np.arange(3), (8, 1)))
        for name in artifact.files:
            assert np.isfinite(artifact[name]).all()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == "1.0"
    assert manifest["verbose_diagnostics"] is True
    assert "kappa_after" in manifest["verbose_tensor_fields"]
    assert "AssignmentEnergy" in manifest["energy_breakdown_fields"]
    assert len(manifest["episode_metadata"]) == 24
    assert all("run_index" in item and "episode" in item for item in manifest["episode_metadata"])
    assert manifest["solver_trace_event_count"] > 0
    assert len(manifest["solver_trace_events"]) == manifest["solver_trace_event_count"]
    assert all("run_index" in item and "episode" in item for item in manifest["solver_trace_events"])
    assert "t+1" in manifest["theta_after"]
    assert "episode t" in manifest["timing"]
    assert len(manifest["records"]) == 8
    assert result["integrity"]["all_checks_pass"] is True
