from __future__ import annotations

import json

import pytest

from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.metrics import EpisodeRecord, compute_ppee, compute_trajectory_summary
from controlled_benchmark.runner import ControlledBenchmarkRunner


def make_records(internal, external):
    return [
        EpisodeRecord(
            episode=i,
            internal_energy=float(internal[i]),
            external_energy=float(external[i]),
            energy_evaluations=1,
            accepted_moves=0,
            iterations=1,
            runtime_sec=0.001,
            termination_reason="completed",
            solver_status="completed",
            is_optimal=None,
            mip_gap=None,
            timeout=False,
            fallback_used=False,
            reconfig_cost=0.0,
            constraint_violations=0.0,
            coordination_score=0.0,
            load_balance=0.0,
            kappa_norm=0.0,
            theta_diff_norm=0.0,
        )
        for i in range(len(external))
    ]


def test_ppee_uses_nonnegative_external_excess_and_exact_window():
    records = make_records([100, 100, 100, 100, 100], [10, 11, 12, 9, 13])
    assert compute_ppee(records, perturb_episode=0, post_ppee_window=5, reference_energy=10.0) == pytest.approx(1.2)


def test_ppee_ignores_internal_energy():
    records = make_records([100, 100, 100], [10, 11, 12])
    assert compute_ppee(records, perturb_episode=0, post_ppee_window=3, reference_energy=10.0) == pytest.approx(1.0)


def test_summary_uses_exact_constant_reference_for_ppee_and_cee():
    external = [0.0] * 25 + [11.0] * 10 + [12.0] * 15
    records = make_records([999.0] * 50, external)
    summary = compute_trajectory_summary(
        records,
        scenario_id="Stationary",
        perturb_episode=25,
        post_ppee_window=10,
        reference_energy=10.0,
        reference_method="exact_optimal",
    )
    assert summary.ppee_10 == pytest.approx(1.0)
    assert summary.cumulative_excess_energy == pytest.approx(10.0 + 15.0 * 2.0)
    assert summary.reference_energy == pytest.approx(10.0)
    assert summary.reference_method == "exact_optimal"


def test_research_configuration_is_central_and_reproducible():
    cfg = BenchmarkConfig.research_mode()
    assert cfg.seeds == list(range(42, 62))
    assert (cfg.num_agents, cfg.num_tasks, cfg.dim) == (5, 10, 8)
    assert (cfg.num_episodes, cfg.perturb_episode) == (50, 25)
    assert (cfg.max_energy_evaluations, cfg.post_ppee_window) == (500, 10)
    assert cfg.run_ilp_dynamic is False
    assert cfg.experiment_id == "controlled_benchmark_paired"


def test_pilot_integrity_and_metadata(tmp_path):
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42],
        scenarios=["Stationary"],
        num_episodes=4,
        perturb_episode=2,
        max_energy_evaluations=20,
        output_dir=str(tmp_path),
    )
    result = ControlledBenchmarkRunner(cfg).run_benchmark()
    assert len(result["runs_df"]) == 5
    assert result["integrity"]["all_checks_pass"] is True
    assert result["integrity"]["duplicate_run_keys"] == 0
    assert result["integrity"]["runs_with_missing_or_duplicate_episodes"] == []
    assert result["integrity"]["ppee_window"] == [2, 3]

    with open(tmp_path / "metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["experiment_id"] == cfg.experiment_id
    assert metadata["git_commit"]
    assert "git_dirty" in metadata
    assert metadata["post_ppee_window"] == 10

    required_episode_columns = {
        "seed", "scenario_id", "solver_id", "adaptation_mode", "episode",
        "reference_method", "reference_energy", "internal_energy", "external_energy",
        "energy_evaluations", "accepted_moves", "iterations", "runtime_sec",
        "termination_reason", "solver_status", "optimal", "mip_gap", "timeout",
        "fallback", "kappa_norm", "theta_norm", "delta_kappa_norm", "delta_theta_norm",
    }
    assert required_episode_columns <= set(result["episodes_df"].columns)


def test_shared_trajectory_and_reference_ids_in_runner(tmp_path):
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42],
        scenarios=["Dependency Change"],
        num_episodes=4,
        perturb_episode=2,
        max_energy_evaluations=20,
        output_dir=str(tmp_path),
    )
    result = ControlledBenchmarkRunner(cfg).run_benchmark()
    runs = result["runs_df"]
    assert runs["trajectory_id"].nunique() == 1
    assert runs["reference_id"].nunique() == 1
    assert runs["reference_energy"].nunique() == 1
    assert (runs["reference_method"] == "exact_optimal").all()


def test_no_leakage_boundary_is_reflected_in_episode_state_deltas(tmp_path):
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42],
        scenarios=["Task Shift"],
        num_episodes=4,
        perturb_episode=2,
        max_energy_evaluations=20,
        output_dir=str(tmp_path),
    )
    result = ControlledBenchmarkRunner(cfg).run_benchmark()
    episodes = result["episodes_df"]
    full_sa = episodes[
        (episodes.solver_id == "Simulated Annealing")
        & (episodes.adaptation_mode == "full")
    ].sort_values("episode")
    # Episode 0 is recorded before its first adaptation step.
    first = full_sa.loc[full_sa.episode == 0].iloc[0]
    assert first["delta_kappa_norm"] == pytest.approx(0.0, abs=1e-12)
    assert first["delta_theta_norm"] == pytest.approx(0.0, abs=1e-12)
    assert (full_sa["delta_kappa_norm"] >= 0.0).all()
    assert (full_sa["delta_theta_norm"] >= 0.0).all()
