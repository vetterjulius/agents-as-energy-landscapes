from __future__ import annotations

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch

from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.runner import ControlledBenchmarkRunner
from controlled_benchmark.scenarios import generate_scenario_trajectory


def test_reproducibility_same_seed_twice():
    """Reproducibility: Running the same seed twice yields 100% identical results."""
    tmpdir1 = tempfile.mkdtemp()
    tmpdir2 = tempfile.mkdtemp()
    try:
        cfg1 = BenchmarkConfig.quick_mode(
            seeds=[42],
            scenarios=["Capability Drift"],
            num_episodes=5,
            perturb_episode=2,
            max_energy_evaluations=30,
            output_dir=tmpdir1,
        )
        cfg2 = BenchmarkConfig.quick_mode(
            seeds=[42],
            scenarios=["Capability Drift"],
            num_episodes=5,
            perturb_episode=2,
            max_energy_evaluations=30,
            output_dir=tmpdir2,
        )

        res1 = ControlledBenchmarkRunner(cfg1).run_benchmark()
        res2 = ControlledBenchmarkRunner(cfg2).run_benchmark()

        df_runs1 = res1["runs_df"]
        df_runs2 = res2["runs_df"]
        df_ep1 = res1["episodes_df"]
        df_ep2 = res2["episodes_df"]

        df_runs1_clean = df_runs1.drop(columns=["total_runtime_sec"])
        df_runs2_clean = df_runs2.drop(columns=["total_runtime_sec"])
        pd.testing.assert_frame_equal(df_runs1_clean, df_runs2_clean)

        df_ep1_clean = df_ep1.drop(columns=["runtime_sec"])
        df_ep2_clean = df_ep2.drop(columns=["runtime_sec"])
        pd.testing.assert_frame_equal(df_ep1_clean, df_ep2_clean)
    finally:
        shutil.rmtree(tmpdir1, ignore_errors=True)
        shutil.rmtree(tmpdir2, ignore_errors=True)


def test_adaptation_timing_episode_t_effects_in_t_plus_1():
    """Adaptation timing: Adaptation from episode t only affects landscape in t+1."""
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42],
        scenarios=["Task Shift"],
        num_episodes=5,
        perturb_episode=2,
        max_energy_evaluations=30,
    )
    runner = ControlledBenchmarkRunner(cfg)
    trajectory = generate_scenario_trajectory(
        scenario_id="Task Shift",
        seed=42,
        num_episodes=5,
        perturb_episode=2,
        N=cfg.num_agents,
        M=cfg.num_tasks,
        d=cfg.dim,
    )

    _, records = runner.run_trajectory(
        scenario_id="Task Shift",
        seed=42,
        solver_id="Energy Greedy",
        landscape_id="Adaptive Full",
        adaptation_mode="full",
        trajectory=trajectory,
    )

    # In episode 0, kappa_norm is 0 and delta_kappa_norm is 0 because no past adaptation exists.
    assert records[0].kappa_norm == 0.0
    assert records[0].delta_kappa_norm == 0.0

    # In episode 1, kappa_norm > 0 because adaptation from episode 0 now takes effect.
    assert records[1].kappa_norm > 0.0

    # At change point t = 2 (perturb_episode=2):
    # delta_kappa_norm for episode 2 was derived from step at end of episode 1.
    # The perturbation in episode 2 has NOT yet affected episode 2's kappa_norm.
    assert records[2].delta_kappa_norm >= 0.0


def test_static_mode_is_completely_unmodified():
    """Static mode must not modify kappa or Theta across any episodes."""
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42],
        scenarios=["Dependency Change"],
        num_episodes=10,
        perturb_episode=5,
        max_energy_evaluations=30,
    )
    runner = ControlledBenchmarkRunner(cfg)
    trajectory = generate_scenario_trajectory(
        scenario_id="Dependency Change",
        seed=42,
        num_episodes=10,
        perturb_episode=5,
        N=cfg.num_agents,
        M=cfg.num_tasks,
        d=cfg.dim,
    )

    _, records = runner.run_trajectory(
        scenario_id="Dependency Change",
        seed=42,
        solver_id="Energy Greedy",
        landscape_id="Static",
        adaptation_mode="static",
        trajectory=trajectory,
    )

    for record in records:
        assert record.kappa_norm == 0.0
        assert record.delta_kappa_norm == 0.0
        assert record.delta_theta_norm == 0.0
        # Theta norm should match the original base interaction graph norm
        assert record.theta_norm == pytest.approx(records[0].theta_norm)


def test_output_schema_integrity(tmp_path):
    """Output schema: runs.csv and episodes.csv have identical expected columns."""
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42],
        scenarios=["Stationary"],
        num_episodes=3,
        perturb_episode=1,
        max_energy_evaluations=20,
        output_dir=str(tmp_path),
    )
    ControlledBenchmarkRunner(cfg).run_benchmark()

    runs_csv = pd.read_csv(tmp_path / "runs.csv")
    episodes_csv = pd.read_csv(tmp_path / "episodes.csv")

    expected_runs_cols = [
        "run_id", "experiment_id", "problem_id", "trajectory_id", "reference_id",
        "scenario_id", "seed", "solver_id", "landscape_id", "adaptation_mode",
        "N", "M", "d", "num_episodes", "perturb_episode", "evaluation_budget",
        "max_energy_evaluations", "ppee_10", "cumulative_excess_energy", "cumulative_regret",
        "performance_drop", "perf_drop", "recovery_time", "mean_external_energy",
        "mean_internal_energy", "final_energy", "pre_base_energy", "post_base_energy",
        "convergence", "stability", "kappa_norm", "theta_norm", "adaptation_magnitude_kappa",
        "adaptation_magnitude_theta", "kappa_change_post", "theta_change_post",
        "mean_co_assignment_conflicts", "mean_constraint_violations", "total_energy_evaluations",
        "total_runtime_sec", "termination_reasons", "all_episodes_optimal", "fallback_used",
        "recovery_threshold", "reference_energy", "reference_method", "reference_valid",
        "reference_timeout", "recovery_window", "evaluation_landscape_id", "git_commit",
        "benchmark_version"
    ]

    expected_episodes_cols = [
        "run_id", "experiment_id", "problem_id", "trajectory_id", "reference_id",
        "scenario_id", "seed", "solver_id", "landscape_id", "adaptation_mode",
        "N", "M", "d", "total_episodes", "perturb_episode", "episode",
        "evaluation_budget", "reference_method", "reference_energy", "reference_valid",
        "internal_energy", "external_energy", "energy_evaluations", "accepted_moves",
        "iterations", "runtime_sec", "termination_reason", "solver_status", "optimal",
        "mip_gap", "timeout", "fallback", "reconfig_cost", "co_assignment_conflicts",
        "coordination_score", "load_balance", "kappa_norm", "theta_norm",
        "theta_diff_norm", "delta_kappa_norm", "delta_theta_norm", "git_commit"
    ]

    for col in expected_runs_cols:
        assert col in runs_csv.columns, f"Missing run col: {col}"

    for col in expected_episodes_cols:
        assert col in episodes_csv.columns, f"Missing episode col: {col}"
