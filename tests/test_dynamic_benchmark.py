import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.dynamic_benchmark import (
    generate_capability_drift_episode,
    generate_task_shift_episode,
    generate_dependency_change_episode,
    generate_specialization_episode,
    generate_robustness_episode,
    MultiEpisodeSimulator,
    compute_adaptation_metrics
)

def test_episode_generators():
    # Test generation of ProblemInstances
    prob_drift = generate_capability_drift_episode(0, seed=42)
    assert len(prob_drift.agents) == 5
    assert len(prob_drift.tasks) == 10

    prob_shift = generate_task_shift_episode(50, seed=42)
    assert len(prob_shift.agents) == 5
    assert len(prob_shift.tasks) == 10

    prob_dep = generate_dependency_change_episode(10, seed=42)
    assert len(prob_dep.agents) == 5
    assert len(prob_dep.tasks) == 10

    prob_spec = generate_specialization_episode(0, seed=42)
    assert len(prob_spec.agents) == 4
    assert len(prob_spec.tasks) == 12

    # Robustness transitions
    prob_robust_init = generate_robustness_episode(10, seed=42)
    assert len(prob_robust_init.agents) == 5

    prob_robust_fail = generate_robustness_episode(30, seed=42)
    # Agent 4 fails, new agent has not joined yet (N_base=5, minus 1 = 4 agents)
    assert len(prob_robust_fail.agents) == 4

    prob_robust_join = generate_robustness_episode(40, seed=42)
    # Agent 4 fails, extra agent joined (N_base-1+1 = 5 agents)
    assert len(prob_robust_join.agents) == 5


def test_multi_episode_simulator():
    # Run a very short simulation to verify compiling and logic
    simulator = MultiEpisodeSimulator(generate_capability_drift_episode, num_episodes=3, seed=42)

    # Run with static
    df_static = simulator.run(kappa_enabled=False, theta_enabled=False)
    assert len(df_static) == 3
    assert "energy" in df_static.columns
    assert "reconfig_cost" in df_static.columns

    # Run with Full EBMAO
    df_ebmao = simulator.run(kappa_enabled=True, theta_enabled=True)
    assert len(df_ebmao) == 3
    assert df_ebmao.iloc[0]["kappa_norm"] >= 0.0


def test_compute_adaptation_metrics():
    # Mock dataframe
    import pandas as pd
    data = {
        "episode": list(range(50)),
        "energy": [1.0] * 25 + [2.5] + [1.8] + [1.0] * 23,
        "reconfig_cost": [0.0] * 50
    }
    df = pd.DataFrame(data)
    metrics = compute_adaptation_metrics(df, perturb_episode=25, total_episodes=50)

    assert metrics["perf_drop"] == 1.5
    assert metrics["recovery_time"] == 2 # episode 25 is 2.5, 26 is 1.8, 27 is 1.0 (baseline 1.0, 1.1*1.0 = 1.1)
    assert metrics["cum_regret"] == 2.3 # (2.5 - 1.0) + (1.8 - 1.0) = 1.5 + 0.8 = 2.3
    assert metrics["convergence"] == 0.0
