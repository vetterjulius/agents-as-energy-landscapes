from __future__ import annotations

import os
import torch
import pytest
import numpy as np
import pandas as pd

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.metrics import (
    EpisodeRecord,
    compute_trajectory_summary,
)
from controlled_benchmark.runner import ControlledBenchmarkRunner
from controlled_benchmark.scenarios import (
    generate_capability_drift_episode,
    generate_dependency_change_episode,
    generate_scenario_trajectory,
    generate_stationary_episode,
    generate_task_shift_episode,
    make_initial_landscape_state,
    problem_instance_to_problem_context,
)
from controlled_benchmark.solvers import (
    BudgetExceededError,
    BudgetedLandscape,
    EnergyAwareGreedySolver,
    EnergyAwareSimulatedAnnealingSolver,
    FixedLandscapeILPSolver,
)
from landscape import Landscape, LandscapeState, ProblemContext


# ---------------------------------------------------------------------------
# Requirement 1: Same seed -> same generated problem instance
# ---------------------------------------------------------------------------
def test_same_seed_generates_same_problem_instance():
    for gen_fn in [
        generate_stationary_episode,
        generate_capability_drift_episode,
        generate_task_shift_episode,
        generate_dependency_change_episode,
    ]:
        inst1 = gen_fn(episode=3, seed=42, perturb_episode=5, N=3, M=6, d=4)
        inst2 = gen_fn(episode=3, seed=42, perturb_episode=5, N=3, M=6, d=4)
        inst_diff = gen_fn(episode=3, seed=99, perturb_episode=5, N=3, M=6, d=4)

        s1 = torch.stack([a.capability_embedding for a in inst1.agents])
        s2 = torch.stack([a.capability_embedding for a in inst2.agents])
        s_diff = torch.stack([a.capability_embedding for a in inst_diff.agents])

        assert torch.equal(s1, s2), "Same seed must generate identical agent capabilities"
        assert not torch.equal(s1, s_diff), "Different seeds should generate different agent capabilities"
        assert torch.equal(inst1.interaction_graph, inst2.interaction_graph)
        assert torch.equal(inst1.co_assignment_costs, inst2.co_assignment_costs)
        assert torch.equal(inst1.risk_weights, inst2.risk_weights)


# ---------------------------------------------------------------------------
# Requirement 2: Compared solvers receive the same problem instance
# ---------------------------------------------------------------------------
def test_compared_solvers_receive_same_problem_instance():
    cfg = BenchmarkConfig.quick_mode(seeds=[42])
    trajectory = generate_scenario_trajectory(
        scenario_id="Capability Drift",
        seed=42,
        num_episodes=cfg.num_episodes,
        perturb_episode=cfg.perturb_episode,
        N=cfg.num_agents,
        M=cfg.num_tasks,
        d=cfg.dim,
    )

    ctx_sa = problem_instance_to_problem_context(trajectory[0])
    ctx_greedy = problem_instance_to_problem_context(trajectory[0])

    assert torch.equal(ctx_sa.s, ctx_greedy.s)
    assert torch.equal(ctx_sa.c, ctx_greedy.c)
    assert torch.equal(ctx_sa.C, ctx_greedy.C)
    assert torch.equal(ctx_sa.W_risk, ctx_greedy.W_risk)
    assert ctx_sa.N == ctx_greedy.N and ctx_sa.M == ctx_greedy.M and ctx_sa.d == ctx_greedy.d


# ---------------------------------------------------------------------------
# Requirement 3: Compared methods receive the same scenario trajectory
# ---------------------------------------------------------------------------
def test_compared_methods_receive_same_scenario_trajectory():
    traj_a = generate_scenario_trajectory(
        scenario_id="Task Shift",
        seed=42,
        num_episodes=10,
        perturb_episode=5,
        N=3,
        M=6,
        d=4,
    )
    traj_b = generate_scenario_trajectory(
        scenario_id="Task Shift",
        seed=42,
        num_episodes=10,
        perturb_episode=5,
        N=3,
        M=6,
        d=4,
    )

    assert len(traj_a) == len(traj_b) == 10
    for ep in range(10):
        s_a = torch.stack([a.capability_embedding for a in traj_a[ep].agents])
        s_b = torch.stack([a.capability_embedding for a in traj_b[ep].agents])
        c_a = torch.stack([t.embedding for t in traj_a[ep].tasks])
        c_b = torch.stack([t.embedding for t in traj_b[ep].tasks])
        assert torch.equal(s_a, s_b)
        assert torch.equal(c_a, c_b)
        assert torch.equal(traj_a[ep].interaction_graph, traj_b[ep].interaction_graph)


# ---------------------------------------------------------------------------
# Requirement 4: Static and adaptive runs differ only in intended landscape
# ---------------------------------------------------------------------------
def test_static_and_adaptive_runs_differ_only_in_intended_landscape():
    cfg = BenchmarkConfig.quick_mode(seeds=[42], num_episodes=4, perturb_episode=2)
    runner = ControlledBenchmarkRunner(cfg)
    trajectory = generate_scenario_trajectory(
        scenario_id="Capability Drift",
        seed=42,
        num_episodes=cfg.num_episodes,
        perturb_episode=cfg.perturb_episode,
        N=cfg.num_agents,
        M=cfg.num_tasks,
        d=cfg.dim,
    )

    summary_stat, records_stat = runner.run_trajectory(
        scenario_id="Capability Drift",
        seed=42,
        solver_id="Simulated Annealing",
        landscape_id="Static",
        adaptation_mode="static",
        trajectory=trajectory,
    )
    summary_adapt, records_adapt = runner.run_trajectory(
        scenario_id="Capability Drift",
        seed=42,
        solver_id="Simulated Annealing",
        landscape_id="Adaptive Full",
        adaptation_mode="full",
        trajectory=trajectory,
    )

    # In episode 0, both methods start with kappa=0
    assert records_stat[0].kappa_norm == 0.0
    assert records_adapt[0].kappa_norm == 0.0

    # In static mode, kappa remains 0 throughout all episodes
    for rec in records_stat:
        assert rec.kappa_norm == 0.0

    # In adaptive mode, kappa updates after episode 0
    assert records_adapt[-1].kappa_norm > 0.0


# ---------------------------------------------------------------------------
# Requirement 5: All stochastic solvers respect the energy-evaluation budget
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("budget", [5, 20, 50, 100])
def test_all_stochastic_solvers_respect_energy_evaluation_budget(budget):
    inst = generate_stationary_episode(episode=0, seed=42, perturb_episode=5, N=3, M=6, d=4)
    ctx = problem_instance_to_problem_context(inst)
    state = make_initial_landscape_state(inst)
    landscape = Landscape(ctx, state)

    initial_X = torch.zeros(3, 6)
    for t in range(6):
        initial_X[t % 3, t] = 1.0

    # 1. Simulated Annealing
    b_land_sa = BudgetedLandscape(landscape, max_evaluations=budget)
    sa = EnergyAwareSimulatedAnnealingSolver()
    res_sa = sa.solve(b_land_sa, initial_X, seed=42)
    assert res_sa.energy_evaluations <= budget
    assert b_land_sa.evaluations_used <= budget

    # 2. Greedy
    b_land_gr = BudgetedLandscape(landscape, max_evaluations=budget)
    greedy = EnergyAwareGreedySolver()
    res_gr = greedy.solve(b_land_gr, initial_X)
    assert res_gr.energy_evaluations <= budget
    assert b_land_gr.evaluations_used <= budget

    # Verify BudgetExceededError is raised if called past budget
    with pytest.raises(BudgetExceededError):
        b_land_sa.evaluate(initial_X)


# ---------------------------------------------------------------------------
# Requirement 6: No hidden warm start exists unless explicitly configured
# ---------------------------------------------------------------------------
def test_no_hidden_warm_start():
    cfg = BenchmarkConfig.quick_mode()
    runner = ControlledBenchmarkRunner(cfg)
    X_init = runner._get_initial_assignment(N=3, M=6)

    # Verify initial assignment is deterministic round robin
    expected = torch.zeros(3, 6)
    for t in range(6):
        expected[t % 3, t] = 1.0
    assert torch.equal(X_init, expected)
    # Task assignment invariant: exactly one agent per task
    assert torch.allclose(X_init.sum(dim=0), torch.ones(6))


# ---------------------------------------------------------------------------
# Requirement 7: Every Landscape.evaluate(X) call uses the explicit Landscape
# ---------------------------------------------------------------------------
def test_every_evaluate_uses_explicit_landscape():
    inst = generate_stationary_episode(episode=0, seed=42, perturb_episode=5, N=3, M=6, d=4)
    ctx = problem_instance_to_problem_context(inst)
    state = make_initial_landscape_state(inst)
    landscape = Landscape(ctx, state)

    budgeted = BudgetedLandscape(landscape, max_evaluations=10)

    X = torch.zeros(3, 6)
    for t in range(6):
        X[t % 3, t] = 1.0

    eval_direct = landscape.evaluate(X)
    eval_budgeted = budgeted.evaluate(X)
    assert eval_direct == eval_budgeted
    assert budgeted.evaluations_used == 1


# ---------------------------------------------------------------------------
# Requirement 8: Solver cannot mutate the supplied LandscapeState
# ---------------------------------------------------------------------------
def test_solver_cannot_mutate_supplied_landscape_state():
    inst = generate_stationary_episode(episode=0, seed=42, perturb_episode=5, N=3, M=6, d=4)
    ctx = problem_instance_to_problem_context(inst)
    original_kappa = torch.randn(3, 4)
    original_theta = inst.interaction_graph.clone()
    state = LandscapeState(kappa=original_kappa.clone(), Theta=original_theta.clone())
    landscape = Landscape(ctx, state)

    budgeted = BudgetedLandscape(landscape, max_evaluations=50)

    # Solve with SA
    sa = EnergyAwareSimulatedAnnealingSolver()
    initial_X = torch.zeros(3, 6)
    for t in range(6):
        initial_X[t % 3, t] = 1.0

    sa.solve(budgeted, initial_X, seed=42)

    # Verify original landscape state was NOT mutated
    assert torch.equal(landscape.state.kappa, original_kappa)
    assert torch.equal(landscape.state.Theta, original_theta)

    # Verify budgeted_landscape.state is a copy and mutating it does not affect landscape
    cloned_state = budgeted.state
    cloned_state.kappa[0, 0] += 999.0
    assert not torch.equal(cloned_state.kappa, landscape.state.kappa)


# ---------------------------------------------------------------------------
# Requirement 9: Results contain seed/problem/solver/landscape metadata
# ---------------------------------------------------------------------------
def test_results_contain_seed_problem_solver_landscape_metadata():
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42],
        scenarios=["Stationary"],
        num_episodes=2,
        perturb_episode=1,
    )
    runner = ControlledBenchmarkRunner(cfg)
    results = runner.run_benchmark()
    runs_df = results["runs_df"]

    required_columns = [
        "run_id",
        "problem_id",
        "scenario_id",
        "seed",
        "solver_id",
        "landscape_id",
        "adaptation_mode",
        "N",
        "M",
        "d",
        "evaluation_budget",
        "recovery_time",
        "perf_drop",
        "cumulative_regret",
        "pre_base_energy",
        "post_base_energy",
        "final_energy",
        "mean_external_energy",
        "mean_internal_energy",
        "convergence",
        "stability",
        "adaptation_magnitude_kappa",
        "adaptation_magnitude_theta",
        "mean_constraint_violations",
        "total_energy_evaluations",
        "total_runtime_sec",
        "all_episodes_optimal",
        "fallback_used",
        "evaluation_landscape_id",
        "git_commit",
        "benchmark_version",
    ]

    for col in required_columns:
        assert col in runs_df.columns, f"Missing required column: {col}"


# ---------------------------------------------------------------------------
# Requirement 10: ILP timeout/fallback is explicitly represented
# ---------------------------------------------------------------------------
def test_ilp_timeout_and_fallback_explicitly_represented():
    # Construct an instance with M=12 where MILP cannot solve in 0.0001 seconds
    inst = generate_stationary_episode(episode=0, seed=42, perturb_episode=5, N=4, M=12, d=4)
    ctx = problem_instance_to_problem_context(inst)
    state = make_initial_landscape_state(inst)
    landscape = Landscape(ctx, state)

    # Extremely short time limit to force timeout
    ilp = FixedLandscapeILPSolver(time_limit_sec=0.0001)
    res = ilp.solve(landscape)

    # Must be explicitly represented as timeout / fallback, NOT silently as optimal ILP
    assert res.timeout is True or res.status in ("timeout", "failed")
    assert res.is_optimal is False
    assert res.fallback_used is True
    assert res.termination_reason in ("timeout", "failed")


# ---------------------------------------------------------------------------
# Requirement 11: Paired result rows can be matched by (problem_id, scenario_id, seed)
# ---------------------------------------------------------------------------
def test_paired_result_rows_matched_by_problem_scenario_seed():
    cfg = BenchmarkConfig.quick_mode(
        seeds=[42, 43],
        scenarios=["Stationary"],
        num_episodes=3,
        perturb_episode=1,
    )
    runner = ControlledBenchmarkRunner(cfg)
    results = runner.run_benchmark()
    runs_df = results["runs_df"]

    sa_adapt = runs_df[(runs_df["solver_id"] == "Simulated Annealing") & (runs_df["adaptation_mode"] == "full")]
    sa_stat = runs_df[(runs_df["solver_id"] == "Simulated Annealing") & (runs_df["adaptation_mode"] == "static")]

    merged = pd.merge(sa_adapt, sa_stat, on=["problem_id", "scenario_id", "seed"], suffixes=("_adapt", "_stat"))
    assert len(merged) == len(cfg.seeds)
    for seed in cfg.seeds:
        assert seed in merged["seed"].values


# ---------------------------------------------------------------------------
# Requirement 12: Primary recovery metric uses the same definition across methods
# ---------------------------------------------------------------------------
def test_primary_recovery_metric_definition_consistent():
    # Test formal computation with known synthetic energy sequences
    # Sequence A: recovers at offset 2 (ep 7)
    records_a = [
        EpisodeRecord(
            episode=ep,
            internal_energy=1.0,
            external_energy=1.0 if ep < 5 else (3.0 if ep == 5 else (2.0 if ep == 6 else 1.05)),
            energy_evaluations=10,
            accepted_moves=2,
            iterations=5,
            runtime_sec=0.01,
            termination_reason="completed",
            solver_status="completed",
            is_optimal=None,
            mip_gap=None,
            timeout=False,
            fallback_used=False,
            reconfig_cost=0.1,
            constraint_violations=0.0,
            coordination_score=1.0,
            load_balance=0.5,
            kappa_norm=0.1,
            theta_diff_norm=0.0,
        )
        for ep in range(10)
    ]
    summary_a = compute_trajectory_summary(
        records=records_a,
        scenario_id="Capability Drift",
        perturb_episode=5,
        pre_window=3,
        post_window=3,
    )
    # Pre-base is 1.0, target is 1.1 * 1.0 = 1.1.
    # At ep 5: E=3.0 > 1.1
    # At ep 6: E=2.0 > 1.1
    # At ep 7: E=1.05 <= 1.1 -> offset 2!
    assert summary_a.recovery_time == 2.0
    assert summary_a.perf_drop == 2.0  # 3.0 - 1.0 = 2.0
    assert summary_a.evaluation_landscape_id == "external_ground_truth"
