"""
Adversarial Tests for P0-1 through P0-4 Blockers.
"""
from __future__ import annotations

import pytest
import torch
import numpy as np
from typing import List

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.metrics import (
    EpisodeRecord,
    RECOVERY_WINDOW,
    compute_recovery_threshold,
    compute_trajectory_summary,
)
from controlled_benchmark.runner import ControlledBenchmarkRunner
from controlled_benchmark.scenarios import (
    generate_base_problem,
    generate_scenario_trajectory,
    generate_stationary_episode,
    make_initial_landscape_state,
    problem_instance_to_problem_context,
)
from controlled_benchmark.solvers import FixedLandscapeILPSolver
from landscape import Landscape


def make_synthetic_records(energies: List[float]) -> List[EpisodeRecord]:
    return [
        EpisodeRecord(
            episode=ep,
            internal_energy=e,
            external_energy=e,
            energy_evaluations=10,
            accepted_moves=1,
            iterations=10,
            runtime_sec=0.001,
            termination_reason="budget_exhausted",
            solver_status="completed",
            is_optimal=None,
            mip_gap=None,
            timeout=False,
            fallback_used=False,
            reconfig_cost=0.0,
            constraint_violations=0.0,
            coordination_score=1.0,
            load_balance=0.5,
            kappa_norm=0.0,
            theta_diff_norm=0.0,
        )
        for ep, e in enumerate(energies)
    ]


def make_landscape_state_after_n_adaptation_steps(n_steps: int, mode: str, seed: int = 42):
    N, M, d = 3, 4, 4
    base = generate_base_problem(seed=seed, N=N, M=M, d=d)
    problem_ctx = problem_instance_to_problem_context(base)
    state = make_initial_landscape_state(base)
    mgr = EpisodeAdaptationManager(adaptation_mode=mode)
    X = torch.zeros(N, M)
    for t in range(M):
        X[t % N, t] = 1.0
    for _ in range(n_steps):
        state = mgr.step(current_state=state, X=X, problem=problem_ctx)
    return state


# ===========================================================================
# 1. Future-Leakage Tests (P0-1)
# ===========================================================================

class TestFutureLeakage:

    def _run_adaptation_to_ep(self, trajectory, mode: str, seed: int):
        N, M, d = 3, 4, 4
        mgr = EpisodeAdaptationManager(adaptation_mode=mode)
        base = generate_base_problem(seed=seed, N=N, M=M, d=d)
        state = make_initial_landscape_state(base)
        X = torch.zeros(N, M)
        for t in range(M):
            X[t % N, t] = 1.0
        for problem_inst in trajectory[:-1]:
            problem_ctx = problem_instance_to_problem_context(problem_inst)
            state = mgr.step(current_state=state, X=X, problem=problem_ctx)
        return state

    def test_static_landscape_state_independent_of_future(self):
        N, M, d = 3, 4, 4
        T = 6
        shared_seed = 42
        prefix = generate_scenario_trajectory("Stationary", seed=shared_seed, num_episodes=T - 1,
                                              perturb_episode=10, N=N, M=M, d=d)
        traj_A = prefix + generate_scenario_trajectory("Capability Drift", seed=shared_seed + 1,
                                                        num_episodes=1, perturb_episode=0, N=N, M=M, d=d)
        traj_B = prefix + generate_scenario_trajectory("Task Shift", seed=shared_seed + 99,
                                                        num_episodes=1, perturb_episode=0, N=N, M=M, d=d)
        state_A = self._run_adaptation_to_ep(traj_A, mode="static", seed=shared_seed)
        state_B = self._run_adaptation_to_ep(traj_B, mode="static", seed=shared_seed)
        assert torch.allclose(state_A.kappa, state_B.kappa, atol=1e-7), \
            "P0-1 FAIL: Static kappa differs between trajectories with identical history! Future leaked."
        assert torch.allclose(state_A.Theta, state_B.Theta, atol=1e-7), \
            "P0-1 FAIL: Static Theta differs between trajectories with identical history! Future leaked."

    def test_kappa_only_landscape_state_independent_of_future(self):
        N, M, d = 3, 4, 4
        T = 6
        shared_seed = 42
        prefix = generate_scenario_trajectory("Stationary", seed=shared_seed, num_episodes=T - 1,
                                              perturb_episode=10, N=N, M=M, d=d)
        traj_A = prefix + generate_scenario_trajectory("Stationary", seed=shared_seed + 1,
                                                        num_episodes=1, perturb_episode=0, N=N, M=M, d=d)
        traj_B = prefix + generate_scenario_trajectory("Dependency Change", seed=shared_seed + 99,
                                                        num_episodes=1, perturb_episode=0, N=N, M=M, d=d)
        state_A = self._run_adaptation_to_ep(traj_A, mode="kappa-only", seed=shared_seed)
        state_B = self._run_adaptation_to_ep(traj_B, mode="kappa-only", seed=shared_seed)
        assert torch.allclose(state_A.kappa, state_B.kappa, atol=1e-7), \
            "P0-1 FAIL: kappa-only kappa differs between trajectories with identical history!"
        assert torch.allclose(state_A.Theta, state_B.Theta, atol=1e-7), \
            "P0-1 FAIL: kappa-only Theta differs between trajectories with identical history!"

    def test_static_kappa_is_always_zero(self):
        state = make_landscape_state_after_n_adaptation_steps(n_steps=5, mode="static")
        assert torch.allclose(state.kappa, torch.zeros_like(state.kappa), atol=1e-9), \
            "P0-1 FAIL: Static kappa is non-zero."

    def test_kappa_only_theta_unchanged(self):
        N, M, d = 3, 4, 4
        base = generate_base_problem(seed=42, N=N, M=M, d=d)
        initial_theta = make_initial_landscape_state(base).Theta.clone()
        state = make_landscape_state_after_n_adaptation_steps(n_steps=5, mode="kappa-only", seed=42)
        assert torch.allclose(state.Theta, initial_theta, atol=1e-7), \
            "P0-1 FAIL: kappa-only mode mutated Theta."

    def test_theta_only_kappa_unchanged(self):
        state = make_landscape_state_after_n_adaptation_steps(n_steps=5, mode="theta-only")
        assert torch.allclose(state.kappa, torch.zeros_like(state.kappa), atol=1e-9), \
            "P0-1 FAIL: theta-only mode mutated kappa."

    def test_static_kappa_zero_in_runner(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42], scenarios=["Stationary"],
                                         num_episodes=8, perturb_episode=4)
        runner = ControlledBenchmarkRunner(cfg)
        trajectory = generate_scenario_trajectory("Stationary", seed=42, num_episodes=8,
                                                  perturb_episode=4, N=cfg.num_agents,
                                                  M=cfg.num_tasks, d=cfg.dim)
        _, records = runner.run_trajectory(
            scenario_id="Stationary", seed=42, solver_id="Simulated Annealing",
            landscape_id="Static", adaptation_mode="static", trajectory=trajectory)
        for rec in records:
            assert rec.kappa_norm == 0.0, \
                f"P0-1 FAIL: Static run kappa_norm={rec.kappa_norm} at ep {rec.episode}."


# ===========================================================================
# 2. Stationary Invariance Tests (P0-2)
# ===========================================================================

class TestStationaryInvariance:

    def test_stationary_all_episodes_identical(self):
        N, M, d = 3, 6, 4
        trajectory = generate_scenario_trajectory("Stationary", seed=42, num_episodes=15,
                                                  perturb_episode=7, N=N, M=M, d=d)
        inst0 = trajectory[0]
        s0 = torch.stack([a.capability_embedding for a in inst0.agents])
        c0 = torch.stack([t.embedding for t in inst0.tasks])
        for ep, inst in enumerate(trajectory[1:], start=1):
            s_ep = torch.stack([a.capability_embedding for a in inst.agents])
            c_ep = torch.stack([t.embedding for t in inst.tasks])
            assert torch.equal(s0, s_ep), f"P0-2 FAIL: Stationary agents differ at ep {ep}."
            assert torch.equal(c0, c_ep), f"P0-2 FAIL: Stationary tasks differ at ep {ep}."
            assert torch.equal(inst0.interaction_graph, inst.interaction_graph), \
                f"P0-2 FAIL: Stationary graph differs at ep {ep}."
            assert torch.equal(inst0.risk_weights, inst.risk_weights), \
                f"P0-2 FAIL: Stationary risk_weights differ at ep {ep}."

    def test_stationary_reproducible_across_calls(self):
        traj_a = generate_scenario_trajectory("Stationary", seed=42, num_episodes=10,
                                              perturb_episode=5, N=3, M=6, d=4)
        traj_b = generate_scenario_trajectory("Stationary", seed=42, num_episodes=10,
                                              perturb_episode=5, N=3, M=6, d=4)
        for ep, (a, b) in enumerate(zip(traj_a, traj_b)):
            s_a = torch.stack([ag.capability_embedding for ag in a.agents])
            s_b = torch.stack([ag.capability_embedding for ag in b.agents])
            assert torch.equal(s_a, s_b), f"P0-2 FAIL: Stationary not reproducible at ep {ep}."

    def test_different_seeds_produce_different_base_problems(self):
        base_a = generate_base_problem(seed=42, N=3, M=6, d=4)
        base_b = generate_base_problem(seed=43, N=3, M=6, d=4)
        s_a = torch.stack([a.capability_embedding for a in base_a.agents])
        s_b = torch.stack([a.capability_embedding for a in base_b.agents])
        assert not torch.equal(s_a, s_b), "P0-2: Different seeds same agent embeddings."


# ===========================================================================
# 3. Perturbation-Isolation Tests (P0-2)
# ===========================================================================

class TestPerturbationIsolation:

    def test_capability_drift_only_changes_agents(self):
        N, M, d, perturb = 3, 6, 4, 5
        traj = generate_scenario_trajectory("Capability Drift", seed=42, num_episodes=10,
                                            perturb_episode=perturb, N=N, M=M, d=d)
        ep_before, ep_after = traj[perturb - 1], traj[perturb]
        c_b = torch.stack([t.embedding for t in ep_before.tasks])
        c_a = torch.stack([t.embedding for t in ep_after.tasks])
        assert torch.equal(c_b, c_a), "P0-2: Capability Drift changed task embeddings."
        assert torch.equal(ep_before.interaction_graph, ep_after.interaction_graph), \
            "P0-2: Capability Drift changed interaction_graph."
        assert torch.equal(ep_before.risk_weights, ep_after.risk_weights), \
            "P0-2: Capability Drift changed risk_weights."
        s_b = torch.stack([a.capability_embedding for a in ep_before.agents])
        s_a = torch.stack([a.capability_embedding for a in ep_after.agents])
        assert not torch.equal(s_b, s_a), "P0-2: Capability Drift did not change agents."

    def test_task_shift_only_changes_embeddings(self):
        N, M, d, perturb = 3, 6, 4, 5
        traj = generate_scenario_trajectory("Task Shift", seed=42, num_episodes=10,
                                            perturb_episode=perturb, N=N, M=M, d=d)
        ep_before, ep_after = traj[perturb - 1], traj[perturb]
        s_b = torch.stack([a.capability_embedding for a in ep_before.agents])
        s_a = torch.stack([a.capability_embedding for a in ep_after.agents])
        assert torch.equal(s_b, s_a), "P0-2: Task Shift changed agent capabilities."
        assert torch.equal(ep_before.interaction_graph, ep_after.interaction_graph), \
            "P0-2: Task Shift changed interaction_graph."
        assert torch.equal(ep_before.risk_weights, ep_after.risk_weights), \
            "P0-2: Task Shift changed risk_weights."
        c_b = torch.stack([t.embedding for t in ep_before.tasks])
        c_a = torch.stack([t.embedding for t in ep_after.tasks])
        assert not torch.equal(c_b, c_a), "P0-2: Task Shift did not change task embeddings."

    def test_task_shift_exactly_15(self):
        N, M, d, perturb = 3, 6, 4, 5
        traj = generate_scenario_trajectory("Task Shift", seed=42, num_episodes=10,
                                            perturb_episode=perturb, N=N, M=M, d=d)
        c_pre = torch.stack([t.embedding for t in traj[perturb - 1].tasks])
        c_post = torch.stack([t.embedding for t in traj[perturb].tasks])
        diff = c_post - c_pre
        expected = (torch.ones(d) * 1.5).unsqueeze(0).expand(M, -1)
        assert torch.allclose(diff, expected, atol=1e-6), \
            "P0-2: Task Shift embedding not exactly +1.5."

    def test_dependency_change_only_changes_graph(self):
        N, M, d, perturb = 3, 6, 4, 5
        traj = generate_scenario_trajectory("Dependency Change", seed=42, num_episodes=10,
                                            perturb_episode=perturb, N=N, M=M, d=d)
        ep_before, ep_after = traj[perturb - 1], traj[perturb]
        s_b = torch.stack([a.capability_embedding for a in ep_before.agents])
        s_a = torch.stack([a.capability_embedding for a in ep_after.agents])
        assert torch.equal(s_b, s_a), "P0-2: Dependency Change changed agents."
        c_b = torch.stack([t.embedding for t in ep_before.tasks])
        c_a = torch.stack([t.embedding for t in ep_after.tasks])
        assert torch.equal(c_b, c_a), "P0-2: Dependency Change changed tasks."
        assert torch.equal(ep_before.risk_weights, ep_after.risk_weights), \
            "P0-2: Dependency Change changed risk_weights."
        assert not torch.equal(ep_before.interaction_graph, ep_after.interaction_graph), \
            "P0-2: Dependency Change did not change interaction_graph."

    def test_pre_perturbation_identical_all_scenarios(self):
        N, M, d, perturb = 3, 6, 4, 5
        for scenario in ["Stationary", "Capability Drift", "Task Shift", "Dependency Change"]:
            traj = generate_scenario_trajectory(scenario, seed=42, num_episodes=10,
                                                perturb_episode=perturb, N=N, M=M, d=d)
            ep0 = traj[0]
            s0 = torch.stack([a.capability_embedding for a in ep0.agents])
            c0 = torch.stack([t.embedding for t in ep0.tasks])
            for ep_idx in range(1, perturb):
                ep = traj[ep_idx]
                s_ep = torch.stack([a.capability_embedding for a in ep.agents])
                c_ep = torch.stack([t.embedding for t in ep.tasks])
                assert torch.equal(s0, s_ep), \
                    f"P0-2 [{scenario}]: ep {ep_idx} agents differ pre-perturbation."
                assert torch.equal(c0, c_ep), \
                    f"P0-2 [{scenario}]: ep {ep_idx} tasks differ pre-perturbation."
                assert torch.equal(ep0.interaction_graph, ep.interaction_graph), \
                    f"P0-2 [{scenario}]: ep {ep_idx} graph differs pre-perturbation."


# ===========================================================================
# 4-6. Solver-Independent Recovery Threshold (P0-3)
# ===========================================================================

class TestRecoveryMetric:

    def test_recovery_threshold_formula(self):
        ref = 1.0
        threshold = compute_recovery_threshold(ref)
        assert threshold == pytest.approx(1.1, rel=1e-6)

    def test_bad_solver_cannot_win_recovery(self):
        perturb = 5
        good_energies = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 0.9, 0.9, 0.9, 0.9]
        bad_energies  = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        good_summary = compute_trajectory_summary(make_synthetic_records(good_energies),
                                                  "Capability Drift", perturb, pre_window=3, post_window=3)
        bad_summary  = compute_trajectory_summary(make_synthetic_records(bad_energies),
                                                  "Capability Drift", perturb, pre_window=3, post_window=3)
        assert good_summary.recovery_time < bad_summary.recovery_time, \
            f"P0-3 FAIL: Bad solver recovered faster ({bad_summary.recovery_time}) than good ({good_summary.recovery_time})!"
        assert bad_summary.recovery_time == float(len(bad_energies) - perturb), \
            f"P0-3 FAIL: Bad solver got non-censored recovery_time={bad_summary.recovery_time}."

    def test_1_hit_not_recovery(self):
        perturb = 5
        threshold = compute_recovery_threshold(1.0)
        energies = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, threshold - 0.01, 3.0, 3.0, 3.0]
        summary = compute_trajectory_summary(make_synthetic_records(energies),
                                             "Capability Drift", perturb, pre_window=3, post_window=3)
        assert summary.recovery_time == float(len(energies) - perturb), \
            f"P0-3 FAIL: 1 hit produced recovery={summary.recovery_time} (should be censored)."

    def test_2_consecutive_not_recovery(self):
        perturb = 5
        threshold = compute_recovery_threshold(1.0)
        energies = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, threshold - 0.01, threshold - 0.01, 3.0, 3.0]
        summary = compute_trajectory_summary(make_synthetic_records(energies),
                                             "Capability Drift", perturb, pre_window=3, post_window=3)
        assert summary.recovery_time == float(len(energies) - perturb), \
            f"P0-3 FAIL: 2 hits produced recovery={summary.recovery_time} (should be censored)."

    def test_3_consecutive_is_recovery_at_offset_1(self):
        perturb = 5
        threshold = compute_recovery_threshold(1.0)
        energies = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, threshold - 0.01, threshold - 0.01, threshold - 0.01, 3.0]
        summary = compute_trajectory_summary(make_synthetic_records(energies),
                                             "Capability Drift", perturb, pre_window=3, post_window=3)
        assert summary.recovery_time == 1.0, \
            f"P0-3 FAIL: 3 consecutive hits returned recovery_time={summary.recovery_time} (expected 1.0)."

    def test_never_recovered_is_censored(self):
        perturb = 5
        energies = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        summary = compute_trajectory_summary(make_synthetic_records(energies),
                                             "Capability Drift", perturb, pre_window=3, post_window=3)
        assert summary.recovery_time == float(len(energies) - perturb)

    def test_recovery_threshold_same_for_all_solvers(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42], scenarios=["Capability Drift"],
                                         num_episodes=10, perturb_episode=5)
        runner = ControlledBenchmarkRunner(cfg)
        results = runner.run_benchmark()
        runs_df = results["runs_df"]
        seed_df = runs_df[(runs_df["scenario_id"] == "Capability Drift") & (runs_df["seed"] == 42)]
        thresholds = seed_df["recovery_threshold"].unique()
        assert len(thresholds) == 1, \
            f"P0-3 FAIL: Multiple recovery thresholds for same (scenario, seed): {thresholds}"


# ===========================================================================
# 7. Static-vs-Adaptive State Isolation (P0-1 + P0-2)
# ===========================================================================

class TestStaticAdaptiveStateIsolation:

    def test_static_kappa_invariant(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42], scenarios=["Dependency Change"],
                                         num_episodes=10, perturb_episode=5)
        runner = ControlledBenchmarkRunner(cfg)
        trajectory = generate_scenario_trajectory("Dependency Change", seed=42, num_episodes=10,
                                                  perturb_episode=5, N=cfg.num_agents,
                                                  M=cfg.num_tasks, d=cfg.dim)
        _, records = runner.run_trajectory(
            scenario_id="Dependency Change", seed=42, solver_id="Simulated Annealing",
            landscape_id="Static", adaptation_mode="static", trajectory=trajectory)
        for rec in records:
            assert rec.kappa_norm == 0.0, \
                f"P0-1 FAIL: Static kappa non-zero at ep {rec.episode}."

    def test_adaptive_kappa_evolves(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42], scenarios=["Stationary"],
                                         num_episodes=8, perturb_episode=4)
        runner = ControlledBenchmarkRunner(cfg)
        trajectory = generate_scenario_trajectory("Stationary", seed=42, num_episodes=8,
                                                  perturb_episode=4, N=cfg.num_agents,
                                                  M=cfg.num_tasks, d=cfg.dim)
        _, records = runner.run_trajectory(
            scenario_id="Stationary", seed=42, solver_id="Simulated Annealing",
            landscape_id="Adaptive Full", adaptation_mode="full", trajectory=trajectory)
        assert any(rec.kappa_norm > 0.0 for rec in records[1:]), \
            "FAIL: Full adaptive mode never updated kappa."


# ===========================================================================
# 8. Reproducibility Test
# ===========================================================================

class TestReproducibility:

    def test_full_trajectory_reproducible(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42], scenarios=["Capability Drift"],
                                         num_episodes=10, perturb_episode=5)
        runner = ControlledBenchmarkRunner(cfg)
        trajectory = generate_scenario_trajectory("Capability Drift", seed=42, num_episodes=10,
                                                  perturb_episode=5, N=cfg.num_agents,
                                                  M=cfg.num_tasks, d=cfg.dim)
        _, records_a = runner.run_trajectory(
            scenario_id="Capability Drift", seed=42, solver_id="Simulated Annealing",
            landscape_id="Static", adaptation_mode="static", trajectory=trajectory)
        _, records_b = runner.run_trajectory(
            scenario_id="Capability Drift", seed=42, solver_id="Simulated Annealing",
            landscape_id="Static", adaptation_mode="static", trajectory=trajectory)
        for ep, (ra, rb) in enumerate(zip(records_a, records_b)):
            assert ra.external_energy == pytest.approx(rb.external_energy, rel=1e-8), \
                f"FAIL: Reproducibility broken at ep {ep}."


# ===========================================================================
# 9. ILP Fallback/Status Correctness Test (P0-4)
# ===========================================================================

class TestILPFallback:

    def test_ilp_timeout_is_explicitly_flagged(self):
        inst = generate_stationary_episode(episode=0, seed=42, perturb_episode=5, N=4, M=12, d=4)
        ctx = problem_instance_to_problem_context(inst)
        state = make_initial_landscape_state(inst)
        landscape = Landscape(ctx, state)
        ilp = FixedLandscapeILPSolver(time_limit_sec=0.0001)
        res = ilp.solve(landscape)
        assert res.is_optimal is False, "P0-4 FAIL: ILP timeout marked as optimal."
        assert res.fallback_used is True, "P0-4 FAIL: ILP timeout did not set fallback_used=True."
        assert res.status in ("timeout", "failed")
        assert res.time_limit_sec == pytest.approx(0.0001, rel=1e-3)

    def test_ilp_success_on_small_instance(self):
        inst = generate_stationary_episode(episode=0, seed=42, perturb_episode=5, N=2, M=3, d=4)
        ctx = problem_instance_to_problem_context(inst)
        state = make_initial_landscape_state(inst)
        landscape = Landscape(ctx, state)
        ilp = FixedLandscapeILPSolver(time_limit_sec=30.0)
        res = ilp.solve(landscape)
        assert res.is_optimal is True
        assert res.fallback_used is False
        assert res.status == "optimal"

    def test_ilp_fallback_not_optimal_in_output(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42], scenarios=["Stationary"],
                                         num_episodes=2, perturb_episode=1, run_ilp_dynamic=True)
        runner = ControlledBenchmarkRunner(cfg)
        results = runner.run_benchmark()
        runs_df = results["runs_df"]
        fallback_rows = runs_df[runs_df["fallback_used"] == True]
        if len(fallback_rows) > 0:
            assert not fallback_rows["all_episodes_optimal"].any(), \
                "P0-4 FAIL: Rows with fallback_used=True marked as all_episodes_optimal."


# ===========================================================================
# 10. Paired Statistical Grouping Test (P1)
# ===========================================================================

class TestPairedStatisticalGrouping:

    def test_hypothesis_family_labels_present(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42, 43], scenarios=["Stationary"],
                                         num_episodes=4, perturb_episode=2)
        runner = ControlledBenchmarkRunner(cfg)
        results = runner.run_benchmark()
        stats_df = results["stats_df"]
        assert "hypothesis_family" in stats_df.columns
        assert not stats_df["hypothesis_family"].isna().any()

    def test_four_families_present(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42, 43], scenarios=["Stationary"],
                                         num_episodes=4, perturb_episode=2)
        runner = ControlledBenchmarkRunner(cfg)
        results = runner.run_benchmark()
        stats_df = results["stats_df"]
        families = set(stats_df["hypothesis_family"].unique())
        expected = {"Family 1: PRIMARY", "Family 2: SECONDARY",
                    "Family 3: ABLATION", "Family 4: SOLVER INTERACTION"}
        missing = expected - families
        assert not missing, f"P1 FAIL: Missing hypothesis families: {missing}"

    def test_provenance_columns_present(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42, 43], scenarios=["Stationary"],
                                         num_episodes=3, perturb_episode=1)
        runner = ControlledBenchmarkRunner(cfg)
        results = runner.run_benchmark()
        runs_df = results["runs_df"]
        for col in ["problem_id", "scenario_id", "seed", "solver_id", "adaptation_mode",
                    "recovery_time", "mean_external_energy", "cumulative_regret",
                    "recovery_threshold", "reference_method", "evaluation_landscape_id",
                    "max_energy_evaluations"]:
            assert col in runs_df.columns, f"P1 FAIL: Missing column '{col}'."


# ===========================================================================
# 11. Recovery Reference Method Consistency Test (P0-3)
# ===========================================================================

class TestReferenceMethodConsistency:

    def test_reference_method_consistent_per_scenario_seed(self):
        cfg = BenchmarkConfig.quick_mode(seeds=[42], scenarios=["Capability Drift", "Task Shift"],
                                         num_episodes=10, perturb_episode=5)
        runner = ControlledBenchmarkRunner(cfg)
        results = runner.run_benchmark()
        runs_df = results["runs_df"]
        for (scenario, seed), group in runs_df.groupby(["scenario_id", "seed"]):
            methods = group["reference_method"].unique()
            assert len(methods) == 1, \
                f"P0-3 FAIL: Multiple reference methods for ({scenario}, seed={seed}): {methods}"
