from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.metrics import (
    EpisodeRecord,
    TrajectorySummary,
    compute_co_assignment_conflicts,
    compute_coordination_score,
    compute_load_balance,
    compute_reconfig_cost,
    compute_trajectory_summary,
)
from controlled_benchmark.scenarios import (
    generate_scenario_trajectory,
    make_initial_landscape_state,
    problem_instance_to_problem_context,
)
from controlled_benchmark.solvers import (
    BudgetedLandscape,
    EnergyAwareGreedySolver,
    EnergyAwareSimulatedAnnealingSolver,
    FixedLandscapeILPSolver,
    SolverResult,
)
from controlled_benchmark.statistics import (
    analyze_paired_comparison,
    analyze_solver_interaction,
    holm_bonferroni_correction,
)
from landscape import Landscape, LandscapeState


def _get_git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return bool(output)
    except Exception:
        return False


def _trajectory_fingerprint(trajectory: List[Any]) -> str:
    digest = hashlib.sha256()
    for problem in trajectory:
        for agent in problem.agents:
            digest.update(agent.capability_embedding.detach().cpu().numpy().tobytes())
        for task in problem.tasks:
            digest.update(task.embedding.detach().cpu().numpy().tobytes())
        for tensor in (
            problem.interaction_graph,
            problem.co_assignment_costs,
            problem.risk_weights,
        ):
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


class ControlledBenchmarkRunner:
    """Runner for the controlled landscape/solver benchmark."""

    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.sa_solver = EnergyAwareSimulatedAnnealingSolver(
            temperature_init=config.sa_temperature_init,
            min_temperature=config.sa_min_temperature,
            cooling_rate=config.sa_cooling_rate,
        )
        self.greedy_solver = EnergyAwareGreedySolver()
        self.ilp_solver = FixedLandscapeILPSolver(time_limit_sec=config.ilp_time_limit_sec)

    def _get_initial_assignment(self, N: int, M: int) -> torch.Tensor:
        X = torch.zeros(N, M, dtype=torch.float32)
        for task in range(M):
            X[task % N, task] = 1.0
        return X

    def _ground_truth_reference(
        self, problem_inst: Any
    ) -> Tuple[float, str, bool, bool]:
        context = problem_instance_to_problem_context(
            problem_inst,
            lambda_align=self.cfg.lambda_align,
            lambda_memory=self.cfg.lambda_memory,
            interaction_weight=self.cfg.interaction_weight,
            cost_weight=self.cfg.cost_weight,
            risk_weight=self.cfg.risk_weight,
        )
        state = LandscapeState(
            kappa=torch.zeros(self.cfg.num_agents, self.cfg.dim),
            Theta=problem_inst.interaction_graph.clone(),
        )
        result = self.ilp_solver.solve(Landscape(context, state))
        method = "exact_optimal" if result.is_optimal is True else "ilp_timeout_or_failure"
        return float(result.energy), method, bool(result.is_optimal is True), bool(result.timeout)

    def run_trajectory(
        self,
        scenario_id: str,
        seed: int,
        solver_id: str,
        landscape_id: str,
        adaptation_mode: str,
        trajectory: List[Any],
        reference_energy: Optional[float] = None,
        reference_method: Optional[str] = None,
        reference_valid: Optional[bool] = None,
    ) -> Tuple[TrajectorySummary, List[EpisodeRecord]]:
        """Run one condition; adaptation at t only creates state for t+1."""
        N, M, d = self.cfg.num_agents, self.cfg.num_tasks, self.cfg.dim
        manager = EpisodeAdaptationManager(
            adaptation_mode=adaptation_mode,
            eta_memory=self.cfg.eta_memory,
            eta_theta=self.cfg.eta_theta,
        )
        current_state = make_initial_landscape_state(trajectory[0])
        records: List[EpisodeRecord] = []
        previous_X: Optional[torch.Tensor] = None
        previous_state: Optional[LandscapeState] = None
        state_snapshots: Dict[int, LandscapeState] = {}

        for episode, problem_inst in enumerate(trajectory):
            context = problem_instance_to_problem_context(
                problem_inst,
                lambda_align=self.cfg.lambda_align,
                lambda_memory=self.cfg.lambda_memory,
                interaction_weight=self.cfg.interaction_weight,
                cost_weight=self.cfg.cost_weight,
                risk_weight=self.cfg.risk_weight,
            )
            solver_landscape = Landscape(context, current_state)
            ground_truth = Landscape(
                context,
                LandscapeState(
                    kappa=torch.zeros(N, d),
                    Theta=problem_inst.interaction_graph.clone(),
                ),
            )
            budgeted = BudgetedLandscape(
                solver_landscape, max_evaluations=self.cfg.max_energy_evaluations
            )
            initial_X = self._get_initial_assignment(N, M)

            if solver_id == "Simulated Annealing":
                solver_result = self.sa_solver.solve(
                    budgeted, initial_X, seed=seed * 10000 + episode
                )
            elif solver_id == "Energy Greedy":
                solver_result = self.greedy_solver.solve(budgeted, initial_X)
            elif solver_id == "Exact ILP":
                solver_result = self.ilp_solver.solve(solver_landscape)
            else:
                raise ValueError(f"Unknown solver_id: {solver_id}")

            X = solver_result.X.clone()
            internal_energy = solver_landscape.evaluate(X)
            external_energy = ground_truth.evaluate(X)
            kappa_norm = float(current_state.kappa.norm().item())
            theta_norm = float(current_state.Theta.norm().item())
            theta_diff_norm = float(
                (current_state.Theta - problem_inst.interaction_graph).norm().item()
            )
            if previous_state is None:
                delta_kappa = 0.0
                delta_theta = 0.0
            else:
                delta_kappa = float((current_state.kappa - previous_state.kappa).norm().item())
                delta_theta = float((current_state.Theta - previous_state.Theta).norm().item())
            state_snapshots[episode] = current_state.clone()

            records.append(
                EpisodeRecord(
                    episode=episode,
                    internal_energy=internal_energy,
                    external_energy=external_energy,
                    energy_evaluations=solver_result.energy_evaluations,
                    accepted_moves=solver_result.accepted_moves,
                    iterations=solver_result.iterations,
                    runtime_sec=solver_result.runtime_sec,
                    termination_reason=solver_result.termination_reason,
                    solver_status=solver_result.status,
                    is_optimal=solver_result.is_optimal,
                    mip_gap=solver_result.mip_gap,
                    timeout=solver_result.timeout,
                    fallback_used=solver_result.fallback_used,
                    reconfig_cost=compute_reconfig_cost(previous_X, X),
                    constraint_violations=compute_co_assignment_conflicts(context.C, X),
                    coordination_score=compute_coordination_score(
                        problem_inst.interaction_graph, X
                    ),
                    load_balance=compute_load_balance(X),
                    kappa_norm=kappa_norm,
                    theta_diff_norm=theta_diff_norm,
                    delta_kappa_norm=delta_kappa,
                    delta_theta_norm=delta_theta,
                    theta_norm=theta_norm,
                )
            )

            # The observation from episode t is applied only after all episode-t metrics.
            previous_state = current_state.clone()
            current_state = manager.step(current_state, X, context)
            previous_X = X

        if reference_energy is None:
            target = trajectory[min(self.cfg.perturb_episode, len(trajectory) - 1)]
            reference_energy, reference_method, reference_valid, _ = self._ground_truth_reference(target)

        snapshot_start = self.cfg.perturb_episode
        snapshot_end = min(
            self.cfg.perturb_episode + self.cfg.post_ppee_window,
            len(trajectory) - 1,
        )
        kappa_change = theta_change = None
        if snapshot_start < len(trajectory) and snapshot_end >= snapshot_start:
            start_state = state_snapshots[snapshot_start]
            end_state = state_snapshots[snapshot_end]
            kappa_change = float((end_state.kappa - start_state.kappa).norm().item())
            theta_change = float((end_state.Theta - start_state.Theta).norm().item())

        summary = compute_trajectory_summary(
            records=records,
            scenario_id=scenario_id,
            perturb_episode=self.cfg.perturb_episode,
            pre_window=self.cfg.pre_window,
            post_window=self.cfg.post_window,
            post_ppee_window=self.cfg.post_ppee_window,
            reference_energy=reference_energy,
            reference_method=reference_method,
            kappa_change_post=kappa_change,
            theta_change_post=theta_change,
        )
        return summary, records

    def _cells(self) -> List[Tuple[str, str, str]]:
        cells = [
            ("Simulated Annealing", "Static", "static"),
            ("Simulated Annealing", "Kappa-only", "kappa-only"),
            ("Simulated Annealing", "Theta-only", "theta-only"),
            ("Simulated Annealing", "Full", "full"),
            ("Energy Greedy", "Static", "static"),
            ("Energy Greedy", "Kappa-only", "kappa-only"),
            ("Energy Greedy", "Theta-only", "theta-only"),
            ("Energy Greedy", "Full", "full"),
        ]
        if self.cfg.run_ilp_dynamic:
            cells.extend(
                [
                    ("Exact ILP", "Static", "static"),
                    ("Exact ILP", "Full", "full"),
                ]
            )
        return cells

    def run_benchmark(self) -> Dict[str, Any]:
        started = time.time()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.cfg.save_json(os.path.join(self.cfg.output_dir, "config.json"))
        cells = self._cells()
        expected_runs = len(self.cfg.seeds) * len(self.cfg.scenarios) * len(cells)
        print(
            f"[benchmark] starting: {len(self.cfg.seeds)} seeds × "
            f"{len(self.cfg.scenarios)} scenarios × {len(cells)} conditions "
            f"= {expected_runs} runs",
            flush=True,
        )
        runs: List[Dict[str, Any]] = []
        episodes: List[Dict[str, Any]] = []

        for scenario_id in self.cfg.scenarios:
            for seed in self.cfg.seeds:
                trajectory = generate_scenario_trajectory(
                    scenario_id=scenario_id,
                    seed=seed,
                    num_episodes=self.cfg.num_episodes,
                    perturb_episode=self.cfg.perturb_episode,
                    N=self.cfg.num_agents,
                    M=self.cfg.num_tasks,
                    d=self.cfg.dim,
                )
                trajectory_id = _trajectory_fingerprint(trajectory)
                target = trajectory[min(self.cfg.perturb_episode, len(trajectory) - 1)]
                reference_energy, reference_method, reference_valid, reference_timeout = (
                    self._ground_truth_reference(target)
                )
                reference_id = hashlib.sha256(
                    f"{scenario_id}|{seed}|{reference_energy:.12g}".encode()
                ).hexdigest()

                print(
                    f"[benchmark] prepared trajectory: scenario={scenario_id} "
                    f"seed={seed} reference={reference_energy:.6f} "
                    f"({reference_method})",
                    flush=True,
                )

                for solver_id, landscape_id, adaptation_mode in cells:
                    run_id = "run_{}_{}_{}_{}_{}".format(
                        _slug(scenario_id), seed, _slug(solver_id), _slug(adaptation_mode), self.cfg.experiment_id
                    )
                    summary, episode_records = self.run_trajectory(
                        scenario_id=scenario_id,
                        seed=seed,
                        solver_id=solver_id,
                        landscape_id=landscape_id,
                        adaptation_mode=adaptation_mode,
                        trajectory=trajectory,
                        reference_energy=reference_energy,
                        reference_method=reference_method,
                        reference_valid=reference_valid,
                    )
                    run = {
                        "run_id": run_id,
                        "experiment_id": self.cfg.experiment_id,
                        "problem_id": f"{scenario_id}_N{self.cfg.num_agents}_M{self.cfg.num_tasks}_seed{seed}",
                        "trajectory_id": trajectory_id,
                        "reference_id": reference_id,
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "solver_id": solver_id,
                        "landscape_id": landscape_id,
                        "adaptation_mode": adaptation_mode,
                        "N": self.cfg.num_agents,
                        "M": self.cfg.num_tasks,
                        "d": self.cfg.dim,
                        "num_episodes": self.cfg.num_episodes,
                        "perturb_episode": self.cfg.perturb_episode,
                        "evaluation_budget": self.cfg.max_energy_evaluations,
                        "max_energy_evaluations": self.cfg.max_energy_evaluations,
                        "ppee_10": summary.ppee_10,
                        "cumulative_excess_energy": summary.cumulative_excess_energy,
                        "cumulative_regret": summary.cumulative_regret,
                        "performance_drop": summary.perf_drop,
                        "perf_drop": summary.perf_drop,
                        "recovery_time": summary.recovery_time,
                        "mean_external_energy": summary.mean_external_energy,
                        "mean_internal_energy": summary.mean_internal_energy,
                        "final_energy": summary.final_energy,
                        "pre_base_energy": summary.pre_base_energy,
                        "post_base_energy": summary.post_base_energy,
                        "convergence": summary.convergence,
                        "stability": summary.stability,
                        "kappa_norm": summary.adaptation_magnitude_kappa,
                        "theta_norm": summary.adaptation_magnitude_theta,
                        "adaptation_magnitude_kappa": summary.adaptation_magnitude_kappa,
                        "adaptation_magnitude_theta": summary.adaptation_magnitude_theta,
                        "kappa_change_post": summary.kappa_change_post,
                        "theta_change_post": summary.theta_change_post,
                        "mean_co_assignment_conflicts": summary.mean_co_assignment_conflicts,
                        "mean_constraint_violations": summary.mean_constraint_violations,
                        "total_energy_evaluations": summary.total_energy_evaluations,
                        "total_runtime_sec": summary.total_runtime_sec,
                        "termination_reasons": summary.termination_reasons_summary,
                        "all_episodes_optimal": summary.all_episodes_optimal,
                        "fallback_used": summary.fallback_used,
                        "recovery_threshold": summary.recovery_threshold,
                        "reference_energy": reference_energy,
                        "reference_method": reference_method,
                        "reference_valid": reference_valid,
                        "reference_timeout": reference_timeout,
                        "recovery_window": summary.recovery_window,
                        "evaluation_landscape_id": summary.evaluation_landscape_id,
                        "git_commit": self.cfg.git_commit,
                        "benchmark_version": self.cfg.benchmark_version,
                    }
                    runs.append(run)
                    completed_runs = len(runs)
                    print(
                        f"[benchmark] [{completed_runs}/{expected_runs}] "
                        f"scenario={scenario_id} seed={seed} solver={solver_id} "
                        f"adaptation={adaptation_mode} ppee_10={summary.ppee_10:.6f} "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
                    for record in episode_records:
                        episodes.append(
                            {
                                "run_id": run_id,
                                "experiment_id": self.cfg.experiment_id,
                                "problem_id": run["problem_id"],
                                "trajectory_id": trajectory_id,
                                "reference_id": reference_id,
                                "scenario_id": scenario_id,
                                "seed": seed,
                                "solver_id": solver_id,
                                "landscape_id": landscape_id,
                                "adaptation_mode": adaptation_mode,
                                "N": self.cfg.num_agents,
                                "M": self.cfg.num_tasks,
                                "d": self.cfg.dim,
                                "total_episodes": self.cfg.num_episodes,
                                "perturb_episode": self.cfg.perturb_episode,
                                "episode": record.episode,
                                "evaluation_budget": self.cfg.max_energy_evaluations,
                                "reference_method": reference_method,
                                "reference_energy": reference_energy,
                                "reference_valid": reference_valid,
                                "internal_energy": record.internal_energy,
                                "external_energy": record.external_energy,
                                "energy_evaluations": record.energy_evaluations,
                                "accepted_moves": record.accepted_moves,
                                "iterations": record.iterations,
                                "runtime_sec": record.runtime_sec,
                                "termination_reason": record.termination_reason,
                                "solver_status": record.solver_status,
                                "optimal": record.is_optimal,
                                "mip_gap": record.mip_gap,
                                "timeout": record.timeout,
                                "fallback": record.fallback_used,
                                "reconfig_cost": record.reconfig_cost,
                                "co_assignment_conflicts": record.constraint_violations,
                                "coordination_score": record.coordination_score,
                                "load_balance": record.load_balance,
                                "kappa_norm": record.kappa_norm,
                                "theta_norm": record.theta_norm,
                                "theta_diff_norm": record.theta_diff_norm,
                                "delta_kappa_norm": record.delta_kappa_norm,
                                "delta_theta_norm": record.delta_theta_norm,
                                "git_commit": self.cfg.git_commit,
                            }
                        )

        runs_df = pd.DataFrame(runs)
        episodes_df = pd.DataFrame(episodes)
        runs_df.to_csv(os.path.join(self.cfg.output_dir, "runs.csv"), index=False)
        episodes_df.to_csv(os.path.join(self.cfg.output_dir, "episodes.csv"), index=False)

        summary_metrics = [
            "ppee_10",
            "cumulative_excess_energy",
            "performance_drop",
            "recovery_time",
            "kappa_change_post",
            "theta_change_post",
            "total_runtime_sec",
            "total_energy_evaluations",
        ]
        group_cols = ["scenario_id", "solver_id", "adaptation_mode"]
        summary_df = runs_df.groupby(group_cols, dropna=False).agg(
            ppee_10_mean=("ppee_10", "mean"),
            ppee_10_median=("ppee_10", "median"),
            ppee_10_std=("ppee_10", "std"),
            cumulative_excess_energy_mean=("cumulative_excess_energy", "mean"),
            cumulative_excess_energy_median=("cumulative_excess_energy", "median"),
            performance_drop_mean=("performance_drop", "mean"),
            recovery_time_mean=("recovery_time", "mean"),
            kappa_change_post_mean=("kappa_change_post", "mean"),
            theta_change_post_mean=("theta_change_post", "mean"),
            runtime_sec_mean=("total_runtime_sec", "mean"),
            energy_evaluations_mean=("total_energy_evaluations", "mean"),
            n=("ppee_10", "count"),
        ).reset_index()
        summary_df.to_csv(os.path.join(self.cfg.output_dir, "summary.csv"), index=False)

        stats_df = pd.DataFrame(self._compute_paired_statistics(runs_df))
        stats_df.to_csv(os.path.join(self.cfg.output_dir, "statistics.csv"), index=False)

        integrity = self._integrity_report(runs_df, episodes_df, cells)
        with open(os.path.join(self.cfg.output_dir, "integrity.json"), "w", encoding="utf-8") as handle:
            json.dump(integrity, handle, indent=2)

        metadata = self._build_metadata(time.time() - started)
        with open(os.path.join(self.cfg.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        self._write_results_readme(os.path.join(self.cfg.output_dir, "README.md"), integrity)
        return {
            "runs_df": runs_df,
            "episodes_df": episodes_df,
            "summary_df": summary_df,
            "stats_df": stats_df,
            "integrity": integrity,
            "output_dir": self.cfg.output_dir,
        }

    def _integrity_report(
        self, runs_df: pd.DataFrame, episodes_df: pd.DataFrame, cells: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        expected_runs = len(self.cfg.seeds) * len(self.cfg.scenarios) * len(cells)
        run_key = ["seed", "scenario_id", "solver_id", "adaptation_mode"]
        duplicates = int(runs_df.duplicated(run_key).sum()) if len(runs_df) else 0
        episode_counts = episodes_df.groupby("run_id")["episode"].nunique() if len(episodes_df) else pd.Series(dtype=int)
        expected_episode_set = set(range(self.cfg.num_episodes))
        missing_episode_runs = [
            run_id for run_id, count in episode_counts.items()
            if count != self.cfg.num_episodes
            or set(episodes_df.loc[episodes_df.run_id == run_id, "episode"]) != expected_episode_set
        ]
        primary = ["ppee_10", "cumulative_excess_energy", "reference_energy"]
        finite = bool(np.isfinite(runs_df[primary].to_numpy(dtype=float)).all()) if len(runs_df) else False
        references_shared = (
            runs_df.groupby(["scenario_id", "seed"])["reference_id"].nunique().max() <= 1
            if len(runs_df) else False
        )
        trajectories_shared = (
            runs_df.groupby(["scenario_id", "seed"])["trajectory_id"].nunique().max() <= 1
            if len(runs_df) else False
        )
        return {
            "expected_runs": expected_runs,
            "actual_runs": int(len(runs_df)),
            "run_count_ok": len(runs_df) == expected_runs,
            "duplicate_run_keys": duplicates,
            "duplicate_keys_ok": duplicates == 0,
            "episodes_per_run": self.cfg.num_episodes,
            "runs_with_missing_or_duplicate_episodes": missing_episode_runs,
            "episode_completeness_ok": not missing_episode_runs,
            "ppee_window": list(range(
                self.cfg.perturb_episode,
                min(self.cfg.perturb_episode + self.cfg.post_ppee_window, self.cfg.num_episodes),
            )),
            "references_shared": bool(references_shared),
            "trajectories_shared": bool(trajectories_shared),
            "finite_primary_values": finite,
            "invalid_reference_runs": int((~runs_df["reference_valid"].astype(bool)).sum()) if len(runs_df) else 0,
            "all_checks_pass": bool(
                len(runs_df) == expected_runs
                and duplicates == 0
                and not missing_episode_runs
                and finite
                and references_shared
                and trajectories_shared
                and (int((~runs_df["reference_valid"].astype(bool)).sum()) == 0 if len(runs_df) else False)
            ),
        }

    def _compute_paired_statistics(self, runs_df: pd.DataFrame) -> List[Dict[str, Any]]:
        families: Dict[str, List[Dict[str, Any]]] = {
            "primary": [], "secondary": [], "ablation": [], "interaction": []
        }
        pair_keys = ["problem_id", "scenario_id", "seed"]
        for scenario in self.cfg.scenarios:
            scoped = runs_df[runs_df.scenario_id == scenario]
            for solver in ["Simulated Annealing", "Energy Greedy"]:
                full = scoped[(scoped.solver_id == solver) & (scoped.adaptation_mode == "full")]
                static = scoped[(scoped.solver_id == solver) & (scoped.adaptation_mode == "static")]
                merged = pd.merge(full, static, on=pair_keys, suffixes=("_full", "_static"))
                if len(merged):
                    result = analyze_paired_comparison(
                        merged.ppee_10_full.tolist(), merged.ppee_10_static.tolist(),
                        "ppee_10", f"{solver}: Full - Static"
                    )
                    row = asdict(result) | {"scenario_id": scenario, "solver_id": solver, "hypothesis_family": "Family 1: PRIMARY"}
                    families["primary"].append(row)
                    for metric in ["cumulative_excess_energy", "performance_drop", "recovery_time"]:
                        result = analyze_paired_comparison(
                            merged[f"{metric}_full"].tolist(), merged[f"{metric}_static"].tolist(),
                            metric, f"{solver}: Full - Static"
                        )
                        families["secondary"].append(
                            asdict(result) | {"scenario_id": scenario, "solver_id": solver, "hypothesis_family": "Family 2: SECONDARY"}
                        )
                for mode in ["kappa-only", "theta-only"]:
                    ablation = scoped[(scoped.solver_id == solver) & (scoped.adaptation_mode == mode)]
                    merged = pd.merge(ablation, static, on=pair_keys, suffixes=("_abl", "_static"))
                    if len(merged):
                        for metric in ["ppee_10", "cumulative_excess_energy"]:
                            result = analyze_paired_comparison(
                                merged[f"{metric}_abl"].tolist(), merged[f"{metric}_static"].tolist(),
                                metric, f"{solver}: {mode} - Static"
                            )
                            families["ablation"].append(
                                asdict(result) | {"scenario_id": scenario, "solver_id": solver, "hypothesis_family": "Family 3: ABLATION"}
                            )

            sa_full = scoped[(scoped.solver_id == "Simulated Annealing") & (scoped.adaptation_mode == "full")]
            sa_static = scoped[(scoped.solver_id == "Simulated Annealing") & (scoped.adaptation_mode == "static")]
            gr_full = scoped[(scoped.solver_id == "Energy Greedy") & (scoped.adaptation_mode == "full")]
            gr_static = scoped[(scoped.solver_id == "Energy Greedy") & (scoped.adaptation_mode == "static")]
            sa = pd.merge(sa_full, sa_static, on=pair_keys, suffixes=("_full", "_static"))
            gr = pd.merge(gr_full, gr_static, on=pair_keys, suffixes=("_full", "_static"))
            both = pd.merge(sa, gr, on=pair_keys, suffixes=("_sa", "_greedy"))
            if len(both):
                for metric in ["ppee_10", "cumulative_excess_energy"]:
                    result = analyze_solver_interaction(
                        both[f"{metric}_full_sa"].tolist(), both[f"{metric}_static_sa"].tolist(),
                        both[f"{metric}_full_greedy"].tolist(), both[f"{metric}_static_greedy"].tolist(), metric
                    )
                    families["interaction"].append({
                        "metric": metric,
                        "comparison": "Solver interaction (SA vs Greedy)",
                        "scenario_id": scenario,
                        "solver_id": "Interaction",
                        "hypothesis_family": "Family 4: SOLVER INTERACTION",
                        "n_pairs": result.n_pairs,
                        "mean_adaptive": result.mean_sa_diff,
                        "mean_static": result.mean_greedy_diff,
                        "mean_difference": result.mean_interaction,
                        "median_difference": result.median_interaction,
                        "ci_95_lower": result.ci_95_lower,
                        "ci_95_upper": result.ci_95_upper,
                        "permutation_p_val": result.permutation_p_val,
                        "wilcoxon_p_val": 1.0,
                        "cohens_d": result.cohens_d,
                    })

        output: List[Dict[str, Any]] = []
        for name, rows in families.items():
            if rows:
                adjusted = holm_bonferroni_correction([row["permutation_p_val"] for row in rows])
                for row, value in zip(rows, adjusted):
                    row["p_val_adjusted"] = value
            output.extend(rows)
        return output

    def _build_metadata(self, runtime: float) -> Dict[str, Any]:
        return {
            "experiment_id": self.cfg.experiment_id,
            "benchmark_version": self.cfg.benchmark_version,
            "git_commit": self.cfg.git_commit,
            "git_dirty": _get_git_dirty(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "N": self.cfg.num_agents,
            "M": self.cfg.num_tasks,
            "d": self.cfg.dim,
            "episodes": self.cfg.num_episodes,
            "perturb_episode": self.cfg.perturb_episode,
            "evaluation_budget": self.cfg.max_energy_evaluations,
            "seeds": self.cfg.seeds,
            "scenarios": self.cfg.scenarios,
            "solvers": self.cfg.solvers,
            "adaptations": self.cfg.ablations,
            "reference_method": "ground_truth_exact_ilp_when_optimal",
            "post_ppee_window": self.cfg.post_ppee_window,
            "ppee_definition": "mean(max(0, external_energy - reference_energy)) over the first post-ppee_window episodes",
            "cee_definition": "sum(max(0, external_energy - reference_energy)) over all post-perturbation episodes",
            "current_problem_observed": True,
            "no_hidden_change_detection": True,
            "no_warm_start": True,
            "deterministic_initialization": self.cfg.initial_x_mode,
            "total_runtime_sec": runtime,
        }

    def _write_results_readme(self, path: str, integrity: Dict[str, Any]) -> None:
        content = f"""# Controlled Landscape × Solver Benchmark

Experiment identifier: `{self.cfg.experiment_id}`
Git commit: `{self.cfg.git_commit}`

The current problem instance is directly observed by the solver in every episode; adaptation is historical landscape information, not hidden change detection. Every episode starts from the same deterministic initialization policy and does not warm-start from the previous solution.

Primary metric: `ppee_10`, based exclusively on external ground-truth energy and the shared post-perturbation ILP reference. Lower energy is better; adaptive-minus-static differences below zero are improvements.

## Integrity checks

```json
{json.dumps(integrity, indent=2)}
```
"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
