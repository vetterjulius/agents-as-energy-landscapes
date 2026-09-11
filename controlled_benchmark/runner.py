from __future__ import annotations

import csv
import json
import os
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
    compute_constraint_violations,
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
    PairedAnalysisResult,
    SolverInteractionResult,
    analyze_paired_comparison,
    analyze_solver_interaction,
    holm_bonferroni_correction,
)
from landscape import Landscape, LandscapeState, ProblemContext


class ControlledBenchmarkRunner:
    """
    Primary Runner for the Controlled Landscape x Solver Experimental Benchmark.
    """

    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.sa_solver = EnergyAwareSimulatedAnnealingSolver(
            temperature_init=self.cfg.sa_temperature_init,
            min_temperature=self.cfg.sa_min_temperature,
            cooling_rate=self.cfg.sa_cooling_rate,
        )
        self.greedy_solver = EnergyAwareGreedySolver()
        self.ilp_solver = FixedLandscapeILPSolver(
            time_limit_sec=self.cfg.ilp_time_limit_sec
        )

    def _get_initial_assignment(self, N: int, M: int) -> torch.Tensor:
        """
        Generate identical deterministic initial assignment for all compared methods.
        Fairness: eliminates solver-specific initialization advantage.
        """
        X = torch.zeros(N, M, dtype=torch.float32)
        for t in range(M):
            X[t % N, t] = 1.0
        return X

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
    ) -> Tuple[TrajectorySummary, List[EpisodeRecord]]:
        """
        Execute one multi-episode run for a given method on a scenario trajectory.
        """
        N = self.cfg.num_agents
        M = self.cfg.num_tasks
        d = self.cfg.dim

        adaptation_mgr = EpisodeAdaptationManager(
            adaptation_mode=adaptation_mode,
            eta_memory=self.cfg.eta_memory,
            eta_theta=self.cfg.eta_theta,
        )

        current_landscape_state = make_initial_landscape_state(trajectory[0])
        records: List[EpisodeRecord] = []
        prev_X: Optional[torch.Tensor] = None

        for ep, problem_inst in enumerate(trajectory):
            # 1. Build immutable ProblemContext for current episode
            problem_ctx = problem_instance_to_problem_context(
                problem_inst,
                lambda_align=self.cfg.lambda_align,
                lambda_memory=self.cfg.lambda_memory,
                interaction_weight=self.cfg.interaction_weight,
                cost_weight=self.cfg.cost_weight,
                risk_weight=self.cfg.risk_weight,
            )

            # 2. Build explicit Landscape for current episode
            # Solver landscape uses current adaptation state
            solver_landscape = Landscape(
                problem=problem_ctx,
                state=current_landscape_state.clone(),
            )

            # Ground truth external landscape: kappa=0, Theta=env interaction graph
            ground_truth_state = LandscapeState(
                kappa=torch.zeros(N, d, dtype=torch.float32),
                Theta=problem_inst.interaction_graph.clone(),
            )
            ground_truth_landscape = Landscape(
                problem=problem_ctx,
                state=ground_truth_state,
            )

            # 3. Solver Optimization under Strict Budget
            initial_X = self._get_initial_assignment(N, M)
            budgeted_landscape = BudgetedLandscape(
                landscape=solver_landscape,
                max_evaluations=self.cfg.max_energy_evaluations,
            )

            solver_res: SolverResult
            if solver_id == "Simulated Annealing":
                # Ensure reproducibility of stochastic search
                solver_seed = seed * 10000 + ep
                solver_res = self.sa_solver.solve(
                    budgeted_landscape=budgeted_landscape,
                    initial_X=initial_X,
                    seed=solver_seed,
                )
            elif solver_id == "Energy Greedy":
                solver_res = self.greedy_solver.solve(
                    budgeted_landscape=budgeted_landscape,
                    initial_X=initial_X,
                )
            elif solver_id == "Exact ILP":
                solver_res = self.ilp_solver.solve(
                    landscape=solver_landscape,
                )
            else:
                raise ValueError(f"Unknown solver_id: {solver_id}")

            X_opt = solver_res.X.clone()

            # 4. Pure Evaluations
            internal_E = solver_landscape.evaluate(X_opt)
            external_E = ground_truth_landscape.evaluate(X_opt)

            reconfig = compute_reconfig_cost(prev_X, X_opt)
            conflicts = compute_constraint_violations(problem_ctx.C, X_opt)
            coordination = compute_coordination_score(problem_inst.interaction_graph, X_opt)
            load_bal = compute_load_balance(X_opt)

            kappa_norm = float(current_landscape_state.kappa.norm().item())
            theta_diff = (current_landscape_state.Theta - problem_inst.interaction_graph).norm().item()

            records.append(
                EpisodeRecord(
                    episode=ep,
                    internal_energy=internal_E,
                    external_energy=external_E,
                    energy_evaluations=solver_res.energy_evaluations,
                    accepted_moves=solver_res.accepted_moves,
                    iterations=solver_res.iterations,
                    runtime_sec=solver_res.runtime_sec,
                    termination_reason=solver_res.termination_reason,
                    solver_status=solver_res.status,
                    is_optimal=solver_res.is_optimal,
                    mip_gap=solver_res.mip_gap,
                    timeout=solver_res.timeout,
                    fallback_used=solver_res.fallback_used,
                    reconfig_cost=reconfig,
                    constraint_violations=conflicts,
                    coordination_score=coordination,
                    load_balance=load_bal,
                    kappa_norm=kappa_norm,
                    theta_diff_norm=theta_diff,
                )
            )

            # 5. Explicit Episode Boundary Adaptation
            # STRICTLY from current episode data only — no future trajectory access
            current_landscape_state = adaptation_mgr.step(
                current_state=current_landscape_state,
                X=X_opt,
                problem=problem_ctx,
            )

            prev_X = X_opt.clone()

        if reference_energy is None:
            target_ep = min(self.cfg.perturb_episode, len(trajectory) - 1)
            target_inst = trajectory[target_ep]
            gt_ctx = problem_instance_to_problem_context(
                target_inst,
                lambda_align=self.cfg.lambda_align,
                lambda_memory=self.cfg.lambda_memory,
                interaction_weight=self.cfg.interaction_weight,
                cost_weight=self.cfg.cost_weight,
                risk_weight=self.cfg.risk_weight,
            )
            gt_state = LandscapeState(
                kappa=torch.zeros(N, d, dtype=torch.float32),
                Theta=target_inst.interaction_graph.clone(),
            )
            gt_landscape = Landscape(problem=gt_ctx, state=gt_state)
            ref_sol = self.ilp_solver.solve(gt_landscape)
            reference_energy = ref_sol.energy
            reference_method = "exact_optimal" if ref_sol.is_optimal else "ilp_solver_fallback"

        summary = compute_trajectory_summary(
            records=records,
            scenario_id=scenario_id,
            perturb_episode=self.cfg.perturb_episode,
            pre_window=self.cfg.pre_window,
            post_window=self.cfg.post_window,
            reference_energy=reference_energy,
            reference_method=reference_method,
        )
        return summary, records

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete controlled experimental matrix across scenarios, solvers, and seeds.
        """
        start_benchmark_time = time.time()
        print("=" * 80)
        print(f"STARTING CONTROLLED LANDSCAPE x SOLVER BENCHMARK (Mode: {self.cfg.mode})")
        print(f"Seeds: {self.cfg.seeds} | Budget: {self.cfg.max_energy_evaluations} evals | Episodes: {self.cfg.num_episodes}")
        print("=" * 80)

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.cfg.save_json(os.path.join(self.cfg.output_dir, "config.json"))

        run_records: List[Dict[str, Any]] = []

        # Define experimental cells
        # Primary Matrix:
        #   (SA, Static), (SA, Adaptive Full)
        #   (Greedy, Static), (Greedy, Adaptive Full)
        #   (ILP, Static), (ILP, Adaptive Full) [if run_ilp_dynamic or small scale]
        # Ablations (for SA):
        #   (SA, kappa-only), (SA, theta-only)
        cells = [
            ("Simulated Annealing", "Static", "static"),
            ("Simulated Annealing", "Adaptive Full", "full"),
            ("Simulated Annealing", "kappa-only", "kappa-only"),
            ("Simulated Annealing", "theta-only", "theta-only"),
            ("Energy Greedy", "Static", "static"),
            ("Energy Greedy", "Adaptive Full", "full"),
        ]

        if self.cfg.run_ilp_dynamic:
            cells.extend([
                ("Exact ILP", "Static", "static"),
                ("Exact ILP", "Adaptive Full", "full"),
            ])

        total_runs = len(self.cfg.scenarios) * len(self.cfg.seeds) * len(cells)
        current_run_idx = 0

        for scenario_id in self.cfg.scenarios:
            print(f"\n>>> Scenario: {scenario_id} <<<")
            for seed in self.cfg.seeds:
                # 1. Pre-generate identical trajectory for this (scenario, seed)
                trajectory = generate_scenario_trajectory(
                    scenario_id=scenario_id,
                    seed=seed,
                    num_episodes=self.cfg.num_episodes,
                    perturb_episode=self.cfg.perturb_episode,
                    N=self.cfg.num_agents,
                    M=self.cfg.num_tasks,
                    d=self.cfg.dim,
                )

                problem_id = f"{scenario_id}_N{self.cfg.num_agents}_M{self.cfg.num_tasks}_seed{seed}"

                # Compute solver-independent reference energy ONCE per (scenario, seed)
                target_ep = min(self.cfg.perturb_episode, len(trajectory) - 1)
                target_inst = trajectory[target_ep]
                ref_ctx = problem_instance_to_problem_context(
                    target_inst,
                    lambda_align=self.cfg.lambda_align,
                    lambda_memory=self.cfg.lambda_memory,
                    interaction_weight=self.cfg.interaction_weight,
                    cost_weight=self.cfg.cost_weight,
                    risk_weight=self.cfg.risk_weight,
                )
                ref_state = LandscapeState(
                    kappa=torch.zeros(self.cfg.num_agents, self.cfg.dim, dtype=torch.float32),
                    Theta=target_inst.interaction_graph.clone(),
                )
                ref_landscape = Landscape(problem=ref_ctx, state=ref_state)
                ref_sol = self.ilp_solver.solve(ref_landscape)
                ref_energy = ref_sol.energy
                ref_method = "exact_optimal" if ref_sol.is_optimal else "ilp_solver_fallback"

                for solver_id, landscape_id, adapt_mode in cells:
                    current_run_idx += 1
                    run_id = f"run_{scenario_id[:4].lower()}_s{seed}_{solver_id[:2].lower()}_{adapt_mode[:4]}"

                    summary, ep_records = self.run_trajectory(
                        scenario_id=scenario_id,
                        seed=seed,
                        solver_id=solver_id,
                        landscape_id=landscape_id,
                        adaptation_mode=adapt_mode,
                        trajectory=trajectory,
                        reference_energy=ref_energy,
                        reference_method=ref_method,
                    )

                    rec = {
                        "run_id": run_id,
                        "problem_id": problem_id,
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "solver_id": solver_id,
                        "landscape_id": landscape_id,
                        "adaptation_mode": adapt_mode,
                        "N": self.cfg.num_agents,
                        "M": self.cfg.num_tasks,
                        "d": self.cfg.dim,
                        "num_episodes": self.cfg.num_episodes,
                        "perturb_episode": self.cfg.perturb_episode,
                        "evaluation_budget": self.cfg.max_energy_evaluations,
                        "max_energy_evaluations": self.cfg.max_energy_evaluations,
                        "recovery_time": summary.recovery_time,
                        "perf_drop": summary.perf_drop,
                        "cumulative_regret": summary.cumulative_regret,
                        "pre_base_energy": summary.pre_base_energy,
                        "post_base_energy": summary.post_base_energy,
                        "final_energy": summary.final_energy,
                        "mean_external_energy": summary.mean_external_energy,
                        "mean_internal_energy": summary.mean_internal_energy,
                        "convergence": summary.convergence,
                        "stability": summary.stability,
                        "adaptation_magnitude_kappa": summary.adaptation_magnitude_kappa,
                        "adaptation_magnitude_theta": summary.adaptation_magnitude_theta,
                        "mean_constraint_violations": summary.mean_constraint_violations,
                        "total_energy_evaluations": summary.total_energy_evaluations,
                        "total_runtime_sec": summary.total_runtime_sec,
                        "termination_reasons": summary.termination_reasons_summary,
                        "all_episodes_optimal": summary.all_episodes_optimal,
                        "fallback_used": summary.fallback_used,
                        "recovery_threshold": summary.recovery_threshold,
                        "reference_energy": summary.reference_energy,
                        "reference_method": summary.reference_method,
                        "recovery_window": summary.recovery_window,
                        "evaluation_landscape_id": summary.evaluation_landscape_id,
                        "git_commit": self.cfg.git_commit,
                        "benchmark_version": self.cfg.benchmark_version,
                    }
                    run_records.append(rec)

                    print(
                        f"  [{current_run_idx:03d}/{total_runs:03d}] "
                        f"Seed={seed:2d} | {solver_id:19s} | {landscape_id:13s} -> "
                        f"RecovTime: {summary.recovery_time:4.1f} | "
                        f"MeanExtE: {summary.mean_external_energy:7.3f} | "
                        f"Regret: {summary.cumulative_regret:7.2f} | "
                        f"Runtime: {summary.total_runtime_sec*1000:6.1f}ms"
                    )

        # Convert to DataFrame
        runs_df = pd.DataFrame(run_records)
        runs_csv_path = os.path.join(self.cfg.output_dir, "runs.csv")
        runs_df.to_csv(runs_csv_path, index=False)
        print(f"\nWrote runs table to {runs_csv_path}")

        # Compute Summary Table
        summary_cols = [
            "recovery_time",
            "perf_drop",
            "cumulative_regret",
            "pre_base_energy",
            "post_base_energy",
            "mean_external_energy",
            "convergence",
            "stability",
            "total_energy_evaluations",
            "total_runtime_sec",
        ]
        group_cols = ["scenario_id", "solver_id", "landscape_id", "adaptation_mode"]
        summary_df = runs_df.groupby(group_cols)[summary_cols].agg(["mean", "std", "count"]).reset_index()
        summary_csv_path = os.path.join(self.cfg.output_dir, "summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Wrote aggregated summary to {summary_csv_path}")

        # Compute Rigorous Paired Statistics
        stats_records = self._compute_paired_statistics(runs_df)
        stats_df = pd.DataFrame(stats_records)
        stats_csv_path = os.path.join(self.cfg.output_dir, "statistics.csv")
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"Wrote paired statistical tests to {stats_csv_path}")

        # Generate README.md in results directory
        readme_path = os.path.join(self.cfg.output_dir, "README.md")
        self._write_results_readme(readme_path, runs_df, stats_df, time.time() - start_benchmark_time)
        print(f"Wrote report to {readme_path}")

        print("\n" + "=" * 80)
        print("BENCHMARK RUN COMPLETED SUCCESSFULLY.")
        print("=" * 80)

        return {
            "runs_df": runs_df,
            "summary_df": summary_df,
            "stats_df": stats_df,
            "output_dir": self.cfg.output_dir,
        }

    def _compute_paired_statistics(self, runs_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Compute paired statistical tests strictly matched on (problem_id, scenario_id, seed).

        Hypothesis families (Holm-Bonferroni applied WITHIN each family):
          Family 1 — PRIMARY: Adaptive Full vs Static on recovery_time.
          Family 2 — SECONDARY: Adaptive Full vs Static on mean_external_energy, cumulative_regret.
          Family 3 — ABLATION: SA kappa-only/theta-only vs Static.
          Family 4 — SOLVER INTERACTION: Adaptive advantage SA vs Adaptive advantage Greedy.
        """
        # Collect rows per family
        family1: List[Dict[str, Any]] = []
        family2: List[Dict[str, Any]] = []
        family3: List[Dict[str, Any]] = []
        family4: List[Dict[str, Any]] = []

        for scenario_id in self.cfg.scenarios:
            scen_df = runs_df[runs_df["scenario_id"] == scenario_id]

            # --- Family 1 & 2: Primary matrix adaptive vs static ---
            for solver_id in ["Simulated Annealing", "Energy Greedy"]:
                adapt_df = scen_df[(scen_df["solver_id"] == solver_id) & (scen_df["adaptation_mode"] == "full")]
                stat_df = scen_df[(scen_df["solver_id"] == solver_id) & (scen_df["adaptation_mode"] == "static")]

                merged = pd.merge(adapt_df, stat_df, on="seed", suffixes=("_adapt", "_stat"))
                if len(merged) == 0:
                    continue

                # Family 1: PRIMARY — recovery_time
                a_vals = merged["recovery_time_adapt"].tolist()
                s_vals = merged["recovery_time_stat"].tolist()
                res = analyze_paired_comparison(a_vals, s_vals, "recovery_time", f"{solver_id}: Adaptive Full - Static")
                row = asdict(res)
                row["scenario_id"] = scenario_id
                row["solver_id"] = solver_id
                row["hypothesis_family"] = "Family 1: PRIMARY"
                family1.append(row)

                # Family 2: SECONDARY — mean_external_energy and cumulative_regret
                for metric in ["mean_external_energy", "cumulative_regret"]:
                    a_vals = merged[f"{metric}_adapt"].tolist()
                    s_vals = merged[f"{metric}_stat"].tolist()
                    res = analyze_paired_comparison(a_vals, s_vals, metric, f"{solver_id}: Adaptive Full - Static")
                    row = asdict(res)
                    row["scenario_id"] = scenario_id
                    row["solver_id"] = solver_id
                    row["hypothesis_family"] = "Family 2: SECONDARY"
                    family2.append(row)

            # --- Family 3: ABLATION — kappa-only and theta-only vs static for SA ---
            for abl_mode in ["kappa-only", "theta-only"]:
                abl_df = scen_df[(scen_df["solver_id"] == "Simulated Annealing") & (scen_df["adaptation_mode"] == abl_mode)]
                stat_df = scen_df[(scen_df["solver_id"] == "Simulated Annealing") & (scen_df["adaptation_mode"] == "static")]
                merged = pd.merge(abl_df, stat_df, on="seed", suffixes=("_abl", "_stat"))
                if len(merged) == 0:
                    continue

                for metric in ["recovery_time", "mean_external_energy", "cumulative_regret"]:
                    a_vals = merged[f"{metric}_abl"].tolist()
                    s_vals = merged[f"{metric}_stat"].tolist()
                    res = analyze_paired_comparison(a_vals, s_vals, metric, f"SA Ablation ({abl_mode}) - Static")
                    row = asdict(res)
                    row["scenario_id"] = scenario_id
                    row["solver_id"] = "Simulated Annealing"
                    row["hypothesis_family"] = "Family 3: ABLATION"
                    family3.append(row)

            # --- Family 4: SOLVER INTERACTION ---
            sa_adapt = scen_df[(scen_df["solver_id"] == "Simulated Annealing") & (scen_df["adaptation_mode"] == "full")]
            sa_stat = scen_df[(scen_df["solver_id"] == "Simulated Annealing") & (scen_df["adaptation_mode"] == "static")]
            gr_adapt = scen_df[(scen_df["solver_id"] == "Energy Greedy") & (scen_df["adaptation_mode"] == "full")]
            gr_stat = scen_df[(scen_df["solver_id"] == "Energy Greedy") & (scen_df["adaptation_mode"] == "static")]

            m_sa = pd.merge(sa_adapt, sa_stat, on="seed", suffixes=("_adapt", "_stat"))
            m_gr = pd.merge(gr_adapt, gr_stat, on="seed", suffixes=("_adapt", "_stat"))
            m_both = pd.merge(m_sa, m_gr, on="seed", suffixes=("_sa", "_gr"))

            if len(m_both) >= 2:
                for metric in ["recovery_time", "mean_external_energy", "cumulative_regret"]:
                    inter_res = analyze_solver_interaction(
                        sa_adaptive=m_both[f"{metric}_adapt_sa"].tolist(),
                        sa_static=m_both[f"{metric}_stat_sa"].tolist(),
                        greedy_adaptive=m_both[f"{metric}_adapt_gr"].tolist(),
                        greedy_static=m_both[f"{metric}_stat_gr"].tolist(),
                        metric_name=metric,
                    )
                    family4.append({
                        "metric": metric,
                        "comparison": "Solver Interaction (SA vs Greedy)",
                        "scenario_id": scenario_id,
                        "solver_id": "Interaction",
                        "hypothesis_family": "Family 4: SOLVER INTERACTION",
                        "n_pairs": inter_res.n_pairs,
                        "mean_adaptive": inter_res.mean_sa_diff,
                        "mean_static": inter_res.mean_greedy_diff,
                        "mean_difference": inter_res.mean_interaction,
                        "median_difference": inter_res.median_interaction,
                        "ci_95_lower": inter_res.ci_95_lower,
                        "ci_95_upper": inter_res.ci_95_upper,
                        "permutation_p_val": inter_res.permutation_p_val,
                        "wilcoxon_p_val": 1.0,
                        "cohens_d": inter_res.cohens_d,
                        "p_val_adjusted": None,
                    })

        # Apply Holm-Bonferroni WITHIN each family (not globally)
        def apply_holm_within_family(rows: List[Dict[str, Any]]) -> None:
            if not rows:
                return
            raw_p = [r["permutation_p_val"] for r in rows]
            adj_p = holm_bonferroni_correction(raw_p)
            for row, adj in zip(rows, adj_p):
                row["p_val_adjusted"] = adj

        apply_holm_within_family(family1)
        apply_holm_within_family(family2)
        apply_holm_within_family(family3)
        apply_holm_within_family(family4)

        return family1 + family2 + family3 + family4

    def _write_results_readme(
        self,
        filepath: str,
        runs_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        total_time_sec: float,
    ) -> None:
        """Generate comprehensive, user-facing markdown report."""
        content = f"""# Controlled Landscape × Solver Benchmark Results

**Benchmark Version:** {self.cfg.benchmark_version}  
**Git Commit:** `{self.cfg.git_commit}`  
**Mode:** {self.cfg.mode}  
**Execution Runtime:** {total_time_sec:.2f} seconds  

---

## 1. Experimental Setup & Fairness Controls

- **Evaluated Scenarios:** {", ".join(self.cfg.scenarios)}
- **Evaluated Solvers:** {", ".join(self.cfg.solvers)}
- **Evaluated Landscapes:** {", ".join(self.cfg.landscapes)}
- **Evaluated Ablations (SA):** {", ".join(self.cfg.ablations)}
- **Seeds ({len(self.cfg.seeds)}):** `{self.cfg.seeds}`
- **Common Budget:** `{self.cfg.max_energy_evaluations}` calls to `Landscape.evaluate(X)`
- **Problem Dimensions:** N={self.cfg.num_agents}, M={self.cfg.num_tasks}, d={self.cfg.dim}
- **Horizon & Perturbation:** {self.cfg.num_episodes} episodes, perturbation at episode {self.cfg.perturb_episode}
- **Initial Assignment:** `{self.cfg.initial_x_mode}` (identical starting assignment for all methods)
- **Primary Metric:** `recovery_time` (formally defined as episodes after perturbation until external energy <= target threshold)
- **Evaluation Landscape:** `external_ground_truth` (external real environment is source of truth for all recovery and performance metrics)

---

## 2. Paired Statistical Findings

The table below summarizes paired comparisons (Adaptive - Static) across matched seeds and trajectories:

| Scenario | Comparison | Metric | Mean Diff | 95% CI | Permutation p-val | Holm Adj p-val | Cohen's d |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for _, row in stats_df.iterrows():
            ci_str = f"[{row['ci_95_lower']:+.3f}, {row['ci_95_upper']:+.3f}]"
            adj_p_str = f"{row['p_val_adjusted']:.4f}" if row.get("p_val_adjusted") is not None else "N/A"
            content += (
                f"| {row['scenario_id']} | {row['comparison']} | {row['metric']} | "
                f"{row['mean_difference']:+.3f} | {ci_str} | {row['permutation_p_val']:.4f} | "
                f"{adj_p_str} | {row['cohens_d']:+.2f} |\n"
            )

        content += """
---

## 3. Scientific Interpretation & Hypotheses Testing

1. **Stationary Control Condition:**
   Evaluates whether unnecessary adaptation degrades performance in a stationary environment.
2. **Capability Drift:**
   Evaluates recovery speed and adaptation loss when agent capabilities abruptly change.
3. **Task Shift:**
   Evaluates adaptability under systematic shifts in task distribution.
4. **Dependency Change:**
   Evaluates adaptation to changing task synergy patterns.
5. **Solver Independence:**
   Tested by the Interaction term: whether the effect of the adaptive landscape differs significantly between Simulated Annealing and Energy Greedy.
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
