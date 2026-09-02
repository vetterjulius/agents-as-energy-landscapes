import json
import csv
import os
import numpy as np
from benchmark.evaluation.metrics import compute_statistical_tests

def generate_markdown_report(all_results, output_path="results/benchmark_report.md", catalog_path="results/figure_catalog.md"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Generate Main Report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Multi-Agent System Energy Landscape Orchestration Benchmark\n\n")
        f.write("This report presents a comparative evaluation of energy-based orchestration methods against baseline approaches across multiple scenarios with $n \\geq 30$ independent random seeds.\n\n")

        f.write("## Table of Contents\n")
        f.write("1. [Performance Metrics by Scenario](#performance-metrics-by-scenario)\n")
        f.write("2. [Emergent Behavior Metrics](#emergent-behavior-metrics)\n")
        f.write("3. [Statistical Comparison](#statistical-comparison)\n")
        f.write("4. [Visualizations](#visualizations)\n\n")

        f.write("## Performance Metrics by Scenario\n\n")

        for scenario_name, scenario_results in all_results.items():
            f.write(f"### Scenario: {scenario_name}\n\n")

            # Table 1: Main Performance Metrics
            f.write("#### Performance Summary (Mean $\\pm$ Standard Deviation)\n\n")
            f.write("| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

            for orch_name, metrics in scenario_results.items():
                e_mean, e_std = np.mean(metrics["energy"]), np.std(metrics["energy"])
                lb_mean, lb_std = np.mean(metrics["load_balance"]), np.std(metrics["load_balance"])
                co_mean, co_std = np.mean(metrics["coordination"]), np.std(metrics["coordination"])
                conf_mean, conf_std = np.mean(metrics["conflicts"]), np.std(metrics["conflicts"])
                rt_mean, rt_std = np.mean(metrics["runtime"]), np.std(metrics["runtime"])

                f.write(f"| {orch_name} | {e_mean:.4f} $\\pm$ {e_std:.4f} | {lb_mean:.4f} $\\pm$ {lb_std:.4f} | {co_mean:.2f} $\\pm$ {co_std:.2f} | {conf_mean:.2f} $\\pm$ {conf_std:.2f} | {rt_mean:.4f} $\\pm$ {rt_std:.4f} |\n")
            f.write("\n")

            # Table 2: Emergent Behavior Metrics
            f.write("#### Emergent Behavior Metrics\n\n")
            f.write("| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")

            for orch_name, metrics in scenario_results.items():
                spec_mean, spec_std = np.mean(metrics["specialization"]), np.std(metrics["specialization"])
                clust_mean, clust_std = np.mean(metrics["task_clustering"]), np.std(metrics["task_clustering"])
                comm_mean, comm_std = np.mean(metrics["communication_cost"]), np.std(metrics["communication_cost"])
                confr_mean, confr_std = np.mean(metrics["conflict_rate"]), np.std(metrics["conflict_rate"])

                f.write(f"| {orch_name} | {spec_mean:.4f} $\\pm$ {spec_std:.4f} | {clust_mean:.4f} $\\pm$ {clust_std:.4f} | {comm_mean:.2f} $\\pm$ {comm_std:.2f} | {confr_mean:.2f} $\\pm$ {confr_std:.2f} |\n")
            f.write("\n")

            # Table 3: Statistical Significance Analysis
            f.write("#### Statistical Comparison with Best Baseline\n\n")
            f.write("Comparison of energy-based solvers against the baseline with lowest mean energy.\n\n")

            # Find best baseline
            baselines = ["Random", "Capability Matching (Greedy)", "GreedyLB", "RuleBased", "Beam Search", "Tabu Search"]
            best_base_name = None
            best_base_energy = float('inf')

            for b_name in baselines:
                if b_name in scenario_results and len(scenario_results[b_name]["energy"]) > 0:
                    mean_e = np.mean(scenario_results[b_name]["energy"])
                    if mean_e < best_base_energy:
                        best_base_energy = mean_e
                        best_base_name = b_name

            if best_base_name:
                f.write(f"**Reference Baseline**: {best_base_name} (Mean Energy: {best_base_energy:.4f})\n\n")
                f.write("| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% CI | Significant (p < 0.05) |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: |\n")

                energy_solvers = ["Energy (Pure Greedy)", "Energy (Pure SA)", "Energy (Hybrid)"]
                for solver_name in energy_solvers:
                    if solver_name in scenario_results and len(scenario_results[solver_name]["energy"]) > 0:
                        stats_res = compute_statistical_tests(
                            scenario_results[solver_name]["energy"],
                            scenario_results[best_base_name]["energy"]
                        )
                        p_t = stats_res["welch_p_val"]
                        p_u = stats_res["mann_whitney_p_val"]
                        ci = stats_res["ci_ref"]
                        sig = "Yes" if (p_t < 0.05 or p_u < 0.05) else "No"

                        f.write(f"| {solver_name} | {p_t:.2e} | {p_u:.2e} | [{ci[0]:.4f}, {ci[1]:.4f}] | {sig} |\n")
            else:
                f.write("No baseline methods evaluated in this scenario.\n")
            f.write("\n---\n\n")

        f.write("## Visualizations\n\n")
        f.write("Detailed figures and plots are available in [figure_catalog.md](figure_catalog.md).\n")

    # 2. Generate Figure Catalog
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("# Figure Catalog\n\n")
        f.write("This document catalogs all figures generated during the benchmark evaluation.\n\n")

        f.write("## Table of Contents\n")
        f.write("1. [Energy Metrics](#1-energy-metrics)\n")
        f.write("2. [Scaling Analysis](#2-scaling-analysis)\n")
        f.write("3. [Parameter Sensitivity](#3-parameter-sensitivity)\n")
        f.write("4. [Constraint Violations](#4-constraint-violations)\n")
        f.write("5. [Multi-Objective Trade-offs](#5-multi-objective-trade-offs)\n")
        f.write("6. [Assignment Structures](#6-assignment-structures)\n\n")

        f.write("## 1. Energy Metrics\n\n")
        f.write("Total energy values and component breakdowns across scenarios.\n\n")

        for scenario_name in all_results.keys():
            img_name = f"energy_{scenario_name}.png"
            f.write(f"### {scenario_name}\n")
            f.write(f"![Total Energy - {scenario_name}](plots/{img_name})\n\n")
            f.write(f"**Figure**: Total energy achieved by each orchestrator (lower values indicate lower energy).\n\n")

            breakdown_img = f"breakdown_{scenario_name}.png"
            f.write(f"![Energy Breakdown - {scenario_name}](plots/{breakdown_img})\n\n")
            f.write(f"**Figure**: Stacked bar chart showing energy component contributions (Assignment, Interaction, Cost, Risk).\n\n")

        f.write("## 2. Scaling Analysis\n\n")
        f.write("![Scaling Energy](plots/scaling_energy.png)\n\n")
        f.write("**Figure**: Final solution energy as a function of problem size (number of tasks: 20-500).\n\n")
        
        f.write("![Scaling Runtime](plots/scaling_runtime.png)\n\n")
        f.write("**Figure**: Computation time as a function of problem size.\n\n")

        f.write("## 3. Parameter Sensitivity\n\n")
        f.write("![Coupling Energy](plots/coupling_energy.png)\n\n")
        f.write("**Figure**: Energy values across varying interaction coupling weights $\\lambda_{\\text{int}}$.\n\n")
        
        f.write("![Coupling Coordination](plots/coupling_coordination.png)\n\n")
        f.write("**Figure**: Coordination scores across varying interaction coupling weights.\n\n")

        f.write("## 4. Constraint Violations\n\n")
        f.write("![Conflict Violations](plots/conflicts_comparison.png)\n\n")
        f.write("**Figure**: Constraint violation counts across scenarios and orchestrators.\n\n")
        
        f.write("![Coupling Conflicts](plots/coupling_conflicts.png)\n\n")
        f.write("**Figure**: Constraint violations as a function of interaction strength.\n\n")

        f.write("## 5. Multi-Objective Trade-offs\n\n")
        f.write("![Pareto Runtime vs Energy](plots/pareto_runtime_energy.png)\n\n")
        f.write("**Figure**: Scatter plot of mean runtime (Y-axis) vs. mean energy (X-axis). Points in the lower-left represent better performance on both objectives.\n\n")

        f.write("## 6. Assignment Structures\n\n")
        f.write("### Assignment Heatmap\n")
        f.write("![Assignment Heatmap](plots/assignment_heatmap.png)\n\n")
        f.write("**Figure**: Binary matrix showing agent-task assignments. Rows represent agents, columns represent tasks. Filled cells indicate assignments.\n\n")

        f.write("### Task Dependency Graph\n")
        f.write("![Task Dependency Graph](plots/task_dependency_graph.png)\n\n")
        f.write("**Figure**: Network graph where nodes are tasks and edges represent positive interaction weights ($\\Theta_{i,j} > 0$).\n\n")

        f.write("### Agent-Task Bipartite Graph\n")
        f.write("![Agent-Task Bipartite Graph](plots/agent_task_bipartite.png)\n\n")
        f.write("**Figure**: Bipartite graph connecting agents to their assigned tasks. Edge color intensity indicates capability similarity.\n\n")

def save_csv_results(all_results, output_path="results/benchmark_results.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Raw individual runs export
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "Scenario", "Orchestrator", "RunIndex", "Energy", "LoadBalance",
            "Coordination", "Conflicts", "Runtime", "Specialization",
            "TaskClustering", "CommunicationCost", "ConflictRate"
        ]
        writer.writerow(header)
        for scenario_name, scenario_results in all_results.items():
            for orch_name, metrics in scenario_results.items():
                num_runs = len(metrics["energy"])
                for run_idx in range(num_runs):
                    row = [
                        scenario_name, orch_name, run_idx,
                        metrics["energy"][run_idx],
                        metrics["load_balance"][run_idx],
                        metrics["coordination"][run_idx],
                        metrics["conflicts"][run_idx],
                        metrics["runtime"][run_idx],
                        metrics["specialization"][run_idx],
                        metrics["task_clustering"][run_idx],
                        metrics["communication_cost"][run_idx],
                        metrics["conflict_rate"][run_idx]
                    ]
                    writer.writerow(row)

    # Aggregated Summary export
    summary_path = "results/benchmark_results_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "Scenario", "Orchestrator",
            "Energy_Mean", "Energy_Std",
            "LoadBalance_Mean", "LoadBalance_Std",
            "Coordination_Mean", "Coordination_Std",
            "Conflicts_Mean", "Conflicts_Std",
            "Runtime_Mean", "Runtime_Std"
        ]
        writer.writerow(header)
        for scenario_name, scenario_results in all_results.items():
            for orch_name, metrics in scenario_results.items():
                writer.writerow([
                    scenario_name, orch_name,
                    np.mean(metrics["energy"]), np.std(metrics["energy"]),
                    np.mean(metrics["load_balance"]), np.std(metrics["load_balance"]),
                    np.mean(metrics["coordination"]), np.std(metrics["coordination"]),
                    np.mean(metrics["conflicts"]), np.std(metrics["conflicts"]),
                    np.mean(metrics["runtime"]), np.std(metrics["runtime"])
                ])
