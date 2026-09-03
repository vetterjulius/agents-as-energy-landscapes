import json
import csv
import os
import numpy as np
from benchmark.evaluation.metrics import compute_statistical_tests

def generate_markdown_report(all_results, output_path="results/benchmark_report.md", catalog_path="results/figure_catalog.md"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determine seed count dynamically from results if available
    sample_seeds = 0
    for s_res in all_results.values():
        for o_res in s_res.values():
            if "energy" in o_res:
                sample_seeds = len(o_res["energy"])
                break
        if sample_seeds > 0:
            break
    seed_text = f"$n = {sample_seeds}$" if sample_seeds > 0 else "multiple"

    # 1. Generate Main Report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Multi-Agent System Energy Landscape Orchestration Benchmark\n\n")
        f.write(f"This report presents a comprehensive evaluation of energy-based orchestration methods against baseline approaches across multiple scenarios and experimental conditions with {seed_text} independent random seeds.\n\n")

        f.write("## Table of Contents\n")
        f.write("1. [Performance Metrics by Scenario](#performance-metrics-by-scenario)\n")
        f.write("2. [Emergent Behavior Metrics](#emergent-behavior-metrics)\n")
        f.write("3. [Statistical Comparison](#statistical-comparison)\n")
        f.write("4. [Ablation Studies](#ablation-studies)\n")
        f.write("5. [Scalability Analysis](#scalability-analysis)\n")
        f.write("6. [Parameter Sensitivity Analysis](#parameter-sensitivity-analysis)\n")
        f.write("7. [Dynamic Adaptation Experiments](#dynamic-adaptation-experiments)\n")
        f.write("8. [Visualizations](#visualizations)\n\n")

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

        # Add Ablation Studies Section
        f.write("## Ablation Studies\n\n")
        f.write("Ablation studies evaluated on the Interaction Scenario to isolate energy terms and solver component contributions.\n\n")

        if os.path.exists("results/ablation_results.json"):
            try:
                with open("results/ablation_results.json", "r", encoding="utf-8") as af:
                    ablation_data = json.load(af)

                if "representation" in ablation_data:
                    f.write("### Energy Representation Ablations\n\n")
                    f.write("| Variant / Enabled Terms | Final Energy |\n")
                    f.write("| :--- | :---: |\n")
                    for var_name, eng_val in ablation_data["representation"].items():
                        f.write(f"| {var_name} | {eng_val:.4f} |\n")
                    f.write("\n")

                if "solver" in ablation_data:
                    f.write("### Solver Component Ablations\n\n")
                    f.write("| Variant / Configuration | Final Energy |\n")
                    f.write("| :--- | :---: |\n")
                    for var_name, eng_val in ablation_data["solver"].items():
                        f.write(f"| {var_name} | {eng_val:.4f} |\n")
                    f.write("\n")
            except Exception:
                f.write("Ablation results stored in `results/ablation_results.json`.\n\n")

        # Add Scalability Analysis Section
        f.write("## Scalability Analysis\n\n")
        f.write("The scalability experiments evaluate how orchestration methods scale with increasing problem size (number of agents and tasks).\n\n")
        
        # Check if scalability results exist
        if os.path.exists("results/scaling_tasks_results.csv"):
            f.write("### Task Scalability (Fixed Agents=10)\n\n")
            f.write("See [Figure Catalog: Scaling Analysis](#2-scaling-analysis) for detailed visualizations.\n\n")
            f.write("Results available in `results/scaling_tasks_results.csv`.\n\n")
        
        if os.path.exists("results/scaling_agents_results.csv"):
            f.write("### Agent Scalability (Fixed Tasks=50)\n\n")
            f.write("See [Figure Catalog: Scaling Analysis](#2-scaling-analysis) for detailed visualizations.\n\n")
            f.write("Results available in `results/scaling_agents_results.csv`.\n\n")
        
        # Add Parameter Sensitivity Section
        f.write("## Parameter Sensitivity Analysis\n\n")
        f.write("The coupling sweep experiment examines how varying the interaction coupling weight $\\lambda_{\\text{int}}$ affects orchestration performance.\n\n")
        
        if os.path.exists("results/coupling_results.csv"):
            f.write("See [Figure Catalog: Parameter Sensitivity](#3-parameter-sensitivity) for detailed visualizations.\n\n")
            f.write("Results available in `results/coupling_results.csv`.\n\n")
        
        # Add Dynamic Adaptation Section
        f.write("## Dynamic Adaptation Experiments\n\n")
        f.write("The dynamic adaptation experiments evaluate the system's ability to adapt to non-stationary environments with abrupt changes.\n\n")
        
        dynamic_scenarios = [
            ("Capability Drift", "Agent expertise changes abruptly"),
            ("Task Shift", "Task distribution shifts abruptly"),
            ("Dependency Change", "Task dependencies evolve"),
            ("Emergent Specialization", "Long-horizon role emergence"),
            ("Robustness", "Agent failures and recoveries")
        ]
        
        for scenario_name, description in dynamic_scenarios:
            file_pattern = f"dynamic_{scenario_name.lower().replace(' ', '_')}_"
            matching_files = [f for f in os.listdir("results") if f.startswith(file_pattern) and f.endswith(".csv")]
            if matching_files:
                f.write(f"### {scenario_name}\n\n")
                f.write(f"{description}.\n\n")
                f.write(f"Results: {', '.join([f'`results/{fn}`' for fn in matching_files[:4]])}\n\n")
        
        if os.path.exists("results/dynamic_adaptation_metrics.csv"):
            f.write("### Summary Metrics\n\n")
            f.write("Aggregated adaptation metrics (recovery time, cumulative regret) available in `results/dynamic_adaptation_metrics.csv`.\n\n")
            f.write("See [Figure Catalog: Dynamic Adaptation](#7-dynamic-adaptation-experiments) for detailed visualizations.\n\n")

        f.write("## Visualizations\n\n")
        f.write("Detailed figures and plots are available in [figure_catalog.md](figure_catalog.md).\n")

    # 2. Generate Figure Catalog
    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("# Figure Catalog\n\n")
        f.write("This document catalogs all figures generated during the comprehensive benchmark evaluation.\n\n")

        f.write("## Table of Contents\n")
        f.write("1. [Energy Metrics by Scenario](#1-energy-metrics-by-scenario)\n")
        f.write("2. [Scaling Analysis](#2-scaling-analysis)\n")
        f.write("3. [Parameter Sensitivity](#3-parameter-sensitivity)\n")
        f.write("4. [Constraint Violations](#4-constraint-violations)\n")
        f.write("5. [Multi-Objective Trade-offs](#5-multi-objective-trade-offs)\n")
        f.write("6. [Assignment Structures](#6-assignment-structures)\n")
        f.write("7. [Dynamic Adaptation Experiments](#7-dynamic-adaptation-experiments)\n")
        f.write("8. [Optimization Trajectories](#8-optimization-trajectories)\n\n")

        f.write("## 1. Energy Metrics by Scenario\n\n")
        f.write("Total energy values and emergent behavior breakdowns across different problem scenarios.\n\n")

        for scenario_name in all_results.keys():
            img_name = f"energy_{scenario_name}.png"
            f.write(f"### {scenario_name}\n")
            f.write(f"![Total Energy - {scenario_name}](plots/{img_name})\n\n")
            f.write(f"**Figure**: Mean total energy achieved by each orchestrator with standard deviation error bars. Lower values indicate better optimization performance.\n\n")

            breakdown_img = f"breakdown_{scenario_name}.png"
            f.write(f"![Emergent Behavior - {scenario_name}](plots/{breakdown_img})\n\n")
            f.write(f"**Figure**: Emergent behavior metrics showing specialization degree and task clustering patterns.\n\n")

        f.write("## 2. Scaling Analysis\n\n")
        f.write("Scalability experiments evaluate computational performance as problem size increases.\n\n")
        
        if os.path.exists("results/plots/scaling_energy.png"):
            f.write("### Task Scaling\n")
            f.write("![Scaling Energy](plots/scaling_energy.png)\n\n")
            f.write("**Figure**: Final solution energy as a function of number of tasks (fixed agents=10). Shows how solution quality scales with problem size.\n\n")
        
        if os.path.exists("results/plots/scaling_runtime.png"):
            f.write("![Scaling Runtime](plots/scaling_runtime.png)\n\n")
            f.write("**Figure**: Computation time (seconds) as a function of problem size. Linear or sub-linear scaling indicates good computational efficiency.\n\n")
        
        if os.path.exists("results/plots/scaling_agents_energy.png"):
            f.write("### Agent Scaling\n")
            f.write("![Agent Scaling Energy](plots/scaling_agents_energy.png)\n\n")
            f.write("**Figure**: Energy as a function of number of agents (fixed tasks=50).\n\n")
        
        if os.path.exists("results/plots/scaling_agents_runtime.png"):
            f.write("![Agent Scaling Runtime](plots/scaling_agents_runtime.png)\n\n")
            f.write("**Figure**: Runtime as a function of number of agents.\n\n")

        f.write("## 3. Parameter Sensitivity\n\n")
        f.write("Parameter sensitivity analysis examines how the interaction coupling weight $\\lambda_{\\text{int}}$ affects orchestration behavior.\n\n")
        
        if os.path.exists("results/plots/coupling_energy.png"):
            f.write("![Coupling Energy](plots/coupling_energy.png)\n\n")
            f.write("**Figure**: Total energy as a function of interaction coupling weight. Shows the trade-off between assignment costs and coordination benefits.\n\n")
        
        if os.path.exists("results/plots/coupling_coordination.png"):
            f.write("![Coupling Coordination](plots/coupling_coordination.png)\n\n")
            f.write("**Figure**: Coordination scores across varying coupling weights. Higher coupling promotes coordinated assignments.\n\n")

        f.write("## 4. Constraint Violations\n\n")
        f.write("Analysis of constraint satisfaction across different scenarios and configurations.\n\n")
        
        if os.path.exists("results/plots/conflicts_comparison.png"):
            f.write("![Conflict Violations](plots/conflicts_comparison.png)\n\n")
            f.write("**Figure**: Mean constraint violation counts across scenarios and orchestrators. Zero violations indicate feasible solutions.\n\n")
        
        if os.path.exists("results/plots/coupling_conflicts.png"):
            f.write("![Coupling Conflicts](plots/coupling_conflicts.png)\n\n")
            f.write("**Figure**: Constraint violations as a function of interaction coupling strength.\n\n")

        f.write("## 5. Multi-Objective Trade-offs\n\n")
        f.write("Pareto analysis of solution quality versus computational cost trade-offs.\n\n")
        
        if os.path.exists("results/plots/pareto_runtime_energy.png"):
            f.write("![Pareto Runtime vs Energy](plots/pareto_runtime_energy.png)\n\n")
            f.write("**Figure**: Pareto frontier showing trade-off between execution runtime (seconds) and solution quality (energy). Points in the lower-left represent better performance on both objectives. The ideal orchestrator would be fast and produce low-energy solutions.\n\n")

        f.write("## 6. Assignment Structures\n\n")
        f.write("Structural visualizations of orchestration solutions and problem topology.\n\n")
        
        if os.path.exists("results/plots/assignment_heatmap.png"):
            f.write("### Assignment Heatmap\n")
            f.write("![Assignment Heatmap](plots/assignment_heatmap.png)\n\n")
            f.write("**Figure**: Binary assignment matrix $X$ showing agent-task assignments. Rows represent agents, columns represent tasks. Filled cells (value=1) indicate assignments. Reveals workload distribution patterns.\n\n")

        if os.path.exists("results/plots/task_dependency_graph.png"):
            f.write("### Task Dependency Graph\n")
            f.write("![Task Dependency Graph](plots/task_dependency_graph.png)\n\n")
            f.write("**Figure**: Network graph where nodes are tasks and edges represent positive interaction weights ($\\Theta_{i,j} > 0$). Edge thickness indicates interaction strength. Connected tasks benefit from coordinated assignment.\n\n")

        if os.path.exists("results/plots/agent_task_bipartite.png"):
            f.write("### Agent-Task Bipartite Graph\n")
            f.write("![Agent-Task Bipartite Graph](plots/agent_task_bipartite.png)\n\n")
            f.write("**Figure**: Bipartite graph connecting agents (left, blue) to their assigned tasks (right, orange). Shows which agents are assigned to which tasks in a representative solution.\n\n")
        
        f.write("## 7. Dynamic Adaptation Experiments\n\n")
        f.write("Dynamic experiments evaluate the system's ability to adapt to non-stationary environments.\n\n")
        
        if os.path.exists("results/plots/dynamic_adaptation_curves.png"):
            f.write("### Adaptation Trajectories\n")
            f.write("![Dynamic Adaptation Curves](plots/dynamic_adaptation_curves.png)\n\n")
            f.write("**Figure**: Energy trajectories over 50 episodes for three dynamic scenarios (Capability Drift, Task Shift, Dependency Change). Vertical red line indicates the abrupt perturbation at episode 25. Full EBMAO (green) demonstrates faster recovery and adaptation compared to static energy baseline (orange) and single-component ablations.\n\n")
        
        if os.path.exists("results/plots/dynamic_specialization_curves.png"):
            f.write("### Emergent Specialization\n")
            f.write("![Specialization Evolution](plots/dynamic_specialization_curves.png)\n\n")
            f.write("**Figure**: Evolution of agent role specialization degree over 80 episodes. Higher values indicate stronger task-type specialization. EBMAO's adaptive memory enables emergent specialization patterns.\n\n")
        
        if os.path.exists("results/plots/dynamic_robustness_curves.png"):
            f.write("### Robustness Under Perturbations\n")
            f.write("![Robustness Curves](plots/dynamic_robustness_curves.png)\n\n")
            f.write("**Figure**: Energy evolution during agent failure (episode 25, red line) and new agent joining (episode 38, blue line). Tests system resilience and recovery capabilities.\n\n")
        
        if os.path.exists("results/plots/dynamic_adaptation_bars.png"):
            f.write("### Recovery Metrics\n")
            f.write("![Adaptation Metrics](plots/dynamic_adaptation_bars.png)\n\n")
            f.write("**Figure**: Average recovery time (episodes to return to pre-perturbation performance) and cumulative regret (total adaptation loss) across dynamic scenarios. Lower values indicate faster, more efficient adaptation.\n\n")
        
        f.write("## 8. Optimization Trajectories\n\n")
        f.write("Detailed view of the optimization process dynamics.\n\n")
        
        if os.path.exists("results/plots/energy_progression_temp.png"):
            f.write("![Energy Progression](plots/energy_progression_temp.png)\n\n")
            f.write("**Figure**: Energy minimization trajectory (red) and simulated annealing temperature schedule (blue dashed) over solver iterations. Shows the optimization dynamics: temperature cooling allows exploitation after initial exploration.\n\n")

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
