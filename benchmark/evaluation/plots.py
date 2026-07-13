import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import networkx as nx
import seaborn as sns

from benchmark.config import config
from benchmark.scenarios.interaction import InteractionScenario
from benchmark.baselines.energy_based import EnergyHybridOrchestrator

def plot_results(all_results, output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Bar plots of Mean Energy per Scenario & Pareto Plots
    # -------------------------------------------------------------
    scenarios = list(all_results.keys())

    # Pareto Plot collection
    pareto_data = []

    for scenario in scenarios:
        orch_names = list(all_results[scenario].keys())
        mean_energies = []
        std_energies = []
        mean_runtimes = []
        orch_labels = []

        for o_name in orch_names:
            metrics = all_results[scenario][o_name]
            if len(metrics["energy"]) > 0:
                mean_energies.append(np.mean(metrics["energy"]))
                std_energies.append(np.std(metrics["energy"]))
                mean_runtimes.append(np.mean(metrics["runtime"]))
                orch_labels.append(o_name)

                # Collect for global Pareto view across main scenario (e.g. Interaction)
                if scenario == "Interaction":
                    pareto_data.append({
                        "name": o_name,
                        "energy": np.mean(metrics["energy"]),
                        "runtime": np.mean(metrics["runtime"])
                    })

        # Plot total Energy bar chart for this scenario
        plt.figure(figsize=(10, 6))
        plt.bar(orch_labels, mean_energies, yerr=std_energies, color='skyblue', capsize=5, edgecolor='black')
        plt.title(f"Mean Total Energy - {scenario} (with StdDev)", fontsize=14, fontweight='bold')
        plt.ylabel("Energy (Lower is better)", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_{scenario}.png")
        plt.close()

        # Plot Stacked Energy Breakdown (using first run's component distribution)
        # Since we have conflict_rate, load_balance, communication_cost, let's draw them
        components = ["Specialization", "Clustering", "Comm Cost", "Conflicts"]
        plt.figure(figsize=(10, 6))

        # We'll normalize and plot the emergent characteristics stacked
        spec_means = [np.mean(all_results[scenario][o]["specialization"]) for o in orch_labels]
        clust_means = [np.mean(all_results[scenario][o]["task_clustering"]) for o in orch_labels]
        comm_means = [np.mean(all_results[scenario][o]["communication_cost"]) for o in orch_labels]
        conf_means = [np.mean(all_results[scenario][o]["conflict_rate"]) for o in orch_labels]

        x_indices = np.arange(len(orch_labels))
        plt.bar(x_indices, spec_means, label="Specialization Degree", alpha=0.85)
        plt.bar(x_indices, clust_means, bottom=spec_means, label="Task Clustering", alpha=0.85)
        plt.xticks(x_indices, orch_labels, rotation=45, ha='right')
        plt.title(f"Emergent Behavior Breakdown - {scenario}", fontsize=14, fontweight='bold')
        plt.ylabel("Normalized Metrics", fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/breakdown_{scenario}.png")
        plt.close()

    # -------------------------------------------------------------
    # 2. Line Chart comparing Constraint Violations
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    all_orch_names = list(all_results[scenarios[0]].keys())

    for o_name in all_orch_names:
        scen_y = []
        for scenario in scenarios:
            if o_name in all_results[scenario] and len(all_results[scenario][o_name]["conflicts"]) > 0:
                scen_y.append(np.mean(all_results[scenario][o_name]["conflicts"]))
            else:
                scen_y.append(0.0)
        plt.plot(scenarios, scen_y, marker='o', label=o_name, linewidth=2)

    plt.title("Constraint Violations (Conflicts) Across Scenarios", fontsize=14, fontweight='bold')
    plt.ylabel("Conflicts (Lower is better)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/conflicts_comparison.png")
    plt.close()

    # -------------------------------------------------------------
    # 3. Pareto Plot (Energy vs. Runtime)
    # -------------------------------------------------------------
    if pareto_data:
        plt.figure(figsize=(10, 6))
        for item in pareto_data:
            plt.scatter(item["runtime"], item["energy"], s=150, label=item["name"], alpha=0.9, edgecolors='black')
        plt.xlabel("Execution Runtime (seconds)", fontsize=12)
        plt.ylabel("Total Energy (Lower is better)", fontsize=12)
        plt.title("Pareto-Plot: Execution Runtime vs. Total Energy (Interaction Scenario)", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pareto_runtime_energy.png")
        plt.close()

    # -------------------------------------------------------------
    # 4. Trajectory, Heatmaps and Graphs from a Representative Run
    # -------------------------------------------------------------
    print("  Generating structural curves, networks, and assignment heatmaps from representative run...")
    seed = config.get("seed", 42)
    rep_scenario = InteractionScenario(num_agents=5, num_tasks=15, dim=8)
    problem = rep_scenario.generate(seed)

    # Run Hybrid Orchestrator to capture step trajectory
    orch = EnergyHybridOrchestrator(config)
    X = orch.solve(problem)

    # A. Energy Progression & Temp Curves
    fig, ax1 = plt.subplots(figsize=(10, 6))

    steps = list(range(len(orch.energy_history)))
    color = 'tab:red'
    ax1.set_xlabel('Solver Iterations (Steps)', fontsize=12)
    ax1.set_ylabel('Total Landscape Energy', color=color, fontsize=12)
    ax1.plot(steps, orch.energy_history, color=color, linewidth=2.5, label="Total Energy")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Simulated Annealing Temperature', color=color, fontsize=12)
    ax2.plot(steps, orch.temp_history, color=color, linestyle='--', linewidth=2, label="Temperature")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Optimization Trajectory: Energy Minimization vs. SA Cooling Schedule', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{output_dir}/energy_progression_temp.png")
    plt.close()

    # B. Assignment Heatmap (X)
    plt.figure(figsize=(10, 5))
    X_np = X.cpu().numpy()
    sns.heatmap(X_np, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5, linecolor='gray',
                xticklabels=[f"Task {i}" for i in range(X_np.shape[1])],
                yticklabels=[f"Agent {j}" for j in range(X_np.shape[0])])
    plt.title("Orchestration Assignment Heatmap ($X$ matrix)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/assignment_heatmap.png")
    plt.close()

    # C. Task-Dependency Graph
    plt.figure(figsize=(10, 8))
    theta_np = problem.interaction_graph.cpu().numpy()
    G_tasks = nx.Graph()
    for i in range(theta_np.shape[0]):
        G_tasks.add_node(f"Task {i}")
        for j in range(i+1, theta_np.shape[1]):
            weight = theta_np[i, j]
            if weight > 0:
                G_tasks.add_edge(f"Task {i}", f"Task {j}", weight=weight)

    pos_tasks = nx.circular_layout(G_tasks)
    nx.draw_networkx_nodes(G_tasks, pos_tasks, node_color='lightgreen', node_size=600, edgecolors='black')
    nx.draw_networkx_labels(G_tasks, pos_tasks, font_size=10, font_weight='bold')

    edges = G_tasks.edges(data=True)
    if edges:
        weights = [d['weight'] * 2 for u, v, d in edges]
        nx.draw_networkx_edges(G_tasks, pos_tasks, width=weights, edge_color='gray', alpha=0.7)

    plt.title("Task Synergy & Dependency Network ($\Theta$ matrix)", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task_dependency_graph.png")
    plt.close()

    # D. Agent-Task Bipartite Graph
    plt.figure(figsize=(12, 8))
    G_bip = nx.Graph()

    agents_nodes = [f"Agent {j}" for j in range(X_np.shape[0])]
    tasks_nodes = [f"Task {i}" for i in range(X_np.shape[1])]

    G_bip.add_nodes_from(agents_nodes, bipartite=0)
    G_bip.add_nodes_from(tasks_nodes, bipartite=1)

    # Add assignment edges
    for j in range(X_np.shape[0]):
        for i in range(X_np.shape[1]):
            if X_np[j, i] > 0.5:
                G_bip.add_edge(f"Agent {j}", f"Task {i}")

    pos_bip = {}
    # Place Agents on left, Tasks on right
    pos_bip.update((node, (1, index)) for index, node in enumerate(agents_nodes))
    pos_bip.update((node, (2, index * (len(agents_nodes)/len(tasks_nodes)))) for index, node in enumerate(tasks_nodes))

    nx.draw_networkx_nodes(G_bip, pos_bip, nodelist=agents_nodes, node_color='skyblue', node_size=1200, edgecolors='black', label="Agents")
    nx.draw_networkx_nodes(G_bip, pos_bip, nodelist=tasks_nodes, node_color='orange', node_size=600, edgecolors='black', label="Tasks")
    nx.draw_networkx_labels(G_bip, pos_bip, font_size=9, font_weight='bold')
    nx.draw_networkx_edges(G_bip, pos_bip, width=2, edge_color='darkblue', alpha=0.8)

    plt.title("Bipartite Agent-Task Assignment Mapping Network", fontsize=14, fontweight='bold')
    plt.legend(loc="upper center", ncol=2, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/agent_task_bipartite.png")
    plt.close()
