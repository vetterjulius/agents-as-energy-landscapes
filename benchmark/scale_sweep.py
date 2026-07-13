import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy

from benchmark.config import config
from benchmark.scenarios.interaction import InteractionScenario
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.energy_based import EnergyPureSAOrchestrator, EnergyHybridOrchestrator, EnergyPureGreedyOrchestrator
from benchmark.evaluation.metrics import compute_energy, load_balance, coordination_score, constraint_violations

def run_scale_sweep():
    print("Starting Scale Sweep (Scaling Experiment)...")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Load scaling parameters from config
    agent_scales = config.get("scalability", {}).get("agents_scaling", [5, 10, 20, 50])
    task_scales = config.get("scalability", {}).get("tasks_scaling", [20, 50, 100, 500])
    sweep_iters = config.get("scalability", {}).get("sweep_iterations", 15)

    scale_config = copy.deepcopy(config)
    if "solver" not in scale_config:
        scale_config["solver"] = {}
    scale_config["solver"]["iterations"] = sweep_iters

    orchestrators = {
        "Capability Matching (Greedy)": GreedyOrchestrator(),
        "Energy (Pure Greedy)": EnergyPureGreedyOrchestrator(scale_config),
        "Energy (Pure SA)": EnergyPureSAOrchestrator(scale_config),
        "Energy (Hybrid)": EnergyHybridOrchestrator(scale_config)
    }

    results_tasks = []
    results_agents = []

    # 1. Scale Tasks (Fix Agent count = 10)
    fixed_agents = 10
    for M in task_scales:
        print(f"  Scaling Tasks Sweep: Agents={fixed_agents}, Tasks={M}...")
        scenario = InteractionScenario(num_agents=fixed_agents, num_tasks=M, dim=8)
        problem = scenario.generate(seed)

        for name, orchestrator in orchestrators.items():
            start_time = time.perf_counter()
            try:
                X = orchestrator.solve(problem)
                elapsed = time.perf_counter() - start_time
                energy, _ = compute_energy(problem, X)

                results_tasks.append({
                    "Tasks": M,
                    "Orchestrator": name,
                    "Energy": float(energy),
                    "Runtime": float(elapsed)
                })
            except Exception as e:
                print(f"    [ERROR] {name} failed on Tasks={M}: {e}")

    # 2. Scale Agents (Fix Task count = 50)
    fixed_tasks = 50
    for N in agent_scales:
        print(f"  Scaling Agents Sweep: Agents={N}, Tasks={fixed_tasks}...")
        scenario = InteractionScenario(num_agents=N, num_tasks=fixed_tasks, dim=8)
        problem = scenario.generate(seed)

        for name, orchestrator in orchestrators.items():
            start_time = time.perf_counter()
            try:
                X = orchestrator.solve(problem)
                elapsed = time.perf_counter() - start_time
                energy, _ = compute_energy(problem, X)

                results_agents.append({
                    "Agents": N,
                    "Orchestrator": name,
                    "Energy": float(energy),
                    "Runtime": float(elapsed)
                })
            except Exception as e:
                print(f"    [ERROR] {name} failed on Agents={N}: {e}")

    # Save to CSV
    df_tasks = pd.DataFrame(results_tasks)
    df_agents = pd.DataFrame(results_agents)
    os.makedirs("results", exist_ok=True)
    df_tasks.to_csv("results/scaling_tasks_results.csv", index=False)
    df_agents.to_csv("results/scaling_agents_results.csv", index=False)
    print("Saved scaling sweep results to results/scaling_tasks_results.csv and scaling_agents_results.csv")

    # Generate plots
    plot_scale_results(df_tasks, df_agents)

def plot_scale_results(df_tasks, df_agents):
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Task Scaling plots (Fix Agents=10)
    # A. Energy vs Tasks
    plt.figure(figsize=(10, 6))
    for name in df_tasks["Orchestrator"].unique():
        subset = df_tasks[df_tasks["Orchestrator"] == name]
        plt.plot(subset["Tasks"], subset["Energy"], marker='o', label=name, linewidth=2)
    plt.xlabel("Number of Tasks (M)")
    plt.ylabel("Total Energy (Lower is better)")
    plt.title("Scaling Tasks: Total Energy vs. Number of Tasks (N=10 Agents)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_energy.png")
    plt.close()

    # B. Runtime vs Tasks
    plt.figure(figsize=(10, 6))
    for name in df_tasks["Orchestrator"].unique():
        subset = df_tasks[df_tasks["Orchestrator"] == name]
        plt.plot(subset["Tasks"], subset["Runtime"], marker='s', label=name, linewidth=2)
    plt.xlabel("Number of Tasks (M)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Scaling Tasks: Execution Runtime vs. Number of Tasks (N=10 Agents)")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_runtime.png")
    plt.close()

    # 2. Agent Scaling plots (Fix Tasks=50)
    # A. Energy vs Agents
    plt.figure(figsize=(10, 6))
    for name in df_agents["Orchestrator"].unique():
        subset = df_agents[df_agents["Orchestrator"] == name]
        plt.plot(subset["Agents"], subset["Energy"], marker='o', label=name, linewidth=2)
    plt.xlabel("Number of Agents (N)")
    plt.ylabel("Total Energy (Lower is better)")
    plt.title("Scaling Agents: Total Energy vs. Number of Agents (M=50 Tasks)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_agents_energy.png")
    plt.close()

    # B. Runtime vs Agents
    plt.figure(figsize=(10, 6))
    for name in df_agents["Orchestrator"].unique():
        subset = df_agents[df_agents["Orchestrator"] == name]
        plt.plot(subset["Agents"], subset["Runtime"], marker='s', label=name, linewidth=2)
    plt.xlabel("Number of Agents (N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Scaling Agents: Execution Runtime vs. Number of Agents (M=50 Tasks)")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_agents_runtime.png")
    plt.close()

    print(f"Generated scaling plots in {output_dir}/")

if __name__ == "__main__":
    run_scale_sweep()
