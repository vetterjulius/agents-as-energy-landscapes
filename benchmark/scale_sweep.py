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

    N_agents = 10
    M_tasks_list = [10, 20, 30, 40, 50]

    # Create a custom config optimized for the scaling sweep to be fast and efficient
    scale_config = copy.deepcopy(config)
    if "solver" not in scale_config:
        scale_config["solver"] = {}
    scale_config["solver"]["iterations"] = 15  # 15 iterations is plenty to show the trend and runs fast

    orchestrators = {
        "Capability Matching (Greedy)": GreedyOrchestrator(),
        "Energy (Pure Greedy)": EnergyPureGreedyOrchestrator(scale_config),
        "Energy (Pure SA)": EnergyPureSAOrchestrator(scale_config),
        "Energy (Hybrid)": EnergyHybridOrchestrator(scale_config)
    }

    results = []

    for M in M_tasks_list:
        print(f"  Running Scale Sweep for N={N_agents}, Tasks M={M}...")

        # Create an interaction scenario of the given size
        scenario = InteractionScenario(num_agents=N_agents, num_tasks=M, dim=8)
        problem = scenario.generate(seed)

        for name, orchestrator in orchestrators.items():
            print(f"    Evaluating {name}...")
            start_time = time.perf_counter()
            X = orchestrator.solve(problem)
            elapsed = time.perf_counter() - start_time

            energy, _ = compute_energy(problem, X)
            lb = load_balance(X)
            coord = coordination_score(problem, X)
            conflicts = constraint_violations(problem, X)

            results.append({
                "Tasks": M,
                "Orchestrator": name,
                "Energy": float(energy),
                "LoadBalance": float(lb),
                "Coordination": float(coord),
                "Conflicts": float(conflicts),
                "Runtime": float(elapsed)
            })

    # Save results to CSV
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/scaling_results.csv", index=False)
    print("Saved scaling results to results/scaling_results.csv")

    # Generate plots
    plot_scale_results(df)

def plot_scale_results(df):
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Energy vs Tasks
    plt.figure(figsize=(10, 6))
    for name in df["Orchestrator"].unique():
        subset = df[df["Orchestrator"] == name]
        plt.plot(subset["Tasks"], subset["Energy"], marker='o', label=name, linewidth=2)
    plt.xlabel("Number of Tasks (M)")
    plt.ylabel("Total Energy (Lower is better)")
    plt.title("Scaling: Total Energy vs. Number of Tasks (N=10 Agents)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_energy.png")
    plt.close()

    # 2. Runtime vs Tasks
    plt.figure(figsize=(10, 6))
    for name in df["Orchestrator"].unique():
        subset = df[df["Orchestrator"] == name]
        plt.plot(subset["Tasks"], subset["Runtime"], marker='s', label=name, linewidth=2)
    plt.xlabel("Number of Tasks (M)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Scaling: Execution Runtime vs. Number of Tasks (N=10 Agents)")
    plt.yscale('log') # Log scale is helpful as complexity grows
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_runtime.png")
    plt.close()

    # 3. Conflicts vs Tasks
    plt.figure(figsize=(10, 6))
    for name in df["Orchestrator"].unique():
        subset = df[df["Orchestrator"] == name]
        plt.plot(subset["Tasks"], subset["Conflicts"], marker='x', label=name, linewidth=2)
    plt.xlabel("Number of Tasks (M)")
    plt.ylabel("Constraint Violations (Lower is better)")
    plt.title("Scaling: Constraint Violations vs. Number of Tasks (N=10 Agents)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_conflicts.png")
    plt.close()

    print(f"Generated scaling plots in {output_dir}/")

if __name__ == "__main__":
    run_scale_sweep()
