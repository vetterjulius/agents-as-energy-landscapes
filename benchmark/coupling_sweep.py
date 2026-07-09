import torch
import copy
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from benchmark.config import config
from benchmark.scenarios.interaction import InteractionScenario
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.energy_based import EnergyPureSAOrchestrator, EnergyHybridOrchestrator, EnergyPureGreedyOrchestrator
from benchmark.evaluation.metrics import compute_energy, load_balance, coordination_score, constraint_violations

def run_coupling_sweep():
    print("Starting Coupling Sweep (Interaction Strength Sweep)...")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # We sweep over interaction weights as specified
    coupling_weights = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    scenario = InteractionScenario(num_agents=5, num_tasks=15, dim=8)
    problem = scenario.generate(seed)

    results = []

    # Initialize the orchestrators
    # We will adjust their internal configurations for each sweep iteration
    for iw in coupling_weights:
        print(f"  Coupling Sweep: interaction_weight={iw}")

        # Deep copy config and set the current interaction weight
        current_cfg = copy.deepcopy(config)
        if "model" not in current_cfg:
            current_cfg["model"] = {}
        current_cfg["model"]["interaction_weight"] = iw
        # Ensure a fast and thorough execution (e.g., 50 iterations)
        if "solver" not in current_cfg:
            current_cfg["solver"] = {}
        current_cfg["solver"]["iterations"] = 25

        orchestrators = {
            "Capability Matching (Greedy)": GreedyOrchestrator(),
            "Energy (Pure Greedy)": EnergyPureGreedyOrchestrator(current_cfg),
            "Energy (Pure SA)": EnergyPureSAOrchestrator(current_cfg),
            "Energy (Hybrid)": EnergyHybridOrchestrator(current_cfg)
        }

        for name, orchestrator in orchestrators.items():
            print(f"    Evaluating {name}...")
            start_time = time.perf_counter()
            X = orchestrator.solve(problem)
            elapsed = time.perf_counter() - start_time

            # Compute metrics using the CURRENT interaction weight
            energy, _ = compute_energy_with_weights(problem, X, iw)
            lb = load_balance(X)
            coord = coordination_score(problem, X)
            conflicts = constraint_violations(problem, X)

            results.append({
                "InteractionWeight": iw,
                "Orchestrator": name,
                "Energy": float(energy),
                "LoadBalance": float(lb),
                "Coordination": float(coord),
                "Conflicts": float(conflicts),
                "Runtime": float(elapsed)
            })

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/coupling_results.csv", index=False)
    print("Saved coupling sweep results to results/coupling_results.csv")

    plot_coupling_results(df)

def compute_energy_with_weights(problem, X, iw):
    from energy.registry import EnergyRegistry
    from energy.assignment import AssignmentEnergy
    from energy.interaction import InteractionEnergy
    from energy.cost import CostEnergy
    from energy.risk import RiskEnergy, RiskPredictor
    from state.orchestration_state import OrchestrationState

    N = len(problem.agents)
    M = len(problem.tasks)
    d = problem.agents[0].capability_embedding.shape[0]

    state = OrchestrationState(
        X=X,
        s=torch.stack([a.capability_embedding for a in problem.agents]),
        c=torch.stack([t.embedding for t in problem.tasks]),
        kappa=torch.zeros(N, d),
        Theta=problem.interaction_graph,
        C=problem.co_assignment_costs,
        N=N, M=M, d=d
    )

    risk_predictor = RiskPredictor(d, W_risk=problem.risk_weights)

    registry = EnergyRegistry()
    registry.add(AssignmentEnergy(lambda_align=config["model"].get("lambda_align", 0.5), weight=1.0))
    registry.add(InteractionEnergy(weight=iw))
    registry.add(CostEnergy(weight=config["model"].get("cost_weight", 1.0)))
    registry.add(RiskEnergy(risk_predictor, weight=config["model"].get("risk_weight", 1.0)))

    total, components = registry.compute(state)
    return total.item(), components

def plot_coupling_results(df):
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Energy vs Interaction Weight
    plt.figure(figsize=(10, 6))
    for name in df["Orchestrator"].unique():
        subset = df[df["Orchestrator"] == name]
        plt.plot(subset["InteractionWeight"], subset["Energy"], marker='o', label=name, linewidth=2)
    plt.xlabel("Interaction Weight (Strength of Coupling)")
    plt.ylabel("Total Energy (Lower is better)")
    plt.title("Coupling Sweep: Total Energy vs. Interaction Strength")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coupling_energy.png")
    plt.close()

    # 2. Coordination Score vs Interaction Weight
    plt.figure(figsize=(10, 6))
    for name in df["Orchestrator"].unique():
        subset = df[df["Orchestrator"] == name]
        plt.plot(subset["InteractionWeight"], subset["Coordination"], marker='s', label=name, linewidth=2)
    plt.xlabel("Interaction Weight (Strength of Coupling)")
    plt.ylabel("Coordination Score (Synergies Exploited)")
    plt.title("Coupling Sweep: Coordination Score vs. Interaction Strength")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coupling_coordination.png")
    plt.close()

    # 3. Conflicts vs Interaction Weight
    plt.figure(figsize=(10, 6))
    for name in df["Orchestrator"].unique():
        subset = df[df["Orchestrator"] == name]
        plt.plot(subset["InteractionWeight"], subset["Conflicts"], marker='x', label=name, linewidth=2)
    plt.xlabel("Interaction Weight (Strength of Coupling)")
    plt.ylabel("Constraint Violations (Conflicts)")
    plt.title("Coupling Sweep: Constraint Violations vs. Interaction Strength")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coupling_conflicts.png")
    plt.close()

    print(f"Generated coupling sweep plots in {output_dir}/")

if __name__ == "__main__":
    run_coupling_sweep()
