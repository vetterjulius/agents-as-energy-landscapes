import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from benchmark.config import config
from benchmark.scenarios.interaction import InteractionScenario
from benchmark.scenarios.independent import IndependentScenario
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.energy_based import EnergyBasedOrchestrator
from benchmark.evaluation.metrics import compute_energy

def run_sweep():
    print("Starting Weight Sweep...")

    sweep_cfg = config.get("sweep", {})
    scenario_name = sweep_cfg.get("scenario", "Interaction")

    if scenario_name == "Interaction":
        scenario = InteractionScenario()
    else:
        scenario = IndependentScenario()

    problem = scenario.generate(config["seed"])

    results = []

    i_weights = sweep_cfg.get("interaction_weights", [1.0])
    r_weights = sweep_cfg.get("risk_weights", [1.0])
    c_weights = sweep_cfg.get("cost_weights", [1.0])

    greedy = GreedyOrchestrator()
    print("Evaluating Greedy baseline...")
    X_greedy = greedy.solve(problem)
    # Note: compute_energy uses default weights (1.0).
    # For a fair comparison in the sweep, we might need to compute energy with the swept weights.

    import copy
    for iw in i_weights:
        for rw in r_weights:
            for cw in c_weights:
                print(f"  Sweep: interaction={iw}, risk={rw}, cost={cw}")

                # Update config for EnergyBasedOrchestrator
                # We need to make sure the orchestrator uses these weights internally
                current_cfg = copy.deepcopy(config)
                current_cfg["model"]["interaction_weight"] = iw
                current_cfg["model"]["risk_weight"] = rw
                current_cfg["model"]["cost_weight"] = cw
                current_cfg["solver"]["iterations"] = sweep_cfg.get("iterations", 100)

                eb = EnergyBasedOrchestrator(current_cfg)
                X_eb = eb.solve(problem)

                # Compute energy for both using the CURRENT weights
                E_eb, _ = compute_energy_with_weights(problem, X_eb, iw, rw, cw)
                E_greedy, _ = compute_energy_with_weights(problem, X_greedy, iw, rw, cw)

                # Ensure we store float values, not tensors
                results.append({
                    "interaction_weight": iw,
                    "risk_weight": rw,
                    "cost_weight": cw,
                    "energy_eb": float(E_eb),
                    "energy_greedy": float(E_greedy),
                    "diff": float(E_greedy - E_eb) # Positive means EB is better (lower energy)
                })

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/sweep_results.csv", index=False)

    plot_sweep(df)
    print("Sweep complete. Results saved in results/sweep_results.csv and results/plots/weight_sweep.png")

def compute_energy_with_weights(problem, X, iw, rw, cw):
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
    registry.add(CostEnergy(weight=cw))
    registry.add(RiskEnergy(risk_predictor, weight=rw))

    total, components = registry.compute(state)
    return total.item(), components

def plot_sweep(df):
    os.makedirs("results/plots", exist_ok=True)

    # Simple plot: Performance Diff vs Interaction Weight (assuming others are constant)
    # If multiple weights vary, this might need more complex logic (like multiple lines or heatmap)

    plt.figure(figsize=(10, 6))

    # Check which weight is varying
    varying = []
    if len(df["interaction_weight"].unique()) > 1: varying.append("interaction_weight")
    if len(df["risk_weight"].unique()) > 1: varying.append("risk_weight")
    if len(df["cost_weight"].unique()) > 1: varying.append("cost_weight")

    if len(varying) == 1:
        v = varying[0]
        plt.plot(df[v], df["diff"], marker='o')
        plt.xlabel(v)
        plt.ylabel("Energy Advantage (Greedy - EnergyBased)")
        plt.title(f"Energy Advantage vs {v}")
    else:
        # Just plot index for now if multiple vary
        plt.plot(df.index, df["diff"], marker='o')
        plt.xlabel("Sweep Index")
        plt.ylabel("Energy Advantage")
        plt.title("Weight Sweep Energy Advantage")

    plt.grid(True)
    plt.axhline(0, color='red', linestyle='--')
    plt.savefig("results/plots/weight_sweep.png")
    plt.close()

if __name__ == "__main__":
    run_sweep()
