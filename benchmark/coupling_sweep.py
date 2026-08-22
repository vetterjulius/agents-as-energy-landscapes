import copy
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import random

from benchmark.config import config
from benchmark.scenarios.interaction import InteractionScenario
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.energy_based import (
    EnergyPureSAOrchestrator,
    EnergyHybridOrchestrator,
    EnergyPureGreedyOrchestrator,
)
from benchmark.baselines.ebmao_based import (
    EBMAOPureSAOrchestrator,
    EBMAOHybridOrchestrator,
    EBMAOPureGreedyOrchestrator,
)
from benchmark.evaluation.metrics import (
    compute_energy,
    load_balance,
    coordination_score,
    constraint_violations,
)


def run_coupling_sweep():
    print("Starting Coupling Sweep (Interaction Strength Sweep)...")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    coupling_weights = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    scenario = InteractionScenario(
        num_agents=5,
        num_tasks=15,
        dim=8,
    )
    problem = scenario.generate(seed)

    results = []

    for iw in coupling_weights:
        print(f"  Coupling Sweep: interaction_weight={iw}")

        current_cfg = copy.deepcopy(config)

        if "model" not in current_cfg:
            current_cfg["model"] = {}

        current_cfg["model"]["interaction_weight"] = iw

        current_cfg["model"].update({
            "warm_start_steps": 0,
            "local_refine_steps": 0,
            "proposal_candidates": 2,
            "proposal_task_sample": 2,
            "agent_sample_size": 2,
            "block_move_size": 1,
            "hybrid_cleanup_prob": 0.0,
        })

        # The benchmark adapter uses training.iterations.
        current_cfg.setdefault("training", {})
        current_cfg["training"]["iterations"] = 25

        # Keep the legacy solver configuration in sync for the
        # classical Energy baselines that still consume it.
        current_cfg.setdefault("solver", {})
        current_cfg["solver"]["iterations"] = 25

        orchestrators = {
            "Capability Matching (Greedy)": GreedyOrchestrator(),
            "Energy (Pure Greedy)": EnergyPureGreedyOrchestrator(current_cfg),
            "Energy (Pure SA)": EnergyPureSAOrchestrator(current_cfg),
            "Energy (Hybrid)": EnergyHybridOrchestrator(current_cfg),
            "EBMAO (Pure Greedy)": EBMAOPureGreedyOrchestrator(current_cfg),
            "EBMAO (Pure SA)": EBMAOPureSAOrchestrator(current_cfg),
            "EBMAO (Hybrid)": EBMAOHybridOrchestrator(current_cfg),
        }

        for name, orchestrator in orchestrators.items():
            print(f"    Evaluating {name}...")

            start_time = time.perf_counter()

            torch.manual_seed(seed)
            random.seed(seed)
            X = orchestrator.solve(problem)

            elapsed = time.perf_counter() - start_time

            energy, _ = compute_energy(
                problem,
                X,
                interaction_weight=iw,
                lambda_align=current_cfg["model"].get("lambda_align", 0.5),
                cost_weight=current_cfg["model"].get("cost_weight", 1.0),
                risk_weight=current_cfg["model"].get("risk_weight", 1.0),
            )

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
                "Runtime": float(elapsed),
            })

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)

    df.to_csv(
        "results/coupling_results.csv",
        index=False,
    )

    print("Saved coupling sweep results to results/coupling_results.csv")

    plot_coupling_results(df)


def plot_coupling_results(df):
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Energy vs Interaction Weight
    plt.figure(figsize=(10, 6))

    for name in df["Orchestrator"].unique():
        subset = df[
            df["Orchestrator"] == name
        ]

        plt.plot(
            subset["InteractionWeight"],
            subset["Energy"],
            marker="o",
            label=name,
            linewidth=2,
        )

    plt.xlabel(
        "Interaction Weight (Strength of Coupling)"
    )
    plt.ylabel(
        "Total Energy (Lower is better)"
    )
    plt.title(
        "Coupling Sweep: Total Energy vs. Interaction Strength"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/coupling_energy.png"
    )
    plt.close()

    # 2. Coordination Score vs Interaction Weight
    plt.figure(figsize=(10, 6))

    for name in df["Orchestrator"].unique():
        subset = df[
            df["Orchestrator"] == name
        ]

        plt.plot(
            subset["InteractionWeight"],
            subset["Coordination"],
            marker="s",
            label=name,
            linewidth=2,
        )

    plt.xlabel(
        "Interaction Weight (Strength of Coupling)"
    )
    plt.ylabel(
        "Coordination Score (Synergies Exploited)"
    )
    plt.title(
        "Coupling Sweep: Coordination Score vs. Interaction Strength"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/coupling_coordination.png"
    )
    plt.close()

    # 3. Conflicts vs Interaction Weight
    plt.figure(figsize=(10, 6))

    for name in df["Orchestrator"].unique():
        subset = df[
            df["Orchestrator"] == name
        ]

        plt.plot(
            subset["InteractionWeight"],
            subset["Conflicts"],
            marker="x",
            label=name,
            linewidth=2,
        )

    plt.xlabel(
        "Interaction Weight (Strength of Coupling)"
    )
    plt.ylabel(
        "Constraint Violations (Conflicts)"
    )
    plt.title(
        "Coupling Sweep: Constraint Violations vs. Interaction Strength"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/coupling_conflicts.png"
    )
    plt.close()

    print(
        f"Generated coupling sweep plots in {output_dir}/"
    )


if __name__ == "__main__":
    run_coupling_sweep()