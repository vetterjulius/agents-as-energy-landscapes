from pathlib import Path

import random

import numpy as np
import pandas as pd
import torch

from benchmark.baselines.energy_based import (
    EnergyGuidedSAOrchestrator,
    EnergyHybridOrchestrator,
    EnergyPureGreedyOrchestrator,
    EnergyPureSAOrchestrator,
)
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.random import RandomOrchestrator
from benchmark.evaluation.metrics import brute_force_optimum, compute_energy
from benchmark.scenarios.frustrated import FrustratedScenario
from benchmark.scenarios.independent import IndependentScenario
from benchmark.scenarios.interaction import InteractionScenario


SCENARIOS = {
    "Independent": IndependentScenario(num_agents=3, num_tasks=6, dim=8),
    "Interaction": InteractionScenario(num_agents=3, num_tasks=6, dim=8),
    "Frustrated": FrustratedScenario(num_agents=3, num_tasks=6, dim=8),
}


def make_config(iterations):
    return {
        "energy": {
            "lambda_align": 0.5,
            "lambda_memory": 0.5,
            "interaction_weight": 1.0,
            "risk_weight": 1.0,
            "risk_scale": 1.0,
            "cost_weight": 1.0,
        },
        "ebmao": {
            "temperature_init": 2.0,
            "min_temperature": 0.1,
            "max_temperature": 5.0,
            "target_accept_rate": 0.3,
            "proposal_candidates": 4,
            "proposal_task_sample": 3,
            "agent_sample_size": 3,
            "block_move_size": 2,
            "warm_start_steps": 0,
            "local_refine_steps": 0,
            "hybrid_cleanup_prob": 0.0,
            "eta_theta": 0.1,
            "eta_memory": 0.05,
        },
        "iterations": iterations,
        "solver": {
            "iterations": iterations,
            "temperature_init": 2.0,
            "min_temperature": 0.1,
            "max_temperature": 5.0,
            "target_accept_rate": 0.3,
        },
    }


def build_methods(config):
    return {
        "Random": RandomOrchestrator(),
        "Capability Greedy": GreedyOrchestrator(),
        "Energy Pure Greedy": EnergyPureGreedyOrchestrator(config),
        "Energy Pure SA": EnergyPureSAOrchestrator(config),
        "Energy Guided SA": EnergyGuidedSAOrchestrator(config),
        "Energy Hybrid": EnergyHybridOrchestrator(config),
    }


def run_controlled_benchmark(
    seeds=tuple(range(10)),
    iterations=25,
    output_path="results/controlled_stationary_benchmark.csv",
):
    rows = []

    for scenario_name, scenario in SCENARIOS.items():
        for seed in seeds:
            problem = scenario.generate(seed)
            optimum = brute_force_optimum(problem)
            config = make_config(iterations)

            for method_name, method in build_methods(config).items():
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                assignment = method.solve(problem)
                energy, _ = compute_energy(problem, assignment)
                rows.append(
                    {
                        "scenario": scenario_name,
                        "seed": seed,
                        "method": method_name,
                        "iterations": iterations,
                        "energy": energy,
                        "optimum": optimum,
                        "absolute_gap": energy - optimum,
                        "relative_gap": (energy - optimum) / max(abs(optimum), 1e-8),
                    }
                )

    result = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    result = run_controlled_benchmark()
    summary = result.groupby(["scenario", "method"], as_index=False).agg(
        mean_energy=("energy", "mean"),
        mean_gap=("absolute_gap", "mean"),
        success_rate=("absolute_gap", lambda values: float(np.mean(np.isclose(values, 0.0, atol=1e-6)))),
    )
    print(summary.round(4).to_string(index=False))
