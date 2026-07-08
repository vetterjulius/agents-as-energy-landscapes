import torch
import random
import time
from benchmark.config import config
from benchmark.scenarios.independent import IndependentScenario
from benchmark.scenarios.interaction import InteractionScenario
from benchmark.scenarios.dynamic import DynamicScenario
from benchmark.scenarios.distribution_shift import DistributionShiftScenario

from benchmark.baselines.random import RandomOrchestrator
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.greedy_load_balancing import GreedyLoadBalancingOrchestrator
from benchmark.baselines.rule_based import RuleBasedOrchestrator
from benchmark.baselines.energy_based import EnergyBasedOrchestrator

from benchmark.evaluation.metrics import compute_energy, load_balance, coordination_score, constraint_violations
from benchmark.evaluation.report import generate_markdown_report, save_csv_results
from benchmark.evaluation.plots import plot_results
from benchmark.ablations import run_representation_ablations, run_solver_ablations

def run_benchmark():
    print("Starting Energy-Based Orchestration Benchmark (EOB)...")

    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)

    scenarios = {
        "Independent": IndependentScenario(),
        "Interaction": InteractionScenario(),
        "Dynamic": DynamicScenario(),
        "DistributionShift": DistributionShiftScenario()
    }

    orchestrators = {
        "Random": RandomOrchestrator(),
        "Greedy": GreedyOrchestrator(),
        "GreedyLB": GreedyLoadBalancingOrchestrator(),
        "RuleBased": RuleBasedOrchestrator(),
        "EnergyBased (SA)": EnergyBasedOrchestrator(config)
    }

    all_results = {}

    for s_name, scenario in scenarios.items():
        print(f"  Running Scenario: {s_name}")
        problem = scenario.generate(seed)
        scenario_results = {}

        for o_name, orchestrator in orchestrators.items():
            print(f"    Evaluating {o_name}...")
            start_time = time.perf_counter()
            X = orchestrator.solve(problem)
            elapsed = time.perf_counter() - start_time

            energy, _ = compute_energy(problem, X)
            lb = load_balance(X)
            coord = coordination_score(problem, X)
            conflicts = constraint_violations(problem, X)

            scenario_results[o_name] = {
                "energy": energy,
                "load_balance": lb,
                "coordination": coord,
                "conflicts": conflicts,
                "runtime": elapsed
            }

        all_results[s_name] = scenario_results

    # Run Ablations on Interaction Scenario
    print("  Running Ablations on Interaction Scenario...")
    interaction_problem = scenarios["Interaction"].generate(seed)
    rep_ablation_results = run_representation_ablations(interaction_problem, config)
    sol_ablation_results = run_solver_ablations(interaction_problem, config)

    # Reporting
    print("Generating reports and plots...")
    generate_markdown_report(all_results)
    save_csv_results(all_results)
    plot_results()

    # Print ablation summary to console
    print("\nRepresentation Ablation Results (Interaction Scenario):")
    for name, energy in rep_ablation_results.items():
        print(f"  {name}: {energy:.4f}")

    print("\nSolver Ablation Results (Interaction Scenario):")
    for name, energy in sol_ablation_results.items():
        print(f"  {name}: {energy:.4f}")

    print("\nBenchmark Complete. Results saved in results/")

if __name__ == "__main__":
    run_benchmark()
