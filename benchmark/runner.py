import torch
import random
import time
import copy
import numpy as np
from benchmark.config import config
from benchmark.scenarios.independent import IndependentScenario
from benchmark.scenarios.interaction import InteractionScenario
from benchmark.scenarios.dynamic import DynamicScenario
from benchmark.scenarios.distribution_shift import DistributionShiftScenario
from benchmark.scenarios.frustrated import FrustratedScenario

from benchmark.baselines.random import RandomOrchestrator
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.greedy_load_balancing import GreedyLoadBalancingOrchestrator
from benchmark.baselines.rule_based import RuleBasedOrchestrator
from benchmark.baselines.beam_search import BeamSearchOrchestrator
from benchmark.baselines.tabu_search import TabuSearchOrchestrator
from benchmark.baselines.energy_based import (
    EnergyPureSAOrchestrator,
    EnergyHybridOrchestrator,
    EnergyPureGreedyOrchestrator
)
from benchmark.baselines.ebmao_based import (
    EBMAOPureSAOrchestrator,
    EBMAOHybridOrchestrator,
    EBMAOPureGreedyOrchestrator
)

from benchmark.evaluation.metrics import (
    compute_energy, load_balance, coordination_score, constraint_violations,
    specialization_degree, task_clustering, communication_cost, conflict_rate
)
from benchmark.evaluation.report import generate_markdown_report, save_csv_results
from benchmark.evaluation.plots import plot_results
from benchmark.ablations import run_representation_ablations, run_solver_ablations
from benchmark.scale_sweep import run_scale_sweep
from benchmark.coupling_sweep import run_coupling_sweep
from benchmark.dynamic_benchmark import run_dynamic_benchmark

def apply_robustness_perturbations(problem, seed, cfg):
    """
    Applies configurable robustness perturbations (noise, agent failures, comm outages)
    to a copy of the problem instance to test system resilience.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    perturbed = copy.deepcopy(problem)
    rob_cfg = cfg.get("robustness", {})

    # 1. Capability Noise
    if rob_cfg.get("capability_noise", {}).get("enabled", False):
        level = rob_cfg["capability_noise"]["level"]
        for agent in perturbed.agents:
            noise = torch.randn_like(agent.capability_embedding) * level
            agent.capability_embedding = agent.capability_embedding + noise

    # 2. Risk Weights Noise (Falsche Risikoschätzungen)
    if rob_cfg.get("risk_weights_noise", {}).get("enabled", False):
        level = rob_cfg["risk_weights_noise"]["level"]
        noise = torch.randn_like(perturbed.risk_weights) * level
        perturbed.risk_weights = perturbed.risk_weights + noise

    # 3. Agent Failure (Randomly drop a percentage of agents)
    if rob_cfg.get("agent_failure", {}).get("enabled", False):
        rate = rob_cfg["agent_failure"]["rate"]
        num_failed = int(len(perturbed.agents) * rate)
        if 0 < num_failed < len(perturbed.agents):
            all_indices = list(range(len(perturbed.agents)))
            failed_indices = random.sample(all_indices, num_failed)
            perturbed.agents = [agent for idx, agent in enumerate(perturbed.agents) if idx not in failed_indices]

    # 4. Communication Outages (Zero out some interaction connections)
    if rob_cfg.get("comm_outages", {}).get("enabled", False):
        rate = rob_cfg["comm_outages"]["rate"]
        mask = (torch.rand_like(perturbed.interaction_graph) > rate).float()
        perturbed.interaction_graph = perturbed.interaction_graph * mask

    return perturbed

def run_benchmark():
    print("Starting Paper-Ready Energy-Based Orchestration Benchmark (EOB)...")

    # Load base seed
    base_seed = config.get("seed", 42)
    num_seeds = config.get("num_evaluation_seeds", 30)
    print(f"Configured to run {num_seeds} seeds per scenario.")

    # Instantiate Scenarios (using config dimensions)
    dim = config.get("dim", 8)
    n_agents = config.get("num_agents", 5)
    n_tasks = config.get("num_tasks", 10)

    scenarios = {
        "Independent": IndependentScenario(num_agents=n_agents, num_tasks=n_tasks, dim=dim),
        "Interaction": InteractionScenario(num_agents=n_agents, num_tasks=n_tasks, dim=dim),
        "Dynamic": DynamicScenario(num_agents=n_agents, num_tasks=n_tasks, dim=dim),
        "DistributionShift": DistributionShiftScenario(dim=dim), # Custom size inside for scale shifts
        "Frustrated": FrustratedScenario(dim=dim)
    }

    # Instantiate Orchestrators (Baselines + Primary Energy Landscape methods)
    orchestrators = {
        # Standard Baselines
        "Random": RandomOrchestrator(),
        "Capability Matching (Greedy)": GreedyOrchestrator(),
        "GreedyLB": GreedyLoadBalancingOrchestrator(),
        "RuleBased": RuleBasedOrchestrator(),

        # New Classical Optimization Baselines
        "Beam Search": BeamSearchOrchestrator(beam_width=config.get("beam_search", {}).get("beam_width", 5)),
        "Tabu Search": TabuSearchOrchestrator(
            max_iterations=config.get("tabu_search", {}).get("max_iterations", 50),
            tabu_tenure=config.get("tabu_search", {}).get("tabu_tenure", 5)
        ),

        # Energy Landscape Solvers
        "Energy (Pure Greedy)": EnergyPureGreedyOrchestrator(config),
        "Energy (Pure SA)": EnergyPureSAOrchestrator(config),
        "Energy (Hybrid)": EnergyHybridOrchestrator(config),
        "EBMAO (Pure Greedy)": EBMAOPureGreedyOrchestrator(config),
        "EBMAO (Pure SA)": EBMAOPureSAOrchestrator(config),
        "EBMAO (Hybrid)": EBMAOHybridOrchestrator(config)
    }

    # Results structure to accumulate multi-seed metrics
    all_results = {}

    for s_name, scenario in scenarios.items():
        print(f"\n  Running Scenario: {s_name} ({num_seeds} seeds)")
        all_results[s_name] = {}

        # Initialize lists for each orchestrator
        for o_name in orchestrators.keys():
            all_results[s_name][o_name] = {
                "energy": [],
                "load_balance": [],
                "coordination": [],
                "conflicts": [],
                "runtime": [],
                "specialization": [],
                "task_clustering": [],
                "communication_cost": [],
                "conflict_rate": []
            }

        # Multi-seed evaluation loop
        for run_idx in range(num_seeds):
            seed = base_seed + run_idx

            # Generate unperturbed scenario problem instance
            base_problem = scenario.generate(seed)

            # Apply robustness perturbations if configured
            problem = apply_robustness_perturbations(base_problem, seed, config)

            for o_name, orchestrator in orchestrators.items():
                start_time = time.perf_counter()
                try:
                    X = orchestrator.solve(problem)
                    elapsed = time.perf_counter() - start_time

                    # Compute all metrics on the perturbed problem (surviving environment)
                    energy, _ = compute_energy(problem, X)
                    lb = load_balance(X)
                    coord = coordination_score(problem, X)
                    conf = constraint_violations(problem, X)

                    spec = specialization_degree(problem, X)
                    clust = task_clustering(problem, X)
                    comm = communication_cost(problem, X)
                    confr = conflict_rate(problem, X)

                    # Accumulate
                    metrics = all_results[s_name][o_name]
                    metrics["energy"].append(energy)
                    metrics["load_balance"].append(lb)
                    metrics["coordination"].append(coord)
                    metrics["conflicts"].append(conf)
                    metrics["runtime"].append(elapsed)
                    metrics["specialization"].append(spec)
                    metrics["task_clustering"].append(clust)
                    metrics["communication_cost"].append(comm)
                    metrics["conflict_rate"].append(confr)
                except Exception as e:
                    print(f"      [ERROR] {o_name} failed on seed {seed}: {e}")

    # Run Ablations on Interaction Scenario
    print("\n  Running Ablations on Interaction Scenario...")
    interaction_problem = scenarios["Interaction"].generate(base_seed)
    rep_ablation_results = run_representation_ablations(interaction_problem, config)
    sol_ablation_results = run_solver_ablations(interaction_problem, config)

    # Reporting and Plots
    print("\nGenerating comprehensive paper-ready reports and plots...")
    generate_markdown_report(all_results)
    save_csv_results(all_results)
    plot_results(all_results)

    # Console summary of ablations
    print("\nRepresentation Ablation Results (Interaction Scenario):")
    for name, energy in rep_ablation_results.items():
        print(f"  {name}: {energy:.4f}")

    print("\nSolver Ablation Results (Interaction Scenario):")
    for name, energy in sol_ablation_results.items():
        print(f"  {name}: {energy:.4f}")

    # Run Scaling Sweep
    print("\nRunning Scale Sweep Experiment...")
    run_scale_sweep()

    # Run Coupling Sweep
    print("\nRunning Coupling Sweep Experiment...")
    run_coupling_sweep()

    # Run Dynamic & Long-Horizon Learning Adaptation Benchmark (EBMAO specific properties)
    print("\nRunning Dynamic & Long-Horizon Adaptation Benchmark...")
    run_dynamic_benchmark()

    print("\nBenchmark Complete. Results and figures saved in results/")

if __name__ == "__main__":
    run_benchmark()
