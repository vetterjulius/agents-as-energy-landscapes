import torch
import random
import time
import copy
import numpy as np
from benchmark.config import config, quick_config
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

def deep_merge_config(base, overrides):
    """
    Recursively merge overrides into a copy of base.

    Nested dictionaries are merged recursively.
    Lists and scalar values are replaced completely.
    """
    result = copy.deepcopy(base)

    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result

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

def run_experiment(
    experiment_name,
    scenarios,
    orchestrators,
    base_seed,
    num_seeds,
    config,
):
    print(f"\n{'=' * 70}")
    print(f"Starting experiment: {experiment_name}")
    print(f"{'=' * 70}")

    all_results = {}

    for s_name, scenario in scenarios.items():
        print(f"\n  Running Scenario: {s_name} ({num_seeds} seeds)")
        all_results[s_name] = {}

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
                "conflict_rate": [],
            }

        for run_idx in range(num_seeds):
            seed = base_seed + run_idx

            base_problem = scenario.generate(seed)
            problem = apply_robustness_perturbations(
                base_problem,
                seed,
                config,
            )

            for o_name, orchestrator in orchestrators.items():
                start_time = time.perf_counter()
                # Full reproducibility: seed all RNG sources
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)

                try:
                    X = orchestrator.solve(problem)
                    elapsed = time.perf_counter() - start_time

                    energy, _ = compute_energy(problem, X)
                    lb = load_balance(X)
                    coord = coordination_score(problem, X)
                    conf = constraint_violations(problem, X)
                    spec = specialization_degree(problem, X)
                    clust = task_clustering(problem, X)
                    comm = communication_cost(problem, X)
                    confr = conflict_rate(problem, X)

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
                    raise RuntimeError(
                        f"{experiment_name}: {o_name} failed "
                        f"on scenario {s_name}, seed {seed}"
                    ) from e

    return all_results

def run_benchmark(quick: bool = False):
    print("=" * 70)
    print("Energy-Based Orchestration Benchmark")
    print("=" * 70)

    # ------------------------------------------------------------
    # Select configuration
    # ------------------------------------------------------------
    if quick:
        cfg = deep_merge_config(config, quick_config)
        mode = "QUICK (validation only)"
    else:
        cfg = copy.deepcopy(config)
        mode = "FULL"

    # ------------------------------------------------------------
    # Global benchmark configuration
    # ------------------------------------------------------------
    base_seed = cfg.get("seed", 42)
    num_seeds = cfg.get("num_evaluation_seeds", 30)

    print(f"\nMode: {mode}")
    print(f"Seeds per scenario: {num_seeds}")
    print(f"Base random seed: {base_seed}")

    # ------------------------------------------------------------
    # Problem configuration
    # ------------------------------------------------------------
    problem_cfg = cfg.get("problem", {})

    dim = problem_cfg.get("dim", 8)
    n_agents = problem_cfg.get("num_agents", 5)
    n_tasks = problem_cfg.get("num_tasks", 10)

    print(f"\nProblem configuration:")
    print(f"  Agents: {n_agents}, Tasks: {n_tasks}, Embedding dim: {dim}")

    exp1_iter = cfg.get("experiment_1", {}).get("iterations", 100)
    exp2_iter = cfg.get("experiment_2", {}).get("iterations", 100)
    print(f"\nSolver iterations:")
    print(f"  Experiment 1 (World): {exp1_iter}")
    print(f"  Experiment 2 (Solver Battle): {exp2_iter}")

    # ------------------------------------------------------------
    # Instantiate scenarios
    # ------------------------------------------------------------
    scenarios = {
        "Independent": IndependentScenario(
            num_agents=n_agents,
            num_tasks=n_tasks,
            dim=dim,
        ),
        "Interaction": InteractionScenario(
            num_agents=n_agents,
            num_tasks=n_tasks,
            dim=dim,
        ),
        "Frustrated": FrustratedScenario(
            num_agents=n_agents,
            num_tasks=n_tasks,
            dim=dim,
        ),
        "Dynamic": DynamicScenario(
            num_agents=n_agents,
            num_tasks=n_tasks,
            dim=dim,
        ),
        "DistributionShift": DistributionShiftScenario(
            num_agents=n_agents,
            num_tasks=n_tasks,
            dim=dim,
        )
    }

    # ------------------------------------------------------------
    # Experiment 1
    #
    # Main comparison:
    #   classical baselines
    #   Energy
    #   EBMAO
    #
    # The experiment configuration is decided here and passed
    # explicitly to the proposed-system adapters.
    # ------------------------------------------------------------
    exp1_cfg = cfg.get("experiment_1", {})

    exp1_iterations = exp1_cfg.get("iterations", 100)
    exp1_solver = exp1_cfg.get("energy_solver", "hybrid")

    experiment_1_cfg = deep_merge_config(
        cfg,
        {
            "iterations": exp1_iterations,
            "solver": {},
        },
    )

    # ------------------------------------------------------------
    # Experiment 1 classical baseline parameters
    #
    # Beam/Tabu are configured globally under experiment_2 because
    # they are solver-comparison baselines.
    # ------------------------------------------------------------
    exp2_cfg = cfg.get("experiment_2", {})

    beam_cfg = exp2_cfg.get("beam_search", {})
    tabu_cfg = exp2_cfg.get("tabu_search", {})

    world_baselines = {
        "Random": RandomOrchestrator(),

        "Capability Matching (Greedy)": GreedyOrchestrator(),

        "GreedyLB": GreedyLoadBalancingOrchestrator(),

        "RuleBased": RuleBasedOrchestrator(),

        "Beam Search": BeamSearchOrchestrator(
            beam_width=beam_cfg.get("beam_width", 5),
        ),

        "Tabu Search": TabuSearchOrchestrator(
            max_iterations=tabu_cfg.get(
                "max_iterations",
                50,
            ),
            tabu_tenure=tabu_cfg.get(
                "tabu_tenure",
                5,
            ),
        ),

        "Energy (Hybrid)": EnergyHybridOrchestrator(
            experiment_1_cfg,
        ),

        "EBMAO (Hybrid)": EBMAOHybridOrchestrator(
            experiment_1_cfg,
        ),
    }

    # ------------------------------------------------------------
    # Experiment 2
    #
    # Solver battle on the same energy landscape.
    #
    # The runner explicitly selects the solver-specific config
    # and passes it to the corresponding adapter.
    # ------------------------------------------------------------
    exp2_iterations = exp2_cfg.get("iterations", 100)

    def build_solver_config(solver_name):
        solver_specific_cfg = exp2_cfg.get(
            solver_name,
            {},
        )

        return deep_merge_config(
            cfg,
            {
                "iterations": exp2_iterations,
                "solver": solver_specific_cfg,
            },
        )

    energy_greedy_cfg = build_solver_config(
        "energy_greedy",
    )

    energy_sa_cfg = build_solver_config(
        "energy_sa",
    )

    energy_hybrid_cfg = build_solver_config(
        "energy_hybrid",
    )

    # EBMAO uses the same solver-comparison structure.
    #
    # There are currently no separate EBMAO-specific sections in
    # config.py, so the shared experiment_2 iteration count is used.
    ebmao_greedy_cfg = build_solver_config(
        "energy_greedy",
    )

    ebmao_sa_cfg = build_solver_config(
        "energy_sa",
    )

    ebmao_hybrid_cfg = build_solver_config(
        "energy_hybrid",
    )

    energy_solver_battle = {
        "Energy (Pure Greedy)": EnergyPureGreedyOrchestrator(
            energy_greedy_cfg,
        ),

        "Energy (Pure SA)": EnergyPureSAOrchestrator(
            energy_sa_cfg,
        ),

        "Energy (Hybrid)": EnergyHybridOrchestrator(
            energy_hybrid_cfg,
        ),

        "EBMAO (Pure Greedy)": EBMAOPureGreedyOrchestrator(
            ebmao_greedy_cfg,
        ),

        "EBMAO (Pure SA)": EBMAOPureSAOrchestrator(
            ebmao_sa_cfg,
        ),

        "EBMAO (Hybrid)": EBMAOHybridOrchestrator(
            ebmao_hybrid_cfg,
        ),
    }

    # ------------------------------------------------------------
    # Run Experiment 1
    # ------------------------------------------------------------
    world_results = run_experiment(
        experiment_name="EOB: Energy/EBMAO vs Classical Baselines",
        scenarios=scenarios,
        orchestrators=world_baselines,
        base_seed=base_seed,
        num_seeds=num_seeds,
        config=cfg,
    )

    # ------------------------------------------------------------
    # Run Experiment 2
    # ------------------------------------------------------------
    solver_results = run_experiment(
        experiment_name="Solver Battle: Energy Landscape Solvers",
        scenarios=scenarios,
        orchestrators=energy_solver_battle,
        base_seed=base_seed,
        num_seeds=num_seeds,
        config=cfg,
    )

    # ------------------------------------------------------------
    # Ablations
    # ------------------------------------------------------------
    print("\n  Running Ablations on Interaction Scenario...")

    interaction_problem = scenarios["Interaction"].generate(
        base_seed,
    )

    ebmao_cfg = cfg.get("ebmao", {})
    ablation_cfg = deep_merge_config(
        cfg,
        {
            "solver": {
                "iterations": exp1_iterations,
                "temperature_init": ebmao_cfg.get("temperature_init", 4.0),
                "min_temperature": ebmao_cfg.get("min_temperature", 1.0),
                "max_temperature": ebmao_cfg.get("max_temperature", 6.0),
                "target_accept_rate": ebmao_cfg.get("target_accept_rate", 0.3),
            }
        },
    )

    rep_ablation_results = run_representation_ablations(
        interaction_problem,
        ablation_cfg,
    )

    sol_ablation_results = run_solver_ablations(
        interaction_problem,
        ablation_cfg,
    )

    # ------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------
    print("\nGenerating report for Experiment 1...")

    generate_markdown_report(world_results)
    save_csv_results(world_results)
    plot_results(world_results)

    print("\nGenerating report for Experiment 2...")

    generate_markdown_report(solver_results)
    save_csv_results(solver_results)
    plot_results(solver_results)

    # ------------------------------------------------------------
    # Ablation summary
    # ------------------------------------------------------------
    print("\nRepresentation Ablation Results (Interaction Scenario):")

    for name, energy in rep_ablation_results.items():
        print(f"  {name}: {energy:.4f}")

    print("\nSolver Ablation Results (Interaction Scenario):")

    for name, energy in sol_ablation_results.items():
        print(f"  {name}: {energy:.4f}")

    # ------------------------------------------------------------
    # Additional experiments
    # ------------------------------------------------------------
    if cfg.get("run_scale_sweep", True):
        print("\nRunning Scale Sweep Experiment...")
        run_scale_sweep()
    else:
        print("\nSkipping Scale Sweep Experiment.")

    if cfg.get("run_coupling_sweep", True):
        print("\nRunning Coupling Sweep Experiment...")
        run_coupling_sweep()
    else:
        print("\nSkipping Coupling Sweep Experiment.")

    if cfg.get("run_dynamic_benchmark", True):
        print("\nRunning Dynamic & Long-Horizon Adaptation Benchmark...")
        run_dynamic_benchmark()
    else:
        print("\nSkipping Dynamic & Long-Horizon Adaptation Benchmark.")

    # Final summary
    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    print("Results saved to:")
    print("  - results/benchmark_results.csv (raw per-run data)")
    print("  - results/benchmark_results_summary.csv (aggregated)")
    print("  - results/benchmark_report.md (markdown report)")
    print("  - results/figure_catalog.md (figure descriptions)")
    print("  - results/plots/ (all visualization figures)")
    print("  - results/ebmao_vs_world_20seeds.csv")
    print("  - results/recurrent_advantage_benchmark_20seeds_statistics.csv")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the Energy-Based Orchestration Benchmark suite."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation mode (2 seeds) instead of full mode (20 seeds)."
    )
    args = parser.parse_args()
    run_benchmark(quick=args.quick)
