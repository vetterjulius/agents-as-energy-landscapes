import torch
import random
import time
import copy
import numpy as np
from benchmark.config import config, quick_config
from benchmark.logging_config import get_logger
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
from benchmark.evaluation.comprehensive_report import generate_comprehensive_report, generate_experiment_summary
from benchmark.ablations import run_representation_ablations, run_solver_ablations, save_ablation_results
from benchmark.scale_sweep import run_scale_sweep
from benchmark.coupling_sweep import run_coupling_sweep
from benchmark.dynamic_benchmark import run_dynamic_benchmark

logger = get_logger("runner")

def deep_merge_config(base, overrides):
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
    torch.manual_seed(seed)
    random.seed(seed)

    perturbed = copy.deepcopy(problem)
    rob_cfg = cfg.get("robustness", {})

    if rob_cfg.get("capability_noise", {}).get("enabled", False):
        level = rob_cfg["capability_noise"]["level"]
        for agent in perturbed.agents:
            noise = torch.randn_like(agent.capability_embedding) * level
            agent.capability_embedding = agent.capability_embedding + noise

    if rob_cfg.get("risk_weights_noise", {}).get("enabled", False):
        level = rob_cfg["risk_weights_noise"]["level"]
        noise = torch.randn_like(perturbed.risk_weights) * level
        perturbed.risk_weights = perturbed.risk_weights + noise

    if rob_cfg.get("agent_failure", {}).get("enabled", False):
        rate = rob_cfg["agent_failure"]["rate"]
        num_failed = int(len(perturbed.agents) * rate)
        if 0 < num_failed < len(perturbed.agents):
            all_indices = list(range(len(perturbed.agents)))
            failed_indices = random.sample(all_indices, num_failed)
            perturbed.agents = [agent for idx, agent in enumerate(perturbed.agents) if idx not in failed_indices]

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
    logger.info("=" * 70)
    logger.info(f"Experiment: {experiment_name}")
    logger.info("=" * 70)

    all_results = {}

    for s_name, scenario in scenarios.items():
        logger.info(f"Scenario: {s_name} | Seeds: {num_seeds}")
        logger.debug(f"  Initializing result storage for {len(orchestrators)} orchestrators")
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
            logger.debug(f"  Seed {run_idx + 1}/{num_seeds} (seed={seed})")

            base_problem = scenario.generate(seed)
            problem = apply_robustness_perturbations(
                base_problem,
                seed,
                config,
            )

            for o_name, orchestrator in orchestrators.items():
                logger.debug(f"    Running {o_name}...")
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
                    
                    logger.debug(f"      → Energy={energy:.4f}, Coord={coord:.2f}, Conflicts={conf}, Time={elapsed:.4f}s")

                except Exception as e:
                    raise RuntimeError(
                        f"{experiment_name}: {o_name} failed "
                        f"on scenario {s_name}, seed {seed}"
                    ) from e

    return all_results

def run_benchmark(quick: bool = False):
    logger.info("=" * 70)
    logger.info("Energy-Based Orchestration Benchmark")
    logger.info("=" * 70)

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

    logger.info(f"Mode: {mode}")
    logger.info(f"Seeds per scenario: {num_seeds}")
    logger.info(f"Base random seed: {base_seed}")

    # ------------------------------------------------------------
    # Problem configuration
    # ------------------------------------------------------------
    problem_cfg = cfg.get("problem", {})

    dim = problem_cfg.get("dim", 8)
    n_agents = problem_cfg.get("num_agents", 5)
    n_tasks = problem_cfg.get("num_tasks", 10)

    logger.info(f"Problem configuration: Agents={n_agents}, Tasks={n_tasks}, Dim={dim}")

    exp1_iter = cfg.get("experiment_1", {}).get("iterations", 100)
    exp2_iter = cfg.get("experiment_2", {}).get("iterations", 100)
    logger.info(f"Solver iterations: Exp1={exp1_iter}, Exp2={exp2_iter}")

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
    logger.info("Running ablations on Interaction scenario")
    logger.debug("  Testing energy component contributions and solver variants")

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

    logger.debug("  Running representation ablations...")
    rep_ablation_results = run_representation_ablations(
        interaction_problem,
        ablation_cfg,
    )

    logger.debug("  Running solver ablations...")
    sol_ablation_results = run_solver_ablations(
        interaction_problem,
        ablation_cfg,
    )

    save_ablation_results(rep_ablation_results, sol_ablation_results)

    # Merge Experiment 2 (solver_results) into world_results so all orchestrators appear in reports
    consolidated_results = copy.deepcopy(world_results)
    for s_name, s_orch_map in solver_results.items():
        if s_name not in consolidated_results:
            consolidated_results[s_name] = {}
        for o_name, o_metrics in s_orch_map.items():
            if o_name not in consolidated_results[s_name]:
                consolidated_results[s_name][o_name] = o_metrics

    # ------------------------------------------------------------
    # Reporting - Initial Results
    # ------------------------------------------------------------
    logger.info("Generating initial reports for Experiments 1 & 2")
    logger.debug(f"  Processing {len(consolidated_results)} scenario results")

    generate_markdown_report(consolidated_results)
    save_csv_results(consolidated_results)
    plot_results(consolidated_results)

    # ------------------------------------------------------------
    # Ablation summary
    # ------------------------------------------------------------
    logger.info("Representation Ablation Results (Interaction Scenario):")

    for name, energy in rep_ablation_results.items():
        logger.info(f"  {name}: {energy:.4f}")

    logger.info("Solver Ablation Results (Interaction Scenario):")

    for name, energy in sol_ablation_results.items():
        logger.info(f"  {name}: {energy:.4f}")

    # ------------------------------------------------------------
    # Additional experiments
    # ------------------------------------------------------------
    if cfg.get("run_scale_sweep", True):
        logger.info("Running Scale Sweep Experiment")
        run_scale_sweep()
    else:
        logger.info("Skipping Scale Sweep Experiment")

    if cfg.get("run_coupling_sweep", True):
        logger.info("Running Coupling Sweep Experiment")
        run_coupling_sweep()
    else:
        logger.info("Skipping Coupling Sweep Experiment")

    if cfg.get("run_dynamic_benchmark", True):
        logger.info("Running Dynamic & Long-Horizon Adaptation Benchmark")
        run_dynamic_benchmark()
    else:
        logger.info("Skipping Dynamic & Long-Horizon Adaptation Benchmark")

    # ------------------------------------------------------------
    # Generate Comprehensive Final Report
    # ------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Generating Comprehensive Final Report")
    logger.info("=" * 70)
    
    # Combine all results for comprehensive report
    logger.info("Consolidating all experiment results...")
    generate_comprehensive_report(consolidated_results)
    generate_experiment_summary()
    
    # Final summary
    logger.info("=" * 70)
    logger.info("Benchmark Complete")
    logger.info("=" * 70)
    logger.info("Results saved to:")
    logger.info("  - results/benchmark_report.md (comprehensive markdown report)")
    logger.info("  - results/figure_catalog.md (detailed figure documentation)")
    logger.info("  - results/experiment_summary.md (execution summary)")
    logger.info("  - results/benchmark_results.csv (raw per-run data)")
    logger.info("  - results/benchmark_results_summary.csv (aggregated statistics)")
    logger.info("  - results/plots/ (all visualization figures)")
    
    # List additional experiment results
    logger.info("\nAdditional Experiment Results:")
    if cfg.get("run_scale_sweep", True):
        logger.info("  - results/scaling_tasks_results.csv")
        logger.info("  - results/scaling_agents_results.csv")
    if cfg.get("run_coupling_sweep", True):
        logger.info("  - results/coupling_results.csv")
    if cfg.get("run_dynamic_benchmark", True):
        logger.info("  - results/dynamic_adaptation_metrics.csv")
        logger.info("  - results/dynamic_*_*.csv (scenario-specific results)")
    
    logger.info("=" * 70)

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
