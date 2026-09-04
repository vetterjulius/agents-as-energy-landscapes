import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.logging_config import setup_logger
from benchmark.runner import run_experiment, deep_merge_config
from benchmark.config import config
from dataset_pipeline.real_scenario import RealGitHubScenario
from benchmark.baselines.random import RandomOrchestrator
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.greedy_load_balancing import GreedyLoadBalancingOrchestrator
from benchmark.baselines.rule_based import RuleBasedOrchestrator
from benchmark.baselines.beam_search import BeamSearchOrchestrator
from benchmark.baselines.tabu_search import TabuSearchOrchestrator
from benchmark.baselines.auction import MarketAuctionOrchestrator
from benchmark.baselines.energy_based import EnergyHybridOrchestrator
from benchmark.baselines.ebmao_based import EBMAOHybridOrchestrator
from benchmark.evaluation.report import generate_markdown_report, save_csv_results
from benchmark.evaluation.plots import plot_results

def main():
    parser = argparse.ArgumentParser(description="Run Benchmark on Real GitHub Issues Dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Optional specific path to dataset JSON")
    args = parser.parse_args()

    logger = setup_logger(level=logging.INFO)
    logger.info("=" * 70)
    logger.info("Running Energy Landscape Benchmark on REAL GitHub Issues Datasets")
    logger.info("=" * 70)

    scenarios = {}
    
    # 1. Quick Validation Dataset (Small: 3 issues)
    quick_path = "dataset_pipeline/processed_gh_dataset.json"
    if os.path.exists(quick_path):
        scenarios["GitHub_Issues_Quick"] = RealGitHubScenario(quick_path)
    
    # 2. Comprehensive Publication Dataset (Large: 50 issues, 10 agents)
    large_path = "dataset_pipeline/processed_gh_dataset_large.json"
    if os.path.exists(large_path):
        scenarios["GitHub_Issues_Large"] = RealGitHubScenario(large_path)

    # 3. Custom path if provided
    if args.dataset and os.path.exists(args.dataset):
        scenarios["GitHub_Issues_Custom"] = RealGitHubScenario(args.dataset)

    if not scenarios:
        logger.error("No processed datasets found! Please run fetch_issues.py and process_issues.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(scenarios)} real-world scenarios: {list(scenarios.keys())}")

    # Prepare Orchestrators
    cfg = deep_merge_config(config, {})
    orchestrators = {
        "Random": RandomOrchestrator(),
        "Greedy": GreedyOrchestrator(),
        "GreedyLB": GreedyLoadBalancingOrchestrator(),
        "RuleBased": RuleBasedOrchestrator(),
        "Beam Search": BeamSearchOrchestrator(beam_width=5),
        "Tabu Search": TabuSearchOrchestrator(max_iterations=50, tabu_tenure=5),
        "Market Auction (SOTA)": MarketAuctionOrchestrator(alpha_load=0.5, beta_synergy=0.5),
        "Energy (Hybrid)": EnergyHybridOrchestrator(cfg),
        "EBMAO (Hybrid)": EBMAOHybridOrchestrator(cfg),
    }

    results = run_experiment(
        experiment_name="Real GitHub Issues Benchmark",
        scenarios=scenarios,
        orchestrators=orchestrators,
        base_seed=42,
        num_seeds=10,
        config=cfg
    )

    logger.info("Generating reports for Real GitHub Issues Benchmark...")
    os.makedirs("results", exist_ok=True)
    generate_markdown_report(results, output_path="results/real_github_benchmark_report.md")
    save_csv_results(results, output_path="results/real_github_benchmark_results.csv")
    plot_results(results)

    logger.info("=" * 70)
    logger.info("Real GitHub Issues Benchmark Completed Successfully!")
    logger.info("  Report saved to: results/real_github_benchmark_report.md")
    logger.info("  CSV saved to: results/real_github_benchmark_results.csv")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
