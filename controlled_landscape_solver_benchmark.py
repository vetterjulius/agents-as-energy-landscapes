#!/usr/bin/env python3
"""
Controlled Landscape x Solver Benchmark
=======================================
Primary scientifically controlled benchmark runner to test:
> Does an adaptive energy landscape improve multi-agent orchestration under non-stationarity,
> and is the effect independent of the optimization solver?

Usage:
    python controlled_landscape_solver_benchmark.py --mode quick
    python controlled_landscape_solver_benchmark.py --mode research
    python controlled_landscape_solver_benchmark.py --mode research --seeds 42,43,44,45
"""

import argparse
import sys

from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.runner import ControlledBenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Controlled Landscape x Solver Experimental Benchmark."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "research"],
        default="quick",
        help="Run mode: 'quick' (fast validation, 2 seeds, small instances) or 'research' (20 seeds, production instances).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of integer seeds, e.g. '42,43,44'. Overrides default mode seeds.",
    )
    parser.add_argument(
        "--eval-budget",
        type=int,
        default=None,
        help="Common maximum energy evaluation budget per episode (overrides default).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes per trajectory (overrides default).",
    )
    parser.add_argument(
        "--perturb-episode",
        type=int,
        default=None,
        help="Episode at which perturbation occurs (overrides default).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/controlled_landscape_solver",
        help="Directory where results (CSV, JSON, README) will be written.",
    )
    parser.add_argument(
        "--skip-ilp",
        action="store_true",
        help="Skip dynamic ILP runs even in quick mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.eval_budget is not None:
        overrides["max_energy_evaluations"] = args.eval_budget
    if args.episodes is not None:
        overrides["num_episodes"] = args.episodes
    if args.perturb_episode is not None:
        overrides["perturb_episode"] = args.perturb_episode
    if args.skip_ilp:
        overrides["run_ilp_dynamic"] = False

    if args.seeds is not None:
        overrides["seeds"] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.mode == "quick":
        config = BenchmarkConfig.quick_mode(**overrides)
    else:
        num_seeds = len(overrides["seeds"]) if "seeds" in overrides else 20
        config = BenchmarkConfig.research_mode(num_seeds=num_seeds, **overrides)

    runner = ControlledBenchmarkRunner(config)
    results = runner.run_benchmark()

    print("\nBenchmark completed successfully.")
    print(f"Total runs recorded: {len(results['runs_df'])}")
    print(f"Summary rows: {len(results['summary_df'])}")
    print(f"Statistical comparisons: {len(results['stats_df'])}")
    print(f"Artifacts saved in: {results['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
