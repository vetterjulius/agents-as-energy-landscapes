import os
import sys
import yaml

# Add the current directory to python path if not already present
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system import load_config
from model.orchestrator import Orchestrator
from baselines.greedy import GreedyOrchestrator
from baselines.random_baseline import RandomOrchestrator
from benchmarks.synthetic import SyntheticBenchmark
from benchmarks.runner import BenchmarkRunner

def main():
    print("Loading config...")
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    
    # Define solvers to run
    # Each solver maps to a factory function: lambda cfg, init_state, W_risk: Solver(...)
    solvers = {
        "Simulated Annealing (SA)": lambda c, s, w: Orchestrator(c, initial_state=s, W_risk=w),
        "Greedy (Local Improvement)": lambda c, s, w: GreedyOrchestrator(c, initial_state=s, W_risk=w, mode="local_improvement"),
        "Greedy (Local Search)": lambda c, s, w: GreedyOrchestrator(c, initial_state=s, W_risk=w, mode="local_search"),
        "Greedy (Construction)": lambda c, s, w: GreedyOrchestrator(c, initial_state=s, W_risk=w, mode="construction"),
        "Random Baseline": lambda c, s, w: RandomOrchestrator(c, initial_state=s, W_risk=w)
    }

    # Initialize benchmark
    benchmark = SyntheticBenchmark(seed=cfg["training"]["seed"])
    runner = BenchmarkRunner(benchmark, cfg)
    
    # We will run for 100 steps for a quick but comprehensive benchmark
    num_steps = 100
    print(f"\nStarting benchmark with {num_steps} iterations per scenario...")
    
    results = runner.run(solvers, num_steps=num_steps)
    
    # Generate report
    report = runner.format_results_markdown(results)
    
    print("\n" + "=" * 50)
    print("BENCHMARK REPORT")
    print("=" * 50)
    print(report)
    print("=" * 50)
    
    
    # Fallback to local workspace if the brain directory doesn't exist
    with open("benchmark_results.md", "w") as f:
        f.write(report)
    print(f"\nArtifact saved to local: benchmark_results.md")

if __name__ == "__main__":
    main()
