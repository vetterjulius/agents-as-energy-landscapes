import time
import copy
import torch
from typing import Dict, List, Any, Callable
from benchmarks.base import BaseBenchmark, BenchmarkScenario

class BenchmarkRunner:
    def __init__(self, benchmark: BaseBenchmark, base_cfg: Dict[str, Any]):
        """
        Coordinates running a benchmark suite against multiple orchestrators.
        
        Args:
            benchmark: The benchmark suite (BaseBenchmark instance).
            base_cfg: The baseline configuration dictionary (e.g. read from config.yaml).
        """
        self.benchmark = benchmark
        self.base_cfg = base_cfg

    def run(self, solvers: Dict[str, Callable], num_steps: int) -> Dict[str, Dict[str, Any]]:
        """
        Runs the benchmark evaluation.
        
        Args:
            solvers: Dictionary mapping solver names to factory functions.
                     The factory function takes (cfg, initial_state, W_risk) and returns the solver instance.
            num_steps: Number of steps/iterations to run each solver.
            
        Returns:
            A dictionary containing results for each scenario and solver.
        """
        scenarios = self.benchmark.load_scenarios()
        results = {}

        for scenario in scenarios:
            print(f"Running scenario: {scenario.name} ...")
            results[scenario.name] = {}
            
            # Prepare configuration for this scenario
            scenario_cfg = copy.deepcopy(self.base_cfg)
            scenario_cfg["model"]["num_agents"] = scenario.N
            scenario_cfg["model"]["num_tasks"] = scenario.M
            scenario_cfg["model"]["dim"] = scenario.d

            for solver_name, solver_factory in solvers.items():
                print(f"  Evaluating solver: {solver_name} ...")
                
                # Fresh initial state and predictor weights
                initial_state = scenario.to_state()
                W_risk = scenario.W_risk.clone()
                
                # Instantiate solver
                solver = solver_factory(scenario_cfg, initial_state, W_risk)
                
                # Record initial metrics
                init_energy = solver.total_energy().item()
                
                # Run steps and measure time
                start_time = time.perf_counter()
                for _ in range(num_steps):
                    solver.step()
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000.0
                
                # Record final metrics
                final_energy = solver.total_energy().item()
                energy_reduction = init_energy - final_energy
                energy_reduction_pct = (energy_reduction / (abs(init_energy) + 1e-8)) * 100.0
                
                # Success probability and risk metrics
                avg_success = 0.0
                if hasattr(solver, 'risk_predictor') and hasattr(solver, 'state'):
                    p = solver.risk_predictor.predict(solver.state)
                    # Average success probability of the actual assignments
                    avg_success = (solver.state.X * p).sum().item() / scenario.M
                
                # Workload distribution / Load balance
                # Compute standard deviation of tasks per agent
                tasks_per_agent = solver.state.X.sum(dim=1)
                load_balance_std = torch.std(tasks_per_agent).item()
                
                results[scenario.name][solver_name] = {
                    "initial_energy": init_energy,
                    "final_energy": final_energy,
                    "energy_reduction": energy_reduction,
                    "energy_reduction_pct": energy_reduction_pct,
                    "avg_success_prob": avg_success,
                    "avg_risk": 1.0 - avg_success,
                    "load_balance_std": load_balance_std,
                    "elapsed_ms": elapsed_ms
                }
                
        return results

    @staticmethod
    def format_results_markdown(results: Dict[str, Dict[str, Any]]) -> str:
        """Formats the evaluation results into a clean markdown report."""
        markdown = []
        markdown.append("# Benchmark Evaluation Results")
        markdown.append("\nThis report evaluates the performance of the Orchestrator system against the baselines.\n")
        
        for scenario_name, solvers_data in results.items():
            markdown.append(f"## Scenario: {scenario_name}")
            markdown.append("| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |")
            markdown.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            
            for solver_name, metrics in solvers_data.items():
                init_e = metrics["initial_energy"]
                final_e = metrics["final_energy"]
                reduction_pct = metrics["energy_reduction_pct"]
                success_p = metrics["avg_success_prob"]
                load_std = metrics["load_balance_std"]
                elapsed = metrics["elapsed_ms"]
                
                markdown.append(
                    f"| {solver_name} | {init_e:.4f} | {final_e:.4f} | {reduction_pct:.2f}% | "
                    f"{success_p:.4f} | {load_std:.2f} | {elapsed:.1f} |"
                )
            markdown.append("\n")
            
        return "\n".join(markdown)
