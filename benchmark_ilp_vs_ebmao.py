import time
import json
import os
import torch
import numpy as np

from benchmark.scenarios.interaction import InteractionScenario
from benchmark.scenarios.frustrated import FrustratedScenario
from benchmark.baselines.random import RandomOrchestrator
from benchmark.baselines.greedy import GreedyOrchestrator
from benchmark.baselines.ilp import ILPOrchestrator
from orchestrator.ebmao_orchestrator import EBMAOOrchestrator
from benchmark.evaluation.metrics import compute_energy


def make_ebmao_config(num_agents, num_tasks, iterations=50):
    return {
        "model": {
            "num_agents": num_agents,
            "num_tasks": num_tasks,
            "dim": 8,
            "lambda_align": 0.5,
            "lambda_memory": 0.5,
            "eta_theta": 0.1,
            "eta_memory": 0.1,
            "risk_weight": 1.0,
            "interaction_weight": 1.0,
            "cost_weight": 1.0,
            "temperature_init": 2.0,
            "min_temperature": 0.1,
            "max_temperature": 5.0,
            "target_accept_rate": 0.3,
            "search_mode": "hybrid",
        },
        "solver": {"iterations": iterations}
    }


def run_ilp_comparison(task_scales=(6, 12, 18, 24), seeds=tuple(range(3))):
    print("=" * 80)
    print("EBMAO Step 2: Ground Truth Optimality & Scalability vs Exact ILP Solver")
    print("=" * 80)

    num_agents = 4
    results = []

    for M in task_scales:
        print(f"\n>>> Problem Scale: N={num_agents} Agents, M={M} Tasks (Search Space: {num_agents}^{M} = {num_agents**M:.1e} assignments) <<<")

        scenario = FrustratedScenario(num_agents=num_agents, num_tasks=M, dim=8)

        for seed in seeds:
            problem = scenario.generate(seed)
            ebmao_cfg = make_ebmao_config(num_agents, M, iterations=60)

            methods = {
                "Random": lambda: RandomOrchestrator(),
                "Greedy": lambda: GreedyOrchestrator(),
                "EBMAO": lambda: EBMAOOrchestrator(ebmao_cfg),
                "Exact ILP": lambda: ILPOrchestrator(time_limit_sec=10.0),
            }

            for name, method_fn in methods.items():
                torch.manual_seed(seed)
                np.random.seed(seed)

                solver = method_fn()

                start_t = time.time()
                X = solver.solve(problem)
                solve_time = time.time() - start_t

                energy, _ = compute_energy(problem, X)

                rec = {
                    "num_agents": num_agents,
                    "num_tasks": M,
                    "seed": seed,
                    "method": name,
                    "energy": float(energy),
                    "solve_time_sec": round(solve_time, 4),
                }
                results.append(rec)

                print(f"  [{name:9s}] Energy: {energy:8.3f} | Solve Time: {solve_time*1000:7.2f} ms")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON ACROSS SCALES")
    print("=" * 80)
    
    summary = {}
    for M in task_scales:
        print(f"\nScale M={M} Tasks:")
        scale_recs = [r for r in results if r["num_tasks"] == M]
        ilp_energies = [r["energy"] for r in scale_recs if r["method"] == "Exact ILP"]
        ilp_mean_e = np.mean(ilp_energies) if ilp_energies else 1.0

        for method_name in ["Random", "Greedy", "EBMAO", "Exact ILP"]:
            m_recs = [r for r in scale_recs if r["method"] == method_name]
            mean_e = np.mean([r["energy"] for r in m_recs])
            mean_time = np.mean([r["solve_time_sec"] for r in m_recs])
            
            # Optimality ratio vs ILP (Ground Truth)
            gap = mean_e - ilp_mean_e
            print(f"  {method_name:9s} -> Avg Energy: {mean_e:8.3f} (Gap to ILP: {gap:+7.3f}) | Avg Time: {mean_time*1000:7.2f} ms")

    # Save to json
    os.makedirs("results", exist_ok=True)
    out_path = "results/ilp_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved ILP comparison results to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_ilp_comparison()
