import os
import time
import json
import torch
import numpy as np
import scipy.stats as stats
from urllib import request, error
from dotenv import load_dotenv

from benchmark.scenarios.independent import IndependentScenario
from benchmark.scenarios.interaction import InteractionScenario
from benchmark.scenarios.frustrated import FrustratedScenario
from benchmark.baselines.random import RandomOrchestrator
from benchmark.baselines.greedy import GreedyOrchestrator
from orchestrator.ebmao_orchestrator import EBMAOOrchestrator
from benchmark.evaluation.metrics import compute_energy

# Model configuration
DEFAULT_MODEL = "openrouter/free"
load_dotenv()

def call_openrouter_with_metrics(prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """Calls OpenRouter API and extracts token usage, latency, and response status."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing in .env")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    start_time = time.time()
    
    try:
        with request.urlopen(req) as resp:
            elapsed_time = time.time() - start_time
            res_data = json.loads(resp.read().decode("utf-8"))
            
            content = res_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = res_data.get("usage", {})
            
            prompt_tokens = usage.get("prompt_tokens", len(prompt) // 4)
            completion_tokens = usage.get("completion_tokens", len(content) // 4)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            return {
                "success": True,
                "content": content,
                "latency_sec": elapsed_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "error": None
            }
    except error.HTTPError as e:
        elapsed_time = time.time() - start_time
        err_body = e.read().decode("utf-8") if e.fp else ""
        return {
            "success": False,
            "content": "",
            "latency_sec": elapsed_time,
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": 0,
            "total_tokens": len(prompt) // 4,
            "error": f"HTTP {e.code}: {err_body}"
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "content": "",
            "latency_sec": elapsed_time,
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": 0,
            "total_tokens": len(prompt) // 4,
            "error": str(e)
        }


def make_ebmao_config(iterations=20):
    return {
        "model": {
            "num_agents": 3,
            "num_tasks": 5,
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


def run_validation_suite(seeds=tuple(range(3)), delay_sec=1.5):
    print("=" * 80)
    print("EBMAO Step 1: Empirical Energy-to-LLM Proxy Validation Suite")
    print("=" * 80)

    scenarios = {
        "Independent": IndependentScenario(num_agents=3, num_tasks=5, dim=8),
        "Interaction": InteractionScenario(num_agents=3, num_tasks=5, dim=8),
        "Frustrated": FrustratedScenario(num_agents=3, num_tasks=5, dim=8),
    }

    orchestrators = {
        "Random": lambda cfg: RandomOrchestrator(),
        "Greedy": lambda cfg: GreedyOrchestrator(),
        "EBMAO": lambda cfg: EBMAOOrchestrator(cfg),
    }

    results = []

    for sc_name, scenario in scenarios.items():
        print(f"\n--- Running Scenario: {sc_name} ---")
        for seed in seeds:
            problem = scenario.generate(seed)
            ebmao_cfg = make_ebmao_config(iterations=25)

            for orch_name, orch_fn in orchestrators.items():
                torch.manual_seed(seed)
                np.random.seed(seed)

                orchestrator = orch_fn(ebmao_cfg)
                X = orchestrator.solve(problem)

                energy, components = compute_energy(problem, X)

                # Execute LLM calls for assigned tasks
                total_prompt_tokens = 0
                total_completion_tokens = 0
                total_latency = 0.0
                failures = 0
                total_calls = 0

                for a_idx, agent in enumerate(problem.agents):
                    assigned_indices = (X[a_idx] == 1.0).nonzero(as_tuple=True)[0]
                    if len(assigned_indices) == 0:
                        continue

                    assigned_tasks = [problem.tasks[t_idx] for t_idx in assigned_indices]
                    task_str = ", ".join([f"Task #{t.id}" for t in assigned_tasks])

                    prompt = (
                        f"Agent Role: '{agent.role}'.\n"
                        f"Assigned tasks: {task_str}.\n"
                        f"Briefly outline in 2 sentences how you will complete these assigned tasks."
                    )

                    total_calls += 1
                    call_res = call_openrouter_with_metrics(prompt)
                    
                    total_prompt_tokens += call_res["prompt_tokens"]
                    total_completion_tokens += call_res["completion_tokens"]
                    total_latency += call_res["latency_sec"]
                    if not call_res["success"]:
                        failures += 1

                    time.sleep(delay_sec)

                rec = {
                    "scenario": sc_name,
                    "seed": seed,
                    "orchestrator": orch_name,
                    "energy": float(energy),
                    "assignment_energy": float(components.get("AssignmentEnergy", 0.0)),
                    "interaction_energy": float(components.get("InteractionEnergy", 0.0)),
                    "cost_energy": float(components.get("CostEnergy", 0.0)),
                    "risk_energy": float(components.get("RiskEnergy", 0.0)),
                    "llm_calls": total_calls,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                    "total_latency_sec": round(total_latency, 3),
                    "failures": failures,
                }
                results.append(rec)

                print(
                    f"[{sc_name}|Seed {seed}|{orch_name:7s}] Energy: {energy:7.3f} | "
                    f"Tokens: {total_prompt_tokens + total_completion_tokens:5d} | "
                    f"Latency: {total_latency:5.2f}s | Failures: {failures}"
                )

    # Statistical Correlation Analysis
    energies = [r["energy"] for r in results]
    tokens = [r["total_tokens"] for r in results]
    latencies = [r["total_latency_sec"] for r in results]

    spearman_tokens, p_tokens = stats.spearmanr(energies, tokens)
    pearson_tokens, p_pearson_tok = stats.pearsonr(energies, tokens)
    spearman_lat, p_lat = stats.spearmanr(energies, latencies)

    print("\n" + "=" * 80)
    print("EMPOWERMENT & PROXY VALIDATION RESULTS")
    print("=" * 80)
    print(f"Energy vs Total Tokens  -> Spearman r = {spearman_tokens:.4f} (p = {p_tokens:.4e})")
    print(f"Energy vs Total Tokens  -> Pearson r  = {pearson_tokens:.4f} (p = {p_pearson_tok:.4e})")
    print(f"Energy vs Total Latency -> Spearman r = {spearman_lat:.4f} (p = {p_lat:.4e})")
    print("-" * 80)

    # Average Energy and Tokens by Orchestrator
    by_orch = {}
    for r in results:
        o = r["orchestrator"]
        if o not in by_orch:
            by_orch[o] = {"energies": [], "tokens": [], "latencies": []}
        by_orch[o]["energies"].append(r["energy"])
        by_orch[o]["tokens"].append(r["total_tokens"])
        by_orch[o]["latencies"].append(r["total_latency_sec"])

    print("\nSummary by Orchestrator Method:")
    for o, d in by_orch.items():
        print(
            f"  {o:8s} -> Avg Energy: {np.mean(d['energies']):7.3f} | "
            f"Avg Tokens: {np.mean(d['tokens']):6.1f} | "
            f"Avg Latency: {np.mean(d['latencies']):5.2f}s"
        )

    # Save validation output to JSON
    os.makedirs("results", exist_ok=True)
    out_file = "results/energy_proxy_validation.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "correlation": {
                    "spearman_energy_vs_tokens": float(spearman_tokens),
                    "p_val_spearman_tokens": float(p_tokens),
                    "pearson_energy_vs_tokens": float(pearson_tokens),
                    "spearman_energy_vs_latency": float(spearman_lat),
                },
                "summary": {
                    o: {
                        "avg_energy": float(np.mean(d["energies"])),
                        "avg_tokens": float(np.mean(d["tokens"])),
                        "avg_latency": float(np.mean(d["latencies"])),
                    }
                    for o, d in by_orch.items()
                },
                "records": results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved empirical validation results to {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_validation_suite()
