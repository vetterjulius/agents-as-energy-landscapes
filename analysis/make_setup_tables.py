
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis.common as common
from analysis.common import load_config, load_manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_path_arguments(parser)
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)
    cfg = load_config()
    tables = common.ARTIFACTS / "tables"
    tables.mkdir(exist_ok=True)

    setup = [
        ("experiment_id", cfg["experiment_id"]),
        ("benchmark_version", cfg["benchmark_version"]),
        ("git_commit", cfg["git_commit"]),
        ("agents (N)", cfg["num_agents"]),
        ("tasks (M)", cfg["num_tasks"]),
        ("embedding dimension (d)", cfg["dim"]),
        ("episodes per trajectory", cfg["num_episodes"]),
        ("perturbation episode", cfg["perturb_episode"]),
        ("pre-window (episodes)", "15-24"),
        ("post-window (episodes)", "25-34"),
        ("primary metric", "PPEE_10 (mean over episodes 25-34 of max(0, external_energy - reference_energy))"),
        ("reference energy", "exact optimum of the episode-25 problem instance (MILP, is_optimal=True for all 400 runs)"),
        ("seeds", "42..61 (20 seeds, paired across conditions)"),
        ("scenarios", "; ".join(cfg["scenarios"])),
        ("conditions", "; ".join(cfg["conditions"])),
        ("energy evaluation budget per episode (B1-B4)", cfg["max_energy_evaluations"]),
        ("energy evaluations B0", "0 (by design)"),
        ("initial assignment", "deterministic round-robin, identical for all solvers, no warm start"),
        ("adaptation rate eta_memory (kappa)", cfg["eta_memory"]),
        ("adaptation rate eta_theta (Theta)", cfg["eta_theta"]),
        ("lambda_align", cfg["lambda_align"]),
        ("lambda_memory", cfg["lambda_memory"]),
        ("interaction_weight", cfg["interaction_weight"]),
        ("cost_weight", cfg["cost_weight"]),
        ("risk_weight", cfg["risk_weight"]),
        ("SA temperature_init", cfg["sa_temperature_init"]),
        ("SA min_temperature", cfg["sa_min_temperature"]),
        ("SA cooling_rate", cfg["sa_cooling_rate"]),
        ("SA move proposal", "uniform random single-task reassignment; best-of-run kept"),
        ("Energy Greedy neighborhood", "1-move steepest descent (M x (N-1) candidates per pass)"),
        ("ILP reference time limit (s)", cfg["ilp_time_limit_sec"]),
        ("multiplicity correction", "Holm-Bonferroni within pre-specified hypothesis family"),
        ("total runs", "5 conditions x 4 scenarios x 20 seeds = 400"),
        ("total episode solves", "400 x 50 = 20,000"),
        ("total energy evaluations (B1-B4)", "6,955,387 (B1: 1,480,900; B2: 2,000,000; B3: 1,474,487; B4: 2,000,000)"),
    ]
    pd.DataFrame(setup, columns=["parameter", "value"]).to_csv(
        tables / "experimental_setup.csv", index=False)

    conditions = [
        ("B0", "conventional_greedy", "None (conventional local allocation)", "Conventional Greedy",
         "none", "no interaction structure", 0,
         "per-task argmin of capability-distance - alignment + risk; no kappa/Theta/C"),
        ("B1", "static_energy_greedy", "Static (kappa_0=0, Theta_0=G_gt pre-change)", "Energy Greedy",
         "none", "fixed Theta_0", 500,
         "steepest-descent local search on static energy landscape"),
        ("B2", "static_energy_sa", "Static (kappa_0=0, Theta_0=G_gt pre-change)", "Simulated Annealing",
         "none", "fixed Theta_0", 500,
         "single-task-reassignment SA on the same static landscape"),
        ("B3", "adaptive_energy_greedy", "Adaptive full (kappa_t, Theta_t via EMA)", "Energy Greedy",
         "kappa (eta=0.05) + Theta (eta=0.10)", "learned co-assignment EMA of observed X_t", 500,
         "same greedy solver on the adapting landscape"),
        ("B4", "adaptive_energy_sa", "Adaptive full (kappa_t, Theta_t via EMA)", "Simulated Annealing",
         "kappa (eta=0.05) + Theta (eta=0.10)", "learned co-assignment EMA of observed X_t", 500,
         "same SA solver on the adapting landscape"),
    ]
    pd.DataFrame(conditions, columns=[
        "ID", "condition_id", "landscape", "solver", "adaptation", "interaction structure",
        "energy_evaluations_per_episode", "notes"]).to_csv(
        tables / "benchmark_conditions.csv", index=False)
    print("wrote experimental_setup.csv and benchmark_conditions.csv")


if __name__ == "__main__":
    main()
