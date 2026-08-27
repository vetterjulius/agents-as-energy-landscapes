from pathlib import Path

import pandas as pd

from benchmark.dynamic_benchmark import (
    MultiEpisodeSimulator,
    generate_regime_switch_episode,
)
from benchmark.evaluation.metrics import brute_force_optimum
from benchmark.temporal_experiment import CONFIGURATIONS, compute_temporal_statistics


REGIME_LENGTH = 8
NUM_AGENTS = 3
NUM_TASKS = 6
DIMENSION = 4


def recurrent_regime_generator(episode, seed=42):
    return generate_regime_switch_episode(
        episode,
        seed=seed,
        num_agents=NUM_AGENTS,
        num_tasks=NUM_TASKS,
        dim=DIMENSION,
        regime_length=REGIME_LENGTH,
    )


def run_recurrent_advantage_benchmark(
    seeds=tuple(range(10)),
    num_episodes=32,
    iterations=2,
    search_mode="guided_sa",
    output_path="results/recurrent_advantage_benchmark.csv",
    statistics_output_path="results/recurrent_advantage_benchmark_statistics.csv",
):
    """Evaluate recurrent regimes under a deliberately constrained search budget."""
    rows = []

    for seed in seeds:
        for configuration, (kappa_enabled, theta_enabled) in CONFIGURATIONS.items():
            history = MultiEpisodeSimulator(
                recurrent_regime_generator,
                num_episodes=num_episodes,
                seed=seed,
            ).run(
                config_override={"solver": {"iterations": iterations}},
                kappa_enabled=kappa_enabled,
                theta_enabled=theta_enabled,
                search_mode=search_mode,
                reference_energy_fn=brute_force_optimum,
            )

            for episode_idx, episode in history.iterrows():
                rows.append(
                    {
                        "seed": seed,
                        "configuration": configuration,
                        "search_mode": search_mode,
                        "iterations": iterations,
                        "regime_length": REGIME_LENGTH,
                        "episode": int(episode_idx),
                        "regime": int(episode_idx // REGIME_LENGTH) % 2,
                        "external_energy": episode["external_energy"],
                        "internal_energy": episode["internal_energy"],
                        "reference_energy": episode["reference_energy"],
                        "absolute_gap": episode["absolute_gap"],
                        "kappa_norm": episode["kappa_norm"],
                        "context_similarity": episode["context_similarity"],
                        "reconfig_cost": episode["reconfig_cost"],
                    }
                )

    result = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    if statistics_output_path is not None:
        statistics_output = Path(statistics_output_path)
        statistics_output.parent.mkdir(parents=True, exist_ok=True)
        compute_temporal_statistics(result).to_csv(statistics_output, index=False)

    return result


if __name__ == "__main__":
    result = run_recurrent_advantage_benchmark()
    summary = result.groupby("configuration", as_index=False).agg(
        mean_external_energy=("external_energy", "mean"),
        mean_internal_energy=("internal_energy", "mean"),
        mean_gap=("absolute_gap", "mean"),
    )
    print(summary.round(4).to_string(index=False))
    print("\nPaired seed statistics:")
    print(compute_temporal_statistics(result).round(4).to_string(index=False))
