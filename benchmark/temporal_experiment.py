from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from benchmark.logging_config import get_logger
from benchmark.dynamic_benchmark import (
    MultiEpisodeSimulator,
    generate_regime_switch_episode,
)
from benchmark.evaluation.metrics import brute_force_optimum

logger = get_logger("temporal_experiment")


CONFIGURATIONS = {
    "Static Energy": (False, False),
    "EBMAO (kappa-only)": (True, False),
    "EBMAO (theta-only)": (False, True),
    "Full EBMAO": (True, True),
}


def _paired_statistics(reference, comparison):
    """Return paired statistics for two seed-aligned one-dimensional series."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    delta = comparison - reference
    sample_size = len(delta)
    mean_delta = float(np.mean(delta)) if sample_size else float("nan")
    std_delta = float(np.std(delta, ddof=1)) if sample_size > 1 else 0.0

    if sample_size > 1 and std_delta > 1e-12:
        t_stat, t_p_value = stats.ttest_rel(comparison, reference)
        ci_half_width = stats.t.ppf(0.975, sample_size - 1) * std_delta / np.sqrt(sample_size)
        cohen_dz = mean_delta / std_delta
    else:
        t_stat, t_p_value = 0.0, 1.0
        ci_half_width = 0.0
        cohen_dz = 0.0 if sample_size else float("nan")

    if sample_size > 1 and np.any(np.abs(delta) > 1e-12):
        try:
            _, wilcoxon_p_value = stats.wilcoxon(delta)
        except ValueError:
            wilcoxon_p_value = 1.0
    else:
        wilcoxon_p_value = 1.0

    return {
        "n_seeds": sample_size,
        "reference_mean": float(np.mean(reference)) if sample_size else float("nan"),
        "comparison_mean": float(np.mean(comparison)) if sample_size else float("nan"),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_low": mean_delta - ci_half_width,
        "ci_high": mean_delta + ci_half_width,
        "paired_t_stat": float(t_stat),
        "paired_t_p_value": float(t_p_value),
        "wilcoxon_p_value": float(wilcoxon_p_value),
        "cohen_dz": float(cohen_dz),
    }


def compute_temporal_statistics(result, reference_configuration="Static Energy"):
    """Compute paired seed statistics and the internal/external trade-off test.

    Negative deltas are improvements for both energies and the external gap.
    ``tradeoff_supported`` is true when a configuration improves mean internal
    energy while worsening mean external assignment gap relative to Static Energy.
    """
    required = {"seed", "configuration", "internal_energy", "absolute_gap"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Temporal results missing columns: {sorted(missing)}")

    seed_means = (
        result.groupby(["seed", "configuration"], as_index=False)[
            ["internal_energy", "absolute_gap"]
        ]
        .mean()
    )
    reference = seed_means[seed_means["configuration"] == reference_configuration].rename(
        columns={
            "internal_energy": "reference_internal_energy",
            "absolute_gap": "reference_absolute_gap",
        }
    )

    rows = []
    for configuration in sorted(seed_means["configuration"].unique()):
        if configuration == reference_configuration:
            continue
        comparison = seed_means[seed_means["configuration"] == configuration].merge(
            reference[["seed", "reference_internal_energy", "reference_absolute_gap"]],
            on="seed",
            how="inner",
        )
        internal = _paired_statistics(
            comparison["reference_internal_energy"], comparison["internal_energy"]
        )
        external = _paired_statistics(
            comparison["reference_absolute_gap"], comparison["absolute_gap"]
        )
        rows.append(
            {
                "configuration": configuration,
                "metric": "internal_energy",
                **internal,
                "tradeoff_supported": False,
            }
        )
        rows.append(
            {
                "configuration": configuration,
                "metric": "absolute_gap",
                **external,
                "tradeoff_supported": bool(
                    internal["comparison_mean"] < internal["reference_mean"]
                    and external["comparison_mean"] > external["reference_mean"]
                ),
            }
        )

    return pd.DataFrame(rows)


def controlled_regime_generator(episode, seed=42):
    return generate_regime_switch_episode(
        episode,
        seed=seed,
        num_agents=3,
        num_tasks=6,
        dim=4,
        regime_length=3,
    )


def run_temporal_controlled_benchmark(
    seeds=(42, 43, 44),
    num_episodes=12,
    iterations=5,
    search_mode="guided_sa",
    output_path="results/controlled_temporal_benchmark.csv",
    statistics_output_path="results/controlled_temporal_benchmark_statistics.csv",
):
    rows = []

    for seed in seeds:
        for configuration, (kappa_enabled, theta_enabled) in CONFIGURATIONS.items():
            simulator = MultiEpisodeSimulator(
                controlled_regime_generator,
                num_episodes=num_episodes,
                seed=seed,
            )
            history = simulator.run(
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
                        "episode": int(episode_idx),
                        "regime": int(episode_idx // 3) % 2,
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
    result = run_temporal_controlled_benchmark()
    summary = result.groupby(["configuration", "regime"], as_index=False).agg(
        mean_external_energy=("external_energy", "mean"),
        mean_internal_energy=("internal_energy", "mean"),
        mean_gap=("absolute_gap", "mean"),
        mean_reconfiguration=("reconfig_cost", "mean"),
    )
    logger.info("Temporal Controlled Benchmark Summary:")
    logger.info("\n" + summary.round(4).to_string(index=False))
    logger.info("\nPaired seed statistics:")
    logger.info("\n" + compute_temporal_statistics(result).round(4).to_string(index=False))
