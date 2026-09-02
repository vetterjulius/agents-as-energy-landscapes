from pathlib import Path

import pandas as pd

from benchmark.logging_config import get_logger
from benchmark.dynamic_benchmark import (
    MultiEpisodeSimulator,
    generate_regime_switch_episode,
)

logger = get_logger("regime_experiment")


CONFIGURATIONS = {
    "Static Energy": (False, False),
    "EBMAO (kappa-only)": (True, False),
    "EBMAO (theta-only)": (False, True),
    "Full EBMAO": (True, True),
}


def run_regime_switch_experiment(
    seeds=(42, 43, 44),
    num_episodes=40,
    iterations=3,
    search_mode="guided_sa",
    regime_length=10,
    memory_retention=1.0,
    theta_retention=1.0,
    adaptive_retention=False,
    output_path="results/regime_switch_guided_sa.csv",
):
    rows = []

    for seed in seeds:
        for configuration, (kappa_enabled, theta_enabled) in CONFIGURATIONS.items():
            simulator = MultiEpisodeSimulator(
                generate_regime_switch_episode,
                num_episodes=num_episodes,
                seed=seed,
            )
            history = simulator.run(
                config_override={"solver": {"iterations": iterations}},
                kappa_enabled=kappa_enabled,
                theta_enabled=theta_enabled,
                search_mode=search_mode,
                memory_retention=memory_retention,
                theta_retention=theta_retention,
                adaptive_retention=adaptive_retention,
            )

            for regime, start in enumerate(range(0, num_episodes, regime_length)):
                end = min(start + regime_length, num_episodes)
                rows.append(
                    {
                        "seed": seed,
                        "configuration": configuration,
                        "search_mode": search_mode,
                        "iterations": iterations,
                        "memory_retention": memory_retention,
                        "theta_retention": theta_retention,
                        "adaptive_retention": adaptive_retention,
                        "regime_block": regime,
                        "regime": regime % 2,
                        "mean_energy": float(history.energy.iloc[start:end].mean()),
                        "final_energy": float(history.energy.iloc[end - 1]),
                        "mean_internal_energy": float(
                            history.internal_energy.iloc[start:end].mean()
                        ),
                        "final_internal_energy": float(
                            history.internal_energy.iloc[end - 1]
                        ),
                        "mean_kappa_norm": float(history.kappa_norm.iloc[start:end].mean()),
                        "final_kappa_norm": float(history.kappa_norm.iloc[end - 1]),
                    }
                )

    result = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def run_retention_sweep(
    retention_values=(0.0, 0.25, 0.5, 0.75, 1.0),
    seeds=(42, 43, 44),
    num_episodes=40,
    iterations=3,
    search_mode="guided_sa",
    output_path="results/regime_retention_sweep.csv",
):
    """Evaluate independent retention factors for the two adaptive states."""
    results = []

    for parameter_name in ("theta_retention", "memory_retention"):
        for retention in retention_values:
            parameters = {
                "memory_retention": 1.0,
                "theta_retention": 1.0,
            }
            parameters[parameter_name] = retention
            result = run_regime_switch_experiment(
                seeds=seeds,
                num_episodes=num_episodes,
                iterations=iterations,
                search_mode=search_mode,
                memory_retention=parameters["memory_retention"],
                theta_retention=parameters["theta_retention"],
                output_path=Path(output_path).with_name(
                    f".{parameter_name}_{retention}.csv"
                ),
            )
            full = result[result["configuration"] == "Full EBMAO"]
            full = full.groupby("regime_block", as_index=False).mean(numeric_only=True)
            full["sweep_parameter"] = parameter_name
            full["retention"] = retention
            results.append(full)

    sweep = pd.concat(results, ignore_index=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output, index=False)
    return sweep


if __name__ == "__main__":
    result = run_retention_sweep()
    summary = result.groupby(["sweep_parameter", "retention"], as_index=False).mean(numeric_only=True)
    logger.info("Regime Experiment Summary:")
    logger.info("\n" + summary.to_string(index=False))
