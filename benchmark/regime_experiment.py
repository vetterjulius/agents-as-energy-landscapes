from pathlib import Path

import pandas as pd

from benchmark.dynamic_benchmark import (
    MultiEpisodeSimulator,
    generate_regime_switch_episode,
)


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
            )

            for regime, start in enumerate(range(0, num_episodes, regime_length)):
                end = min(start + regime_length, num_episodes)
                rows.append(
                    {
                        "seed": seed,
                        "configuration": configuration,
                        "search_mode": search_mode,
                        "iterations": iterations,
                        "regime_block": regime,
                        "regime": regime % 2,
                        "mean_energy": float(history.energy.iloc[start:end].mean()),
                        "final_energy": float(history.energy.iloc[end - 1]),
                        "mean_kappa_norm": float(history.kappa_norm.iloc[start:end].mean()),
                        "final_kappa_norm": float(history.kappa_norm.iloc[end - 1]),
                    }
                )

    result = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    result = run_regime_switch_experiment()
    summary = result.groupby(["configuration", "regime_block"], as_index=False).mean(numeric_only=True)
    print(summary.to_string(index=False))
