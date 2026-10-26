from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis.common as common


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_path_arguments(parser)
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)
    runs_path = common.RESULTS / "runs.csv"
    episodes_path = common.RESULTS / "episodes.csv"
    common.require_file(runs_path)
    common.require_file(episodes_path)
    runs = pd.read_csv(runs_path)
    episodes = pd.read_csv(episodes_path)

    nan_run_columns = [
        "kappa_norm", "theta_norm", "adaptation_magnitude_kappa",
        "adaptation_magnitude_theta", "kappa_change_post", "theta_change_post",
    ]
    print("=== runs NaN (all-NaN per condition?) ===")
    print(runs.groupby("condition_id")[nan_run_columns].apply(
        lambda group: group.isna().all()).to_string())
    print("ppee_10 NaN:", runs.ppee_10.isna().sum(),
          "| inf:", np.isinf(runs.ppee_10.fillna(0)).sum())
    print("=== episodes NaN (all-NaN per condition?) ===")
    for column in [
        "optimal", "mip_gap", "kappa_norm", "theta_norm", "theta_diff_norm",
        "delta_kappa_norm", "delta_theta_norm",
    ]:
        values = episodes.groupby("condition_id")[column].apply(lambda item: item.isna().all())
        print(column, "->", {key: bool(value) for key, value in values.items()})
    print("all_episodes_optimal True:", int((runs.all_episodes_optimal == True).sum()),
          "of", int((runs.condition_id != "conventional_greedy").sum()), "energy runs")
    print("fallback_used True:", int((runs.fallback_used == True).sum()))
    energy_episodes = episodes[episodes.condition_id != "conventional_greedy"]
    print("episode-level optimal True (energy conds):",
          int((energy_episodes.optimal == True).sum()), "/", len(energy_episodes))

    labels = {
        "conventional_greedy": "B0",
        "static_energy_greedy": "B1",
        "static_energy_sa": "B2",
        "adaptive_energy_greedy": "B3",
        "adaptive_energy_sa": "B4",
    }
    runs["condition"] = runs.condition_id.map(labels)
    means = runs.pivot_table(index="scenario_id", columns="condition",
                              values="ppee_10", aggfunc="mean")
    print("\n=== fresh PPEE10 means per (condition, scenario) ===")
    print(means[["B0", "B1", "B2", "B3", "B4"]].round(4).to_string())

    zero_fraction = runs.pivot_table(
        index="scenario_id", columns="condition", values="ppee_10",
        aggfunc=lambda values: (values == 0).mean(),
    )
    print("\n=== zero fraction per (condition, scenario) ===")
    print(zero_fraction[["B0", "B1", "B2", "B3", "B4"]].round(2).to_string())

    frozen_path = common.ARTIFACTS / "tables" / "main_ppee_results.csv"
    common.require_file(frozen_path)
    frozen = pd.read_csv(frozen_path)
    print("\n=== compare with selected-run main_ppee_results.csv ===")
    print("columns:", list(frozen.columns))
    print(frozen.head(25).to_string())


if __name__ == "__main__":
    main()
