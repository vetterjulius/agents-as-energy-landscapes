
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
    parser.add_argument(
        "--reference-artifacts-dir",
        type=Path,
        default=common.DEFAULT_ARTIFACTS,
        help="Frozen artifact directory used for comparison.",
    )
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)
    runs = pd.read_csv(common.require_file(common.RESULTS / "runs.csv"))
    reference = args.reference_artifacts_dir.resolve()
    if reference == common.ARTIFACTS.resolve():
        raise RuntimeError(
            "Frozen comparison requires a reference artifact directory separate "
            "from the selected output directory."
        )
    frozen = pd.read_csv(common.require_file(
        reference / "tables" / "main_ppee_results.csv"))
    labels = {
        "conventional_greedy": "B0",
        "static_energy_greedy": "B1",
        "static_energy_sa": "B2",
        "adaptive_energy_greedy": "B3",
        "adaptive_energy_sa": "B4",
    }
    runs["condition"] = runs.condition_id.map(labels)
    frozen_means = {(row.scenario, row.condition): row.mean
                    for row in frozen.itertuples()}
    max_diff = 0.0
    worst = None
    for (scenario, condition), group in runs.groupby(["scenario_id", "condition"]):
        mean = group.ppee_10.mean()
        difference = abs(mean - frozen_means[(scenario, condition)])
        if difference > max_diff:
            max_diff = difference
            worst = (scenario, condition, mean, frozen_means[(scenario, condition)])
    if max_diff > 1e-12:
        raise RuntimeError(
            "Selected run does not match the frozen reference within 1e-12: "
            f"max difference={max_diff} at {worst}"
        )
    print("selected run matches frozen reference means:", max_diff)

    paired = pd.read_csv(common.require_file(
        common.ARTIFACTS / "figure_data" / "paired_differences_ppee10.csv"))
    reference_paired = pd.read_csv(common.require_file(
        reference / "figure_data" / "paired_differences_ppee10.csv"))
    key = ["scenario", "comparison", "seed"]
    required = set(key + ["diff"])
    if not required <= set(paired) or not required <= set(reference_paired):
        raise RuntimeError("Frozen paired-difference reference has incompatible columns.")
    merged = paired[key + ["diff"]].merge(
        reference_paired[key + ["diff"]],
        on=key,
        suffixes=("_selected", "_reference"),
    )
    if len(merged) != len(paired) or len(merged) != len(reference_paired):
        raise RuntimeError("Selected and frozen paired-difference keys do not match.")
    max_paired_difference = float(
        (merged["diff_selected"] - merged["diff_reference"]).abs().max()
    )
    if max_paired_difference > 1e-12:
        raise RuntimeError(
            "Selected paired differences do not match the frozen reference within 1e-12: "
            f"max difference={max_paired_difference}"
        )
    print("selected paired differences match frozen reference:", max_paired_difference)


if __name__ == "__main__":
    main()
