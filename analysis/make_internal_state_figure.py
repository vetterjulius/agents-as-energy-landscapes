
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import analysis.common as common
from analysis.make_figures import COLORS, LABEL_SHORT


SCENARIOS = ("Stationary", "Dependency Change")
CONDITIONS = ("B1", "B2", "B3", "B4")
SERIES = (
    ("theta_norm", r"$\|\Theta_t\|_F$"),
    ("kappa_norm", r"$\|\kappa_t\|_F$"),
    ("theta_diff_norm", r"$\|\Theta_t-\Theta_{\mathrm{gt},t}\|_F$"),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_path_arguments(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=common.ROOT / "paper" / "figures" / "internal_state_trajectories.pdf",
        help="Output PDF path.",
    )
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)

    source = common.ARTIFACTS / "figure_data" / "state_trajectories_episode_means.csv"
    common.require_file(source)
    data = pd.read_csv(source)
    required = {"scenario_id", "condition", "episode", *(name for name, _ in SERIES)}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"State trajectory data is missing columns: {sorted(missing)}")
    selected = data[data.scenario_id.isin(SCENARIOS) & data.condition.isin(CONDITIONS)]
    if selected.empty or set(selected.scenario_id.unique()) != set(SCENARIOS):
        raise ValueError("State trajectory data does not contain both required manuscript scenarios.")
    if set(selected.condition.unique()) != set(CONDITIONS):
        raise ValueError("State trajectory data does not contain all B1-B4 conditions.")
    counts = selected.groupby(["scenario_id", "condition"])["episode"].nunique()
    if len(counts) != len(SCENARIOS) * len(CONDITIONS) or not (counts == 50).all():
        raise ValueError("State trajectory data must contain 50 episodes for every scenario and condition.")
    if selected[list(name for name, _ in SERIES)].isna().any().any():
        raise ValueError("State trajectory data contains missing values in plotted series.")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)
    for row, scenario in enumerate(SCENARIOS):
        scenario_data = selected[selected.scenario_id == scenario]
        for col, (field, label) in enumerate(SERIES):
            ax = axes[row, col]
            for condition in CONDITIONS:
                curve = scenario_data[scenario_data.condition == condition].sort_values("episode")
                if curve.empty:
                    raise ValueError(f"Missing {condition} data for {scenario}.")
                ax.plot(
                    curve.episode,
                    curve[field],
                    color=COLORS[condition],
                    label=LABEL_SHORT[condition],
                    linewidth=1.3,
                )
            ax.axvline(25, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(label)
            ax.set_ylabel(scenario if col == 0 else "")
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        r"Internal adaptation states: mean over 20 seeds; dashed line marks episode 25",
        y=0.995,
    )
    fig.tight_layout()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
