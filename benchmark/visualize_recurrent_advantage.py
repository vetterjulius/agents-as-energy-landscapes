"""Verify and visualize recurrent advantage benchmark outputs."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGURATION_ORDER = [
    "Static Energy",
    "EBMAO (kappa-only)",
    "EBMAO (theta-only)",
    "Full EBMAO",
]
COLORS = {
    "Static Energy": "#343a40",
    "EBMAO (kappa-only)": "#007c91",
    "EBMAO (theta-only)": "#d95f02",
    "Full EBMAO": "#6a3d9a",
}


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def verify_inputs(raw, statistics):
    expected = set(CONFIGURATION_ORDER)
    assert set(raw["configuration"]) == expected
    assert len(raw["seed"].unique()) == 20
    assert len(raw["episode"].unique()) == 32
    assert len(raw) == 20 * 32 * 4
    assert raw.isna().sum().sum() == 0
    assert len(statistics) == 6
    assert set(statistics["configuration"]) == expected - {"Static Energy"}
    assert set(statistics["metric"]) == {"internal_energy", "absolute_gap"}
    assert raw.groupby("seed").size().eq(128).all()
    assert raw.groupby("configuration").size().eq(640).all()


def _save_gap_summary(raw, output_dir):
    seed_means = raw.groupby(["seed", "configuration"], as_index=False)["absolute_gap"].mean()
    means, errors = [], []
    for configuration in CONFIGURATION_ORDER:
        values = seed_means.loc[
            seed_means.configuration == configuration, "absolute_gap"
        ]
        mean, error = _mean_ci(values)
        means.append(mean)
        errors.append(error)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(CONFIGURATION_ORDER))
    ax.bar(x, means, yerr=errors, capsize=5,
           color=[COLORS[name] for name in CONFIGURATION_ORDER],
           edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Absolute gap to exact optimum (lower is better)")
    ax.set_title("Recurrent regimes: external solution quality")
    ax.set_xticks(x, ["Static", "Kappa-only", "Theta-only", "Full EBMAO"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "recurrent_gap_summary.png", dpi=220)
    plt.close(fig)


def _save_paired_effects(statistics, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, metric, label in zip(
        axes, ["absolute_gap", "internal_energy"], ["External gap", "Internal energy"]
    ):
        subset = statistics[statistics.metric == metric].copy()
        subset["configuration"] = pd.Categorical(
            subset.configuration, categories=CONFIGURATION_ORDER[1:], ordered=True
        )
        subset = subset.sort_values("configuration").reset_index(drop=True)
        y = subset.mean_delta.to_numpy()
        errors = [y - subset.ci_low.to_numpy(), subset.ci_high.to_numpy() - y]
        x = np.arange(len(subset))
        ax.errorbar(x, y, yerr=errors, fmt="o", markersize=7,
                    capsize=5, color="#007c91")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, ["Kappa-only", "Theta-only", "Full EBMAO"])
        ax.set_ylabel("Paired delta vs Static Energy")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        for index, row in subset.iterrows():
            ax.annotate(f"p={row.paired_t_p_value:.3f}", (index, row.mean_delta),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8)
    fig.suptitle("Paired seed effects with 95% confidence intervals", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "recurrent_paired_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_episode_trajectories(raw, output_dir):
    means = raw.groupby(["episode", "configuration"], as_index=False).agg(
        absolute_gap=("absolute_gap", "mean"),
        internal_energy=("internal_energy", "mean"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for configuration in CONFIGURATION_ORDER:
        subset = means[means.configuration == configuration]
        axes[0].plot(subset.episode, subset.absolute_gap,
                     label=configuration, color=COLORS[configuration], linewidth=2)
        axes[1].plot(subset.episode, subset.internal_energy,
                     label=configuration, color=COLORS[configuration], linewidth=2)
    for ax, ylabel in zip(axes, ["Mean absolute gap", "Mean internal energy"]):
        for boundary in (8, 16, 24):
            ax.axvline(boundary - 0.5, color="#adb5bd", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, frameon=False)
    axes[1].set_xlabel("Episode")
    fig.suptitle("Learning dynamics across recurring regimes")
    fig.tight_layout()
    fig.savefig(output_dir / "recurrent_episode_trajectories.png", dpi=220)
    plt.close(fig)


def _save_internal_external_scatter(raw, output_dir):
    means = raw.groupby(["seed", "configuration"], as_index=False).agg(
        internal_energy=("internal_energy", "mean"),
        absolute_gap=("absolute_gap", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for configuration in CONFIGURATION_ORDER:
        subset = means[means.configuration == configuration]
        ax.scatter(subset.internal_energy, subset.absolute_gap,
                   label=configuration, color=COLORS[configuration], alpha=0.8,
                   edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Mean internal energy")
    ax.set_ylabel("Mean absolute gap to optimum")
    ax.set_title("Internal model quality does not imply external improvement")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "recurrent_internal_external_scatter.png", dpi=220)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=Path("results/recurrent_advantage_benchmark_20seeds.csv"))
    parser.add_argument("--statistics", type=Path,
                        default=Path("results/recurrent_advantage_benchmark_20seeds_statistics.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/plots/recurrent_advantage"))
    args = parser.parse_args()
    raw = pd.read_csv(args.raw)
    statistics = pd.read_csv(args.statistics)
    verify_inputs(raw, statistics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_gap_summary(raw, args.output_dir)
    _save_paired_effects(statistics, args.output_dir)
    _save_episode_trajectories(raw, args.output_dir)
    _save_internal_external_scatter(raw, args.output_dir)
    print(f"Verified {len(raw)} rows across {raw.seed.nunique()} seeds.")
    print(f"Wrote 4 figures to {args.output_dir}")


if __name__ == "__main__":
    main()