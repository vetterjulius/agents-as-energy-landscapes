
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis.common as common
from analysis.common import COMPARISONS, CONDITION_ORDER, PERTURB_EPISODE, SCENARIOS
from analysis.make_figures import COLORS, COMP_COLORS, LABEL_SHORT, MARKERS


PAPER_FIGURES = common.ROOT / "paper" / "figures"
T_CRIT = stats.t.ppf(0.975, df=19)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def require_finite(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite plotted values")


def validate_runs(runs: pd.DataFrame) -> None:
    require_columns(runs, {"scenario_id", "condition", "seed", "ppee_10"}, "runs.csv")
    if set(runs.scenario_id) != set(SCENARIOS):
        raise ValueError("runs.csv does not contain exactly the four required scenarios")
    if set(runs.condition) != set(CONDITION_ORDER):
        raise ValueError("runs.csv does not contain exactly conditions B0-B4")
    counts = runs.groupby(["scenario_id", "condition"]).seed.nunique()
    if len(counts) != 20 or not (counts == 20).all():
        raise ValueError("runs.csv must contain 20 distinct seeds per scenario and condition")
    if runs.duplicated(["scenario_id", "condition", "seed"]).any():
        raise ValueError("runs.csv contains duplicate scenario/condition/seed rows")
    require_finite(runs, {"ppee_10"}, "runs.csv")


def validate_episodes(eps: pd.DataFrame) -> None:
    require_columns(
        eps,
        {"scenario_id", "condition", "seed", "episode", "external_energy",
         "reference_energy", "reconfig_cost", "kappa_norm", "theta_norm",
         "theta_diff_norm"},
        "episodes.csv",
    )
    if set(eps.scenario_id) != set(SCENARIOS):
        raise ValueError("episodes.csv does not contain exactly the four required scenarios")
    if set(eps.condition) != set(CONDITION_ORDER):
        raise ValueError("episodes.csv does not contain exactly conditions B0-B4")
    counts = eps.groupby(["scenario_id", "condition", "seed"]).episode.nunique()
    if len(counts) != 400 or not (counts == 50).all():
        raise ValueError("episodes.csv must contain 50 episodes for each of 400 runs")
    if sorted(eps.episode.unique()) != list(range(50)):
        raise ValueError("episodes.csv must contain episode indices 0..49")
    require_finite(
        eps,
        {"external_energy", "reference_energy", "reconfig_cost"},
        "episodes.csv",
    )
    state = eps[eps.condition != "B0"]
    require_finite(state, {"kappa_norm", "theta_norm", "theta_diff_norm"}, "episodes.csv")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{name}.pdf"
    png = output_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_data(data: pd.DataFrame, artifact_dir: Path, name: str) -> Path:
    path = artifact_dir / "figure_data" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def make_primary(runs: pd.DataFrame, output_dir: Path, artifact_dir: Path) -> list[Path]:
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        sub = runs[runs.scenario_id == scenario]
        panel_max = 0.0
        for x, condition in enumerate(CONDITION_ORDER):
            condition_data = sub[sub.condition == condition].sort_values("seed")
            values = condition_data.ppee_10.to_numpy(float)
            order = np.argsort(values, kind="stable")
            values = values[order]
            seeds = condition_data.seed.to_numpy()[order]
            n = len(values)
            mean = float(values.mean())
            half = T_CRIT * float(values.std(ddof=1)) / np.sqrt(n)
            median = float(np.median(values))
            panel_max = max(panel_max, float(values.max()))
            ax.plot(
                x + np.linspace(-0.17, 0.17, n), values, ls="none",
                marker=MARKERS[condition], ms=3.6, color=COLORS[condition],
                alpha=0.55,
            )
            ax.errorbar([x], [mean], yerr=[[half], [half]], fmt="D",
                        color=COLORS[condition], ecolor="0.25", capsize=3.5)
            ax.plot([x], [median], marker="_", color="black", ms=15, mew=1.6)
            for seed, value in zip(seeds, values):
                rows.append({
                    "scenario": scenario, "condition": condition, "seed": int(seed),
                    "ppee_10": float(value), "mean": mean,
                    "ci95_t_low": mean - half, "ci95_t_high": mean + half,
                    "median": median, "n": n,
                })
        ax.set_title(f"{scenario} ($n$=20)")
        ax.set_xticks(range(5), CONDITION_ORDER)
        ax.set_ylim(-0.07 * panel_max, 1.10 * panel_max)
        ax.set_ylabel("PPEE$_{10}$ (lower is better)")
    axes[1, 0].set_xlabel("Condition")
    axes[1, 1].set_xlabel("Condition")
    handles = [
        plt.Line2D([], [], marker=MARKERS[c], ls="none", color=COLORS[c],
                   label=LABEL_SHORT[c]) for c in CONDITION_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Primary PPEE$_{10}$ outcome: individual seeds, mean ± 95% t CI, and median",
                 y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    outputs = list(save_figure(fig, output_dir, "main_ppee10"))
    write_data(pd.DataFrame(rows), artifact_dir, "main_ppee10")
    return outputs


def make_paired(runs: pd.DataFrame, comparisons: pd.DataFrame,
                output_dir: Path, artifact_dir: Path) -> list[Path]:
    require_columns(
        comparisons,
        {"comparison", "scenario", "n_pairs", "mean_diff", "median_diff",
         "ci95_t_low", "ci95_t_high"},
        "statistical_comparisons.csv",
    )
    comparisons = comparisons[comparisons.metric == "ppee_10"].copy()
    expected = {name for name, *_ in COMPARISONS}
    if set(comparisons.comparison) != expected or len(comparisons) != 24:
        raise ValueError("statistical_comparisons.csv has incomplete primary contrasts")
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True)
    for ax, (name, a, b, _) in zip(axes.ravel(), COMPARISONS):
        for y, scenario in enumerate(reversed(SCENARIOS)):
            sub = runs[runs.scenario_id == scenario].pivot(
                index="seed", columns="condition", values="ppee_10")
            diff = (sub[a] - sub[b]).dropna()
            stat = comparisons[
                (comparisons.comparison == name) & (comparisons.scenario == scenario)
            ]
            if len(stat) != 1 or int(stat.iloc[0].n_pairs) != len(diff):
                raise ValueError(f"Pair-count mismatch for {name} / {scenario}")
            row = stat.iloc[0]
            if not np.isclose(diff.mean(), row.mean_diff) or not np.isclose(
                    diff.median(), row.median_diff):
                raise ValueError(f"Paired estimates mismatch for {name} / {scenario}")
            if not np.isclose(diff.mean() - T_CRIT * diff.std(ddof=1) / np.sqrt(len(diff)),
                              row.ci95_t_low) or not np.isclose(
                                  diff.mean() + T_CRIT * diff.std(ddof=1) / np.sqrt(len(diff)),
                                  row.ci95_t_high):
                raise ValueError(f"Paired t CI mismatch for {name} / {scenario}")
            spread = np.linspace(-0.18, 0.18, len(diff))
            ax.plot(diff.to_numpy(), y + spread, ".", color=COMP_COLORS[name], alpha=0.5)
            ax.plot([row.mean_diff], [y], "D", color=COMP_COLORS[name], ms=5)
            ax.plot([row.median_diff], [y], "o", color="black", ms=5,
                    mfc="white", label="median" if y == 0 else None)
            ax.plot([row.ci95_t_low, row.ci95_t_high], [y, y], "|-",
                    color=COMP_COLORS[name], ms=10, lw=1.4)
            for seed, value in diff.items():
                rows.append({
                    "comparison": name, "scenario": scenario, "seed": int(seed),
                    "diff": float(value), "mean_diff": float(row.mean_diff),
                    "median_diff": float(row.median_diff),
                    "ci95_t_low": float(row.ci95_t_low),
                    "ci95_t_high": float(row.ci95_t_high),
                    "n_pairs": int(row.n_pairs),
                })
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_title(name)
        ax.set_yticks(range(4), list(reversed(SCENARIOS)))
        ax.set_xlabel("Paired Δ PPEE$_{10}$ (first − second; negative favors first)")
    axes[0, 0].set_ylabel("Scenario")
    fig.suptitle("Paired PPEE$_{10}$ effects: dots, mean ± 95% t CI, and median; n=20",
                 y=0.995)
    fig.tight_layout()
    outputs = list(save_figure(fig, output_dir, "paired_effects"))
    write_data(pd.DataFrame(rows), artifact_dir, "paired_effects")
    return outputs


def make_temporal(eps: pd.DataFrame, output_dir: Path, artifact_dir: Path) -> list[Path]:
    data = eps.copy()
    data["excess"] = np.maximum(0.0, data.external_energy - data.reference_energy)
    data = data[(data.episode >= 15) & (data.episode <= 34)]
    grouped = data.groupby(["scenario_id", "condition", "episode"]).excess.agg(
        mean="mean", sd="std", n="size").reset_index()
    grouped["ci95_t_low"] = grouped["mean"] - T_CRIT * grouped.sd / np.sqrt(grouped.n)
    grouped["ci95_t_high"] = grouped["mean"] + T_CRIT * grouped.sd / np.sqrt(grouped.n)
    require_finite(grouped, {"mean", "ci95_t_low", "ci95_t_high"}, "temporal data")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        for condition in CONDITION_ORDER:
            sub = grouped[(grouped.scenario_id == scenario) &
                          (grouped.condition == condition)]
            ax.plot(sub.episode, sub["mean"], color=COLORS[condition],
                    label=LABEL_SHORT[condition])
            ax.fill_between(sub.episode, sub.ci95_t_low, sub.ci95_t_high,
                            color=COLORS[condition], alpha=0.12)
        ax.axvline(25, color="black", ls="--", lw=0.8)
        ax.axvline(26, color="black", ls=":", lw=0.8)
        ax.set_title(scenario)
        ax.set_ylabel("Excess external energy")
    axes[1, 0].set_xlabel("Episode (25: changed environment; 26: first updated-state decision)")
    axes[1, 1].set_xlabel("Episode")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("External-energy trajectories around episode 25; mean ± 95% t CI",
                 y=0.995)
    fig.tight_layout()
    outputs = list(save_figure(fig, output_dir, "external_energy_trajectories"))
    write_data(grouped.rename(columns={"scenario_id": "scenario"}), artifact_dir,
               "external_energy_trajectories")
    return outputs


def make_internal(eps: pd.DataFrame, output_dir: Path, artifact_dir: Path) -> list[Path]:
    selected = eps[(eps.scenario_id.isin(["Stationary", "Dependency Change"])) &
                   (eps.condition.isin(["B1", "B2", "B3", "B4"]))]
    grouped = selected.groupby(["scenario_id", "condition", "episode"])[
        ["theta_norm", "kappa_norm", "theta_diff_norm"]].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped = grouped.reset_index()
    for metric in ["theta_norm", "kappa_norm", "theta_diff_norm"]:
        grouped[f"{metric}_ci95_low"] = (
            grouped[f"{metric}_mean"] -
            T_CRIT * grouped[f"{metric}_std"] / np.sqrt(grouped[f"{metric}_count"]))
        grouped[f"{metric}_ci95_high"] = (
            grouped[f"{metric}_mean"] +
            T_CRIT * grouped[f"{metric}_std"] / np.sqrt(grouped[f"{metric}_count"]))
    require_finite(
        grouped,
        {"theta_norm_mean", "kappa_norm_mean", "theta_diff_norm_mean"},
        "internal-state data",
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)
    panels = [("theta_norm", r"$\|\Theta_t\|_F$"),
              ("kappa_norm", r"$\|\kappa_t\|_F$"),
              ("theta_diff_norm", r"$\|\Theta_t-\Theta_{\mathrm{gt},t}\|_F$")]
    for row, scenario in enumerate(["Stationary", "Dependency Change"]):
        for col, (metric, label) in enumerate(panels):
            ax = axes[row, col]
            for condition in ["B1", "B2", "B3", "B4"]:
                sub = grouped[(grouped.scenario_id == scenario) &
                              (grouped.condition == condition)]
                ax.plot(sub.episode, sub[f"{metric}_mean"], color=COLORS[condition],
                        label=LABEL_SHORT[condition])
                ax.fill_between(sub.episode, sub[f"{metric}_ci95_low"],
                                sub[f"{metric}_ci95_high"], color=COLORS[condition],
                                alpha=0.10)
            ax.axvline(25, color="black", ls="--", lw=0.8)
            ax.axvline(26, color="black", ls=":", lw=0.8)
            ax.set_title(label)
            ax.set_ylabel(scenario if col == 0 else "")
            ax.set_xlabel("Episode")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Internal states: mean ± 95% t CI; ground-truth distance is post-hoc only",
                 y=0.995)
    fig.tight_layout()
    outputs = list(save_figure(fig, output_dir, "internal_state_trajectories"))
    write_data(grouped.rename(columns={"scenario_id": "scenario"}), artifact_dir,
               "internal_state_trajectories")
    return outputs


def make_ecdf(runs: pd.DataFrame, output_dir: Path, artifact_dir: Path) -> list[Path]:
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        for condition in CONDITION_ORDER:
            values = np.sort(runs[(runs.scenario_id == scenario) &
                                  (runs.condition == condition)].ppee_10.to_numpy(float))
            ecdf = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, ecdf, where="post", color=COLORS[condition],
                    label=LABEL_SHORT[condition])
            rows.extend({"scenario": scenario, "condition": condition, "seed_rank": i,
                         "ppee_10": float(value), "ecdf": float(y)}
                        for i, (value, y) in enumerate(zip(values, ecdf), start=1))
        ax.set_title(f"{scenario} ($n$=20)")
        ax.set_xlabel("PPEE$_{10}$ (lower is better)")
    axes[0].set_ylabel("ECDF")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("PPEE$_{10}$ empirical cumulative distributions", y=1.04)
    fig.tight_layout()
    outputs = list(save_figure(fig, output_dir, "ppee_distribution"))
    write_data(pd.DataFrame(rows), artifact_dir, "ppee_distribution")
    return outputs


def make_diagnostics(eps: pd.DataFrame, artifact_dir: Path,
                     output_dir: Path) -> list[Path]:
    table = pd.read_csv(common.require_file(
        artifact_dir / "tables" / "adaptation_diagnostics.csv"))
    churn = pd.read_csv(common.require_file(
        artifact_dir / "figure_data" / "assignment_churn_by_window.csv"))
    theta = pd.read_csv(common.require_file(
        artifact_dir / "figure_data" / "theta_accuracy_by_condition.csv"))
    require_columns(table, {"scenario", "condition", "kappa_change_post_mean",
                            "theta_change_post_mean", "theta_diff_norm_final_mean"},
                    "adaptation_diagnostics.csv")
    require_columns(churn, {"scenario_id", "condition", "window", "mean_reconfig"},
                    "assignment_churn_by_window.csv")
    require_columns(theta, {"scenario", "condition", "final_theta_diff_mean"},
                    "theta_accuracy_by_condition.csv")
    require_finite(
        table[table.condition.isin(["B1", "B2", "B3", "B4"])],
        {"kappa_change_post_mean", "theta_change_post_mean",
         "theta_diff_norm_final_mean"},
        "adaptation_diagnostics.csv",
    )
    require_finite(churn, {"mean_reconfig"}, "assignment_churn_by_window.csv")
    require_finite(theta, {"final_theta_diff_mean"}, "theta_accuracy_by_condition.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    rows = []
    post = churn[churn.window == "post-window(25-34)"]
    for condition in ["B1", "B2", "B3", "B4"]:
        sub = post[post.condition == condition]
        axes[0, 0].plot(sub.scenario_id, sub.mean_reconfig, marker="o",
                        color=COLORS[condition], label=LABEL_SHORT[condition])
        for row in sub.itertuples():
            rows.append({"panel": "churn", "scenario": row.scenario_id,
                         "condition": row.condition, "value": row.mean_reconfig})
    axes[0, 0].set_title("Assignment churn (post-change window)")
    axes[0, 0].set_ylabel("Mean reconfiguration fraction")
    for scenario in SCENARIOS:
        sub = table[(table.scenario == scenario) &
                    (table.condition.isin(["B1", "B2", "B3", "B4"]))]
        axes[0, 1].plot(sub.condition, sub.kappa_change_post_mean, marker="o",
                        label=scenario)
        axes[1, 0].plot(sub.condition, sub.theta_change_post_mean, marker="o",
                        label=scenario)
        for row in sub.itertuples():
            rows.extend([
                {"panel": "kappa_movement", "scenario": row.scenario,
                 "condition": row.condition, "value": row.kappa_change_post_mean},
                {"panel": "theta_movement", "scenario": row.scenario,
                 "condition": row.condition, "value": row.theta_change_post_mean},
            ])
    axes[0, 1].set_title("Kappa movement (episodes 25–34)")
    axes[0, 1].set_ylabel(r"$\|\kappa_{34}-\kappa_{25}\|$")
    axes[1, 0].set_title("Theta movement (episodes 25–34)")
    axes[1, 0].set_ylabel(r"$\|\Theta_{34}-\Theta_{25}\|$")
    for scenario in SCENARIOS:
        sub = theta[theta.scenario == scenario]
        axes[1, 1].plot(sub.condition, sub.final_theta_diff_mean, marker="o",
                        label=scenario)
        for row in sub.itertuples():
            rows.append({"panel": "final_representation_error", "scenario": row.scenario,
                         "condition": row.condition, "value": row.final_theta_diff_mean})
    axes[1, 1].set_title("Final representation error (post-hoc)")
    axes[1, 1].set_ylabel(r"$\|\Theta_{49}-\Theta_{\mathrm{gt},49}\|_F$")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].legend(frameon=False, fontsize=7)
    for ax in axes.ravel():
        ax.set_xlabel("Condition")
    fig.suptitle("Secondary adaptation diagnostics (descriptive; mean over seeds)",
                 y=0.995)
    fig.tight_layout()
    outputs = list(save_figure(fig, output_dir, "adaptation_diagnostics"))
    write_data(pd.DataFrame(rows), artifact_dir, "adaptation_diagnostics")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_path_arguments(parser)
    parser.add_argument("--paper-figures-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)
    output_dir = args.paper_figures_dir.resolve()
    runs = common.load_runs()
    eps = common.load_episodes()
    runs["condition"] = runs["condition_id"].map(common.CONDITION_MAP)
    eps["condition"] = eps["condition_id"].map(common.CONDITION_MAP)
    validate_runs(runs)
    validate_episodes(eps)
    comparisons = pd.read_csv(common.require_file(
        common.ARTIFACTS / "tables" / "statistical_comparisons.csv"))
    make_primary(runs, output_dir, common.ARTIFACTS)
    make_paired(runs, comparisons, output_dir, common.ARTIFACTS)
    make_temporal(eps, output_dir, common.ARTIFACTS)
    make_internal(eps, output_dir, common.ARTIFACTS)
    make_ecdf(runs, output_dir, common.ARTIFACTS)
    make_diagnostics(eps, common.ARTIFACTS, output_dir)
    print(f"generated six numerical figures in {output_dir}")


if __name__ == "__main__":
    main()
