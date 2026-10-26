
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis.common as common
from analysis.common import CONDITION_LABEL, CONDITION_ORDER, PERTURB_EPISODE, SCENARIOS, \
    load_episodes, load_runs

FIG = common.ARTIFACTS / "figures"
FIGDATA = common.ARTIFACTS / "figure_data"

COLORS = {"B0": "#7f7f7f", "B1": "#1f77b4", "B2": "#2ca02c",
          "B3": "#d62728", "B4": "#9467bd"}
MARKERS = {"B0": "s", "B1": "o", "B2": "^", "B3": "D", "B4": "v"}
LABEL_SHORT = {"B0": "B0 Conv. Greedy", "B1": "B1 Static Greedy", "B2": "B2 Static SA",
               "B3": "B3 Adapt. Greedy", "B4": "B4 Adapt. SA"}
COMP_COLORS = {"B0-B1": "#1f77b4", "B0-B2": "#2ca02c", "B1-B3": "#d62728",
               "B2-B4": "#9467bd", "B1-B2": "#17becf", "B3-B4": "#8c564b"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png/.pdf")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_path_arguments(parser)
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)
    global FIG, FIGDATA
    FIG = common.ARTIFACTS / "figures"
    FIGDATA = common.ARTIFACTS / "figure_data"
    FIG.mkdir(parents=True, exist_ok=True)
    FIGDATA.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    eps = load_episodes()
    eps = eps.copy()
    eps["excess"] = np.maximum(0.0, eps.external_energy - eps.reference_energy)

    # ------------------------------------------------ Figure 1: episode trajectories
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        sub = eps[(eps.scenario_id == scenario) & (eps.episode >= 15) & (eps.episode <= 34)]
        g = sub.groupby(["condition", "episode"])["excess"].agg(["mean", "sem"]).reset_index()
        for cond in CONDITION_ORDER:
            gc = g[g.condition == cond]
            ax.plot(gc.episode, gc["mean"], color=COLORS[cond], marker=MARKERS[cond],
                    ms=2.5, lw=1.2, label=LABEL_SHORT[cond])
            ax.fill_between(gc.episode, gc["mean"] - 1.96 * gc["sem"],
                            gc["mean"] + 1.96 * gc["sem"], color=COLORS[cond], alpha=0.15, lw=0)
        ax.axvline(PERTURB_EPISODE, color="k", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(scenario)
        ax.set_ylabel("Excess energy  $\\max(0, E_{ext}-E^*)$")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[0, 0].legend(loc="upper left", ncol=1, frameon=False)
    for ax in axes.ravel():
        ax.text(0.98, 0.03, f"$n$ = 20 seeds", transform=ax.transAxes, ha="right",
                fontsize=7, color="0.35")
    fig.suptitle("Figure 1: External-energy excess around the perturbation (episode 25, dashed); "
                 "mean $\\pm$ 95% CI over 20 seeds", y=0.995)
    fig.tight_layout()
    save(fig, "figure1_trajectory")
    (eps[(eps.episode >= 15) & (eps.episode <= 34)]
     .groupby(["scenario_id", "condition", "episode"])["excess"]
     .agg(["mean", "sem", "size"]).reset_index()
     .to_csv(FIGDATA / "figure1_trajectory.csv", index=False))

    # ------------------------------------------------ Figure 2: static vs adaptive paired diff
    traj = pd.read_csv(FIGDATA / "static_vs_adaptive_excess_trajectory.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    comp_styles = {"B3-B1": ("-", "B3 (adaptive) $-$ B1 (static), Greedy", "#d62728"),
                   "B4-B2": ("--", "B4 (adaptive) $-$ B2 (static), SA", "#9467bd")}
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        for comp, (ls, label, color) in comp_styles.items():
            g = traj[(traj.scenario == scenario) & (traj.comparison == comp) &
                     (traj.episode >= 15) & (traj.episode <= 34)]
            ax.plot(g.episode, g["mean"], ls=ls, color=color, lw=1.4, label=label)
            ax.fill_between(g.episode, g["mean"] - 1.96 * g["se"], g["mean"] + 1.96 * g["se"],
                            color=color, alpha=0.15, lw=0)
        ax.axvline(PERTURB_EPISODE, color="k", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(0.0, color="0.5", lw=0.7, alpha=0.8)
        ax.set_title(scenario)
        ax.set_ylabel("Paired excess difference\n(adaptive $-$ static)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.suptitle("Figure 2: Adaptive $-$ static excess-energy difference; negative = adaptation "
                 "improves; mean $\\pm$ 95% CI over 20 paired seeds", y=0.995)
    fig.tight_layout()
    save(fig, "figure2_static_vs_adaptive")

    # ------------------------------------------------ Figure 3: Theta adaptation (Dependency Change vs Stationary)
    st = pd.read_csv(FIGDATA / "state_trajectories_episode_means.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    panels = [("Dependency Change", "theta_diff_norm",
               "Dependency Change:  $\\|\\Theta_t - G_{gt}(t)\\|$"),
              ("Stationary", "theta_diff_norm",
               "Stationary:  $\\|\\Theta_t - G_{gt}(t)\\|$"),
              ("Dependency Change", "theta_norm", "Dependency Change:  $\\|\\Theta_t\\|_F$")]
    for ax, (scenario, col, title) in zip(axes, panels):
        for cond in ("B1", "B2", "B3", "B4"):
            g = st[(st.scenario_id == scenario) & (st.condition == cond)]
            ax.plot(g.episode, g[col], color=COLORS[cond], lw=1.3, label=LABEL_SHORT[cond])
        ax.axvline(PERTURB_EPISODE, color="k", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Episode")
    axes[0].set_ylabel("Frobenius norm")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 3: $\\Theta$ adaptation (mean over 20 seeds; dashed = perturbation at "
                 "episode 25). Adaptive $\\Theta$ drifts under Stationarity but tracks the new "
                 "structure after Dependency Change.", y=1.04)
    fig.tight_layout()
    save(fig, "figure3_theta_adaptation")
    sel = st[st.scenario_id.isin(["Dependency Change", "Stationary"])]
    sel.to_csv(FIGDATA / "figure3_theta_adaptation.csv", index=False)

    # ------------------------------------------------ Figure 4: kappa adaptation (all scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), sharex=True)
    for ax, scenario in zip(axes.ravel(), SCENARIOS):
        for cond in ("B1", "B2", "B3", "B4"):
            g = st[(st.scenario_id == scenario) & (st.condition == cond)]
            ax.plot(g.episode, g.kappa_norm, color=COLORS[cond], lw=1.3, label=LABEL_SHORT[cond])
        ax.axvline(PERTURB_EPISODE, color="k", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(scenario)
        ax.set_ylabel("$\\|\\kappa_t\\|_F$ (mean)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 4: $\\kappa$ (memory) norm trajectories, mean over 20 seeds; "
                 "movement scales with perturbation magnitude", y=0.995)
    fig.tight_layout()
    save(fig, "figure4_kappa_adaptation")
    st.to_csv(FIGDATA / "figure4_kappa_adaptation.csv", index=False)

    # ------------------------------------------------ Figure 5: PPEE_10 distributions (ECDF)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=True)
    ecdf_rows = []
    for ax, scenario in zip(axes, SCENARIOS):
        for cond in CONDITION_ORDER:
            v = np.sort(runs[(runs.scenario_id == scenario) &
                             (runs.condition == cond)]["ppee_10"].to_numpy(dtype=float))
            p = np.arange(1, len(v) + 1) / len(v)
            ax.step(v, p, where="post", color=COLORS[cond], lw=1.4, label=LABEL_SHORT[cond])
            for x, y in zip(v, p):
                ecdf_rows.append({"scenario": scenario, "condition": cond, "ppee_10": x,
                                  "ecdf": y})
        ax.set_title(f"{scenario}  ($n$=20)")
        ax.set_xlabel("PPEE$_{10}$")
    axes[0].set_ylabel("ECDF")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Figure 5: ECDFs of PPEE$_{10}$ per scenario (zero-inflated; the jump at 0 is "
                 "the zero fraction). Left = better.", y=1.04)
    fig.tight_layout()
    save(fig, "figure5_ppee_distribution")
    pd.DataFrame(ecdf_rows).to_csv(FIGDATA / "figure5_ppee_distribution.csv", index=False)

    # ------------------------------------------------ Figure 6: paired effects with CIs
    diffs = pd.read_csv(FIGDATA / "paired_differences_ppee10.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True)
    order = ["B0-B1", "B0-B2", "B1-B3", "B2-B4", "B1-B2", "B3-B4"]
    for ax, comp in zip(axes.ravel(), order):
        for i, scenario in enumerate(SCENARIOS):
            sub = diffs[(diffs.comparison == comp) & (diffs.scenario == scenario)]
            median = sub["diff"].median()
            # bootstrap CI of the median (deterministic seed, BC percentile)
            rng = np.random.default_rng(20260923)
            boots = np.array([np.median(rng.choice(sub["diff"].to_numpy(), len(sub), replace=True))
                              for _ in range(5000)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            y = len(SCENARIOS) - 1 - i
            ax.plot(sub["diff"], np.full(len(sub), y) + np.linspace(-0.18, 0.18, len(sub)),
                    ".", color=COMP_COLORS[comp], ms=3, alpha=0.45)
            ax.plot([median], [y], "o", color=COMP_COLORS[comp], ms=7)
            ax.plot([lo, hi], [y, y], "|-", color=COMP_COLORS[comp], ms=10, lw=1.4)
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels(SCENARIOS[::-1])
        ax.set_title(comp, color=COMP_COLORS[comp])
        ax.set_xlabel("Paired $\\Delta$ PPEE$_{10}$\n(first $-$ second; <0 = first better)",
                      fontsize=7.5)
    fig.suptitle("Figure 6: Per-seed paired PPEE$_{10}$ differences (dots), median (circle) and "
                 "bootstrap 95% CI (bar); 20 seeds each", y=0.995)
    fig.tight_layout()
    save(fig, "figure6_paired_effects")
    med_rows = []
    for comp in order:
        for scenario in SCENARIOS:
            sub = diffs[(diffs.comparison == comp) & (diffs.scenario == scenario)]["diff"]
            rng = np.random.default_rng(20260923)
            boots = np.array([np.median(rng.choice(sub.to_numpy(), len(sub), replace=True))
                              for _ in range(5000)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            med_rows.append({"comparison": comp, "scenario": scenario,
                             "median_diff": float(sub.median()), "boot_ci_low": float(lo),
                             "boot_ci_high": float(hi), "mean_diff": float(sub.mean()),
                             "n": len(sub)})
    pd.DataFrame(med_rows).to_csv(FIGDATA / "figure6_paired_effects.csv", index=False)

    # ------------------------------------------------ Figure 7: scenario x condition heatmap
    main = pd.read_csv(FIGDATA / "main_ppee_results.csv")
    mat_med = main.pivot(index="condition", columns="scenario", values="median")
    mat_mean = main.pivot(index="condition", columns="scenario", values="mean")
    mat_zero = main.pivot(index="condition", columns="scenario", values="zero_fraction")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
    for ax, mat, title, fmt in (
            (axes[0], mat_med.loc[CONDITION_ORDER], "Median PPEE$_{10}$", "%.4f"),
            (axes[1], mat_mean.loc[CONDITION_ORDER], "Mean PPEE$_{10}$", "%.4f"),
            (axes[2], mat_zero.loc[CONDITION_ORDER], "$P($PPEE$_{10}=0)$", "%.2f")):
        im = ax.imshow(mat.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=20, ha="right")
        ax.set_yticks(range(len(mat.index)), mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt % mat.iloc[i, j], ha="center", va="center",
                        color="white" if mat.iloc[i, j] > mat.to_numpy().max() / 2 else "w",
                        fontsize=8)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Figure 7: Scenario $\\times$ condition overview of the primary metric "
                 "(lower = better)", y=1.02)
    fig.tight_layout()
    save(fig, "figure7_scenario_heatmap")
    main.to_csv(FIGDATA / "figure7_scenario_heatmap.csv", index=False)

    # ------------------------------------------------ Figure 8: adaptation diagnostics
    churn = pd.read_csv(FIGDATA / "assignment_churn_by_window.csv")
    diag = pd.read_csv(common.ARTIFACTS / "tables" / "adaptation_diagnostics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))

    ax = axes[0, 0]
    churn_all = (eps.groupby(["scenario_id", "condition"])["reconfig_cost"].mean()
                 .reset_index())
    for cond in ("B1", "B2", "B3", "B4"):
        vals = [churn_all[(churn_all.scenario_id == s) & (churn_all.condition == cond)]
                .reconfig_cost.mean() for s in SCENARIOS]
        ax.plot(range(4), vals, color=COLORS[cond], marker=MARKERS[cond], ms=5, lw=1.2,
                label=LABEL_SHORT[cond])
    ax.set_xticks(range(4), [s.replace(" ", "\n") for s in SCENARIOS])
    ax.set_ylabel("Assignment churn (mean reconfig.\nfraction, all episodes)")
    ax.set_title("A: Assignment churn (adaptive > static)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    for scenario in SCENARIOS:
        sub = diag[(diag.scenario == scenario)]
        ax.plot(sub.condition, sub.kappa_change_post_mean, marker="o", ms=5,
                label=scenario)
    ax.set_ylabel("$\\|\\kappa_{34}-\\kappa_{25}\\|$ (mean)")
    ax.set_title("B: $\\kappa$ movement over the PPEE window")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    for scenario in SCENARIOS:
        sub = diag[(diag.scenario == scenario)]
        ax.plot(sub.condition, sub.theta_change_post_mean, marker="o", ms=5, label=scenario)
    ax.set_ylabel("$\\|\\Theta_{34}-\\Theta_{25}\\|$ (mean)")
    ax.set_title("C: $\\Theta$ movement over the PPEE window")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    theta_acc = pd.read_csv(FIGDATA / "theta_accuracy_by_condition.csv")
    for scenario in SCENARIOS:
        sub = theta_acc[theta_acc.scenario == scenario]
        ax.plot(sub.condition, sub.final_theta_diff_mean, marker="o", ms=5, label=scenario)
    ax.set_ylabel("Final $\\|\\Theta - G_{gt}\\|$ (mean)")
    ax.set_title("D: $\\Theta$ accuracy vs ground truth (final episode)")
    ax.legend(frameon=False, fontsize=7)
    for ax in axes.ravel():
        ax.set_xlabel("Condition")
    fig.suptitle("Figure 8: Adaptation-state diagnostics (mean over 20 seeds)", y=0.995)
    fig.tight_layout()
    save(fig, "figure8_adaptation_diagnostics")
    churn.to_csv(FIGDATA / "figure8_adaptation_diagnostics_churn.csv", index=False)
    diag[["scenario", "condition", "kappa_change_post_mean", "theta_change_post_mean"]].to_csv(
        FIGDATA / "figure8_adaptation_diagnostics_state.csv", index=False)
    theta_acc.to_csv(FIGDATA / "figure8_adaptation_diagnostics_theta_accuracy.csv", index=False)

    print("all figures done")


if __name__ == "__main__":
    main()
