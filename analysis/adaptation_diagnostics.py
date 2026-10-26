
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis.common as common
from analysis.common import (CONDITION_ORDER, PERTURB_EPISODE, SCENARIOS, SEEDS,
                    load_episodes, load_runs)


def paired_diff(runs: pd.DataFrame, metric: str, a: str, b: str, scenario: str) -> np.ndarray:
    sub = runs[(runs.scenario_id == scenario) & (runs.condition.isin([a, b]))]
    wide = sub.pivot_table(index="seed", columns="condition", values=metric, aggfunc="first").dropna()
    return (wide[a] - wide[b]).to_numpy(dtype=float)


def perm_p(diff: np.ndarray) -> float:
    n = len(diff)
    if n == 0 or abs(np.mean(diff)) < 1e-15:
        return 1.0
    obs = abs(float(np.mean(diff)))
    total = 1 << n
    arange_n = np.arange(n)
    chunk = 1 << 16
    count = 0
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        signs = 1.0 - 2.0 * ((idx[:, None] >> arange_n) & 1).astype(np.float64)
        count += int(np.sum(np.abs(signs @ diff / n) >= obs - 1e-12))
    return count / total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_path_arguments(parser)
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)
    runs = load_runs()
    eps = load_episodes()
    tables = common.ARTIFACTS / "tables"
    figdata = common.ARTIFACTS / "figure_data"
    tables.mkdir(exist_ok=True)
    figdata.mkdir(exist_ok=True)

    # ------------------------------------------------ Θ accuracy vs ground truth (episodes)
    # theta_diff_norm = ||Theta_t - G_gt(t)|| stored per episode for B1-B4.
    theta_rows = []
    for scenario in SCENARIOS:
        for cond in ("B1", "B2", "B3", "B4"):
            sub = eps[(eps.scenario_id == scenario) & (eps.condition == cond)]
            pre = sub[sub.episode < PERTURB_EPISODE]
            post = sub[sub.episode >= PERTURB_EPISODE]
            theta_rows.append({
                "scenario": scenario, "condition": cond,
                "mean_theta_diff_pre": float(pre.theta_diff_norm.mean()),
                "mean_theta_diff_post": float(post.theta_diff_norm.mean()),
                "final_theta_diff_mean": float(sub[sub.episode == 49].theta_diff_norm.mean()),
                "mean_theta_norm_pre": float(pre.theta_norm.mean()),
                "mean_theta_norm_post": float(post.theta_norm.mean()),
            })
    theta_df = pd.DataFrame(theta_rows)
    theta_df.to_csv(figdata / "theta_accuracy_by_condition.csv", index=False)

    # ------------------------------------------------ Table 5: adaptation diagnostics
    diag_rows = []
    for scenario in SCENARIOS:
        for cond in CONDITION_ORDER:
            sub_runs = runs[(runs.scenario_id == scenario) & (runs.condition == cond)]
            sub_eps = eps[(eps.scenario_id == scenario) & (eps.condition == cond)]
            pre_eps = sub_eps[sub_eps.episode < PERTURB_EPISODE]
            post_eps = sub_eps[sub_eps.episode >= PERTURB_EPISODE]
            late_eps = sub_eps[sub_eps.episode >= 35]
            row = {
                "scenario": scenario, "condition": cond,
                # adaptation magnitude over the PPEE window (runs.csv, state snapshots ep25 -> ep34)
                "kappa_change_post_mean": sub_runs.kappa_change_post.mean(),
                "theta_change_post_mean": sub_runs.theta_change_post.mean(),
                "theta_change_post_sd": sub_runs.theta_change_post.std(),
                # stationary drift: per-seed |Theta_49 - Theta_0| and |kappa_49 - kappa_0|
                # proxies from norms at final episode
                "kappa_norm_final_mean": sub_eps[sub_eps.episode == 49].kappa_norm.mean(),
                "theta_norm_final_mean": sub_eps[sub_eps.episode == 49].theta_norm.mean(),
                "theta_diff_norm_final_mean": sub_eps[sub_eps.episode == 49].theta_diff_norm.mean(),
                # assignment churn (reconfig cost) by window
                "churn_pre_window_mean": pre_eps[pre_eps.episode >= 15].reconfig_cost.mean(),
                "churn_post_window_mean": post_eps[post_eps.episode < 35].reconfig_cost.mean(),
                "churn_late_mean": late_eps.reconfig_cost.mean(),
                "churn_stationary_drift": sub_eps.reconfig_cost.mean(),
                # co-assignment conflicts & coordination
                "co_assignment_conflicts_mean": sub_eps.co_assignment_conflicts.mean(),
                "coordination_score_mean": sub_eps.coordination_score.mean(),
                # recovery
                "recovery_time_mean": sub_runs.recovery_time.mean(),
                "recovery_time_median": sub_runs.recovery_time.median(),
                "recovery_time_censored_frac": float((sub_runs.recovery_time == 25.0).mean()),
                # convergence / stability (late-window std of external energy; late reconfig)
                "convergence_mean": sub_runs.convergence.mean(),
                "stability_mean": sub_runs.stability.mean(),
                # solver behavior
                "total_energy_evaluations_mean": sub_runs.total_energy_evaluations.mean(),
                "mean_accepted_moves_per_episode": sub_eps.accepted_moves.mean(),
                "mean_iterations_per_episode": sub_eps.iterations.mean(),
                "mean_runtime_sec_per_episode": sub_eps.runtime_sec.mean(),
            }
            diag_rows.append(row)
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(tables / "adaptation_diagnostics.csv", index=False)

    # ------------------------------------------------ paired diagnostics tests (B3-B1, B4-B2)
    tests_rows = []
    for scenario in SCENARIOS:
        for metric, label in (("kappa_change_post", "kappa movement ep25-34"),
                              ("theta_change_post", "theta movement ep25-34"),
                              ("recovery_time", "recovery_time (25 = censored)"),
                              ("stability", "late-window reconfig (stability)"),
                              ("convergence", "late-window energy std (convergence)"),
                              ("mean_co_assignment_conflicts", "mean co-assignment conflicts")):
            for a, b, comp in (("B3", "B1", "B3-B1"), ("B4", "B2", "B4-B2")):
                d = paired_diff(runs, metric, a, b, scenario)
                tests_rows.append({
                    "scenario": scenario, "metric": label, "comparison": comp,
                    "mean_diff": float(np.mean(d)), "median_diff": float(np.median(d)),
                    "ci95_low": float(np.mean(d) - 1.96 * np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                    "ci95_high": float(np.mean(d) + 1.96 * np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                    "permutation_p": perm_p(d),
                })
    tests_df = pd.DataFrame(tests_rows)
    tests_df.to_csv(tables / "adaptation_diagnostics_tests.csv", index=False)

    # ------------------------------------------------ per-seed churn paired (for Fig 8)
    per_seed = (eps.groupby(["scenario_id", "condition", "seed"])
                .agg(mean_reconfig=("reconfig_cost", "mean"),
                     mean_conflicts=("co_assignment_conflicts", "mean"),
                     kappa_norm_final=("kappa_norm", "last"),
                     theta_norm_final=("theta_norm", "last"),
                     theta_diff_final=("theta_diff_norm", "last"))
                .reset_index())
    churn_rows = []
    for scenario in SCENARIOS:
        for a, b, comp in (("B3", "B1", "B3-B1"), ("B4", "B2", "B4-B2")):
            sub = per_seed[(per_seed.scenario_id == scenario) & (per_seed.condition.isin([a, b]))]
            wide = sub.pivot_table(index="seed", columns="condition",
                                   values=["mean_reconfig", "mean_conflicts",
                                           "kappa_norm_final", "theta_diff_final"]).dropna()
            d_churn = (wide[("mean_reconfig", a)] - wide[("mean_reconfig", b)]).to_numpy()
            d_conf = (wide[("mean_conflicts", a)] - wide[("mean_conflicts", b)]).to_numpy()
            d_kappa = (wide[("kappa_norm_final", a)] - wide[("kappa_norm_final", b)]).to_numpy()
            d_theta = (wide[("theta_diff_final", a)] - wide[("theta_diff_final", b)]).to_numpy()
            churn_rows.append({
                "scenario": scenario, "comparison": comp,
                "mean_churn_diff": float(np.mean(d_churn)),
                "churn_perm_p": perm_p(d_churn),
                "mean_conflicts_diff": float(np.mean(d_conf)),
                "conflicts_perm_p": perm_p(d_conf),
                "mean_kappa_norm_diff": float(np.mean(d_kappa)),
                "kappa_perm_p": perm_p(d_kappa),
                "mean_theta_accuracy_diff": float(np.mean(d_theta)),
                "theta_accuracy_perm_p": perm_p(d_theta),
                "wins_churn_a_higher": int(np.sum(d_churn > 1e-12)),
                "wins_churn_b_higher": int(np.sum(d_churn < -1e-12)),
                "ties": int(len(d_churn) - np.sum(d_churn > 1e-12) - np.sum(d_churn < -1e-12)),
            })
    churn_paired = pd.DataFrame(churn_rows)
    churn_paired.to_csv(figdata / "churn_paired_summary.csv", index=False)
    per_seed.to_csv(figdata / "per_seed_state_summary.csv", index=False)

    # ------------------------------------------------ Θ-accuracy hypothesis tests (B3 vs B1, B4 vs B2)
    acc_rows = []
    for scenario in SCENARIOS:
        for a, b, comp in (("B3", "B1", "B3-B1"), ("B4", "B2", "B4-B2")):
            sub = per_seed[(per_seed.scenario_id == scenario) & (per_seed.condition.isin([a, b]))]
            wide = sub.pivot_table(index="seed", columns="condition", values="theta_diff_final").dropna()
            d = (wide[a] - wide[b]).to_numpy(dtype=float)
            acc_rows.append({
                "scenario": scenario, "comparison": comp,
                "mean_final_theta_diff_A_minus_B": float(np.mean(d)),
                "median": float(np.median(d)),
                "permutation_p": perm_p(d),
                "wins_A_lower_error": int(np.sum(d < -1e-12)),
                "wins_B_lower_error": int(np.sum(d > 1e-12)),
                "ties": int(len(d) - np.sum(np.abs(d) > 1e-12)),
            })
    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(tables / "theta_accuracy_tests.csv", index=False)

    # ------------------------------------------------ digest
    digest = {
        "theta_accuracy": theta_df.to_dict(orient="records"),
        "theta_accuracy_tests": acc_df.to_dict(orient="records"),
        "churn_paired": churn_paired.to_dict(orient="records"),
        "diagnostics_tests_selected": tests_df.to_dict(orient="records"),
    }
    (common.ARTIFACTS / "adaptation_diagnostics_summary.json").write_text(
        json.dumps(digest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print("=== Theta accuracy (mean ||Theta_t - G_gt||) ===")
    print(theta_df.to_string(index=False))
    print()
    print("=== Theta accuracy paired tests (final episode) ===")
    print(acc_df.to_string(index=False))
    print()
    print("=== Churn paired summary ===")
    print(churn_paired.to_string(index=False))
    print()
    print("=== Selected diagnostics (adaptive minus static) ===")
    sel = tests_df[tests_df.metric.isin(["kappa movement ep25-34", "theta movement ep25-34"])]
    print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
