
from __future__ import annotations

import itertools
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis.common as common
from analysis.common import (COMPARISONS, CONDITION_LABEL, CONDITION_ORDER, DEFINITION_SUFFIX,
                    PPEE_WINDOW, SCENARIOS, SEEDS, paired_differences, load_runs,
                    load_episodes, summarize_metric)

RNG_SEED = 20260923
BOOTSTRAP_DRAWS = 20000


# --------------------------------------------------------------------------- tests
def exact_signflip_permutation_p(diff: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation test on the mean of paired differences.

    n = 20 seeds -> 2^20 = 1,048,576 sign patterns, enumerated exactly.
    Returns (number of patterns with |mean| >= |observed mean|) / 2^n.
    The observed configuration is part of the enumeration, so the minimum
    attainable p-value is 1/2^n.
    """
    n = len(diff)
    if n == 0 or np.abs(np.mean(diff)) < 1e-15:
        return 1.0
    obs = abs(float(np.mean(diff)))
    total = 1 << n
    chunk = 1 << 16
    count = 0
    arange_n = np.arange(n)
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        signs = 1.0 - 2.0 * ((idx[:, None] >> arange_n) & 1).astype(np.float64)
        means = np.abs(signs @ diff / n)
        count += int(np.sum(means >= obs - 1e-12))
    return count / total


def wilcoxon_p(diff: np.ndarray) -> tuple[float, int, int]:
    """Wilcoxon signed-rank p-value; zero differences are dropped (zero_method='wilcox').

    Returns (p, n_nonzero, n_zero).
    """
    nonzero = diff[np.abs(diff) > 1e-12]
    n_zero = int(len(diff) - len(nonzero))
    if len(nonzero) == 0:
        return 1.0, 0, n_zero
    res = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    return float(res.pvalue), int(len(nonzero)), n_zero


def cohens_dz(diff: np.ndarray) -> float:
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def bootstrap_ci(diff: np.ndarray, statistic=np.mean, draws: int = BOOTSTRAP_DRAWS,
                 seed: int = RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n < 2:
        v = float(statistic(diff)) if n else np.nan
        return v, v
    idx = rng.integers(0, n, size=(draws, n))
    stats_boot = statistic(diff[idx], axis=1)
    return float(np.percentile(stats_boot, 2.5)), float(np.percentile(stats_boot, 97.5))


def t_ci(diff: np.ndarray) -> tuple[float, float]:
    n = len(diff)
    if n < 2:
        v = float(np.mean(diff)) if n else np.nan
        return v, v
    sem = float(stats.sem(diff))
    if sem < 1e-15:
        m = float(np.mean(diff))
        return m, m
    h = sem * stats.t.ppf(0.975, df=n - 1)
    return float(np.mean(diff) - h), float(np.mean(diff) + h)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(np.asarray(p_values))
    sorted_p = np.asarray(p_values)[order]
    adjusted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out.tolist()


# --------------------------------------------------------------------------- pipeline
def paired_comparison_row(runs: pd.DataFrame, metric: str, comp_name: str, cond_a: str,
                          cond_b: str, family: str, scenario: str) -> dict:
    pdiff = paired_differences(runs, metric, cond_a, cond_b, scenario)
    diff = pdiff["diff"].to_numpy(dtype=float)
    n = len(diff)
    mean_d = float(np.mean(diff))
    med_d = float(np.median(diff))
    sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    t_lo, t_hi = t_ci(diff)
    b_mean_lo, b_mean_hi = bootstrap_ci(diff, np.mean)
    b_med_lo, b_med_hi = bootstrap_ci(diff, np.median)
    perm_p = exact_signflip_permutation_p(diff)
    wil_p, n_nonzero, n_zero = wilcoxon_p(diff)
    wins_a = int(np.sum(diff < -1e-12))
    wins_b = int(np.sum(diff > 1e-12))
    ties = n - wins_a - wins_b
    return {
        "family": family,
        "comparison": comp_name,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "scenario": scenario,
        "metric": metric,
        "n_pairs": n,
        "mean_a": float(np.mean(runs[(runs.condition == cond_a) & (runs.scenario_id == scenario)][metric])),
        "mean_b": float(np.mean(runs[(runs.condition == cond_b) & (runs.scenario_id == scenario)][metric])),
        "mean_diff": mean_d,
        "median_diff": med_d,
        "sd_diff": sd_d,
        "ci95_t_low": t_lo,
        "ci95_t_high": t_hi,
        "ci95_boot_mean_low": b_mean_lo,
        "ci95_boot_mean_high": b_mean_hi,
        "ci95_boot_median_low": b_med_lo,
        "ci95_boot_median_high": b_med_hi,
        "cohens_dz": cohens_dz(diff),
        "wilcoxon_p": wil_p,
        "wilcoxon_nonzero_pairs": n_nonzero,
        "wilcoxon_zero_pairs": n_zero,
        "permutation_p_exact": perm_p,
        "wins_a_better": wins_a,
        "wins_b_better": wins_b,
        "ties": ties,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_path_arguments(parser)
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)
    runs = load_runs()
    eps = load_episodes()
    artifacts = common.ARTIFACTS
    tables = artifacts / "tables"
    figdata = artifacts / "figure_data"
    tables.mkdir(exist_ok=True)
    figdata.mkdir(exist_ok=True)

    # ---------------------------------------------------------- Table 3: main PPEE results
    rows = []
    for scenario in SCENARIOS:
        for cond in CONDITION_ORDER:
            s = summarize_metric(runs[(runs.scenario_id == scenario) & (runs.condition == cond)]["ppee_10"])
            rows.append({"scenario": scenario, "condition": cond,
                         "label": CONDITION_LABEL[cond], **s})
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(tables / "main_ppee_results.csv", index=False)
    main_tbl.to_csv(figdata / "main_ppee_results.csv", index=False)

    # aggregate over scenarios (descriptive only)
    agg_rows = []
    for cond in CONDITION_ORDER:
        s = summarize_metric(runs[runs.condition == cond]["ppee_10"])
        agg_rows.append({"scenario": "ALL (descriptive)", "condition": cond,
                         "label": CONDITION_LABEL[cond], **s})
    pd.DataFrame(agg_rows).to_csv(figdata / "main_ppee_results_aggregate.csv", index=False)

    # ---------------------------------------------------------- primary paired comparisons
    comp_rows = []
    for comp_name, a, b, family in COMPARISONS:
        for scenario in SCENARIOS:
            comp_rows.append(paired_comparison_row(runs, "ppee_10", comp_name, a, b, family, scenario))

    # Holm within each pre-specified family (8 tests per family: 2 comparisons x 4 scenarios)
    comp_df = pd.DataFrame(comp_rows)
    comp_df["holm_p_within_family"] = np.nan
    for family in comp_df.family.unique():
        mask = comp_df.family == family
        comp_df.loc[mask, "holm_p_within_family"] = holm_bonferroni(
            comp_df.loc[mask, "permutation_p_exact"].tolist())
    # exploratory alternative: Holm within family x scenario (4 tests)
    comp_df["holm_p_within_family_scenario"] = np.nan
    for (family, scenario), mask in comp_df.groupby(["family", "scenario"]).groups.items():
        idx = list(mask)
        comp_df.loc[idx, "holm_p_within_family_scenario"] = holm_bonferroni(
            comp_df.loc[idx, "permutation_p_exact"].tolist())
    comp_df.to_csv(tables / "statistical_comparisons.csv", index=False)

    # ---------------------------------------------------------- paired differences export (Fig 6)
    diff_rows = []
    for comp_name, a, b, family in COMPARISONS:
        for scenario in SCENARIOS:
            pdiff = paired_differences(runs, "ppee_10", a, b, scenario)
            for _, r in pdiff.iterrows():
                diff_rows.append({"comparison": comp_name, "scenario": scenario,
                                  "seed": int(r["seed"]), "diff": float(r["diff"])})
    pd.DataFrame(diff_rows).to_csv(figdata / "paired_differences_ppee10.csv", index=False)

    # ---------------------------------------------------------- secondary metrics (exploratory)
    secondary = {}
    for metric in ("cumulative_excess_energy", "recovery_time", "mean_external_energy",
                   "performance_drop"):
        sec_rows = []
        for comp_name, a, b, family in COMPARISONS:
            for scenario in SCENARIOS:
                sec_rows.append(paired_comparison_row(runs, metric, comp_name, a, b, family, scenario))
        sec_df = pd.DataFrame(sec_rows)
        sec_df["holm_p_within_family"] = holm_bonferroni(sec_df["permutation_p_exact"].tolist())
        secondary[metric] = sec_df
    sec_all = pd.concat(secondary.values(), ignore_index=True)
    sec_all.insert(0, "status", "secondary/exploratory")
    sec_all.to_csv(tables / "secondary_metric_comparisons.csv", index=False)

    # ---------------------------------------------------------- B0 vs energy mean decomposition
    decomp_rows = []
    for scenario in SCENARIOS:
        means = {c: float(runs[(runs.scenario_id == scenario) & (runs.condition == c)]["ppee_10"].mean())
                 for c in CONDITION_ORDER}
        decomp_rows.append({
            "scenario": scenario,
            "B0_mean_ppee10": means["B0"],
            "B1_mean_ppee10": means["B1"], "B2_mean_ppee10": means["B2"],
            "B3_mean_ppee10": means["B3"], "B4_mean_ppee10": means["B4"],
            "formulation_effect_greedy_B1_minus_B0": means["B1"] - means["B0"],
            "formulation_effect_sa_B2_minus_B0": means["B2"] - means["B0"],
            "adaptation_effect_greedy_B3_minus_B1": means["B3"] - means["B1"],
            "adaptation_effect_sa_B4_minus_B2": means["B4"] - means["B2"],
            "solver_effect_static_B2_minus_B1": means["B2"] - means["B1"],
            "solver_effect_adaptive_B4_minus_B3": means["B4"] - means["B3"],
        })
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(figdata / "effect_decomposition_means.csv", index=False)

    # ---------------------------------------------------------- episode-level trajectories
    eps = eps.copy()
    eps["excess"] = np.maximum(0.0, eps.external_energy - eps.reference_energy)
    traj = (eps.groupby(["scenario_id", "condition", "episode"])
            .agg(excess_mean=("excess", "mean"), excess_median=("excess", "median"),
                 excess_sd=("excess", "std"), external_mean=("external_energy", "mean"),
                 external_median=("external_energy", "median"),
                 external_sd=("external_energy", "std"),
                 n=("excess", "size"))
            .reset_index())
    traj["excess_se"] = traj.excess_sd / np.sqrt(traj.n)
    traj.to_csv(figdata / "trajectory_episode_means.csv", index=False)

    # paired static-vs-adaptive excess trajectories (Fig 2) and full conditions (Fig 1)
    per_seed_excess = (eps.groupby(["scenario_id", "condition", "seed", "episode"])
                       .excess.mean().reset_index())
    per_seed_excess.to_csv(figdata / "per_seed_episode_excess.csv", index=False)

    pair_traj_rows = []
    for scenario in SCENARIOS:
        sub = per_seed_excess[per_seed_excess.scenario_id == scenario]
        for comp_name, a, b in (("B3-B1", "B3", "B1"), ("B4-B2", "B4", "B2")):
            wide = sub[sub.condition.isin([a, b])].pivot_table(
                index=["seed", "episode"], columns="condition", values="excess")
            wide = wide.dropna()
            d = (wide[a] - wide[b]).rename("diff").reset_index()
            g = d.groupby("episode")["diff"].agg(["mean", "std", "size"]).reset_index()
            g["se"] = g["std"] / np.sqrt(g["size"])
            g["comparison"] = comp_name
            g["scenario"] = scenario
            pair_traj_rows.append(g[["comparison", "scenario", "episode", "mean", "se", "size"]])
    pd.concat(pair_traj_rows).to_csv(figdata / "static_vs_adaptive_excess_trajectory.csv", index=False)

    # ---------------------------------------------------------- adaptation-state trajectories (from episodes.csv)
    state_cols = ["kappa_norm", "theta_norm", "theta_diff_norm", "delta_kappa_norm", "delta_theta_norm"]
    state_traj = (eps[eps.condition != "B0"]
                  .groupby(["scenario_id", "condition", "episode"])[state_cols]
                  .mean().reset_index())
    state_traj.to_csv(figdata / "state_trajectories_episode_means.csv", index=False)

    # assignment churn (reconfig cost) by window
    def window(e: int) -> str:
        if e < 15:
            return "pre(0-14)"
        if e < 25:
            return "pre-window(15-24)"
        if e < 35:
            return "post-window(25-34)"
        return "late(35-49)"
    eps["window"] = eps.episode.map(window)
    churn = (eps.groupby(["scenario_id", "condition", "window"])
             .agg(mean_reconfig=("reconfig_cost", "mean"), sd_reconfig=("reconfig_cost", "std"),
                  mean_conflicts=("co_assignment_conflicts", "mean"),
                  mean_coordination=("coordination_score", "mean"),
                  mean_load_balance=("load_balance", "mean"), n=("reconfig_cost", "size"))
             .reset_index())
    churn.to_csv(figdata / "assignment_churn_by_window.csv", index=False)
    churn.to_csv(tables / "assignment_churn_by_window.csv", index=False)

    # ---------------------------------------------------------- cross-check against frozen statistics.csv
    frozen = pd.read_csv(common.RESULTS / "statistics.csv")
    frozen_map = {}
    label_to_comp = {
        "B0 vs B1 (ConvGreedy vs StaticGreedy)": "B0-B1",
        "B0 vs B2 (ConvGreedy vs StaticSA)": "B0-B2",
        "B1 vs B3 (Greedy Static vs Adaptive)": "B1-B3",
        "B2 vs B4 (SA Static vs Adaptive)": "B2-B4",
        "B1 vs B2 (Static Greedy vs SA)": "B1-B2",
        "B3 vs B4 (Adaptive Greedy vs SA)": "B3-B4",
    }
    # frozen: adaptive/static or greedy/sa orientation differs per family; map to common sign
    for _, r in frozen.iterrows():
        comp = label_to_comp[r.comparison]
        family = str(r.hypothesis_family).split(":")[1].strip().lower()
        frozen_map[(comp, r.scenario_id)] = {
            "mean_difference_frozen": float(r.mean_difference),
            "permutation_p_frozen": float(r.permutation_p_val),
            "holm_p_frozen": float(r.p_val_adjusted),
        }
    verify_rows = []
    for _, r in comp_df.iterrows():
        key = (r.comparison, r.scenario)
        f = frozen_map.get(key)
        if f is None:
            continue
        comp, a, b, family = next((cn, ca, cb, fam) for cn, ca, cb, fam in COMPARISONS
                                  if cn == r.comparison)
        # frozen orientation for adaptation family is adaptive - static = B1-B3 orientation? No:
        # frozen B1 vs B3 uses adapt - static = (B3 - B1) = -(B1-B3). Solver family uses greedy-sa = B1-B2.
        if family == "adaptation":
            frozen_sign = -1.0
        elif family == "formulation":
            frozen_sign = 1.0  # frozen b0 - target = B0 - B1 (same as ours)
        else:
            frozen_sign = 1.0
        verify_rows.append({
            "comparison": r.comparison, "scenario": r.scenario,
            "our_mean_diff": r.mean_diff,
            "frozen_mean_diff_same_sign": frozen_sign * f["mean_difference_frozen"],
            "our_perm_p": r.permutation_p_exact,
            "frozen_perm_p": f["permutation_p_frozen"],
        })
    verify_df = pd.DataFrame(verify_rows)
    verify_df["mean_diff_abs_dev"] = (verify_df.our_mean_diff - verify_df.frozen_mean_diff_same_sign).abs()
    verify_df["perm_p_abs_dev"] = (verify_df.our_perm_p - verify_df.frozen_perm_p).abs()
    verify_df.to_csv(figdata / "frozen_statistics_crosscheck.csv", index=False)

    # ---------------------------------------------------------- headline JSON
    def get(comp, scenario, field):
        return float(comp_df[(comp_df.comparison == comp) & (comp_df.scenario == scenario)][field].iloc[0])

    headline = {
        "n_runs": int(len(runs)),
        "n_episodes": int(len(eps)),
        "main_results": main_tbl.to_dict(orient="records"),
        "comparisons": comp_df.to_dict(orient="records"),
        "crosscheck_max_mean_dev": float(verify_df.mean_diff_abs_dev.max()),
        "crosscheck_max_perm_p_dev": float(verify_df.perm_p_abs_dev.max()),
        "zero_fractions": {
            f"{r.condition}|{r.scenario}": r.zero_fraction for r in main_tbl.itertuples()},
        "definition_note": DEFINITION_SUFFIX,
    }
    (common.ARTIFACTS / "stats_summary.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")

    # ---------------------------------------------------------- console digest
    pd.set_option("display.width", 200)
    print("=== Main PPEE_10 results (n / median / mean / zero-frac) ===")
    print(main_tbl[["scenario", "condition", "n", "median", "mean", "sd", "zero_fraction"]]
          .to_string(index=False))
    print()
    print("=== Primary comparisons (permutation p exact, Holm within family) ===")
    show = comp_df[["family", "comparison", "scenario", "mean_diff", "median_diff",
                    "ci95_t_low", "ci95_t_high", "cohens_dz", "wilcoxon_p",
                    "permutation_p_exact", "holm_p_within_family",
                    "wins_a_better", "wins_b_better", "ties"]]
    print(show.to_string(index=False))
    print()
    print(f"Cross-check vs frozen statistics.csv: max |mean diff dev| = "
          f"{verify_df.mean_diff_abs_dev.max():.3e}, max |perm p dev| = "
          f"{verify_df.perm_p_abs_dev.max():.4f} (frozen p is Monte-Carlo, 10k draws)")


if __name__ == "__main__":
    main()
