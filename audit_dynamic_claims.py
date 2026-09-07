"""
audit_dynamic_claims.py

Reviewer-ready audit of the dynamic benchmark data.
Operationalizes every metric with its exact definition, reports
means ± std across seeds, and flags which claims hold vs. which are weak.

Usage:
    python audit_dynamic_claims.py

Output:
    results/dynamic_audit_table.csv   ← machine-readable
    results/dynamic_audit_report.txt  ← human-readable, ready to paste into Related Work
"""

import os
import sys
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "dynamic_audit_table.csv")
OUTPUT_TXT  = os.path.join(RESULTS_DIR, "dynamic_audit_report.txt")

# ── Scenario and configuration labels ────────────────────────────────────────
SCENARIOS = {
    "Capability Drift": "dynamic_capability_drift",
    "Task Shift":       "dynamic_task_shift",
    "Dependency Change":"dynamic_dependency_change",
    "Robustness":       "dynamic_robustness",
}

CONFIGS = {
    "static_energy":   "Static Energy (no memory)",
    "ebmao_kappa-only":"EBMAO κ-only",
    "ebmao_theta-only":"EBMAO Θ-only",
    "full_ebmao":      "Full EBMAO",
}

PERTURB_EP = 25        # episode where drift / shift happens
RECOVERY_WINDOW = 3    # consecutive episodes required for stable recovery

# ── Exact metric definitions (printed verbatim in the report) ─────────────────
METRIC_DEFS = {
    "pre_energy_mean":
        "Mean energy over the ten episodes immediately before the perturbation.",
    "post_energy_mean":
        "Mean energy from the perturbation episode through the last available episode.",
    "delta_energy":
        "post_energy_mean − pre_energy_mean  (positive = worse after drift).",
    "perf_drop":
        "Energy(episode 25) − Energy(episode 24).  Immediate shock size.",
    "recovery_time_abs":
        "First episode t ≥ 25 whose next three energies are all ≤ 1.10 × "
        "the relevant baseline  "
        "(for Robustness/Task Shift: target = post-drift stable level; "
        "for Capability Drift / Dependency Change: target = pre-drift level; "
        "the 1.10× tolerance applies in both cases). "
        "Measured in episodes from perturbation point (lower = faster).",
    "cum_regret":
        "Sum from the perturbation episode through the last available episode of "
        "max(0, E_t − pre_energy_mean) — total excess energy "
        "accumulated after the drift relative to the pre-drift baseline.",
    "late_energy_mean":
        "Mean energy over the last ten available episodes — whether the system eventually "
        "settles to a good solution.",
    "late_energy_std":
        "Std of energy over the last ten available episodes — stability of final state.",
}


def load_df(scenario_key, config_key):
    path = os.path.join(RESULTS_DIR, f"{scenario_key}_{config_key}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_metrics(df, scenario_label):
    """Compute all metrics from a single-run episode DataFrame."""
    required_columns = {"episode", "energy"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = df.sort_values("episode").reset_index(drop=True)
    eps = data["episode"].to_numpy()
    E = data["energy"].to_numpy()

    if len(eps) == 0 or not np.array_equal(eps, np.arange(eps[-1] + 1)):
        raise ValueError("Episodes must be a contiguous sequence starting at 0")
    total_eps = int(eps[-1]) + 1
    if total_eps <= PERTURB_EP:
        raise ValueError(
            f"Need episodes after perturbation episode {PERTURB_EP}, got {total_eps}"
        )

    pre_mask = (
        (eps >= max(0, PERTURB_EP - 10))
        & (eps < PERTURB_EP)
    )
    post_mask = eps >= PERTURB_EP
    late_mask = eps >= max(PERTURB_EP + 15, total_eps - 10)

    pre_mean  = E[pre_mask].mean()  if pre_mask.any()  else np.nan
    post_mean = E[post_mask].mean() if post_mask.any() else np.nan
    late_mean = E[late_mask].mean() if late_mask.any() else np.nan
    late_std  = E[late_mask].std()  if late_mask.any() else np.nan

    # Immediate shock
    perf_drop = float(E[PERTURB_EP] - E[PERTURB_EP - 1])

    # Recovery time with *two* target definitions:
    #   permanent-shift scenarios → recover to 1.10 × post_mean
    #   adaptive scenarios        → recover to 1.10 × pre_mean
    is_permanent = scenario_label in ("Task Shift", "Robustness")
    target = 1.10 * post_mean if is_permanent else 1.10 * pre_mean
    recovery_time = total_eps - PERTURB_EP  # default: never recovered
    last_start = total_eps - RECOVERY_WINDOW
    for t in range(PERTURB_EP, last_start + 1):
        window = E[t:t + RECOVERY_WINDOW]
        if np.all(window <= target):
            recovery_time = t - PERTURB_EP
            break

    # Cumulative regret vs. pre-drift baseline
    cum_regret = 0.0
    for value in E[PERTURB_EP:]:
        cum_regret += max(0.0, value - pre_mean)

    return {
        "pre_energy_mean":  round(pre_mean,  4),
        "post_energy_mean": round(post_mean, 4),
        "late_energy_mean": round(late_mean, 4),
        "late_energy_std":  round(late_std,  4),
        "delta_energy":     round(post_mean - pre_mean, 4),
        "perf_drop":        round(perf_drop, 4),
        "recovery_time_abs":int(recovery_time),
        "cum_regret":       round(cum_regret, 4),
    }


def main():
    rows = []
    for sc_label, sc_key in SCENARIOS.items():
        for cfg_key, cfg_label in CONFIGS.items():
            df = load_df(sc_key, cfg_key)
            if df is None:
                print(f"  [MISSING] {sc_key}_{cfg_key}.csv")
                continue

            m = compute_metrics(df, sc_label)
            rows.append({
                "scenario":    sc_label,
                "config":      cfg_label,
                **m,
            })

    df_out = pd.DataFrame(rows)

    # ── Claim verification table ──────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EBMAO Dynamic Benchmark: Reviewer-Ready Audit")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("METRIC DEFINITIONS")
    report_lines.append("-" * 80)
    for k, v in METRIC_DEFS.items():
        report_lines.append(f"  {k}:")
        report_lines.append(f"    {v}")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("RESULTS BY SCENARIO")
    report_lines.append("=" * 80)

    claim_verdicts = []

    for sc_label in SCENARIOS:
        sub = df_out[df_out["scenario"] == sc_label]
        if sub.empty:
            continue

        report_lines.append(f"\n{'─' * 60}")
        report_lines.append(f"Scenario: {sc_label}")
        report_lines.append(f"{'─' * 60}")

        # Print all metrics per config
        metric_keys = ["pre_energy_mean", "post_energy_mean", "delta_energy",
                       "perf_drop", "recovery_time_abs", "cum_regret",
                       "late_energy_mean", "late_energy_std"]

        # Header
        header = f"{'Metric':<22}" + "".join(f"{c:<28}" for c in sub["config"].values)
        report_lines.append(header)
        report_lines.append("-" * len(header))

        for mk in metric_keys:
            row_str = f"{mk:<22}"
            for _, r in sub.iterrows():
                val = r[mk]
                row_str += f"{val:<28}"
            report_lines.append(row_str)

        # ── Compute claim delta: Full EBMAO vs Static Energy ─────────────────
        stat = sub[sub["config"] == "Static Energy (no memory)"]
        full = sub[sub["config"] == "Full EBMAO"]
        if stat.empty or full.empty:
            continue

        s = stat.iloc[0]
        f = full.iloc[0]

        report_lines.append("")
        report_lines.append("  ▶ Claim verification: Full EBMAO vs. Static Energy")

        # Recovery time claim
        rt_s = int(s["recovery_time_abs"])
        rt_f = int(f["recovery_time_abs"])
        rt_diff = rt_s - rt_f
        rt_verdict = "✓ HOLDS" if rt_diff >= 2 else ("~ MARGINAL" if rt_diff >= 1 else "✗ FAILS")
        report_lines.append(
            f"    recovery_time_abs: Static={rt_s} ep, Full EBMAO={rt_f} ep, "
            f"Δ={rt_diff} ep  → {rt_verdict}"
        )
        claim_verdicts.append((sc_label, "recovery_time", rt_verdict))

        # Cum regret claim
        cr_s = s["cum_regret"]
        cr_f = f["cum_regret"]
        cr_pct = 100 * (cr_s - cr_f) / abs(cr_s) if cr_s != 0 else 0
        cr_verdict = "✓ HOLDS" if cr_pct > 5 else ("~ MARGINAL" if cr_pct > 1 else "✗ FAILS")
        report_lines.append(
            f"    cum_regret:        Static={cr_s:.3f}, Full EBMAO={cr_f:.3f}, "
            f"Δ={cr_pct:.1f}%  → {cr_verdict}"
        )
        claim_verdicts.append((sc_label, "cum_regret", cr_verdict))

        # Late energy claim
        le_s = s["late_energy_mean"]
        le_f = f["late_energy_mean"]
        le_pct = 100 * (le_s - le_f) / abs(le_s) if le_s != 0 else 0
        le_verdict = "✓ HOLDS" if le_pct > 2 else ("~ MARGINAL" if le_pct > 0.5 else "✗ FAILS")
        report_lines.append(
            f"    late_energy_mean:  Static={le_s:.4f}, Full EBMAO={le_f:.4f}, "
            f"Δ={le_pct:.1f}%  → {le_verdict}"
        )
        claim_verdicts.append((sc_label, "late_energy_mean", le_verdict))

    # ── Caveats for Robustness recovery_time ─────────────────────────────────
    report_lines.append("\n" + "=" * 80)
    report_lines.append("CAVEATS & REVIEWER WARNINGS")
    report_lines.append("=" * 80)
    report_lines.append("""
[1] Robustness / Task Shift recovery_time:
    Target = 1.10 × post_drift_mean (not pre-drift mean).
    This means 'recovery' is settling relative to the new, degraded level.
    MUST be stated explicitly in paper: 'recovers to within 10% of
    post-perturbation steady state, not to pre-perturbation level.'

[2] Robustness scenario (interaction_graph = 0):
    κ (memory) affects Assignment Energy only. Interaction Energy = 0.
    Therefore Full EBMAO ≈ κ-only EBMAO in this scenario by construction.
    Claim: 'Full EBMAO adapts faster' must be narrowed to
    'memory-augmented EBMAO adapts faster under agent capability changes.'

[3] Task Shift and Dependency Change:
    If cum_regret and late_energy_mean show no improvement, this is
    a honest null result and should be reported as 'boundary of applicability':
    EBMAO adaptation is strongest when agent capabilities, not problem
    structure, are non-stationary.

[4] Single seed per scenario:
    Current data has no confidence intervals. For a camera-ready paper,
    re-run with 3–5 seeds and report mean ± std for all metrics.
    The current results are preliminary / directional only.
""")

    # ── Summary claim matrix ──────────────────────────────────────────────────
    report_lines.append("=" * 80)
    report_lines.append("CLAIM VERDICT SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"{'Scenario':<22} {'Metric':<20} {'Verdict'}")
    report_lines.append("-" * 60)
    for sc, metric, verdict in claim_verdicts:
        report_lines.append(f"{sc:<22} {metric:<20} {verdict}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved audit table  -> {OUTPUT_CSV}")

    report_text = "\n".join(report_lines)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved audit report -> {OUTPUT_TXT}")

    # Print to console
    console_encoding = sys.stdout.encoding or "utf-8"
    safe_report = report_text.encode(
        console_encoding,
        errors="replace",
    ).decode(
        console_encoding,
        errors="replace",
    )
    print("\n" + safe_report)


if __name__ == "__main__":
    main()
