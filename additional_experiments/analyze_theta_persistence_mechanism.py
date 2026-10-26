#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_NPZ = REPO_ROOT / "additional_experiments" / "data" / "theta_ablation_traces.npz"
DEFAULT_CONTROL_CSV = REPO_ROOT / "additional_experiments" / "data" / "control_checks.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "additional_experiments" / "theta-persistence-mechanism"

INPUT_NPZ = DEFAULT_INPUT_NPZ
CONTROL_CSV = DEFAULT_CONTROL_CSV
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PLOTS_DIR = OUTPUT_DIR / "plots"

# Parameters
TAU_LOW = 0.70
TAU_HIGH = 0.90
ETA_MAX = 0.10
EPISODES = 90
CHANGE_EPISODE = 30

PHASES = {
    "Phase I": range(0, 30),
    "Phase II": range(30, 60),
    "Phase III": range(60, 90),
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(INPUT_NPZ)
    seeds = sorted(list(set([int(k.split("_")[1]) for k in data.keys() if k.startswith("seed_")])))
    gt_A = data["ground_truth_A"]
    gt_B = data["ground_truth_B"]

    # Collect per-seed series
    consistency_all = []
    eta_all = []
    obs_change_all = []
    theta_step_all = []
    dist_B_all = []
    B_progress_all = []

    per_seed_rows = []

    for s in seeds:
        c = data[f"seed_{s}_persistent_dynamic_consistency"]
        eta = data[f"seed_{s}_persistent_dynamic_eta"]
        tb = data[f"seed_{s}_persistent_dynamic_theta_before"]
        ta = data[f"seed_{s}_persistent_dynamic_theta_after"]
        C = data[f"seed_{s}_C"]

        # Observation change: ||C_t - C_{t-1}||_F for t >= 1
        obs_change = np.zeros(EPISODES, dtype=np.float64)
        for t in range(1, EPISODES):
            obs_change[t] = np.linalg.norm(C[t] - C[t - 1])

        # Theta step: ||Theta_after[t] - Theta_before[t]||_F
        theta_step = np.zeros(EPISODES, dtype=np.float64)
        for t in range(EPISODES):
            theta_step[t] = np.linalg.norm(ta[t] - tb[t])

        # Distance to B: ||Theta_before[t] - GroundTruth_B||_F
        dist_B = np.zeros(EPISODES, dtype=np.float64)
        for t in range(EPISODES):
            dist_B[t] = np.linalg.norm(tb[t] - gt_B)

        # B progress: distance_B_{t-1} - distance_B_t for t >= 1
        B_progress = np.zeros(EPISODES, dtype=np.float64)
        for t in range(1, EPISODES):
            B_progress[t] = dist_B[t - 1] - dist_B[t]

        consistency_all.append(c)
        eta_all.append(eta)
        obs_change_all.append(obs_change)
        theta_step_all.append(theta_step)
        dist_B_all.append(dist_B)
        B_progress_all.append(B_progress)

        # Lags for this seed across full trajectory (t=1..89)
        lags = {}
        for k in [0, 1, 2, 3]:
            eta_slice = eta[1 : EPISODES - k]
            prog_slice = B_progress[1 + k : EPISODES]
            corr = float(np.corrcoef(eta_slice, prog_slice)[0, 1])
            lags[k] = corr

        row = {
            "seed": s,
            "consistency_phase_I": float(np.mean(c[PHASES["Phase I"]])),
            "consistency_phase_II": float(np.mean(c[PHASES["Phase II"]])),
            "consistency_phase_III": float(np.mean(c[PHASES["Phase III"]])),
            "consistency_ep_29": float(c[29]),
            "consistency_ep_30": float(c[30]),
            "consistency_ep_31": float(c[31]),
            "eta_phase_I": float(np.mean(eta[PHASES["Phase I"]])),
            "eta_phase_II": float(np.mean(eta[PHASES["Phase II"]])),
            "eta_phase_III": float(np.mean(eta[PHASES["Phase III"]])),
            "eta_ep_29": float(eta[29]),
            "eta_ep_30": float(eta[30]),
            "eta_ep_31": float(eta[31]),
            "observation_change_phase_I": float(np.mean(obs_change[1:30])),
            "observation_change_phase_II": float(np.mean(obs_change[PHASES["Phase II"]])),
            "observation_change_ep_30": float(obs_change[30]),
            "theta_step_phase_I": float(np.mean(theta_step[PHASES["Phase I"]])),
            "theta_step_phase_II": float(np.mean(theta_step[PHASES["Phase II"]])),
            "theta_step_phase_III": float(np.mean(theta_step[PHASES["Phase III"]])),
            "final_distance_B": float(dist_B[89]),
            "mean_B_progress_phase_II": float(np.mean(B_progress[PHASES["Phase II"]])),
            "lag0_eta_B_progress": lags[0],
            "lag1_eta_B_progress": lags[1],
            "lag2_eta_B_progress": lags[2],
            "lag3_eta_B_progress": lags[3],
        }
        per_seed_rows.append(row)

    df_per_seed = pd.DataFrame(per_seed_rows)
    df_per_seed.to_csv(OUTPUT_DIR / "per_seed_mechanism.csv", index=False)

    consistency_all = np.array(consistency_all)
    eta_all = np.array(eta_all)
    obs_change_all = np.array(obs_change_all)
    theta_step_all = np.array(theta_step_all)
    dist_B_all = np.array(dist_B_all)
    B_progress_all = np.array(B_progress_all)

    # Compute summary stats
    def calc_stats(arr, window):
        sub = arr[:, window]
        return {
            "mean": float(np.mean(sub)),
            "median": float(np.median(sub)),
            "std": float(np.std(sub)),
        }

    summary_rows = []
    quantities = [
        ("consistency", consistency_all),
        ("eta", eta_all),
        ("observation_change", obs_change_all),
        ("theta_step", theta_step_all),
        ("distance_B", dist_B_all),
        ("B_progress", B_progress_all),
    ]

    for name, arr in quantities:
        p1 = calc_stats(arr, range(0, 30))
        cw = calc_stats(arr, range(27, 36))
        p2 = calc_stats(arr, range(30, 60))
        p3 = calc_stats(arr, range(60, 90))

        summary_rows.append({
            "quantity": name,
            "phase_I_mean": p1["mean"], "phase_I_median": p1["median"], "phase_I_std": p1["std"],
            "change_window_mean": cw["mean"], "change_window_median": cw["median"], "change_window_std": cw["std"],
            "phase_II_mean": p2["mean"], "phase_II_median": p2["median"], "phase_II_std": p2["std"],
            "phase_III_mean": p3["mean"], "phase_III_median": p3["median"], "phase_III_std": p3["std"],
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "mechanism_summary.csv", index=False)

    # Save detailed window table around ep 27-35
    window_data = []
    for ep in range(27, 36):
        window_data.append({
            "episode": ep,
            "consistency_mean": float(np.mean(consistency_all[:, ep])),
            "consistency_median": float(np.median(consistency_all[:, ep])),
            "consistency_std": float(np.std(consistency_all[:, ep])),
            "eta_mean": float(np.mean(eta_all[:, ep])),
            "eta_median": float(np.median(eta_all[:, ep])),
            "eta_std": float(np.std(eta_all[:, ep])),
            "obs_change_mean": float(np.mean(obs_change_all[:, ep])),
            "obs_change_std": float(np.std(obs_change_all[:, ep])),
            "theta_step_mean": float(np.mean(theta_step_all[:, ep])),
            "theta_step_std": float(np.std(theta_step_all[:, ep])),
            "dist_B_mean": float(np.mean(dist_B_all[:, ep])),
            "dist_B_median": float(np.median(dist_B_all[:, ep])),
            "dist_B_std": float(np.std(dist_B_all[:, ep])),
            "B_progress_mean": float(np.mean(B_progress_all[:, ep])),
            "B_progress_median": float(np.median(B_progress_all[:, ep])),
            "B_progress_std": float(np.std(B_progress_all[:, ep])),
        })
    df_window = pd.DataFrame(window_data)
    df_window.to_csv(OUTPUT_DIR / "episodes_27_35_details.csv", index=False)

    # Generate Plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Observation change
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    mean_obs = np.mean(obs_change_all, axis=0)
    std_obs = np.std(obs_change_all, axis=0)
    episodes = np.arange(EPISODES)
    ax.plot(episodes[1:], mean_obs[1:], color='#1f77b4', lw=2, label=r'||C_t - C_{t-1}||_F (Mean)')
    ax.fill_between(episodes[1:], mean_obs[1:] - std_obs[1:], mean_obs[1:] + std_obs[1:], color='#1f77b4', alpha=0.2)
    ax.axvline(30, color='red', linestyle='--', linewidth=1.5, label='Structural Switch (A -> B)')
    ax.set_title('Observation Change Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode t')
    ax.set_ylabel(r'||C_t - C_{t-1}||_F')
    ax.set_xlim(0, 89)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot1_observation_change.png")
    plt.close()

    # Plot 2: Consistency
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    mean_c = np.mean(consistency_all, axis=0)
    std_c = np.std(consistency_all, axis=0)
    ax.plot(episodes, mean_c, color='#2ca02c', lw=2, label=r'Consistency c_t (Mean)')
    ax.fill_between(episodes, mean_c - std_c, mean_c + std_c, color='#2ca02c', alpha=0.2, label=r'+/- 1 Std Dev')
    ax.axvline(30, color='red', linestyle='--', linewidth=1.5, label='Structural Switch (A -> B)')
    ax.axhline(TAU_LOW, color='black', linestyle=':', linewidth=1.2, label=r'tau_low = 0.70')
    ax.axhline(TAU_HIGH, color='black', linestyle='-.', linewidth=1.2, label=r'tau_high = 0.90')
    ax.set_title('Running Co-occurrence Consistency', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode t')
    ax.set_ylabel('Consistency')
    ax.set_xlim(0, 89)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot2_consistency.png")
    plt.close()

    # Plot 3: Adaptation activity (eta)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    mean_eta = np.mean(eta_all, axis=0)
    std_eta = np.std(eta_all, axis=0)
    ax.plot(episodes, mean_eta, color='#ff7f0e', lw=2, label=r'Adaptation Rate eta_t (Mean)')
    ax.fill_between(episodes, mean_eta - std_eta, mean_eta + std_eta, color='#ff7f0e', alpha=0.2, label=r'+/- 1 Std Dev')
    ax.axvline(30, color='red', linestyle='--', linewidth=1.5, label='Structural Switch (A -> B)')
    ax.set_title('Adaptation Activity (Effective Eta)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode t')
    ax.set_ylabel(r'Effective eta_t')
    ax.set_xlim(0, 89)
    ax.set_ylim(-0.005, 0.105)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot3_adaptation_activity.png")
    plt.close()

    # Plot 4: Theta response (Distance to B & Theta step)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)

    mean_dist_B = np.mean(dist_B_all, axis=0)
    std_dist_B = np.std(dist_B_all, axis=0)
    ax1.plot(episodes, mean_dist_B, color='#d62728', lw=2, label=r'||Theta_before[t] - GT_B||_F (Mean)')
    ax1.fill_between(episodes, mean_dist_B - std_dist_B, mean_dist_B + std_dist_B, color='#d62728', alpha=0.2)
    ax1.axvline(30, color='red', linestyle='--', linewidth=1.5, label='Structural Switch (A -> B)')
    ax1.set_ylabel('Distance to GT B')
    ax1.set_title('Theta Convergence and Step Magnitude', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')

    mean_step = np.mean(theta_step_all, axis=0)
    std_step = np.std(theta_step_all, axis=0)
    ax2.plot(episodes, mean_step, color='#9467bd', lw=2, label=r'||Theta_after[t] - Theta_before[t]||_F (Mean)')
    ax2.fill_between(episodes, mean_step - std_step, mean_step + std_step, color='#9467bd', alpha=0.2)
    ax2.axvline(30, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Episode t')
    ax2.set_ylabel('Theta Step Size')
    ax2.set_xlim(0, 89)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot4_theta_response.png")
    plt.close()

    # Supplementary Scatter plot of Consistency vs Eta with theoretical gate overlay
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cons_flat = consistency_all.flatten()
    eta_flat = eta_all.flatten()
    ax.scatter(cons_flat, eta_flat, color='#1f77b4', alpha=0.3, edgecolors='none', label='Observed (N=1800)')
    
    # Gate curve
    c_gate = np.linspace(0.0, 1.0, 500)
    eta_gate = ETA_MAX * np.clip((c_gate - TAU_LOW) / (TAU_HIGH - TAU_LOW), 0.0, 1.0)
    ax.plot(c_gate, eta_gate, color='red', linestyle='--', linewidth=2, label='Theoretical Gate')
    
    ax.set_title('Consistency vs. Eta (Gate Verification)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Consistency c_t')
    ax.set_ylabel(r'Effective eta_t')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_consistency_vs_eta_gate.png")
    plt.close()

    # Generate Manifest JSON
    manifest = {
        "experiment": "theta-persistence-mechanism-20260917",
        "input_source": str(INPUT_NPZ),
        "control_checks_source": str(CONTROL_CSV),
        "num_seeds": len(seeds),
        "episodes": EPISODES,
        "change_episode": CHANGE_EPISODE,
        "files_generated": [
            "report.md",
            "mechanism_summary.csv",
            "per_seed_mechanism.csv",
            "episodes_27_35_details.csv",
            "manifest.json",
            "plots/plot1_observation_change.png",
            "plots/plot2_consistency.png",
            "plots/plot3_adaptation_activity.png",
            "plots/plot4_theta_response.png",
            "plots/plot_consistency_vs_eta_gate.png",
        ]
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate report.md
    metric_labels = {
        "consistency": r"Consistency ($c_t$)",
        "eta": r"Adaptation rate ($\eta_t$)",
        "observation_change": r"Observation change ($||C_t - C_{t-1}||_F$)",
        "theta_step": r"Theta step ($||\Delta \Theta_t||_F$)",
        "distance_B": r"Distance to GT B ($||\Theta_t - \Theta^*_B||_F$)",
        "B_progress": r"Progress toward B ($\Delta d_{B, t}$)",
    }

    report_lines = [
        "# Persistent Dynamic Theta Adaptation Mechanism",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "| :--- | ---: |",
        f"| Seeds | {len(seeds)} |",
        f"| Episodes | {EPISODES} |",
        f"| Change episode | {CHANGE_EPISODE} |",
        f"| Tau low | {TAU_LOW:.2f} |",
        f"| Tau high | {TAU_HIGH:.2f} |",
        f"| Eta max | {ETA_MAX:.2f} |",
        "",
        "## Phase Summary Statistics (Mean \u00b1 Std)",
        "",
        "| Metric | Phase I (ep 0-29) | Change Window (ep 27-35) | Phase II (ep 30-59) | Phase III (ep 60-89) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]

    for _, row in df_summary.iterrows():
        label = metric_labels.get(row["quantity"], str(row["quantity"]))
        p1 = f"{row['phase_I_mean']:.4f} \u00b1 {row['phase_I_std']:.4f}"
        cw = f"{row['change_window_mean']:.4f} \u00b1 {row['change_window_std']:.4f}"
        p2 = f"{row['phase_II_mean']:.4f} \u00b1 {row['phase_II_std']:.4f}"
        p3 = f"{row['phase_III_mean']:.4f} \u00b1 {row['phase_III_std']:.4f}"
        report_lines.append(f"| {label} | {p1} | {cw} | {p2} | {p3} |")

    report_lines.extend([
        "",
        "## Transition Window (Episodes 27-35, Mean \u00b1 Std)",
        "",
        r"| Episode | Observation Change ($||C_t - C_{t-1}||_F$) | Consistency ($c_t$) | Adaptation Rate ($\eta_t$) | Theta Step ($||\Delta \Theta_t||_F$) | Distance to GT B ($||\Theta_t - \Theta^*_B||_F$) | Progress toward B ($\Delta d_{B, t}$) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for _, row in df_window.iterrows():
        ep = int(row["episode"])
        obs = f"{row['obs_change_mean']:.4f} \u00b1 {row['obs_change_std']:.4f}"
        c = f"{row['consistency_mean']:.4f} \u00b1 {row['consistency_std']:.4f}"
        eta = f"{row['eta_mean']:.4f} \u00b1 {row['eta_std']:.4f}"
        step = f"{row['theta_step_mean']:.4f} \u00b1 {row['theta_step_std']:.4f}"
        dist = f"{row['dist_B_mean']:.4f} \u00b1 {row['dist_B_std']:.4f}"
        prog = f"{row['B_progress_mean']:.4f} \u00b1 {row['B_progress_std']:.4f}"
        report_lines.append(f"| {ep} | {obs} | {c} | {eta} | {step} | {dist} | {prog} |")

    report_lines.extend([
        "",
        r"## Lag Correlations ($\eta_t$ vs $\Delta d_{B, t+k}$)",
        "",
        "| Lag | Mean | Std | Median | Min | Max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for k in [0, 1, 2, 3]:
        col = f"lag{k}_eta_B_progress"
        vals = df_per_seed[col].values
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        med_val = float(np.median(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        report_lines.append(f"| {k} | {mean_val:.4f} | {std_val:.4f} | {med_val:.4f} | {min_val:.4f} | {max_val:.4f} |")

    report_content = "\n".join(report_lines) + "\n"

    with open(OUTPUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("Analysis script completed successfully.")


if __name__ == "__main__":
    main()
