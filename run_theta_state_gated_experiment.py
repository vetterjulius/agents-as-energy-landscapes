#!/usr/bin/env python3
"""
Proof-of-Mechanism Test for State-Gated Persistent Dynamic Theta Adaptation.

Isolated experiment script: does NOT modify production code, runner.py, adaptation.py,
or existing benchmark files.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Constants & Configurations
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(r"D:\Julius Vetter\Code\energy-landscape\.storage\analysis\theta-state-gated-mechanism-20260917")
PLOTS_DIR = OUTPUT_DIR / "plots"

M = 6
N = 3
EPISODES = 90
CHANGE_EPISODE = 30
SEEDS = list(range(42, 62)) # 20 seeds
GAMMA = 0.20

VARIANTS = [
    "Static",
    "Dynamic",
    "Persistent Dynamic",
    "State-Gated Persistent Dynamic",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def offdiag(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(result, 0.0)
    return result

def normalize_theta(theta: np.ndarray) -> np.ndarray:
    return offdiag((theta + theta.T) / 2.0)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else 0.0

def canonical_assignment(structure: str) -> np.ndarray:
    X = np.zeros((N, M), dtype=np.float64)
    groups = ((0, 1), (2, 3), (4, 5)) if structure == "A" else ((0, 2), (1, 3), (4, 5))
    for agent, tasks in enumerate(groups):
        for task in tasks:
            X[agent, task] = 1.0
    return X

def observed_cooccurrence(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    co = X.T @ X
    total = float(co.sum())
    if total < epsilon:
        return np.zeros_like(co, dtype=np.float64)
    normalized = co / (total + epsilon)
    return offdiag((normalized + normalized.T) / 2.0)

def get_ground_truths() -> tuple[np.ndarray, np.ndarray]:
    gt_A = observed_cooccurrence(canonical_assignment("A"))
    gt_B = observed_cooccurrence(canonical_assignment("B"))
    return gt_A, gt_B

def generate_trajectory(seed: int, scenario: str, noise_prob: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate assignments Xs, co-occurrences Cs, and ground truth trajectory GTs.
    Scenarios:
    - 'A->A': Stationary A throughout
    - 'A->B': Change from A to B at episode 30
    - 'B->A': Change from B to A at episode 30
    """
    rng = np.random.default_rng(seed)
    gt_A, gt_B = get_ground_truths()
    
    Xs = []
    Cs = []
    GTs = []
    
    for ep in range(EPISODES):
        if scenario == "A->A":
            curr_gt = gt_A
            base_struct = "A"
        elif scenario == "A->B":
            curr_gt = gt_A if ep < CHANGE_EPISODE else gt_B
            base_struct = "A" if ep < CHANGE_EPISODE else "B"
        elif scenario == "B->A":
            curr_gt = gt_B if ep < CHANGE_EPISODE else gt_A
            base_struct = "B" if ep < CHANGE_EPISODE else "A"
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
        X = canonical_assignment(base_struct)
        if rng.random() < noise_prob:
            task = int(rng.integers(0, M))
            source = int(np.argmax(X[:, task]))
            target = int(rng.choice([a for a in range(N) if a != source]))
            X[source, task] = 0.0
            X[target, task] = 1.0
            
        C = observed_cooccurrence(X)
        Xs.append(X)
        Cs.append(C)
        GTs.append(curr_gt)
        
    return np.asarray(Xs), np.asarray(Cs), np.asarray(GTs)

# -----------------------------------------------------------------------------
# Simulation of a single run
# -----------------------------------------------------------------------------
def run_simulation(
    variant: str,
    Cs: np.ndarray,
    GTs: np.ndarray,
    theta_init: np.ndarray,
) -> dict:
    theta = theta_init.copy()
    cbar_prev = None
    C_prev = None
    
    state = "STABLE" # initial state for State-Gated
    suspect_evidence = 0
    converged_counter = 0
    
    theta_befores = []
    theta_afters = []
    etas = []
    consistencies = []
    c_changes = []
    cbar_theta_dists = []
    states = []
    transitions = []
    
    for t in range(EPISODES):
        C = Cs[t]
        theta_before = theta.copy()
        
        if t == 0:
            consistency = 1.0
            c_change = 0.0
            cbar = C.copy()
        else:
            consistency = cosine_similarity(C, cbar_prev)
            c_change = float(np.linalg.norm(C - C_prev))
            cbar = (1.0 - GAMMA) * cbar_prev + GAMMA * C
            
        cbar_theta_dist = float(np.linalg.norm(cbar - theta_before))
        
        # Determine state & eta for variant
        if variant == "Static":
            curr_state = "STABLE"
            eta = 0.0
        elif variant == "Dynamic":
            curr_state = "ADAPTING"
            eta = 0.10
        elif variant == "Persistent Dynamic":
            alpha = float(np.clip((consistency - 0.70) / (0.90 - 0.70), 0.0, 1.0))
            eta = 0.10 * alpha
            if eta > 0.05:
                curr_state = "ADAPTING"
            elif eta > 0.005:
                curr_state = "CONVERGED"
            else:
                curr_state = "STABLE"
        elif variant == "State-Gated Persistent Dynamic":
            prev_state = state
            is_mismatch = (consistency < 0.70) or (c_change >= 0.15)
            
            if state == "STABLE":
                if is_mismatch:
                    suspect_evidence += 1
                    if suspect_evidence >= 2:
                        state = "SUSPECTED_CHANGE"
                        suspect_evidence = 0
                else:
                    suspect_evidence = 0
                    
            elif state == "SUSPECTED_CHANGE":
                if consistency >= 0.75:
                    if cbar_theta_dist < 0.08:
                        state = "CONVERGED"
                        converged_counter = 0
                    else:
                        state = "ADAPTING"
                        converged_counter = 0
                    suspect_evidence = 0
                elif consistency >= 0.70:
                    state = "ADAPTING"
                    converged_counter = 0
                    suspect_evidence = 0
                    
            elif state == "ADAPTING":
                if consistency < 0.65:
                    suspect_evidence += 1
                    if suspect_evidence >= 2:
                        state = "SUSPECTED_CHANGE"
                        suspect_evidence = 0
                        converged_counter = 0
                else:
                    suspect_evidence = 0
                    if cbar_theta_dist < 0.08:
                        converged_counter += 1
                        if converged_counter >= 2: # patience of 2
                            state = "CONVERGED"
                    else:
                        converged_counter = 0
                        
            elif state == "CONVERGED":
                if consistency < 0.65 or cbar_theta_dist > 0.10:
                    suspect_evidence += 1
                    if suspect_evidence >= 2:
                        state = "SUSPECTED_CHANGE"
                        suspect_evidence = 0
                        converged_counter = 0
                else:
                    suspect_evidence = 0
                    
            if state != prev_state:
                transitions.append({
                    "episode": t,
                    "from_state": prev_state,
                    "to_state": state,
                    "consistency": consistency,
                    "c_change": c_change,
                    "cbar_theta_dist": cbar_theta_dist,
                })
                
            curr_state = state
            
            if curr_state == "STABLE":
                eta = 0.0
            elif curr_state == "SUSPECTED_CHANGE":
                eta = 0.0
            elif curr_state == "ADAPTING":
                eta = 0.10
            elif curr_state == "CONVERGED":
                eta = 0.0
            else:
                eta = 0.0
        else:
            raise ValueError(f"Unknown variant {variant}")
            
        # Update Theta
        theta_next = normalize_theta((1.0 - eta) * theta_before + eta * C)
        theta = theta_next
        
        theta_befores.append(theta_before)
        theta_afters.append(theta_next)
        etas.append(eta)
        consistencies.append(consistency)
        c_changes.append(c_change)
        cbar_theta_dists.append(cbar_theta_dist)
        states.append(curr_state)
        
        cbar_prev = cbar.copy()
        C_prev = C.copy()
        
    return {
        "theta_before": np.asarray(theta_befores),
        "theta_after": np.asarray(theta_afters),
        "eta": np.asarray(etas),
        "consistency": np.asarray(consistencies),
        "c_change": np.asarray(c_changes),
        "cbar_theta_dist": np.asarray(cbar_theta_dists),
        "state": np.asarray(states),
        "transitions": transitions,
    }

# -----------------------------------------------------------------------------
# Main Execution & Evaluation
# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    gt_A, gt_B = get_ground_truths()
    
    scenarios = ["A->A", "A->B", "B->A"]
    noise_levels = [0.00, 0.20, 0.40]
    
    all_diagnostics = []
    all_transitions = []
    all_per_seed = []
    
    print("Running proof-of-mechanism simulations...")
    
    for sc in scenarios:
        for noise in noise_levels:
            for seed in SEEDS:
                Xs, Cs, GTs = generate_trajectory(seed, sc, noise)
                theta_init = gt_A if sc in ("A->A", "A->B") else gt_B
                
                for variant in VARIANTS:
                    res = run_simulation(variant, Cs, GTs, theta_init)
                    
                    tb = res["theta_before"]
                    ta = res["theta_after"]
                    eta = res["eta"]
                    cons = res["consistency"]
                    cc = res["c_change"]
                    ctd = res["cbar_theta_dist"]
                    st = res["state"]
                    
                    # Record episode diagnostics
                    for ep in range(EPISODES):
                        theta_step = float(np.linalg.norm(ta[ep] - tb[ep]))
                        dist_gt = float(np.linalg.norm(tb[ep] - GTs[ep]))
                        sim_gt = cosine_similarity(tb[ep], GTs[ep])
                        
                        all_diagnostics.append({
                            "scenario": sc,
                            "noise_prob": noise,
                            "seed": seed,
                            "episode": ep,
                            "variant": variant,
                            "state": st[ep],
                            "consistency": cons[ep],
                            "C_change": cc[ep],
                            "Cbar_theta_distance": ctd[ep],
                            "theta_step": theta_step,
                            "eta": eta[ep],
                            "distance_to_GT": dist_gt,
                            "similarity_to_GT": sim_gt,
                        })
                        
                    # Record transitions
                    for tr in res["transitions"]:
                        all_transitions.append({
                            "scenario": sc,
                            "noise_prob": noise,
                            "seed": seed,
                            "variant": variant,
                            "episode": tr["episode"],
                            "from_state": tr["from_state"],
                            "to_state": tr["to_state"],
                            "consistency": tr["consistency"],
                            "c_change": tr["c_change"],
                            "cbar_theta_dist": tr["cbar_theta_dist"],
                        })
                        
                    # Per-seed aggregate metrics
                    drifts = [float(np.linalg.norm(tb[ep] - theta_init)) for ep in range(30)]
                    phase_1_mean_drift = float(np.mean(drifts))
                    phase_1_max_drift = float(np.max(drifts))
                    phase_1_final_drift = float(drifts[29])
                    
                    time_in_adapting_p1 = float(np.mean(st[:30] == "ADAPTING"))
                    time_in_suspected_p1 = float(np.mean(st[:30] == "SUSPECTED_CHANGE"))
                    
                    # Phase II/III metrics for change scenarios
                    if sc in ("A->B", "B->A"):
                        target_gt = gt_B if sc == "A->B" else gt_A
                        dists_target = [float(np.linalg.norm(tb[ep] - target_gt)) for ep in range(EPISODES)]
                        sims_target = [cosine_similarity(tb[ep], target_gt) for ep in range(EPISODES)]
                        
                        # Delays after ep 30
                        adapting_eps = [ep for ep in range(30, EPISODES) if st[ep] == "ADAPTING"]
                        suspected_eps = [ep for ep in range(30, EPISODES) if st[ep] == "SUSPECTED_CHANGE"]
                        converged_eps = [ep for ep in range(30, EPISODES) if st[ep] == "CONVERGED"]
                        
                        time_to_suspected = (suspected_eps[0] - 30) if suspected_eps else None
                        time_to_adapting = (adapting_eps[0] - 30) if adapting_eps else None
                        time_to_converged = (converged_eps[0] - 30) if converged_eps else None
                        duration_adapting = len(adapting_eps)
                        
                        final_dist = dists_target[89]
                        mean_dist_p2 = float(np.mean(dists_target[30:60]))
                        mean_dist_p3 = float(np.mean(dists_target[60:90]))
                    else:
                        time_to_suspected = None
                        time_to_adapting = None
                        time_to_converged = None
                        duration_adapting = 0
                        final_dist = float(np.linalg.norm(tb[89] - theta_init))
                        mean_dist_p2 = float(np.mean([np.linalg.norm(tb[ep] - theta_init) for ep in range(30, 60)]))
                        mean_dist_p3 = float(np.mean([np.linalg.norm(tb[ep] - theta_init) for ep in range(60, 90)]))
                        
                    all_per_seed.append({
                        "scenario": sc,
                        "noise_prob": noise,
                        "seed": seed,
                        "variant": variant,
                        "phase_1_mean_drift": phase_1_mean_drift,
                        "phase_1_max_drift": phase_1_max_drift,
                        "phase_1_final_drift": phase_1_final_drift,
                        "time_in_adapting_p1": time_in_adapting_p1,
                        "time_in_suspected_p1": time_in_suspected_p1,
                        "time_to_suspected": time_to_suspected,
                        "time_to_adapting": time_to_adapting,
                        "time_to_converged": time_to_converged,
                        "duration_adapting": duration_adapting,
                        "final_dist_target": final_dist,
                        "mean_dist_p2": mean_dist_p2,
                        "mean_dist_p3": mean_dist_p3,
                        "false_adapting_p1_eps": int(np.sum(st[:30] == "ADAPTING")),
                        "false_suspected_p1_eps": int(np.sum(st[:30] == "SUSPECTED_CHANGE")),
                    })

    df_diag = pd.DataFrame(all_diagnostics)
    df_trans = pd.DataFrame(all_transitions)
    df_per_seed = pd.DataFrame(all_per_seed)
    
    df_diag.to_csv(OUTPUT_DIR / "episode_diagnostics.csv", index=False)
    df_trans.to_csv(OUTPUT_DIR / "state_transitions.csv", index=False)
    df_per_seed.to_csv(OUTPUT_DIR / "per_seed_metrics.csv", index=False)
    
    # -------------------------------------------------------------------------
    # Summary Metrics
    # -------------------------------------------------------------------------
    summary_rows = []
    for (sc, noise, variant), group in df_per_seed.groupby(["scenario", "noise_prob", "variant"]):
        summary_rows.append({
            "scenario": sc,
            "noise_prob": noise,
            "variant": variant,
            "mean_p1_drift": group["phase_1_mean_drift"].mean(),
            "max_p1_drift": group["phase_1_max_drift"].max(),
            "final_p1_drift": group["phase_1_final_drift"].mean(),
            "mean_time_in_adapting_p1": group["time_in_adapting_p1"].mean(),
            "mean_time_in_suspected_p1": group["time_in_suspected_p1"].mean(),
            "mean_time_to_adapting": group["time_to_adapting"].dropna().mean(),
            "mean_time_to_converged": group["time_to_converged"].dropna().mean(),
            "mean_duration_adapting": group["duration_adapting"].mean(),
            "mean_final_dist": group["final_dist_target"].mean(),
            "mean_dist_p2": group["mean_dist_p2"].mean(),
            "mean_dist_p3": group["mean_dist_p3"].mean(),
            "false_adapting_p1_total_eps": group["false_adapting_p1_eps"].sum(),
            "false_suspected_p1_total_eps": group["false_suspected_p1_eps"].sum(),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    
    # Noise metrics breakdown
    df_noise = df_summary[df_summary["scenario"] == "A->B"].copy()
    df_noise.to_csv(OUTPUT_DIR / "noise_metrics.csv", index=False)
    
    # Manifest JSON
    manifest = {
        "timestamp": "2026-09-17T18:10:00Z",
        "experiment": "State-Gated Persistent Dynamic Proof of Mechanism",
        "M": M,
        "N": N,
        "episodes": EPISODES,
        "change_episode": CHANGE_EPISODE,
        "seeds": SEEDS,
        "scenarios": scenarios,
        "noise_levels": noise_levels,
        "variants": VARIANTS,
        "hyperparameters": {
            "gamma": GAMMA,
            "suspect_consistency_thresh": 0.70,
            "suspect_c_change_thresh": 0.15,
            "suspect_patience": 2,
            "adapting_consistency_thresh": 0.70,
            "adapting_hysteresis_thresh": 0.75,
            "converged_dist_thresh": 0.08,
            "converged_leave_dist_thresh": 0.10,
            "converged_leave_consistency_thresh": 0.65,
            "converged_patience": 2,
            "eta_adapting": 0.10,
            "eta_converged": 0.0,
            "eta_stable": 0.0,
            "eta_suspected": 0.0,
        },
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Metrics CSVs and Manifest written successfully.")
    
    # -------------------------------------------------------------------------
    # Visualizations (6 Plots Max)
    # -------------------------------------------------------------------------
    print("Generating 6 diagnostic plots...")
    
    # Color palette
    colors = {
        "Static": "#7f7f7f",
        "Dynamic": "#d62728",
        "Persistent Dynamic": "#ff7f0e",
        "State-Gated Persistent Dynamic": "#1f77b4",
    }
    
    # Plot 1: Distance to Ground Truth over time (A->B, noise=0.20)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_sub1 = df_diag[(df_diag["scenario"] == "A->B") & (df_diag["noise_prob"] == 0.20)]
    for v in VARIANTS:
        v_data = df_sub1[df_sub1["variant"] == v]
        means = v_data.groupby("episode")["distance_to_GT"].mean()
        stds = v_data.groupby("episode")["distance_to_GT"].std()
        ax.plot(means.index, means.values, label=v, color=colors[v], linewidth=2)
        ax.fill_between(means.index, means.values - stds.values, means.values + stds.values, color=colors[v], alpha=0.15)
    ax.axvline(30, color="black", linestyle="--", alpha=0.7, label="Structural Change (ep 30)")
    ax.set_title("Plot 1: Distance to Active Ground Truth over Time (A->B, noise=0.20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("||Theta_t - GT_t||_F")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot1_distance_to_gt.png")
    plt.close(fig)
    
    # Plot 2: State trajectory over time for State-Gated Persistent Dynamic (A->B, noise=0.20)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    sg_data = df_sub1[df_sub1["variant"] == "State-Gated Persistent Dynamic"]
    state_map = {"STABLE": 0, "SUSPECTED_CHANGE": 1, "ADAPTING": 2, "CONVERGED": 3}
    for s in SEEDS:
        s_data = sg_data[sg_data["seed"] == s]
        y_vals = [state_map[st] for st in s_data["state"]]
        ax.plot(s_data["episode"], y_vals, color="#1f77b4", alpha=0.25, linewidth=1)
    # Mode state across seeds
    mode_states = sg_data.groupby("episode")["state"].agg(lambda x: x.mode()[0])
    y_mode = [state_map[st] for st in mode_states]
    ax.plot(range(EPISODES), y_mode, color="black", linewidth=2.5, label="Mode State (Majority)")
    ax.axvline(30, color="red", linestyle="--", alpha=0.8, label="Structural Change (ep 30)")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["STABLE", "SUSPECTED_CHANGE", "ADAPTING", "CONVERGED"])
    ax.set_title("Plot 2: State Trajectory over Episode (State-Gated, A->B, noise=0.20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("State")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot2_state_trajectory.png")
    plt.close(fig)
    
    # Plot 3: Consistency and Cbar-Theta Distance together (Central Mechanism Plot)
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax2 = ax1.twinx()
    
    cons_means = sg_data.groupby("episode")["consistency"].mean()
    dist_means = sg_data.groupby("episode")["Cbar_theta_distance"].mean()
    
    line1 = ax1.plot(cons_means.index, cons_means.values, color="green", linewidth=2, label="Consistency (C_t vs Cbar_{t-1})")
    line2 = ax2.plot(dist_means.index, dist_means.values, color="purple", linewidth=2, label="||Cbar_t - Theta_t||_F")
    
    ax1.axhline(0.70, color="green", linestyle=":", alpha=0.7, label="Consistency Threshold (0.70)")
    ax2.axhline(0.08, color="purple", linestyle=":", alpha=0.7, label="Convergence Dist Threshold (0.08)")
    ax1.axvline(30, color="black", linestyle="--", alpha=0.7, label="Change at ep 30")
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Consistency", color="green")
    ax2.set_ylabel("||Cbar_t - Theta_t||_F", color="purple")
    ax1.set_title("Plot 3: Mechanism Signals — Consistency & Cbar-Theta Distance (State-Gated)")
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot3_consistency_and_cbar_theta_distance.png")
    plt.close(fig)
    
    # Plot 4: Theta Step and Eta over time (State-Gated vs Persistent Dynamic, A->B, noise=0.20)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)
    
    for v in ["Persistent Dynamic", "State-Gated Persistent Dynamic"]:
        v_data = df_sub1[df_sub1["variant"] == v]
        eta_means = v_data.groupby("episode")["eta"].mean()
        step_means = v_data.groupby("episode")["theta_step"].mean()
        
        ax1.plot(eta_means.index, eta_means.values, label=v, color=colors[v], linewidth=2)
        ax2.plot(step_means.index, step_means.values, label=v, color=colors[v], linewidth=2)
        
    ax1.axvline(30, color="black", linestyle="--", alpha=0.7)
    ax2.axvline(30, color="black", linestyle="--", alpha=0.7)
    
    ax1.set_ylabel("Learning Rate (eta)")
    ax2.set_ylabel("Theta Step ||Delta Theta||_F")
    ax2.set_xlabel("Episode")
    ax1.set_title("Plot 4: Eta and Theta Step Profiles over Time (A->B, noise=0.20)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot4_eta_and_theta_step.png")
    plt.close(fig)
    
    # Plot 5: Stationary Noise Sensitivity (Phase 1 Mean Drift vs Noise Level)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_stat = df_summary[df_summary["scenario"] == "A->A"]
    for v in VARIANTS:
        v_data = df_stat[df_stat["variant"] == v].sort_values("noise_prob")
        ax.plot(v_data["noise_prob"], v_data["mean_p1_drift"], marker="o", label=v, color=colors[v], linewidth=2)
    ax.set_title("Plot 5: Stationary Noise Sensitivity — Phase I Theta Drift vs Noise Level (A->A)")
    ax.set_xlabel("Observation Noise Probability")
    ax.set_ylabel("Phase I Mean Drift ||Theta_t - Theta_0||_F")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot5_stationary_noise_sensitivity.png")
    plt.close(fig)
    
    # Plot 6: Detection Delay vs Stationary Drift (Trade-off Plot)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    # Filter for noise = 0.20
    df_ab_02 = df_summary[(df_summary["scenario"] == "A->B") & (df_summary["noise_prob"] == 0.20)]
    df_aa_02 = df_summary[(df_summary["scenario"] == "A->A") & (df_summary["noise_prob"] == 0.20)]
    
    for v in VARIANTS:
        drift_val = df_aa_02[df_aa_02["variant"] == v]["mean_p1_drift"].values[0]
        delay_val = df_ab_02[df_ab_02["variant"] == v]["mean_time_to_adapting"].values[0]
        if math.isnan(delay_val):
            delay_val = 0.0 # for Static
        ax.scatter(drift_val, delay_val, label=v, color=colors[v], s=120, zorder=5)
        ax.annotate(v, (drift_val, delay_val), textcoords="offset points", xytext=(5,5), ha='left')
        
    ax.set_title("Plot 6: Trade-off — Stationary Drift (noise=0.20) vs Time to ADAPTING (A->B)")
    ax.set_xlabel("Stationary Phase I Drift (A->A, noise=0.20)")
    ax.set_ylabel("Mean Time to ADAPTING after ep 30 (episodes)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot6_tradeoff_drift_vs_delay.png")
    plt.close(fig)
    
    print("All 6 plots generated and saved successfully.")

if __name__ == "__main__":
    main()
