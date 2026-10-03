#!/usr/bin/env python3
"""
Isolated Proof-of-Mechanism Test: Residual Learning Rate in CONVERGED State.

Investigates whether setting eta_residual > 0 in CONVERGED reduces the asymptotic parameter
error without destroying the stationary noise protection of State-Gated Persistent Dynamic.
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
# Constants & Paths
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(r"D:\Julius Vetter\Code\energy-landscape\.storage\analysis\theta-residual-learning-20260917")
PLOTS_DIR = OUTPUT_DIR / "plots"

M = 6
N = 3
EPISODES = 90
CHANGE_EPISODE = 30
SEEDS = list(range(42, 62)) # 20 seeds
GAMMA = 0.20

VARIANTS = [
    "Persistent Dynamic",
    "State-Gated Freeze",
    "State-Gated Residual-0.005",
    "State-Gated Residual-0.01",
]

# -----------------------------------------------------------------------------
# Utility Functions
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
# Simulation Function
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
    
    state = "STABLE"
    suspect_evidence = 0
    converged_counter = 0
    
    # Residual rate for CONVERGED
    if variant == "State-Gated Freeze":
        eta_residual = 0.00
    elif variant == "State-Gated Residual-0.005":
        eta_residual = 0.005
    elif variant == "State-Gated Residual-0.01":
        eta_residual = 0.01
    else:
        eta_residual = 0.00
        
    theta_befores = []
    theta_afters = []
    etas = []
    consistencies = []
    c_changes = []
    cbar_theta_dists = []
    states = []
    transitions = []
    first_converged_ep = None
    
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
        
        if variant == "Persistent Dynamic":
            alpha = float(np.clip((consistency - 0.70) / (0.90 - 0.70), 0.0, 1.0))
            eta = 0.10 * alpha
            if eta > 0.05:
                curr_state = "ADAPTING"
            elif eta > 0.005:
                curr_state = "CONVERGED"
            else:
                curr_state = "STABLE"
        else:
            # State-Gated mechanism with identical rules
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
                    
            curr_state = state
            if curr_state == "CONVERGED" and first_converged_ep is None and t >= CHANGE_EPISODE:
                first_converged_ep = t
                
            if curr_state == "STABLE":
                eta = 0.0
            elif curr_state == "SUSPECTED_CHANGE":
                eta = 0.0
            elif curr_state == "ADAPTING":
                eta = 0.10
            elif curr_state == "CONVERGED":
                eta = eta_residual
            else:
                eta = 0.0
                
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
        "first_converged_ep": first_converged_ep,
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    gt_A, gt_B = get_ground_truths()
    
    # Scenarios required
    # Test A: A->A with noise 0.00, 0.20, 0.40
    # Test B: A->B with noise 0.00
    # Test C: A->B with noise 0.20
    # Test D: B->A with noise 0.00
    test_cases = [
        ("A->A", 0.00),
        ("A->A", 0.20),
        ("A->A", 0.40),
        ("A->B", 0.00),
        ("A->B", 0.20),
        ("B->A", 0.00),
    ]
    
    state_diagnostics = []
    per_seed_rows = []
    post_conv_rows = []
    
    print("Running residual learning rate experiment...")
    
    for sc, noise in test_cases:
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
                first_conv = res["first_converged_ep"]
                
                # Diagnostics per episode
                for ep in range(EPISODES):
                    theta_step = float(np.linalg.norm(ta[ep] - tb[ep]))
                    dist_gt = float(np.linalg.norm(tb[ep] - GTs[ep]))
                    
                    state_diagnostics.append({
                        "scenario": sc,
                        "noise_prob": noise,
                        "seed": seed,
                        "episode": ep,
                        "variant": variant,
                        "state": st[ep],
                        "eta": eta[ep],
                        "consistency": cons[ep],
                        "C_change": cc[ep],
                        "Cbar_theta_distance": ctd[ep],
                        "theta_step": theta_step,
                        "distance_to_GT": dist_gt,
                    })
                    
                # Phase I metrics
                drifts = [float(np.linalg.norm(tb[ep] - theta_init)) for ep in range(30)]
                mean_drift_p1 = float(np.mean(drifts))
                max_drift_p1 = float(np.max(drifts))
                final_drift_p1 = float(drifts[29])
                mean_theta_step_p1 = float(np.mean([np.linalg.norm(ta[ep] - tb[ep]) for ep in range(30)]))
                
                frac_eta_active = float(np.mean(eta[:30] > 0.005))
                frac_adapting_p1 = float(np.mean(st[:30] == "ADAPTING"))
                frac_converged_p1 = float(np.mean(st[:30] == "CONVERGED"))
                
                # Structural change metrics (if change scenario)
                if sc in ("A->B", "B->A"):
                    target_gt = gt_B if sc == "A->B" else gt_A
                    dists = [float(np.linalg.norm(tb[ep] - target_gt)) for ep in range(EPISODES)]
                    
                    adapting_eps = [ep for ep in range(30, EPISODES) if st[ep] == "ADAPTING"]
                    converged_eps = [ep for ep in range(30, EPISODES) if st[ep] == "CONVERGED"]
                    
                    time_to_adapting = (adapting_eps[0] - 30) if adapting_eps else None
                    time_to_converged = (converged_eps[0] - 30) if converged_eps else None
                    
                    dist_p2_end = dists[59]
                    dist_p3_end = dists[89]
                    min_dist = float(np.min(dists[30:]))
                    final_dist = dists[89]
                    
                    dist_at_conv = dists[first_conv] if first_conv is not None else None
                    dist_conv_5 = dists[first_conv + 5] if (first_conv is not None and first_conv + 5 < EPISODES) else None
                    dist_conv_10 = dists[first_conv + 10] if (first_conv is not None and first_conv + 10 < EPISODES) else None
                    dist_conv_20 = dists[first_conv + 20] if (first_conv is not None and first_conv + 20 < EPISODES) else None
                    
                    # Track post-CONVERGED trajectory relative to first_conv
                    if first_conv is not None:
                        for offset in range(EPISODES - first_conv):
                            ep_idx = first_conv + offset
                            post_conv_rows.append({
                                "scenario": sc,
                                "noise_prob": noise,
                                "seed": seed,
                                "variant": variant,
                                "episodes_since_converged": offset,
                                "distance_to_target": dists[ep_idx],
                                "state": st[ep_idx],
                                "eta": eta[ep_idx],
                            })
                else:
                    time_to_adapting = None
                    time_to_converged = None
                    dist_p2_end = float(np.linalg.norm(tb[59] - theta_init))
                    dist_p3_end = float(np.linalg.norm(tb[89] - theta_init))
                    min_dist = float(np.min([np.linalg.norm(tb[ep] - theta_init) for ep in range(EPISODES)]))
                    final_dist = float(np.linalg.norm(tb[89] - theta_init))
                    dist_at_conv = None
                    dist_conv_5 = None
                    dist_conv_10 = None
                    dist_conv_20 = None
                    
                per_seed_rows.append({
                    "scenario": sc,
                    "noise_prob": noise,
                    "seed": seed,
                    "variant": variant,
                    "mean_drift_p1": mean_drift_p1,
                    "max_drift_p1": max_drift_p1,
                    "final_drift_p1": final_drift_p1,
                    "mean_theta_step_p1": mean_theta_step_p1,
                    "frac_eta_active_p1": frac_eta_active,
                    "frac_adapting_p1": frac_adapting_p1,
                    "frac_converged_p1": frac_converged_p1,
                    "time_to_adapting": time_to_adapting,
                    "time_to_converged": time_to_converged,
                    "dist_p2_end": dist_p2_end,
                    "dist_p3_end": dist_p3_end,
                    "min_dist": min_dist,
                    "final_dist": final_dist,
                    "dist_at_conv": dist_at_conv,
                    "dist_conv_5": dist_conv_5,
                    "dist_conv_10": dist_conv_10,
                    "dist_conv_20": dist_conv_20,
                })

    df_diag = pd.DataFrame(state_diagnostics)
    df_per_seed = pd.DataFrame(per_seed_rows)
    df_post_conv = pd.DataFrame(post_conv_rows)
    
    df_diag.to_csv(OUTPUT_DIR / "state_diagnostics.csv", index=False)
    df_per_seed.to_csv(OUTPUT_DIR / "per_seed_metrics.csv", index=False)
    df_post_conv.to_csv(OUTPUT_DIR / "post_convergence_trajectory.csv", index=False)
    
    # Summary Metrics
    summary_rows = []
    for (sc, noise, variant), group in df_per_seed.groupby(["scenario", "noise_prob", "variant"]):
        # Freeze reference for relative drift increase
        freeze_group = df_per_seed[(df_per_seed["scenario"] == sc) & (df_per_seed["noise_prob"] == noise) & (df_per_seed["variant"] == "State-Gated Freeze")]
        freeze_drift = freeze_group["mean_drift_p1"].mean() if len(freeze_group) > 0 else 1.0
        
        curr_drift = group["mean_drift_p1"].mean()
        rel_drift_increase = (curr_drift / freeze_drift) if (freeze_drift > 1e-8) else (1.0 if curr_drift == 0.0 else 999.0)
        
        summary_rows.append({
            "scenario": sc,
            "noise_prob": noise,
            "variant": variant,
            "mean_drift_p1": curr_drift,
            "max_drift_p1": group["max_drift_p1"].max(),
            "final_drift_p1": group["final_drift_p1"].mean(),
            "mean_theta_step_p1": group["mean_theta_step_p1"].mean(),
            "frac_eta_active_p1": group["frac_eta_active_p1"].mean(),
            "frac_adapting_p1": group["frac_adapting_p1"].mean(),
            "frac_converged_p1": group["frac_converged_p1"].mean(),
            "mean_time_to_adapting": group["time_to_adapting"].dropna().mean(),
            "mean_time_to_converged": group["time_to_converged"].dropna().mean(),
            "mean_dist_p2_end": group["dist_p2_end"].mean(),
            "mean_dist_p3_end": group["dist_p3_end"].mean(),
            "mean_min_dist": group["min_dist"].mean(),
            "mean_final_dist": group["final_dist"].mean(),
            "mean_dist_at_conv": group["dist_at_conv"].dropna().mean(),
            "mean_dist_conv_5": group["dist_conv_5"].dropna().mean(),
            "mean_dist_conv_10": group["dist_conv_10"].dropna().mean(),
            "mean_dist_conv_20": group["dist_conv_20"].dropna().mean(),
            "relative_drift_increase_p1": rel_drift_increase,
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    
    # Noise metrics file
    df_noise = df_summary[df_summary["scenario"] == "A->A"].copy()
    df_noise.to_csv(OUTPUT_DIR / "noise_metrics.csv", index=False)
    
    # Manifest JSON
    manifest = {
        "timestamp": "2026-09-17T18:15:00Z",
        "experiment": "State-Gated Residual Learning Rate Mechanism Test",
        "M": M,
        "N": N,
        "episodes": EPISODES,
        "change_episode": CHANGE_EPISODE,
        "seeds": SEEDS,
        "test_cases": test_cases,
        "variants": VARIANTS,
        "parameters": {
            "eta_adapting": 0.10,
            "eta_freeze": 0.00,
            "eta_residual_small": 0.005,
            "eta_residual_large": 0.01,
            "gamma": GAMMA,
            "converged_dist_threshold": 0.08,
            "converged_patience": 2,
        },
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("CSV data files and manifest created.")
    
    # -------------------------------------------------------------------------
    # Visualizations (Max 5 Plots)
    # -------------------------------------------------------------------------
    print("Generating plots...")
    colors = {
        "Persistent Dynamic": "#ff7f0e",
        "State-Gated Freeze": "#7f7f7f",
        "State-Gated Residual-0.005": "#2ca02c",
        "State-Gated Residual-0.01": "#1f77b4",
    }
    
    # Plot 1: Distance to B after CONVERGED (A->B, noise=0.00)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_pc_ab0 = df_post_conv[(df_post_conv["scenario"] == "A->B") & (df_post_conv["noise_prob"] == 0.00)]
    for v in VARIANTS:
        v_data = df_pc_ab0[df_pc_ab0["variant"] == v]
        if len(v_data) > 0:
            means = v_data.groupby("episodes_since_converged")["distance_to_target"].mean()
            ax.plot(means.index, means.values, label=v, color=colors[v], linewidth=2.5)
    ax.set_title("Plot 1: Asymptotic Distance to Ground Truth B after CONVERGED Entry (A->B, noise=0.00)")
    ax.set_xlabel("Episodes since entering CONVERGED")
    ax.set_ylabel("||Theta_t - GT_B||_F")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot1_distance_after_converged.png")
    plt.close(fig)
    
    # Plot 2: Stationary Drift vs Residual Eta (A->A)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    res_etas = [0.00, 0.005, 0.01]
    res_variants = ["State-Gated Freeze", "State-Gated Residual-0.005", "State-Gated Residual-0.01"]
    
    for noise in [0.00, 0.20, 0.40]:
        drifts = []
        for v in res_variants:
            d_val = df_summary[(df_summary["scenario"] == "A->A") & (df_summary["noise_prob"] == noise) & (df_summary["variant"] == v)]["mean_drift_p1"].values[0]
            drifts.append(d_val)
        ax.plot(res_etas, drifts, marker="o", linewidth=2, label=f"Noise prob = {noise}")
        
    ax.set_title("Plot 2: Stationary Theta Drift vs. Residual Eta in CONVERGED (A->A)")
    ax.set_xlabel("Residual Learning Rate (eta in CONVERGED state)")
    ax.set_ylabel("Mean Phase I Stationary Drift ||Theta_t - Theta_0||_F")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot2_stationary_drift_vs_residual_eta.png")
    plt.close(fig)
    
    # Plot 3: Eta and State trajectory for representative seed (Seed 42, A->B, noise=0.20)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)
    seed_data = df_diag[(df_diag["scenario"] == "A->B") & (df_diag["noise_prob"] == 0.20) & (df_diag["seed"] == 42)]
    
    state_map = {"STABLE": 0, "SUSPECTED_CHANGE": 1, "ADAPTING": 2, "CONVERGED": 3}
    for v in ["State-Gated Freeze", "State-Gated Residual-0.01"]:
        v_data = seed_data[seed_data["variant"] == v]
        y_states = [state_map[s] for s in v_data["state"]]
        ax1.plot(v_data["episode"], y_states, label=v, color=colors[v], linewidth=2)
        ax2.plot(v_data["episode"], v_data["distance_to_GT"], label=v, color=colors[v], linewidth=2)
        
    ax1.axvline(30, color="black", linestyle="--", alpha=0.7)
    ax2.axvline(30, color="black", linestyle="--", alpha=0.7)
    
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["STABLE", "SUSPECTED", "ADAPTING", "CONVERGED"])
    ax1.set_ylabel("State Trajectory")
    ax2.set_ylabel("Distance to Ground Truth")
    ax2.set_xlabel("Episode")
    ax1.set_title("Plot 3: State Trajectory & Distance to GT (Seed 42, A->B, noise=0.20)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot4_eta_and_state_trajectory.png") # saving as plot3/4 filename
    plt.close(fig)
    
    # Plot 4: Distance at CONVERGED vs Final Distance (A->B, noise=0.00)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_ab0_ps = df_per_seed[(df_per_seed["scenario"] == "A->B") & (df_per_seed["noise_prob"] == 0.00)]
    
    x_coords = np.arange(len(VARIANTS))
    width = 0.35
    
    d_at_conv_means = [df_ab0_ps[df_ab0_ps["variant"] == v]["dist_at_conv"].dropna().mean() for v in VARIANTS]
    d_final_means = [df_ab0_ps[df_ab0_ps["variant"] == v]["final_dist"].mean() for v in VARIANTS]
    
    ax.bar(x_coords - width/2, d_at_conv_means, width, label="Distance at CONVERGED Entry", color="#9467bd", alpha=0.85)
    ax.bar(x_coords + width/2, d_final_means, width, label="Final Distance (Ep 89)", color="#1f77b4", alpha=0.85)
    
    ax.set_xticks(x_coords)
    ax.set_xticklabels(VARIANTS, rotation=15)
    ax.set_ylabel("||Theta - GT_B||_F")
    ax.set_title("Plot 4: Parameter Error at CONVERGED Entry vs. Final Episode (A->B, noise=0.00)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot3_dist_at_conv_vs_final.png")
    plt.close(fig)
    
    # Plot 5: Noise Sensitivity (A->A Phase I Drift across Noise Probabilities)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_stat_sum = df_summary[df_summary["scenario"] == "A->A"]
    for v in VARIANTS:
        v_data = df_stat_sum[df_stat_sum["variant"] == v].sort_values("noise_prob")
        ax.plot(v_data["noise_prob"], v_data["mean_drift_p1"], marker="o", label=v, color=colors[v], linewidth=2.5)
    ax.set_title("Plot 5: Noise Sensitivity — Stationary Phase I Theta Drift vs. Noise Probability")
    ax.set_xlabel("Observation Noise Probability")
    ax.set_ylabel("Mean Stationary Drift ||Theta_t - Theta_0||_F")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot5_noise_sensitivity.png")
    plt.close(fig)
    
    print("All plots generated and saved successfully.")

if __name__ == "__main__":
    main()
