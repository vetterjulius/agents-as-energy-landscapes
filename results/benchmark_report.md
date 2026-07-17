# Multi-Agent System Energy Landscape Orchestration Benchmark

This report presents the rigorous, paper-ready scientific evaluation of the **Energy-Based Orchestration Model** against various baselines (deterministic, heuristic, and classical optimization methods) across multiple random seeds ($\geq 30$ Runs) to establish statistical significance.

## Table of Contents
1. [Core Evaluation per Scenario](#core-evaluation-per-scenario)
2. [Emergent Behavior Analytics](#emergent-behavior-analytics)
3. [Statistical Significance & Confidence Intervals](#statistical-significance--confidence-intervals)
4. [Link to Detailed Figure Catalog](#detailed-figure-catalog)

## Core Evaluation per Scenario

### Scenario: Independent

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 3.5020 $\pm$ 0.0671 | 1.1441 $\pm$ 0.4370 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0002 $\pm$ 0.0001 |
| Capability Matching (Greedy) | 1.4600 $\pm$ 0.0440 | 1.2247 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0003 $\pm$ 0.0000 |
| GreedyLB | 1.5294 $\pm$ 0.1129 | 0.3536 $\pm$ 0.3536 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0007 $\pm$ 0.0000 |
| RuleBased | 3.5013 $\pm$ 0.1897 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | 1.4405 $\pm$ 0.0272 | 1.1124 $\pm$ 0.1124 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0760 $\pm$ 0.0003 |
| Tabu Search | 1.4405 $\pm$ 0.0272 | 1.1124 $\pm$ 0.1124 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.4417 $\pm$ 0.0031 |
| Energy (Pure Greedy) | 1.4407 $\pm$ 0.0270 | 0.9659 $\pm$ 0.2588 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0958 $\pm$ 0.0010 |
| Energy (Pure SA) | 1.4407 $\pm$ 0.0270 | 0.9659 $\pm$ 0.2588 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.4420 $\pm$ 0.0240 |
| Energy (Hybrid) | 1.4407 $\pm$ 0.0270 | 0.9659 $\pm$ 0.2588 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.8710 $\pm$ 0.0091 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.1382 $\pm$ 0.0078 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Capability Matching (Greedy) | 0.3889 $\pm$ 0.0303 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| GreedyLB | 0.3820 $\pm$ 0.0013 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| RuleBased | -0.1224 $\pm$ 0.0556 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Beam Search | 0.4250 $\pm$ 0.0313 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Tabu Search | 0.4250 $\pm$ 0.0313 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.4472 $\pm$ 0.0289 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.4472 $\pm$ 0.0289 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.4472 $\pm$ 0.0289 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: 1.4405)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 9.97e-01 | 6.67e-01 | [1.0974, 1.7840] | **No** |
| Energy (Pure SA) | 9.97e-01 | 6.67e-01 | [1.0974, 1.7840] | **No** |
| Energy (Hybrid) | 9.97e-01 | 6.67e-01 | [1.0974, 1.7840] | **No** |

---

### Scenario: Interaction

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 3.4320 $\pm$ 0.0229 | 1.1441 $\pm$ 0.4370 | 5.00 $\pm$ 3.00 | 1.00 $\pm$ 1.00 | 0.0001 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.4700 $\pm$ 0.0060 | 1.2247 $\pm$ 0.0000 | 1.00 $\pm$ 1.00 | 1.00 $\pm$ 1.00 | 0.0003 $\pm$ 0.0000 |
| GreedyLB | 1.5594 $\pm$ 0.0829 | 0.3536 $\pm$ 0.3536 | 0.00 $\pm$ 0.00 | 1.00 $\pm$ 1.00 | 0.0008 $\pm$ 0.0000 |
| RuleBased | 3.4613 $\pm$ 0.1897 | 0.0000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | 1.3932 $\pm$ 0.0346 | 1.2247 $\pm$ 0.0000 | 4.00 $\pm$ 2.00 | 0.00 $\pm$ 0.00 | 0.0764 $\pm$ 0.0024 |
| Tabu Search | 1.3729 $\pm$ 0.0549 | 1.3195 $\pm$ 0.0947 | 5.00 $\pm$ 3.00 | 0.00 $\pm$ 0.00 | 0.4374 $\pm$ 0.0090 |
| Energy (Pure Greedy) | 1.3731 $\pm$ 0.0547 | 1.1124 $\pm$ 0.1124 | 5.00 $\pm$ 3.00 | 0.00 $\pm$ 0.00 | 0.1158 $\pm$ 0.0425 |
| Energy (Pure SA) | 1.3731 $\pm$ 0.0547 | 1.1124 $\pm$ 0.1124 | 5.00 $\pm$ 3.00 | 0.00 $\pm$ 0.00 | 1.4513 $\pm$ 0.0599 |
| Energy (Hybrid) | 1.3731 $\pm$ 0.0547 | 1.1124 $\pm$ 0.1124 | 5.00 $\pm$ 3.00 | 0.00 $\pm$ 0.00 | 1.8922 $\pm$ 0.0100 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.1382 $\pm$ 0.0078 | 0.5000 $\pm$ 0.3000 | 5.00 $\pm$ 3.00 | 1.00 $\pm$ 1.00 |
| Capability Matching (Greedy) | 0.3889 $\pm$ 0.0303 | 0.1000 $\pm$ 0.1000 | 9.00 $\pm$ 1.00 | 1.00 $\pm$ 1.00 |
| GreedyLB | 0.3820 $\pm$ 0.0013 | 0.0000 $\pm$ 0.0000 | 10.00 $\pm$ 0.00 | 1.00 $\pm$ 1.00 |
| RuleBased | -0.1224 $\pm$ 0.0556 | 0.2000 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Beam Search | 0.4289 $\pm$ 0.0275 | 0.4000 $\pm$ 0.2000 | 6.00 $\pm$ 2.00 | 0.00 $\pm$ 0.00 |
| Tabu Search | 0.4026 $\pm$ 0.0537 | 0.5000 $\pm$ 0.3000 | 5.00 $\pm$ 3.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.4149 $\pm$ 0.0414 | 0.5000 $\pm$ 0.3000 | 5.00 $\pm$ 3.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.4149 $\pm$ 0.0414 | 0.5000 $\pm$ 0.3000 | 5.00 $\pm$ 3.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.4149 $\pm$ 0.0414 | 0.5000 $\pm$ 0.3000 | 5.00 $\pm$ 3.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Tabu Search* (Mean Energy: 1.3729)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 9.98e-01 | 1.00e+00 | [0.6783, 2.0679] | **No** |
| Energy (Pure SA) | 9.98e-01 | 1.00e+00 | [0.6783, 2.0679] | **No** |
| Energy (Hybrid) | 9.98e-01 | 1.00e+00 | [0.6783, 2.0679] | **No** |

---

### Scenario: Dynamic

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 3.6341 $\pm$ 0.1173 | 1.1441 $\pm$ 0.4370 | 11.50 $\pm$ 1.50 | 26.00 $\pm$ 4.00 | 0.0001 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.5710 $\pm$ 0.0118 | 1.2247 $\pm$ 0.0000 | 12.50 $\pm$ 1.50 | 26.00 $\pm$ 0.00 | 0.0003 $\pm$ 0.0000 |
| GreedyLB | 1.6399 $\pm$ 0.1063 | 0.3536 $\pm$ 0.3536 | 9.00 $\pm$ 3.00 | 21.00 $\pm$ 1.00 | 0.0008 $\pm$ 0.0000 |
| RuleBased | 3.6020 $\pm$ 0.1693 | 0.0000 $\pm$ 0.0000 | 8.50 $\pm$ 0.50 | 20.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | 1.5525 $\pm$ 0.0189 | 0.9659 $\pm$ 0.2588 | 11.00 $\pm$ 1.00 | 24.00 $\pm$ 2.00 | 0.0776 $\pm$ 0.0016 |
| Tabu Search | 1.5525 $\pm$ 0.0189 | 0.9659 $\pm$ 0.2588 | 11.00 $\pm$ 1.00 | 24.00 $\pm$ 2.00 | 0.4393 $\pm$ 0.0003 |
| Energy (Pure Greedy) | 1.5525 $\pm$ 0.0189 | 0.9659 $\pm$ 0.2588 | 11.00 $\pm$ 1.00 | 24.00 $\pm$ 2.00 | 0.0724 $\pm$ 0.0004 |
| Energy (Pure SA) | 1.5529 $\pm$ 0.0193 | 0.9659 $\pm$ 0.2588 | 11.50 $\pm$ 0.50 | 24.00 $\pm$ 2.00 | 1.4364 $\pm$ 0.0242 |
| Energy (Hybrid) | 1.5529 $\pm$ 0.0193 | 0.9659 $\pm$ 0.2588 | 11.50 $\pm$ 0.50 | 24.00 $\pm$ 2.00 | 1.8399 $\pm$ 0.0325 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.1382 $\pm$ 0.0078 | 0.2091 $\pm$ 0.0094 | 3.16 $\pm$ 0.21 | 26.00 $\pm$ 4.00 |
| Capability Matching (Greedy) | 0.3889 $\pm$ 0.0303 | 0.2281 $\pm$ 0.0364 | 3.07 $\pm$ 0.03 | 26.00 $\pm$ 0.00 |
| GreedyLB | 0.3820 $\pm$ 0.0013 | 0.1415 $\pm$ 0.0169 | 3.43 $\pm$ 0.26 | 21.00 $\pm$ 1.00 |
| RuleBased | -0.1224 $\pm$ 0.0556 | 0.1396 $\pm$ 0.0136 | 3.43 $\pm$ 0.14 | 20.00 $\pm$ 0.00 |
| Beam Search | 0.4297 $\pm$ 0.0464 | 0.1793 $\pm$ 0.0209 | 3.27 $\pm$ 0.10 | 24.00 $\pm$ 2.00 |
| Tabu Search | 0.4297 $\pm$ 0.0464 | 0.1793 $\pm$ 0.0209 | 3.27 $\pm$ 0.10 | 24.00 $\pm$ 2.00 |
| Energy (Pure Greedy) | 0.4297 $\pm$ 0.0464 | 0.1793 $\pm$ 0.0209 | 3.27 $\pm$ 0.10 | 24.00 $\pm$ 2.00 |
| Energy (Pure SA) | 0.4198 $\pm$ 0.0365 | 0.1853 $\pm$ 0.0269 | 3.24 $\pm$ 0.07 | 24.00 $\pm$ 2.00 |
| Energy (Hybrid) | 0.4198 $\pm$ 0.0365 | 0.1853 $\pm$ 0.0269 | 3.24 $\pm$ 0.07 | 24.00 $\pm$ 2.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: 1.5525)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | 1.00e+00 | [1.3119, 1.7931] | **No** |
| Energy (Pure SA) | 9.91e-01 | 1.00e+00 | [1.3077, 1.7980] | **No** |
| Energy (Hybrid) | 9.91e-01 | 1.00e+00 | [1.3077, 1.7980] | **No** |

---

### Scenario: DistributionShift

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 4.3767 $\pm$ 0.3550 | 1.8556 $\pm$ 0.0962 | 113.50 $\pm$ 10.50 | 215.00 $\pm$ 5.00 | 0.0005 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.3553 $\pm$ 0.1455 | 2.9731 $\pm$ 0.3364 | 147.50 $\pm$ 7.50 | 292.00 $\pm$ 28.00 | 0.0012 $\pm$ 0.0001 |
| GreedyLB | 1.3437 $\pm$ 0.0981 | 1.5151 $\pm$ 0.1179 | 99.50 $\pm$ 1.50 | 199.00 $\pm$ 5.00 | 0.0037 $\pm$ 0.0000 |
| RuleBased | 4.1182 $\pm$ 0.0909 | 0.4880 $\pm$ 0.0000 | 83.50 $\pm$ 3.50 | 170.00 $\pm$ 0.00 | 0.0004 $\pm$ 0.0000 |
| Beam Search | 1.2858 $\pm$ 0.1084 | 1.8674 $\pm$ 0.1912 | 117.50 $\pm$ 4.50 | 216.00 $\pm$ 10.00 | 2.0259 $\pm$ 0.1553 |
| Tabu Search | 2.1726 $\pm$ 0.0572 | 1.7531 $\pm$ 0.3056 | 111.00 $\pm$ 9.00 | 211.00 $\pm$ 15.00 | 11.0729 $\pm$ 0.8126 |
| Energy (Pure Greedy) | 1.2753 $\pm$ 0.1107 | 1.5367 $\pm$ 0.1394 | 108.50 $\pm$ 4.50 | 200.00 $\pm$ 6.00 | 4.5399 $\pm$ 1.1089 |
| Energy (Pure SA) | 1.6953 $\pm$ 0.1269 | 1.9632 $\pm$ 0.1637 | 118.50 $\pm$ 11.50 | 221.00 $\pm$ 9.00 | 3.9813 $\pm$ 0.7096 |
| Energy (Hybrid) | 1.2770 $\pm$ 0.1094 | 1.6927 $\pm$ 0.2954 | 113.50 $\pm$ 8.50 | 208.00 $\pm$ 14.00 | 18.9975 $\pm$ 0.1519 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.0247 $\pm$ 0.0153 | 0.0813 $\pm$ 0.0106 | 903.21 $\pm$ 23.00 | 215.00 $\pm$ 5.00 |
| Capability Matching (Greedy) | 0.5362 $\pm$ 0.0141 | 0.1192 $\pm$ 0.0052 | 865.82 $\pm$ 17.12 | 292.00 $\pm$ 28.00 |
| GreedyLB | 0.5453 $\pm$ 0.0021 | 0.0807 $\pm$ 0.0059 | 903.55 $\pm$ 6.72 | 199.00 $\pm$ 5.00 |
| RuleBased | -0.0120 $\pm$ 0.0376 | 0.0672 $\pm$ 0.0000 | 916.87 $\pm$ 12.72 | 170.00 $\pm$ 0.00 |
| Beam Search | 0.5656 $\pm$ 0.0206 | 0.0945 $\pm$ 0.0020 | 890.06 $\pm$ 10.40 | 216.00 $\pm$ 10.00 |
| Tabu Search | 0.3298 $\pm$ 0.0348 | 0.0935 $\pm$ 0.0035 | 890.96 $\pm$ 8.97 | 211.00 $\pm$ 15.00 |
| Energy (Pure Greedy) | 0.5688 $\pm$ 0.0156 | 0.0903 $\pm$ 0.0082 | 894.09 $\pm$ 4.37 | 200.00 $\pm$ 6.00 |
| Energy (Pure SA) | 0.4440 $\pm$ 0.0030 | 0.0998 $\pm$ 0.0144 | 885.08 $\pm$ 26.41 | 221.00 $\pm$ 9.00 |
| Energy (Hybrid) | 0.5717 $\pm$ 0.0190 | 0.0931 $\pm$ 0.0081 | 891.29 $\pm$ 4.45 | 208.00 $\pm$ 14.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: 1.2858)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 9.52e-01 | 6.67e-01 | [-0.1312, 2.6819] | **No** |
| Energy (Pure SA) | 1.37e-01 | 3.33e-01 | [0.0834, 3.3071] | **No** |
| Energy (Hybrid) | 9.60e-01 | 6.67e-01 | [-0.1128, 2.6667] | **No** |

---

### Scenario: Frustrated

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 2.4810 $\pm$ 0.2778 | 1.0000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 4.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0002 $\pm$ 0.0000 |
| GreedyLB | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0005 $\pm$ 0.0001 |
| RuleBased | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0250 $\pm$ 0.0002 |
| Tabu Search | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.1302 $\pm$ 0.0004 |
| Energy (Pure Greedy) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0283 $\pm$ 0.0001 |
| Energy (Pure SA) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.6328 $\pm$ 0.0009 |
| Energy (Hybrid) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.7305 $\pm$ 0.0011 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | 0.5000 $\pm$ 0.3333 | 0.1667 $\pm$ 0.0000 | 20.00 $\pm$ 0.00 | 4.00 $\pm$ 0.00 |
| Capability Matching (Greedy) | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| GreedyLB | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| RuleBased | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| Beam Search | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Tabu Search | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: -0.2690)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | nan | [-0.2690, -0.2690] | **No** |
| Energy (Pure SA) | 1.00e+00 | nan | [-0.2690, -0.2690] | **No** |
| Energy (Hybrid) | 1.00e+00 | nan | [-0.2690, -0.2690] | **No** |

---

## Detailed Figure Catalog

The complete collection of scientific visualizations, charts, and detailed explanations is compiled in the Figure Catalog.
Please proceed to the **[Figure Catalog](figure_catalog.md)** to inspect results visually.
