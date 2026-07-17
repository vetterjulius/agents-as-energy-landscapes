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
| Random | 3.3333 $\pm$ 0.6406 | 1.2537 $\pm$ 0.4019 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0024 $\pm$ 0.0119 |
| Capability Matching (Greedy) | 1.6286 $\pm$ 0.3863 | 1.8516 $\pm$ 0.5815 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0005 $\pm$ 0.0003 |
| GreedyLB | 1.7891 $\pm$ 0.4536 | 0.9122 $\pm$ 0.3669 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0013 $\pm$ 0.0001 |
| RuleBased | 3.2871 $\pm$ 0.6163 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0002 $\pm$ 0.0000 |
| Beam Search | 1.6195 $\pm$ 0.3866 | 1.7846 $\pm$ 0.5309 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0937 $\pm$ 0.0064 |
| Tabu Search | 1.6195 $\pm$ 0.3866 | 1.7846 $\pm$ 0.5309 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.5035 $\pm$ 0.0386 |
| Energy (Pure Greedy) | 1.6195 $\pm$ 0.3866 | 1.7748 $\pm$ 0.5477 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0882 $\pm$ 0.0200 |
| Energy (Pure SA) | 1.6203 $\pm$ 0.3878 | 1.7676 $\pm$ 0.5559 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.6088 $\pm$ 0.1441 |
| Energy (Hybrid) | 1.6207 $\pm$ 0.3877 | 1.7770 $\pm$ 0.5558 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 2.0247 $\pm$ 0.1909 |
| EBMAO (Pure Greedy) | 1.6248 $\pm$ 0.3845 | 1.8704 $\pm$ 0.5641 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0904 $\pm$ 0.0108 |
| EBMAO (Pure SA) | 1.6251 $\pm$ 0.3853 | 1.8710 $\pm$ 0.5767 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.8882 $\pm$ 0.2590 |
| EBMAO (Hybrid) | 1.6248 $\pm$ 0.3854 | 1.8595 $\pm$ 0.5706 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 2.3205 $\pm$ 0.3178 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.0050 $\pm$ 0.1158 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Capability Matching (Greedy) | 0.3843 $\pm$ 0.1015 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| GreedyLB | 0.3701 $\pm$ 0.1073 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| RuleBased | -0.0071 $\pm$ 0.1093 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Beam Search | 0.4002 $\pm$ 0.0973 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Tabu Search | 0.4002 $\pm$ 0.0973 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.4016 $\pm$ 0.0978 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.4026 $\pm$ 0.0994 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.4022 $\pm$ 0.0990 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.3853 $\pm$ 0.1023 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.3857 $\pm$ 0.1021 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.3874 $\pm$ 0.1015 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: 1.6195)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | 9.94e-01 | [1.4727, 1.7663] | **No** |
| Energy (Pure SA) | 9.93e-01 | 9.76e-01 | [1.4730, 1.7676] | **No** |
| Energy (Hybrid) | 9.90e-01 | 9.59e-01 | [1.4735, 1.7680] | **No** |

---

### Scenario: Interaction

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 3.3193 $\pm$ 0.6404 | 1.2537 $\pm$ 0.4019 | 1.80 $\pm$ 1.74 | 0.73 $\pm$ 1.09 | 0.0002 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.6033 $\pm$ 0.3763 | 1.8516 $\pm$ 0.5815 | 2.67 $\pm$ 1.89 | 0.93 $\pm$ 1.44 | 0.0004 $\pm$ 0.0000 |
| GreedyLB | 1.7724 $\pm$ 0.4469 | 0.9122 $\pm$ 0.3669 | 1.53 $\pm$ 1.12 | 0.47 $\pm$ 0.85 | 0.0012 $\pm$ 0.0001 |
| RuleBased | 3.2704 $\pm$ 0.6283 | 0.0000 $\pm$ 0.0000 | 1.13 $\pm$ 1.61 | 0.20 $\pm$ 0.60 | 0.0001 $\pm$ 0.0000 |
| Beam Search | 1.5636 $\pm$ 0.3785 | 1.8499 $\pm$ 0.5727 | 4.13 $\pm$ 2.12 | 0.27 $\pm$ 0.68 | 0.0876 $\pm$ 0.0058 |
| Tabu Search | 1.5592 $\pm$ 0.3793 | 1.8614 $\pm$ 0.5643 | 4.47 $\pm$ 2.40 | 0.27 $\pm$ 0.68 | 0.4923 $\pm$ 0.0567 |
| Energy (Pure Greedy) | 1.5602 $\pm$ 0.3800 | 1.8438 $\pm$ 0.5327 | 4.27 $\pm$ 2.17 | 0.27 $\pm$ 0.68 | 0.0933 $\pm$ 0.0240 |
| Energy (Pure SA) | 1.5611 $\pm$ 0.3800 | 1.8378 $\pm$ 0.6237 | 4.47 $\pm$ 2.40 | 0.27 $\pm$ 0.68 | 1.5621 $\pm$ 0.1233 |
| Energy (Hybrid) | 1.5612 $\pm$ 0.3808 | 1.8363 $\pm$ 0.5429 | 4.27 $\pm$ 2.17 | 0.27 $\pm$ 0.68 | 1.9551 $\pm$ 0.1009 |
| EBMAO (Pure Greedy) | 1.6007 $\pm$ 0.3787 | 1.9756 $\pm$ 0.6302 | 3.87 $\pm$ 1.86 | 1.33 $\pm$ 1.40 | 0.0962 $\pm$ 0.0164 |
| EBMAO (Pure SA) | 1.5992 $\pm$ 0.3791 | 2.0363 $\pm$ 0.6353 | 4.07 $\pm$ 1.90 | 1.33 $\pm$ 1.40 | 1.8037 $\pm$ 0.3059 |
| EBMAO (Hybrid) | 1.6001 $\pm$ 0.3808 | 1.9953 $\pm$ 0.6072 | 3.80 $\pm$ 1.89 | 1.33 $\pm$ 1.40 | 2.2522 $\pm$ 0.5908 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.0050 $\pm$ 0.1158 | 0.1883 $\pm$ 0.1801 | 7.87 $\pm$ 1.93 | 0.73 $\pm$ 1.09 |
| Capability Matching (Greedy) | 0.3843 $\pm$ 0.1015 | 0.2750 $\pm$ 0.1905 | 7.00 $\pm$ 1.91 | 0.93 $\pm$ 1.44 |
| GreedyLB | 0.3701 $\pm$ 0.1073 | 0.1567 $\pm$ 0.1138 | 8.13 $\pm$ 1.15 | 0.47 $\pm$ 0.85 |
| RuleBased | -0.0071 $\pm$ 0.1093 | 0.1133 $\pm$ 0.1607 | 8.53 $\pm$ 1.54 | 0.20 $\pm$ 0.60 |
| Beam Search | 0.3932 $\pm$ 0.1064 | 0.4300 $\pm$ 0.2283 | 5.53 $\pm$ 2.23 | 0.27 $\pm$ 0.68 |
| Tabu Search | 0.3903 $\pm$ 0.1081 | 0.4633 $\pm$ 0.2523 | 5.20 $\pm$ 2.45 | 0.27 $\pm$ 0.68 |
| Energy (Pure Greedy) | 0.3936 $\pm$ 0.1025 | 0.4433 $\pm$ 0.2319 | 5.40 $\pm$ 2.26 | 0.27 $\pm$ 0.68 |
| Energy (Pure SA) | 0.3946 $\pm$ 0.1059 | 0.4633 $\pm$ 0.2523 | 5.20 $\pm$ 2.45 | 0.27 $\pm$ 0.68 |
| Energy (Hybrid) | 0.3941 $\pm$ 0.1020 | 0.4433 $\pm$ 0.2319 | 5.40 $\pm$ 2.26 | 0.27 $\pm$ 0.68 |
| EBMAO (Pure Greedy) | 0.3802 $\pm$ 0.1071 | 0.3983 $\pm$ 0.1832 | 5.80 $\pm$ 1.81 | 1.33 $\pm$ 1.40 |
| EBMAO (Pure SA) | 0.3792 $\pm$ 0.1083 | 0.4200 $\pm$ 0.1882 | 5.60 $\pm$ 1.89 | 1.33 $\pm$ 1.40 |
| EBMAO (Hybrid) | 0.3827 $\pm$ 0.1068 | 0.3917 $\pm$ 0.1867 | 5.87 $\pm$ 1.86 | 1.33 $\pm$ 1.40 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Tabu Search* (Mean Energy: 1.5592)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 9.92e-01 | 9.65e-01 | [1.4159, 1.7046] | **No** |
| Energy (Pure SA) | 9.85e-01 | 9.47e-01 | [1.4168, 1.7054] | **No** |
| Energy (Hybrid) | 9.84e-01 | 9.53e-01 | [1.4166, 1.7058] | **No** |

---

### Scenario: Dynamic

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 3.4751 $\pm$ 0.6357 | 1.2537 $\pm$ 0.4019 | 13.83 $\pm$ 3.52 | 26.93 $\pm$ 4.12 | 0.0002 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.8121 $\pm$ 0.3888 | 1.8516 $\pm$ 0.5815 | 18.00 $\pm$ 6.31 | 35.07 $\pm$ 8.88 | 0.0004 $\pm$ 0.0001 |
| GreedyLB | 1.9142 $\pm$ 0.4386 | 0.9122 $\pm$ 0.3669 | 12.20 $\pm$ 2.98 | 23.87 $\pm$ 2.53 | 0.0012 $\pm$ 0.0001 |
| RuleBased | 3.3867 $\pm$ 0.6233 | 0.0000 $\pm$ 0.0000 | 10.13 $\pm$ 2.32 | 20.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | 1.7764 $\pm$ 0.3814 | 1.4707 $\pm$ 0.5037 | 15.73 $\pm$ 5.03 | 29.67 $\pm$ 6.43 | 0.0872 $\pm$ 0.0056 |
| Tabu Search | 1.7764 $\pm$ 0.3814 | 1.4707 $\pm$ 0.5037 | 15.73 $\pm$ 5.03 | 29.67 $\pm$ 6.43 | 0.4804 $\pm$ 0.0084 |
| Energy (Pure Greedy) | 1.7764 $\pm$ 0.3813 | 1.4664 $\pm$ 0.4997 | 15.80 $\pm$ 5.02 | 29.60 $\pm$ 6.37 | 0.0874 $\pm$ 0.0107 |
| Energy (Pure SA) | 1.7771 $\pm$ 0.3812 | 1.4578 $\pm$ 0.5552 | 15.80 $\pm$ 5.23 | 29.73 $\pm$ 7.00 | 1.5346 $\pm$ 0.0714 |
| Energy (Hybrid) | 1.7773 $\pm$ 0.3817 | 1.4697 $\pm$ 0.5538 | 15.87 $\pm$ 5.19 | 29.87 $\pm$ 6.97 | 1.9390 $\pm$ 0.0987 |
| EBMAO (Pure Greedy) | 1.8526 $\pm$ 0.4068 | 2.1342 $\pm$ 0.7495 | 21.00 $\pm$ 8.69 | 40.47 $\pm$ 13.96 | 0.0953 $\pm$ 0.0333 |
| EBMAO (Pure SA) | 1.8448 $\pm$ 0.4032 | 2.0732 $\pm$ 0.7759 | 20.73 $\pm$ 8.81 | 39.60 $\pm$ 14.31 | 1.7162 $\pm$ 0.0857 |
| EBMAO (Hybrid) | 1.8502 $\pm$ 0.4072 | 2.1035 $\pm$ 0.7801 | 20.90 $\pm$ 8.75 | 40.13 $\pm$ 14.29 | 2.0895 $\pm$ 0.0923 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.0050 $\pm$ 0.1158 | 0.2952 $\pm$ 0.0933 | 2.79 $\pm$ 0.47 | 26.93 $\pm$ 4.12 |
| Capability Matching (Greedy) | 0.3843 $\pm$ 0.1015 | 0.3552 $\pm$ 0.1210 | 2.56 $\pm$ 0.57 | 35.07 $\pm$ 8.88 |
| GreedyLB | 0.3701 $\pm$ 0.1073 | 0.2462 $\pm$ 0.0729 | 3.00 $\pm$ 0.51 | 23.87 $\pm$ 2.53 |
| RuleBased | -0.0071 $\pm$ 0.1093 | 0.2038 $\pm$ 0.0567 | 3.15 $\pm$ 0.38 | 20.00 $\pm$ 0.00 |
| Beam Search | 0.4023 $\pm$ 0.0921 | 0.3148 $\pm$ 0.1072 | 2.71 $\pm$ 0.52 | 29.67 $\pm$ 6.43 |
| Tabu Search | 0.4023 $\pm$ 0.0921 | 0.3148 $\pm$ 0.1072 | 2.71 $\pm$ 0.52 | 29.67 $\pm$ 6.43 |
| Energy (Pure Greedy) | 0.4024 $\pm$ 0.0924 | 0.3140 $\pm$ 0.1067 | 2.72 $\pm$ 0.52 | 29.60 $\pm$ 6.37 |
| Energy (Pure SA) | 0.4020 $\pm$ 0.0924 | 0.3120 $\pm$ 0.1110 | 2.72 $\pm$ 0.53 | 29.73 $\pm$ 7.00 |
| Energy (Hybrid) | 0.4014 $\pm$ 0.0932 | 0.3146 $\pm$ 0.1113 | 2.71 $\pm$ 0.53 | 29.87 $\pm$ 6.97 |
| EBMAO (Pure Greedy) | 0.3689 $\pm$ 0.1107 | 0.4184 $\pm$ 0.1579 | 2.31 $\pm$ 0.72 | 40.47 $\pm$ 13.96 |
| EBMAO (Pure SA) | 0.3757 $\pm$ 0.1044 | 0.4064 $\pm$ 0.1637 | 2.35 $\pm$ 0.74 | 39.60 $\pm$ 14.31 |
| EBMAO (Hybrid) | 0.3729 $\pm$ 0.1094 | 0.4127 $\pm$ 0.1634 | 2.33 $\pm$ 0.74 | 40.13 $\pm$ 14.29 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: 1.7764)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | 9.94e-01 | [1.6316, 1.9212] | **No** |
| Energy (Pure SA) | 9.94e-01 | 9.47e-01 | [1.6324, 1.9219] | **No** |
| Energy (Hybrid) | 9.92e-01 | 9.41e-01 | [1.6324, 1.9223] | **No** |

---

### Scenario: DistributionShift

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 4.5159 $\pm$ 0.4276 | 1.7310 $\pm$ 0.2741 | 105.97 $\pm$ 10.01 | 209.67 $\pm$ 13.33 | 0.0007 $\pm$ 0.0001 |
| Capability Matching (Greedy) | 1.4815 $\pm$ 0.1670 | 2.9957 $\pm$ 0.6450 | 148.97 $\pm$ 30.20 | 298.13 $\pm$ 62.16 | 0.0017 $\pm$ 0.0001 |
| GreedyLB | 1.4953 $\pm$ 0.1709 | 1.5114 $\pm$ 0.3182 | 98.97 $\pm$ 9.57 | 200.07 $\pm$ 14.23 | 0.0054 $\pm$ 0.0004 |
| RuleBased | 4.4495 $\pm$ 0.4380 | 0.4880 $\pm$ 0.0000 | 84.33 $\pm$ 7.05 | 170.00 $\pm$ 0.00 | 0.0006 $\pm$ 0.0001 |
| Beam Search | 1.4027 $\pm$ 0.1552 | 2.1060 $\pm$ 0.4397 | 122.07 $\pm$ 14.59 | 231.47 $\pm$ 26.13 | 1.7693 $\pm$ 0.0358 |
| Tabu Search | 2.3673 $\pm$ 0.2779 | 1.6867 $\pm$ 0.3049 | 109.43 $\pm$ 10.64 | 207.80 $\pm$ 14.29 | 11.4098 $\pm$ 0.1123 |
| Energy (Pure Greedy) | 1.3998 $\pm$ 0.1546 | 1.9189 $\pm$ 0.4008 | 116.27 $\pm$ 14.44 | 220.47 $\pm$ 21.35 | 4.4989 $\pm$ 1.4853 |
| Energy (Pure SA) | 1.9038 $\pm$ 0.2701 | 2.0325 $\pm$ 0.3931 | 116.57 $\pm$ 13.37 | 226.67 $\pm$ 22.41 | 5.9181 $\pm$ 1.3784 |
| Energy (Hybrid) | 1.4032 $\pm$ 0.1549 | 1.9163 $\pm$ 0.4014 | 116.73 $\pm$ 14.09 | 220.33 $\pm$ 21.43 | 17.1491 $\pm$ 1.6776 |
| EBMAO (Pure Greedy) | 1.6101 $\pm$ 0.2612 | 4.0830 $\pm$ 1.3003 | 216.27 $\pm$ 87.61 | 423.73 $\pm$ 180.66 | 5.5153 $\pm$ 1.9777 |
| EBMAO (Pure SA) | 2.0239 $\pm$ 0.2891 | 2.6856 $\pm$ 0.5709 | 138.23 $\pm$ 25.31 | 272.20 $\pm$ 46.31 | 6.0542 $\pm$ 1.1969 |
| EBMAO (Hybrid) | 1.6050 $\pm$ 0.2564 | 4.0320 $\pm$ 1.2891 | 213.33 $\pm$ 83.97 | 417.53 $\pm$ 174.10 | 17.8561 $\pm$ 1.8327 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.0048 $\pm$ 0.0484 | 0.0835 $\pm$ 0.0101 | 914.00 $\pm$ 31.10 | 209.67 $\pm$ 13.33 |
| Capability Matching (Greedy) | 0.5436 $\pm$ 0.0261 | 0.1211 $\pm$ 0.0262 | 876.46 $\pm$ 37.04 | 298.13 $\pm$ 62.16 |
| GreedyLB | 0.5507 $\pm$ 0.0247 | 0.0805 $\pm$ 0.0094 | 916.90 $\pm$ 29.54 | 200.07 $\pm$ 14.23 |
| RuleBased | 0.0034 $\pm$ 0.0487 | 0.0690 $\pm$ 0.0072 | 928.31 $\pm$ 28.20 | 170.00 $\pm$ 0.00 |
| Beam Search | 0.5669 $\pm$ 0.0243 | 0.1030 $\pm$ 0.0137 | 894.44 $\pm$ 29.82 | 231.47 $\pm$ 26.13 |
| Tabu Search | 0.3455 $\pm$ 0.0391 | 0.0913 $\pm$ 0.0091 | 906.08 $\pm$ 26.56 | 207.80 $\pm$ 14.29 |
| Energy (Pure Greedy) | 0.5691 $\pm$ 0.0235 | 0.0970 $\pm$ 0.0135 | 900.41 $\pm$ 29.28 | 220.47 $\pm$ 21.35 |
| Energy (Pure SA) | 0.4371 $\pm$ 0.0473 | 0.0977 $\pm$ 0.0142 | 899.81 $\pm$ 31.28 | 226.67 $\pm$ 22.41 |
| Energy (Hybrid) | 0.5700 $\pm$ 0.0237 | 0.0975 $\pm$ 0.0129 | 899.95 $\pm$ 30.55 | 220.33 $\pm$ 21.43 |
| EBMAO (Pure Greedy) | 0.5125 $\pm$ 0.0497 | 0.1779 $\pm$ 0.0719 | 819.67 $\pm$ 74.14 | 423.73 $\pm$ 180.66 |
| EBMAO (Pure SA) | 0.4003 $\pm$ 0.0523 | 0.1147 $\pm$ 0.0221 | 882.90 $\pm$ 35.66 | 272.20 $\pm$ 46.31 |
| EBMAO (Hybrid) | 0.5155 $\pm$ 0.0488 | 0.1747 $\pm$ 0.0683 | 822.79 $\pm$ 70.98 | 417.53 $\pm$ 174.10 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: 1.4027)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 9.44e-01 | 9.35e-01 | [1.3411, 1.4585] | **No** |
| Energy (Pure SA) | 3.03e-11 | 4.62e-10 | [1.8012, 2.0063] | **Yes** |
| Energy (Hybrid) | 9.89e-01 | 9.71e-01 | [1.3444, 1.4621] | **No** |

---

### Scenario: Frustrated

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 1.2551 $\pm$ 0.9687 | 1.3805 $\pm$ 0.5723 | 4.53 $\pm$ 2.25 | 2.13 $\pm$ 1.63 | 0.0001 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0003 $\pm$ 0.0001 |
| GreedyLB | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0007 $\pm$ 0.0001 |
| RuleBased | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0288 $\pm$ 0.0021 |
| Tabu Search | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.1485 $\pm$ 0.0066 |
| Energy (Pure Greedy) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0316 $\pm$ 0.0006 |
| Energy (Pure SA) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.6876 $\pm$ 0.0222 |
| Energy (Hybrid) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.8030 $\pm$ 0.0358 |
| EBMAO (Pure Greedy) | 2.6199 $\pm$ 0.0000 | 3.4641 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0279 $\pm$ 0.0026 |
| EBMAO (Pure SA) | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.7397 $\pm$ 0.0355 |
| EBMAO (Hybrid) | 2.6199 $\pm$ 0.0000 | 3.4641 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.8641 $\pm$ 0.0326 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | 0.3889 $\pm$ 0.2208 | 0.3778 $\pm$ 0.1872 | 14.93 $\pm$ 4.49 | 2.13 $\pm$ 1.63 |
| Capability Matching (Greedy) | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| GreedyLB | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| RuleBased | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| Beam Search | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Tabu Search | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.3333 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.3333 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |

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
