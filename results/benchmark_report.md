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
| Random | 3.5690 $\pm$ 0.0000 | 1.5811 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0003 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.4160 $\pm$ 0.0000 | 1.2247 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0007 $\pm$ 0.0000 |
| GreedyLB | 1.4164 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0009 $\pm$ 0.0000 |
| RuleBased | 3.3115 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | 1.4133 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.1123 $\pm$ 0.0000 |
| Tabu Search | 1.4133 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.4638 $\pm$ 0.0000 |
| Energy (Pure Greedy) | 1.4137 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0971 $\pm$ 0.0000 |
| Energy (Pure SA) | 1.4137 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.6073 $\pm$ 0.0000 |
| Energy (Hybrid) | 1.4137 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.7545 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 1.4160 $\pm$ 0.0000 | 1.2247 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.1202 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 1.4133 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 2.1951 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 1.4133 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 2.9735 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.1459 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Capability Matching (Greedy) | 0.3587 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| GreedyLB | 0.3833 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| RuleBased | -0.1780 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Beam Search | 0.3936 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Tabu Search | 0.3936 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.4183 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.4183 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.4183 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.3587 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.3936 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.3936 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: 1.4133)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | 1.00e+00 | [1.4137, 1.4137] | **No** |
| Energy (Pure SA) | 1.00e+00 | 1.00e+00 | [1.4137, 1.4137] | **No** |
| Energy (Hybrid) | 1.00e+00 | 1.00e+00 | [1.4137, 1.4137] | **No** |

---

### Scenario: Interaction

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 3.4090 $\pm$ 0.0000 | 1.5811 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0002 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.4760 $\pm$ 0.0000 | 1.2247 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 2.00 $\pm$ 0.00 | 0.0003 $\pm$ 0.0000 |
| GreedyLB | 1.4764 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 2.00 $\pm$ 0.00 | 0.0009 $\pm$ 0.0000 |
| RuleBased | 3.2715 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | 1.3586 $\pm$ 0.0000 | 1.2247 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0828 $\pm$ 0.0000 |
| Tabu Search | 1.3180 $\pm$ 0.0000 | 1.4142 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.4638 $\pm$ 0.0000 |
| Energy (Pure Greedy) | 1.3184 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.1672 $\pm$ 0.0000 |
| Energy (Pure SA) | 1.3184 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.6743 $\pm$ 0.0000 |
| Energy (Hybrid) | 1.3184 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.9497 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 1.3385 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.2316 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 1.3180 $\pm$ 0.0000 | 1.4142 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.4447 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 1.3385 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 1.8681 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.1459 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Capability Matching (Greedy) | 0.3587 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 10.00 $\pm$ 0.00 | 2.00 $\pm$ 0.00 |
| GreedyLB | 0.3833 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 10.00 $\pm$ 0.00 | 2.00 $\pm$ 0.00 |
| RuleBased | -0.1780 $\pm$ 0.0000 | 0.2000 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Beam Search | 0.4014 $\pm$ 0.0000 | 0.6000 $\pm$ 0.0000 | 4.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Tabu Search | 0.3489 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.3736 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.3736 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.3736 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.3369 $\pm$ 0.0000 | 0.6000 $\pm$ 0.0000 | 4.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.3489 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.3369 $\pm$ 0.0000 | 0.6000 $\pm$ 0.0000 | 4.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Tabu Search* (Mean Energy: 1.3180)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | 1.00e+00 | [1.3184, 1.3184] | **No** |
| Energy (Pure SA) | 1.00e+00 | 1.00e+00 | [1.3184, 1.3184] | **No** |
| Energy (Hybrid) | 1.00e+00 | 1.00e+00 | [1.3184, 1.3184] | **No** |

---

### Scenario: Dynamic

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 3.7514 $\pm$ 0.0000 | 1.5811 $\pm$ 0.0000 | 13.00 $\pm$ 0.00 | 30.00 $\pm$ 0.00 | 0.0002 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.5592 $\pm$ 0.0000 | 1.2247 $\pm$ 0.0000 | 14.00 $\pm$ 0.00 | 26.00 $\pm$ 0.00 | 0.0004 $\pm$ 0.0000 |
| GreedyLB | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 0.0008 $\pm$ 0.0000 |
| RuleBased | 3.4327 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 9.00 $\pm$ 0.00 | 20.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 0.0806 $\pm$ 0.0000 |
| Tabu Search | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 0.4613 $\pm$ 0.0000 |
| Energy (Pure Greedy) | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 0.0778 $\pm$ 0.0000 |
| Energy (Pure SA) | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 1.6989 $\pm$ 0.0000 |
| Energy (Hybrid) | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 1.8322 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 0.1055 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 1.5287 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 1.5336 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 2.1516 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.1459 $\pm$ 0.0000 | 0.2185 $\pm$ 0.0000 | 2.94 $\pm$ 0.00 | 30.00 $\pm$ 0.00 |
| Capability Matching (Greedy) | 0.3587 $\pm$ 0.0000 | 0.1918 $\pm$ 0.0000 | 3.04 $\pm$ 0.00 | 26.00 $\pm$ 0.00 |
| GreedyLB | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| RuleBased | -0.1780 $\pm$ 0.0000 | 0.1260 $\pm$ 0.0000 | 3.29 $\pm$ 0.00 | 20.00 $\pm$ 0.00 |
| Beam Search | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| Tabu Search | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *GreedyLB* (Mean Energy: 1.5336)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | 1.00e+00 | [1.5336, 1.5336] | **No** |
| Energy (Pure SA) | 1.00e+00 | 1.00e+00 | [1.5336, 1.5336] | **No** |
| Energy (Hybrid) | 1.00e+00 | 1.00e+00 | [1.5336, 1.5336] | **No** |

---

### Scenario: DistributionShift

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 4.0217 $\pm$ 0.0000 | 1.7593 $\pm$ 0.0000 | 103.00 $\pm$ 0.00 | 210.00 $\pm$ 0.00 | 0.0005 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 1.2097 $\pm$ 0.0000 | 2.6367 $\pm$ 0.0000 | 140.00 $\pm$ 0.00 | 264.00 $\pm$ 0.00 | 0.0013 $\pm$ 0.0000 |
| GreedyLB | 1.2456 $\pm$ 0.0000 | 1.3973 $\pm$ 0.0000 | 98.00 $\pm$ 0.00 | 194.00 $\pm$ 0.00 | 0.0035 $\pm$ 0.0000 |
| RuleBased | 4.0273 $\pm$ 0.0000 | 0.4880 $\pm$ 0.0000 | 87.00 $\pm$ 0.00 | 170.00 $\pm$ 0.00 | 0.0004 $\pm$ 0.0000 |
| Beam Search | 1.1774 $\pm$ 0.0000 | 2.0587 $\pm$ 0.0000 | 122.00 $\pm$ 0.00 | 226.00 $\pm$ 0.00 | 1.7198 $\pm$ 0.0000 |
| Tabu Search | 2.1154 $\pm$ 0.0000 | 2.0587 $\pm$ 0.0000 | 120.00 $\pm$ 0.00 | 226.00 $\pm$ 0.00 | 11.6438 $\pm$ 0.0000 |
| Energy (Pure Greedy) | 1.1646 $\pm$ 0.0000 | 1.6762 $\pm$ 0.0000 | 113.00 $\pm$ 0.00 | 206.00 $\pm$ 0.00 | 5.2123 $\pm$ 0.0000 |
| Energy (Pure SA) | 1.5684 $\pm$ 0.0000 | 1.7995 $\pm$ 0.0000 | 107.00 $\pm$ 0.00 | 212.00 $\pm$ 0.00 | 6.0669 $\pm$ 0.0000 |
| Energy (Hybrid) | 1.1676 $\pm$ 0.0000 | 1.9881 $\pm$ 0.0000 | 122.00 $\pm$ 0.00 | 222.00 $\pm$ 0.00 | 16.2093 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 1.1716 $\pm$ 0.0000 | 2.0931 $\pm$ 0.0000 | 123.00 $\pm$ 0.00 | 228.00 $\pm$ 0.00 | 5.6443 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 1.6890 $\pm$ 0.0000 | 2.4976 $\pm$ 0.0000 | 122.00 $\pm$ 0.00 | 254.00 $\pm$ 0.00 | 6.0390 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 1.1779 $\pm$ 0.0000 | 1.9881 $\pm$ 0.0000 | 122.00 $\pm$ 0.00 | 222.00 $\pm$ 0.00 | 21.9704 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | -0.0094 $\pm$ 0.0000 | 0.0706 $\pm$ 0.0000 | 926.21 $\pm$ 0.00 | 210.00 $\pm$ 0.00 |
| Capability Matching (Greedy) | 0.5503 $\pm$ 0.0000 | 0.1140 $\pm$ 0.0000 | 882.94 $\pm$ 0.00 | 264.00 $\pm$ 0.00 |
| GreedyLB | 0.5474 $\pm$ 0.0000 | 0.0866 $\pm$ 0.0000 | 910.27 $\pm$ 0.00 | 194.00 $\pm$ 0.00 |
| RuleBased | -0.0497 $\pm$ 0.0000 | 0.0672 $\pm$ 0.0000 | 929.59 $\pm$ 0.00 | 170.00 $\pm$ 0.00 |
| Beam Search | 0.5862 $\pm$ 0.0000 | 0.0965 $\pm$ 0.0000 | 900.46 $\pm$ 0.00 | 226.00 $\pm$ 0.00 |
| Tabu Search | 0.2950 $\pm$ 0.0000 | 0.0970 $\pm$ 0.0000 | 899.93 $\pm$ 0.00 | 226.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.5844 $\pm$ 0.0000 | 0.0985 $\pm$ 0.0000 | 898.46 $\pm$ 0.00 | 206.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.4471 $\pm$ 0.0000 | 0.0854 $\pm$ 0.0000 | 911.49 $\pm$ 0.00 | 212.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.5907 $\pm$ 0.0000 | 0.1012 $\pm$ 0.0000 | 895.74 $\pm$ 0.00 | 222.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.5695 $\pm$ 0.0000 | 0.1100 $\pm$ 0.0000 | 886.96 $\pm$ 0.00 | 228.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.3968 $\pm$ 0.0000 | 0.0960 $\pm$ 0.0000 | 900.93 $\pm$ 0.00 | 254.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.5596 $\pm$ 0.0000 | 0.1096 $\pm$ 0.0000 | 887.33 $\pm$ 0.00 | 222.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: 1.1774)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | 1.00e+00 | [1.1646, 1.1646] | **No** |
| Energy (Pure SA) | 1.00e+00 | 1.00e+00 | [1.5684, 1.5684] | **No** |
| Energy (Hybrid) | 1.00e+00 | 1.00e+00 | [1.1676, 1.1676] | **No** |

---

### Scenario: Frustrated

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 2.7588 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 4.00 $\pm$ 0.00 | 0.0002 $\pm$ 0.0000 |
| Capability Matching (Greedy) | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0002 $\pm$ 0.0000 |
| GreedyLB | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0005 $\pm$ 0.0000 |
| RuleBased | 3.3977 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 | 0.0001 $\pm$ 0.0000 |
| Beam Search | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0255 $\pm$ 0.0000 |
| Tabu Search | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.1320 $\pm$ 0.0000 |
| Energy (Pure Greedy) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0283 $\pm$ 0.0000 |
| Energy (Pure SA) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.6953 $\pm$ 0.0000 |
| Energy (Hybrid) | -0.2690 $\pm$ 0.0000 | 1.7321 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.7284 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | -0.1856 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0406 $\pm$ 0.0000 |
| EBMAO (Pure SA) | -0.1856 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.5991 $\pm$ 0.0000 |
| EBMAO (Hybrid) | -0.1856 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.7643 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Random | 0.1667 $\pm$ 0.0000 | 0.1667 $\pm$ 0.0000 | 20.00 $\pm$ 0.00 | 4.00 $\pm$ 0.00 |
| Capability Matching (Greedy) | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| GreedyLB | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| RuleBased | 1.0000 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 24.00 $\pm$ 0.00 | 6.00 $\pm$ 0.00 |
| Beam Search | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Tabu Search | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure Greedy) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.3333 $\pm$ 0.0000 | 0.6667 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

**Identified Best Baseline**: *Beam Search* (Mean Energy: -0.2690)

| Energy Solver | Welch's t-test p-value | Mann-Whitney U p-value | Solver 95% Confidence Interval | Statistically Significant (p < 0.05)? |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.00e+00 | 1.00e+00 | [-0.2690, -0.2690] | **No** |
| Energy (Pure SA) | 1.00e+00 | 1.00e+00 | [-0.2690, -0.2690] | **No** |
| Energy (Hybrid) | 1.00e+00 | 1.00e+00 | [-0.2690, -0.2690] | **No** |

---

## Detailed Figure Catalog

The complete collection of scientific visualizations, charts, and detailed explanations is compiled in the Figure Catalog. 
Please proceed to the **[Figure Catalog](figure_catalog.md)** to inspect results visually.
