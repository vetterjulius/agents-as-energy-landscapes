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
| Energy (Pure Greedy) | 1.4133 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.2895 $\pm$ 0.0000 |
| Energy (Pure SA) | 2.7695 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0242 $\pm$ 0.0000 |
| Energy (Hybrid) | 1.4133 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 2.5896 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 1.4133 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0896 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 2.7695 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0178 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 1.4133 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 2.5593 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 0.3936 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.0576 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.3936 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.3936 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.0576 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.3936 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

No baselines evaluated for significance.

---

### Scenario: Interaction

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.3980 $\pm$ 0.0000 | 1.4142 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.2593 $\pm$ 0.0000 |
| Energy (Pure SA) | 2.7695 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0255 $\pm$ 0.0000 |
| Energy (Hybrid) | 1.3980 $\pm$ 0.0000 | 1.4142 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 2.6932 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 1.3980 $\pm$ 0.0000 | 1.4142 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.1791 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 2.7695 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0173 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 1.3980 $\pm$ 0.0000 | 1.4142 $\pm$ 0.0000 | 8.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 2.6200 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 0.3489 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.0576 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 10.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.3489 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.3489 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.0576 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 10.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.3489 $\pm$ 0.0000 | 0.8000 $\pm$ 0.0000 | 2.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

No baselines evaluated for significance.

---

### Scenario: Dynamic

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 1.5329 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 0.2538 $\pm$ 0.0000 |
| Energy (Pure SA) | 2.8946 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 11.00 $\pm$ 0.00 | 20.00 $\pm$ 0.00 | 0.0247 $\pm$ 0.0000 |
| Energy (Hybrid) | 1.5329 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 2.5989 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 1.5329 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 0.0913 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 2.8946 $\pm$ 0.0000 | 0.0000 $\pm$ 0.0000 | 11.00 $\pm$ 0.00 | 20.00 $\pm$ 0.00 | 0.0159 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 1.5329 $\pm$ 0.0000 | 0.7071 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 22.00 $\pm$ 0.00 | 2.5789 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.0576 $\pm$ 0.0000 | 0.1848 $\pm$ 0.0000 | 3.07 $\pm$ 0.00 | 20.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.0576 $\pm$ 0.0000 | 0.1848 $\pm$ 0.0000 | 3.07 $\pm$ 0.00 | 20.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.3833 $\pm$ 0.0000 | 0.1584 $\pm$ 0.0000 | 3.17 $\pm$ 0.00 | 22.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

No baselines evaluated for significance.

---

### Scenario: DistributionShift

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 2.8522 $\pm$ 0.0000 | 1.3973 $\pm$ 0.0000 | 103.00 $\pm$ 0.00 | 194.00 $\pm$ 0.00 | 6.0654 $\pm$ 0.0000 |
| Energy (Pure SA) | 4.1019 $\pm$ 0.0000 | 1.0465 $\pm$ 0.0000 | 87.00 $\pm$ 0.00 | 182.00 $\pm$ 0.00 | 0.0303 $\pm$ 0.0000 |
| Energy (Hybrid) | 1.1589 $\pm$ 0.0000 | 1.8387 $\pm$ 0.0000 | 116.00 $\pm$ 0.00 | 214.00 $\pm$ 0.00 | 19.6654 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 1.1589 $\pm$ 0.0000 | 1.8387 $\pm$ 0.0000 | 116.00 $\pm$ 0.00 | 214.00 $\pm$ 0.00 | 3.6077 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 4.1019 $\pm$ 0.0000 | 1.0465 $\pm$ 0.0000 | 87.00 $\pm$ 0.00 | 182.00 $\pm$ 0.00 | 0.0231 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 1.1589 $\pm$ 0.0000 | 1.8387 $\pm$ 0.0000 | 116.00 $\pm$ 0.00 | 214.00 $\pm$ 0.00 | 19.5339 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 0.1380 $\pm$ 0.0000 | 0.0785 $\pm$ 0.0000 | 918.34 $\pm$ 0.00 | 194.00 $\pm$ 0.00 |
| Energy (Pure SA) | -0.0737 $\pm$ 0.0000 | 0.0665 $\pm$ 0.0000 | 930.35 $\pm$ 0.00 | 182.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.5904 $\pm$ 0.0000 | 0.0963 $\pm$ 0.0000 | 900.59 $\pm$ 0.00 | 214.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.5904 $\pm$ 0.0000 | 0.0963 $\pm$ 0.0000 | 900.59 $\pm$ 0.00 | 214.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | -0.0737 $\pm$ 0.0000 | 0.0665 $\pm$ 0.0000 | 930.35 $\pm$ 0.00 | 182.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.5904 $\pm$ 0.0000 | 0.0963 $\pm$ 0.0000 | 900.59 $\pm$ 0.00 | 214.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

No baselines evaluated for significance.

---

### Scenario: Frustrated

#### Performance Summary (Mean $\pm$ Standard Deviation)

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts (Violations) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 0.1477 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0435 $\pm$ 0.0000 |
| Energy (Pure SA) | 0.1477 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0254 $\pm$ 0.0000 |
| Energy (Hybrid) | 0.1477 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.9806 $\pm$ 0.0000 |
| EBMAO (Pure Greedy) | 0.1477 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0324 $\pm$ 0.0000 |
| EBMAO (Pure SA) | 0.1477 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.0166 $\pm$ 0.0000 |
| EBMAO (Hybrid) | 0.1477 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 | 6.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | 0.9907 $\pm$ 0.0000 |

#### Emergent Behavior Analytics

| Orchestrator | Specialization Degree | Task Clustering | Communication Cost | Conflict Rate |
| :--- | :---: | :---: | :---: | :---: |
| Energy (Pure Greedy) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Pure SA) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| Energy (Hybrid) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure Greedy) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Pure SA) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |
| EBMAO (Hybrid) | 0.5000 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | 12.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 |

#### Statistical Significance vs. Best Baseline

We compare the primary Energy solvers (Pure SA, Pure Greedy, Hybrid) against the best baseline (lowest mean energy among non-Energy methods).

No baselines evaluated for significance.

---

## Detailed Figure Catalog

The complete collection of scientific visualizations, charts, and detailed explanations is compiled in the Figure Catalog. 
Please proceed to the **[Figure Catalog](figure_catalog.md)** to inspect results visually.
