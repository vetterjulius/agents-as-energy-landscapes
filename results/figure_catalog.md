# MAS Energy Landscape Benchmark: Visualizations & Figure Catalog

This document compiles and structures all figures generated during the benchmark evaluation. It serves as a visual companion to the primary report, providing scientific interpretation for each plot.

## Table of Contents
1. [Energy and Optimization Landscapes](#1-energy-and-optimization-landscapes)
2. [System Scaling Characteristics](#2-system-scaling-characteristics)
3. [Interaction Strength Sweeps](#3-interaction-strength-sweeps)
4. [Conflict and Constraint Violations](#4-conflict-and-constraint-violations)
5. [Pareto-Plot (Runtime vs. Energy)](#5-pareto-plot-runtime-vs-energy)
6. [Emergent Networks & Heatmaps](#6-emergent-networks--heatmaps)

## 1. Energy and Optimization Landscapes

These plots show the energy profiles across different scenarios. Minimizing energy translates directly to optimal multi-agent orchestration.

### Scenario: Independent
![Total Energy - Independent](plots/energy_Independent.png)

**Interpretation**:
- **What this shows**: The total optimized energy achieved by each orchestrator. Lower is better.
- **Analysis**: Energy-based orchestrators (particularly Hybrid) consistently find assignments with significantly lower total energy. This highlights the effectiveness of treating MAS orchestration as an energy minimization problem over a continuous/discrete landscape.

#### Energy Component Breakdown - Independent
![Energy Breakdown - Independent](plots/breakdown_Independent.png)

**Interpretation**:
- **What this shows**: Stacked energy components (Assignment, Interaction, Cost, Risk) making up the total energy.
- **Analysis**: This verifies that the energy model successfully balances trade-offs. For instance, capability alignment might be trade-off against co-assignment costs or high communication risk, which is visible in how different models distribute energy components.

### Scenario: Interaction
![Total Energy - Interaction](plots/energy_Interaction.png)

**Interpretation**:
- **What this shows**: The total optimized energy achieved by each orchestrator. Lower is better.
- **Analysis**: Energy-based orchestrators (particularly Hybrid) consistently find assignments with significantly lower total energy. This highlights the effectiveness of treating MAS orchestration as an energy minimization problem over a continuous/discrete landscape.

#### Energy Component Breakdown - Interaction
![Energy Breakdown - Interaction](plots/breakdown_Interaction.png)

**Interpretation**:
- **What this shows**: Stacked energy components (Assignment, Interaction, Cost, Risk) making up the total energy.
- **Analysis**: This verifies that the energy model successfully balances trade-offs. For instance, capability alignment might be trade-off against co-assignment costs or high communication risk, which is visible in how different models distribute energy components.

### Scenario: Dynamic
![Total Energy - Dynamic](plots/energy_Dynamic.png)

**Interpretation**:
- **What this shows**: The total optimized energy achieved by each orchestrator. Lower is better.
- **Analysis**: Energy-based orchestrators (particularly Hybrid) consistently find assignments with significantly lower total energy. This highlights the effectiveness of treating MAS orchestration as an energy minimization problem over a continuous/discrete landscape.

#### Energy Component Breakdown - Dynamic
![Energy Breakdown - Dynamic](plots/breakdown_Dynamic.png)

**Interpretation**:
- **What this shows**: Stacked energy components (Assignment, Interaction, Cost, Risk) making up the total energy.
- **Analysis**: This verifies that the energy model successfully balances trade-offs. For instance, capability alignment might be trade-off against co-assignment costs or high communication risk, which is visible in how different models distribute energy components.

### Scenario: DistributionShift
![Total Energy - DistributionShift](plots/energy_DistributionShift.png)

**Interpretation**:
- **What this shows**: The total optimized energy achieved by each orchestrator. Lower is better.
- **Analysis**: Energy-based orchestrators (particularly Hybrid) consistently find assignments with significantly lower total energy. This highlights the effectiveness of treating MAS orchestration as an energy minimization problem over a continuous/discrete landscape.

#### Energy Component Breakdown - DistributionShift
![Energy Breakdown - DistributionShift](plots/breakdown_DistributionShift.png)

**Interpretation**:
- **What this shows**: Stacked energy components (Assignment, Interaction, Cost, Risk) making up the total energy.
- **Analysis**: This verifies that the energy model successfully balances trade-offs. For instance, capability alignment might be trade-off against co-assignment costs or high communication risk, which is visible in how different models distribute energy components.

### Scenario: Frustrated
![Total Energy - Frustrated](plots/energy_Frustrated.png)

**Interpretation**:
- **What this shows**: The total optimized energy achieved by each orchestrator. Lower is better.
- **Analysis**: Energy-based orchestrators (particularly Hybrid) consistently find assignments with significantly lower total energy. This highlights the effectiveness of treating MAS orchestration as an energy minimization problem over a continuous/discrete landscape.

#### Energy Component Breakdown - Frustrated
![Energy Breakdown - Frustrated](plots/breakdown_Frustrated.png)

**Interpretation**:
- **What this shows**: Stacked energy components (Assignment, Interaction, Cost, Risk) making up the total energy.
- **Analysis**: This verifies that the energy model successfully balances trade-offs. For instance, capability alignment might be trade-off against co-assignment costs or high communication risk, which is visible in how different models distribute energy components.

## 2. System Scaling Characteristics

![Scaling Energy](plots/scaling_energy.png)
![Scaling Runtime](plots/scaling_runtime.png)

**Interpretation**:
- **Energy vs Problem Size**: Shows how final solution energy scales as we increase the number of tasks from 20 to 500. Lower is better.
- **Runtime vs Problem Size**: Measures computational overhead as task size scales. Hybrid search strikes an ideal balance, optimizing energy to near-minimum while running significantly faster than naive solvers.

## 3. Interaction Strength Sweeps

![Coupling Energy](plots/coupling_energy.png)
![Coupling Coordination](plots/coupling_coordination.png)

**Interpretation**:
- **Energy Sweep**: Analyzes sensitivity of final energy to the interaction coupling weight $\lambda_{int}$.
- **Coordination Sweep**: Realized synergies (coordination scores) as coupling weight scales. Heuristics are flat because they ignore task interactions, whereas energy models dynamically pool tasks on the same agent as synergies become more valuable.

## 4. Conflict and Constraint Violations

![Conflict Violations](plots/conflicts_comparison.png)
![Coupling Conflicts](plots/coupling_conflicts.png)

**Interpretation**:
- **Conflicts cross scenarios**: Shows rule-based vs. energy solvers' constraint violation rates. Rule-based models incur high conflicts because they lack foresight, whereas the energy framework pushes assignment states out of high-cost boundaries.

## 5. Pareto-Plot (Runtime vs. Energy)

![Pareto Runtime vs Energy](plots/pareto_runtime_energy.png)

**Interpretation**:
- **What this shows**: Multi-objective visualization plotting mean Runtime (Y-axis) against mean Energy (X-axis). The lower-left corner represents the optimal Pareto frontier (fastest runtime and lowest energy).
- **Analysis**: The Energy (Hybrid) model generally dominates, achieving near-optimal SA energy while staying close to Greedy runtimes.

## 6. Emergent Networks & Heatmaps

### Assignment Heatmap
![Assignment Heatmap](plots/assignment_heatmap.png)

**Interpretation**:
- **What this shows**: Density grid representing how tasks (columns) are distributed among agents (rows). Highlighted cells indicate assigned pairs.
- **Analysis**: Reveals load balancing or specialization patterns (e.g. specialists clustering on specific columns, generalists spreading evenly).

### Task-Dependency Graph
![Task Dependency Graph](plots/task_dependency_graph.png)

**Interpretation**:
- **What this shows**: Synergy/dependency network graph where edges denote positive interactions ($\Theta_{i,j}$) between tasks.
- **Analysis**: Visualizes problem complexity. Cliques or dense clusters indicate task groups that must be co-assigned to reduce interaction energy.

### Agent-Task Bipartite Graph
![Agent-Task Bipartite Graph](plots/agent_task_bipartite.png)

**Interpretation**:
- **What this shows**: Bipartite graph mapping agents directly to their assigned tasks, with edge colors representing capability similarity.
- **Analysis**: Provides a direct, intuitive visual representation of the final orchestration layout.
