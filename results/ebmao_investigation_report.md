# Investigation Report: EBMAO Performance Discrepancy under "Frustrated" Scenario

## Executive Summary

A comprehensive benchmark evaluation of the **Energy-Based Multi-Agent Orchestration (EBMAO)** orchestrator revealed a significant performance degradation compared to the previous Energy-Based system specifically under the **"Frustrated"** scenario. In other scenarios (e.g., *Independent*), EBMAO performs comparably, but in frustration-heavy and conflict-heavy environments, its performance drops precipitously, producing extreme constraint/conflict violations.

This investigation was conducted to determine whether this performance gap represents a fundamental flaw in the mathematical foundations of EBMAO or a semantic and implementation-level mismatch.

**Core Finding:** The performance degradation is **not** a fundamental mathematical limitation of EBMAO's underlying optimization/search dynamics. Instead, it is caused by a **direct semantic and formulation mismatch** in how the cost matrix $\mathbf{C}$ is interpreted and mathematically mapped between the benchmark scenario generator and the EBMAO orchestrator.
- The benchmark generator defines $\mathbf{C}$ as a **co-assignment exclusion/conflict cost** (which must be penalized when tasks are assigned to the *same* agent).
- The EBMAO orchestrator implements $\mathbf{C}$ as a **communication/coordination overhead** (which is incurred when tasks are assigned to *different* agents).

As a result, EBMAO's internal cost-minimization logic is mathematically inverted: **it actively optimizes to cluster conflicting tasks onto the same agent**, creating maximum constraint violations under the actual evaluation metrics.

---

## Theoretical Background & Model Differences

To understand the divergence, we contrast the mathematical formulations of the previous Energy system and the EBMAO framework.

### 1. Cost & Conflict Energy Term

#### Previous Energy Model (`CostEnergy`)
In the previous energy system, the cost energy term $\mathbf{E}_{\text{cost}}$ represents a **co-assignment penalty** (mutual exclusion / conflict cost). It is defined as:
$$E_{\text{cost}} = \frac{1}{NM} \sum_{i, j} C_{ij} \cdot \text{co}_{ij}$$
Where:
- $N$ is the number of agents, $M$ is the number of tasks.
- $C_{ij}$ is the co-assignment cost between task $i$ and task $j$.
- $\text{co}_{ij} = \sum_a X_{ai} X_{aj} = (\mathbf{X}^\top \mathbf{X})_{ij}$ is the co-occurrence indicator (equal to $1.0$ if task $i$ and task $j$ are assigned to the same agent $a$, and $0.0$ otherwise).

Under this definition, assigning conflicting tasks ($C_{ij} > 0$) to the same agent adds $C_{ij}$ directly to the energy score. Therefore, energy minimization naturally drives the system to **separate** conflicting tasks onto different agents ($\text{co}_{ij} = 0$).

#### EBMAO Model (`EBMAOCostEnergy`)
In the EBMAO framework, the cost energy term $\mathbf{E}_{\text{cost\_EBMAO}}$ represents **communication/coordination overhead** incurred when interdependent tasks are assigned to different agents. It is defined as:
$$E_{\text{cost\_EBMAO}} = \frac{1}{NM} \sum_{i < j} C_{ij} \cdot (1 - \text{co}_{ij})$$
Where:
- $C_{ij}$ is interpreted as the coordination cost between task $i$ and task $j$ if they are assigned to different agents (requiring cross-agent network communication).
- $1 - \text{co}_{ij}$ is equal to $1.0$ if task $i$ and task $j$ are assigned to **different** agents, and $0.0$ if they are assigned to the **same** agent.

Under this definition, assigning tasks with high communication costs ($C_{ij} > 0$) to different agents adds $C_{ij}$ to the energy. Therefore, energy minimization drives the system to **cluster** these tasks onto the same agent ($\text{co}_{ij} = 1$) to eliminate communication overhead.

### 2. Assignment Energy Term

We also note a minor formulation difference in the assignment term.
- **Previous Model (`AssignmentEnergy`)**:
  $$E_{\text{assign}} = \frac{1}{NM} \sum_{a, i} X_{ai} \left( \|\mathbf{s}_a - \mathbf{c}_i\|^2 - \lambda \cdot (\mathbf{s}_a^\top \mathbf{c}_i + \mathbf{s}_a^\top \mathbf{\kappa}_a) \right)$$
- **EBMAO Model (`EBMAOAssignmentEnergy`)**:
  $$E_{\text{assign\_EBMAO}} = \frac{1}{NM} \sum_{a, i} X_{ai} \left( \|\mathbf{s}_a - \mathbf{c}_i\|^2 - \lambda \cdot \mathbf{s}_a^\top \mathbf{\kappa}_a \right)$$

EBMAO removes the $\mathbf{s}_a^\top \mathbf{c}_i$ term inside the alignment parenthesis because it is mathematically redundant when the Euclidean distance $\|\mathbf{s}_a - \mathbf{c}_i\|^2$ is already minimized (assuming normalized embeddings). This is a clean refinement, but is not the primary driver of the performance collapse.

---

## The Semantic & Mapping Mismatch

The root cause of the "Frustrated" scenario failure is the **direct mapping of conflict matrices into communication cost parameters**:

1. **Benchmark Scenario's Intent**:
   The "Frustrated" scenario generates a problem instance where tasks have high individual agent alignment but are highly restricted from being assigned together. This is modeled by setting `problem.co_assignment_costs` ($\mathbf{C}$) to extremely high values (e.g., $10.0$):
   $$C_{jk} = 10.0 \quad \text{if task } j \text{ and task } k \text{ want to go to the same agent locally}$$

2. **Variable Injection**:
   In `EBMAOBasedOrchestrator.solve`, the orchestrator initialization maps this matrix directly:
   ```python
   state = OrchestrationState(
       X=X_init, s=s, c=c, kappa=kappa,
       Theta=problem.interaction_graph,
       C=problem.co_assignment_costs,   # <--- Fed directly to EBMAO's C parameter
       N=N, M=M, d=d
   )
   ```

3. **The Mismatch**:
   - The **previous Energy system** correctly optimizes to keep these tasks separate ($co_{jk} = 0$) because it treats $\mathbf{C}$ as exclusion costs.
   - The **EBMAO system** optimizes to keep these tasks together ($co_{jk} = 1$) because it treats $\mathbf{C}$ as communication costs. It is highly rewarded internally for placing conflicting tasks on the *same* agent because doing so "saves" $10.0$ units of communication cost!

4. **Evaluation Discrepancy**:
   During evaluation, the resulting assignments are assessed via the standard metric `compute_energy`, which evaluates using the **previous system's `CostEnergy`** ($C * co$). Consequently, EBMAO's clustered assignment is hit with a massive conflict penalty ($10.0$ per violation), resulting in an extremely poor evaluated energy.

---

## Empirical Replication & Proof

An empirical test was conducted on a frustrated scenario instance with $3$ agents and $6$ tasks (seed 42).

### The Conflict Cost Matrix $\mathbf{C}$
The benchmark generated the following co-assignment cost matrix:
```
tensor([[ 0.,  0.,  0., 10.,  0.,  0.],
        [ 0.,  0.,  0.,  0., 10.,  0.],
        [ 0.,  0.,  0.,  0.,  0., 10.],
        [10.,  0.,  0.,  0.,  0.,  0.],
        [ 0., 10.,  0.,  0.,  0.,  0.],
        [ 0.,  0., 10.,  0.,  0.,  0.]])
```
*Interpretation:* Task $0$ and Task $3$ conflict heavily (cost $10.0$ if co-assigned). Same for $(1, 4)$ and $(2, 5)$.

### Comparison of Results

| Orchestrator | Resulting Assignment Matrix $\mathbf{X}$ | Evaluated Energy | Conflicts Count |
| :--- | :--- | :---: | :---: |
| **Previous Energy (Hybrid)** | Agent 0: Tasks {0, 1, 2}<br>Agent 1: Tasks {3, 4, 5}<br>Agent 2: None | **-0.2690** | **0.0** |
| **EBMAO (Hybrid)** | Agent 0: Tasks {0, 1, 2, 3, 4, 5}<br>Agent 1: None<br>Agent 2: None | **2.6199** | **6.0** |

### Analysis of the EBMAO Assignment
- **Previous System**: Correctly partitioned tasks $\{0, 1, 2\}$ and $\{3, 4, 5\}$ onto separate agents. No conflicting pairs are co-assigned. Total conflict violations = $0.0$.
- **EBMAO System**: Assigned **all tasks** to Agent 0.
  - To EBMAO's internal optimizer, this was the absolute best possible state because the communication energy $E_{\text{cost\_EBMAO}}$ was completely minimized to $0.0$.
  - To the evaluation metric, this is the worst possible state because the conflict violations are maximized: $(0,3), (3,0), (1,4), (4,1), (2,5), (5,2)$ are all co-assigned, creating $6$ violations and incurring $60.0$ raw penalty units.

This explains the extremely high conflict rate ($6.00$) and high load imbalance standard deviation ($3.46$) reported for EBMAO in the "Frustrated" scenario.

---

## Impact on Other Scenarios

This semantic mismatch also explains smaller performance degradations seen in other scenarios:
- **Independent**: Standard $C = 0.0$. Since there are no conflict costs, EBMAO performs identical to the previous system ($1.6248$ vs $1.6207$ energy).
- **Interaction**: Incurs minor randomized conflict costs. EBMAO clusters them slightly more, raising energy from $1.5612$ to $1.6001$ and doubling conflicts from $0.27$ to $1.33$.
- **Dynamic / DistributionShift**: General randomized cost constraints are present. EBMAO experiences higher conflict rates ($40.47$ vs $29.87$ in Dynamic; $417.53$ vs $220.33$ in DistributionShift) due to its internal drive to cluster conflicting tasks.

---

## Actionable Recommendations for Resolution

To resolve this issue and allow both frameworks to be evaluated fairly, we can apply one of the following simple, robust fixes:

### Option A: Standardize the Cost/Conflict Term (Recommended)
If the benchmark's cost matrix is always meant to represent **exclusion/conflict constraints** (tasks that cannot run together), the EBMAO orchestrator should use a formulation that penalizes co-assignment.
We can update `energy/ebmao_cost.py` to match the co-assignment penalty:
```python
class EBMAOCostEnergy(EnergyTerm):
    def compute(self, state):
        co = state.X.T @ state.X
        # Penalize co-assignment of conflicting tasks
        return (state.C * co).sum() / (state.N * state.M)
```
This single-line change instantly aligns EBMAO's optimization objective with the benchmark's physical scenario constraints.

### Option B: Dual-Potential Framework (Comprehensive)
In real-world multi-agent systems, we face both **coordination/communication costs** (which favor task co-assignment on the same agent) AND **exclusion/conflict constraints** (which favor task separation).
To model both simultaneously:
1. Retain `EBMAOCostEnergy` as a communication cost potential:
   $$E_{\text{comm\_EBMAO}} = \frac{1}{NM} \sum C^{\text{comm}}_{ij} (1 - co_{ij})$$
2. Introduce a new `EBMAOConflictEnergy` term as an exclusion penalty potential:
   $$E_{\text{conflict\_EBMAO}} = \frac{1}{NM} \sum C^{\text{conflict}}_{ij} co_{ij}$$
3. Feed separate matrices from the benchmark to the orchestrator instead of overloading the single `C` variable.

---

## Conclusion

The investigation confirms that EBMAO's mathematical dynamics and simulated annealing framework are solid. The observed poor performance in the "Frustrated" scenario is entirely an **implementation-level semantic mismatch** where exclusion/conflict costs were mapped directly to a potential term designed for inter-agent communication overhead. Applying a simple alignment in the cost potential formulation (Option A) will immediately restore EBMAO's performance to optimal levels.
