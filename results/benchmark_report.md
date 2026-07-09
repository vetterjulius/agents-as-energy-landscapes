# Energy-Based Orchestration Benchmark (EOB) Report

## Scenario: Independent

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 2.2700 | 1.8708 | 0.00 | 0.00 | 0.0003 | AssignmentEnergy: 2.1405, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1295 |
| Capability Matching (Greedy) | 1.4160 | 1.2247 | 0.00 | 0.00 | 0.0003 | AssignmentEnergy: 1.2751, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1409 |
| GreedyLB | 1.4164 | 0.7071 | 0.00 | 0.00 | 0.0010 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1470 |
| RuleBased | 3.3115 | 0.0000 | 0.00 | 0.00 | 0.0001 | AssignmentEnergy: 3.1640, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1475 |
| Energy (Pure Greedy) | 1.4137 | 0.7071 | 0.00 | 0.00 | 0.0897 | AssignmentEnergy: 1.2735, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1402 |
| Energy (Pure SA) | 1.4137 | 0.7071 | 0.00 | 0.00 | 3.4860 | AssignmentEnergy: 1.2735, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1402 |
| Energy (Hybrid) | 1.4137 | 0.7071 | 0.00 | 0.00 | 5.0321 | AssignmentEnergy: 1.2735, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1402 |

## Scenario: Interaction

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 2.2866 | 1.5811 | 2.00 | 0.00 | 0.0002 | AssignmentEnergy: 2.1967, InteractionEnergy: -0.0400, CostEnergy: 0.0000, RiskEnergy: 0.1298 |
| Capability Matching (Greedy) | 1.4760 | 1.2247 | 0.00 | 2.00 | 0.0007 | AssignmentEnergy: 1.2751, InteractionEnergy: -0.0000, CostEnergy: 0.0600, RiskEnergy: 0.1409 |
| GreedyLB | 1.4764 | 0.7071 | 0.00 | 2.00 | 0.0020 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0000, CostEnergy: 0.0600, RiskEnergy: 0.1470 |
| RuleBased | 3.2715 | 0.0000 | 2.00 | 0.00 | 0.0003 | AssignmentEnergy: 3.1640, InteractionEnergy: -0.0400, CostEnergy: 0.0000, RiskEnergy: 0.1475 |
| Energy (Pure Greedy) | 1.3184 | 1.0000 | 8.00 | 0.00 | 0.3703 | AssignmentEnergy: 1.3339, InteractionEnergy: -0.1600, CostEnergy: 0.0000, RiskEnergy: 0.1445 |
| Energy (Pure SA) | 1.3184 | 1.0000 | 8.00 | 0.00 | 5.1362 | AssignmentEnergy: 1.3339, InteractionEnergy: -0.1600, CostEnergy: 0.0000, RiskEnergy: 0.1445 |
| Energy (Hybrid) | 1.3184 | 1.0000 | 8.00 | 0.00 | 4.1526 | AssignmentEnergy: 1.3339, InteractionEnergy: -0.1600, CostEnergy: 0.0000, RiskEnergy: 0.1445 |

## Scenario: Dynamic

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 3.7514 | 1.5811 | 13.00 | 30.00 | 0.0001 | AssignmentEnergy: 3.4136, InteractionEnergy: 0.0042, CostEnergy: 0.1495, RiskEnergy: 0.1841 |
| Capability Matching (Greedy) | 1.5592 | 1.2247 | 14.00 | 26.00 | 0.0003 | AssignmentEnergy: 1.2751, InteractionEnergy: -0.0029, CostEnergy: 0.1348, RiskEnergy: 0.1522 |
| GreedyLB | 1.5336 | 0.7071 | 12.00 | 22.00 | 0.0009 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0008, CostEnergy: 0.1154, RiskEnergy: 0.1496 |
| RuleBased | 3.4327 | 0.0000 | 9.00 | 20.00 | 0.0001 | AssignmentEnergy: 3.1640, InteractionEnergy: 0.0018, CostEnergy: 0.0949, RiskEnergy: 0.1720 |
| Energy (Pure Greedy) | 1.5336 | 0.7071 | 12.00 | 22.00 | 0.0660 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0008, CostEnergy: 0.1154, RiskEnergy: 0.1496 |
| Energy (Pure SA) | 1.5336 | 0.7071 | 12.00 | 22.00 | 3.5295 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0008, CostEnergy: 0.1154, RiskEnergy: 0.1496 |
| Energy (Hybrid) | 1.5336 | 0.7071 | 12.00 | 22.00 | 4.0844 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0008, CostEnergy: 0.1154, RiskEnergy: 0.1496 |

## Scenario: DistributionShift

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 4.4682 | 1.6762 | 114.00 | 206.00 | 0.0005 | AssignmentEnergy: 4.3132, InteractionEnergy: -0.0355, CostEnergy: 0.1423, RiskEnergy: 0.0481 |
| Capability Matching (Greedy) | 1.2097 | 2.6367 | 140.00 | 264.00 | 0.0012 | AssignmentEnergy: 0.9974, InteractionEnergy: -0.0306, CostEnergy: 0.1823, RiskEnergy: 0.0606 |
| GreedyLB | 1.2456 | 1.3973 | 98.00 | 194.00 | 0.0035 | AssignmentEnergy: 1.0605, InteractionEnergy: -0.0118, CostEnergy: 0.1378, RiskEnergy: 0.0592 |
| RuleBased | 4.0273 | 0.4880 | 87.00 | 170.00 | 0.0004 | AssignmentEnergy: 3.8622, InteractionEnergy: -0.0054, CostEnergy: 0.1169, RiskEnergy: 0.0536 |
| Energy (Pure Greedy) | 1.1646 | 1.6762 | 113.00 | 206.00 | 4.6488 | AssignmentEnergy: 1.0025, InteractionEnergy: -0.0407, CostEnergy: 0.1445, RiskEnergy: 0.0583 |
| Energy (Pure SA) | 1.2273 | 1.5430 | 113.00 | 200.00 | 13.1116 | AssignmentEnergy: 1.0527, InteractionEnergy: -0.0347, CostEnergy: 0.1476, RiskEnergy: 0.0618 |
| Energy (Hybrid) | 1.1652 | 1.8387 | 118.00 | 214.00 | 34.4481 | AssignmentEnergy: 0.9987, InteractionEnergy: -0.0417, CostEnergy: 0.1498, RiskEnergy: 0.0584 |

## Scenario: Frustrated

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 2.7588 | 1.0000 | 2.00 | 4.00 | 0.0001 | AssignmentEnergy: 0.5278, InteractionEnergy: -0.2222, CostEnergy: 2.2222, RiskEnergy: 0.2310 |
| Capability Matching (Greedy) | 3.3977 | 0.0000 | 0.00 | 6.00 | 0.0002 | AssignmentEnergy: -0.1667, InteractionEnergy: -0.0000, CostEnergy: 3.3333, RiskEnergy: 0.2310 |
| GreedyLB | 3.3977 | 0.0000 | 0.00 | 6.00 | 0.0006 | AssignmentEnergy: -0.1667, InteractionEnergy: -0.0000, CostEnergy: 3.3333, RiskEnergy: 0.2310 |
| RuleBased | 3.3977 | 0.0000 | 0.00 | 6.00 | 0.0001 | AssignmentEnergy: -0.1667, InteractionEnergy: -0.0000, CostEnergy: 3.3333, RiskEnergy: 0.2310 |
| Energy (Pure Greedy) | -0.2690 | 1.7321 | 8.00 | 0.00 | 0.0222 | AssignmentEnergy: 0.3889, InteractionEnergy: -0.8889, CostEnergy: 0.0000, RiskEnergy: 0.2310 |
| Energy (Pure SA) | -0.2690 | 1.7321 | 8.00 | 0.00 | 1.5298 | AssignmentEnergy: 0.3889, InteractionEnergy: -0.8889, CostEnergy: 0.0000, RiskEnergy: 0.2310 |
| Energy (Hybrid) | -0.2690 | 1.7321 | 8.00 | 0.00 | 1.6633 | AssignmentEnergy: 0.3889, InteractionEnergy: -0.8889, CostEnergy: 0.0000, RiskEnergy: 0.2310 |
