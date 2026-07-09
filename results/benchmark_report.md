# Energy-Based Orchestration Benchmark (EOB) Report

## Scenario: Independent

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 2.2700 | 1.8708 | 0.00 | 0.00 | 0.0003 | AssignmentEnergy: 2.1405, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1295 |
| Greedy | 1.4160 | 1.2247 | 0.00 | 0.00 | 0.0004 | AssignmentEnergy: 1.2751, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1409 |
| GreedyLB | 1.4164 | 0.7071 | 0.00 | 0.00 | 0.0010 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1470 |
| RuleBased | 3.3115 | 0.0000 | 0.00 | 0.00 | 0.0001 | AssignmentEnergy: 3.1640, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1475 |
| EnergyBased (SA) | 1.4241 | 1.0000 | 0.00 | 0.00 | 19.8659 | AssignmentEnergy: 1.2783, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.1459 |

## Scenario: Interaction

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 2.2866 | 1.5811 | 2.00 | 0.00 | 0.0001 | AssignmentEnergy: 2.1967, InteractionEnergy: -0.0400, CostEnergy: 0.0000, RiskEnergy: 0.1298 |
| Greedy | 1.4760 | 1.2247 | 0.00 | 2.00 | 0.0003 | AssignmentEnergy: 1.2751, InteractionEnergy: -0.0000, CostEnergy: 0.0600, RiskEnergy: 0.1409 |
| GreedyLB | 1.4764 | 0.7071 | 0.00 | 2.00 | 0.0010 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0000, CostEnergy: 0.0600, RiskEnergy: 0.1470 |
| RuleBased | 3.2715 | 0.0000 | 2.00 | 0.00 | 0.0001 | AssignmentEnergy: 3.1640, InteractionEnergy: -0.0400, CostEnergy: 0.0000, RiskEnergy: 0.1475 |
| EnergyBased (SA) | 1.4102 | 1.0000 | 2.00 | 0.00 | 17.0753 | AssignmentEnergy: 1.2927, InteractionEnergy: -0.0400, CostEnergy: 0.0000, RiskEnergy: 0.1575 |

## Scenario: Dynamic

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 3.7514 | 1.5811 | 13.00 | 30.00 | 0.0001 | AssignmentEnergy: 3.4136, InteractionEnergy: 0.0042, CostEnergy: 0.1495, RiskEnergy: 0.1841 |
| Greedy | 1.5592 | 1.2247 | 14.00 | 26.00 | 0.0004 | AssignmentEnergy: 1.2751, InteractionEnergy: -0.0029, CostEnergy: 0.1348, RiskEnergy: 0.1522 |
| GreedyLB | 1.5336 | 0.7071 | 12.00 | 22.00 | 0.0008 | AssignmentEnergy: 1.2694, InteractionEnergy: -0.0008, CostEnergy: 0.1154, RiskEnergy: 0.1496 |
| RuleBased | 3.4327 | 0.0000 | 9.00 | 20.00 | 0.0002 | AssignmentEnergy: 3.1640, InteractionEnergy: 0.0018, CostEnergy: 0.0949, RiskEnergy: 0.1720 |
| EnergyBased (SA) | 1.5524 | 0.7071 | 10.00 | 22.00 | 17.6309 | AssignmentEnergy: 1.2735, InteractionEnergy: 0.0062, CostEnergy: 0.1129, RiskEnergy: 0.1598 |

## Scenario: DistributionShift

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 4.4682 | 1.6762 | 114.00 | 206.00 | 0.0005 | AssignmentEnergy: 4.3132, InteractionEnergy: -0.0355, CostEnergy: 0.1423, RiskEnergy: 0.0481 |
| Greedy | 1.2097 | 2.6367 | 140.00 | 264.00 | 0.0014 | AssignmentEnergy: 0.9974, InteractionEnergy: -0.0306, CostEnergy: 0.1823, RiskEnergy: 0.0606 |
| GreedyLB | 1.2456 | 1.3973 | 98.00 | 194.00 | 0.0037 | AssignmentEnergy: 1.0605, InteractionEnergy: -0.0118, CostEnergy: 0.1378, RiskEnergy: 0.0592 |
| RuleBased | 4.0273 | 0.4880 | 87.00 | 170.00 | 0.0005 | AssignmentEnergy: 3.8622, InteractionEnergy: -0.0054, CostEnergy: 0.1169, RiskEnergy: 0.0536 |
| EnergyBased (SA) | 1.1995 | 1.8387 | 108.00 | 214.00 | 117.9301 | AssignmentEnergy: 1.0037, InteractionEnergy: -0.0056, CostEnergy: 0.1472, RiskEnergy: 0.0543 |

## Scenario: Frustrated

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random | 2.7588 | 1.0000 | 2.00 | 4.00 | 0.0002 | AssignmentEnergy: 0.5278, InteractionEnergy: -0.2222, CostEnergy: 2.2222, RiskEnergy: 0.2310 |
| Greedy | 3.3977 | 0.0000 | 0.00 | 6.00 | 0.0002 | AssignmentEnergy: -0.1667, InteractionEnergy: -0.0000, CostEnergy: 3.3333, RiskEnergy: 0.2310 |
| GreedyLB | 3.3977 | 0.0000 | 0.00 | 6.00 | 0.0006 | AssignmentEnergy: -0.1667, InteractionEnergy: -0.0000, CostEnergy: 3.3333, RiskEnergy: 0.2310 |
| RuleBased | 3.3977 | 0.0000 | 0.00 | 6.00 | 0.0001 | AssignmentEnergy: -0.1667, InteractionEnergy: -0.0000, CostEnergy: 3.3333, RiskEnergy: 0.2310 |
| EnergyBased (SA) | 0.4810 | 1.0000 | 0.00 | 0.00 | 7.4545 | AssignmentEnergy: 0.2500, InteractionEnergy: -0.0000, CostEnergy: 0.0000, RiskEnergy: 0.2310 |
