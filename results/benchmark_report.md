# Energy-Based Orchestration Benchmark (EOB) Report

## Scenario: Independent

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 2.2700 | 1.8708 | 0.00 | 0.00 | 0.0003 |
| Greedy | 1.4160 | 1.2247 | 0.00 | 0.00 | 0.0004 |
| GreedyLB | 1.4164 | 0.7071 | 0.00 | 0.00 | 0.0009 |
| RuleBased | 3.3115 | 0.0000 | 0.00 | 0.00 | 0.0001 |
| EnergyBased (SA) | 1.4241 | 1.0000 | 0.00 | 0.00 | 16.9830 |

## Scenario: Interaction

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 2.2866 | 1.5811 | 2.00 | 0.00 | 0.0001 |
| Greedy | 1.4760 | 1.2247 | 0.00 | 2.00 | 0.0003 |
| GreedyLB | 1.4764 | 0.7071 | 0.00 | 2.00 | 0.0008 |
| RuleBased | 3.2715 | 0.0000 | 2.00 | 0.00 | 0.0001 |
| EnergyBased (SA) | 1.4102 | 1.0000 | 2.00 | 0.00 | 17.0957 |

## Scenario: Dynamic

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 3.7514 | 1.5811 | 13.00 | 30.00 | 0.0001 |
| Greedy | 1.5592 | 1.2247 | 14.00 | 26.00 | 0.0004 |
| GreedyLB | 1.5336 | 0.7071 | 12.00 | 22.00 | 0.0009 |
| RuleBased | 3.4327 | 0.0000 | 9.00 | 20.00 | 0.0001 |
| EnergyBased (SA) | 1.5524 | 0.7071 | 10.00 | 22.00 | 16.9872 |

## Scenario: DistributionShift

| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random | 4.4682 | 1.6762 | 114.00 | 206.00 | 0.0005 |
| Greedy | 1.2097 | 2.6367 | 140.00 | 264.00 | 0.0013 |
| GreedyLB | 1.2456 | 1.3973 | 98.00 | 194.00 | 0.0037 |
| RuleBased | 4.0273 | 0.4880 | 87.00 | 170.00 | 0.0004 |
| EnergyBased (SA) | 1.1995 | 1.8387 | 108.00 | 214.00 | 113.8237 |
