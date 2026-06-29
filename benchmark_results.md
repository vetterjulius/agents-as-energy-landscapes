# Benchmark Evaluation Results

This report evaluates the performance of the Orchestrator system against the baselines.

## Scenario: Small (5x15)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.3400 | 4.1194 | 5.08% | 0.4587 | 2.45 | 20802.5 |
| Greedy (Local Improvement) | 7.5821 | 4.1202 | 45.66% | 0.4585 | 2.45 | 3444.6 |
| Greedy (Local Search) | 7.5821 | 4.1196 | 45.67% | 0.4587 | 2.45 | 3944.9 |
| Greedy (Construction) | 7.5821 | 4.1196 | 45.67% | 0.4587 | 2.45 | 3450.2 |
| Random Baseline | 7.5821 | 6.4107 | 15.45% | 0.5311 | 1.87 | 120.2 |


## Scenario: Medium (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5232 | 4.3855 | 3.04% | 0.4177 | 1.70 | 52250.5 |
| Greedy (Local Improvement) | 6.4638 | 4.3868 | 32.13% | 0.4173 | 1.70 | 19896.1 |
| Greedy (Local Search) | 6.4638 | 4.3860 | 32.14% | 0.4262 | 1.83 | 30522.2 |
| Greedy (Construction) | 6.4638 | 4.3859 | 32.15% | 0.4177 | 1.70 | 19766.7 |
| Random Baseline | 6.4638 | 6.6675 | -3.15% | 0.3995 | 1.70 | 239.9 |


## Scenario: Large (20x60)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5776 | 4.4325 | 3.17% | 0.5450 | 2.32 | 158800.4 |
| Greedy (Local Improvement) | 6.5545 | 4.4371 | 32.30% | 0.5425 | 2.32 | 106138.0 |
| Greedy (Local Search) | 6.5545 | 4.4327 | 32.37% | 0.5450 | 2.32 | 259116.3 |
| Greedy (Construction) | 6.5545 | 4.4327 | 32.37% | 0.5450 | 2.32 | 105399.5 |
| Random Baseline | 6.5545 | 6.4982 | 0.86% | 0.5007 | 1.75 | 464.0 |


## Scenario: Conflict-Heavy (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.6090 | 4.4584 | 3.27% | 0.5492 | 0.94 | 53598.6 |
| Greedy (Local Improvement) | 6.8061 | 4.4728 | 34.28% | 0.5474 | 0.94 | 19572.6 |
| Greedy (Local Search) | 6.8061 | 4.4721 | 34.29% | 0.5475 | 0.94 | 26195.6 |
| Greedy (Construction) | 6.8061 | 4.4586 | 34.49% | 0.5492 | 0.94 | 19679.2 |
| Random Baseline | 6.8061 | 6.9670 | -2.36% | 0.5159 | 1.63 | 238.4 |


## Scenario: Risk-Heavy (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5096 | 4.3195 | 4.22% | 0.5742 | 1.63 | 62079.3 |
| Greedy (Local Improvement) | 6.8405 | 4.3278 | 36.73% | 0.5683 | 1.63 | 19696.6 |
| Greedy (Local Search) | 6.8405 | 4.3197 | 36.85% | 0.5744 | 1.63 | 36850.2 |
| Greedy (Construction) | 6.8405 | 4.3200 | 36.85% | 0.5741 | 1.63 | 19708.8 |
| Random Baseline | 6.8405 | 7.3487 | -7.43% | 0.5772 | 1.05 | 228.7 |

