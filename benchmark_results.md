# Benchmark Evaluation Results

This report evaluates the performance of the Orchestrator system against the baselines.

## Scenario: Small (5x15)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.3400 | 4.1194 | 5.08% | 0.4587 | 2.45 | 20648.3 |
| Greedy (Local Improvement) | 7.5821 | 4.1202 | 45.66% | 0.4585 | 2.45 | 3451.3 |
| Greedy (Local Search) | 7.5821 | 4.1196 | 45.67% | 0.4587 | 2.45 | 3909.4 |
| Greedy (Construction) | 7.5821 | 4.1196 | 45.67% | 0.4587 | 2.45 | 3512.9 |
| Random Baseline | 7.5821 | 7.0258 | 7.34% | 0.4917 | 1.58 | 120.3 |


## Scenario: Medium (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5232 | 4.3855 | 3.04% | 0.4177 | 1.70 | 66154.9 |
| Greedy (Local Improvement) | 6.4638 | 4.3868 | 32.13% | 0.4173 | 1.70 | 19895.6 |
| Greedy (Local Search) | 6.4638 | 4.3860 | 32.14% | 0.4262 | 1.83 | 31090.8 |
| Greedy (Construction) | 6.4638 | 4.3859 | 32.15% | 0.4177 | 1.70 | 21129.7 |
| Random Baseline | 6.4638 | 7.0229 | -8.65% | 0.4150 | 1.83 | 259.9 |


## Scenario: Large (20x60)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5776 | 4.4325 | 3.17% | 0.5450 | 2.32 | 272505.8 |
| Greedy (Local Improvement) | 6.5545 | 4.4371 | 32.30% | 0.5425 | 2.32 | 110804.3 |
| Greedy (Local Search) | 6.5545 | 4.4327 | 32.37% | 0.5450 | 2.32 | 267045.5 |
| Greedy (Construction) | 6.5545 | 4.4327 | 32.37% | 0.5450 | 2.32 | 111310.3 |
| Random Baseline | 6.5545 | 6.5866 | -0.49% | 0.4803 | 1.49 | 509.2 |


## Scenario: Conflict-Heavy (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.6090 | 4.4584 | 3.27% | 0.5492 | 0.94 | 71302.1 |
| Greedy (Local Improvement) | 6.8061 | 4.4728 | 34.28% | 0.5474 | 0.94 | 20550.8 |
| Greedy (Local Search) | 6.8061 | 4.4721 | 34.29% | 0.5475 | 0.94 | 26494.5 |
| Greedy (Construction) | 6.8061 | 4.4586 | 34.49% | 0.5492 | 0.94 | 19919.7 |
| Random Baseline | 6.8061 | 6.7445 | 0.91% | 0.4927 | 1.70 | 239.3 |


## Scenario: Risk-Heavy (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5096 | 4.3195 | 4.22% | 0.5742 | 1.63 | 71173.4 |
| Greedy (Local Improvement) | 6.8405 | 4.3278 | 36.73% | 0.5683 | 1.63 | 20152.9 |
| Greedy (Local Search) | 6.8405 | 4.3197 | 36.85% | 0.5744 | 1.63 | 37438.5 |
| Greedy (Construction) | 6.8405 | 4.3200 | 36.85% | 0.5741 | 1.63 | 20226.5 |
| Random Baseline | 6.8405 | 7.4747 | -9.27% | 0.6345 | 2.11 | 242.1 |

