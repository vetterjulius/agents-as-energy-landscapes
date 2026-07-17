# Benchmark Evaluation Results

This report evaluates the performance of the Orchestrator system against the baselines.

## Scenario: Small (5x15)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.3400 | 4.2929 | 1.08% | 0.4393 | 2.45 | 839.9 |
| EBMAO (Proposed) | 4.7845 | 4.7502 | 0.72% | 0.3846 | 4.12 | 934.1 |
| Greedy (Local Improvement) | 7.5821 | 5.6583 | 25.37% | 0.5205 | 1.00 | 150.9 |
| Greedy (Local Search) | 7.5821 | 4.3103 | 43.15% | 0.4377 | 2.45 | 555.8 |
| Greedy (Construction) | 7.5821 | 4.3101 | 43.15% | 0.4376 | 2.45 | 191.0 |
| Random Baseline | 7.5821 | 7.6070 | -0.33% | 0.4987 | 2.24 | 5.2 |


## Scenario: Medium (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5232 | 4.4935 | 0.66% | 0.4092 | 1.70 | 3052.0 |
| EBMAO (Proposed) | 5.3284 | 5.3065 | 0.41% | 0.3471 | 4.83 | 3015.5 |
| Greedy (Local Improvement) | 6.4638 | 5.7770 | 10.62% | 0.4253 | 0.82 | 765.8 |
| Greedy (Local Search) | 6.4638 | 4.5245 | 30.00% | 0.4045 | 1.70 | 5227.4 |
| Greedy (Construction) | 6.4638 | 4.5232 | 30.02% | 0.4037 | 1.70 | 791.5 |
| Random Baseline | 6.4638 | 6.4036 | 0.93% | 0.4158 | 1.25 | 8.2 |


## Scenario: Large (20x60)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5775 | 4.5448 | 0.72% | 0.5232 | 2.34 | 6469.5 |
| EBMAO (Proposed) | 5.4422 | 5.4155 | 0.49% | 0.5656 | 4.00 | 11717.4 |
| Greedy (Local Improvement) | 6.5545 | 6.1654 | 5.94% | 0.5108 | 1.78 | 3701.8 |
| Greedy (Local Search) | 6.5545 | 4.5635 | 30.38% | 0.5240 | 2.34 | 45344.2 |
| Greedy (Construction) | 6.5545 | 4.5634 | 30.38% | 0.5238 | 2.34 | 3710.1 |
| Random Baseline | 6.5545 | 6.2203 | 5.10% | 0.5359 | 1.65 | 14.9 |


## Scenario: Conflict-Heavy (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.6090 | 4.5745 | 0.75% | 0.5554 | 0.82 | 3252.6 |
| EBMAO (Proposed) | 5.9788 | 5.6035 | 6.28% | 0.5372 | 9.49 | 3558.3 |
| Greedy (Local Improvement) | 6.8061 | 6.0077 | 11.73% | 0.5723 | 1.05 | 775.2 |
| Greedy (Local Search) | 6.8061 | 4.6154 | 32.19% | 0.5602 | 0.82 | 5872.1 |
| Greedy (Construction) | 6.8061 | 4.6013 | 32.39% | 0.5612 | 0.82 | 781.4 |
| Random Baseline | 6.8061 | 7.3321 | -7.73% | 0.5164 | 1.70 | 7.8 |


## Scenario: Risk-Heavy (10x30)
| Solver | Init Energy | Final Energy | Reduction | Success Prob | Load Bal Std | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Simulated Annealing (SA) | 4.5096 | 4.4638 | 1.02% | 0.5844 | 1.49 | 2498.4 |
| EBMAO (Proposed) | 5.3670 | 5.3362 | 0.57% | 0.5702 | 3.33 | 2582.2 |
| Greedy (Local Improvement) | 6.8405 | 5.9091 | 13.62% | 0.5920 | 1.70 | 783.2 |
| Greedy (Local Search) | 6.8405 | 4.4996 | 34.22% | 0.5861 | 1.49 | 4788.6 |
| Greedy (Construction) | 6.8405 | 4.4994 | 34.22% | 0.5860 | 1.49 | 785.9 |
| Random Baseline | 6.8405 | 7.0232 | -2.67% | 0.5631 | 1.15 | 7.7 |

