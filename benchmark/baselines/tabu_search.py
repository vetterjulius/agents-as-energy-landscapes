import torch
from .base import Orchestrator
from ..scenarios.base import ProblemInstance
from ..evaluation.metrics import compute_energy

class TabuSearchOrchestrator(Orchestrator):
    def __init__(self, max_iterations=50, tabu_tenure=5):
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)

        # Initial assignment: modulo assignment
        X_curr = torch.zeros(N, M)
        for t in range(M):
            X_curr[t % N, t] = 1.0

        best_X = X_curr.clone()
        best_E, _ = compute_energy(problem, best_X)

        curr_E = best_E

        # Tabu list: maps (task_idx, agent_idx) to the iteration number until which it is tabu
        tabu_list = {}

        for iteration in range(self.max_iterations):
            best_move = None
            best_move_E = float('inf')
            best_move_X = None

            # Explore all single-task reassignments
            for t in range(M):
                a_curr = torch.argmax(X_curr[:, t]).item()
                for a in range(N):
                    if a == a_curr:
                        continue

                    # Propose move: move task t to agent a
                    X_prop = X_curr.clone()
                    X_prop[a_curr, t] = 0.0
                    X_prop[a, t] = 1.0

                    E_prop, _ = compute_energy(problem, X_prop)

                    # Check tabu status
                    is_tabu = (t, a) in tabu_list and iteration < tabu_list[(t, a)]

                    # Aspiration criterion: if it's better than best_E, we accept even if tabu
                    if E_prop < best_E - 1e-6:
                        is_tabu = False

                    if not is_tabu:
                        if E_prop < best_move_E:
                            best_move_E = E_prop
                            best_move = (t, a_curr, a)  # task t, from a_curr to a
                            best_move_X = X_prop

            if best_move is None:
                # No non-tabu moves available, break
                break

            # Apply best move
            t, a_old, a_new = best_move
            X_curr = best_move_X
            curr_E = best_move_E

            # Update tabu list: the reverse move (moving task t back to a_old) is tabu
            tabu_list[(t, a_old)] = iteration + self.tabu_tenure

            # Update best assignment seen so far
            if curr_E < best_E - 1e-6:
                best_E = curr_E
                best_X = X_curr.clone()

        return best_X
