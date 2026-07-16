import torch
from .base import Orchestrator
from ..scenarios.base import ProblemInstance
from ..evaluation.metrics import compute_energy

class BeamSearchOrchestrator(Orchestrator):
    def __init__(self, beam_width=5):
        self.beam_width = beam_width

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)

        # Start with an empty assignment (all zeros)
        # Each beam state is a tuple: (energy_value, assignment_matrix_X)
        X_init = torch.zeros(N, M)
        beam = [(0.0, X_init)]

        for t in range(M):
            candidates = []
            for _, X_curr in beam:
                for a in range(N):
                    X_next = X_curr.clone()
                    X_next[a, t] = 1.0

                    # Compute energy of this partial assignment
                    energy, _ = compute_energy(problem, X_next)
                    candidates.append((energy, X_next))

            # Sort candidates by energy (lower is better) and keep top beam_width
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:self.beam_width]

        # Return the best assignment from the final beam
        return beam[0][1]
