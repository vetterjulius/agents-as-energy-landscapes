import torch
from energy.base import EnergyTerm

class EBMAOAssignmentEnergy(EnergyTerm):
    def __init__(self, lambda_align, weight=1.0):
        super().__init__(weight)
        self.lambda_align = lambda_align

    def compute(self, state):
        # s shape: (N, d)
        # c shape: (M, d)
        # kappa shape: (N, d)
        # X shape: (N, M)

        # Semantic distance: ||s_a - c_i||^2
        dist = torch.cdist(state.s, state.c) ** 2  # (N, M)

        # Memory coupling: s_a^\top \kappa_a
        # align_mem is (N,), we unsqueeze(1) to make it (N, 1) and expand/broadcast to (N, M)
        align_mem = torch.einsum("nd,nd->n", state.s, state.kappa).unsqueeze(1) # (N, 1)

        score = dist - self.lambda_align * align_mem # (N, M)

        # Return normalized potential
        return (state.X * score).sum() / (state.N * state.M)
