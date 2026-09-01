import torch
from energy.base import EnergyTerm


class InteractionEnergy(EnergyTerm):
    def compute(self, state):
        co = state.X.T @ state.X

        term_matrix = state.Theta * co

        upper_tri_mask = torch.triu(
            torch.ones_like(term_matrix),
            diagonal=1
        )

        total_sum = (term_matrix * upper_tri_mask).sum()

        # Normalize by M² since interaction matrix is [M, M]
        # Upper triangular has M*(M-1)/2 elements, but we normalize by M² for consistency
        return -total_sum / (state.M * state.M)