import torch
from energy.base import EnergyTerm

class EBMAOInteractionEnergy(EnergyTerm):
    def compute(self, state):
        # E_inter = - \sum_{i < j} \Theta_{ij} \sum_a X_{a, i} X_{a, j}
        # Let's compute this exactly as specified in the new formalisation.
        # X shape: (N, M)
        # Theta shape: (M, M)
        # co-occurrence matrix of assignments: co_a = \sum_a X_{a, i} X_{a, j}
        # which is co = X.T @ X of shape (M, M)

        co = state.X.T @ state.X # (M, M)

        # We want the sum over i < j.
        # Since state.Theta and co are of shape (M, M), we can extract the strictly upper triangular part
        # of (state.Theta * co).
        term_matrix = state.Theta * co
        upper_tri_mask = torch.triu(torch.ones_like(term_matrix), diagonal=1)
        total_sum = (term_matrix * upper_tri_mask).sum()

        # Return negative sum normalized
        return -total_sum / (state.N * state.M)
