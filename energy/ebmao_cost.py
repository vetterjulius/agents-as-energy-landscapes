import torch
from energy.base import EnergyTerm

class EBMAOCostEnergy(EnergyTerm):
    def compute(self, state):
        # E_cost = \sum_{i < j} c_{ij} \sum_{a \neq b} X_{a, i} X_{b, j}
        # Since \sum_{a \neq b} X_{a, i} X_{b, j} = 1 - \sum_a X_{a, i} X_{a, j} = 1 - co_{i, j}

        co = state.X.T @ state.X # (M, M)
        cost_matrix = state.C * (1.0 - co)

        upper_tri_mask = torch.triu(torch.ones_like(cost_matrix), diagonal=1)
        total_sum = (cost_matrix * upper_tri_mask).sum()

        # Return normalized communication cost potential
        return total_sum / (state.N * state.M)
