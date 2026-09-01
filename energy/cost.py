import torch
from energy.base import EnergyTerm

class CostEnergy(EnergyTerm):
    def compute(self, state):
        co = state.X.T @ state.X
        # Normalize by M² since co-occurrence matrix is [M, M]
        # This ensures scale-invariant energy across problem sizes
        return (state.C * co).sum() / (state.M * state.M)
