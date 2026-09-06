import torch
from energy.base import EnergyTerm

class CostEnergy(EnergyTerm):
    def compute(self, state):
        co = state.X.T @ state.X
        # Normalize by N * M for scale-invariance across agents and tasks
        return (state.C * co).sum() / (state.N * state.M)
