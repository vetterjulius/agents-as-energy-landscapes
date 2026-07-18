import torch
from energy.base import EnergyTerm

class EBMAOCostEnergy(EnergyTerm):
    def compute(self, state):
        co = state.X.T @ state.X
        return (state.C * co).sum() / (state.N * state.M)
