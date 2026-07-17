import torch
from energy.base import EnergyTerm

class EBMAORiskEnergy(EnergyTerm):
    def __init__(self, predictor, weight=1.0):
        super().__init__(weight)
        self.predictor = predictor

    def compute(self, state):
        p = self.predictor.predict(state)
        eps = 1e-8
        # Return negative sum normalized
        return -(state.X * torch.log(p + eps)).sum() / (state.N * state.M)
