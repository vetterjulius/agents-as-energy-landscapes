import math
import torch
from energy.base import EnergyTerm

class RiskPredictor:
    def __init__(self, d, W_risk=None, scale=1.0):
        if W_risk is not None:
            self.W_risk = W_risk
        else:
            self.W_risk = torch.randn(3 * d, 1) * scale
        self.d = d

    def predict(self, state):
        s_exp = state.s.unsqueeze(1).expand(-1, state.M, -1)
        c_exp = state.c.unsqueeze(0).expand(state.N, -1, -1)
        k_exp = state.kappa.unsqueeze(1).expand(-1, state.M, -1)

        x = torch.cat([s_exp, c_exp, k_exp], dim=-1)

        h = torch.matmul(x, self.W_risk).squeeze(-1)
        h = h / math.sqrt(self.d)

        return torch.sigmoid(h)

class RiskEnergy(EnergyTerm):
    def __init__(self, predictor, weight=1.0):
        super().__init__(weight)
        self.predictor = predictor

    def compute(self, state):
        p = self.predictor.predict(state)
        eps = 1e-8
        return -(state.X * torch.log(p + eps)).sum() / (state.N * state.M)
