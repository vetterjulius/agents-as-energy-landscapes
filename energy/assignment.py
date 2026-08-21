import torch
from energy.base import EnergyTerm


class AssignmentEnergy(EnergyTerm):
    def __init__(
        self,
        lambda_align,
        lambda_memory=None,
        weight=1.0,
    ):
        super().__init__(weight)

        self.lambda_align = lambda_align
        self.lambda_memory = (
            lambda_align
            if lambda_memory is None
            else lambda_memory
        )

    def compute(self, state):
        dist = torch.cdist(state.s, state.c) ** 2

        align_sc = torch.einsum(
            "nd,md->nm",
            state.s,
            state.c,
        )

        align_mem = torch.einsum(
            "nd,nd->n",
            state.s,
            state.kappa,
        ).unsqueeze(1)

        score = (
            dist
            - self.lambda_align * align_sc
            - self.lambda_memory * align_mem
        )

        return (state.X * score).sum() / (state.N * state.M)