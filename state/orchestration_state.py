from dataclasses import dataclass
import torch

@dataclass
class OrchestrationState:
    X: torch.Tensor
    s: torch.Tensor
    c: torch.Tensor
    kappa: torch.Tensor
    Theta: torch.Tensor
    C: torch.Tensor
    N: int
    M: int
    d: int

    def clone(self):
        return OrchestrationState(
            X=self.X.clone(),
            s=self.s.clone(),
            c=self.c.clone(),
            kappa=self.kappa.clone(),
            Theta=self.Theta.clone(),
            C=self.C.clone(),
            N=self.N,
            M=self.M,
            d=self.d
        )

