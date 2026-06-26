import math
import random
import torch

class SimulatedAnnealingSampler:
    def __init__(self, proposal_mechanism, energy_registry, T_init=2.0, target_accept=0.3):
        self.proposal_mechanism = proposal_mechanism
        self.energy_registry = energy_registry
        self.T = T_init
        self.acc_rate = 0.3
        self._acc_buffer = []

    def step(self, state):
        X_old = state.X.clone()
        E_old, _ = self.energy_registry.compute(state)

        X_new = self.proposal_mechanism.propose(state)
        state.X = X_new
        E_new, _ = self.energy_registry.compute(state)

        dE = E_new - E_old

        log_p = -dE / max(self.T, 1e-8)
        if isinstance(log_p, torch.Tensor):
            log_p = log_p.item()
        log_p = max(min(log_p, 20.0), -20.0)

        accept_prob = math.exp(log_p)
        accepted = random.random() < accept_prob

        if not accepted:
            state.X = X_old

        val = 1 if accepted else 0
        self._acc_buffer.append(val)
        self.acc_rate = 0.9 * self.acc_rate + 0.1 * val

        return accepted
