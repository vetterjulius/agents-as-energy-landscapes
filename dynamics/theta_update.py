import torch

class ThetaUpdater:
    def __init__(self, eta_theta, epsilon=1e-8):
        self.eta_theta = eta_theta
        self.epsilon = epsilon
        self.running_co = None

    def apply(self, state):
        co = state.X.T @ state.X
        co_norm = co / (co.sum() + self.epsilon)

        if self.running_co is None:
            self.running_co = co_norm.clone()
            old_running = co_norm.clone()
        else:
            old_running = self.running_co.clone()
            self.running_co = (1 - self.eta_theta) * self.running_co + self.eta_theta * co_norm

        # Update Theta with the difference between normalized current co-occurrence and its historical baseline
        state.Theta = (1 - self.eta_theta) * state.Theta + self.eta_theta * (co_norm - old_running)
