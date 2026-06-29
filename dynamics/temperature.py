class TemperatureController:
    def __init__(self, target_accept, T_min, T_max):
        self.target_accept = target_accept
        self.T_min = T_min
        self.T_max = T_max

    def apply(self, sampler_state):
        # Decrease temperature when acceptance is too high and increase it when acceptance is too low.
        error = self.target_accept - sampler_state.acc_rate
        sampler_state.T += 0.1 * error
        sampler_state.T = max(self.T_min, min(sampler_state.T, self.T_max))
