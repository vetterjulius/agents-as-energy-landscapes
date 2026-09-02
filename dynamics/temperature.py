class TemperatureController:
    def __init__(
        self,
        target_accept,
        T_min,
        T_max,
        adaptation_rate=0.05,
    ):
        self.target_accept = float(target_accept)
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.adaptation_rate = float(adaptation_rate)

        if self.T_min <= 0:
            raise ValueError("T_min must be > 0.")

        if self.T_max < self.T_min:
            raise ValueError("T_max must be >= T_min.")

        if not 0.0 < self.target_accept < 1.0:
            raise ValueError(
                "target_accept must be between 0 and 1."
            )

        if self.adaptation_rate < 0:
            raise ValueError(
                "adaptation_rate must be >= 0."
            )

    def apply(self, sampler_state):
        error = self.target_accept - sampler_state.acc_rate

        # Multiplicative temperature adaptation.
        #
        # acc_rate < target:
        #     error > 0 -> T increases
        #
        # acc_rate > target:
        #     error < 0 -> T decreases
        factor = 1.0 + self.adaptation_rate * error

        sampler_state.T *= factor

        sampler_state.T = max(
            self.T_min,
            min(sampler_state.T, self.T_max),
        )