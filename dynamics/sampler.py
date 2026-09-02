import math
import random


class SimulatedAnnealingSampler:
    def __init__(
        self,
        proposal_mechanism,
        energy_registry,
        T_init=2.0,
        target_accept=0.3,
        num_candidates=4,
        mode="sa",
    ):
        self.proposal_mechanism = proposal_mechanism
        self.energy_registry = energy_registry

        self.T = float(T_init)
        self.target_accept = float(target_accept)

        self.acc_rate = 0.3
        self._acc_buffer = []

        self.num_candidates = max(1, int(num_candidates))

        if mode not in ("sa", "hybrid"):
            raise ValueError(
                f"Unknown sampler mode: {mode}. "
                f"Expected 'sa' or 'hybrid'."
            )

        self.mode = mode

    def step(self, state):
        X_old = state.X.clone()
        E_old, _ = self.energy_registry.compute(state)
        E_old = E_old.item()

        if self.mode == "sa":

            state.X = X_old

            X_new = self.proposal_mechanism.propose(state)

            state.X = X_new

            E_new, _ = self.energy_registry.compute(state)
            E_new = E_new.item()

            dE = E_new - E_old

        else:  # self.mode == "hybrid"

            best_X = X_old.clone()
            best_E = E_old

            for _ in range(self.num_candidates):

                # Every candidate starts from the SAME state.
                state.X = X_old

                X_new = self.proposal_mechanism.propose(state)

                state.X = X_new

                E_new, _ = self.energy_registry.compute(state)
                E_new = E_new.item()

                if E_new < best_E:
                    best_E = E_new
                    best_X = X_new.clone()

            state.X = best_X

            dE = best_E - E_old

        # --------------------------------------------------------
        # Metropolis acceptance
        # --------------------------------------------------------
        if dE <= 0.0:
            accept_prob = 1.0
        else:
            log_p = -dE / max(self.T, 1e-8)

            # Numerical safety.
            log_p = max(min(log_p, 20.0), -20.0)

            accept_prob = math.exp(log_p)

        accepted = random.random() < accept_prob

        # --------------------------------------------------------
        # Reject:
        # restore original state.
        # --------------------------------------------------------
        if not accepted:
            state.X = X_old

        # --------------------------------------------------------
        # Acceptance statistics
        # --------------------------------------------------------
        self._acc_buffer.append(1 if accepted else 0)

        # Keep a rolling window of the last 100 decisions.
        if len(self._acc_buffer) > 100:
            self._acc_buffer.pop(0)

        self.acc_rate = (
            sum(self._acc_buffer) / len(self._acc_buffer)
        )

        return accepted