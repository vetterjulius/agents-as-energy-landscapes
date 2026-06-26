import random
import torch

class AssignmentProposal:
    def propose(self, state):
        X_new = state.X.clone()

        if random.random() < 0.7:
            # single swap
            t = random.randint(0, state.M - 1)
            a_old = torch.argmax(state.X[:, t]).item()
            candidates = [a for a in range(state.N) if a != a_old]
            if candidates:
                a_new = random.choice(candidates)
                X_new[a_old, t] = 0.0
                X_new[a_new, t] = 1.0
        else:
            # block move
            tasks = random.sample(range(state.M), k=min(3, state.M))

            for t in tasks:
                a_old = torch.argmax(state.X[:, t]).item()
                candidates = [a for a in range(state.N) if a != a_old]
                if candidates:
                    a_new = random.choice(candidates)
                    X_new[a_old, t] = 0.0
                    X_new[a_new, t] = 1.0

        return X_new
