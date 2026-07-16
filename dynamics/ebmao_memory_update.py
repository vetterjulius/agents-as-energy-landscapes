import torch

class EBMAOMemoryUpdater:
    def __init__(self, eta_memory):
        self.eta_memory = eta_memory

    def apply(self, state, risk_predictor):
        p = risk_predictor.predict(state) # shape (N, M)

        for a in range(state.N):
            tasks = (state.X[a] > 0).nonzero(as_tuple=True)[0]

            if len(tasks) > 0:
                # Get context embeddings for tasks assigned to agent a
                task_emb = state.c[tasks] # shape (len(tasks), d)

                # Get success probabilities for these tasks for agent a
                success_probs = p[a, tasks].unsqueeze(1) # shape (len(tasks), 1)

                # Compute weighted update: mean of p_{a, i} * c_i over assigned tasks
                weighted_update = (task_emb * success_probs).mean(dim=0) # shape (d,)

                # Update with decay
                state.kappa[a] = (
                    (1 - self.eta_memory) * state.kappa[a]
                    + self.eta_memory * weighted_update
                )
