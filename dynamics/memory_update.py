import torch

class MemoryUpdater:
    def __init__(self, eta_memory):
        self.eta_memory = eta_memory

    def apply(self, state, risk_predictor):
        p = risk_predictor.predict(state)

        for a in range(state.N):
            tasks = (state.X[a] > 0).nonzero(as_tuple=True)[0]

            if len(tasks) > 0:
                # Agent has assigned tasks: update based on task embeddings and success
                task_emb = state.c[tasks]
                mean_emb = task_emb.mean(dim=0)
                success = p[a, tasks].mean()

                state.kappa[a] = (
                    (1 - self.eta_memory) * state.kappa[a]
                    + self.eta_memory * mean_emb * success
                )
            else:
                # Agent has no tasks: decay memory toward zero to prevent cold-start bias
                # This allows previously idle agents to compete fairly in future assignments
                state.kappa[a] = (1 - self.eta_memory * 0.1) * state.kappa[a]
