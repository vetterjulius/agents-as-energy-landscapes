import random
import torch

class EBMAOAssignmentProposal:
    def __init__(self, energy_registry, lambda_align=0.5, num_tasks=4, block_size=3, agent_sample_size=None):
        self.energy_registry = energy_registry
        self.lambda_align = lambda_align
        self.num_tasks = max(1, int(num_tasks))
        self.block_size = max(1, int(block_size))
        self.agent_sample_size = agent_sample_size

    def propose(self, state):
        r = random.random()
        if r < 0.45:
            prop = self._guided_single_swap(state)
        elif r < 0.75:
            prop = self._guided_block_move(state)
        elif r < 0.95:
            prop = self._random_swap(state)
        else:
            prop = self._full_single_reassignment(state)

        if torch.equal(prop, state.X):
            prop = self._random_swap(state)
        return prop

    def _guided_single_swap(self, state):
        tasks = self._select_tasks(state, self.num_tasks)
        return self._best_swap_among_tasks(state, tasks)

    def _guided_block_move(self, state):
        tasks = self._select_tasks(state, self.block_size)
        return self._best_block_move(state, tasks)

    def _random_swap(self, state):
        X_new = state.X.clone()
        t = random.randint(0, state.M - 1)
        a_old = torch.argmax(state.X[:, t]).item()
        candidates = [a for a in range(state.N) if a != a_old]
        if candidates:
            a_new = random.choice(candidates)
            X_new[a_old, t] = 0.0
            X_new[a_new, t] = 1.0
        return X_new

    def _select_tasks(self, state, k):
        if k >= state.M:
            return list(range(state.M))

        assigned = torch.argmax(state.X, dim=0)
        agent_s = state.s[assigned]
        agent_k = state.kappa[assigned]

        # In EBMAO, the assignment term inside parenthesis is: ||s_a - c_i||^2 - \lambda s_a^\top \kappa_a
        dist = torch.sum((agent_s - state.c) ** 2, dim=1)
        align_mem = torch.sum(agent_s * agent_k, dim=1)

        importance = dist - self.lambda_align * align_mem
        _, top_idxs = torch.topk(importance, k, largest=True)
        return top_idxs.tolist()

    def _candidate_agents(self, state, a_old):
        candidates = [a for a in range(state.N) if a != a_old]
        if self.agent_sample_size is None or self.agent_sample_size >= len(candidates):
            return candidates
        return random.sample(candidates, self.agent_sample_size)

    def _best_swap_among_tasks(self, state, tasks):
        X_orig = state.X.clone()
        best_X = X_orig.clone()
        best_E, _ = self.energy_registry.compute(state)
        best_E = best_E.item()

        for t in tasks:
            a_old = torch.argmax(X_orig[:, t]).item()
            for a_new in self._candidate_agents(state, a_old):
                X_prop = X_orig.clone()
                X_prop[a_old, t] = 0.0
                X_prop[a_new, t] = 1.0
                state.X = X_prop
                E, _ = self.energy_registry.compute(state)
                E_val = E.item()
                if E_val < best_E:
                    best_E = E_val
                    best_X = X_prop.clone()

        state.X = X_orig
        return best_X

    def _best_block_move(self, state, tasks):
        X_orig = state.X.clone()
        X_prop = X_orig.clone()
        best_E, _ = self.energy_registry.compute(state)
        best_E = best_E.item()

        for t in tasks:
            a_old = torch.argmax(X_prop[:, t]).item()
            candidates = self._candidate_agents(state, a_old)
            best_local_X = X_prop.clone()
            best_local_E = best_E

            for a_new in candidates:
                X_trial = X_prop.clone()
                X_trial[a_old, t] = 0.0
                X_trial[a_new, t] = 1.0
                state.X = X_trial
                E, _ = self.energy_registry.compute(state)
                E_val = E.item()
                if E_val < best_local_E:
                    best_local_E = E_val
                    best_local_X = X_trial.clone()

            X_prop = best_local_X
            if best_local_E < best_E:
                best_E = best_local_E

        state.X = X_orig
        return X_prop

    def _full_single_reassignment(self, state):
        X_orig = state.X.clone()
        best_X = X_orig.clone()
        best_E, _ = self.energy_registry.compute(state)
        best_E = best_E.item()

        for t in range(state.M):
            a_old = torch.argmax(X_orig[:, t]).item()
            for a_new in range(state.N):
                if a_new == a_old:
                    continue
                X_prop = X_orig.clone()
                X_prop[a_old, t] = 0.0
                X_prop[a_new, t] = 1.0
                state.X = X_prop
                E, _ = self.energy_registry.compute(state)
                E_val = E.item()
                if E_val < best_E:
                    best_E = E_val
                    best_X = X_prop.clone()

        state.X = X_orig
        return best_X
