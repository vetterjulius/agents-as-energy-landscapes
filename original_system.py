import torch
import random
import yaml
import math


class EnergyModel:
    def __init__(self, cfg):
        self.cfg = cfg
        m = cfg["model"]

        # --------------------
        # dimensions
        # --------------------
        self.N = m["num_agents"]
        self.M = m["num_tasks"]
        self.d = m["dim"]

        # --------------------
        # hyperparameters
        # --------------------
        self.lambda_align = m["lambda_align"]
        self.eta_theta = m["eta_theta"]
        self.eta_memory = m["eta_memory"]

        self.w_risk = m["risk_weight"]
        self.w_int = m["interaction_weight"]
        self.w_cost = m["cost_weight"]

        # --------------------
        # temperature (STABLE RANGE)
        # --------------------
        self.T = m["temperature_init"]
        self.T_min = m["min_temperature"]
        self.T_max = m["max_temperature"]

        self.target_accept = m["target_accept_rate"]

        self._acc_buffer = []
        self.acc_window = 100
        self.acc_rate = 0.3  # EMA initialization

        # --------------------
        # embeddings
        # --------------------
        self.s = torch.randn(self.N, self.d)
        self.c = torch.randn(self.M, self.d)
        self.kappa = torch.zeros(self.N, self.d)

        # interaction + dependency
        self.Theta = torch.zeros(self.M, self.M)
        self.C = torch.rand(self.M, self.M)
        self.C.fill_diagonal_(0)

        # --------------------
        # assignment
        # --------------------
        self.X = torch.zeros(self.N, self.M)
        for t in range(self.M):
            a = random.randint(0, self.N - 1)
            self.X[a, t] = 1.0

        # --------------------
        # risk model
        # --------------------
        self.W_risk = torch.randn(3 * self.d, 1) * 0.01

    # =========================================================
    # NORMALIZATION HELPERS (CRITICAL FIX)
    # =========================================================

    def norm(self, x):
        return x / (self.N * self.M)

    # =========================================================
    # ENERGY TERMS (SCALED)
    # =========================================================

    def energy_assign(self):
        dist = torch.cdist(self.s, self.c) ** 2

        align_sc = torch.einsum("nd,md->nm", self.s, self.c)
        align_mem = torch.einsum("nd,nd->n", self.s, self.kappa).unsqueeze(1)

        score = dist - self.lambda_align * (align_sc + align_mem)

        return self.norm((self.X * score).sum())

    def energy_interaction(self):
        co = self.X.T @ self.X
        return self.norm(-(self.Theta * co).sum())

    def energy_cost(self):
        co = self.X.T @ self.X
        return self.norm((self.C * co).sum())

    # =========================================================
    # RISK MODEL (stable + learnable)
    # =========================================================

    def risk_probs(self):
        s_exp = self.s.unsqueeze(1).expand(-1, self.M, -1)
        c_exp = self.c.unsqueeze(0).expand(self.N, -1, -1)
        k_exp = self.kappa.unsqueeze(1).expand(-1, self.M, -1)

        x = torch.cat([s_exp, c_exp, k_exp], dim=-1)

        h = torch.matmul(x, self.W_risk).squeeze(-1)
        h = h / math.sqrt(self.d)

        return torch.sigmoid(h)

    def energy_risk(self):
        p = self.risk_probs()
        eps = 1e-8
        return self.norm(-(self.X * torch.log(p + eps)).sum())

    # =========================================================
    # TOTAL ENERGY
    # =========================================================

    def total_energy(self):
        return (
            self.energy_assign()
            + self.w_int * self.energy_interaction()
            + self.w_cost * self.energy_cost()
            + self.w_risk * self.energy_risk()
        )

    # =========================================================
    # PROPOSAL DISTRIBUTION (STABLE MIXTURE)
    # =========================================================

    def propose(self):
        X_new = self.X.clone()

        if random.random() < 0.7:
            # single swap
            t = random.randint(0, self.M - 1)
            a_old = torch.argmax(self.X[:, t]).item()
            a_new = random.randint(0, self.N - 1)

            X_new[a_old, t] = 0.0
            X_new[a_new, t] = 1.0
        else:
            # block move
            tasks = random.sample(range(self.M), k=min(3, self.M))

            for t in tasks:
                a_old = torch.argmax(self.X[:, t]).item()
                a_new = random.randint(0, self.N - 1)

                X_new[a_old, t] = 0.0
                X_new[a_new, t] = 1.0

        return X_new

    # =========================================================
    # SA STEP (NUMERICALLY STABLE)
    # =========================================================

    def step_sa(self):
        X_old = self.X.clone()
        E_old = self.total_energy()

        self.X = self.propose()
        E_new = self.total_energy()

        dE = E_new - E_old

        # stable acceptance
        log_p = -dE / max(self.T, 1e-8)
        log_p = max(min(log_p, 20), -20)

        accept_prob = math.exp(log_p)

        accepted = random.random() < accept_prob

        if not accepted:
            self.X = X_old

        # EMA acceptance rate (IMPORTANT FIX)
        self._acc_buffer.append(1 if accepted else 0)
        self.acc_rate = 0.9 * self.acc_rate + 0.1 * (1 if accepted else 0)

    # =========================================================
    # INTERACTION LEARNING
    # =========================================================

    def update_theta(self):
        co = self.X.T @ self.X

        expected = co / (co.sum() + 1e-8)

        self.Theta = (
            (1 - self.eta_theta) * self.Theta
            + self.eta_theta * (co - expected)
        )

    # =========================================================
    # MEMORY UPDATE
    # =========================================================

    def update_memory(self):
        p = self.risk_probs()

        for a in range(self.N):
            tasks = (self.X[a] > 0).nonzero(as_tuple=True)[0]

            if len(tasks) > 0:
                task_emb = self.c[tasks]
                mean_emb = task_emb.mean(dim=0)

                success = p[a, tasks].mean()

                self.kappa[a] = (
                    (1 - self.eta_memory) * self.kappa[a]
                    + self.eta_memory * mean_emb * success
                )

    # =========================================================
    # STABLE TEMPERATURE CONTROL
    # =========================================================

    def update_temperature(self):
        error = self.acc_rate - self.target_accept

        # linear controller (stable!)
        self.T += 0.1 * error

        self.T = max(self.T_min, min(self.T, self.T_max))

    # =========================================================
    # FULL STEP
    # =========================================================

    def step(self):
        self.step_sa()
        self.update_theta()
        self.update_memory()
        self.update_temperature()

    # =========================================================
    # LOGGING
    # =========================================================

    def log(self, step):
        print(
            f"Step {step} | "
            f"E={self.total_energy().item():.4f} | "
            f"T={self.T:.3f} | "
            f"acc={self.acc_rate:.3f}"
        )
