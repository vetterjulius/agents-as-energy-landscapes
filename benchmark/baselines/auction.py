import torch
from .base import Orchestrator
from ..scenarios.base import ProblemInstance

class MarketAuctionOrchestrator(Orchestrator):
    """
    Market-Based Auction Orchestrator (Consensus-Based Auction / Sequential Single-Item Auction).
    
    In each round of the auction:
    1. Unassigned tasks are put up for bid.
    2. Agents compute their bid for each available task based on:
       - Marginal Capability Fit: -||s_i - c_t||
       - Synergies / Co-assignment: Bonus for taking tasks with strong coupling (Theta) to already won tasks.
       - Load Penalty / Marginal Cost: Penalty proportional to current workload and co-assignment costs.
    3. The task-agent pair with the highest overall bid is matched, and the agent updates its internal state.
    4. Repeats until all tasks are assigned.
    """
    def __init__(self, alpha_load: float = 0.5, beta_synergy: float = 0.5):
        self.alpha_load = alpha_load
        self.beta_synergy = beta_synergy

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)
        X = torch.zeros(N, M)

        s = torch.stack([a.capability_embedding for a in problem.agents])
        c = torch.stack([t.embedding for t in problem.tasks])

        Theta = problem.interaction_graph  # (M, M)
        C = problem.co_assignment_costs    # (M, M)

        unassigned_tasks = list(range(M))
        agent_assigned_tasks = [[] for _ in range(N)]

        while unassigned_tasks:
            best_bid = -float('inf')
            winning_agent = -1
            winning_task = -1

            for t in unassigned_tasks:
                c_t = c[t]
                # Distance cost
                dist = torch.norm(s - c_t, dim=1) # (N,)

                for a_idx in range(N):
                    current_won = agent_assigned_tasks[a_idx]
                    
                    # Synergy bonus with already assigned tasks for this agent
                    synergy = 0.0
                    co_cost = 0.0
                    if current_won:
                        won_tensor = torch.tensor(current_won, dtype=torch.long)
                        synergy = Theta[t, won_tensor].sum().item()
                        co_cost = C[t, won_tensor].sum().item()

                    # Marginal utility (Bid) = Capability Match + Synergy - Load Penalty - Co-Assignment Cost
                    workload = len(current_won)
                    bid = -dist[a_idx].item() + self.beta_synergy * synergy - self.alpha_load * workload - co_cost

                    if bid > best_bid:
                        best_bid = bid
                        winning_agent = a_idx
                        winning_task = t

            # Assign winner
            X[winning_agent, winning_task] = 1.0
            agent_assigned_tasks[winning_agent].append(winning_task)
            unassigned_tasks.remove(winning_task)

        return X
