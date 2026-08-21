import torch

from model.ebmao_orchestrator import EBMAOOrchestrator


def test_ebmao_runs_multiple_steps():
    cfg = {
        "model": {
            "num_agents": 4,
            "num_tasks": 8,
            "dim": 3,

            "lambda_align": 0.5,
            "lambda_memory": 0.5,

            "eta_theta": 0.1,
            "eta_memory": 0.1,

            "risk_weight": 1.0,
            "interaction_weight": 1.0,
            "cost_weight": 1.0,

            "temperature_init": 2.0,
            "min_temperature": 0.1,
            "max_temperature": 5.0,
            "target_accept_rate": 0.3,

            "search_mode": "hybrid",

            "proposal_candidates": 4,
            "proposal_task_sample": 3,
            "agent_sample_size": 3,
            "block_move_size": 2,

            "warm_start_steps": 1,
            "local_refine_steps": 1,
        }
    }

    torch.manual_seed(0)

    orchestrator = EBMAOOrchestrator(cfg)

    for _ in range(10):
        orchestrator.step()

        X = orchestrator.X

        # Jede Task genau einem Agenten zugeordnet
        assert torch.allclose(
            X.sum(dim=0),
            torch.ones(orchestrator.M),
        )

        # Nur 0/1
        assert torch.all((X == 0) | (X == 1))

        # Energie muss endlich sein
        E = orchestrator.total_energy()
        assert torch.isfinite(E)