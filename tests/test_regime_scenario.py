import torch

from benchmark.dynamic_benchmark import (
    MultiEpisodeSimulator,
    context_similarity,
    generate_regime_switch_episode,
)
from benchmark.evaluation.metrics import compute_energy
from benchmark.evaluation.metrics import brute_force_optimum
from benchmark.scenarios.interaction import InteractionScenario


def test_regime_switch_repeats_structured_regimes():
    episode_a0 = generate_regime_switch_episode(0, seed=7)
    episode_b = generate_regime_switch_episode(10, seed=7)
    episode_a1 = generate_regime_switch_episode(20, seed=7)

    assert len(episode_a0.agents) == 10
    assert len(episode_a0.tasks) == 25
    assert episode_a0.tasks[0].embedding.shape == (8,)
    assert episode_a0.constraints["regime"] == 0
    assert episode_b.constraints["regime"] == 1
    assert episode_a1.constraints["regime"] == 0

    assert torch.equal(
        episode_a0.tasks[0].embedding,
        episode_a1.tasks[0].embedding,
    )
    assert not torch.equal(
        episode_a0.tasks[0].embedding,
        episode_b.tasks[0].embedding,
    )
    assert not torch.equal(
        episode_a0.interaction_graph,
        episode_b.interaction_graph,
    )


def test_regime_runner_applies_memory_retention():
    simulator = MultiEpisodeSimulator(
        generate_regime_switch_episode,
        num_episodes=2,
        seed=7,
    )
    full_memory = simulator.run(
        config_override={"solver": {"iterations": 1}},
        kappa_enabled=True,
        theta_enabled=False,
        search_mode="guided_sa",
        memory_retention=1.0,
    )
    forgotten_memory = simulator.run(
        config_override={"solver": {"iterations": 1}},
        kappa_enabled=True,
        theta_enabled=False,
        search_mode="guided_sa",
        memory_retention=0.0,
    )

    assert full_memory.kappa_norm.iloc[1] > forgotten_memory.kappa_norm.iloc[1]


def test_context_similarity_distinguishes_repeated_regimes():
    episode_a = generate_regime_switch_episode(0, seed=7)
    episode_b = generate_regime_switch_episode(10, seed=7)
    context_a = torch.stack([task.embedding for task in episode_a.tasks])
    context_b = torch.stack([task.embedding for task in episode_b.tasks])

    same_regime = context_similarity(
        context_a,
        episode_a.interaction_graph,
        (context_a, episode_a.interaction_graph),
    )
    different_regime = context_similarity(
        context_b,
        episode_b.interaction_graph,
        (context_a, episode_a.interaction_graph),
    )

    assert same_regime == 1.0
    assert 0.0 <= different_regime < same_regime


def test_adaptive_retention_records_context_similarity():
    history = MultiEpisodeSimulator(
        generate_regime_switch_episode,
        num_episodes=11,
        seed=7,
    ).run(
        config_override={"solver": {"iterations": 1}},
        kappa_enabled=True,
        theta_enabled=True,
        search_mode="guided_sa",
        adaptive_retention=True,
    )

    assert "context_similarity" in history
    assert history.context_similarity.iloc[0] == 1.0
    assert history.context_similarity.iloc[10] < 1.0


def test_energy_evaluation_can_use_adaptive_state():
    problem = generate_regime_switch_episode(0, seed=7)
    X = torch.zeros(len(problem.agents), len(problem.tasks))
    for task_idx in range(len(problem.tasks)):
        X[task_idx % len(problem.agents), task_idx] = 1.0

    dimension = problem.agents[0].capability_embedding.shape[0]
    kappa = torch.ones(len(problem.agents), dimension)
    ground_truth_energy, _ = compute_energy(problem, X)
    adaptive_energy, _ = compute_energy(
        problem,
        X,
        kappa=kappa,
        theta=torch.ones_like(problem.interaction_graph),
    )

    assert ground_truth_energy != adaptive_energy


def test_dynamic_runner_can_record_reference_gaps():
    scenario = InteractionScenario(num_agents=2, num_tasks=3, dim=2)
    history = MultiEpisodeSimulator(
        lambda episode, seed: scenario.generate(seed + episode),
        num_episodes=2,
        seed=7,
    ).run(
        config_override={"solver": {"iterations": 1}},
        kappa_enabled=False,
        theta_enabled=False,
        search_mode="pure_sa",
        reference_energy_fn=brute_force_optimum,
    )

    assert history.reference_energy.notna().all()
    assert history.absolute_gap.notna().all()
