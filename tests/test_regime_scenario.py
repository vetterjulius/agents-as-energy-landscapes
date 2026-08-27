import torch

from benchmark.dynamic_benchmark import generate_regime_switch_episode


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
