from benchmark.recurrent_advantage_experiment import (
    run_recurrent_advantage_benchmark,
)


def test_recurrent_advantage_benchmark_is_reproducible_and_paired(tmp_path):
    result = run_recurrent_advantage_benchmark(
        seeds=(0, 1),
        num_episodes=4,
        iterations=1,
        output_path=tmp_path / "recurrent.csv",
        statistics_output_path=tmp_path / "recurrent_statistics.csv",
    )

    assert len(result) == 2 * 4 * 4
    assert result["configuration"].nunique() == 4
    assert result["reference_energy"].notna().all()
    assert result["absolute_gap"].notna().all()
    assert (tmp_path / "recurrent_statistics.csv").exists()
