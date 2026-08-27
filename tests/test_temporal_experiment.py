import pandas as pd

from benchmark.temporal_experiment import (
    compute_temporal_statistics,
    run_temporal_controlled_benchmark,
)


def test_temporal_controlled_benchmark_reports_reference_gaps(tmp_path):
    result = run_temporal_controlled_benchmark(
        seeds=(0,),
        num_episodes=4,
        iterations=1,
        output_path=tmp_path / "temporal.csv",
        statistics_output_path=tmp_path / "temporal_statistics.csv",
    )

    assert len(result) == 16
    assert result["configuration"].nunique() == 4
    assert result["reference_energy"].notna().all()
    assert result["absolute_gap"].notna().all()
    assert result["internal_energy"].notna().all()
    statistics = compute_temporal_statistics(result)
    assert len(statistics) == 6
    assert (tmp_path / "temporal_statistics.csv").exists()
    assert {"paired_t_p_value", "wilcoxon_p_value", "cohen_dz"}.issubset(
        statistics.columns
    )


def test_temporal_statistics_detects_internal_external_tradeoff():
    rows = []
    for seed in (1, 2, 3):
        rows.extend(
            [
                {
                    "seed": seed,
                    "configuration": "Static Energy",
                    "internal_energy": 10.0,
                    "absolute_gap": 1.0,
                },
                {
                    "seed": seed,
                    "configuration": "Full EBMAO",
                    "internal_energy": 8.0,
                    "absolute_gap": 2.0,
                },
            ]
        )

    statistics = compute_temporal_statistics(pd.DataFrame(rows))
    tradeoff = statistics[
        (statistics["configuration"] == "Full EBMAO")
        & (statistics["metric"] == "absolute_gap")
    ].iloc[0]
    assert tradeoff["tradeoff_supported"]
    assert tradeoff["mean_delta"] == 1.0
