import numpy as np

from benchmark.controlled_experiment import run_controlled_benchmark


def test_controlled_benchmark_reports_exact_gaps(tmp_path):
    result = run_controlled_benchmark(
        seeds=(0,),
        iterations=2,
        output_path=tmp_path / "controlled_test.csv",
    )

    assert set(result["scenario"]) == {
        "Independent",
        "Interaction",
        "Frustrated",
    }
    assert result["method"].nunique() == 6
    assert result["optimum"].notna().all()
    assert np.allclose(
        result["absolute_gap"],
        result["energy"] - result["optimum"],
    )