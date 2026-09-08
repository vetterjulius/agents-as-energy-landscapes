
# Legacy & Regression Benchmark Suite

> **NOTICE: FROZEN BENCHMARK SUITE (LEGACY / REGRESSION ONLY)**
> 
> The benchmark runners and experiment scripts in this directory
> (`runner.py`, `dynamic_benchmark.py`, `controlled_experiment.py`, `temporal_experiment.py`,
> `regime_experiment.py`, `recurrent_advantage_experiment.py`, `scale_sweep.py`, `coupling_sweep.py`, etc.)
> are permanently **FROZEN** for legacy and regression testing purposes only.
> 
> Their scientific semantics, energy definitions, and existing historical result artifacts are preserved
> without alteration to avoid breaking backwards compatibility or regression tests.
> 
> **Primary Controlled Benchmark:**
> For the scientifically controlled benchmark addressing the core research question:
> *"Does an adaptive energy landscape improve multi-agent orchestration under non-stationarity, and is the effect independent of the optimization solver?"*
> see:
> - Entrypoint: `controlled_landscape_solver_benchmark.py`
> - Package: `controlled_benchmark/`
> - Results: `results/controlled_landscape_solver/`
