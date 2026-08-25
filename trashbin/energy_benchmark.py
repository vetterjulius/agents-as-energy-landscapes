from __future__ import annotations

from typing import List

from benchmark.scenarios.base import ProblemInstance, Scenario
from orchestrator.energy_orchestrator import EnergyOrchestrator
from orchestrator.ebmao_orchestrator import EBMAOOrchestrator


class EnergyBenchmark:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(
        self,
        scenarios: List[Scenario],
        seeds: List[int],
    ):
        results = []

        for scenario in scenarios:
            for seed in seeds:
                problem: ProblemInstance = scenario.generate(seed)

                energy = EnergyOrchestrator(self.cfg)
                ebmao = EBMAOOrchestrator(self.cfg)

                X_energy = energy.solve(problem)
                X_ebmao = ebmao.solve(problem)

                results.append(
                    {
                        "scenario": scenario.__class__.__name__,
                        "seed": seed,
                        "energy_X": X_energy,
                        "ebmao_X": X_ebmao,
                        "energy_total": float(
                            energy.total_energy().item()
                        ),
                        "ebmao_total": float(
                            ebmao.total_energy().item()
                        ),
                    }
                )

        return results