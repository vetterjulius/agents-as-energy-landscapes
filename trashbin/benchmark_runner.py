from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from benchmark.agentbench.adapter import AgentBenchAdapter
from trashbin.capability_matching import CapabilityMatchingOrchestrator
from baselines.random_baseline import RandomOrchestrator
from orchestrator.base import Agent
from orchestrator.executor import Executor
from trashbin.llm_planner import LLMPlannerOrchestrator
from orchestrator.energy_orchestrator import EnergyOrchestrator
from orchestrator.ebmao_orchestrator import EBMAOOrchestrator
from orchestrator.task_decomposer import TaskDecomposer


class PublicBenchmarkRunner:
    def __init__(
        self,
        adapter: AgentBenchAdapter | None = None,
        config_path: str | Path = "config.yaml",
    ):
        self.adapter = adapter or AgentBenchAdapter()
        self.config_path = Path(config_path)
        self.cfg = self._load_config()

        model_cfg = self.cfg.get("model", {})
        self.dim = int(model_cfg.get("dim", 8))

        self.decomposer = TaskDecomposer(dim=self.dim)

    def _load_config(self) -> Dict[str, object]:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Benchmark config not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}

        if "model" not in cfg:
            raise ValueError(
                f"Benchmark config {self.config_path} "
                "does not contain a 'model' section."
            )

        return cfg

    def run(
        self,
        orchestrator_name: str | None = None,
        orchestrator_names: List[str] | None = None,
        export_dir: str | None = None,
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []

        benchmark_tasks = self.adapter.load_tasks()

        orchestrators = (
            orchestrator_names
            or ([orchestrator_name] if orchestrator_name else ["energy"])
        )

        export_path = Path(export_dir) if export_dir else None
        if export_path is not None:
            export_path.mkdir(parents=True, exist_ok=True)

        for benchmark_task in benchmark_tasks:
            subtasks = self.decomposer.decompose(benchmark_task)
            agents = self._build_agents()

            for name in orchestrators:
                orchestrator = self._build_orchestrator(name)

                assignment = orchestrator.solve(
                    subtasks,
                    agents,
                )

                executor = Executor(benchmark_task)

                outputs, metadata = executor.execute(
                    assignment,
                    subtasks,
                    agents,
                )

                metrics = {
                    **metadata,
                    "orchestrator_name": name,
                    "num_subtasks": len(subtasks),
                    "num_agents": len(agents),
                }

                total_energy = getattr(
                    orchestrator,
                    "total_energy",
                    None,
                )

                if callable(total_energy):
                    try:
                        energy = total_energy()
                        metrics["total_energy"] = float(
                            energy.item()
                        )
                    except (
                        AttributeError,
                        RuntimeError,
                        ValueError,
                    ):
                        pass

                results.append(
                    {
                        "benchmark_task": benchmark_task.id,
                        "orchestrator_name": name,
                        "assignment": assignment,
                        "outputs": outputs,
                        "metrics": metrics,
                    }
                )

        if export_path is not None:
            self._export_results(
                results,
                export_path,
            )

        return results

    def _build_orchestrator(self, orchestrator_name: str):
        model_cfg = dict(
            self.cfg.get("model", {})
        )

        # The actual benchmark controls the number of agents.
        model_cfg["num_agents"] = len(
            self._build_agents()
        )

        cfg = {
            "model": model_cfg,
            "training": dict(
                self.cfg.get("training", {})
            ),
            "sweep": dict(
                self.cfg.get("sweep", {})
            ),
        }

        if orchestrator_name == "llm":
            return LLMPlannerOrchestrator(cfg={})

        if orchestrator_name == "random":
            return RandomOrchestrator(cfg)

        if orchestrator_name == "capability":
            return CapabilityMatchingOrchestrator(cfg)

        if orchestrator_name == "energy":
            return EnergyOrchestrator(cfg)

        if orchestrator_name == "ebmao":
            return EBMAOOrchestrator(cfg)

        raise ValueError(
            f"Unknown orchestrator: {orchestrator_name!r}. "
            "Expected one of: "
            "llm, random, capability, energy, ebmao."
        )

    def _build_agents(self) -> List[Agent]:
        def embedding(values: List[float]) -> torch.Tensor:
            if len(values) > self.dim:
                raise ValueError(
                    "Agent embedding is larger than configured "
                    f"dimension {self.dim}."
                )

            result = torch.zeros(
                self.dim,
                dtype=torch.float32,
            )

            result[: len(values)] = torch.tensor(
                values,
                dtype=torch.float32,
            )

            return result

        return [
            Agent(
                id="planner",
                role="Planner",
                capability_embedding=embedding(
                    [
                        1.0,
                        0.2,
                        0.1,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
            Agent(
                id="sql-expert",
                role="SQL Expert",
                capability_embedding=embedding(
                    [
                        0.8,
                        0.9,
                        0.7,
                        0.2,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
            Agent(
                id="verifier",
                role="Verifier",
                capability_embedding=embedding(
                    [
                        0.6,
                        0.2,
                        0.8,
                        0.7,
                        0.1,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
            Agent(
                id="debugger",
                role="Debugger",
                capability_embedding=embedding(
                    [
                        0.7,
                        0.4,
                        0.3,
                        0.9,
                        0.2,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
        ]

    @staticmethod
    def _export_results(
        results: List[Dict[str, object]],
        export_dir: Path,
    ) -> None:
        payload_path = export_dir / "benchmark_results.json"

        payload_path.write_text(
            json.dumps(
                results,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        csv_path = export_dir / "benchmark_results.csv"

        fieldnames = [
            "benchmark_task",
            "orchestrator_name",
            "num_subtasks",
            "num_agents",
            "total_energy",
        ]

        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for item in results:
                metrics = item.get(
                    "metrics",
                    {},
                )

                writer.writerow(
                    {
                        "benchmark_task": item.get(
                            "benchmark_task"
                        ),
                        "orchestrator_name": item.get(
                            "orchestrator_name"
                        ),
                        "num_subtasks": metrics.get(
                            "num_subtasks"
                        ),
                        "num_agents": metrics.get(
                            "num_agents"
                        ),
                        "total_energy": metrics.get(
                            "total_energy"
                        ),
                    }
                )