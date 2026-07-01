from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from benchmark.agentbench.adapter import AgentBenchAdapter, BenchmarkTask
from baselines.capability_matching import CapabilityMatchingOrchestrator
from baselines.random_baseline import RandomOrchestrator
from orchestrator.base import Agent, Assignment, Task
from orchestrator.executor import Executor
from orchestrator.llm_planner import LLMPlannerOrchestrator
from orchestrator.energy_orchestrator import EnergyOrchestrator
from orchestrator.task_decomposer import TaskDecomposer


class PublicBenchmarkRunner:
    def __init__(self, adapter: AgentBenchAdapter | None = None):
        self.adapter = adapter or AgentBenchAdapter()
        self.decomposer = TaskDecomposer()

    def run(self, orchestrator_name: str | None = None, orchestrator_names: List[str] | None = None, export_dir: str | None = None) -> List[Dict[str, object]]:
        results = []
        tasks = self.adapter.load_tasks()
        orchestrators = orchestrator_names or ([orchestrator_name] if orchestrator_name else ["energy"])
        export_path = Path(export_dir) if export_dir else None
        if export_path is not None:
            export_path.mkdir(parents=True, exist_ok=True)

        for benchmark_task in tasks:
            subtasks = self.decomposer.decompose(benchmark_task)
            agents = self._build_agents()
            for name in orchestrators:
                orchestrator = self._build_orchestrator(name)
                assignment = orchestrator.solve(subtasks, agents)
                executor = Executor(benchmark_task)
                outputs, metadata = executor.execute(assignment, subtasks, agents)
                results.append({
                    "benchmark_task": benchmark_task.id,
                    "orchestrator_name": name,
                    "assignment": assignment,
                    "outputs": outputs,
                    "metrics": {
                        **metadata,
                        "orchestrator_name": name,
                        "num_subtasks": len(subtasks),
                        "num_agents": len(agents),
                    },
                })

        if export_path is not None:
            self._export_results(results, export_path)
        return results

    def _build_orchestrator(self, orchestrator_name: str):
        cfg = {
            "model": {
                "num_agents": 4,
                "num_tasks": 4,
                "dim": 8,
                "lambda_align": 0.5,
                "eta_theta": 0.1,
                "eta_memory": 0.05,
                "temperature_init": 4.0,
                "min_temperature": 1.0,
                "max_temperature": 6.0,
                "proposal_candidates": 4,
                "proposal_task_sample": 2,
                "agent_sample_size": 2,
                "block_move_size": 2,
                "warm_start_steps": 0,
                "warm_start_type": "greedy",
                "hybrid_cleanup_prob": 0.0,
                "local_refine_steps": 0,
                "target_accept_rate": 0.3,
                "risk_weight": 1.0,
                "risk_scale": 1.0,
                "interaction_weight": 1.0,
                "cost_weight": 1.0,
            }
        }
        if orchestrator_name == "llm":
            return LLMPlannerOrchestrator(cfg={})
        if orchestrator_name == "random":
            return RandomOrchestrator(cfg)
        if orchestrator_name == "capability":
            return CapabilityMatchingOrchestrator(cfg)
        if orchestrator_name == "energy":
            return EnergyOrchestrator(cfg)
        return EnergyOrchestrator(cfg)

    def _build_agents(self) -> List[Agent]:
        return [
            Agent(id="planner", role="Planner", capability_embedding=[1.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Agent(id="sql-expert", role="SQL Expert", capability_embedding=[0.8, 0.9, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0]),
            Agent(id="verifier", role="Verifier", capability_embedding=[0.6, 0.2, 0.8, 0.7, 0.1, 0.0, 0.0, 0.0]),
            Agent(id="debugger", role="Debugger", capability_embedding=[0.7, 0.4, 0.3, 0.9, 0.2, 0.0, 0.0, 0.0]),
        ]

    @staticmethod
    def _export_results(results: List[Dict[str, object]], export_dir: Path) -> None:
        payload_path = export_dir / "benchmark_results.json"
        payload_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

        csv_path = export_dir / "benchmark_results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["benchmark_task", "orchestrator_name", "num_subtasks", "num_agents"])
            writer.writeheader()
            for item in results:
                writer.writerow({
                    "benchmark_task": item.get("benchmark_task"),
                    "orchestrator_name": item.get("orchestrator_name"),
                    "num_subtasks": item.get("metrics", {}).get("num_subtasks"),
                    "num_agents": item.get("metrics", {}).get("num_agents"),
                })
