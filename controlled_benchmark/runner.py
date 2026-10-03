from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.metrics import (
    EpisodeRecord,
    TrajectorySummary,
    compute_co_assignment_conflicts,
    compute_coordination_score,
    compute_load_balance,
    compute_reconfig_cost,
    compute_trajectory_summary,
)
from controlled_benchmark.scenarios import (
    generate_scenario_trajectory,
    make_initial_landscape_state,
    problem_instance_to_problem_context,
)
from controlled_benchmark.solvers import (
    BudgetedLandscape,
    ConventionalGreedySolver,
    EnergyAwareGreedySolver,
    EnergyAwareSimulatedAnnealingSolver,
    FixedLandscapeILPSolver,
    SolverResult,
)
from controlled_benchmark.statistics import (
    analyze_paired_comparison,
    analyze_solver_interaction,
    holm_bonferroni_correction,
)
from landscape import Landscape, LandscapeState


def _get_git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return bool(output)
    except Exception:
        return False


def _trajectory_fingerprint(trajectory: List[Any]) -> str:
    digest = hashlib.sha256()
    for problem in trajectory:
        for agent in problem.agents:
            digest.update(agent.capability_embedding.detach().cpu().numpy().tobytes())
        for task in problem.tasks:
            digest.update(task.embedding.detach().cpu().numpy().tobytes())
        for tensor in (
            problem.interaction_graph,
            problem.co_assignment_costs,
            problem.risk_weights,
        ):
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def _observed_cooccurrence(X: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Reproduce the adaptation module's normalized, symmetric off-diagonal C_t."""
    co = X.T @ X
    co_sum = co.sum().item()
    if co_sum < epsilon:
        return torch.zeros_like(co)
    co_norm = co / (co_sum + epsilon)
    result = (co_norm + co_norm.T) / 2.0
    result.fill_diagonal_(0.0)
    return result


def _validate_diagnostic_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Mechanism diagnostic {name} contains NaN or Inf values")


def _json_safe(value: Any) -> Any:
    """Convert scalar numpy values in trace records to standard JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class _IncrementalDiagnosticWriter:
    """Persist mechanism diagnostics in bounded per-run chunks.

    NPZ archives are not appendable.  Each completed run is therefore written to
    its own compressed chunk immediately; the manifest is updated atomically so
    an interrupted benchmark still leaves readable completed-run diagnostics.
    """

    def __init__(self, output_dir: str, verbose: bool, expected_runs: int):
        self.output_dir = Path(output_dir) / "mechanism_diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.expected_runs = expected_runs
        self.chunks: List[Dict[str, Any]] = []
        self.manifest_path = self.output_dir.parent / "mechanism_diagnostics.json"
        self._write_manifest()

    @staticmethod
    def _stack(diagnostics: List[Dict[str, Any]], name: str) -> np.ndarray:
        return np.stack([item[name] for item in diagnostics])

    def write_run(self, diagnostics: List[Dict[str, Any]], run_metadata: Dict[str, Any]) -> None:
        if not diagnostics:
            return
        run_index = int(run_metadata["run_index"])
        payload: Dict[str, np.ndarray] = {
            name: self._stack(diagnostics, name)
            for name in (
                "assignment_matrix", "cooccurrence_matrix", "theta_before",
                "theta_after", "ground_truth_dependency",
            )
        }
        payload["episode"] = np.asarray([item["episode"] for item in diagnostics], dtype=np.int64)
        tensor_fields: List[str] = []
        breakdown_fields: List[str] = []
        if self.verbose:
            tensor_fields = [
                "initial_assignment", "agent_capabilities", "task_embeddings",
                "co_assignment_costs", "risk_weights", "kappa_before", "kappa_after",
                "adaptation_risk_probabilities", "adaptation_kappa_target",
                "adaptation_theta_observation",
            ]
            for field in tensor_fields:
                payload[field] = self._stack(diagnostics, field)
            breakdown_fields = sorted(diagnostics[0]["internal_energy_breakdown"])
            for key in breakdown_fields:
                payload[f"internal_energy_{key}"] = np.asarray(
                    [item["internal_energy_breakdown"][key] for item in diagnostics], dtype=np.float64
                )
                payload[f"external_energy_{key}"] = np.asarray(
                    [item["external_energy_breakdown"][key] for item in diagnostics], dtype=np.float64
                )

        trace_events: List[Dict[str, Any]] = []
        trace_assignments: List[np.ndarray] = []
        if self.verbose:
            for item in diagnostics:
                for event_index, event in enumerate(item.get("solver_trace", [])):
                    trace_assignments.append(event["X"].detach().cpu().numpy())
                    trace_events.append({
                        "run_index": run_index,
                        "episode": int(item["episode"]),
                        "event_index": event_index,
                        **{key: _json_safe(value) for key, value in event.items() if key != "X"},
                    })
            if trace_assignments:
                payload["trace_assignment_matrix"] = np.stack(trace_assignments)
                payload["trace_event_index"] = np.asarray([e["event_index"] for e in trace_events], dtype=np.int64)
                payload["trace_run_index"] = np.full(len(trace_events), run_index, dtype=np.int64)
                payload["trace_episode"] = np.asarray([e["episode"] for e in trace_events], dtype=np.int64)

        filename = f"chunk_{run_index:06d}.npz"
        np.savez_compressed(self.output_dir / filename, **payload)
        self.chunks.append({
            "run_index": run_index,
            "file": f"mechanism_diagnostics/{filename}",
            "run_metadata": run_metadata,
            "episode_metadata": [
                {"run_index": run_index, "episode": int(item["episode"]), **item["episode_metadata"]}
                for item in diagnostics
            ] if self.verbose else [],
            "solver_trace_events": trace_events,
            "tensor_fields": list(payload),
        })
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "format_version": "2.0-chunked",
            "tensor_order": ["episode", "row", "column"],
            "verbose_diagnostics": self.verbose,
            "timing": "X_t and C_t belong to labeled episode t; theta_before is Theta_t; theta_after is Theta_{t+1}. The update from t affects only t+1.",
            "chunks_directory": "mechanism_diagnostics",
            "expected_runs": self.expected_runs,
            "completed_runs": len(self.chunks),
            "chunks": self.chunks,
            "assignment_matrix": "X_t, the solver assignment selected during labeled episode t",
            "cooccurrence_matrix": "C_t, computed from X_t before the boundary update",
            "theta_before": "Theta_t, the state used to solve episode t",
            "theta_after": "Theta_{t+1}, produced from X_t and used to solve episode t+1",
            "ground_truth_dependency": "Episode-t interaction_graph for structural comparison",
            "solver_trace": "trace_assignment_matrix and solver_trace_events are stored per chunk; event order is local to the episode",
        }
        temp = self.manifest_path.with_suffix(".json.tmp")
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(temp, self.manifest_path)


class ControlledBenchmarkRunner:
    """Runner for the controlled landscape/solver benchmark."""

    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.sa_solver = EnergyAwareSimulatedAnnealingSolver(
            temperature_init=config.sa_temperature_init,
            min_temperature=config.sa_min_temperature,
            cooling_rate=config.sa_cooling_rate,
        )
        self.greedy_solver = EnergyAwareGreedySolver()
        self.conventional_greedy_solver = ConventionalGreedySolver()
        self.ilp_solver = FixedLandscapeILPSolver(time_limit_sec=config.ilp_time_limit_sec)
        # Populated by run_trajectory; kept separate from the tabular episode log.
        self.last_mechanism_diagnostics: List[Dict[str, Any]] = []

    def _get_initial_assignment(self, N: int, M: int) -> torch.Tensor:
        X = torch.zeros(N, M, dtype=torch.float32)
        for task in range(M):
            X[task % N, task] = 1.0
        return X

    def _ground_truth_reference(
        self, problem_inst: Any
    ) -> Tuple[float, str, bool, bool]:
        context = problem_instance_to_problem_context(
            problem_inst,
            lambda_align=self.cfg.lambda_align,
            lambda_memory=self.cfg.lambda_memory,
            interaction_weight=self.cfg.interaction_weight,
            cost_weight=self.cfg.cost_weight,
            risk_weight=self.cfg.risk_weight,
        )
        state = LandscapeState(
            kappa=torch.zeros(self.cfg.num_agents, self.cfg.dim),
            Theta=problem_inst.interaction_graph.clone(),
        )
        result = self.ilp_solver.solve(Landscape(context, state))
        method = "exact_optimal" if result.is_optimal is True else "ilp_timeout_or_failure"
        return float(result.energy), method, bool(result.is_optimal is True), bool(result.timeout)

    def run_trajectory(
        self,
        scenario_id: str,
        seed: int,
        solver_id: str,
        landscape_id: str,
        adaptation_mode: str,
        trajectory: List[Any],
        reference_energy: Optional[float] = None,
        reference_method: Optional[str] = None,
        reference_valid: Optional[bool] = None,
    ) -> Tuple[TrajectorySummary, List[EpisodeRecord]]:
        """Run one condition; adaptation at t only creates state for t+1."""
        N, M, d = self.cfg.num_agents, self.cfg.num_tasks, self.cfg.dim
        manager = EpisodeAdaptationManager(
            adaptation_mode=adaptation_mode,
            eta_memory=self.cfg.eta_memory,
            eta_theta=self.cfg.eta_theta,
        )
        current_state = make_initial_landscape_state(trajectory[0])
        records: List[EpisodeRecord] = []
        mechanism_diagnostics: List[Dict[str, Any]] = []
        previous_X: Optional[torch.Tensor] = None
        previous_state: Optional[LandscapeState] = None
        state_snapshots: Dict[int, LandscapeState] = {}
        snapshot_start = self.cfg.perturb_episode
        snapshot_end = min(
            self.cfg.perturb_episode + self.cfg.post_ppee_window,
            len(trajectory) - 1,
        )
        snapshot_episodes = {snapshot_start, snapshot_end}

        for episode, problem_inst in enumerate(trajectory):
            context = problem_instance_to_problem_context(
                problem_inst,
                lambda_align=self.cfg.lambda_align,
                lambda_memory=self.cfg.lambda_memory,
                interaction_weight=self.cfg.interaction_weight,
                cost_weight=self.cfg.cost_weight,
                risk_weight=self.cfg.risk_weight,
            )
            solver_landscape = Landscape(context, current_state)
            ground_truth = Landscape(
                context,
                LandscapeState(
                    kappa=torch.zeros(N, d),
                    Theta=problem_inst.interaction_graph.clone(),
                ),
            )
            budgeted = BudgetedLandscape(
                solver_landscape, max_evaluations=self.cfg.max_energy_evaluations
            )
            initial_X = self._get_initial_assignment(N, M)
            solver_trace: Optional[List[Dict[str, Any]]] = [] if self.cfg.verbose_diagnostics else None

            if solver_id in ("Conventional Greedy", "conventional_greedy"):
                solver_result = self.conventional_greedy_solver.solve(
                    context, trace=solver_trace
                )
            elif solver_id == "Simulated Annealing":
                solver_result = self.sa_solver.solve(
                    budgeted,
                    initial_X,
                    seed=seed * 10000 + episode,
                    trace=solver_trace,
                )
            elif solver_id == "Energy Greedy":
                solver_result = self.greedy_solver.solve(
                    budgeted, initial_X, trace=solver_trace
                )
            elif solver_id == "Exact ILP":
                solver_result = self.ilp_solver.solve(solver_landscape)
            else:
                raise ValueError(f"Unknown solver_id: {solver_id}")

            X = solver_result.X.clone()
            if solver_trace is not None and not solver_trace:
                solver_trace.append(
                    {
                        "event_type": "final_solution",
                        "X": X.clone(),
                        "energy": float(solver_result.energy),
                        "termination_reason": solver_result.termination_reason,
                        "iterations": int(solver_result.iterations),
                    }
                )
            theta_before = current_state.Theta.clone()
            kappa_before = current_state.kappa.clone()
            C_t = _observed_cooccurrence(X)
            internal_energy = solver_landscape.evaluate(X)
            external_energy = ground_truth.evaluate(X)
            if self.cfg.verbose_diagnostics:
                internal_breakdown = solver_landscape.breakdown(X)
                external_breakdown = ground_truth.breakdown(X)
            if solver_id in ("Conventional Greedy", "conventional_greedy"):
                kappa_norm = float("nan")
                theta_norm = float("nan")
                theta_diff_norm = float("nan")
                delta_kappa = float("nan")
                delta_theta = float("nan")
            else:
                kappa_norm = float(current_state.kappa.norm().item())
                theta_norm = float(current_state.Theta.norm().item())
                theta_diff_norm = float(
                    (current_state.Theta - problem_inst.interaction_graph).norm().item()
                )
                if previous_state is None:
                    delta_kappa = 0.0
                    delta_theta = 0.0
                else:
                    delta_kappa = float((current_state.kappa - previous_state.kappa).norm().item())
                    delta_theta = float((current_state.Theta - previous_state.Theta).norm().item())
            if episode in snapshot_episodes:
                state_snapshots[episode] = current_state.clone()

            records.append(
                EpisodeRecord(
                    episode=episode,
                    internal_energy=internal_energy,
                    external_energy=external_energy,
                    energy_evaluations=solver_result.energy_evaluations,
                    accepted_moves=solver_result.accepted_moves,
                    iterations=solver_result.iterations,
                    runtime_sec=solver_result.runtime_sec,
                    termination_reason=solver_result.termination_reason,
                    solver_status=solver_result.status,
                    is_optimal=solver_result.is_optimal,
                    mip_gap=solver_result.mip_gap,
                    timeout=solver_result.timeout,
                    fallback_used=solver_result.fallback_used,
                    reconfig_cost=compute_reconfig_cost(previous_X, X),
                    constraint_violations=compute_co_assignment_conflicts(context.C, X),
                    coordination_score=compute_coordination_score(
                        problem_inst.interaction_graph, X
                    ),
                    load_balance=compute_load_balance(X),
                    kappa_norm=kappa_norm,
                    theta_diff_norm=theta_diff_norm,
                    delta_kappa_norm=delta_kappa,
                    delta_theta_norm=delta_theta,
                    theta_norm=theta_norm,
                )
            )

            # The observation from episode t is applied only after all episode-t metrics.
            # theta_before is the state used to solve episode t; theta_after is the state
            # produced by X_t and is the state used to solve episode t+1.
            previous_state = current_state.clone()
            adaptation_trace: Optional[Dict[str, Any]] = {} if self.cfg.verbose_diagnostics else None
            if solver_id in ("Conventional Greedy", "conventional_greedy"):
                theta_after = current_state.clone()
                if adaptation_trace is not None:
                    adaptation_trace.update({
                        "kappa_update_active": False,
                        "theta_update_active": False,
                        "risk_probabilities": torch.zeros(N, M),
                        "kappa_target": torch.zeros(N, d),
                        "theta_observation": torch.zeros(M, M),
                    })
            else:
                theta_after = manager.step(
                    current_state, X, context, diagnostics=adaptation_trace
                )
            for name, value in (
                ("assignment_matrix", X),
                ("cooccurrence_matrix", C_t),
                ("theta_before", theta_before),
                ("theta_after", theta_after.Theta),
                ("ground_truth_dependency", problem_inst.interaction_graph),
            ):
                _validate_diagnostic_tensor(name, value)
            diagnostic_item = {
                "episode": episode,
                "assignment_matrix": X.detach().cpu().numpy().copy(),
                "cooccurrence_matrix": C_t.detach().cpu().numpy().copy(),
                "theta_before": theta_before.detach().cpu().numpy().copy(),
                "theta_after": theta_after.Theta.detach().cpu().numpy().copy(),
                "ground_truth_dependency": problem_inst.interaction_graph.detach().cpu().numpy().copy(),
            }
            if self.cfg.verbose_diagnostics:
                verbose_tensors = {
                    "initial_assignment": initial_X,
                    "agent_capabilities": context.s,
                    "task_embeddings": context.c,
                    "co_assignment_costs": context.C,
                    "risk_weights": context.W_risk,
                    "kappa_before": kappa_before,
                    "kappa_after": theta_after.kappa,
                    "adaptation_risk_probabilities": adaptation_trace["risk_probabilities"],
                    "adaptation_kappa_target": adaptation_trace["kappa_target"],
                    "adaptation_theta_observation": adaptation_trace["theta_observation"],
                }
                for name, value in verbose_tensors.items():
                    _validate_diagnostic_tensor(name, value)
                    diagnostic_item[name] = value.detach().cpu().numpy().copy()
                diagnostic_item["internal_energy_breakdown"] = {
                    key: float(value) for key, value in internal_breakdown.items()
                }
                diagnostic_item["external_energy_breakdown"] = {
                    key: float(value) for key, value in external_breakdown.items()
                }
                diagnostic_item["episode_metadata"] = {
                    "solver_seed": (seed * 10000 + episode) if solver_id == "Simulated Annealing" else None,
                    "solver_reported_energy": float(solver_result.energy),
                    "internal_energy": internal_energy,
                    "external_energy": external_energy,
                    "energy_evaluations": int(solver_result.energy_evaluations),
                    "accepted_moves": int(solver_result.accepted_moves),
                    "iterations": int(solver_result.iterations),
                    "runtime_sec": float(solver_result.runtime_sec),
                    "termination_reason": solver_result.termination_reason,
                    "solver_status": solver_result.status,
                    "is_optimal": solver_result.is_optimal,
                    "mip_gap": solver_result.mip_gap,
                    "timeout": bool(solver_result.timeout),
                    "fallback_used": bool(solver_result.fallback_used),
                    "reconfig_cost": records[-1].reconfig_cost,
                    "co_assignment_conflicts": records[-1].constraint_violations,
                    "coordination_score": records[-1].coordination_score,
                    "load_balance": records[-1].load_balance,
                    "kappa_norm_before": records[-1].kappa_norm,
                    "theta_norm_before": records[-1].theta_norm,
                    "theta_diff_norm_before": records[-1].theta_diff_norm,
                    "delta_kappa_norm": records[-1].delta_kappa_norm,
                    "delta_theta_norm": records[-1].delta_theta_norm,
                    "kappa_update_active": adaptation_trace["kappa_update_active"],
                    "theta_update_active": adaptation_trace["theta_update_active"],
                }
                diagnostic_item["solver_trace"] = solver_trace or []
            mechanism_diagnostics.append(diagnostic_item)
            current_state = theta_after
            previous_X = X

        if reference_energy is None:
            target = trajectory[min(self.cfg.perturb_episode, len(trajectory) - 1)]
            reference_energy, reference_method, reference_valid, _ = self._ground_truth_reference(target)

        kappa_change = theta_change = None
        if solver_id in ("Conventional Greedy", "conventional_greedy"):
            kappa_change = float("nan")
            theta_change = float("nan")
        elif snapshot_start < len(trajectory) and snapshot_end >= snapshot_start:
            start_state = state_snapshots[snapshot_start]
            end_state = state_snapshots[snapshot_end]
            kappa_change = float((end_state.kappa - start_state.kappa).norm().item())
            theta_change = float((end_state.Theta - start_state.Theta).norm().item())

        self.last_mechanism_diagnostics = mechanism_diagnostics
        summary = compute_trajectory_summary(
            records=records,
            scenario_id=scenario_id,
            perturb_episode=self.cfg.perturb_episode,
            pre_window=self.cfg.pre_window,
            post_window=self.cfg.post_window,
            post_ppee_window=self.cfg.post_ppee_window,
            reference_energy=reference_energy,
            reference_method=reference_method,
            kappa_change_post=kappa_change,
            theta_change_post=theta_change,
        )
        return summary, records

    def _cells(self) -> List[Tuple[str, str, str, str]]:
        cells = [
            ("conventional_greedy", "Conventional Greedy", "Conventional", "static"),
            ("static_energy_greedy", "Energy Greedy", "Static", "static"),
            ("static_energy_sa", "Simulated Annealing", "Static", "static"),
            ("adaptive_energy_greedy", "Energy Greedy", "Adaptive Full", "full"),
            ("adaptive_energy_sa", "Simulated Annealing", "Adaptive Full", "full"),
        ]
        if self.cfg.run_ilp_dynamic:
            cells.extend(
                [
                    ("exact_ilp_static", "Exact ILP", "Static", "static"),
                    ("exact_ilp_full", "Exact ILP", "Full", "full"),
                ]
            )
        return cells

    @staticmethod
    def _save_mechanism_diagnostics(
        output_dir: str,
        diagnostics: List[Dict[str, Any]],
        run_metadata: List[Dict[str, Any]],
        verbose: bool = False,
    ) -> None:
        """Write dense per-episode mechanism tensors plus a JSON-readable manifest."""
        if not diagnostics:
            return
        os.makedirs(output_dir, exist_ok=True)
        run_count = len(run_metadata)
        episode_counts = [int(item["episode_count"]) for item in run_metadata]
        if len(set(episode_counts)) != 1 or sum(episode_counts) != len(diagnostics):
            raise RuntimeError("Mechanism diagnostic runs have inconsistent episode counts")
        episode_count = episode_counts[0]

        def _stack(name: str) -> np.ndarray:
            values = np.stack([item[name] for item in diagnostics])
            return values.reshape((run_count, episode_count) + values.shape[1:])

        payload: Dict[str, np.ndarray] = {
            "assignment_matrix": _stack("assignment_matrix"),
            "cooccurrence_matrix": _stack("cooccurrence_matrix"),
            "theta_before": _stack("theta_before"),
            "theta_after": _stack("theta_after"),
            "ground_truth_dependency": _stack("ground_truth_dependency"),
            "episode": np.asarray(
                [item["episode"] for item in diagnostics], dtype=np.int64
            ).reshape(run_count, episode_count),
        }
        verbose_tensor_fields: List[str] = []
        if verbose:
            verbose_tensor_fields = [
                "initial_assignment",
                "agent_capabilities",
                "task_embeddings",
                "co_assignment_costs",
                "risk_weights",
                "kappa_before",
                "kappa_after",
                "adaptation_risk_probabilities",
                "adaptation_kappa_target",
                "adaptation_theta_observation",
            ]
            for field_name in verbose_tensor_fields:
                payload[field_name] = _stack(field_name)
            breakdown_keys = sorted(diagnostics[0]["internal_energy_breakdown"])
            for key in breakdown_keys:
                payload[f"internal_energy_{key}"] = np.asarray(
                    [item["internal_energy_breakdown"][key] for item in diagnostics],
                    dtype=np.float64,
                ).reshape(run_count, episode_count)
                payload[f"external_energy_{key}"] = np.asarray(
                    [item["external_energy_breakdown"][key] for item in diagnostics],
                    dtype=np.float64,
                ).reshape(run_count, episode_count)
        trace_events: List[Dict[str, Any]] = []
        trace_assignments: List[np.ndarray] = []
        if verbose:
            for run_index, item in enumerate(diagnostics):
                for event_index, event in enumerate(item.get("solver_trace", [])):
                    trace_assignments.append(event["X"].detach().cpu().numpy().copy())
                    trace_events.append(
                        {
                            "run_index": run_index,
                            "episode": int(item["episode"]),
                            "event_index": event_index,
                            **{
                                key: value
                                for key, value in event.items()
                                if key != "X"
                            },
                        }
                    )
            if trace_assignments:
                payload["trace_assignment_matrix"] = np.stack(trace_assignments)
                payload["trace_event_index"] = np.asarray(
                    [event["event_index"] for event in trace_events], dtype=np.int64
                )
                payload["trace_run_index"] = np.asarray(
                    [event["run_index"] for event in trace_events], dtype=np.int64
                )
                payload["trace_episode"] = np.asarray(
                    [event["episode"] for event in trace_events], dtype=np.int64
                )
        np.savez_compressed(
            os.path.join(output_dir, "mechanism_diagnostics.npz"), **payload
        )
        with open(
            os.path.join(output_dir, "mechanism_diagnostics.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "format_version": "1.0",
                    "tensor_file": "mechanism_diagnostics.npz",
                    "tensor_order": ["run", "episode", "row", "column"],
                    "assignment_matrix": "X_t, the solver assignment selected during labeled episode t",
                    "cooccurrence_matrix": "C_t, computed from X_t before the boundary update",
                    "theta_before": "Theta_t, the state used to solve episode t",
                    "theta_after": "Theta_{t+1}, produced from X_t and used to solve episode t+1",
                    "ground_truth_dependency": "Episode-t interaction_graph for structural comparison",
                    "timing": "The update from episode t is applied only after all episode-t metrics and affects episode t+1.",
                    "verbose_diagnostics": verbose,
                    "verbose_tensor_fields": verbose_tensor_fields,
                    "energy_breakdown_fields": (
                        sorted(diagnostics[0]["internal_energy_breakdown"])
                        if verbose else []
                    ),
                    "solver_trace": "trace_assignment_matrix plus solver_trace_events; one event per evaluated/proposed candidate and final solution",
                    "solver_trace_tensor_order": ["event", "row", "column"],
                    "episode_metadata": (
                        [
                            {
                                "run_index": run_index,
                                "episode": int(item["episode"]),
                                **item["episode_metadata"],
                            }
                            for run_index, item in enumerate(diagnostics)
                        ]
                        if verbose else []
                    ),
                    "solver_trace_event_count": len(trace_events),
                    "solver_trace_events": trace_events,
                    "records": run_metadata,
                },
                handle,
                indent=2,
            )

    def run_benchmark(self) -> Dict[str, Any]:
        started = time.time()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.cfg.save_json(os.path.join(self.cfg.output_dir, "config.json"))
        cells = self._cells()
        expected_runs = len(self.cfg.seeds) * len(self.cfg.scenarios) * len(cells)
        print(
            f"[benchmark] starting: {len(self.cfg.seeds)} seeds × "
            f"{len(self.cfg.scenarios)} scenarios × {len(cells)} conditions "
            f"= {expected_runs} runs",
            flush=True,
        )
        # Persist tabular rows and mechanism chunks as each run completes.  This
        # keeps the peak memory bounded by one trajectory/run rather than the
        # complete verbose benchmark.
        runs_path = os.path.join(self.cfg.output_dir, "runs.csv")
        episodes_path = os.path.join(self.cfg.output_dir, "episodes.csv")
        for path in (runs_path, episodes_path):
            if os.path.exists(path):
                os.remove(path)
        mechanism_run_metadata: List[Dict[str, Any]] = []
        diagnostic_bytes_per_episode = (
            self.cfg.num_agents * self.cfg.num_tasks * 4
            + 4 * self.cfg.num_tasks * self.cfg.num_tasks * 4
        )
        if self.cfg.verbose_diagnostics:
            diagnostic_bytes_per_episode *= 4
            diagnostic_bytes_per_episode += self.cfg.max_energy_evaluations * self.cfg.num_agents * self.cfg.num_tasks * 4
        estimated_diagnostic_bytes = expected_runs * self.cfg.num_episodes * diagnostic_bytes_per_episode
        # Keep the legacy consolidated NPZ only for genuinely small test/quick
        # artifacts.  Larger runs use the chunked writer exclusively.
        legacy_allowed = (
            not self.cfg.verbose_diagnostics
            or expected_runs * self.cfg.num_episodes * self.cfg.max_energy_evaluations <= 10_000
        )
        legacy_diagnostics: Optional[List[Dict[str, Any]]] = (
            [] if legacy_allowed and estimated_diagnostic_bytes <= 16 * 1024 * 1024 else None
        )
        diagnostic_writer = _IncrementalDiagnosticWriter(
            self.cfg.output_dir, self.cfg.verbose_diagnostics, expected_runs
        )
        run_rows_written = episode_rows_written = False

        for scenario_id in self.cfg.scenarios:
            for seed in self.cfg.seeds:
                trajectory = generate_scenario_trajectory(
                    scenario_id=scenario_id,
                    seed=seed,
                    num_episodes=self.cfg.num_episodes,
                    perturb_episode=self.cfg.perturb_episode,
                    N=self.cfg.num_agents,
                    M=self.cfg.num_tasks,
                    d=self.cfg.dim,
                )
                trajectory_id = _trajectory_fingerprint(trajectory)
                target = trajectory[min(self.cfg.perturb_episode, len(trajectory) - 1)]
                reference_energy, reference_method, reference_valid, reference_timeout = (
                    self._ground_truth_reference(target)
                )
                reference_id = hashlib.sha256(
                    f"{scenario_id}|{seed}|{reference_energy:.12g}".encode()
                ).hexdigest()

                print(
                    f"[benchmark] prepared trajectory: scenario={scenario_id} "
                    f"seed={seed} reference={reference_energy:.6f} "
                    f"({reference_method})",
                    flush=True,
                )

                for condition_id, solver_id, landscape_id, adaptation_mode in cells:
                    run_id = "run_{}_{}_{}_{}".format(
                        _slug(scenario_id), seed, _slug(condition_id), self.cfg.experiment_id
                    )
                    summary, episode_records = self.run_trajectory(
                        scenario_id=scenario_id,
                        seed=seed,
                        solver_id=solver_id,
                        landscape_id=landscape_id,
                        adaptation_mode=adaptation_mode,
                        trajectory=trajectory,
                        reference_energy=reference_energy,
                        reference_method=reference_method,
                        reference_valid=reference_valid,
                    )
                    run_mechanism_diagnostics = self.last_mechanism_diagnostics
                    mechanism_run_index = len(mechanism_run_metadata)
                    run_mechanism_metadata = {
                        "run_index": mechanism_run_index,
                        "run_id": run_id,
                        "condition_id": condition_id,
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "solver_id": solver_id,
                        "landscape_id": landscape_id,
                        "adaptation_mode": adaptation_mode,
                        "episode_count": len(run_mechanism_diagnostics),
                    }
                    mechanism_run_metadata.append(run_mechanism_metadata)
                    diagnostic_writer.write_run(run_mechanism_diagnostics, run_mechanism_metadata)
                    if legacy_diagnostics is not None:
                        legacy_diagnostics.extend(run_mechanism_diagnostics)
                    # Release the runner's reference as soon as the completed
                    # run has been serialized; the next run must not retain it.
                    self.last_mechanism_diagnostics = []
                    run = {
                        "run_id": run_id,
                        "condition_id": condition_id,
                        "experiment_id": self.cfg.experiment_id,
                        "problem_id": f"{scenario_id}_N{self.cfg.num_agents}_M{self.cfg.num_tasks}_seed{seed}",
                        "trajectory_id": trajectory_id,
                        "reference_id": reference_id,
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "solver_id": solver_id,
                        "landscape_id": landscape_id,
                        "adaptation_mode": adaptation_mode,
                        "N": self.cfg.num_agents,
                        "M": self.cfg.num_tasks,
                        "d": self.cfg.dim,
                        "num_episodes": self.cfg.num_episodes,
                        "perturb_episode": self.cfg.perturb_episode,
                        "evaluation_budget": self.cfg.max_energy_evaluations,
                        "max_energy_evaluations": self.cfg.max_energy_evaluations,
                        "ppee_10": summary.ppee_10,
                        "cumulative_excess_energy": summary.cumulative_excess_energy,
                        "cumulative_regret": summary.cumulative_regret,
                        "performance_drop": summary.perf_drop,
                        "perf_drop": summary.perf_drop,
                        "recovery_time": summary.recovery_time,
                        "mean_external_energy": summary.mean_external_energy,
                        "mean_internal_energy": summary.mean_internal_energy,
                        "final_energy": summary.final_energy,
                        "pre_base_energy": summary.pre_base_energy,
                        "post_base_energy": summary.post_base_energy,
                        "convergence": summary.convergence,
                        "stability": summary.stability,
                        "kappa_norm": summary.adaptation_magnitude_kappa,
                        "theta_norm": summary.adaptation_magnitude_theta,
                        "adaptation_magnitude_kappa": summary.adaptation_magnitude_kappa,
                        "adaptation_magnitude_theta": summary.adaptation_magnitude_theta,
                        "kappa_change_post": summary.kappa_change_post,
                        "theta_change_post": summary.theta_change_post,
                        "mean_co_assignment_conflicts": summary.mean_co_assignment_conflicts,
                        "mean_constraint_violations": summary.mean_constraint_violations,
                        "total_energy_evaluations": summary.total_energy_evaluations,
                        "total_runtime_sec": summary.total_runtime_sec,
                        "termination_reasons": summary.termination_reasons_summary,
                        "all_episodes_optimal": summary.all_episodes_optimal,
                        "fallback_used": summary.fallback_used,
                        "recovery_threshold": summary.recovery_threshold,
                        "reference_energy": reference_energy,
                        "reference_method": reference_method,
                        "reference_valid": reference_valid,
                        "reference_timeout": reference_timeout,
                        "recovery_window": summary.recovery_window,
                        "evaluation_landscape_id": summary.evaluation_landscape_id,
                        "git_commit": self.cfg.git_commit,
                        "benchmark_version": self.cfg.benchmark_version,
                    }
                    pd.DataFrame([run]).to_csv(
                        runs_path, mode="a", header=not run_rows_written, index=False
                    )
                    run_rows_written = True
                    completed_runs = mechanism_run_index + 1
                    print(
                        f"[benchmark] [{completed_runs}/{expected_runs}] "
                        f"scenario={scenario_id} seed={seed} solver={solver_id} "
                        f"adaptation={adaptation_mode} ppee_10={summary.ppee_10:.6f} "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )
                    episode_rows = []
                    for record in episode_records:
                        episode_rows.append(
                            {
                                "run_id": run_id,
                                "condition_id": condition_id,
                                "experiment_id": self.cfg.experiment_id,
                                "problem_id": run["problem_id"],
                                "trajectory_id": trajectory_id,
                                "reference_id": reference_id,
                                "scenario_id": scenario_id,
                                "seed": seed,
                                "solver_id": solver_id,
                                "landscape_id": landscape_id,
                                "adaptation_mode": adaptation_mode,
                                "N": self.cfg.num_agents,
                                "M": self.cfg.num_tasks,
                                "d": self.cfg.dim,
                                "total_episodes": self.cfg.num_episodes,
                                "perturb_episode": self.cfg.perturb_episode,
                                "episode": record.episode,
                                "evaluation_budget": self.cfg.max_energy_evaluations,
                                "reference_method": reference_method,
                                "reference_energy": reference_energy,
                                "reference_valid": reference_valid,
                                "internal_energy": record.internal_energy,
                                "external_energy": record.external_energy,
                                "energy_evaluations": record.energy_evaluations,
                                "accepted_moves": record.accepted_moves,
                                "iterations": record.iterations,
                                "runtime_sec": record.runtime_sec,
                                "termination_reason": record.termination_reason,
                                "solver_status": record.solver_status,
                                "optimal": record.is_optimal,
                                "mip_gap": record.mip_gap,
                                "timeout": record.timeout,
                                "fallback": record.fallback_used,
                                "reconfig_cost": record.reconfig_cost,
                                "co_assignment_conflicts": record.constraint_violations,
                                "coordination_score": record.coordination_score,
                                "load_balance": record.load_balance,
                                "kappa_norm": record.kappa_norm,
                                "theta_norm": record.theta_norm,
                                "theta_diff_norm": record.theta_diff_norm,
                                "delta_kappa_norm": record.delta_kappa_norm,
                                "delta_theta_norm": record.delta_theta_norm,
                                "git_commit": self.cfg.git_commit,
                            }
                        )
                    if episode_rows:
                        pd.DataFrame(episode_rows).to_csv(
                            episodes_path, mode="a", header=not episode_rows_written, index=False
                        )
                        episode_rows_written = True

        # Small artifacts retain the historical single-NPZ compatibility format;
        # large artifacts remain chunked and never need a full in-memory stack.
        if legacy_diagnostics is not None:
            self._save_mechanism_diagnostics(
                self.cfg.output_dir,
                legacy_diagnostics,
                mechanism_run_metadata,
                verbose=self.cfg.verbose_diagnostics,
            )
        runs_df = pd.read_csv(runs_path)
        episodes_df = pd.read_csv(episodes_path)

        summary_metrics = [
            "ppee_10",
            "cumulative_excess_energy",
            "performance_drop",
            "recovery_time",
            "kappa_change_post",
            "theta_change_post",
            "total_runtime_sec",
            "total_energy_evaluations",
        ]
        group_cols = ["scenario_id", "solver_id", "adaptation_mode"]
        summary_df = runs_df.groupby(group_cols, dropna=False).agg(
            ppee_10_mean=("ppee_10", "mean"),
            ppee_10_median=("ppee_10", "median"),
            ppee_10_std=("ppee_10", "std"),
            cumulative_excess_energy_mean=("cumulative_excess_energy", "mean"),
            cumulative_excess_energy_median=("cumulative_excess_energy", "median"),
            performance_drop_mean=("performance_drop", "mean"),
            recovery_time_mean=("recovery_time", "mean"),
            kappa_change_post_mean=("kappa_change_post", "mean"),
            theta_change_post_mean=("theta_change_post", "mean"),
            runtime_sec_mean=("total_runtime_sec", "mean"),
            energy_evaluations_mean=("total_energy_evaluations", "mean"),
            n=("ppee_10", "count"),
        ).reset_index()
        summary_df.to_csv(os.path.join(self.cfg.output_dir, "summary.csv"), index=False)

        stats_df = pd.DataFrame(self._compute_paired_statistics(runs_df))
        stats_df.to_csv(os.path.join(self.cfg.output_dir, "statistics.csv"), index=False)

        integrity = self._integrity_report(runs_df, episodes_df, cells)
        with open(os.path.join(self.cfg.output_dir, "integrity.json"), "w", encoding="utf-8") as handle:
            json.dump(integrity, handle, indent=2)

        metadata = self._build_metadata(time.time() - started)
        with open(os.path.join(self.cfg.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        self._write_results_readme(os.path.join(self.cfg.output_dir, "README.md"), integrity)
        return {
            "runs_df": runs_df,
            "episodes_df": episodes_df,
            "summary_df": summary_df,
            "stats_df": stats_df,
            "integrity": integrity,
            "output_dir": self.cfg.output_dir,
        }

    def _integrity_report(
        self, runs_df: pd.DataFrame, episodes_df: pd.DataFrame, cells: List[Tuple[str, str, str, str]]
    ) -> Dict[str, Any]:
        expected_runs = len(self.cfg.seeds) * len(self.cfg.scenarios) * len(cells)
        run_key = ["seed", "scenario_id", "condition_id"] if "condition_id" in runs_df.columns else ["seed", "scenario_id", "solver_id", "adaptation_mode"]
        duplicates = int(runs_df.duplicated(run_key).sum()) if len(runs_df) else 0
        episode_counts = episodes_df.groupby("run_id")["episode"].nunique() if len(episodes_df) else pd.Series(dtype=int)
        expected_episode_set = set(range(self.cfg.num_episodes))
        missing_episode_runs = [
            run_id for run_id, count in episode_counts.items()
            if count != self.cfg.num_episodes
            or set(episodes_df.loc[episodes_df.run_id == run_id, "episode"]) != expected_episode_set
        ]
        primary = ["ppee_10", "cumulative_excess_energy", "reference_energy"]
        finite = bool(np.isfinite(runs_df[primary].to_numpy(dtype=float)).all()) if len(runs_df) else False
        references_shared = (
            runs_df.groupby(["scenario_id", "seed"])["reference_id"].nunique().max() <= 1
            if len(runs_df) else False
        )
        trajectories_shared = (
            runs_df.groupby(["scenario_id", "seed"])["trajectory_id"].nunique().max() <= 1
            if len(runs_df) else False
        )
        return {
            "expected_runs": expected_runs,
            "actual_runs": int(len(runs_df)),
            "run_count_ok": len(runs_df) == expected_runs,
            "duplicate_run_keys": duplicates,
            "duplicate_keys_ok": duplicates == 0,
            "episodes_per_run": self.cfg.num_episodes,
            "runs_with_missing_or_duplicate_episodes": missing_episode_runs,
            "episode_completeness_ok": not missing_episode_runs,
            "ppee_window": list(range(
                self.cfg.perturb_episode,
                min(self.cfg.perturb_episode + self.cfg.post_ppee_window, self.cfg.num_episodes),
            )),
            "references_shared": bool(references_shared),
            "trajectories_shared": bool(trajectories_shared),
            "finite_primary_values": finite,
            "invalid_reference_runs": int((~runs_df["reference_valid"].astype(bool)).sum()) if len(runs_df) else 0,
            "all_checks_pass": bool(
                len(runs_df) == expected_runs
                and duplicates == 0
                and not missing_episode_runs
                and finite
                and references_shared
                and trajectories_shared
                and (int((~runs_df["reference_valid"].astype(bool)).sum()) == 0 if len(runs_df) else False)
            ),
        }

    def _compute_paired_statistics(self, runs_df: pd.DataFrame) -> List[Dict[str, Any]]:
        families: Dict[str, List[Dict[str, Any]]] = {
            "formulation": [],
            "adaptation": [],
            "solver": [],
        }
        pair_keys = ["problem_id", "scenario_id", "seed"]

        for scenario in self.cfg.scenarios:
            scoped = runs_df[runs_df.scenario_id == scenario]

            def get_cond(cond_id: str, solver: str, mode: str):
                if "condition_id" in scoped.columns:
                    match = scoped[scoped.condition_id == cond_id]
                    if len(match) > 0:
                        return match
                return scoped[(scoped.solver_id == solver) & (scoped.adaptation_mode == mode)]

            b0 = get_cond("conventional_greedy", "Conventional Greedy", "static")
            b1 = get_cond("static_energy_greedy", "Energy Greedy", "static")
            b2 = get_cond("static_energy_sa", "Simulated Annealing", "static")
            b3 = get_cond("adaptive_energy_greedy", "Energy Greedy", "full")
            b4 = get_cond("adaptive_energy_sa", "Simulated Annealing", "full")

            # 1. Formulation comparisons (B0 vs B1, B0 vs B2)
            for target_df, label in [(b1, "B0 vs B1 (ConvGreedy vs StaticGreedy)"), (b2, "B0 vs B2 (ConvGreedy vs StaticSA)")]:
                merged = pd.merge(b0, target_df, on=pair_keys, suffixes=("_b0", "_target"))
                if len(merged):
                    res = analyze_paired_comparison(
                        merged.ppee_10_b0.tolist(), merged.ppee_10_target.tolist(),
                        "ppee_10", label
                    )
                    families["formulation"].append(
                        asdict(res) | {"scenario_id": scenario, "solver_id": "Formulation", "hypothesis_family": "Family 1: FORMULATION"}
                    )

            # 2. Adaptation comparisons (B1 vs B3, B2 vs B4)
            for static_df, adapt_df, label in [(b1, b3, "B1 vs B3 (Greedy Static vs Adaptive)"), (b2, b4, "B2 vs B4 (SA Static vs Adaptive)")]:
                merged = pd.merge(adapt_df, static_df, on=pair_keys, suffixes=("_adapt", "_static"))
                if len(merged):
                    res = analyze_paired_comparison(
                        merged.ppee_10_adapt.tolist(), merged.ppee_10_static.tolist(),
                        "ppee_10", label
                    )
                    families["adaptation"].append(
                        asdict(res) | {"scenario_id": scenario, "solver_id": "Adaptation", "hypothesis_family": "Family 2: ADAPTATION"}
                    )

            # 3. Solver comparisons (B1 vs B2, B3 vs B4)
            for greedy_df, sa_df, label in [(b1, b2, "B1 vs B2 (Static Greedy vs SA)"), (b3, b4, "B3 vs B4 (Adaptive Greedy vs SA)")]:
                merged = pd.merge(greedy_df, sa_df, on=pair_keys, suffixes=("_greedy", "_sa"))
                if len(merged):
                    res = analyze_paired_comparison(
                        merged.ppee_10_greedy.tolist(), merged.ppee_10_sa.tolist(),
                        "ppee_10", label
                    )
                    families["solver"].append(
                        asdict(res) | {"scenario_id": scenario, "solver_id": "Solver", "hypothesis_family": "Family 3: SOLVER"}
                    )

        output: List[Dict[str, Any]] = []
        for name, rows in families.items():
            if rows:
                adjusted = holm_bonferroni_correction([row["permutation_p_val"] for row in rows])
                for row, value in zip(rows, adjusted):
                    row["p_val_adjusted"] = value
            output.extend(rows)
        return output

    def _build_metadata(self, runtime: float) -> Dict[str, Any]:
        return {
            "experiment_id": self.cfg.experiment_id,
            "benchmark_version": self.cfg.benchmark_version,
            "git_commit": self.cfg.git_commit,
            "git_dirty": _get_git_dirty(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "N": self.cfg.num_agents,
            "M": self.cfg.num_tasks,
            "d": self.cfg.dim,
            "episodes": self.cfg.num_episodes,
            "perturb_episode": self.cfg.perturb_episode,
            "evaluation_budget": self.cfg.max_energy_evaluations,
            "seeds": self.cfg.seeds,
            "scenarios": self.cfg.scenarios,
            "solvers": self.cfg.solvers,
            "adaptations": self.cfg.ablations,
            "reference_method": "ground_truth_exact_ilp_when_optimal",
            "post_ppee_window": self.cfg.post_ppee_window,
            "ppee_definition": "mean(max(0, external_energy - reference_energy)) over the first post-ppee_window episodes",
            "cee_definition": "sum(max(0, external_energy - reference_energy)) over all post-perturbation episodes",
            "current_problem_observed": True,
            "no_hidden_change_detection": True,
            "no_warm_start": True,
            "deterministic_initialization": self.cfg.initial_x_mode,
            "verbose_diagnostics": self.cfg.verbose_diagnostics,
            "total_runtime_sec": runtime,
        }

    def _write_results_readme(self, path: str, integrity: Dict[str, Any]) -> None:
        content = f"""# Controlled Landscape × Solver Benchmark

Experiment identifier: `{self.cfg.experiment_id}`
Git commit: `{self.cfg.git_commit}`

The current problem instance is directly observed by the solver in every episode; adaptation is historical landscape information, not hidden change detection. Every episode starts from the same deterministic initialization policy and does not warm-start from the previous solution.

Primary metric: `ppee_10`, based exclusively on external ground-truth energy and the shared post-perturbation ILP reference. Lower energy is better; adaptive-minus-static differences below zero are improvements.

## Integrity checks

```json
{json.dumps(integrity, indent=2)}
```
"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
