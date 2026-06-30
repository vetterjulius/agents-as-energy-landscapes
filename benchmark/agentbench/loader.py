from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from benchmark.agentbench.adapter import BenchmarkTask


def load_agentbench_tasks(path: Optional[str] = None) -> List[BenchmarkTask]:
    """Load benchmark tasks from a JSON/YAML file when available.

    The repository does not ship a bundled AgentBench dataset, so the loader falls back to
    a small deterministic set of public-style tasks for local integration testing.
    """
    if path and Path(path).exists():
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            import json

            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return [BenchmarkTask(**item) for item in payload]
        if suffix in {".yaml", ".yml"}:
            import yaml

            with open(path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
            return [BenchmarkTask(**item) for item in payload]

    return [
        BenchmarkTask(
            id="agentbench-1",
            instruction="Generate a SQL query that lists active users from the users table.",
            ground_truth="SELECT id, name FROM users WHERE active = 1;",
        ),
        BenchmarkTask(
            id="agentbench-2",
            instruction="Write a SQL query that joins orders with customers and filters by status.",
            ground_truth="SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'paid';",
        ),
    ]
