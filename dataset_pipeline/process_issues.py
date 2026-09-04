import os
import json
import argparse
import torch
import torch.nn as nn

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class FeatureEmbedder:
    """
    Encodes text into continuous embedding vectors.
    Uses sentence-transformers if available, otherwise falls back to a deterministic BoW Projection.
    """
    def __init__(self, target_dim: int = 8):
        self.target_dim = target_dim
        if HAS_ST:
            # Compact offline/online sentence transformer model (384-dim)
            self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
            # Linear projection layer down to benchmark target_dim (e.g. 8)
            torch.manual_seed(42)
            self.proj = nn.Linear(384, target_dim, bias=False)
            nn.init.orthogonal_(self.proj.weight)
        else:
            print("[Warning] 'sentence_transformers' not installed. Using Bag-of-Words fallback projection.")

    def embed_text(self, text: str) -> torch.Tensor:
        if HAS_ST:
            emb_384 = torch.tensor(self.st_model.encode(text), dtype=torch.float32)
            with torch.no_grad():
                emb_d = self.proj(emb_384)
            # Normalize vector to unit length
            return emb_d / (torch.norm(emb_d) + 1e-8)
        else:
            # Deterministic fallback BoW hashing projection to target_dim
            tokens = text.lower().split()
            v = torch.zeros(self.target_dim)
            for t in tokens:
                idx = abs(hash(t)) % self.target_dim
                v[idx] += 1.0
            return v / (torch.norm(v) + 1e-8)


# Default Agent Roster for Software Engineering MAS (Scaled to 10 Agents)
DEFAULT_AGENTS = [
    {
        "id": "Agent-Backend-Core",
        "role": "Backend Engineer (Python, Core Architecture, WSGI, Routing, Dispatcher)",
        "tags": ["backend", "core", "wsgi"]
    },
    {
        "id": "Agent-Security-Auth",
        "role": "Security Specialist (Authentication, Session Management, Cookies, Cryptography)",
        "tags": ["security", "auth", "sessions"]
    },
    {
        "id": "Agent-Frontend-Templates",
        "role": "Frontend & Template Developer (Jinja2, HTML, Rendering, Context Helpers)",
        "tags": ["frontend", "templates", "jinja"]
    },
    {
        "id": "Agent-Async-Signals",
        "role": "Concurrency Specialist (AsyncIO, Event Loop, Signals, Hooks, Callbacks)",
        "tags": ["async", "signals", "events"]
    },
    {
        "id": "Agent-DevOps-CI",
        "role": "DevOps Specialist (GitHub Actions, Packaging, PyPI, Tox, Deployment)",
        "tags": ["devops", "ci", "packaging"]
    },
    {
        "id": "Agent-QA-Testing",
        "role": "QA Engineer (Pytest, Fixtures, Coverage, Mocking, Integration Tests)",
        "tags": ["qa", "testing", "pytest"]
    },
    {
        "id": "Agent-Doc-Maintainer",
        "role": "Technical Writer (Sphinx, Readme, API Documentation, Tutorials)",
        "tags": ["doc", "sphinx", "tutorials"]
    },
    {
        "id": "Agent-CLI-Tooling",
        "role": "CLI Specialist (Click, Command Line, Arguments, Shell Completion)",
        "tags": ["cli", "click", "tooling"]
    },
    {
        "id": "Agent-Data-DB",
        "role": "Database & ORM Engineer (SQLAlchemy, Migration, Schemas, JSON Storage)",
        "tags": ["database", "sql", "orm"]
    },
    {
        "id": "Agent-Performance-Optimization",
        "role": "Performance Engineer (Profiling, Caching, Memory Leak, Benchmarking)",
        "tags": ["performance", "caching", "profiling"]
    }
]


def process_dataset(raw_issues_path: str, output_path: str, dim: int = 8):
    print(f"[2/3] Processing raw issues from {raw_issues_path} into benchmark tensors (dim={dim})...")
    with open(raw_issues_path, "r", encoding="utf-8") as f:
        issues = json.load(f)

    embedder = FeatureEmbedder(target_dim=dim)

    # 1. Embed Tasks
    tasks_data = []
    for issue in issues:
        full_text = f"{issue['title']}. {issue['body']}"
        emb = embedder.embed_text(full_text)

        tasks_data.append({
            "id": issue["id"],
            "title": issue["title"],
            "embedding": emb.tolist(),
            "labels": issue.get("labels", []),
            "file_references": issue.get("file_references", [])
        })

    # 2. Embed Agent Roster
    agents_data = []
    for agent in DEFAULT_AGENTS:
        emb = embedder.embed_text(agent["role"])
        agents_data.append({
            "id": agent["id"],
            "role": agent["role"],
            "capability_embedding": emb.tolist(),
            "tags": agent["tags"]
        })

    # 3. Build Interaction Matrix (Theta) based on Shared File References & Label Co-occurrence
    M = len(tasks_data)
    Theta = torch.zeros(M, M)
    C = torch.zeros(M, M)

    for i in range(M):
        for j in range(i + 1, M):
            files_i = set(tasks_data[i]["file_references"])
            files_j = set(tasks_data[j]["file_references"])
            labels_i = set(tasks_data[i]["labels"])
            labels_j = set(tasks_data[j]["labels"])

            # Shared file coupling (strong interaction)
            shared_files = len(files_i.intersection(files_j))
            shared_labels = len(labels_i.intersection(labels_j))

            coupling = 0.5 * shared_files + 0.2 * shared_labels
            if coupling > 0:
                Theta[i, j] = coupling
                Theta[j, i] = coupling

            # Co-assignment communication cost
            C[i, j] = 0.1
            C[j, i] = 0.1

    # 4. Generate Risk Weights W_risk (3*dim, 1)
    torch.manual_seed(42)
    W_risk = (torch.randn(3 * dim, 1) * 0.5).tolist()

    processed_data = {
        "metadata": {
            "num_agents": len(agents_data),
            "num_tasks": len(tasks_data),
            "dim": dim
        },
        "agents": agents_data,
        "tasks": tasks_data,
        "interaction_graph": Theta.tolist(),
        "co_assignment_costs": C.tolist(),
        "risk_weights": W_risk
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)

    print(f" -> Processed {len(tasks_data)} tasks and {len(agents_data)} agents.")
    print(f" -> Saved processed benchmark dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert raw GitHub issues to Benchmark Tensors")
    parser.add_argument("--input", type=str, default="dataset_pipeline/raw_issues.json", help="Input raw issues JSON")
    parser.add_argument("--output", type=str, default="dataset_pipeline/processed_gh_dataset.json", help="Output benchmark dataset JSON")
    parser.add_argument("--dim", type=int, default=8, help="Target embedding dimension for benchmark")
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.dim)

if __name__ == "__main__":
    main()
