import torch
import random
from typing import List
from benchmarks.base import BaseBenchmark, BenchmarkScenario

class SyntheticBenchmark(BaseBenchmark):
    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def name(self) -> str:
        return "Synthetic Orchestration Benchmark"

    def load_scenarios(self) -> List[BenchmarkScenario]:
        # Lock in seed for reproducibility across different evaluations
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        
        scenarios = []
        
        # 1. Small Scale
        scenarios.append(self._create_scenario(
            "Small (5x15)", N=5, M=15, d=16
        ))
        
        # 2. Medium Scale (Standard)
        scenarios.append(self._create_scenario(
            "Medium (10x30)", N=10, M=30, d=32
        ))
        
        # 3. Large Scale
        scenarios.append(self._create_scenario(
            "Large (20x60)", N=20, M=60, d=64
        ))
        
        # 4. Conflict-Heavy (More constraints, higher costs)
        scenarios.append(self._create_scenario(
            "Conflict-Heavy (10x30)", N=10, M=30, d=32, cost_multiplier=5.0
        ))
        
        # 5. Risk-Heavy (Higher risk sensitivity)
        scenarios.append(self._create_scenario(
            "Risk-Heavy (10x30)", N=10, M=30, d=32, risk_multiplier=3.0
        ))
        
        return scenarios

    def _create_scenario(self, name: str, N: int, M: int, d: int,
                        cost_multiplier: float = 1.0,
                        risk_multiplier: float = 1.0) -> BenchmarkScenario:
        # Generate agent and task embeddings
        s = torch.randn(N, d)
        c = torch.randn(M, d)
        
        # Co-assignment cost matrix
        C = torch.rand(M, M) * cost_multiplier
        C.fill_diagonal_(0.0)
        
        # Risk weights (3 * d, 1)
        W_risk = torch.randn(3 * d, 1) * risk_multiplier
        
        # Initial assignment: each task randomly assigned to one agent
        X_init = torch.zeros(N, M)
        for t in range(M):
            a = random.randint(0, N - 1)
            X_init[a, t] = 1.0
            
        return BenchmarkScenario(name, N, M, d, s, c, C, W_risk, X_init)
