from benchmark.agentbench.adapter import AgentBenchAdapter
from orchestrator.benchmark_runner import PublicBenchmarkRunner


def main():
    runner = PublicBenchmarkRunner(adapter=AgentBenchAdapter())
    results = runner.run(orchestrator_name="energy")
    print(f"Processed {len(results)} benchmark tasks")
    for item in results:
        print(item["benchmark_task"], "->", item["assignment"])


if __name__ == "__main__":
    main()
