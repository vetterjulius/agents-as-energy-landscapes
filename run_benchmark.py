import os
import sys
import argparse

# Add the current directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Energy Landscape MAS Benchmark"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced validation benchmark.",
    )

    args = parser.parse_args()

    print("==========================================================")
    print("   Starting One-Command Replication: Energy Landscape MAS")
    print("==========================================================")

    if args.quick:
        print("   MODE: QUICK VALIDATION")
    else:
        print("   MODE: FULL REPLICATION")

    run_benchmark(quick=args.quick)

    print("==========================================================")
    print("   Replication Successful. All results are under results/")
    print("==========================================================")


if __name__ == "__main__":
    main()