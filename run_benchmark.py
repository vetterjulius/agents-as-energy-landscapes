import os
import sys
import argparse
import logging

# Add the current directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.logging_config import setup_logger
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
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )

    args = parser.parse_args()
    
    # Configure logging based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(level=log_level)
    
    logger.info("=" * 70)
    logger.info("Energy Landscape MAS Benchmark - One-Command Replication")
    logger.info("=" * 70)

    mode = "QUICK VALIDATION" if args.quick else "FULL REPLICATION"
    logger.info(f"Mode: {mode}")

    run_benchmark(quick=args.quick)

    logger.info("=" * 70)
    logger.info("Benchmark completed successfully")
    logger.info("Results available in: results/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()