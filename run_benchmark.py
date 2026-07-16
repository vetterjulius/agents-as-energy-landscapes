import os
import sys

# Add the current directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.runner import run_benchmark

def main():
    print("==========================================================")
    print("   Starting One-Command Replication: Energy Landscape MAS")
    print("==========================================================")
    run_benchmark()
    print("==========================================================")
    print("   Replication Successful. All results are under results/")
    print("==========================================================")

if __name__ == "__main__":
    main()
