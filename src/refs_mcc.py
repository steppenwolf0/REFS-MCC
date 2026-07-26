import argparse
import os
import sys

from aBioInf100 import main as part1
from summaryMulti import fakeBootStrapper as part2
from sumFig import create_summary_figure as part3

class REFS_MCC:
    def __init__(self, threads: int = 10, totalRuns: int = 10, numberOfFolds: int = 10, data: str = "../data", output: str = "."):
        self.threads = threads
        self.totalRuns = totalRuns
        self.numberOfFolds = numberOfFolds
        self.data = data
        self.output = output

    def run(self):
        if self.output != "." and os.path.exists(self.output):
            print(f"Output folder '{self.output}' already exists. Please choose a different folder or remove the existing one.")
            sys.exit(1)

        os.makedirs(self.output, exist_ok=True)
        
        part1(self.threads, self.totalRuns, self.numberOfFolds, data=self.data, output=self.output)
        part2(self.totalRuns, self.numberOfFolds, output=self.output)
        part3(self.totalRuns, output=self.output)

def main():
    parser = argparse.ArgumentParser(description="REFS-MCC")
    parser.add_argument('--threads', type=int, default=10, help='Number of threads (default: 10)')
    parser.add_argument('--totalRuns', type=int, default=10, help='Total number of runs (default: 10)')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
    parser.add_argument('--data', type=str, default="../data", help='Path to the data folder (default: ../data)')
    parser.add_argument('--output', type=str, default=".", help='Path to the output folder (default: .)')
    args = parser.parse_args()

    refs_mcc = REFS_MCC(args.threads, args.totalRuns, args.folds, args.data, args.output)
    return refs_mcc.run()

if __name__ == "__main__":
    sys.exit(main())