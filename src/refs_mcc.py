import argparse
import sys

from aBioInf100 import main as part1
from summaryMulti import fakeBootStrapper as part2
from sumFig import create_summary_figure as part3

class REFC_MCC:
    def __init__(self, threads: int = 10, totalRuns: int = 10, numberOfFolds: int = 10):
        self.threads = threads
        self.totalRuns = totalRuns
        self.numberOfFolds = numberOfFolds

    def run(self):
        part1(self.threads, self.totalRuns, self.numberOfFolds)
        part2(self.totalRuns, self.numberOfFolds)
        part3(self.totalRuns)

def main():
    parser = argparse.ArgumentParser(description="REFS-MCC")
    parser.add_argument('--threads', type=int, default=10, help='Number of threads (default: 10)')
    parser.add_argument('--totalRuns', type=int, default=10, help='Total number of runs (default: 10)')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
    args = parser.parse_args()

    refc_mcc = REFC_MCC(args.threads, args.totalRuns, args.folds)
    return refc_mcc.run()

if __name__ == "__main__":
    sys.exit(main())