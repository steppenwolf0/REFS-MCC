import argparse
import sys
from timeit import main

from aBioInf100 import main as part1
from summaryMulti import fakeBootStrapper as part2
from sumFig import create_summary_figure as part3

def main(args):
    threads = args.threads
    totalRuns = args.totalRuns
    numberOfFolds = args.folds

    part1(threads, totalRuns, numberOfFolds)
    part2(totalRuns, numberOfFolds)
    part3(totalRuns)
    return

if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description="REFS-MCC")
	parser.add_argument('--threads', type=int, default=10, help='Number of threads (default: 10)')
	parser.add_argument('--totalRuns', type=int, default=10, help='Total number of runs (default: 10)')
	parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
	args = parser.parse_args()

	sys.exit( main(args) )