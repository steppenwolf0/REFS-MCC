from features import *
from reduceData import *

from joblib import Parallel, delayed
import multiprocessing
import time
import argparse

def mainRun(indexRun, numberOfFolds, data, output) :
	run=indexRun
	start_time = time.time()
	globalAnt=0.0
	globalIndex=0
	globalAccuracy=0.0

	X, y, biomarkerNames = loadDatasetOriginal(run, data, output)
	
	if (int(len(X[0]))>1000):
		numberOfTopFeatures = 1000
	else :
		numberOfTopFeatures = int(len(X[0])*0.80)

	variableSize=numberOfTopFeatures;
	while True:
		globalAnt=globalAccuracy
		globalAccuracy=featureSelection(globalIndex,variableSize, run, numberOfFolds, data, output)
		print(globalAccuracy)
		print(globalIndex)
		print(variableSize)
		size,sizereduced=reduceDataset(globalIndex, run, data, output)
		
		if(variableSize==0):
			break
		variableSize=int(variableSize*0.80)
		
		globalIndex=globalIndex + 1

	elapsed_time = time.time() - start_time
	print("time")
	print(elapsed_time)
	return

def main(threads, totalRuns, numberOfFolds, data, output=".") :
	Parallel(n_jobs=threads, verbose=5, backend="multiprocessing")(delayed(mainRun)(i, numberOfFolds, data, output) for i in range(0,totalRuns))
	return

if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description="Run REFS-MCC - part 1")
	parser.add_argument('--threads', type=int, default=10, help='Number of threads (default: 10)')
	parser.add_argument('--totalRuns', type=int, default=10, help='Total number of runs (default: 10)')
	parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
	parser.add_argument('--data', type=str, default="../data", help='Path to the data folder (default: ../data)')
	parser.add_argument('--output', type=str, default=".", help='Path to the output folder (default: .)')
	args = parser.parse_args()

	threads=args.threads
	totalRuns=args.totalRuns
	numberOfFolds=args.folds
	data=args.data
	output=args.output

	sys.exit( main(threads, totalRuns, numberOfFolds, data, output) )
