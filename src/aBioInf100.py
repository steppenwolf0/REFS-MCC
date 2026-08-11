from features import *
from reduceData import *

from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import time
import argparse
import traceback


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

def mainRun(X: np.ndarray, y: np.ndarray, biomarkerNames: np.ndarray, 
			classifierList: list, criterion, indexRun: int, numberOfFolds: int, output: str, verbose=1):
	try:
		run=indexRun

		folderName = os.path.join(output, f"run{run}")

		if verbose:
			# create folder
			if not os.path.exists(folderName) : os.makedirs(folderName)

		start_time = time.time()
		globalIndex=0
		globalAccuracy=0.0
		globalAccuracies=[]
		nrFeaturesPerBlock=[]

		bestAccuracy=0.0
		bestIdsReduced=None
		bestGlobalIndex=0

		if verbose:
			saveDatasetAsData0(X, y, biomarkerNames, folderName)
		
		if (int(len(X[0]))>1000):
			numberOfTopFeatures = 1000
		else :
			numberOfTopFeatures = int(len(X[0])*0.80)

		variableSize=numberOfTopFeatures

		while True:
			numberOfTopFeatures=int(variableSize)
			globalAccuracy, idsReduced = featureSelection(X, y, biomarkerNames, numberOfTopFeatures, numberOfFolds, classifierList, criterion, verbose, folderName)

			globalAccuracies.append(globalAccuracy)
			nrFeaturesPerBlock.append(len(idsReduced))

			if (globalAccuracy > bestAccuracy):
				bestAccuracy=globalAccuracy
				bestIdsReduced=idsReduced.to_numpy()
				bestGlobalIndex=globalIndex

			if verbose:
				# save most important features overall
				with open( os.path.join(folderName, f"global_{globalIndex}.csv"), "w" ) as fp :
					fp.write("feature,frequencyInTop" + str(numberOfTopFeatures) + "\n")
					for feature, frequency in idsReduced.values :
						fp.write( str(feature) + "," + str(float(frequency)) + "\n")

			print(globalAccuracy)
			print(globalIndex)
			print(variableSize)

			dataReduced, featuresReduced = reduceDataset(X, biomarkerNames, idsReduced.to_numpy())
			
			size = len(biomarkerNames)
			sizereduced = len(featuresReduced)

			if verbose:
				pd.DataFrame(dataReduced).to_csv(os.path.join(folderName, f"data_{globalIndex+1}.csv"), header=None, index=None)	
				pd.DataFrame(featuresReduced).to_csv(os.path.join(folderName, f"features_{globalIndex+1}.csv"), header=None, index=None)

			if (variableSize==0):
				break

			variableSize=int(variableSize*0.80)

			globalIndex=globalIndex + 1

			X = np.array(dataReduced)
			y = y
			biomarkerNames = np.array(featuresReduced)

		elapsed_time = time.time() - start_time
		print("time")
		print(elapsed_time)

		nrBlocks = globalIndex + 1

		return indexRun, nrBlocks, nrFeaturesPerBlock, globalAccuracies, bestAccuracy, bestIdsReduced, bestGlobalIndex
	except Exception as exc:
		tb = traceback.format_exc()
		raise RuntimeError(f"Worker failed in mainRun with original traceback:\n{tb}") from exc


def main(X: np.ndarray, y: np.ndarray, biomarkerNames: np.ndarray, 
		 threads: int, totalRuns: int, numberOfFolds: int, output: str = ".", classifierList = None, criterion = None, verbose=1):

	if classifierList is None:
		# list of classifiers to be used
		classifierList = [
				# ensemble
				[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
				[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
				[LogisticRegression(), "LogisticRegression"],
				[PassiveAggressiveClassifier(),"PassiveAggressiveClassifier"],
				[SGDClassifier(), "SGDClassifier"],
				[SVC(kernel='linear'), "SVC(linear)"],
				[RidgeClassifier(), "RidgeClassifier"],
				[BaggingClassifier(n_estimators=300), "BaggingClassifier(n_estimators=300)"],
		]

	if criterion is None:
		criterion = matthews_corrcoef

	# run parallel and gather results in a list of tuples
	# use tdqm to show progress bar for the runs
	runResults = Parallel(n_jobs=threads, verbose=5, backend="multiprocessing")(
		delayed(mainRun)(X, y, biomarkerNames, classifierList, criterion, i, numberOfFolds, output, verbose) 
		for i in range(0,totalRuns)
	)

	# unpack the results into separate lists
	indexRunList, nrBlocksList, nrFeaturesPerBlockList, globalAccuraciesList, bestAccuracyList, bestIdsReducedList, bestGlobalIndexList = zip(*runResults)

	# order the results by indexRunList
	orderedResults = sorted(zip(indexRunList, nrBlocksList, nrFeaturesPerBlockList, globalAccuraciesList, bestAccuracyList, bestIdsReducedList, bestGlobalIndexList), key=lambda x: x[0])
	
	results=np.zeros((nrBlocksList[0], totalRuns + 1))

	for j in range(0,totalRuns):
		for i in range (0, nrBlocksList[0]):
			results[i,0]=orderedResults[0][2][i]  # number of features for block i in run 0
			results[i,j+1]=orderedResults[j][3][i]  # accuracy for block i in run j


	bestVal=np.zeros(totalRuns)
	bestSize=np.zeros(totalRuns)
	bestFeatures=[]

	for j in range(0,totalRuns):
		idsReduced = orderedResults[j][5]  # best features for run j
		bestFeatures.append(idsReduced[:, 0])
		bestVal[j]=orderedResults[j][4]
		bestSize[j]=len(idsReduced[:, 0])

	runBest=int(np.argmax(bestVal))
	idsReduced = orderedResults[runBest][5]

	if verbose:
		# create folder
		folderName = os.path.join(output, "best")
		if not os.path.exists(folderName) : os.makedirs(folderName)

		# signatures - for logging only
		signatures = []
		for j in range(0,totalRuns):
			features = bestFeatures[j]
			signatures.append([f"run_{j}", ",".join(map(str, features))])
		signatures.append(["bestVal", ",".join(map(str, bestVal))])
		signatures.append(["bestSize", ",".join(map(str, bestSize))])
		pd.DataFrame(signatures).to_csv(os.path.join(output, "best", "signatures.csv"), header=None, index=None)

	# resultsFeatures - logging only
	unique, counts = np.unique(np.concatenate(bestFeatures), return_counts=True)
	resultsFeatures=np.zeros((len(unique),2), 'U16')
	for j in range(0,len(unique)):
		resultsFeatures[j,0]=unique[j]
		resultsFeatures[j,1]=counts[j]

	if verbose:
		pd.DataFrame(resultsFeatures).to_csv(os.path.join(output, "best", "resultsFeatures.csv"), header=None, index=None)

	return results, idsReduced, resultsFeatures

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

	print("Loading dataset...")
	X, y, biomarkerNames = loadDatasetOriginal(data)

	results, idsReduced, resultsFeatures = main(X, y, biomarkerNames, threads, totalRuns, numberOfFolds, output)
