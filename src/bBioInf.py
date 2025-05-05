from features import *
from reduceData import *

from joblib import Parallel, delayed
import multiprocessing
#from runClassifiers import *
import time
import pandas as pd
import numpy as np
# used for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

def loadDatasetRAW() :
	
	# data used for the predictions
	dfData = read_csv("../data/data.csv", header=None, sep=',', dtype=float)
	dfLabels = read_csv("../data/labels.csv", header=None)
	biomarkers = read_csv("../data/features.csv", header=None)
	dfIds = read_csv("../data/ids.csv", header=None)
	
	X=dfData.values
	#X=np.transpose(X)
	
	y=dfLabels.values.ravel()
	features=biomarkers.values.ravel()
	ids=dfIds.values.ravel()
	
	# prepare folds
	#skf = StratifiedKFold(n_splits=10, shuffle=True)
	skf = StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=None)
	indexes = [ (training, test) for training, test in skf.split(X, y) ]
	
	
	X_train = X[indexes[0][0]]
	X_test =  X[indexes[0][1]]
	
	y_train = y[indexes[0][0]]
	y_test =  y[indexes[0][1]]
	
	ids_train = ids[indexes[0][0]]
	ids_test  =  ids[indexes[0][1]]
	
	print(np.shape(X_train))
	print(np.shape(X_test))
	print(np.shape(y_train))
	print(np.shape(y_test))
	print(np.shape(ids_train))
	print(np.shape(ids_test))
	

	pd.DataFrame(X_train).to_csv("../data/data_Train.csv", header=None, index =None)
	pd.DataFrame(features).to_csv("../data/features_0.csv", header=None, index =None)
	pd.DataFrame(y_train).to_csv("../data/labels_Train.csv", header=None, index =None)
	pd.DataFrame(ids_train).to_csv("../data/ids_Train.csv", header=None, index =None)
	
	
	pd.DataFrame(X_test).to_csv("../data/data_Test.csv", header=None, index =None)
	pd.DataFrame(y_test).to_csv("../data/labels_Test.csv", header=None, index =None)
	pd.DataFrame(ids_test).to_csv("../data/ids_test.csv", header=None, index =None)
	return 


def mainRun(indexRun) :
	run=indexRun
	start_time = time.time()
	globalAnt=0.0
	globalIndex=0
	globalAccuracy=0.0

	X, y, biomarkerNames = loadDatasetOriginal(run)
	
	if (int(len(X[0]))>1000):
		numberOfTopFeatures = 1000
	else :
		numberOfTopFeatures = int(len(X[0])*0.80)

	variableSize=numberOfTopFeatures;
	while True:
		globalAnt=globalAccuracy
		globalAccuracy=featureSelection(globalIndex,variableSize, run)
		print(globalAccuracy)
		print(globalIndex)
		print(variableSize)
		size,sizereduced=reduceDataset(globalIndex, run)
		
		#if((globalAccuracy<(0.50)) and (globalIndex!=0)):
		#	break
		if(variableSize==0):
			break
		variableSize=int(variableSize*0.80)
		
		globalIndex=globalIndex+1
	elapsed_time = time.time() - start_time
	print("time")
	print(elapsed_time)
	return

def main():

	loadDatasetRAW()
	threads=10
	totalRuns=10
	Parallel(n_jobs=threads, verbose=5, backend="multiprocessing")(delayed(mainRun)(i) for i in range(0,totalRuns))
	return

if __name__ == "__main__" :
	sys.exit( main() )
