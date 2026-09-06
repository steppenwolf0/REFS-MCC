import copy
import datetime
import logging
import numpy as np
import os
import sys
import pandas as pd 
import argparse

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

# used for normalization
from sklearn.preprocessing import StandardScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from numpy import interp
import matplotlib.pyplot as plt
from pandas import read_csv

from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import f1_score

def loadDataset(output) :
	
	# data used for the predictions
	dfData = read_csv(os.path.join(output, "best", "data_0.csv"), header=None, sep=',')
	dfLabels = read_csv(os.path.join(output, "best", "labels.csv"), header=None)
		
	return dfData.values, dfLabels.values.ravel() # to have it in the format that the classifiers like


def runFeatureReduce(X, y, numberOfFolds, output, verbose = 1, classifierList = None) :

	if classifierList is None:
		classifierList = [
			[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			[LogisticRegression(solver='lbfgs',), "LogisticRegression"],
			[PassiveAggressiveClassifier(),"PassiveAggressiveClassifier"],
			[SGDClassifier(), "SGDClassifier"],
			[SVC(kernel='linear'), "SVC(linear)"],
			[RidgeClassifier(), "RidgeClassifier"],
			[BaggingClassifier(n_estimators=300), "BaggingClassifier(n_estimators=300)"],

			# tree
			[AdaBoostClassifier(n_estimators=300), "AdaBoostClassifier(n_estimators=300)"],
			[ExtraTreesClassifier(n_estimators=300), "ExtraTreesClassifier(n_estimators=300)"],
			[KNeighborsClassifier(), "KNeighborsClassifier"],
			[MLPClassifier(), "MLPClassifier"],
			[LassoCV(), "LassoCV"]
		]

	labels=np.max(y)+1
	# prepare folds
	skf = StratifiedKFold(n_splits=numberOfFolds, shuffle=True)
	indexes = [ (training, test) for training, test in skf.split(X, y) ]
	
	# this will be used for the top features
	topFeatures = dict()
	
	# iterate over all classifiers
	classifierIndex = 0

	classifierResults = []
	
	for originalClassifier, classifierName in classifierList :
		
		print("\nClassifier " + classifierName)
		classifierPerformance = []
		F1score= []

		cMatrix=np.zeros((labels, labels))
		# iterate over all folds
		
		indexFold = 0

		yTest=[]
		yNew=[]

		for train_index, test_index in indexes :
			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize, anyway
			# MinMaxScaler StandardScaler Normalizer
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

		
			
			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			scoreTraining = classifier.score(X_train, y_train)
			scoreTest = classifier.score(X_test, y_test)
			
			y_new = classifier.predict(X_test)
			
			
			
			yNew.append(y_new)
			yTest.append(y_test)
			
			
			for i in range(0,len(y_new)):
				rounded_y_new = round(y_new[i])
				if (rounded_y_new<0 or rounded_y_new>=labels):
					rounded_y_new=0 if (rounded_y_new<0) else labels-1

				cMatrix[y_test[i]][rounded_y_new]+=1
				y_new[i]=rounded_y_new

			print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			classifierPerformance.append( scoreTest )
			
	
			F1scoreTest = f1_score(y_test, y_new, average='weighted')
			F1score.append(F1scoreTest)

		if verbose:
			pd.DataFrame(cMatrix).to_csv(os.path.join(output, "best", "cMatrix"+str(classifierIndex)+".csv"), header=None, index =None)

		classifierIndex+=1

		classifierResults.append({
			"classifier": classifierName,
			"mean": np.mean(classifierPerformance),
			"std": np.std(classifierPerformance),
			"mean_F1": np.mean(F1score), 
			"std_F1": np.std(F1score)
		})

	if verbose:
		with open( os.path.join(output, "best", "results.txt"), "w" ) as fp :
			for result in classifierResults :
				line = "%s \t %.4f \t %.4f \t %.4f \t %.4f\n" % (result["classifier"], result["mean"], result["std"], result["mean_F1"], result["std_F1"])
				print(line)
				fp.write(line)
	
	return classifierResults

if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description="Run multiple classifiers based on the best run")
	parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
	parser.add_argument('--output', type=str, default=".", help='Path to the output folder (default: .)')
	args = parser.parse_args()

	print("Loading dataset...")
	X, y = loadDataset(args.output)
	print(len(X))
	print(len(X[0]))
	print(len(y))

	runFeatureReduce(X, y, args.folds, args.output)