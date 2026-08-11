import copy
import datetime
import logging
import numpy as np
import os
import sys

# used for normalization
from sklearn.preprocessing import StandardScaler

# used for cross-validation
from sklearn.model_selection import StratifiedKFold

from pandas import DataFrame, read_csv
import pandas as pd 

def loadDatasetOriginal(data: str) :
	dfData = read_csv(os.path.join(data, "data_0.csv"), header=None, sep=',')
	dfLabels = read_csv(os.path.join(data, "labels.csv"), header=None)
	biomarkers = read_csv(os.path.join(data, "features_0.csv"), header=None)
	return dfData.values, dfLabels.values.ravel(), biomarkers.values.ravel() # to have it in the format that the classifiers like

def saveDatasetAsData0(X: np.ndarray, y: np.ndarray, biomarkerNames: np.ndarray, folderName: str) :
	pd.DataFrame(X).to_csv(os.path.join(folderName, "data_0.csv"), header=None, index=None)
	pd.DataFrame(biomarkerNames).to_csv(os.path.join(folderName, "features_0.csv"), header=None, index=None)
	pd.DataFrame(y).to_csv(os.path.join(folderName, "labels.csv"), header=None, index=None)

def loadDataset(globalIndex: int, run: int, runFolderPath: str) :
	
	# data used for the predictions
	dfData = read_csv(os.path.join(runFolderPath, f"run{run}", f"data_{globalIndex}.csv"), header=None, sep=',')
	dfLabels = read_csv(os.path.join(runFolderPath, f"run{run}", "labels.csv"), header=None)
	biomarkers = read_csv(os.path.join(runFolderPath, f"run{run}", f"features_{globalIndex}.csv"), header=None)

	return dfData.values, dfLabels.values.ravel(), biomarkers.values.ravel() # to have it in the format that the classifiers like

# this function returns a list of features, in relative order of importance
def relativeFeatureImportance(classifier) :
	
	# this is the output; it will be a sorted list of tuples (importance, index)
	# the index is going to be used to find the "true name" of the feature
	orderedFeatures = []

	# the simplest case: the classifier already has a method that returns relative importance of features
	if hasattr(classifier, "feature_importances_") :
		orderedFeatures = zip(classifier.feature_importances_ , range(0, len(classifier.feature_importances_)))
		orderedFeatures = sorted(orderedFeatures, key = lambda x : x[0], reverse=True)
	
	# some classifiers are ensembles, and if each element in the ensemble is able to return a list of feature importances
	# (that are going to be all scored following the same logic, so they could be easily aggregated, in theory)
	elif hasattr(classifier, "estimators_") and hasattr(classifier.estimators_[0], "feature_importances_") :

		# add up the scores given by each estimator to all features
		global_score = np.zeros(classifier.estimators_[0].feature_importances_.shape[0])

		for estimator in classifier.estimators_ :
			for i in range(0, estimator.feature_importances_.shape[0]) :
				global_score[i] += estimator.feature_importances_[i]

		# "normalize", dividing by the number of estimators
		for i in range(0, global_score.shape[0]) : global_score[i] /= len(classifier.estimators_)

		# proceed as above to obtain the ranked list of features
		orderedFeatures = zip(global_score, range(0, len(global_score)))
		orderedFeatures = sorted(orderedFeatures, key = lambda x : x[0], reverse=True)
	
	# the classifier does not have "feature_importances_" but can return a list
	# of all features used by a lot of estimators (typical of ensembles)
	elif hasattr(classifier, "estimators_features_") :

		numberOfFeaturesUsed = 0
		featureFrequency = dict()
		for listOfFeatures in classifier.estimators_features_ :
			for feature in listOfFeatures :
				if feature in featureFrequency :
					featureFrequency[feature] += 1
				else :
					featureFrequency[feature] = 1
			numberOfFeaturesUsed += len(listOfFeatures)
		
		for feature in featureFrequency : 
			featureFrequency[feature] /= numberOfFeaturesUsed

		# prepare a list of tuples (name, value), to be sorted
		orderedFeatures = [ (featureFrequency[feature], feature) for feature in featureFrequency ]
		orderedFeatures = sorted(orderedFeatures, key=lambda x : x[0], reverse=True)

	# the classifier does not even have the "estimators_features_", but it's
	# some sort of linear/hyperplane classifier, so it does have a list of
	# coefficients; for the coefficients, the absolute value might be relevant
	elif hasattr(classifier, "coef_") :
	
		# now, "coef_" is usually multi-dimensional, so we iterate on
		# all dimensions, and take a look at the features whose coefficients
		# more often appear close to the top; but it could be mono-dimensional,
		# so we need two special cases
		dimensions = len(classifier.coef_.shape)
		#print("dimensions=", len(dimensions))
		featureFrequency = None # to be initialized later
		
		# check on the dimensions
		if dimensions == 1 :
			featureFrequency = np.zeros(len(classifier.coef_))
			
			relativeFeatures = zip(classifier.coef_, range(0, len(classifier.coef_)))
			relativeFeatures = sorted(relativeFeatures, key=lambda x : abs(x[0]), reverse=True)
			
			for index, values in enumerate(relativeFeatures) :
				value, feature = values
				featureFrequency[feature] += 1/(1+index)

		elif dimensions > 1 :
			featureFrequency = np.zeros(len(classifier.coef_[0]))
			
			# so, for each dimension (corresponding to a class, I guess)
			for i in range(0, len(classifier.coef_)) :
				# we give a bonus to the feature proportional to
				# its relative order, good ol' 1/(1+index)
				relativeFeatures = zip(classifier.coef_[i], range(0, len(classifier.coef_[i])))
				relativeFeatures = sorted(relativeFeatures, key=lambda x : abs(x[0]), reverse=True)
				
				for index, values in enumerate(relativeFeatures) :
					value, feature = values
					featureFrequency[feature] += 1/(1+index)
		else:
			print("The classifier does not have any way to return a list with the relative importance of the features")
			return np.array(orderedFeatures)
			
		# finally, let's sort
		orderedFeatures = [ (featureFrequency[feature], feature) for feature in range(0, len(featureFrequency)) ]
		orderedFeatures = sorted(orderedFeatures, key=lambda x : x[0], reverse=True)

	else :
		print("The classifier does not have any way to return a list with the relative importance of the features")

	return np.array(orderedFeatures)

def getMostImportantFeatures(biomarkerNames: np.ndarray, numberOfTopFeatures: int, numberOfFolds: int, topFeatures: dict[int, int]):
    # transform dictionary into list
	listOfTopFeatures = [ (key, topFeatures[key]) for key in topFeatures ]
	listOfTopFeatures = sorted( listOfTopFeatures, key = lambda x : x[1], reverse=True )

	tempIndex=0
	idsRedRows = []
	for feature, frequency in listOfTopFeatures :
		if tempIndex<numberOfTopFeatures:
			idsRedRows.append([ biomarkerNames[feature], float(frequency/numberOfFolds) ])
		tempIndex=tempIndex+1

	idsRed: DataFrame = pd.DataFrame(idsRedRows)
	return idsRed

def featureSelection(X: np.ndarray, y: np.ndarray, biomarkerNames: np.ndarray, 
					 numberOfTopFeatures: int, numberOfFolds: int, 
					 classifierList, criterion, 
					 verbose: int, folderName: str) :
	# prepare folds
	skf = StratifiedKFold(n_splits=numberOfFolds, shuffle=True)
	indexes = [ (training, test) for training, test in skf.split(X, y) ]
	
	# this will be used for the top features
	topFeatures = dict()
	
	classifierResults = []
	globalAccuracy=0

	# iterate over all classifiers
	for originalClassifier, classifierName in classifierList :
		
		print("\nClassifier " + classifierName)
		classifierPerformance = []
		classifierTopFeatures = dict()

		# iterate over all folds
		for train_index, test_index in indexes :
			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize, anyway
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			#scoreTraining = classifier.score(X_train, y_train)
			#scoreTest = classifier.score(X_test, y_test)
			
			y_new_test = classifier.predict(X_test)
			for i in range(0, len(y_new_test)):
				y_new_test[i]=round(y_new_test[i])
			y_new_train = classifier.predict(X_train)
			for i in range(0, len(y_new_train)):
				y_new_train[i]=round(y_new_train[i])
				
			scoreTest = criterion(y_test, y_new_test)
			scoreTraining = criterion(y_train, y_new_train)
				
			print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			classifierPerformance.append( scoreTest )
			
			# now, let's get a list of the most important features, then mark the ones in the top X
			orderedFeatures = relativeFeatureImportance(classifier) 
			for i in range(0, numberOfTopFeatures) :
				
				feature = int(orderedFeatures[i][1])

				if feature in topFeatures :
					topFeatures[ feature ] += 1
				else :
					topFeatures[ feature ] = 1
				
				if feature in classifierTopFeatures :
					classifierTopFeatures[ feature ] += 1
				else :
					classifierTopFeatures[ feature ] = 1

		classifierResults.append({
			"classifier": classifierName,
			"mean": np.mean(classifierPerformance),
			"std": np.std(classifierPerformance),
		})

		globalAccuracy=globalAccuracy+np.mean(classifierPerformance)
		
		if verbose:
			idsClassifier = getMostImportantFeatures(biomarkerNames, numberOfTopFeatures, numberOfFolds, classifierTopFeatures)

			if idsClassifier.values.any():
				# save most important features for the classifier
				idsClassifier.columns = ["feature"] + [f"frequencyInTop{numberOfTopFeatures}"]
				idsClassifier.to_csv(os.path.join(folderName, f"{classifierName}.csv"), index=False)

	if verbose:
		# save the results of all classifiers

		with open( os.path.join(folderName, "results.txt"), "a" ) as fp :
			for result in classifierResults :
				fp.write("%s\t%.4f\t%.4f\n" % (result["classifier"], result["mean"], result["std"]))
					
	globalAccuracy = globalAccuracy / len(classifierList)
	
	# save most important features overall
	idsReduced = getMostImportantFeatures(biomarkerNames, numberOfTopFeatures, numberOfFolds, topFeatures)

	return globalAccuracy, idsReduced

