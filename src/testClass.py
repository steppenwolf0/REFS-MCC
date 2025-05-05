# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import pandas as pd 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
# this is an incredibly useful function
from pandas import read_csv

from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from scipy import stats

def reduceData(dfData,biomarkers,dfbioMarkers):
	X_train=np.zeros((len(dfData),len(biomarkers)))
	count=0
	for i in range(0,len(biomarkers)):
		for j in range(0,len(dfbioMarkers)):
			if(biomarkers[i]==dfbioMarkers[j]):
				for k in range(0,len(dfData)):
					X_train[k,count]=dfData[k,j]
				count=count+1
	return X_train

def runFeatureReduce() :
	
	# list of classifiers, selected on the basis of our previous paper "
	classifierList = [
		
			[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			[LogisticRegression(solver='lbfgs',), "LogisticRegression"],
			[PassiveAggressiveClassifier(),"PassiveAggressiveClassifier"],
			[SGDClassifier(), "SGDClassifier"],
			[SVC(kernel='linear'), "SVC(linear)"],
			[RidgeClassifier(), "RidgeClassifier"],
			[BaggingClassifier(n_estimators=300), "BaggingClassifier(n_estimators=300)"],
			# ensemble
			#[AdaBoostClassifier(), "AdaBoostClassifier"],
			#[AdaBoostClassifier(n_estimators=300), "AdaBoostClassifier(n_estimators=300)"],
			#[AdaBoostClassifier(n_estimators=1500), "AdaBoostClassifier(n_estimators=1500)"],
			#[BaggingClassifier(), "BaggingClassifier"],
			
			#[ExtraTreesClassifier(), "ExtraTreesClassifier"],
			#[ExtraTreesClassifier(n_estimators=300), "ExtraTreesClassifier(n_estimators=300)"],
			 # features_importances_
			#[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			#[GradientBoostingClassifier(n_estimators=1000), "GradientBoostingClassifier(n_estimators=1000)"],
			
			#[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			#[RandomForestClassifier(n_estimators=1000), "RandomForestClassifier(n_estimators=1000)"], # features_importances_

			# linear
			#[ElasticNet(), "ElasticNet"],
			#[ElasticNetCV(), "ElasticNetCV"],
			#[Lasso(), "Lasso"],
			#[LassoCV(), "LassoCV"],
			 # coef_
			#[LogisticRegressionCV(), "LogisticRegressionCV"],
			  # coef_
			 # coef_
			#[RidgeClassifierCV(), "RidgeClassifierCV"],
			 # coef_
			 # coef_, but only if the kernel is linear...the default is 'rbf', which is NOT linear
			
			# naive Bayes
			#[BernoulliNB(), "BernoulliNB"],
			#[GaussianNB(), "GaussianNB"],
			#[MultinomialNB(), "MultinomialNB"],
			
			# neighbors
			#[KNeighborsClassifier(), "KNeighborsClassifier"], # no way to return feature importance
			# TODO this one creates issues
			#[NearestCentroid(), "NearestCentroid"], # it does not have some necessary methods, apparently
			#[RadiusNeighborsClassifier(), "RadiusNeighborsClassifier"],
			
			# tree
			#[DecisionTreeClassifier(), "DecisionTreeClassifier"],
			#[ExtraTreeClassifier(), "ExtraTreeClassifier"],

			]
	
	# this is just a hack to check a few things
	#classifierList = [
	#		[RandomForestClassifier(), "RandomForestClassifier"]
	#		]

	print("Loading dataset...")
	
	
	# create folder
	folderName ="../results/"
	if not os.path.exists(folderName) : os.makedirs(folderName)
	
	
	biomarkers = read_csv("./best/features_0.csv", header=None).values.ravel()
	
	# data used for the predictions
	dfData = read_csv("../data/data_0.csv", header=None, sep=',', dtype=float).values
	dfLabels = read_csv("../data/labels_0.csv", header=None).values.ravel()
	dfbioMarkers = read_csv("../data/features_0.csv", header=None).values.ravel()

	print(biomarkers)

	# data used for the predictions
	dfDataTest = read_csv("../data/data_Test.csv", header=None, sep=',', dtype=float).values
	dfLabelsTest = read_csv("../data/labels_Test.csv", header=None).values.ravel()
	
	# iterate over all classifiers
	classifierIndex = 0
	
	X_train=reduceData(dfData,biomarkers,dfbioMarkers)
	pd.DataFrame(X_train).to_csv("../results/X_train.csv", header=None, index =None)
	
	
	X_test=reduceData(dfDataTest,biomarkers,dfbioMarkers)
	pd.DataFrame(X_test).to_csv("../results/X_test.csv", header=None, index =None)
	
	y_train=dfLabels
	y_test=dfLabelsTest
	
	labels=np.max(y_train)+1
	
	aucs=[]
	
	for originalClassifier, classifierName in classifierList :
		
		# let's normalize, anyway
		# MinMaxScaler StandardScaler Normalizer
		scaler = MinMaxScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.fit_transform(X_test)

		
		classifier = copy.deepcopy(originalClassifier)
		classifier.fit(X_train, y_train)
		scoreTraining = classifier.score(X_train, y_train)
		scoreTest = classifier.score(X_test, y_test)
		
		y_new = classifier.predict(X_test)
		
		print("\nClassifier " + classifierName)
		classifierPerformance = []
		cMatrix=np.zeros((labels, labels))		
		for i in range(0,len(y_new)):
			cMatrix[y_test[i]][y_new[i]]+=1


		print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
		classifierPerformance.append( scoreTest )
		
		plt.clf()
		fpr, tpr, thresholds = roc_curve(y_test, y_new)
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr,  label='ROC (AUC = %0.2f)' % ( roc_auc))
		
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic ')
		plt.legend(loc="lower right")
		#plt.show()
		plt.savefig("../results/%s.png"% ( classifierName), dpi=300)

		pd.DataFrame(cMatrix).to_csv("../results/cMatrix"+str(classifierIndex)+"Test.csv", header=None, index =None)
		classifierIndex+=1
		line ="%s \t %.4f \t %.4f \n" % (classifierName, np.mean(classifierPerformance), np.std(classifierPerformance))
		
		
		aucs.append(round(roc_auc, 4))
		
		print(line)
		fo = open("../results/resultsTest.txt", 'a')
		fo.write( line )
		fo.close()
		
	pd.DataFrame(aucs).to_csv("../results/AUCS.csv", header=None, index =None)
	return

if __name__ == "__main__" :
	sys.exit( runFeatureReduce() )