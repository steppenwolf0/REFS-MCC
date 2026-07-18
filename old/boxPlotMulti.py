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

# this is an incredibly useful function
from pandas import read_csv
import matplotlib.pyplot as plt

directory="best"

def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./"+directory+"/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./"+directory+"/labels.csv", header=None)
	dfFeats = read_csv("./"+directory+"/features_0.csv", header=None)
	dfClasses = read_csv("./"+directory+"/classes.csv", header=None)		
	return dfData.values, dfLabels.values.ravel(), dfFeats.values.ravel(), dfClasses.values.ravel() # to have it in the format that the classifiers like


def fakeBootStrapper2():

	
	X,Y, feats, classes=loadDataset()
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X)
	labels=np.max(Y)+1
	
	xClass=[]
	
	for j in range(0,labels):
		temp=[]
		for i in range(0,len(Y)):
			if (Y[i]==j):
				values=[]
				values=X[i,:]
				temp.append(values)
		xClass.append(temp)

	
	green_diamond = dict(markerfacecolor='r', marker='D')
	
	
	axisVal=[]
	for i in range (1,len(xClass[0][0])+1):
		axisVal.append(i)
	
	for j in range(0,labels):
	
		fig1, ax1 = plt.subplots()
		ax1.set_title(classes[j])
		ax1.boxplot(np.array(xClass[j]), flierprops=green_diamond)
		
		labels = feats
		plt.xticks(axisVal, labels, rotation='vertical')
		plt.subplots_adjust(bottom=0.22)
	
	
	plt.show()
	return

if __name__ == "__main__" :
	sys.exit( fakeBootStrapper2() )