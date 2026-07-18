import copy
import datetime
import logging
import numpy as np
import os
import sys
from pandas import read_csv
import pandas as pd
from sklearn.impute import SimpleImputer

def loadDataset() :
	dfData = read_csv("./dataT.csv", header=None, sep=',', dtype=float).values
	biomarkers = read_csv("./features.csv", header=None)
	return dfData, biomarkers.values.ravel()

def unique_cols(a):

	return (a[0] == a).all(0)
	
def featureSelection() :
	#load Dataset
	X,y = loadDataset()
	X=np.transpose(X)
	Xnew=X
	uniqueCols=unique_cols(X)
	
	uniqueList=[]
	for i in range(0,len(uniqueCols)):
		if uniqueCols[i]:
			uniqueList.append(i)
	
	Xnew=np.delete(Xnew, uniqueList, 1)
	y=np.delete(y, uniqueList, 0)
	
	
	pd.DataFrame(uniqueList).to_csv("uniqueList.csv", header=None, index =None)
	pd.DataFrame(X).to_csv("data.csv", header=None, index =None)
	pd.DataFrame(Xnew).to_csv("data_0.csv", header=None, index =None)
	pd.DataFrame(y).to_csv("features_0.csv", header=None, index =None)

if __name__ == "__main__" :
	sys.exit( featureSelection() )