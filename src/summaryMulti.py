import numpy as np
import os
import sys
import pandas as pd 
import argparse

from classifiersMulti import *

from pandas import read_csv

def fakeBootStrapper(runs, numberOfFolds, output = ".") :
	# create folder
	folderName = os.path.join(output, "best")
	if not os.path.exists(folderName) : os.makedirs(folderName)

	orig_stdout = sys.stdout
	f = open(os.path.join(output, "best", "out.txt"), 'w')
	sys.stdout = f
	
	directory= f"run{0}"
	f=open(os.path.join(output, directory, "results.txt"), "r")
	fl=f.readlines()
	blocks=int(len(fl)/8)
	print(blocks)
	results=np.zeros((blocks,runs+1))
	for j in range(0,runs):
		directory= f"run{j}"
		f=open(os.path.join(output, directory, "results.txt"), "r")
		fl =f.readlines()
		blocks=int(len(fl)/8)
		count=0
		value=0
		variables=np.zeros(blocks)
		accuracy=np.zeros(blocks)
		indexResults=0
		for x in fl:
			a=x.split("\t")
			value=value+float(a[1])/8.0
			count=count+1
			if (count==8):
				accuracy[indexResults]=value
				value=0
				count=0
				indexResults=indexResults+1
		indexResults=0
		
		for i in range (0,blocks):
			dfFeats = (read_csv(os.path.join(output, directory, f"features_{i}.csv"), header=None)).values.ravel() 
			variables[i]=len(dfFeats)
			results[i,j+1]=accuracy[i]
			results[i,0]=variables[i]
		
		
	
	pd.DataFrame(results).to_csv(os.path.join(output, "best", "sum.csv"), header=None, index =None)
	
	bestVal=np.zeros(runs)
	bestSize=np.zeros(runs)
	bestPos=np.zeros(runs)
	for j in range(0,runs):
		bestVal[j]=np.max(results[:,j+1])
	
	for j in range(0,runs):	
		for i in range (0,blocks):
			if ( bestVal[j]==results[i,j+1]):
				bestSize[j]=int(results[i,0])
				bestPos[j]=i
	print(bestVal)
	print(bestSize)
	print(bestPos)
	
	bestFeatures=[]
	signatures=[]
	for j in range(0,runs):
		dfFeats = (read_csv(os.path.join(output, f"run{j}", f"features_{int(bestPos[j])}.csv"), header=None))
		bestFeatures.extend(dfFeats.values.ravel())
		signatures.append(dfFeats.values.ravel())
	#print(bestFeatures)
	signatures.append(bestVal)
	signatures.append(bestSize)
	pd.DataFrame(signatures).to_csv(os.path.join(output, "best", "signatures.csv"), header=None, index =None)
	
	unique, counts = np.unique(bestFeatures, return_counts=True)
	
	resultsFeatures=np.zeros((len(unique),2), 'U16')
	for j in range(0,len(unique)):
		resultsFeatures[j,0]=unique[j]
		resultsFeatures[j,1]=counts[j]
	
	
	pd.DataFrame(resultsFeatures).to_csv(os.path.join(output, "best", "resultsFeatures.csv"), header=None, index =None)
	
	print(np.max(bestVal))
	print(np.argmax(bestVal))
	print(int(bestPos[np.argmax(bestVal)]))
	
	runBest=int(np.argmax(bestVal))
	indexBest=int(bestPos[np.argmax(bestVal)])
	
	
	
	# data used for the predictions
	dfData = read_csv(os.path.join(output, f"run{runBest}", f"data_{indexBest}.csv"), header=None, sep=',')
	dfLabels = read_csv(os.path.join(output, f"run{runBest}", "labels.csv"), header=None)
	biomarkers = read_csv(os.path.join(output, f"run{runBest}", f"features_{indexBest}.csv"), header=None)
	
	pd.DataFrame(dfData.values).to_csv(os.path.join(output, "best", "data_0.csv"), header=None, index =None)
	pd.DataFrame(biomarkers.values.ravel()).to_csv(os.path.join(output, "best", "features_0.csv"), header=None, index =None)
	pd.DataFrame(dfLabels.values.ravel()).to_csv(os.path.join(output, "best", "labels.csv"), header=None, index =None)
	
	runFeatureReduce(numberOfFolds, output)
	sys.stdout = orig_stdout
	f.close()
	return

if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description="Run REFS-MCC - part 2")
	parser.add_argument('--totalRuns', type=int, default=10, help='Total number of runs (default: 10)')
	parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
	parser.add_argument('--output', type=str, default=".", help='Path to the output folder (default: .)')
	args = parser.parse_args()

	runs=args.totalRuns
	numberOfFolds=args.folds
	output=args.output

	sys.exit( fakeBootStrapper(runs, numberOfFolds, output) )