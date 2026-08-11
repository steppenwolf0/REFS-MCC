import numpy as np


def reduceDataset(data: np.ndarray, ids: np.ndarray, idsReduced: np.ndarray):
	print("data Y %d" %(len(data)))
	print("data X %d" %(len(data[0])))
	print(len(ids))
	print(len(idsReduced))

	featuresReduced=[]
	for i in range(0,len(idsReduced)):
		feature = idsReduced[i, 0]
		frequency = idsReduced[i, 1]
		if float(frequency) >= 1:
			featuresReduced.append(feature)

	print(len(featuresReduced))
	count=0
	
	dataReduced=np.zeros((len(data),len(featuresReduced)))

	for i in range(0,len(featuresReduced)):
		for j in range (0,len(ids)):
			if (ids[j]== featuresReduced[i]):
				count=count+1
				for k in range(0,len(data)):
					dataReduced[k,i]=data[k,j]
			
	print(count)
	
	return dataReduced, featuresReduced
