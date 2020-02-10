import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class KNN(object):
	def __init__(self, X, y, topK):
		self.x = X
		self.y = y
		self.k = topK

	def euclidian(self, pointA, pointB):

		'''
		pointA -- 1D vector, 1 by n
		pointB -- 1D vector, 1 by n'''

		assert pointA.shape == pointB.shape
		dist = 0
		for coord in range(len(pointA)):
			dist += (pointA[coord] - pointB[coord])**2
		return np.sqrt(dist)

	def distance(self, testpoint, resultArray):
		'''
		This function calculate every test point with the training point, to get the distance. 
		Then, the distance is ordered in the ascending order. What we want the is the top K training points as the associated point to the test point.

		testpont    -- int[][]  in the shape of m by n
		resultArray -- list<list<int[2]>>  x by m by 2, x means the x test point, m is the size of the train set. This would record each test point to every training points' [distance, idx]
		'''
		assert testpoint[0].shape == self.x[0].shape
		#for each test point
		for i in range(testpoint.shape[0]):
			whichTestPoint = testpoint[i]
			#we calculate the distance between each test point to every training point 
			for j in range(self.x.shape[0]):
				whichTrain = self.x[j]
				#then for this test point, we record the distance and the associated traning point
				resultArray[i].append([self.euclidian(whichTestPoint, whichTrain), j])


	def predict(self, testpoint, mode):
		'''
		distanceArray -- list<list<int[2]>>  x by m by 2, x means the x test point, m is the size of the train set. This would record each test point to every training points' [distance, idx]
		topK          -- list<list<int[2]>>  x by k by 2, the top k element for each test set
		prediction    -- list<int>,          x by 1
		'''
		distanceArray = [[] for row in range(testpoint.shape[0])]

		self.distance(testpoint, distanceArray)
	
		for eachSample in distanceArray:
			#within each test sample, we sort the distance between it and all other train point
			eachSample.sort(key = lambda x : x[0])

		assert len(distanceArray[0]) == self.x.shape[0]

		topK = []
		for eachSample in distanceArray:
			topK.append(eachSample[0 : self.k])

		topK = np.array(topK)

		assert topK.shape[0] == len(distanceArray) and topK.shape[1] == self.k and topK.shape[2] == 2

		#if the testing task is the regression, then, for each testpoint, we calculate all the trainsamples associated ys, and take the average
		if mode == 'regression':
			prediction = []
			#for each test sample
			for sample in topK:
				ypred = 0
				#we take out all the distance and idx associated with it, and take out the associated y value within the same test point
				#and the the average
				for tuples in sample:
					whichIdx = int(tuples[1])
					ypred += self.y[whichIdx]
				prediction.append(ypred / self.k)
			return prediction
		#if the test task is the classification, then for each test point, we take out the associated k train sample's y which is the class
		#and take the mode cls as the cls predicted for this test point
		if mode == 'classification':
			from scipy import stats
			prediction = []

			for sample in topK:
				temp = []
				for tuples in sample:
					whichIdx = int(tuples[1])
					temp.append(self.y[whichIdx])
				whichCls,_ = stats.mode(temp)
				prediction.append(whichCls[0])

			return prediction

	def plot(self, ypred, test):
		fig = plt.figure(figsize = (8, 6))
		plt.scatter(X[:,0], X[:,1], c=y)
		plt.scatter(test[:,0], test[:,1], c = 'r')
		plt.scatter(test[:,0], test[:,1], c = ypred)
		# plt.title("Dataset with 4 clusters")

		plt.xlabel("First feature")
		plt.ylabel("Second feature")
		plt.show()

if __name__ == '__main__':
	X, y = make_blobs(centers=6, n_samples=1000)
	fig = plt.figure(figsize=(8,6))
	plt.scatter(X[:,0], X[:,1], c=y)
	plt.title("Dataset with 4 clusters")
	plt.xlabel("First feature")
	plt.ylabel("Second feature")
	plt.show()


	knn = KNN(X, y, 3)
	samplePoint = np.array([[2,3],[4,5],[10,24]])
	yPred = knn.predict(samplePoint, "regression")
	knn.plot(yPred, samplePoint)




