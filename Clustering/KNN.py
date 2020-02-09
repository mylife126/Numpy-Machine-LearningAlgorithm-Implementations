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
		assert pointA.shape == pointB.shape
		dist = 0
		for coord in range(len(pointA)):
			dist += (pointA[coord] - pointB[coord])**2
		return np.sqrt(dist)

	def distance(self, testpoint, resultArray):
		assert testpoint[0].shape == self.x[0].shape
		for i in range(testpoint.shape[0]):
			whichTestPoint = testpoint[i]
			for j in range(self.x.shape[0]):
				whichTrain = self.x[j]
				resultArray[i].append([self.euclidian(whichTestPoint, whichTrain), j])


	def predict(self, testpoint, mode):
		distanceArray = [[] for row in range(testpoint.shape[0])]

		self.distance(testpoint, distanceArray)
		for eachSample in distanceArray:
			eachSample.sort(key = lambda x : x[0])
		assert len(distanceArray[0]) == self.x.shape[0]

		topK = []
		for eachSample in distanceArray:
			topK.append(eachSample[0 : self.k])

		topK = np.array(topK)

		assert topK.shape[0] == len(distanceArray) and topK.shape[1] == self.k and topK.shape[2] == 2

		if mode == 'regression':
			prediction = []
			for sample in topK:
				ypred = 0
				for tuples in sample:
					whichIdx = int(tuples[1])
					ypred += self.y[whichIdx]
				prediction.append(ypred / self.k)
			return prediction

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

	def plot(self, ypred, mode):
		fig = plt.figure(figsize = (8, 6))
		plt.scatter(X[:,0], X[:,1], c=y)
		plt.title("Dataset with 4 clusters")
		plt.xlabel("First feature")
		plt.ylabel("Second feature")

if __name__ == '__main__':
	X, y = make_blobs(centers=6, n_samples=1000)
	Xcls, ycls = make_classification(n_samples = 1000, n_features = 2, n_classes = 2, n_informative = 2, n_redundant = 0)
	fig = plt.figure(figsize=(8,6))
	plt.scatter(X[:,0], X[:,1], c=y)
	plt.title("Dataset with 4 clusters")
	plt.xlabel("First feature")
	plt.ylabel("Second feature")
	plt.show()


	knn = KNN(X, y, 3)
	samplePoint = np.array([[2,3],[4,5],[10,24]])
	yPred = knn.predict(samplePoint, "classification")
	print(y)
	print(yPred)




