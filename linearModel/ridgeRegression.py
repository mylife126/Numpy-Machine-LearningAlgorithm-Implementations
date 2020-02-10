from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.datasets import make_regression


class ridgeRegression():
	def __init__(self, alpha, intercept):
		'''
		using normal equation to fit a ridge regression problem

		W = (AT@A + aI)^-1 AT@Y
		   n by n @ n * m @ m by 1 => n by 1 

		
		self.theta    -- int[N] the weight for the model X@theta  
		self.intercep -- boolean  decide if wanna include the interception or not
		'''
		self.a = alpha
		self.intercept = intercept
		self.mRows = None
		self.nFeatures = None
		self.theta = None

	def addOneToFeature(self, X):
		ones = np.ones(self.mRows)    #dim = (m,)
		ones = np.expand_dims(ones, 1) #give an extra dimention along the col direction
		print(ones.shape)
		X = np.concatenate((ones, X), 1) #concatenate along the col direction

	def fit(self, X, y):
		'''
		X   -- the training data in the shape of m by n 
		y   -- the training label in the shape of m by 1
		
		linear Model:  y = X @ weights
		Normal Equation :  weight  =  (XT@X + lambda *I)^-1 @ XT @ Y

		return:
		weights -- in the shape of n by 1
		'''
		#if we want to include an extra intercept， we need to introduce an extra column with all the ones to the X
		self.mRows = X.shape[0]
		if self.intercept:
			self.addOneToFeature(X)

		
		self.nFeatures = X.shape[1]			
		firstMatrix =np.linalg.inv((X.T @ X + self.a * np.eye(self.nFeatures)))
		self.theta = firstMatrix @ X.T @ y
		# print(self.theta.shape)

	def predict(self, testData):
		'''
		testData -- in the shape of k by n

		return:
		yPred    -- in the shape of k by 1
		'''

		if self.intercept:
			self.addOneToFeature(testData)

		return testData @ self.theta

	def plot(self, X, y, ypred):
		y = np.expand_dims(y, 1)
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.scatter3D(X[:,0], X[:,1], y, label = 'groundTruth')
		
		ypred = np.expand_dims(ypred, 1)
		ax.scatter3D(X[:,0], X[:,1], ypred, marker = '^', s = 100, label = 'predict')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('Value')
		ax.legend()

		plt.show()

if __name__ == '__main__':
	X, y = make_regression(n_samples = 100, n_features = 2)
	print(X.shape)


	ridge = ridgeRegression(2, True)
	ridge.fit(X,y)
	ypred = ridge.predict(X)

	ridge.plot(X, y, ypred)


