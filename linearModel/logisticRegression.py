import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

class logistic(object):
	def __init__(self, intercept):
		'''
		1. Logistic regression: 1/ (1 + exp(-X @ thetaT))
		2. Z = -X @ thetaT
		3. gradient: (y - ypred) * x
		
		alpha          -- the penalty term, int
		regularization -- boolean, l1 or l2
		theta          -- the weights, in the shape of n by n
		intercept      -- boolean, fit the intercept or not
		'''

		# self.a = alpha
		# self.reg = regularization
		self.theta = None
		self.mRows = 0
		self.nFeatures = 0
		self.intercept = intercept

	def sigmoid(self, z):
		'''
		z -- the input vector, (m,)
		retrun:
		the probability, in the shape of (m,)
		'''
		return 1 / (1 + np.exp(-z))

	def addOneFeature(self, X):
		'''
		ones -- a vector of m rows
		return:
		concated X by added one more feature --  m by (n + 1)
		'''
		ones = np.ones(X.shape[0])
		ones = np.expand_dims(ones, 1)
		X = np.concatenate((ones, X), 1)
		return X

	def mseLoss(self, y, ypred):
		'''
		y     -- the given label, in the shape of m by 1
		ypred -- the predictoin, in the shape of m by 1
		'''
		loss = 0
		for (yi, ypredi) in zip(y, ypred):
			loss += (yi - ypredi)**2

		return loss / len(y)

	def gradient(self, x, y, ypred):
		'''
		x             -- the training data
		y             -- the training label
		gradient:  sum((y_i - ypred_i) * x_i) for i in range of msamples
		in matrix form:  X.T @ (Y - Ypred)
		Also, by derivation, this is derived from the maximum likilhood, thus, we should operate
		gradient ascent. However, let us change the gradient to be negative to perform the gradient descent
		return:
		gradient      -- in the shape of n by 1
		'''
		print(X.shape)
		return -X.T @ (y - ypred)

	def fit(self,X,y,maxIter,lr):
		'''
		X           - training data, in the shape of m by n
		y           - trianing label, in the shape of m by 1
		self. theta - weights, in the shape of n by 

		'''
		self.mRows = X.shape[0]
		if self.intercept:
			X = self.addOneFeature(X)

		self.nFeatures = X.shape[1]
		self.theta = np.random.rand(self.nFeatures)
		print("training")
		for i in range(maxIter):

			z = X @ self.theta    #m by n by n b 1 = > m by 1
			ypred = self.sigmoid(z)
			loss = self.mseLoss(y, ypred)
			print("iter {} loss {}".format(i,loss))

			g = -X.T @ (y - ypred)
			self.theta -= lr * g

		print("done")

	def predict(self, testpoint, theshould):
		if self.intercept:
			testpoint = self.addOneFeature(testpoint)
			
		z = testpoint @ self.theta
		prob = self.sigmoid(z)
		
		for i in range(len(prob)):
			if prob[i] >= theshould:
				prob[i] = 1
			else:
				prob[i] = 0
		return prob

	def plot(self, ypred, X, y):
		f, ax = plt.subplots(figsize=(8, 6))
		ax.scatter(X[:,0], X[:,1], c=y,  marker = 'o', s = 100, label = 'groundTruth')
		ax.scatter(X[:,0], X[:,1], c=-ypred,  marker = '^', label = 'prediction')
		ax.legend()
		plt.title("Accuracy for classification is {}. Missed classifications are those not filled with Δ".format(self.accuracy(ypred,y)))
		plt.show()

	def accuracy(self, ypred, y):
		count = 0
		for (i, j) in zip(ypred, y):
			if i == j:
				count+=1
		return count / len(y)

if __name__ == '__main__':
	X, y = make_classification(n_samples = 100, n_features = 2, n_informative=2, n_redundant=0)
	logistic = logistic(True)
	logistic.fit(X, y, 500, 0.001)
	ypred = logistic.predict(X, 0.5)
	logistic.plot(ypred, X, y)














