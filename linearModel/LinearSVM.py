import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

class SVM(object):
	'''This is the numpy implmentation of the soft linear svm using hinge loss.
	The regularization comes with the norm2 regularization

	L(𝑦,𝑋,𝜃,𝜆)=sum{max{1−𝑦⋅𝜃𝑇𝑋,0}}+𝜆/2||𝜃||

	𝑣=𝟙{𝑦∗(𝑋𝜃))<=1}    
	dL/d𝜃= sumcol{−𝑦∗𝑣∗𝑋}+𝜆𝜃
	'''
	def __init__(self, reg, X, y, intercept):
		'''
		X          -- the training data, in the shape of m by n
		y          -- the ground truth, in the shape of m,
		reg        -- lambda, int
		theta      -- the weights initialized, in the shape of n by 1
		intercept  -- boolean, if we want to create an extra feature for the intercept
		'''
		self.X = X
		self.y = np.expand_dims(y, 1)   #expand the y in the shape of (m,1) from (m,)
		self.reg = reg
		self.mRows = X.shape[0]
		self.nFeatures = X.shape[1]
		self.theta = None
		self.intercept = intercept

	def fit(self, maxIters, lr):
		'''
		L(𝑦,𝑋,𝜃,𝜆)=sum{max{1−𝑦⋅𝜃𝑇𝑋,0}}+𝜆/2||𝜃||

		To train the model
		maxIters         -- int, to train the model to what iterations
		lr               -- int, the learning rate
		ypred            -- array, m by 1
		Yypred           -- is 𝑦⋅𝜃𝑇𝑋, array, in the shape of m by 1 
		g                -- the gradient of the dl/ dtheta with norm2 regularazion, in the shape of n by 1
		'''
		if self.intercept:
			ones = np.ones((self.mRows, 1))
			self.X = np.concatenate((ones, self.X), 1)

		self.nFeatures = self.X.shape[1]
		self.theta = np.random.randn(self.nFeatures,1)

		for i in range(maxIters):
		    ypred = self.X @ self.theta
		    assert ypred.shape == self.y.shape, "shape of ypred {} and shape of y".format(ypred.squeeze().shape, self.y.shape)
		    
		    Yypred   = self.y * ypred                    #m by 1 * m by 1 = > m by 1
		    loss  = np.maximum(1 - Yypred, 0).sum()      #a value, the hinge loss

		    g = self.gradient(Yypred)            
		    self.theta -= lr * g

		    print("iter {:.2f} loss {:.2f}".format(i, loss))

	def gradient(self, Yypred):
		'''
		𝑣=𝟙{𝑦∗(𝑋𝜃))<=1}           -- vector by shape of m, contains 1 or 0
	    dL/d𝜃= sumcol{−𝑦∗𝑣∗𝑋}+𝜆𝜃  -- y*X. T @ v, n by 1 
		'''
		indicator = Yypred <= 1 
		g = -((self.y * self.X).T @ indicator) + self.reg * self.theta
		assert g.shape == self.theta.shape,"shape of g {} and shape of theta".format(g.shape, self.theta.shape)

		return g

	def calIndicator(self, singleVal):
		'''
		singleVal   -- int, a value in the prediction array
		'''
		if singleVal > 0 :
			return 1
		else:
			return 0

	def predict(self, testX):
		if self.intercept:
			ones = np.ones((testX.shape[0], 1))
			testX = np.concatenate((ones, testX), 1)

		ypred = testX @ self.theta
		prediction = list(map(self.calIndicator, ypred))   #use the mapping function to iterate every calculation wihitn the prediction array
		return np.array(prediction)

	def accuracy(self, prediction, y):
		count = 0
		for (p, yi) in zip(prediction, y):
			if p == yi :
				count +=1
		return count / len(y)

	def plot(self, X, y, ypred):
		f, ax = plt.subplots(figsize=(8, 6))
		ax.scatter(X[:,0], X[:,1], c=y,  marker = 'o', s = 100, label = 'groundTruth')
		ax.scatter(X[:,0], X[:,1], c=-ypred,  marker = '^', label = 'prediction')
		ax.legend()
		plt.title("Accuracy for classification is {}. Missed classifications are those not filled with Δ".format(self.accuracy(ypred,y)))
		plt.show()


if __name__ == '__main__':
	#X in shape of m by n, y in shape of m,
	X, y = make_classification(n_samples = 100, n_features = 2, n_informative=2, n_redundant=0)
	# print(X.shape, y.shape)
	svm = SVM(1.1,X, y, False)   #important, SVM doesnot need the intercept!
	svm.fit(500,0.001)
	prediction = svm.predict(X)
	svm.plot(X, y, prediction)











