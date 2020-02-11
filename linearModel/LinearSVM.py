import numpy as np

class (object):
	'''This is the numpy implmentation of the soft linear svm using hinge loss.
	The regularization comes with the norm2 regularization

	L(𝑦,𝑋,𝜃,𝜆)=sum{max{1−𝑦⋅𝜃𝑇𝑋,0}}+𝜆/2||𝜃||

	𝑣=𝟙{𝑦∗(𝑋𝜃))<=1}    
	dL/d𝜃= sumcol{−𝑦∗𝑣∗𝑋}+𝜆𝜃
	'''
	def __init__(self, reg, X, y, intercept):
		'''
		X          -- the training data, in the shape of m by n
		y          -- the ground truth, in the shape of m by 1
		reg        -- lambda, int
		theta      -- the weights initialized, in the shape of n by 1
		intercept  -- boolean, if we want to create an extra feature for the intercept
		'''
		self.X = X
		self.y = y
		self.reg = reg
		self.mRows = X.shape[0]
		self.nFeatures = X.shape[1]
		self.theta = np.random.randn((nFeatures,1))
		self.intercept = intercept

	def fit(self, maxIters, lr):
		if self.intercept:
			ones = np.ones((self.mRows, 1))
			self.X = np.concatenate((ones, self.x), 1)

		for i in range(maxIters):
		    ypred = self.x @ self.theta
		    assert ypred.squeeze().shape == self.y.shape, "shape of ypred {} and shape of y".format(ypred.squeeze().shape, self.y.shape)
		    Yypred   = self.y * ypred.squeeze()
		    loss  = np.max(1 - Yypred).sum()

		    g = self.gradient(Yypred)
		    self.theta -= lr * g

		    print("iter {:.2f} loss {:.2f}".format(i, loss))

	def gradient(self, Yypred):
		'''
		𝑣=𝟙{𝑦∗(𝑋𝜃))<=1}           -- vector by shape of m, contains 1 or 0
	    dL/d𝜃= sumcol{−𝑦∗𝑣∗𝑋}+𝜆𝜃  -- y*X. T @ v, n by 1 
		'''
		indicator = Yypred <= 1 
		g = (self.y * self.X).T @ indicator + self.reg * self.theta
		return g

    def calIndicator(self, singleVal):
    	if singleVal > 0 :
    		return 1
    	else:
    		return 0

	def predict(self, testX):
		if self.intercept:
			ones = np.ones((testX.shape[0], 1))
			testX = np.concatenate((ones, testX), 1)

		ypred = testX @ self.theta
		prediction = list(map(self.calIndicator, ypred))
		return np.array(prediction)








