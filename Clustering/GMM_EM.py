import numpy as np
from scipy.special import logsumexp

class GMM(object):
	def __init__(self, kCluster, X):
		'''init the class, 
		x        -- data, mSamples, nFeatures
		kCluster -- define how many clusters needed
		X        -- training data
		mu       -- clusters shape, in the shape of [k, n]
		sig      -- clusters covariance, in the shape of [n, n] by k 
		pi       -- priors, which is the phi_i = clusterJ, in the shape of [k,]
		Q        -- guessed posterious, in the shape of [m, k], each sample yields a prob for each cluster
		'''
		self.x = X
		self.k  = kCluster
		self.mRows = X.shape[0]
		self.nFeatures = X.shape[1]

		#initialize the mu matrix; in the uniform distribution at the first
		#in the dimension of [k, nFeatures]; meaning each data point has one center
		self.mu = np.random.uniform(-10,10,self.nFeatures * self.k).reshape(self.k, self.nFeatures) 

		#generate the priors for phi, in the shape of [k,]
		self.phi = np.random.rand(self.k) / sum(np.random.rand(self.k))

		#generate the covariance; in the shape n by n, but since we have k cluster, we need to have k covariance
		self.sig = np.array([np.identity(self.nFeatures) for k in range(self.k)])

		#initialize the Q matrix, which is the W{j, i}  for the guessed P(y_i | x(i))
		self.Q = np.zeros((self.mRows, self.k))

	def gaussian(self, xSample, phi_k, sig_k, mu_k):
		'''calculate the log prob of the multi- variate gaussian
		
		xSample -- in shape [1, n]
		phi_k   -- in shape 1
		sig_k   -- in shape n by n 
		mu_k    -- in shape 1 by n
		'''
		# assert len(xSample) == sig_k.shape[0] == sig_k.shape[1] == len(mu_k)
		# det = np.linalg.det(sig_k)

		# # a = -(np.log((2 * np.pi)**1/2) + np.log((det)** 1/2))   # a value
		# a = 1/ ((2 * np.pi) ** 1/2 * det**1/2)
  #       #1 by n @ n by n @ n by ==> return size 1
		# # b = -1/2 * (xSample - mu_k) @ np.linalg.inv(sig_k) @ (xSample - mu_k).T
		# b = np.exp(-1/2 * (xSample - mu_k) @ np.linalg.inv(sig_k) @ (xSample - mu_k).T)
		# # c = np.log(phi_k)
		# c = phi_k
		# # print(a, b, c)
		# return a* b *c
		"""
		Compute log N(x_i | mu, sigma)
		"""
		n = len(mu_k)
		a = n * np.log(2 * np.pi)
		_, b = np.linalg.slogdet(sig_k)

		y = np.linalg.solve(sig_k, xSample - mu_k)
		c = np.dot(xSample - mu_k, y)
		return -0.5 * (a + b + c) + np.log(phi_k)

	def Estep(self):
		'''in the estep, we calculate the upper bound of the estimated p(y_j | x_i) for every cluster
		w{j,i} = p ( x_i | y_j)  * phi(k)   /  sum(p ( x_i | y_j)  * phi(k))

		for the conditional prob, 
		1/ ((2pi)^1/2 * det(sig)^1/2) * exp(-1/2 - (x - mu_j)T @ inv(sig) @ (x - mu_j)) * phi_j ,

		1/ ((2pi)^1/2 * det(sig)^1/2)                   -- in dimension 1
		exp(-1/2 - (x - mu_j)T @ inv(sig) @ (x - mu_j)) -- in dimension 1
		phi_j in dimension k                            -- in dimension 1 
		marginal                                        -- array, in dimension [1, k], p(x|y_j) * phi(y_j)

		return                                          --prob [1, k] for each sample'''

		mSamples = self.mRows
		temp = []

		#for each sample, we calculate the conditional prob for each cluster
		for i in range(mSamples):
			sample = self.x[i,:]
			marginal = []
			#for each cluster we calculate the conditional prob
			for k in range(self.k):
				phi_k = self.phi[k]                                        #phi = int[k]
				sig_k = self.sig[k, :, :]                                  #sig = int[k][n][n]
				mu_k = self.mu[k, :]                                       #mu = int[k][n]
				conditional_k = self.gaussian(sample, phi_k, sig_k, mu_k)
				marginal.append(conditional_k)

			sumOfMarginal = logsumexp(marginal)                            #因为分母是所有的marginalPrb和，但是我们的Gaussian求出来的是log后的，所以需要先exp再求和
			conditionalProbForK = [i - sumOfMarginal for i in marginal]    #log(x/ sum(x)) = log(x) - logsum(x)
			assert len(conditionalProbForK) == self.k
			self.Q[i, :] = conditionalProbForK                             #Q = int[m][k]

	def Mstep(self):
		'''update the prior of each cluster
		self.Q       --conditional prob P(y | x) for each sample m by k
		self.phi     --prior prior for each cluster, 1 by k 
		'''

		denom = np.sum(self.Q, 0)   # 1 by k
		self.phi =  denom / self.mRows

		temp = []
		for k in range(self.k):
			''' mu_k = sum(q_k * x_i)  /  sum(q_k)'''
			numerator = 0
			#sum up all the (prob in each cluster times the x in each sample)
			for i in range(self.mRows):
				numerator += self.Q[i, k] * self.x[i, :]  # 1 * 1 by n, sample to sample
				# print(q_j.shape)
			temp.append(numerator/ denom[k])

		assert len(self.mu) == len(temp) 
		self.mu = np.array(temp)

		temp = []
		for k in range(self.k):
			numerator = 0
			for i in range(self.mRows):
				assert self.mu[k].shape == self.x[i,:].shape
				xMinusMu_j = np.array([self.x[i, :] - self.mu[k]])
				numerator += self.Q[i,k] * (xMinusMu_j.T @ xMinusMu_j)  # 1 by 1 * n by 1 @ 1 by n
			temp.append(q_j / denom[k])

		temp = np.array(temp)    # k by n by n
		assert temp.shape == self.sig.shape
		self.sig = temp

	def train(self, iterations, verbose):
		for i in range(iterations):
			if verbose:
				print("iter: ", i)
			self.Estep()
			self.Mstep()

	def predict(self, data):
		sig = self.sig   # k by n by n
		mu = self.mu     # k by n
		phi = self.phi   # k

		predictions = [[0 for c in range(self.k)] for i in range(data.shape[0])]   #m by k
		for c in range(self.k):
			for i in range(data.shape[0]):
				thisSample = data[i,:]
				thisSig = sig[c, :, :]
				thisMu = mu[c, :]
				thisPhi = phi[c]
				posterious = self.gaussian(thisSample, thisPhi, thisSig, thisMu)
			predictions[i][c] = posterious
		return predictions

# if __name__ == '__main__':
# 	data = np.random.randn(4,3)
# 	gmm =GMM(2, data)
# 	gmm.train(1, False)
# 	# print(gmm.predict(data))
# 	print(gmm.sig)












			



