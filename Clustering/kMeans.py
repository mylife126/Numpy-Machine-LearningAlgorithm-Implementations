import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.datasets import make_blobs
np.random.seed(123)
class KMeans():
    def __init__(self, k, X):
        '''
        k  -- numbers of clusters 
        X  -- the given training data'''
        self.k = k
        self.mRows = X.shape[0]
        self.nFeatures = X.shape[1]
        self.centroids = np.zeros((self.k, self.nFeatures)) 
    
    def euclidian(self, pointA, pointB):
        '''calculate the euclidian distance between two points
        pointA -- in the shape of 1 by n
        pinttB -- in the shape of 1 by n'''
        
        dist = 0
        assert len(pointA) == len(pointB)
        for coord in range(len(pointA)):
            dist += (pointA[coord] - pointB[coord])**2
            
        return np.sqrt(dist)

    def fit(self, X, maxIter):
        """
        Fits the k-means model to the given dataset.
        Implement the core algorithm here.
        data [[x, y],
              [x, y],]
              
              
        would return the centriod founded
        centroid -- k by n 
        """
        from collections import defaultdict
        
        assigned = defaultdict()
        center2points = defaultdict(list)
        for i in range(maxIter):
            print("training at iter, " ,i)
            for sample in X:
                # print(sample)
                distances = []
                for k in range(self.k):
                    whichCenter = self.centroids[k]
                    # print(sample, whichCenter)
                    dist = self.euclidian(sample, whichCenter)
                    distances.append(dist)
                    
                assignedCenter = np.argmin(distances)
                # print(assignedCenter)
                center2points[assignedCenter].append(sample)
                
            for center in center2points:
                allThePointsAssigned = center2points[center]
                howMany = len(allThePointsAssigned)
                temp = [0 for i in range (self.nFeatures)]
                for points in allThePointsAssigned:
                    temp += points
                
                newCenter = [coord / howMany for coord in temp]
                self.centroids[center] = newCenter
        print("done")
        
    def plot_clusters(self, data):
        plt.figure(figsize=(12,10))
        plt.title("Initial centers in black, final centers in red")
        plt.scatter(data[:,0], data[:,1], c=y)
        plt.scatter(self.centroids[:, 0], self.centroids[:,1], c='r')
        plt.show()

        
if __name__ == '__main__':
    X, y = make_blobs(centers=4, n_samples=1000)
    print(f'Shape of dataset: {X.shape}')

    fig = plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title("Dataset with 4 clusters")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()
    kmeans = KMeans(4, X)
    kmeans.fit(X, 100)
    kmeans.plot_clusters(X)