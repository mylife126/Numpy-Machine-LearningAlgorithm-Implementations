import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.datasets import make_blobs
np.random.seed(123)
class KMeans():
    def __init__(self, k, X):
        '''
        k              -- numbers of clusters, in the shape of 1 by k
        X              -- the given training data, in the shape of m by n
        self.centroids -- the predefined centriod, with the shape of k by n'''
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

        centroid        -- k by n 
        center2points   -- map<Integer, list<Integer>>, for each centeroid, we record the assigned data points to this cluster

        """
        from collections import defaultdict
        
        # assigned = defaultdict()
        center2points = defaultdict(list)
        for i in range(maxIter):
            print("training at iter, " ,i)
            #STEP1 for each data point, we need to first give them a correct assignment of which centroid
            for sample in X:
                # print(sample)
                distances = []
                #try out every centroid, by calculating the distance between each
                for k in range(self.k):
                    whichCenter = self.centroids[k]
                    # print(sample, whichCenter)
                    dist = self.euclidian(sample, whichCenter)
                    distances.append(dist)
                #after find all the distanced between this datapoint and centroids, we only want to have
                #the minimal distance. Argmin is used to find the index
                assignedCenter = np.argmin(distances)

                #assign the datapoint to its assigned centorid
                center2points[assignedCenter].append(sample)
            
            #STEP2 minimizing the intradistance between points to the cluster  
            #within each cluster, fix the centroid by averaging the points assigned to this cluster 
            for center in center2points:
                allThePointsAssigned = center2points[center]         #get the list out of from the hash
                howMany = len(allThePointsAssigned)                  #total points assigned to this cluster
                runningSum = [0 for i in range (self.nFeatures)]     #assigned an empty list to contain the same size of feature
                for points in allThePointsAssigned:                  #accumulate the features within the centroid
                    runningSum += points
                # print(temp)
                newCenter = [coord / howMany for coord in runningSum ]#take the average
                self.centroids[center] = newCenter                    #update the centroid
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

    # fig = plt.figure(figsize=(8,6))
    # plt.scatter(X[:,0], X[:,1], c=y)
    # plt.title("Dataset with 4 clusters")
    # plt.xlabel("First feature")
    # plt.ylabel("Second feature")
    # plt.show()
    kmeans = KMeans(4, X)
    kmeans.fit(X, 100)
    kmeans.plot_clusters(X)