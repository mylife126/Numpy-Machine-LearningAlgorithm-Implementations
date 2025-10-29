"""
Step 1 initialize the clusters
Step 2 for each data point x_i <with n features> calculate the euclidean distance between the clusters
    -> assign the x_i to the right cluster
Step 3 after the assignment, reduce the intradistance within each cluster and update the cluster coordinates

Key point:
1. check the dimensionality to be matched with the X <m, n> m samples, n features
2. euclidean distance between each pair of clusters calculation
3. hash to save the distance to points

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

np.random.seed(123)
from collections import defaultdict


class KMeans():
    def __init__(self, n_clusters, X):
        self.n_clusters = n_clusters
        self.data = X
        self.m, self.n = X.shape[0], X.shape[1]
        self.centroids = self.init_k_kmeanspp()

    def init_k(self):
        all_zeros = np.zeros((self.n_clusters, self.m, self.n))  # k by n
        return all_zeros

    def init_k_kmeanspp(self):
        """
        This is the better form of the initialization of the k-means algorithm.
        The concept is, we want to avoid the pure randomization or zero-init to have the case where
        all init centroids are heavily clustered, this would cause the local optimal results

        Thus, we want to make sure the centroids are sparse as much as possible.

        Step 1: randomly select the point from the X as the starting point as the centroid
        Step 2: for every data point, calculate the euclidean distance between the chosen centroid
                D(x) = argmin ( x_i - c) ** 2
                Chose the minimal distance between x_i and c

        Step 3:
        Calculate the probability for each point:
            P(x_i) = D(x_i)^2 / sum(D(x_j)^2)

        This ensures that points far from the existing centroids
        are more likely to be selected as the next centroid. But this is the discrete probability meaning that
        we only know the likelihood of x_i = 0.1, but if we want to use its probability as the weight to select, meaning
        that if we randomly generate a number (0, 1] such as 0.75, we do not know which x_i to select because this is equal to P(r <= x_i)
        which is the CDF.

        The CDF tells a slot of P(X <= x_i)

        •	P(x_i) 表示“单个样本的独立概率”，属于 离散概率分布 (PMF)；
        •	但当你想“根据概率抽样”时，仅知道单个 P(x_i) 不够；
        •	因为随机数 r 是一个连续值，我们要知道 r 落在哪个区间；
        •	所以把概率累加成 CDF（累积分布函数），
        让每个样本在 [0,1) 上占一个“区间槽”；
        •	当随机数 r 落入这个槽，就选中对应的样本。

        Step 4:
        To sample the next centroid:
            - Compute the cumulative distribution function (CDF) of P(x_i).
            - Randomly draw a number r in [0, 1).
            - Find the first x_i where CDF(x_i) ≥ r.
              That x_i becomes the next centroid.

        Key point:
        We want to pick the next centroid *probabilistically*,
        favoring points farther from existing centroids,
        but still allowing randomness to avoid deterministic bias.

        Step 4: repeat step 2 and 3
        """

        centroids = []
        starting = np.random.randint(self.m - 1)
        centroids.append(self.data[starting])
        for _ in range(1, self.n_clusters):
            latest_centroid = centroids[-1]
            distances = self.euclidian(self.data, latest_centroid)  # m by 1
            total_distance = np.sum(distances)

            probabilities = distances / total_distance  # m by 1 [D(x_1)**2/ total D, D(x_2)**2/ total D, ...]
            cdf = np.cumsum(probabilities)
            r = np.random.rand()
            temp = []
            for idx, p in enumerate(cdf):
                if p < r:
                    temp.append((p, idx))

            temp = sorted(temp, key=lambda x: -x[0])
            index = temp[0][-1]
            next_centroid = self.data[index]
            centroids.append(next_centroid)
        return np.array(centroids)

    def euclidian(self, x, y):
        """
        x y are two data points represented as the numpy array, n by m or single (n,)
        such that the calculation of (x - y)**2 is valid
        """
        if len(x.shape) == 1:
            # this is a single point, the dimension is (n,), so we sum along the row to get the sum of all feature;s distance
            # x - y => ([1,3,4] - [2,4,5]) **2, => [-1, -1, -1] **2 =>[1, 1, 1] => sum along the column 1+1+1 =3
            distance = np.sum((x - y) ** 2, axis=0)
        else:
            # this is the batch data means multiple data samples
            # then x - y gives the m by n , and we want to find all samples feature wise distance so sum along the column
            # this would give (m,) data
            distance = np.sum((x - y) ** 2, axis=1)
        return np.sqrt(distance)

    def fit(self, X, max_iter=100):
        for iter in range(max_iter):
            center2points = defaultdict(list)
            # Step 1 : E, get the centroids updated
            distances = []
            for k in self.centroids:
                all_points_distances = self.euclidian(self.data, k)
                distances.append(all_points_distances)
            D = np.stack(distances)  # shape (K, m)
            index = np.argmin(D, axis=0)  # shape (m,) telling you each point's assigned centroid index
            for idx, point in zip(index, self.data):
                center2points[idx].append(point)

            # step 2, for each cluster, optimize for the new centroids
            # calculate the average
            for center in center2points:
                all_points = center2points[center]  # m, n
                feature_wise_avg = np.mean(all_points, axis=0)  # avg out along the row -> shape (n,)
                new_center = feature_wise_avg
                self.centroids[center] = new_center

    def predict(self, X):
        distances = []
        for k in self.centroids:
            all_points_distances = self.euclidian(X, k)
            distances.append(all_points_distances)
        D = np.stack(distances)  # shape (K, m)
        index = np.argmin(D, axis=0)
        return index

    def plot_clusters(self, data, x):
        plt.figure(figsize=(12, 10))
        plt.title("Initial centers in black, final centers in red")
        plt.scatter(data[:, 0], data[:, 1], c=y, marker="o")
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='r', marker='^')

        if x is not None:
            plt.scatter(x[:, 0], x[:, 1], c="r", marker="H")
        plt.show()


if __name__ == '__main__':
    X, y = make_blobs(centers=5, n_samples=1000)
    kmeans = KMeans(5, X)

    fig = plt.figure(figsize=(8, 6))

    kmeans = KMeans(4, X)
    kmeans.fit(X, 100)


    x = np.array([[11, 3], [-1, 3.1]])
    index = kmeans.predict(x)

    kmeans.plot_clusters(X,x)
