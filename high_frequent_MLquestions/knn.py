import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        """
        k: number of neighbors
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Since KNN is lazy learning, just memorize the training data
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        """Compute L2 distance between two vectors"""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))  # (n_train,)

    def predict(self, X_test):
        """
        Predict labels for X_test
        X_test: (m, n_features)
        return: (m,)
        """
        preds = []
        for x in X_test:
            # 1️⃣ compute distance to all training points
            distances = self.euclidean_distance(self.X_train, x)
            # 2️⃣ find the indices of k nearest neighbors
            k_idx = np.argsort(distances)[:self.k]
            # 3️⃣ get corresponding labels
            k_neighbor_labels = self.y_train[k_idx]
            # 4️⃣ vote: choose most common label
            label = Counter(k_neighbor_labels).most_common(1)[0][0]
            preds.append(label)
        return np.array(preds)

if __name__ == "__main__":
    X_train = np.array([
        [1, 2], [2, 3], [3, 3],  # label = 0 (蓝)
        [6, 6], [7, 7], [8, 7],  # label = 1 (红)
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[5, 5]])

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    print("Prediction:", pred)