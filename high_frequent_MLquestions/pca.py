"""
Xc = X - mean(mean) for normalization
find the eigen vectors with top K component
SVD(Xc) -> VT S V  _> k by n, n by 1, n by k

find k <- sum(s**2) till K / sum(s**2) <= 0.95
-> ratio = s**2 / sum(s**2)
distribution/cdf = np.cumsum(ratio)
            for idx, p in enumerate(cdf):
                if p < r:
                    temp.append((p, idx))
            to find the idx of the top K

transformation: A = X @ V_k  m by n @ n by k -> m by k

"""
import numpy as np

class PCA:
    def __init__(self, X):
        """
        PCA implementation using SVD.
        X : ndarray of shape (m, n)
        """
        self.X = X
        self.U = None
        self.V = None
        self.s = None
        self.variance_ratio = None
        self.components = None
        self.mean_ = None
        self.X_centered = None

    def fit(self):
        """Center data and perform SVD"""
        # Center
        self.mean_ = np.mean(self.X, axis=0)
        self.X_centered = self.X - self.mean_

        # SVD decomposition
        # np.linalg.svd returns: X = U @ np.diag(s) @ Vt
        U, s, Vt = np.linalg.svd(self.X_centered, full_matrices=False)
        self.U = U
        self.s = s
        self.V = Vt.T  # V has shape (n, n)

        # Variance ratio
        self.variance_ratio = (s**2) / np.sum(s**2)

    def get_how_many_k(self, ratio_threshold):
        """
        Given a cumulative variance threshold (e.g. 0.95),
        return the smallest number of components K satisfying that ratio.
        """
        cumulative = np.cumsum(self.variance_ratio)
        K = np.searchsorted(cumulative, ratio_threshold) + 1
        print("number of components K =", K)
        return K

    def transform(self, X, K):
        """Project X into the principal component space"""
        if isinstance(K, int) and K <= len(self.s):
            transformation_matrix = self.V[:, :K]
        elif isinstance(K, float) and 0 < K < 1:
            K = self.get_how_many_k(K)
            transformation_matrix = self.V[:, :K]
        else:
            raise ValueError("K must be an int or a float in (0,1)")

        self.components = transformation_matrix
        X_centered = X - self.mean_
        A = X_centered @ transformation_matrix  # (m, n) @ (n, K) -> (m, K)
        return A

    def inverse_transform(self, A):
        """
        Reconstruct data from projected space A back to original space.
        """
        X_reconstructed = A @ self.components.T + self.mean_
        return X_reconstructed


# ====== TEST CODE ======
if __name__ == "__main__":
    X = np.random.randn(10000, 100)  # (m=10000, n=100)
    mypca = PCA(X)
    mypca.fit()

    # Project to retain 95% variance
    A = mypca.transform(X, 0.95)
    print("Projected shape:", A.shape)

    # Reconstruct back to original space
    X_approx = mypca.inverse_transform(A)
    print("Reconstructed shape:", X_approx.shape)

    # Reconstruction error
    err = np.linalg.norm(X - X_approx) / np.linalg.norm(X)
    print("Relative reconstruction error:", err)