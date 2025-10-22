

import numpy as np
from sklearn.datasets import make_classification


class RegressionGLM():
    def __init__(self, X, y, reg_lambda, which_regression, lr):
        """
        reg_lambda -- the regularization parameter, set to be 0 by default if no regularization is desired
        """
        self.X = X
        self.reg_lambda = reg_lambda
        self.which_regression = which_regression
        self.y = y
        self.thetas = np.random.rand(self.X.shape[1])
        self.lr = lr

    def ridge_regression(self, X=None, y=None):
        """
        for linear regression
        X in the shape of m by n
        y in the shape of m by 1
        """
        if X is None and y is None:
            X = self.X
            y = self.y
        m = X.shape[0]
        h = X @ self.thetas  # m by n dot n by 1 -> m by 1
        diff = h - y  # m by 1
        loss = np.sum(diff ** 2) * 0.5 / m + self.reg_lambda * np.sum(self.thetas ** 2) / m  # axis = 0

        # n by m @ m by 1 -> n by 1  + n by 1 -> n by 1
        grad = 1 / m * X.T @ diff + self.reg_lambda / m * self.thetas
        return loss, grad, h

    def lasso_regression(self, X=None, y=None):
        """
        J(θ) = (1/2m)*||Xθ - y||² + (λ/m)||θ||₁
        ∇J(θ) = (1/m)*Xᵀ(Xθ - y) + (λ/m)*sign(θ)
        """
        if X is None and y is None:
            X = self.X
            y = self.y

        m = X.shape[0]
        h = X @ self.thetas
        diff = h - y
        loss = 0.5 / m * np.sum(diff ** 2) + self.reg_lambda / m * np.sum(np.abs(self.thetas))

        # n by m @ m by 1 -> n by 1 + n by 1 = n by 1
        grad = 1 / m * X.T @ diff + self.reg_lambda / m * np.sign(self.thetas)
        return loss, grad, h

    def sigmoid(self, X):
        h = X @ self.thetas  # m by n * n by 1 -> m by 1
        return 1 / (1 + np.exp(-h))

    def logistic_regression(self, X=None, y=None):
        if X is None and y is None:
            X = self.X
            y = self.y

        m = X.shape[0]
        p = self.sigmoid(X)
        eps = 1e-9
        diff = p - y
        loss = -1 / m * np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss += self.reg_lambda / m * np.sum(self.thetas ** 2)

        # n by m @ m by 1 -> n by 1 +
        grad = 1 / m * X.T @ diff + self.reg_lambda / m * (self.thetas)

        return loss, grad, p

    def gradient_descent(self, grad):
        """
        This implementation uses gradient descent to update the parameters.
        If you use the full batches' gradients for descending, then it is the batch gradient descent;

        SGD: uses random single data point to calculate the gradient.
        """
        self.thetas -= self.lr * grad

    def fit(self, iterations=100):
        for i in range(iterations):
            if self.which_regression == "regression":
                loss, grad, pred = self.ridge_regression()
                print(f"iteration {i} loss: {loss}")
                self.gradient_descent(grad)

            elif self.which_regression == "lasso":
                loss, grad, pred = self.lasso_regression()
                self.gradient_descent(grad)
                print(f"iteration {i} loss: {loss}")

            elif self.which_regression == "logistic":
                loss, grad, pred = self.logistic_regression()
                self.gradient_descent(grad)
                print(f"iteration {i} loss: {loss}")

            else:
                raise ValueError("Unsupported regression type")

    def fit_sgd(self, iterations=100):
        for i in range(iterations):
            idx = np.random.randint(0, self.X.shape[0] - 1)
            x_idx = self.X[idx: idx + 1]  # cannot use self.X[idx]. this will break the dim to be (n,) instead of 1 by n
            y_idx = self.y[idx: idx + 1] # cannot use self.X[idx]. this will break the dim to be (n,) instead of 1 by n
            if self.which_regression == "regression":

                loss, grad, pred = self.ridge_regression(x_idx, y_idx)
                print(f"iteration {i} loss: {loss}")
                self.gradient_descent(grad)

            elif self.which_regression == "lasso":
                loss, grad, pred = self.lasso_regression(x_idx, y_idx)
                self.gradient_descent(grad)
                print(f"iteration {i} loss: {loss}")

            elif self.which_regression == "logistic":
                loss, grad, pred = self.logistic_regression(x_idx, y_idx)
                self.gradient_descent(grad)
                print(f"iteration {i} loss: {loss}")

            else:
                raise ValueError("Unsupported regression type")
    '''
    def mini_bath_sgd(self, iterations=100):
        batch_size = 64
        for epoch in range(iterations):
            for start in range(0, self.X.shape[0], batch_size):
                X_batch = X[start:start + batch_size]
                y_batch = y[start:start + batch_size]
                grad = compute_grad(X_batch, y_batch)
                theta -= lr * grad
                
        pass
    '''

    def predict(self, X, threshold=0.5):
        """
        Generate predictions for input X depending on the regression type.
        For logistic regression, returns class labels (0/1).
        For ridge/lasso, returns continuous predictions.
        """
        if self.which_regression == "logistic":
            probs = self.sigmoid(X)
            preds = (probs >= threshold).astype(int)
            return preds
        elif self.which_regression in ["regression", "lasso"]:
            return X @ self.thetas
        else:
            raise ValueError("Unsupported regression type")


import numpy as np


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    手动实现 train-test split
    ------------------------------------
    X : np.ndarray, shape (m, n)
    y : np.ndarray, shape (m,)
    test_size : float, 比例（0~1），表示测试集占比
    shuffle : bool, 是否打乱
    random_state : int, 可选，用于随机种子
    ------------------------------------
    return: X_train, X_test, y_train, y_test
    """
    m = X.shape[0]
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]

    test_count = int(m * test_size)
    X_test = X[:test_count]
    y_test = y[:test_count]
    X_train = X[test_count:]
    y_train = y[test_count:]
    return X_train, X_test, y_train, y_test


def accuracy(ypred, y):
    count = 0
    for (i, j) in zip(ypred, y):
        if i == j:
            count += 1
    return count / len(y)


if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    type_ = "logistic"
    reg_lambda = 0.5
    lr = 0.01
    myModel = RegressionGLM(X_train, y_train, reg_lambda, type_, lr)
    myModel.fit(iterations=1000)
    predictions = myModel.predict(X_test, threshold=0.5)
    print(f"Accuracy: {accuracy(predictions, y_test)}")

    myModel = RegressionGLM(X_train, y_train, reg_lambda, type_, lr)
    myModel.fit_sgd(iterations=1000)
    predictions = myModel.predict(X_test, threshold=0.5)
    print(f"Accuracy: {accuracy(predictions, y_test)}")
