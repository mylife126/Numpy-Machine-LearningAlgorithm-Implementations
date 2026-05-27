# ML Knowledge Interview Answer Scripts

Source file: fileciteturn3file0L1-L1

---

# Linear Regression

Linear regression minimizes the mean squared error between predictions and targets.

The model assumes a linear relationship between features and the output. Since the MSE loss is convex, gradient descent is guaranteed to converge to the global optimum.

In practice, linear regression can be solved either using the closed-form normal equation or iterative optimization methods such as gradient descent.

---

# Ridge Regression

Ridge regression extends linear regression by adding L2 regularization.

The L2 penalty discourages large weights, which helps reduce overfitting and improves model stability, especially when features are correlated.

Unlike L1 regularization, Ridge typically shrinks weights smoothly rather than driving them exactly to zero.

---

# Lasso Regression

Lasso regression uses L1 regularization, which encourages sparsity in the model parameters.

Because the L1 penalty pushes some weights exactly to zero, Lasso can perform implicit feature selection.

The optimization is more challenging because the L1 term is not differentiable at zero, so methods such as coordinate descent or proximal gradient descent are commonly used.

---

# Logistic Regression

Logistic regression is a linear model for binary classification.

Instead of predicting a continuous value directly, it applies a sigmoid function to map the linear output into a probability between 0 and 1.

The model is typically trained using cross-entropy loss, which provides better optimization behavior than MSE for classification tasks.

Since the objective is convex, gradient descent converges to the global optimum.

---

# Ridge Logistic Regression

Ridge logistic regression combines logistic regression with L2 regularization.

The regularization term prevents the model from overfitting by controlling the magnitude of the weights.

This is one of the most common forms of logistic regression used in industry because it balances predictive performance and stability.

---

# Softmax Regression

Softmax regression extends logistic regression to multi-class classification.

Instead of using a sigmoid function, it applies the softmax function to produce a probability distribution across all classes.

The model is trained using multi-class cross-entropy loss, which compares the predicted probability distribution with the one-hot encoded ground truth labels.

This is also the standard output layer formulation used in many neural networks.

---

# Poisson Regression

Poisson regression is a generalized linear model designed for count data.

The model assumes the target follows a Poisson distribution, and the expected count is modeled using an exponential link function.

It is commonly used in scenarios such as event counts, traffic estimation, and occurrence prediction.

---

# Generalized Linear Models (GLM)

Linear regression, logistic regression, softmax regression, and Poisson regression can all be viewed under the generalized linear model framework.

The main difference between these models lies in the link function, the loss function, and the choice of regularization.

Despite these differences, their gradients generally follow the same structure: prediction error multiplied by the feature matrix, plus optional regularization terms.

---

# Batch Gradient Descent

Batch Gradient Descent computes the gradient using the entire training dataset for every update step.

Because it uses the exact gradient, the optimization trajectory is smooth and stable.

However, it is computationally expensive for large datasets and requires the full dataset to fit into memory.

---

# Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent updates parameters using only a single randomly selected sample at each step.

This makes the updates very fast and allows the optimizer to escape shallow local minima due to gradient noise.

However, the optimization path becomes noisy and unstable, so the loss may oscillate instead of converging smoothly.

---

# Mini-Batch Gradient Descent

Mini-Batch Gradient Descent uses a small subset of samples for each parameter update.

It provides a balance between the stability of batch gradient descent and the speed of SGD.

This is the standard optimization approach used in modern deep learning because it works efficiently with GPU-based parallel computation.

---

# Gradient Descent Variants Summary

All gradient descent variants follow the same core update rule, but they differ in how the gradient is estimated.

Batch Gradient Descent uses all samples and produces stable updates.
Stochastic Gradient Descent uses one sample and produces noisy but fast updates.
Mini-Batch Gradient Descent strikes a balance between efficiency and stability, which is why it is most commonly used in practice.

---

# Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that projects data onto directions of maximum variance.

The core idea is to find a lower-dimensional representation that preserves as much information as possible.

These projection directions are called principal components, and they correspond to the eigenvectors of the covariance matrix.

---

# Why PCA Uses Variance Maximization

PCA assumes that directions with larger variance contain more useful information.

By projecting the data onto the directions with the highest variance, PCA preserves the most important structure while discarding low-variance noise.

---

# PCA and Eigenvectors

PCA can be derived as an optimization problem that maximizes projected variance subject to a unit norm constraint.

Solving this optimization leads to an eigenvalue problem.

The eigenvectors of the covariance matrix define the principal directions, while the eigenvalues represent the amount of variance explained by each component.

---

# PCA Projection and Reconstruction

After selecting the top principal components, we project the original data into a lower-dimensional space.

This compressed representation preserves most of the important information.

The original data can then be approximately reconstructed from the low-dimensional representation.

---

# PCA Using SVD

In practice, PCA is usually implemented using Singular Value Decomposition instead of directly computing eigenvectors of the covariance matrix.

SVD is numerically more stable and computationally more efficient, especially when the feature dimension is large.

The right singular vectors correspond to the principal directions, and the squared singular values correspond to explained variance.

---

# Choosing the Number of PCA Components

The number of principal components is typically selected using the cumulative explained variance ratio.

A common practice is to choose the smallest number of components that preserves around 95% of the total variance.

This provides a good balance between dimensionality reduction and information preservation.

---

# PCA Geometric Intuition

Geometrically, PCA finds orthogonal directions along which the data spread is largest.

Early principal components capture the main structure of the data, while later components often correspond to noise.

---

# Cross Validation

Cross Validation is a model evaluation technique used to estimate how well a model generalizes to unseen data.

Instead of relying on a single train-test split, it repeatedly rotates training and validation partitions so that every sample can be used for validation.

This provides a more reliable estimate of model performance.

---

# K-Fold Cross Validation

In K-Fold Cross Validation, the dataset is split into K folds.

For each iteration, the model is trained on K−1 folds and evaluated on the remaining fold.

The final performance is computed as the average validation score across all folds.

---

# Leave-One-Out Cross Validation

Leave-One-Out Cross Validation is a special case where each validation set contains only one sample.

It maximizes training data usage and produces a low-bias estimate, but it is computationally expensive and has high variance.

---

# Data Leakage in Cross Validation

One common mistake in cross validation is performing preprocessing before splitting the data.

For example, normalizing the entire dataset before train-validation splitting leaks information from the validation set into training.

To avoid this, preprocessing steps such as scaling should be fitted only on the training fold and then applied to the validation fold.

---

# Choosing Different Cross Validation Strategies

Different tasks require different cross validation strategies.

For classification, Stratified K-Fold is preferred because it preserves class balance.
For time series data, TimeSeriesSplit is used to preserve chronological order.
For grouped data, GroupKFold prevents samples from the same group from appearing in both training and validation.

---

# Bias-Variance Effect of K in Cross Validation

The choice of K affects the bias-variance tradeoff of cross validation.

Smaller K values produce higher bias but lower variance estimates.
Larger K values reduce bias because more data are used for training, but increase variance because the validation sets become smaller.

---

# Grid Search

Grid Search is a hyperparameter optimization method that systematically evaluates different parameter combinations.

Each configuration is typically evaluated using cross validation, and the combination with the best average validation performance is selected.

---

# Grid Search and Bias-Variance Tradeoff

Grid Search indirectly controls the bias-variance tradeoff by tuning model complexity.

For example, increasing regularization strength usually increases bias but reduces variance.

Cross validation is then used to identify the configuration that minimizes overall generalization error.

---

# Bias-Variance Tradeoff

The bias-variance tradeoff explains the relationship between model complexity and generalization performance.

Simple models usually have high bias and low variance, which leads to underfitting.
Complex models tend to have low bias and high variance, which increases the risk of overfitting.

The goal is to find a balance that minimizes total prediction error.

---

# Classification Metrics

Classification metrics are used to evaluate different aspects of model performance.

Accuracy measures overall correctness.
Precision measures how many predicted positives are correct.
Recall measures how many actual positives are successfully detected.
F1 score balances precision and recall.
AUC measures the ranking ability of the classifier across thresholds.

---

# Precision vs Recall

Precision and recall reflect different priorities.

High recall is important when missing positive cases is costly, such as fraud detection or disease diagnosis.
High precision is important when false positives are expensive, such as spam filtering or recommendation systems.

---

# Choosing Evaluation Metrics

The choice of evaluation metric depends on the business objective.

For imbalanced classification tasks, metrics such as F1 score or AUC are often more informative than accuracy.

In ranking systems and recommendation systems, ranking-oriented metrics are usually preferred.

---

# Cross Entropy Loss

Cross entropy measures the difference between the predicted probability distribution and the true label distribution.

It is widely used for classification tasks because it strongly penalizes incorrect confident predictions.

When the predicted probability for the correct class is high, the cross entropy loss becomes small.

---

# Binary Cross Entropy

Binary Cross Entropy is the standard loss function for binary classification.

It measures how well the predicted probability matches the true binary label.

The loss can be interpreted as the negative log probability assigned to the correct class.

---

# Multi-Class Cross Entropy

Multi-class cross entropy extends the same idea to classification problems with more than two classes.

The model outputs a probability distribution using softmax, and the loss penalizes low probability assigned to the correct class.

---

# Multi-Label Binary Cross Entropy

In multi-label classification, each label is treated independently.

The model predicts a separate probability for each label, and binary cross entropy is applied independently to every label.

This is different from softmax classification, where only one class can be correct.

---

# Cross Entropy in Language Models

Language model training can be viewed as a large-scale multi-class classification problem.

At each position, the model predicts a probability distribution over the entire vocabulary.

Cross entropy loss is then computed using the probability assigned to the correct next token.

---

# Geometric Median

The geometric median is the point that minimizes the sum of Euclidean distances to all points in a dataset.

Unlike the mean used in K-Means, this optimization problem does not have a closed-form solution because the distance term is nonlinear.

As a result, iterative optimization methods are required.

---

# Why K-Means Mean Update Does Not Work for Geometric Median

K-Means minimizes squared Euclidean distance, which leads to a simple closed-form mean update.

The geometric median instead minimizes raw Euclidean distance, which produces a nonlinear optimization problem where the variable appears inside the denominator.

Therefore, the standard mean update used in K-Means cannot be directly applied.

---

# Weiszfeld Algorithm

Weiszfeld’s algorithm is a standard iterative method for solving the geometric median problem.

The algorithm repeatedly computes a weighted average of all points, where nearby points receive larger weights and distant points receive smaller weights.

The iterations continue until the solution converges.

---

# 1D Minimum Manhattan Distance

In one-dimensional space, the minimum Manhattan distance for a point only depends on its neighboring points after sorting.

Therefore, the problem can be solved efficiently by sorting the array while preserving original indices and comparing each element with its adjacent neighbors.

---

# Closest Manhattan Distance in 2D

In two-dimensional space, Manhattan distance can be decomposed into four directional linear projections.

By sorting points along these transformed directions and only comparing neighboring points, we can efficiently identify candidate nearest neighbors.

This reduces the complexity from quadratic time to roughly O(N log N).

---

# Self-Attention

Self-attention allows each token in a sequence to dynamically aggregate information from other tokens based on learned relevance scores.

The model first projects the input into Query, Key, and Value representations.

Attention scores are computed using the similarity between queries and keys, normalized with softmax, and then used to compute a weighted sum of the values.

This produces contextualized representations where each token can incorporate information from the entire sequence.

---

# Single-Head Self-Attention Workflow

The self-attention process consists of several steps.

First, the input embeddings are projected into Query, Key, and Value matrices.

Second, similarity scores are computed between all token pairs.

Third, softmax converts these scores into attention weights.

Finally, the weighted combination of Value vectors produces the contextualized output representation.

---

# Attention Intuition

Attention can be interpreted as a dynamic weighted aggregation mechanism.

Instead of treating all tokens equally, the model learns which tokens are most relevant for understanding the current token.

This enables the model to capture long-range dependencies much more effectively than traditional sequential architectures.

