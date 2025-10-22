
# 🙋1 GRADIENT DESCENT TYPES
========================

There are three main variants of Gradient Descent used in optimization:

1️⃣ Batch Gradient Descent (BGD)
--------------------------------
- Uses the **entire training dataset** to compute the gradient at each step.
- Update rule: θ ← θ - α * (1/m) * Xᵀ(Xθ - y)
- Pros:
    • Produces an exact gradient (no noise)
    • Converges smoothly and stably
- Cons:
    • Very slow for large datasets (one update per epoch)
    • Requires full data to fit in memory

2️⃣ Stochastic Gradient Descent (SGD)
-------------------------------------
- Uses **a single random sample** to compute the gradient and update parameters.
- Update rule: θ ← θ - α * ∇J(xᵢ, yᵢ)
- Pros:
    • Fast updates (one per sample)
    • Can escape shallow local minima due to noise
    • Suitable for online learning
- Cons:
    • Noisy convergence (loss oscillates)
    • Sensitive to learning rate
    • May never fully converge, only fluctuate around the optimum

3️⃣ Mini-Batch Gradient Descent (MBGD)
--------------------------------------
- Uses a **small random subset of samples** (e.g., 32, 64, 128) per update.
- Update rule: θ ← θ - α * (1/batch_size) * X_batchᵀ(X_batchθ - y_batch)
- Pros:
    • Trade-off between speed and stability
    • Works efficiently on GPUs (vectorized computation)
    • Most commonly used in deep learning and large-scale ML
- Cons:
    • Requires tuning of batch size
    • Still introduces small gradient noise

🧠 Summary Table
--------------------------------------
| Method | Gradient Computed From | Noise | Convergence | Speed | Use Case |
|---------|------------------------|--------|--------------|--------|-----------|
| Batch | All samples | None | Very stable | Slow | Small datasets |
| SGD | One sample | High | Noisy | Fast per step | Online / streaming data |
| Mini-Batch | Small subset | Moderate | Smooth | Fast + stable | Deep learning standard |

⚙️ Practical Tip
--------------------------------------
In modern ML systems, "SGD" almost always means **Mini-Batch SGD** — 
each iteration uses a small random batch to approximate the full gradient.

All three follow the same core update rule:
    θ ← θ - α * ∇J(θ)
but differ in how the gradient is estimated per iteration.
"""

# 🙋 2 Principal Component Analysis (PCA) — Full Summary

## 1️⃣ Problem Definition
Given a data matrix  
$X \in \mathbb{R}^{m \times n}$
where:
- \(m\) = number of samples  
- \(n\) = number of features  

**Goal:** Find a low-dimensional representation \(Z \in \mathbb{R}^{m \times k}\) (where \(k < n\)) that preserves as much variance (information) as possible.

We seek a transformation matrix:
$
U_k \in \mathbb{R}^{n \times k}
$
such that:
$
Z = X_c U_k
$
where \(X_c\) is the **centered** data (zero-mean).

---

## 2️⃣ Center the Data
Since variance is computed around the mean, first subtract the column-wise mean:
$
X_c = X - \mathbf{1}\mu^\top,\qquad
\mu = \frac{1}{m}\sum_{i=1}^m x_i
$

---

## 3️⃣ Optimization Objective
Find a unit vector \(u\) that maximizes the **variance** of the data projected onto \(u\):
$
\max_{u}\ \mathrm{Var}(X_c u)\quad \text{s.t. } \|u\|=1
$

For each data point \(x_i\), its projection onto \(u\) is a scalar:
$
z_i = x_i^\top u
$

Since the data are centered (\(\bar z = 0\)):
$
\mathrm{Var}(z) = \frac{1}{m}\sum_{i=1}^m (x_i^\top u)^2
$

Matrix form:
$
\mathrm{Var}(z) = \frac{1}{m}u^\top X_c^\top X_c\, u \;=\; u^\top \Sigma u
$
with the covariance matrix:
$
\Sigma = \frac{1}{m}\, X_c^\top X_c
$

---

## 4️⃣ Lagrange Multiplier ⇒ Eigenvalue Problem
Maximize \(u^\top \Sigma u\) subject to \(\|u\|=1\):
$
\mathcal{L}(u,\lambda) = u^\top \Sigma u \;-\; \lambda (u^\top u - 1)
$
Set derivative to zero:
$
2\Sigma u - 2\lambda u = 0 \quad\Rightarrow\quad \Sigma u = \lambda u
$

**Therefore:**
- \(u_i\): eigenvectors of \(\Sigma\) (principal component directions)  
- \(\lambda_i\): eigenvalues = variance explained by each component  

Sort eigenvalues in descending order, take the top \(k\) eigenvectors → \(U_k\).

---

## 5️⃣ Dimensionality Reduction & Reconstruction
- **Projection:**
  $
  Z = X_c U_k
  $
- **Reconstruction:**
  $
  \hat{X} = Z U_k^\top + \mu
  $

---

## 6️⃣ Why Use SVD Instead of Eigen Decomposition
Directly computing eigenvectors of \(\Sigma = \frac{1}{m} X_c^\top X_c\) (shape \(n \times n\)) can be unstable or expensive when \(n\) is large.

Instead, perform **SVD**:
$
X_c = U\, S\, V^\top
$
Then:
$
X_c^\top X_c \;=\; V\, S^2\, V^\top
$

Hence:
- **Principal directions:** columns of \(V\)  
- **Variance per component:** \(S_i^2/(m-1)\)  
- **Explained variance ratio:**  
  $
  r_i \;=\; \frac{S_i^2}{\sum_j S_j^2}
  $

---

## 7️⃣ Selecting the Number of Components (k)
Cumulative explained variance ratio:
$
R_k \;=\; \frac{\sum_{i=1}^{k} S_i^2}{\sum_{i=1}^{r} S_i^2}
$

Choose the smallest \(k\) such that \(R_k \ge 0.95\).

```python
svals = S  # singular values from SVD
explained_var_ratio = svals**2 / np.sum(svals**2)
cum_ratio = np.cumsum(explained_var_ratio)
k = np.searchsorted(cum_ratio, 0.95) + 1
```

---
## 8️⃣ Minimal PCA (SVD) Implementation
```
python
import numpy as np

# 1) Center
Xc = X - X.mean(axis=0)

# 2) SVD
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

# 3) Explained variance ratio
explained_var_ratio = (S**2) / np.sum(S**2)

# 4) Choose k (e.g., 95%)
cum_ratio = np.cumsum(explained_var_ratio)
k = np.searchsorted(cum_ratio, 0.95) + 1

# 5) Project
V_k = Vt[:k, :].T       # (n, k)
Z = Xc @ V_k            # (m, k)

# (Optional) Reconstruct
X_rec = Z @ V_k.T + X.mean(axis=0)
```

## 9️⃣ Geometric Intuition
	•	Principal components are orthogonal axes pointing along directions of maximum variance.
	•	Components are uncorrelated.
	•	Early components capture most information; later ones are often noise.
---

## 🔟 Key Takeaways (Cheat Sheet)
| Concept | Expression | Meaning |
|----------|-------------|----------|
| Projection of a point | $z_i = x_i^\top u$ | Coordinate of \(x_i\) along direction \(u\) |
| Variance along \(u\) | $\mathrm{Var}(z) = u^\top \Sigma u$ | How widely data spread along \(u\) |
| Optimization | $\max_u u^\top \Sigma u \ \text{s.t.}\ \|u\|=1$ | Find the most informative direction |
| Eigen equation | $\Sigma u = \lambda u$ | \(u\): direction; \(\lambda\): variance |
| SVD link | $X_c = U S V^\top$ | \(V\) gives principal directions |
| Variance from SVD | $S_i^2/(m-1)$ | Variance explained by component \(i\) |
| Explained var. ratio | $S_i^2 \Big/ \sum_j S_j^2$ | Fraction of total variance |
| Cumulative ratio | $\sum_{i\le k} S_i^2 \Big/ \sum_j S_j^2$ | Info kept by first \(k\) components |
| Projection matrix | $U_k \in \mathbb{R}^{n\times k}$ | New basis (orthonormal) |
| Low-dim data | $Z = X_c U_k$ | Shape \((m, k)\) |

