"""
attention_score = softmax(Q @ KT/ sqrt(dk))

Q <- X @ W_q : m by n @ n by d_q, to map the original high dim X into a low dim space. The "search" representations
K <- X @ W_k : m by n @ n by d_k, to map the high dim X into a low dim space.  The "matched tokens" representations
V <- X @ W_v: m by n @ n by d_v


Here d_q == d_k, because later there needs a dot product between this two quantities. And d_v is adjustable hyperparameters

NOTE: usually the d_q = d_k = n/ num_heads , where n is the token's model_dim (such as 512)
      m is the sequence length

For example I love you, suppose the model dim is 512
I -> [0.5, ......., ] length = 512
love -> [0.1, ......] length = 512
you -> [0.2, .....] length = 512 .   m=3


Now:
Q = X @ W_q , m by d_q
K = X @ W_k , m by b by d_q
V = X @ W_v , m by b by d_v


A_ = Q @ K.T = (m, d_q) @ (d_q, m) -> this gives u a head map telling you how similar or important is between a key to a value

In this case, it is a 3by3 matrix:
[[1,2,3],
[4,5,6],
[7,8,9]]
]

A = softmax(A_/sqrt(dk)), along the axis = -1 which is the feature dim

Finally the output = A @ V = (3, 3) @ (3, 2) = (3,2) This gives u the attended information
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPytorch(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model):
        super().__init__()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)

    def softmax(self, x):
        """
        nd matrix , batch, m , m
        """
        x = x - torch.max(x, dim=-1, keepdim=True).values  # make it a scalar

        e = torch.exp(x)
        sum_exp = torch.sum(e, dim=-1, keepdim=True)  # keep dim is to keep the dimension do not remove that dimension

        return e / sum_exp

    def forward(self, X):
        """
        X - batch, sequence length (m), model_dim
        """
        Q = self.W_q(X)  # (batch, sequence_length, d_k)
        K = self.W_k(X)  # (batch, sequence_length, d_k)
        V = self.W_v(X)  # (batch, sequence_length, d_v)

        dk = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))  # scalar
        A = torch.matmul(Q, K.transpose(-2, -1)) / dk  # m by m switch the last dim to the second last dim
        weights = F.softmax(A, dim=-1)
        weights_ = self.softmax(A)

        print(weights - weights_)

        context = torch.matmul(weights, V)
        return context, weights


class AttentionNumpy():
    def __init__(self, d_q, d_k, d_v, d_model):
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.W_q = np.random.randn(d_model, d_q).astype(np.float32)
        self.W_k = np.random.randn(d_model, d_k).astype(np.float32)
        self.W_v = np.random.randn(d_model, d_v).astype(np.float32)

    def softmax(self, values):
        """
        nd numpy array
        batch, m by m
        """
        x = values - np.max(values, axis=-1, keepdims=True)  # to prevent leakage
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def attend(self, X):
        """
        X -> batch, sequence length (m), model_dim
        """

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        k = np.sqrt(self.d_k)

        A_ = Q @ np.transpose(K, (0, 2, 1)) / k  # batch, sequence length, sequence length
        weights = self.softmax(A_)  # sum row wise and divided by the sum(row) , each row is an attention scores

        context = np.matmul(weights, V)
        return context, weights

if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, d_model, d_k, d_v = 1, 6, 8, 4, 4

    X = torch.randn(batch, seq_len, d_model)
    attn = AttentionPytorch(d_model=d_model, d_k=d_k, d_v=d_v, d_q=d_k)
    out, weights = attn(X)

    print("Input X:", X.shape)
    print("Output:", out.shape)
    print("Attention weights:", weights.shape)
    print("Row sums (≈1):", torch.sum(weights, dim=-1))

    X_numpy = np.random.randn(batch, seq_len, d_model)
    attn = AttentionNumpy(d_k, d_k, d_v, d_model)
    context, weights = attn.attend(X_numpy)
