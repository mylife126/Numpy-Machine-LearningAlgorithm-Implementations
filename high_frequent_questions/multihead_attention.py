"""
Self-Attention and Multi-Head Attention Intuition

The core attention mechanism is based on:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

----------------------------------------------------------
Single-Head Attention
----------------------------------------------------------

Given input X of shape (m, n):
    - m: sequence length (number of tokens)
    - n: model dimension (embedding size)

We project X into three subspaces via learnable matrices:
    Q = X @ W_Q : (m, n) @ (n, d_q) -> (m, d_q)
    K = X @ W_K : (m, n) @ (n, d_k) -> (m, d_k)
    V = X @ W_V : (m, n) @ (n, d_v)

Intuitively:
    - Q ("queries"): what each token is *looking for*
    - K ("keys"): what each token *contains* (matchable info)
    - V ("values"): what information to *deliver* if matched

Usually we set d_q == d_k, since they must align in the dot product (Q @ K^T).

Example:
    Suppose "I love you" and d_model = 512
    - Each token embedding → vector of length 512 → m = 3

Then:
    A_ = Q @ K^T       # (m, d_k) @ (d_k, m) -> (m, m)
    A  = softmax(A_ / sqrt(d_k))   # Normalize attention weights
    Output = A @ V     # (m, m) @ (m, d_v) -> (m, d_v)

Result: an "attended" representation for each token (how it relates to others).


----------------------------------------------------------
Multi-Head Attention
----------------------------------------------------------

Intuition:
    Instead of using a single (Q, K, V) space,
    we create multiple independent projection spaces ("heads").
    Each head learns a different relational pattern among tokens.

For h heads:
    d_q = d_k = d_model / h
    d_v = d_model / h     # smaller subspace per head

Each head i has its own learnable parameters:
    W_Q[i], W_K[i], W_V[i]  -> all of shape (d_model, d_model / h)

Then each head computes:
    head_i = softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i   # (m, d_v)

Finally, we concatenate all heads:
    Concat(head_1, ..., head_h) -> (m, h * d_v)

Then apply an output linear projection:
    Output = Concat(...) @ W_O   # (m, h * d_v) @ (h * d_v, d_model) -> (m, d_model)

This final step fuses information from all heads
back into the original model dimension space.

Summary:
    - Multiple W matrices (W_Q, W_K, W_V per head)
    - d_q = d_k = d_model / num_heads
    - Produces multiple (m, d_v) contexts
    - Concatenate → (m, h * d_v)
    - Project back via W_O → (m, d_model)

所以对于multihead ， intuition为： 对于X 做多组线形变换 ， 也就是有多个W 权重， 多个学习参数。
那么此刻dk dq = model dim / k， k = heads
最后就会得到多个context outputs， 那么假设context的维度是 seq_length by q_v, concat 得到了 seq_length by q_v*k

一个很长的context feature， 这个时候用output linear transformation将其转换到 m by qv即可
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Each head will have its own sub-dimension
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Combined linear layers for efficiency
        # Each maps d_model -> num_heads * d_k (or d_v)
        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * self.d_v, bias=False)

        # Output linear layer to map concatenated heads back to d_model
        self.W_o = nn.Linear(num_heads * self.d_v, d_model, bias=False)

    def forward(self, X):
        """
        X: batch, seq_len, d_model
        """
        batch_size, seq_len, _ = X.size()

        # Step 1: Linear projections -> (batch, seq_len, num_heads * d_k)
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # Step 2: Reshape into multiple heads
        # -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # Step 3: Compute attention for each head
        dk = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / dk  # (batch, heads, seq_len, seq_len)
        weights = F.softmax(scores, dim=-1)

        # Step 4: Weighted sum -> (batch, heads, seq_len, d_v)
        context = torch.matmul(weights, V)

        # Step 5: Combine heads （concat）
        # -> (batch, seq_len, heads, d_v) -> (batch, seq_len, heads * d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_v)

        # Step 6: Final linear projection back to d_model
        out = self.W_o(context)  # (batch, seq_len, d_model)

        return out, weights

if __name__ == "__main__":
    torch.manual_seed(42)

    # 定义参数
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 4

    # 构造输入
    X = torch.randn(batch_size, seq_len, d_model)
    print("Input shape:", X.shape)

    # 初始化多头注意力层
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # 前向传播
    out, weights = mha(X)

    # 输出结果
    print("\n✅ Output shapes:")
    print("Output:", out.shape)  # (batch, seq_len, d_model)
    print("Attention weights:", weights.shape)  # (batch, heads, seq_len, seq_len)

    # 验证softmax正确性（每个query的注意力分布之和应≈1）
    print("\nRow sums (should be ≈1):")
    print(torch.sum(weights, dim=-1))