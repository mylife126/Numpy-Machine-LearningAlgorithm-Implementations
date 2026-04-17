import numpy as np

"""
假设：
	•	模型输出 logits（未归一化分数） z_i
	•	通过 softmax 得到概率分布：
p_i = e^z_i / sum(e^z_j) 
	•	真实标签 y 是 one-hot 向量（长度 = num_classes）

Cross Entropy Loss：
L = -sum_i(y_i * log(p_i))

如果 y 是类的 index（不是 one-hot）：
L = -log(p_y_true)

"""


def cross_entropy_loss(logits, y_true):
    """
    logits: shape (num_classes,)
    y_true: scalar, true class index
    """
    # 1️⃣ softmax
    exp_logits = np.exp(logits - np.max(logits))  # 防止溢出
    probs = exp_logits / np.sum(exp_logits)

    # 2️⃣ 取出正确类别的概率
    p_true = probs[y_true]

    # 3️⃣ cross entropy loss
    loss = -np.log(p_true + 1e-9)  # 加epsilon防止log(0)
    return loss, probs


# example
logits = np.array([1.2, 0.5, 2.1])  # 3类
y_true = 2  # 正确类别索引
loss, probs = cross_entropy_loss(logits, y_true)
print("Softmax Probabilities:", probs)
print("Loss:", loss)


def cross_entropy_batch(logits, y_true):
    """
    logits: shape (batch_size, num_classes)
    y_true: shape (batch_size,)
    """
    # softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # gather 正确类概率
    batch_indices = np.arange(len(y_true))
    p_true = probs[batch_indices, y_true]

    # 平均loss
    loss = -np.mean(np.log(p_true + 1e-9))
    return loss


##################################################

import torch
import torch.nn.functional as F


def cross_entropy_manual(logits, y_true):
    """
    logits: (batch, num_classes)
    y_true: (batch,)
    """
    # softmax
    exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True).values)
    probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

    # pick correct class prob
    batch_indices = torch.arange(logits.size(0))
    p_true = probs[batch_indices, y_true]

    # loss
    loss = -torch.mean(torch.log(p_true + 1e-9))
    return loss


# Example
logits = torch.tensor([[1.2, 0.5, 2.1], [2.0, 1.0, 0.1]])
y_true = torch.tensor([2, 0])

print(cross_entropy_manual(logits, y_true))


##################################################
def binary_cross_entropy(pred, y_true):
    """
    L = -[y\log(p) + (1-y)\log(1-p)]
    pred: sigmoid output, shape (batch,)
    y_true: (batch,)
    """
    eps = 1e-9
    return -np.mean(y_true * np.log(pred + eps) + (1 - y_true) * np.log(1 - pred + eps))

##################################################
#When input is one hot vector instead of the index#
##################################################
import numpy as np

def cross_entropy_onehot(logits, y_onehot):
    """
    logits: (num_classes,) - raw model scores
    y_onehot: (num_classes,) - one-hot vector
    """
    # 1️⃣ softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # 2️⃣ cross entropy
    loss = -np.sum(y_onehot * np.log(probs + 1e-9))
    return loss, probs

def cross_entropy_onehot_batch(logits, y_onehot):
    """
    logits: (batch, num_classes)
    y_onehot: (batch, num_classes)
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # element-wise multiply & sum over classes
    loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
    return loss

def cross_entropy_onehot_torch(logits, y_onehot):
    """
    logits: (batch, num_classes)
    y_onehot: (batch, num_classes)
    """
    # softmax
    log_probs = torch.log_softmax(logits, dim=1)  # 直接log-softmax更稳定
    loss = -(y_onehot * log_probs).sum(dim=1).mean()
    return loss

# Example
logits = torch.tensor([[1.2, 0.5, 2.1]], requires_grad=True)
y_onehot = torch.tensor([[0., 0., 1.]])
loss = cross_entropy_onehot_torch(logits, y_onehot)
print(loss)  # tensor(0.4253)