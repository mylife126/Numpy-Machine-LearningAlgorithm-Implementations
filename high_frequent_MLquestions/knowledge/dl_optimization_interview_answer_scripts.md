# DL Optimization Interview Answer Scripts

Source file: fileciteturn0file0L1-L1

---

# Learning Rate

The learning rate controls the step size of parameter updates during optimization.

- If the learning rate is too large, the model may overshoot the optimal point, causing the loss to oscillate or fluctuate instead of converging.
- If it is too small, the updates become very slow and the model may take a long time to converge or get stuck in flat regions.

So in practice, we want a balance between fast convergence and stable optimization, and we often use techniques like learning rate scheduling or adaptive optimizers such as Adam.

---

# Gradient Vanishing

Gradient vanishing is more severe in layers closer to the input layer.

- During backpropagation, gradients are computed via the chain rule, which involves multiplying gradients across layers.
- If each layer contributes a factor smaller than 1, the gradient shrinks exponentially as it propagates backward.

As a result, early layers receive very small gradients and learn very slowly.
This is why modern deep networks use ReLU activations and residual connections to help gradients flow more effectively.

---

# Gradient Explosion

Gradient explosion happens when gradients grow exponentially during backpropagation.
This is usually due to repeated multiplication of large weights, especially in deep networks or RNNs.
It’s not mainly caused by activation functions, but by weight matrices with large norms.

---

# Adam Optimizer

Adam is more stable because it combines both momentum and adaptive learning rate.

- The first moment estimate smooths the gradient direction, reducing oscillations.
- The second moment estimate scales the update based on gradient magnitude, so each parameter has its own adaptive learning rate.
- Bias correction ensures accurate estimates in early training.

Compared to SGD, Adam converges faster and is more robust to noisy gradients.

---

# Residual Connections (ResNet)

Residual connections help by introducing a shortcut path for gradients.
Instead of learning a direct mapping, the network learns a residual function.
During backpropagation, gradients can flow directly through the identity path, which mitigates vanishing gradients.

---

# Sigmoid Saturation

Sigmoid saturation happens when inputs are very large or very small.
In those regions, the derivative becomes close to zero, which leads to vanishing gradients.

---

# Learning Rate Fluctuation vs SGD Noise

LR fluctuation and SGD noise are different. Large learning rates cause deterministic oscillations due to overshooting.
SGD noise comes from stochastic sampling and introduces randomness, which can actually help exploration.

---

# SGD / Momentum / RMSProp / Adam Overview

I usually explain these optimizers as an evolution from SGD.

Plain SGD updates parameters directly using the current mini-batch gradient. The problem is that the gradient can be noisy, especially in mini-batch training, so the optimization path may zig-zag and become unstable.

Momentum was introduced to address that issue. It keeps an exponential moving average of past gradients, so instead of reacting only to the current gradient, it accumulates a notion of velocity. This helps smooth the update direction, reduce oscillation, and accelerate training along consistent directions.

RMSProp addresses a different problem. In practice, different parameters can have very different gradient magnitudes. If we use the same learning rate for all parameters, some dimensions may update too aggressively while others move too slowly. RMSProp keeps an exponential moving average of squared gradients and uses it to normalize the update, so each parameter gets its own adaptive effective learning rate.

Adam combines both ideas. It tracks the first moment, which is the momentum-like moving average of gradients, and the second moment, which is the moving average of squared gradients. So it both smooths the direction and rescales the step size. That is why Adam is often more stable and easier to tune in practice.

One more detail is bias correction. Since both moving averages start from zero, they are biased toward zero in the early steps. Adam corrects for that, which makes the estimates more accurate at the beginning of training.

---

# SGD

Plain SGD is simple, but it has two major issues in practice. First, the mini-batch gradient is noisy, so updates can oscillate a lot. Second, all parameters share the same learning rate, even though different dimensions may have very different gradient scales. These issues motivate Momentum and adaptive optimizers like RMSProp and Adam.

---

# Momentum

Momentum is motivated by the instability of plain SGD. In mini-batch training, the gradient can be noisy, so the update direction may oscillate a lot. Momentum fixes this by keeping an exponential moving average of past gradients. The momentum term acts like a velocity, so the optimizer does not react only to the current gradient, but also to the recent history. This reduces zig-zag behavior and accelerates learning along consistent directions.

---

# RMSProp

RMSProp addresses a different issue from Momentum. The main idea is that different parameters can have very different gradient magnitudes, so using one shared learning rate is often suboptimal. RMSProp keeps an exponential moving average of squared gradients. If a parameter has consistently large gradients, its denominator becomes larger, so the update is reduced. If its gradients are small, the update becomes relatively larger. So RMSProp gives each parameter an adaptive effective learning rate.

---

# Adam Detailed Explanation

Adam combines Momentum and RMSProp. The first moment term is an exponential moving average of gradients, which smooths the direction like Momentum. The second moment term is an exponential moving average of squared gradients, which rescales the update like RMSProp. So Adam both stabilizes the direction and adapts the step size for each parameter. In addition, it uses bias correction because these moving averages are initialized at zero and would otherwise be biased toward zero in the early steps. That is why Adam is usually more stable and easier to tune in practice.

---

# Momentum vs RMSProp vs Adam

If I compare them directly, Momentum mainly improves direction stability, RMSProp mainly improves per-parameter step-size adaptation, and Adam combines both. So when people say Adam is more stable, they usually mean it is stable both in direction and in scale.

---

# Overfitting vs Underfitting

Underfitting happens when the model is too simple and cannot capture the underlying pattern, so both training and test errors are high.
Overfitting happens when the model is too complex and starts fitting noise, so training error is low but test error is high.
This can be explained by the bias–variance tradeoff, where simpler models have high bias and complex models have high variance.

---

# Regularization

Regularization adds a penalty term to the loss function to control model complexity. By discouraging large weights, it prevents the model from fitting noise and improves generalization.

---

# L2 Regularization

L2 regularization penalizes the squared magnitude of weights, which encourages smaller but non-zero parameters. This leads to smoother models and better stability, especially when features are correlated.

---

# L1 Regularization

L1 regularization penalizes the absolute value of weights. It tends to push some weights exactly to zero, which makes the model sparse and performs implicit feature selection.

---

# L1 vs L2

L1 encourages sparsity and is useful for feature selection, while L2 provides more stable solutions by shrinking weights smoothly. In practice, L2 is more commonly used, especially in large-scale systems.

---

# Dropout

Dropout randomly deactivates neurons during training, which prevents neurons from co-adapting too much. It can be seen as training an ensemble of subnetworks, which improves generalization and reduces overfitting.

---

# Preventing Overfitting

To reduce overfitting, we can control model complexity using regularization or simpler models, increase data or use data augmentation, and apply training techniques like early stopping or cross-validation.

---

# Complete Overfitting / Underfitting Answer

Overfitting and underfitting are both related to model complexity.

Underfitting happens when the model is too simple and cannot capture the underlying pattern, leading to high training and test error. Overfitting happens when the model is too complex and starts fitting noise, resulting in low training error but high test error.

This is explained by the bias–variance tradeoff. Simpler models have high bias, while complex models have high variance.

To address overfitting, we use regularization, which adds a penalty term to the loss function. L2 regularization shrinks weights smoothly and improves stability, while L1 regularization promotes sparsity and performs feature selection.

In deep learning, dropout is also widely used. It randomly drops neurons during training, which prevents co-adaptation and behaves like an ensemble of subnetworks.

In practice, we often combine multiple techniques, such as regularization, more data, and early stopping, to achieve better generalization.

---

# Activation Functions

Activation functions introduce non-linearity into neural networks. Without them, stacking multiple layers would still result in a linear model. They allow the network to learn complex patterns and increase model expressiveness.

---

# Loss Functions

The loss function measures how far the model’s predictions are from the ground truth. It defines the optimization objective and provides the gradient signal used in backpropagation.

---

# Activation vs Loss Function

Activation functions and loss functions play different roles in neural networks.

Activation functions are used inside the network to introduce non-linearity and improve model expressiveness, while the loss function is used at the output to measure prediction error and define the optimization objective.

In short, activation functions determine what the model can represent, and the loss function determines what the model tries to optimize.

---

# Common Activation + Loss Combinations

In classification tasks, we usually use ReLU in hidden layers, sigmoid or softmax in the output layer, and cross-entropy loss. For regression, we typically use a linear output and MSE loss.

---

# ROC-AUC

ROC-AUC measures the probability that a randomly chosen positive sample is ranked higher than a negative one. It evaluates the ranking ability of a classifier across all thresholds, but it can be overly optimistic in highly imbalanced datasets.

---

# PR-AUC

PR-AUC focuses on precision and recall, which makes it more suitable for highly imbalanced datasets. In recommendation systems, where positives are rare, PR-AUC better reflects the quality of the predicted positives.

---

# nDCG

nDCG evaluates ranking quality by considering both relevance and position. It assigns higher importance to items ranked at the top, and normalizes the score by the ideal ranking. This makes it very suitable for ranking tasks where position matters.

---

# Hit@K

Hit@K measures whether at least one relevant item appears in the top K results. It is a simple metric that focuses on recall in the top positions but does not consider ranking order.

---

# Classification Metrics vs Ranking Metrics

ROC-AUC and PR-AUC are classification metrics, while nDCG and Hit@K are ranking metrics.

In practice, for recommendation systems like Pinterest, we care more about ranking quality, so nDCG is commonly used because it accounts for position. Hit@K is useful when we only care whether relevant items appear in the top results.

For click prediction, which is usually highly imbalanced, PR-AUC is more informative than ROC-AUC.

---

# nDCG vs Hit@K Example

For example, if a relevant item appears at position 2 instead of position 1, Hit@K would still consider it a success, but nDCG would penalize it due to the lower rank. This makes nDCG more sensitive to ranking quality.

---

# Final Evaluation Metrics Summary

ROC-AUC and PR-AUC measure classification quality, while nDCG and Hit@K measure ranking quality. In recommendation systems, nDCG is often preferred because it accounts for both relevance and position.

---

# Attention Mechanism

In attention, we first project the input features into three representations: Query, Key, and Value using linear transformations.

We then compute the similarity between tokens by taking the dot product between queries and keys, followed by a softmax. This gives us attention weights that represent how much each token attends to others.

These weights are then used to compute a weighted sum of the values, which produces the final output representation for each token. This is called self-attention because each token attends to all other tokens in the same sequence.

---

# Multi-Head Attention

Multi-head attention extends this idea by applying multiple attention operations in parallel, each with its own projections. This allows the model to learn different types of relationships in different representation subspaces, instead of forcing everything into a single attention pattern.

---

# Positional Encoding

One limitation of attention is that it is permutation-invariant, meaning it does not encode positional information. To address this, we add positional encodings to the input embeddings before computing attention.

These encodings are typically sinusoidal functions with different frequencies, which allow the model to capture both absolute and relative positions. After adding them to the embeddings, positional information is naturally incorporated into the attention computation.

---

# Positional Encoding + Attention Integration

Positional encoding is a vector added to each token embedding to inject position information. For each position in the sequence, we generate a vector using sinusoidal functions with different frequencies. This vector has the same dimension as the embedding, so we can directly add them together.

After that, the combined representation is used to compute Q, K, and V. So positional information is implicitly included in the attention computation.

---

# Transformer

Transformer is a neural architecture built entirely on attention mechanisms. It replaces recurrence with self-attention, allowing each token to directly attend to all other tokens. This enables better modeling of long-range dependencies and allows for highly parallel computation.

Compared to RNNs and CNNs, Transformer replaces recurrence and convolution with self-attention. Instead of processing tokens sequentially or locally, each token can directly attend to all other tokens, which makes it more effective at modeling long-range dependencies.

---

# Transformer Encoder vs Decoder

The Transformer encoder and decoder have similar overall structures, but they serve different purposes.

The encoder is mainly responsible for understanding the input sequence and producing contextualized representations. It uses bidirectional self-attention, meaning each token can attend to all other tokens in the input.

The decoder is responsible for generating the output sequence autoregressively. It contains two attention modules. The first is masked self-attention, which prevents the model from seeing future tokens during generation. The second is cross-attention, where the decoder attends to the encoder outputs to retrieve relevant input information.

So intuitively, the encoder focuses on representation learning, while the decoder focuses on conditional sequence generation.

---

# Encoder-Decoder Attention

Cross-attention connects the decoder with the encoder outputs.

In this module, the Query comes from the decoder hidden states, while the Key and Value come from the encoder outputs. This allows the decoder to dynamically retrieve the most relevant information from the input sequence while generating each output token.

For example, in machine translation, when generating a target word, the decoder can attend to the most relevant source words.

---

# Masked Self-Attention

Masked self-attention is used in the decoder to prevent information leakage during training.

When predicting the next token, the model should only attend to previous tokens and not future ones. To achieve this, we apply a causal mask that blocks attention to future positions.

This ensures that training matches autoregressive inference behavior.

---

# Transformer Encoder-only vs Decoder-only vs Encoder-Decoder Models

Encoder-only models such as BERT are mainly used for understanding tasks like classification and retrieval because they learn strong bidirectional representations.

Decoder-only models such as GPT are mainly used for generative tasks because they generate tokens autoregressively using causal attention.

Encoder-decoder models such as T5 are designed for sequence-to-sequence tasks like translation and summarization, where the encoder processes the input and the decoder generates the output.

---

# Mixed Precision Training

Mixed precision training uses both lower-precision and higher-precision numerical formats during training.

Typically, most tensor operations are performed in FP16 or BF16 for efficiency, while certain critical operations such as gradient accumulation or optimizer states are maintained in FP32 for numerical stability.

The main motivation is to reduce memory usage and accelerate computation, especially on modern GPUs with Tensor Cores.

---

# Why Mixed Precision Training Is Faster

Mixed precision training is faster mainly because lower-precision formats require less memory bandwidth and allow GPUs to process more operations in parallel.

FP16 and BF16 tensors are smaller than FP32 tensors, so data transfer becomes more efficient and larger batch sizes can fit into GPU memory.

Modern GPUs also have specialized hardware such as Tensor Cores that are optimized for low-precision matrix multiplication.

---

# FP16 vs BF16

FP16 and BF16 are both low-precision formats used in mixed precision training, but they differ in numerical range and precision.

FP16 has fewer exponent bits, so it has a smaller dynamic range and is more prone to numerical underflow or overflow.

BF16 keeps the same exponent range as FP32, which makes it more numerically stable while still reducing memory and computation cost.

In practice, BF16 is often preferred on newer hardware because it provides better stability with minimal performance tradeoff.

---

# Loss Scaling in Mixed Precision Training

One challenge in FP16 training is numerical underflow, where very small gradients become zero.

To address this, we use loss scaling. The idea is to multiply the loss by a large constant before backpropagation so that gradients become numerically representable in FP16.

After gradients are computed, we divide them back by the same scaling factor before the optimizer update.

This preserves gradient information while maintaining the efficiency benefits of low-precision computation.

---

# Mixed Precision Training Stability

Mixed precision training improves efficiency, but some operations are still kept in FP32 for stability.

For example, optimizer states, gradient accumulation, and normalization statistics are often maintained in higher precision to avoid numerical instability.

This combination achieves a balance between training speed and stable optimization.

