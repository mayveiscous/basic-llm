# LLM

A minimal transformer-based language model implemented from scratch in Go. It includes a custom tensor library, BPE tokenizer, and full forward/backward training loop with manual gradient computation.

## Overview

This project implements a small GPT-style decoder-only transformer with:

- Byte Pair Encoding (BPE) tokenizer
- Embedding + positional encoding
- Single-block transformer (LayerNorm → Attention → FFN)
- Causal self-attention
- Manual backpropagation (no autograd)
- Cross-entropy loss with softmax
- Gradient descent optimization
- Text generation with sampling

Everything, including matrix operations and backpropagation, is implemented manually.

## Key Components

- **EmbeddingWeights**: token embeddings
- **PosWeights**: positional embeddings
- **Wq, Wk, Wv**: attention projections
- **W1, W2**: feed-forward network
- **Wlm**: output projection to vocabulary
- **Gamma/Beta**: LayerNorm parameters

---

## Tensor System

A custom tensor implementation is used instead of external ML libraries.

Supports:

- Matrix multiplication (with transpose variants)
- Layer normalization
- Softmax
- GELU activation
- Gradient buffers
- Manual memory reuse for performance

All operations are explicit and allocation-aware.

---

## Tokenizer (BPE)

A simple Byte Pair Encoding tokenizer.

### Pipeline

Text → Rune Split → Symbol Merging → Vocabulary IDs

### Behavior

- Starts at character-level tokens
- Iteratively merges most frequent adjacent pairs
- Builds vocabulary dynamically
- Encodes text into integer token IDs
- Decodes IDs back into strings

This enables subword-level representation instead of raw characters.

---

## Training

### Loss Function

- Cross entropy over vocabulary distribution
- Softmax applied per token position

### Backpropagation

Manually implemented gradients for:

- Attention
- Feed-forward network
- LayerNorm
- Embeddings
- Output projection

No autograd or ML framework is used.

### Optimization

- Gradient descent
- Learning rate decay
- Gradient clipping (L2 norm threshold)

---

## Attention Mechanism

Scaled dot-product attention:

Attention(Q, K, V) = softmax(QKᵀ / √d) V

With:

- Causal masking (prevents future token access)
- Per-sequence softmax normalization
- Batch support via manual reshaping

---

## Generation

- Sliding context window (fixed sequence length)
- Forward pass inference
- Sampling from logits:
  - Temperature = 0 → greedy
  - Temperature > 0 → stochastic sampling

---

## Limitations

- Single transformer block (not deep)
- No multi-head attention
- No KV caching for inference
- CPU **only** implementation
- BPE implementation is simplified
- Training stability depends heavily on hyperparameters

---

## Goals

- Understand transformer implementation
- Gain hands-on experience in ml infrastructure
- Strengthen linear algebra, calculus, and probability knowledge
- Build a foundation for larger, more optimized models