---
title: "LLM Fundamentals"
layout: default
parent: "LLM & AI Fundamentals"
nav_order: 1
---

# LLM Fundamentals Interview Guide for AI Engineers (2025-2026)
# Comprehensive Deep-Dive Reference

---

# TABLE OF CONTENTS

1. [Core Concepts: Transformer Architecture, Self-Attention, Positional Encoding](#section-1)
2. [Tokenization, Pre-training vs Fine-tuning](#section-2)
3. [RLHF, DPO, Constitutional AI, Alignment](#section-3)
4. [Decoding Strategies, Sampling, Evaluation Metrics](#section-4)
5. [Hallucination, Scaling Laws, Emergent Abilities, CoT, ICL](#section-5)
6. [Common Interview Questions with Detailed Answers](#section-6)
7. [Latest Trends 2025-2026](#section-7)
8. [Comparison of Major LLM Providers](#section-8)
9. [AI Agent Frameworks Landscape 2025-2026](#section-9)
10. [Model Context Protocol (MCP) by Anthropic](#section-10)

---

<a id="section-1"></a>
# SECTION 1: CORE CONCEPTS - TRANSFORMER ARCHITECTURE, SELF-ATTENTION, POSITIONAL ENCODING

## 1.1 The Transformer Architecture

### Q: Explain the Transformer architecture from the ground up.

**Answer:**

The Transformer was introduced in "Attention Is All You Need" (Vaswani et al., 2017). It replaced recurrent architectures (RNN/LSTM) with a purely attention-based mechanism that enables massive parallelization.

**High-Level Architecture:**

```
Input Tokens
    |
    v
[Input Embedding + Positional Encoding]
    |
    v
+---------------------------+
| ENCODER (N layers)        |
|  - Multi-Head Self-Attn   |
|  - Add & Norm             |
|  - Feed-Forward Network   |
|  - Add & Norm             |
+---------------------------+
    |
    v
+---------------------------+
| DECODER (N layers)        |
|  - Masked Multi-Head      |
|    Self-Attention          |
|  - Add & Norm             |
|  - Multi-Head Cross-Attn  |
|    (attends to encoder)   |
|  - Add & Norm             |
|  - Feed-Forward Network   |
|  - Add & Norm             |
+---------------------------+
    |
    v
[Linear + Softmax]
    |
    v
Output Probabilities
```

**Key Components:**

1. **Embedding Layer**: Converts token IDs into dense vectors of dimension d_model (typically 512, 768, 1024, or larger).

2. **Positional Encoding**: Since transformers have no inherent notion of sequence order, positional information is injected. The original paper used sinusoidal functions:
   - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

3. **Multi-Head Self-Attention**: The core innovation. Allows each token to attend to every other token.

4. **Feed-Forward Network (FFN)**: Two linear layers with a nonlinearity (ReLU or GELU):
   - FFN(x) = max(0, xW1 + b1)W2 + b2
   - In modern LLMs, this is often replaced with SwiGLU: SwiGLU(x) = (xW1 * sigmoid(xV)) W2

5. **Layer Normalization**: Stabilizes training. Two variants:
   - Post-LayerNorm (original): LN applied after residual connection
   - Pre-LayerNorm (GPT-2+): LN applied before attention/FFN blocks (more stable training)

6. **Residual Connections**: Skip connections around each sub-layer: output = LayerNorm(x + SubLayer(x))

**Three Architectural Variants Used in Modern LLMs:**

| Variant | Architecture | Examples | Use Case |
|---------|-------------|----------|----------|
| Encoder-Only | Only encoder stack | BERT, RoBERTa, DeBERTa | Classification, NER, embeddings |
| Decoder-Only | Only decoder stack (causal) | GPT-4, Claude, Llama, Mistral | Text generation, chat |
| Encoder-Decoder | Both stacks | T5, BART, Flan-T5 | Translation, summarization |

**Why Decoder-Only Dominates for LLMs:**
- Simpler architecture with fewer components
- Naturally suited for autoregressive generation
- Easier to scale (one stack, not two)
- Unified pre-training objective (next-token prediction)
- GPT-3 demonstrated that decoder-only + scale = strong few-shot performance

---

## 1.2 Self-Attention Mechanism

### Q: Explain the self-attention mechanism mathematically and intuitively.

**Answer:**

**Intuition:** Self-attention allows each token in a sequence to "look at" every other token and decide how much to "pay attention to" each one. For example, in "The cat sat on the mat because it was tired," the word "it" should attend strongly to "cat."

**Mathematical Formulation:**

Given an input sequence X of shape (seq_len, d_model):

**Step 1: Create Q, K, V projections**
```
Q = X * W_Q    (Query: "What am I looking for?")
K = X * W_K    (Key: "What do I contain?")
V = X * W_V    (Value: "What information do I provide?")
```
Where W_Q, W_K, W_V are learned weight matrices of shape (d_model, d_k).

**Step 2: Compute attention scores**
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

**Step-by-step breakdown:**
1. `Q * K^T` produces a (seq_len x seq_len) matrix of raw attention scores
2. `/ sqrt(d_k)` scaling prevents dot products from growing too large (which would push softmax into regions with vanishing gradients)
3. `softmax(...)` normalizes scores to probabilities (each row sums to 1)
4. `* V` produces a weighted combination of value vectors

**Python implementation:**
```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_len, d_k)
    K: (batch, heads, seq_len, d_k)
    V: (batch, heads, seq_len, d_v)
    mask: optional (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # Step 1: Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, heads, seq_len, seq_len)

    # Step 2: Apply mask (for causal/padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax normalization
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### Q: What is Multi-Head Attention and why is it important?

**Answer:**

Multi-Head Attention runs multiple attention operations in parallel, each with different learned projections, allowing the model to attend to information from different representation subspaces.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
```

**Why Multiple Heads?**
- **Different attention patterns**: One head might capture syntactic relationships, another semantic relationships, another coreference
- **Richer representations**: Single attention can only produce one weighted average; multiple heads provide multiple perspectives
- **Computational efficiency**: If d_model = 768 and h = 12, each head works with d_k = 64. The total computation is similar to single-head with d_model, but with more expressiveness

**Implementation:**
```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = torch.nn.Linear(d_model, d_model)
        self.W_K = torch.nn.Linear(d_model, d_model)
        self.W_V = torch.nn.Linear(d_model, d_model)
        self.W_O = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Project and reshape to (batch, heads, seq_len, d_k)
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concat heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_O(attn_output)

        return output
```

### Q: What is causal (masked) self-attention and why do decoder-only models use it?

**Answer:**

Causal self-attention restricts each position to only attend to previous positions (and itself). This is essential for autoregressive generation because during training, the model should not "see the future."

**Mask structure (for sequence length 5):**
```
Token:    t1  t2  t3  t4  t5
t1:      [ 1   0   0   0   0 ]   <- t1 only sees itself
t2:      [ 1   1   0   0   0 ]   <- t2 sees t1, t2
t3:      [ 1   1   1   0   0 ]   <- t3 sees t1, t2, t3
t4:      [ 1   1   1   1   0 ]   <- t4 sees t1-t4
t5:      [ 1   1   1   1   1 ]   <- t5 sees all
```

Positions with 0 are set to -infinity before softmax, making their attention weights 0.

```python
def create_causal_mask(seq_len):
    """Creates a lower-triangular causal mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # (seq_len, seq_len)
```

### Q: Explain Grouped Query Attention (GQA) and Multi-Query Attention (MQA).

**Answer:**

These are efficiency optimizations for the KV cache during inference:

**Standard Multi-Head Attention (MHA):**
- Each head has its own Q, K, V projections
- KV cache size: 2 * n_layers * n_heads * d_k * seq_len
- Used in: Original Transformer, GPT-2/3

**Multi-Query Attention (MQA):**
- Each head has its own Q, but ALL heads share ONE K and ONE V projection
- KV cache size reduced by factor of n_heads
- Used in: PaLM, Falcon
- Tradeoff: Slightly lower quality, much faster inference

**Grouped Query Attention (GQA):**
- Compromise between MHA and MQA
- Heads are divided into groups; each group shares K, V
- If n_heads=32 and n_kv_heads=8, then groups of 4 query heads share KV
- Used in: Llama 2 (70B), Llama 3, Mistral, Claude
- Best balance of quality and efficiency

```
MHA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8   (8 heads, 8 KV pairs)
      K1 K2 K3 K4 K5 K6 K7 K8
      V1 V2 V3 V4 V5 V6 V7 V8

GQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8   (8 heads, 2 KV groups)
      K1 K1 K1 K1 K2 K2 K2 K2
      V1 V1 V1 V1 V2 V2 V2 V2

MQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8   (8 heads, 1 KV pair)
      K1 K1 K1 K1 K1 K1 K1 K1
      V1 V1 V1 V1 V1 V1 V1 V1
```

---

## 1.3 Positional Encoding

### Q: Why do Transformers need positional encoding? Compare different approaches.

**Answer:**

**Why needed:** Self-attention is permutation-equivariant -- it treats the input as a SET, not a SEQUENCE. Without positional information, "The cat chased the dog" and "The dog chased the cat" would produce identical representations.

**Approach 1: Sinusoidal (Absolute) Positional Encoding (Original Transformer)**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Fixed, not learned
- Can theoretically generalize to longer sequences than seen during training
- Each dimension has a different frequency, creating a unique "fingerprint" for each position
- Limitation: Struggles to extrapolate beyond training length in practice

**Approach 2: Learned Positional Embeddings (GPT-2, BERT)**
```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)
# Positions 0, 1, 2, ... max_seq_len-1 each get a learned vector
```
- Simple and effective
- Limitation: Cannot generalize beyond max_seq_len

**Approach 3: Rotary Position Embedding (RoPE) -- DOMINANT IN MODERN LLMs**
- Used in: Llama, Llama 2, Llama 3, Mistral, Qwen, DeepSeek, and most modern models
- Instead of adding positional info to embeddings, RoPE encodes position by ROTATING the query and key vectors
- Key insight: The dot product of rotated Q and K depends only on their RELATIVE position

```python
def apply_rotary_pos_emb(x, cos, sin):
    """
    x: (batch, heads, seq_len, d_k)
    cos, sin: (seq_len, d_k)
    """
    # Split into pairs and rotate
    x1, x2 = x[..., ::2], x[..., 1::2]  # even and odd dimensions

    # Apply rotation
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)

    return rotated
```

**Why RoPE is preferred:**
- Encodes RELATIVE position (more linguistically meaningful)
- Decays attention with distance (nearby tokens attend more strongly by default)
- Can be extended to longer contexts via interpolation techniques (NTK-aware scaling, YaRN)

**Approach 4: ALiBi (Attention with Linear Biases) -- Used in BLOOM, MPT**
- No positional embedding at all
- Instead, adds a linear bias to attention scores based on distance:
  `score(i,j) = q_i * k_j - m * |i - j|`
- Where m is a head-specific slope
- Very simple, strong length generalization
- Limitation: Can underperform RoPE on some tasks

**Comparison Table:**

| Method | Type | Length Generalization | Used In | Key Advantage |
|--------|------|---------------------|---------|---------------|
| Sinusoidal | Absolute, Fixed | Limited | Original Transformer | No learned params |
| Learned | Absolute, Learned | None (hard limit) | GPT-2, BERT | Simple |
| RoPE | Relative, Fixed formula | Good (with scaling) | Llama, Mistral, Qwen | Relative position, efficient |
| ALiBi | Relative, Linear bias | Excellent | BLOOM, MPT | Simplest, best extrapolation |

---

## 1.4 Context Windows

### Q: What is a context window? What are the challenges with long context windows?

**Answer:**

The **context window** (or context length) is the maximum number of tokens a model can process in a single forward pass. It includes both the input prompt and the generated output.

**Context window sizes across models (as of mid-2025):**
| Model | Context Window |
|-------|---------------|
| GPT-3 | 2,048 tokens |
| GPT-3.5 Turbo | 4,096 / 16,384 |
| GPT-4 | 8,192 / 32,768 / 128,000 |
| GPT-4o | 128,000 |
| GPT-4.5 | 128,000 |
| o1 / o3 | 200,000 |
| o4-mini | 200,000 |
| Claude 3.5 Sonnet | 200,000 |
| Claude 3.7 Sonnet | 200,000 |
| Claude Opus 4 / Sonnet 4 | 200,000 |
| Gemini 2.0 Flash | 1,000,000 |
| Gemini 2.5 Pro | 1,000,000 |
| Gemini 1.5 Pro | 2,000,000 |
| Llama 3.1 | 128,000 |
| Llama 4 Scout | 10,000,000 (10M!) |
| Llama 4 Maverick | 1,000,000 |
| Mistral Large | 128,000 |
| DeepSeek V3 | 128,000 |
| DeepSeek R1 | 128,000 |

**Challenges with Long Contexts:**

1. **Quadratic Complexity**: Standard self-attention is O(n^2) in both time and memory. Doubling context length quadruples compute.

2. **KV Cache Memory**: During inference, the KV cache grows linearly with sequence length. For a 70B model with 128K context, KV cache alone can require ~40GB.

3. **Lost in the Middle Problem** (Liu et al., 2023): Models tend to recall information at the beginning and end of the context well, but struggle with information in the middle. This has implications for RAG systems.

4. **Position Extrapolation**: Models trained on shorter sequences may not generalize to longer ones without specific techniques.

**Solutions for Long Context:**

- **Sliding Window Attention**: Each token attends to only a fixed-size local window (used in Mistral 7B with window=4096)
- **Ring Attention**: Distributes long sequences across multiple devices
- **Flash Attention**: IO-aware exact attention that reduces memory from O(n^2) to O(n) by tiling
- **RoPE Scaling (NTK, YaRN)**: Extends RoPE to support longer contexts than training
- **Sparse Attention**: Only attend to a subset of positions (Longformer, BigBird patterns)

---

---

<a id="section-2"></a>
# SECTION 2: TOKENIZATION, PRE-TRAINING VS FINE-TUNING

## 2.1 Tokenization

### Q: What is tokenization and why is it critical for LLMs?

**Answer:**

Tokenization is the process of converting raw text into a sequence of integer token IDs that the model can process. It is the first and last step in any LLM pipeline.

**Why it matters:**
- Determines the vocabulary size (affects embedding table size and output softmax)
- Affects how efficiently information is encoded (tokens per word ratio)
- Impacts multilingual performance (some tokenizers undertokenize non-English text)
- Influences context window utilization (more tokens = less text fits in context)
- Directly affects cost (API pricing is per token)

**Key tokenization approaches:**

### Byte Pair Encoding (BPE)

**Used in:** GPT-2, GPT-3, GPT-4, Llama, Mistral

**Algorithm:**
1. Start with a base vocabulary of individual characters (or bytes)
2. Count all adjacent pairs of tokens in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size is reached

**Example walkthrough:**
```
Corpus: "low low low low low lowest lowest newer newer newer"

Step 0 (character level):
  l o w _ l o w _ l o w _ l o w _ l o w e s t _ ...

Step 1: Most frequent pair = (l, o) -> merge to "lo"
  lo w _ lo w _ lo w _ lo w _ lo w e s t _ ...

Step 2: Most frequent pair = (lo, w) -> merge to "low"
  low _ low _ low _ low _ low e s t _ ...

Step 3: Most frequent pair = (low, _) -> merge to "low_"
  ... and so on
```

**Modern BPE (Byte-level BPE, used in GPT-2+):**
- Uses raw bytes (256 base tokens) instead of Unicode characters
- Can represent ANY text without unknown tokens
- Vocabulary typically 50,000-100,000+ tokens

```python
# Using tiktoken (OpenAI's tokenizer)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Hello, how are you?")
print(tokens)        # [9906, 11, 1268, 527, 499, 30]
print(len(tokens))   # 6
print([enc.decode([t]) for t in tokens])
# ['Hello', ',', ' how', ' are', ' you', '?']
```

### WordPiece

**Used in:** BERT, DistilBERT, Electra

**Key differences from BPE:**
- Merges are chosen based on likelihood (not raw frequency)
- Selects the pair that maximizes the likelihood of the training data
- Uses "##" prefix for sub-word tokens that are continuations

```
"unbelievable" -> ["un", "##believ", "##able"]
"playing"      -> ["play", "##ing"]
```

**Merge criterion:**
```
score(pair) = freq(pair) / (freq(first) * freq(second))
```
This favors merging pairs where both parts rarely occur independently.

### SentencePiece

**Used in:** T5, Llama, Llama 2, Llama 3, Mistral, ALBERT, XLNet

**Key features:**
- Language-agnostic: treats text as a raw byte stream (no pre-tokenization needed)
- Implements both BPE and Unigram algorithms
- Handles whitespace explicitly (uses Unicode character U+2581 "lower one eighth block" to represent spaces)
- Works directly on raw text without language-specific preprocessing

```python
import sentencepiece as spm

# Training
spm.SentencePieceTrainer.train(
    input='training_data.txt',
    model_prefix='my_model',
    vocab_size=32000,
    model_type='bpe'  # or 'unigram'
)

# Usage
sp = spm.SentencePieceProcessor()
sp.load('my_model.model')

tokens = sp.encode("Hello world", out_type=str)
# ['_Hello', '_world']

ids = sp.encode("Hello world", out_type=int)
# [8774, 296]
```

### Unigram Language Model Tokenization

**Used by:** SentencePiece (as an option), XLNet, ALBERT

**Algorithm (opposite of BPE):**
1. Start with a LARGE vocabulary (all substrings up to some length)
2. Compute the loss (negative log-likelihood) on training data
3. Remove the token whose removal LEAST increases the loss
4. Repeat until desired vocabulary size is reached

**Advantage:** Produces a probabilistic model; can output multiple tokenizations with probabilities (useful for data augmentation).

### Comparison Table

| Feature | BPE | WordPiece | SentencePiece (BPE) | Unigram |
|---------|-----|-----------|-------------------|---------|
| Build direction | Bottom-up merging | Bottom-up merging | Bottom-up merging | Top-down pruning |
| Merge criterion | Frequency | Likelihood | Frequency | Loss-based removal |
| Pre-tokenization | Required | Required | Not required | Not required |
| Unknown tokens | Possible (char BPE: no) | Possible | No (byte fallback) | No |
| Used in | GPT-2/3/4 | BERT | Llama, T5 | XLNet, ALBERT |

### Q: What are special tokens and why are they important?

**Answer:**

Special tokens are reserved tokens with specific roles in the model's processing:

| Token | Purpose | Example |
|-------|---------|---------|
| [BOS] / <s> | Beginning of sequence | Marks start of text |
| [EOS] / </s> | End of sequence | Signals generation should stop |
| [PAD] | Padding | Fills shorter sequences in a batch |
| [UNK] | Unknown token | Replaces out-of-vocabulary words |
| [SEP] | Separator | Separates segments (BERT) |
| [CLS] | Classification | Used for classification tasks (BERT) |
| [MASK] | Masking | MLM pre-training (BERT) |
| <|im_start|> | Chat role start | ChatML format |
| <|im_end|> | Chat role end | ChatML format |

**Chat templates (critical for instruction-tuned models):**
```
# Llama 3 format
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is machine learning?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

# ChatML format (OpenAI)
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is machine learning?<|im_end|>
<|im_start|>assistant
```

---

## 2.2 Pre-training vs Fine-tuning

### Q: Explain the difference between pre-training and fine-tuning. What are the different fine-tuning approaches?

**Answer:**

### Pre-training

Pre-training is the initial, large-scale training phase where a model learns general language understanding from massive unlabeled text corpora.

**Pre-training Objectives:**

1. **Causal Language Modeling (CLM)** -- Decoder-only models (GPT family):
   - Predict the next token given all previous tokens
   - Loss: Cross-entropy on next-token prediction
   - `P(x_t | x_1, x_2, ..., x_{t-1})`

2. **Masked Language Modeling (MLM)** -- Encoder models (BERT):
   - Randomly mask 15% of tokens; predict the masked tokens
   - Bidirectional context (can see both left and right)

3. **Span Corruption** -- Encoder-Decoder models (T5):
   - Replace random spans of text with sentinel tokens
   - Decoder reconstructs the original spans

**Pre-training Data Scale (examples):**

| Model | Training Tokens | Data Sources |
|-------|----------------|--------------|
| GPT-3 | 300B tokens | CommonCrawl, WebText2, Books, Wikipedia |
| Llama 2 | 2T tokens | Publicly available data |
| Llama 3 | 15T+ tokens | Broader multilingual corpus |
| DeepSeek V3 | 14.8T tokens | Diverse internet text |

**Pre-training Compute:**
- GPT-3 (175B): ~3,640 petaflop-days
- Llama 3 (405B): ~30,840,000 GPU-hours (H100)
- Cost: Millions to hundreds of millions of dollars

### Fine-tuning Approaches

**1. Full Fine-tuning:**
- Update ALL model parameters on task-specific data
- Highest quality but most expensive
- Requires storing a full copy of the model per task
- Risk of catastrophic forgetting

```python
# Full fine-tuning
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
# All parameters are trainable
for param in model.parameters():
    param.requires_grad = True
```

**2. LoRA (Low-Rank Adaptation):**
- Freezes original weights; adds small trainable rank-decomposition matrices
- Key insight: Weight updates during fine-tuning have low intrinsic rank
- Instead of updating W (d x d), learn A (d x r) and B (r x d) where r << d
- W' = W + A * B (merge at inference for zero overhead)

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 8,043,631,616 || trainable%: 0.1695
```

**3. QLoRA (Quantized LoRA):**
- Combines 4-bit quantization with LoRA
- Base model in 4-bit (NF4 data type), LoRA adapters in fp16/bf16
- Enables fine-tuning 65B parameter models on a single 48GB GPU

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantization_config=bnb_config
)
model = get_peft_model(model, lora_config)
```

**4. Prefix Tuning:**
- Prepends trainable "virtual tokens" to the input
- Only these prefix embeddings are trained
- Very parameter-efficient but can underperform LoRA

**5. Adapter Layers:**
- Inserts small trainable bottleneck layers between transformer layers
- Original weights frozen; only adapter parameters trained

**Comparison of Fine-tuning Methods:**

| Method | Trainable Params | Memory | Quality | Inference Overhead |
|--------|-----------------|--------|---------|-------------------|
| Full Fine-tuning | 100% | Very High | Best | None |
| LoRA | 0.1-1% | Low | Near-full | None (after merge) |
| QLoRA | 0.1-1% | Very Low | Good | Quantization loss |
| Prefix Tuning | <0.1% | Very Low | Lower | Slight |
| Adapters | 1-5% | Low | Good | Slight |

### Q: What is Supervised Fine-Tuning (SFT) and how does it work?

**Answer:**

SFT is the process of fine-tuning a pre-trained model on curated (instruction, response) pairs to make it follow instructions.

**Data format:**
```json
{
    "instruction": "Summarize the following article in 3 bullet points.",
    "input": "The article text goes here...",
    "output": "- Point 1\n- Point 2\n- Point 3"
}
```

**Training:**
- Only compute loss on the RESPONSE tokens (mask the instruction tokens)
- This is called "packing" or "completion-only" training

```python
# Loss masking for SFT
# Tokens: [INST] What is AI? [/INST] AI is the simulation of...
# Labels: [-100] [-100]  [-100] [-100]   AI  is the simulation of...
# -100 is ignored by CrossEntropyLoss
```

**Key SFT datasets:**
- OpenAssistant (OASST)
- Dolly
- ShareGPT (human ChatGPT conversations)
- Alpaca (GPT-4 generated)
- UltraChat
- FLAN Collection

---

---

---

<a id="section-3"></a>
# SECTION 3: RLHF, DPO, CONSTITUTIONAL AI, ALIGNMENT TECHNIQUES

## 3.1 The LLM Training Pipeline

### Q: Describe the complete training pipeline for a modern instruction-following LLM.

**Answer:**

The modern LLM training pipeline consists of three main stages:

```
Stage 1: Pre-training (Unsupervised)
    |  Massive text corpus (trillions of tokens)
    |  Objective: Next-token prediction
    |  Result: Base model (good at completion, not at following instructions)
    v
Stage 2: Supervised Fine-Tuning (SFT)
    |  Curated (instruction, response) pairs (~100K-1M examples)
    |  Objective: Learn to follow instructions
    |  Result: SFT model (follows instructions but may produce harmful/unhelpful content)
    v
Stage 3: Alignment (RLHF / DPO / etc.)
    |  Human preference data (chosen vs rejected responses)
    |  Objective: Align with human preferences (helpful, harmless, honest)
    |  Result: Aligned model (ready for deployment)
```

---

## 3.2 RLHF (Reinforcement Learning from Human Feedback)

### Q: Explain RLHF in detail. How does it work step by step?

**Answer:**

RLHF was popularized by InstructGPT (Ouyang et al., 2022) and is the technique behind ChatGPT's helpfulness.

**Step 1: Train a Reward Model (RM)**

Human annotators compare pairs of model outputs for the same prompt and indicate which response is better.

```
Prompt: "Explain quantum computing"

Response A: "Quantum computing uses qubits that can exist in superposition..."
Response B: "Quantum computing is a type of computing that uses quantum stuff..."

Human preference: A > B
```

The reward model is trained to predict human preferences:
```
Loss_RM = -log(sigmoid(r(x, y_chosen) - r(x, y_rejected)))
```

This is the Bradley-Terry model of preferences: the probability that response A is preferred over B is:
```
P(A > B) = sigmoid(r(A) - r(B))
```

```python
# Reward Model Training (simplified)
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_size, 1)  # scalar reward

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # last token
        reward = self.reward_head(last_hidden)
        return reward

# Training loop
for prompt, chosen, rejected in preference_data:
    r_chosen = reward_model(chosen)
    r_rejected = reward_model(rejected)
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    loss.backward()
```

**Step 2: Optimize Policy with PPO (Proximal Policy Optimization)**

Use the reward model to provide reward signals and optimize the LLM using RL:

```
Objective = E[r(x, y)] - beta * KL(pi_theta || pi_ref)
```

Where:
- `r(x, y)` is the reward model score
- `KL(pi_theta || pi_ref)` is the KL divergence between the current policy and the original SFT model
- `beta` controls the strength of the KL penalty (prevents reward hacking)

**Why the KL penalty?**
Without it, the model learns to "hack" the reward model -- producing outputs that get high reward scores but are not actually good (e.g., being excessively verbose, repeating phrases the RM likes).

```python
# PPO for RLHF (simplified with trl library)
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=16,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    kl_penalty="kl",         # KL divergence type
    init_kl_coef=0.2,        # Initial KL coefficient (beta)
    target_kl=6.0,           # Target KL divergence
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,              # SFT model being optimized
    ref_model=ref_model,      # Frozen copy of SFT model
    tokenizer=tokenizer,
    dataset=dataset,
)

for batch in ppo_trainer.dataloader:
    # Generate responses
    response_tensors = ppo_trainer.generate(batch["input_ids"])

    # Get reward scores
    rewards = reward_model(batch["input_ids"], response_tensors)

    # PPO update step
    stats = ppo_trainer.step(batch["input_ids"], response_tensors, rewards)
```

**Challenges with RLHF:**
1. **Reward Hacking**: Model exploits weaknesses in the reward model
2. **Training Instability**: PPO is notoriously unstable and sensitive to hyperparameters
3. **Expensive**: Requires training 4 models (SFT, RM, Policy, Reference)
4. **Human Labeling Cost**: High-quality preference data is expensive to collect
5. **Reward Model Limitations**: RM has its own biases and errors

---

## 3.3 DPO (Direct Preference Optimization)

### Q: What is DPO and how does it differ from RLHF?

**Answer:**

DPO (Rafailov et al., 2023) is an elegant alternative to RLHF that eliminates the need for a separate reward model and RL optimization loop.

**Key Insight:** The optimal policy under the RLHF objective has a closed-form solution. DPO reparameterizes the reward function in terms of the policy itself:

```
r(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x)) + beta * log(Z(x))
```

This means we can directly optimize the policy using preference data without ever training a reward model:

**DPO Loss:**
```
L_DPO = -E[log sigmoid(beta * (log(pi_theta(y_w|x)/pi_ref(y_w|x)) - log(pi_theta(y_l|x)/pi_ref(y_l|x))))]
```

Where:
- `y_w` = preferred (winning) response
- `y_l` = dispreferred (losing) response
- `pi_theta` = current policy being trained
- `pi_ref` = frozen reference (SFT) model
- `beta` = temperature parameter controlling deviation from reference

**Intuition:** DPO increases the probability of preferred responses and decreases the probability of dispreferred responses, relative to the reference model.

```python
# DPO Training (simplified)
import torch.nn.functional as F

def dpo_loss(pi_logprobs_chosen, pi_logprobs_rejected,
             ref_logprobs_chosen, ref_logprobs_rejected, beta=0.1):
    """
    Compute DPO loss.
    All inputs are sum of log probs over response tokens.
    """
    pi_logratios = pi_logprobs_chosen - pi_logprobs_rejected
    ref_logratios = ref_logprobs_chosen - ref_logprobs_rejected

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Useful metrics
    chosen_rewards = beta * (pi_logprobs_chosen - ref_logprobs_chosen)
    rejected_rewards = beta * (pi_logprobs_rejected - ref_logprobs_rejected)
    reward_margin = (chosen_rewards - rejected_rewards).mean()

    return loss, reward_margin

# Using trl library
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    loss_type="sigmoid",  # or "hinge", "ipo"
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

**DPO vs RLHF Comparison:**

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Reward Model | Required (separate training) | Not needed |
| RL Loop | Required (PPO) | Not needed |
| Models in Memory | 4 (policy, ref, RM, value) | 2 (policy, ref) |
| Training Stability | Unstable, sensitive to HPs | More stable |
| Compute Cost | Very high | Moderate |
| Data Efficiency | Can generate new data online | Offline (fixed dataset) |
| Performance | State-of-the-art | Competitive, sometimes better |
| Implementation | Complex | Simple (standard supervised) |

**DPO Variants:**
- **IPO (Identity Preference Optimization)**: Removes the log-sigmoid, more robust
- **KTO (Kahneman-Tversky Optimization)**: Only needs binary feedback (good/bad), not pairs
- **ORPO (Odds Ratio Preference Optimization)**: Combines SFT and preference optimization in one step
- **SimPO**: Simplified, reference-model-free preference optimization

---

## 3.4 Constitutional AI (CAI)

### Q: What is Constitutional AI and how does it work?

**Answer:**

Constitutional AI (Bai et al., 2022, Anthropic) is a method for training AI systems to be helpful, harmless, and honest using a set of principles ("constitution") rather than relying solely on human feedback.

**Two Phases:**

**Phase 1: Supervised Learning (Critique and Revision)**
1. Generate responses to harmful prompts using the model
2. Ask the model to critique its own response based on constitutional principles
3. Ask the model to revise its response based on the critique
4. Fine-tune on the revised responses

```
Prompt: "How do I pick a lock?"

Initial Response: "Here's how to pick a lock: First, get a tension wrench..."

Critique (based on principle "Choose the response that is least likely to be
used for illegal activities"):
"My response provides detailed instructions for an activity that could
facilitate illegal entry. This could be harmful."

Revision: "I can explain how locks work from an educational/locksmithing
perspective. Locks use pin tumbler mechanisms where..."
```

**Phase 2: RLAIF (RL from AI Feedback)**
- Instead of human annotators comparing responses, the AI model itself evaluates pairs of responses against constitutional principles
- Train a reward model on AI-generated preference labels
- Use RL (PPO) to optimize against this reward model

**Constitutional Principles (examples from Anthropic's paper):**
1. Choose the response that is most helpful to the human
2. Choose the response that is least harmful or toxic
3. Choose the response that is most honest and truthful
4. Choose the response that best supports human oversight of AI
5. Choose the response that is least likely to be used for illegal activities

**Advantages of CAI:**
- Scales better than human feedback (AI can generate unlimited comparisons)
- More consistent than human annotators
- Principles are transparent and auditable
- Can be updated by modifying the constitution
- Reduces the need for humans to interact with harmful content

---

## 3.5 Other Alignment Approaches

### Q: What other alignment techniques exist beyond RLHF and DPO?

**Answer:**

**1. RLAIF (RL from AI Feedback)**
- Replace human annotators with a strong AI model
- Used by Anthropic (Constitutional AI) and Google
- Trade off: cheaper and more scalable, but bounded by AI judge quality

**2. Rejection Sampling (Best-of-N)**
- Generate N responses, score them with a reward model, keep the best
- Simple but effective; used as a data generation strategy for training
- Llama 2 used rejection sampling extensively

**3. SPIN (Self-Play Fine-Tuning)**
- Model plays against itself: current model generates responses, previous model serves as baseline
- No human preference data needed after initial SFT

**4. Self-Rewarding Language Models (Meta, 2024)**
- Model acts as its own reward model
- Iteratively: generate -> self-judge -> train on preferences
- Can improve beyond the initial human data quality

**5. Iterative DPO / Online DPO**
- Instead of training on a fixed offline dataset, generate new preference pairs each iteration
- Combines benefits of DPO simplicity with online data generation

**6. GRPO (Group Relative Policy Optimization) -- DeepSeek**
- Used in DeepSeek R1
- Groups multiple responses and uses group-relative rewards
- Eliminates need for a critic/value model
- More memory efficient than PPO

```
Standard PPO: reward = R(y) - V(s)  (requires value model)
GRPO: reward = (R(y) - mean(R(group))) / std(R(group))  (no value model)
```

---

---

---

<a id="section-4"></a>
# SECTION 4: DECODING STRATEGIES, SAMPLING, EVALUATION METRICS

## 4.1 Decoding / Text Generation Strategies

### Q: Explain the different decoding strategies for LLMs. When would you use each?

**Answer:**

When an LLM generates text, at each step it produces a probability distribution over the entire vocabulary. The **decoding strategy** determines how we select the next token from that distribution.

### Greedy Decoding

**Method:** Always select the token with the highest probability.

```python
def greedy_decode(model, input_ids, max_length):
    for _ in range(max_length):
        logits = model(input_ids).logits[:, -1, :]  # last token logits
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == eos_token_id:
            break
    return input_ids
```

**Pros:** Fast, deterministic, reproducible
**Cons:** Often produces repetitive, dull text. Can miss globally optimal sequences.
**Use when:** You need deterministic output (code generation, structured data extraction)

### Beam Search

**Method:** Maintain top-k (beam_width) candidate sequences at each step. Expand each candidate, score all possibilities, keep the top-k.

```
Beam Width = 3, generating after "The cat"

Step 1:
  Beam 1: "The cat sat"     (score: -2.1)
  Beam 2: "The cat is"      (score: -2.3)
  Beam 3: "The cat was"     (score: -2.5)

Step 2 (expand each beam, keep top 3 overall):
  Beam 1: "The cat sat on"      (score: -4.0)
  Beam 2: "The cat is a"        (score: -4.2)
  Beam 3: "The cat sat down"    (score: -4.3)
  (others pruned)
```

```python
def beam_search(model, input_ids, beam_width=5, max_length=50):
    beams = [(input_ids, 0.0)]  # (sequence, cumulative log prob)

    for _ in range(max_length):
        all_candidates = []
        for seq, score in beams:
            logits = model(seq).logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            top_k_log_probs, top_k_ids = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                new_seq = torch.cat([seq, top_k_ids[:, i:i+1]], dim=-1)
                new_score = score + top_k_log_probs[0, i].item()
                all_candidates.append((new_seq, new_score))

        # Keep top beam_width candidates (length-normalized)
        beams = sorted(all_candidates, key=lambda x: x[1]/x[0].size(1),
                       reverse=True)[:beam_width]

    return beams[0][0]
```

**Pros:** Finds higher-probability sequences than greedy
**Cons:** Still deterministic, tends to produce generic text, computationally expensive
**Use when:** Machine translation, summarization (where accuracy > creativity)

**Length Penalty in Beam Search:**
Beam search favors shorter sequences (fewer log prob terms to sum). Length penalty corrects this:
```
score(Y) = log P(Y) / |Y|^alpha
```
Where alpha > 1 favors longer sequences, alpha < 1 favors shorter ones.

---

## 4.2 Sampling Strategies

### Q: Explain Temperature, Top-k, and Top-p (Nucleus) Sampling.

**Answer:**

### Temperature Sampling

**Method:** Scale the logits by a temperature parameter T before applying softmax.

```
P(token_i) = exp(logit_i / T) / sum(exp(logit_j / T))
```

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| T = 0 | Equivalent to greedy (argmax) | Deterministic tasks |
| T = 0.1-0.3 | Very focused, low randomness | Code generation, factual QA |
| T = 0.7-0.9 | Balanced creativity/coherence | General conversation |
| T = 1.0 | Original distribution | Baseline |
| T > 1.0 | More random, flatter distribution | Creative writing, brainstorming |
| T -> infinity | Uniform distribution | Maximum randomness |

```python
def temperature_sample(logits, temperature=0.7):
    """Apply temperature and sample."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Mathematical intuition:** Lower temperature "sharpens" the distribution (makes high-probability tokens even more likely), while higher temperature "flattens" it (gives low-probability tokens more chance).

### Top-k Sampling

**Method:** Only consider the top k tokens with highest probability. Redistribute probability mass among them.

```python
def top_k_sample(logits, k=50, temperature=1.0):
    """Top-k sampling."""
    scaled_logits = logits / temperature

    # Keep only top-k
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k)

    # Zero out everything else
    filtered_logits = torch.full_like(scaled_logits, float('-inf'))
    filtered_logits.scatter_(1, top_k_indices, top_k_logits)

    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Example:**
```
Vocabulary: [the, cat, sat, dog, xyz, abc, ...]
Probabilities: [0.4, 0.25, 0.15, 0.1, 0.001, 0.0005, ...]

Top-k=3: Only consider [the, cat, sat]
Renormalized: [0.5, 0.3125, 0.1875]
```

**Problem with fixed k:** If the distribution is very peaked (one token has 0.95 probability), k=50 includes many irrelevant tokens. If the distribution is flat, k=50 might be too restrictive.

### Top-p (Nucleus) Sampling

**Method:** Select the smallest set of tokens whose cumulative probability exceeds p. This adapts the number of candidates based on the distribution shape.

```python
def top_p_sample(logits, p=0.9, temperature=1.0):
    """Nucleus (top-p) sampling."""
    scaled_logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift right to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Set removed logits to -inf
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scaled_logits[indices_to_remove] = float('-inf')

    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Example:**
```
Sorted probabilities: [0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02, ...]

Top-p=0.9:
  Cumsum: [0.4, 0.65, 0.80, 0.90, ...]
  Include tokens until cumsum >= 0.9
  Result: top 4 tokens selected (adaptive!)

If distribution is peaked: [0.92, 0.05, 0.02, ...]
  Top-p=0.9: Only 1 token selected!

If distribution is flat: [0.1, 0.09, 0.08, 0.08, ...]
  Top-p=0.9: ~12 tokens selected!
```

**Typical production settings:**
```python
# OpenAI API defaults
temperature = 1.0
top_p = 1.0

# Conservative (factual)
temperature = 0.2
top_p = 0.1

# Balanced
temperature = 0.7
top_p = 0.9

# Creative
temperature = 1.0
top_p = 0.95
```

### Min-p Sampling (Newer approach)

**Method:** Set a minimum probability threshold relative to the top token. Include only tokens with `P(token) >= min_p * P(top_token)`.

```python
def min_p_sample(logits, min_p=0.05, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    top_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * top_prob
    mask = probs < threshold
    logits[mask] = float('-inf')
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Advantage:** More intuitive than top-p; adapts naturally to distribution shape.

### Repetition Penalty

Penalize tokens that have already appeared in the generated text:
```python
def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    for token_id in set(generated_tokens):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits
```

### Frequency Penalty vs Presence Penalty (OpenAI API)

- **Frequency Penalty**: Penalizes tokens proportional to how many times they appear
  - `logit -= frequency_penalty * count(token)`
- **Presence Penalty**: Flat penalty if token has appeared at all
  - `logit -= presence_penalty * (1 if token appeared else 0)`

---

## 4.3 Model Evaluation Metrics

### Q: Explain the major evaluation metrics for LLMs.

**Answer:**

### Perplexity (PPL)

**Definition:** Exponentiated average negative log-likelihood per token. Measures how "surprised" the model is by the text.

```
PPL = exp(-1/N * sum(log P(token_i | context_i)))
```

```python
import torch
import torch.nn.functional as F

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss  # average NLL per token

    perplexity = torch.exp(neg_log_likelihood)
    return perplexity.item()

# Example
# PPL of GPT-3 on WikiText-103: ~20.5
# PPL of GPT-4 on same: ~8-10 (estimated)
```

**Interpretation:**
- PPL = 1: Perfect prediction (impossible in practice)
- PPL = 10: Model is as uncertain as choosing among 10 equally likely tokens
- PPL = 100: Very uncertain
- Lower is better (the model predicts the actual text well)

**Limitations:** Only measures next-token prediction ability, not generation quality, helpfulness, or safety.

### BLEU (Bilingual Evaluation Understudy)

**Originally for:** Machine translation
**How it works:** Computes n-gram precision between generated text and reference text.

```
BLEU = BP * exp(sum(w_n * log(p_n)))

BP (Brevity Penalty) = min(1, exp(1 - ref_len/gen_len))
p_n = modified n-gram precision
w_n = weight for each n-gram (typically uniform: 1/4 for BLEU-4)
```

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "on", "the", "mat"]

# BLEU-4 (considers 1,2,3,4-grams)
score = sentence_bleu(reference, candidate)
print(f"BLEU: {score:.4f}")  # ~0.61
```

**Limitations:**
- Only measures lexical overlap (synonyms score 0)
- Does not capture semantic similarity
- Brevity penalty can be harsh
- Scores are not comparable across different test sets

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Originally for:** Summarization
**Key difference from BLEU:** Measures RECALL (how much of the reference is captured) vs BLEU's precision.

**Variants:**
- **ROUGE-1**: Unigram overlap (word-level recall)
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest Common Subsequence (captures sentence-level structure)
- **ROUGE-Lsum**: ROUGE-L at summary level (splits by newlines)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference = "The cat sat on the mat near the door."
generated = "A cat was sitting on the mat."

scores = scorer.score(reference, generated)
# ROUGE-1: Precision=0.71, Recall=0.56, F1=0.62
# ROUGE-2: Precision=0.40, Recall=0.29, F1=0.33
# ROUGE-L: Precision=0.57, Recall=0.44, F1=0.50
```

### BERTScore

**Method:** Uses contextual embeddings (BERT) to compute semantic similarity between generated and reference text.

```
BERTScore = F1(Precision_BERT, Recall_BERT)

Where:
Precision_BERT = avg over generated tokens of max cosine similarity to any reference token
Recall_BERT = avg over reference tokens of max cosine similarity to any generated token
```

```python
from bert_score import score

references = ["The cat sat on the mat."]
candidates = ["A feline rested on the rug."]

P, R, F1 = score(candidates, references, lang="en", model_type="microsoft/deberta-xlarge-mnli")
print(f"BERTScore F1: {F1.item():.4f}")  # Much higher than BLEU for paraphrases
```

**Advantages over BLEU/ROUGE:**
- Captures semantic similarity (synonyms, paraphrases)
- Correlates better with human judgments
- Language-agnostic (works with multilingual BERT)

### LLM-as-a-Judge

**Modern trend:** Use a strong LLM (GPT-4, Claude) to evaluate outputs.

```python
evaluation_prompt = """
Rate the following response on a scale of 1-10 for:
1. Helpfulness
2. Accuracy
3. Relevance
4. Completeness

Question: {question}
Response: {response}

Provide your ratings and justification.
"""
```

**Common frameworks:**
- **MT-Bench**: Multi-turn benchmark scored by GPT-4
- **AlpacaEval**: Automated evaluation using LLM judges
- **Chatbot Arena**: Human preference voting (Elo ratings)

### Other Important Metrics

| Metric | Purpose | Used For |
|--------|---------|----------|
| METEOR | MT evaluation (includes synonyms, stemming) | Translation |
| CIDEr | Image captioning consensus | Vision-Language |
| MAUVE | Distribution similarity for open-ended generation | Text generation |
| TruthfulQA | Measures truthfulness | Hallucination evaluation |
| HumanEval / MBPP | Code correctness (pass@k) | Code generation |
| GSM8K | Math reasoning accuracy | Reasoning evaluation |
| MMLU | Multi-task accuracy across 57 subjects | General knowledge |
| HellaSwag | Commonsense reasoning | NLU evaluation |
| ARC | Science question answering | Reasoning |
| WinoGrande | Coreference resolution | NLU |

---

---

---

<a id="section-5"></a>
# SECTION 5: HALLUCINATION, SCALING LAWS, EMERGENT ABILITIES, CHAIN-OF-THOUGHT, IN-CONTEXT LEARNING

## 5.1 Hallucination

### Q: What are LLM hallucinations? What causes them and how can they be mitigated?

**Answer:**

**Definition:** Hallucination occurs when an LLM generates content that is factually incorrect, fabricated, or inconsistent with the provided context, but presents it with apparent confidence.

**Types of Hallucination:**

1. **Intrinsic Hallucination**: Output contradicts the source material
   - Example: Given a document saying "Revenue was $5B," the model says "Revenue was $8B"

2. **Extrinsic Hallucination**: Output contains information not present in the source and cannot be verified
   - Example: Model invents a citation: "According to Smith et al. (2023)..." (paper does not exist)

3. **Factual Hallucination**: Generates false facts about the world
   - Example: "The Eiffel Tower is located in London"

4. **Faithfulness Hallucination**: Fails to follow given context/instructions
   - Example: Summarization task where model adds information not in the original

**Root Causes:**

1. **Training Data Issues:**
   - Contradictory information in training corpus
   - Outdated information (knowledge cutoff)
   - Noise, errors, and biases in web-scraped data
   - Memorization of incorrect patterns

2. **Architectural / Objective Limitations:**
   - Models are trained to predict plausible next tokens, not truthful ones
   - Autoregressive generation: errors compound (exposure bias)
   - No explicit mechanism for "I don't know"
   - Softmax over vocabulary always assigns non-zero probability to every token

3. **Decoding Issues:**
   - Sampling can lead to low-probability (nonsensical) continuations
   - Long generation amplifies error accumulation

4. **Knowledge Boundary Problem:**
   - Models cannot distinguish between what they "know" (high confidence from training) and what they are "making up" (confabulation)

**Mitigation Strategies:**

| Strategy | Description | Effectiveness |
|----------|-------------|---------------|
| **RAG** | Ground responses in retrieved documents | High |
| **Grounding** | Require citations / source attribution | High |
| **Chain-of-Thought** | Step-by-step reasoning reduces errors | Medium-High |
| **Self-Consistency** | Sample multiple answers, pick majority | Medium-High |
| **Fine-tuning** | Train on factually verified data | Medium |
| **RLHF/DPO** | Penalize hallucinated content | Medium |
| **Confidence Calibration** | Train model to express uncertainty | Medium |
| **Tool Use** | Let model call APIs, search engines, calculators | High |
| **Knowledge Graphs** | Ground in structured knowledge | Medium |
| **Prompt Engineering** | "If you don't know, say so" | Low-Medium |

**RAG (Retrieval-Augmented Generation) in detail:**
```python
# RAG Pipeline
def rag_generate(query, model, retriever, top_k=5):
    # Step 1: Retrieve relevant documents
    docs = retriever.retrieve(query, top_k=top_k)

    # Step 2: Construct augmented prompt
    context = "\n\n".join([doc.text for doc in docs])
    prompt = f"""Answer the question based ONLY on the following context.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question: {query}
    Answer:"""

    # Step 3: Generate grounded response
    response = model.generate(prompt)
    return response
```

**Self-Consistency Decoding:**
```python
def self_consistency(model, prompt, n_samples=10, temperature=0.7):
    """Generate multiple answers and pick the most common one."""
    answers = []
    for _ in range(n_samples):
        response = model.generate(prompt, temperature=temperature)
        answer = extract_final_answer(response)
        answers.append(answer)

    # Majority vote
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

---

## 5.2 Scaling Laws

### Q: What are neural scaling laws? Explain the Chinchilla scaling laws.

**Answer:**

**Scaling laws** describe predictable relationships between model performance and key factors: model size (N), dataset size (D), and compute budget (C).

**Kaplan et al. (2020) - OpenAI Scaling Laws:**

Found that loss follows power law relationships:
```
L(N) = (N_c / N)^alpha_N         (varying model size)
L(D) = (D_c / D)^alpha_D         (varying data size)
L(C) = (C_c / C)^alpha_C         (varying compute)

Where alpha_N ~ 0.076, alpha_D ~ 0.095, alpha_C ~ 0.050
```

**Key finding:** Performance improves smoothly and predictably as a power law. No plateau observed across 7 orders of magnitude.

**Original conclusion (OpenAI):** Model size matters more than data size. For a fixed compute budget, train a LARGER model on LESS data.

**Chinchilla Scaling Laws (Hoffmann et al., 2022 - DeepMind):**

Re-analyzed scaling and found the OPTIMAL ratio:
```
For compute-optimal training:
N_opt proportional to C^0.5
D_opt proportional to C^0.5

Rule of thumb: Tokens = 20 * Parameters
```

**This means:** A 10B parameter model should be trained on ~200B tokens for compute-optimal training.

**Impact on the field:**

| Model | Parameters | Training Tokens | Tokens/Params Ratio | Chinchilla-Optimal? |
|-------|-----------|----------------|--------------------|--------------------|
| GPT-3 | 175B | 300B | 1.7x | Undertrained |
| Chinchilla | 70B | 1.4T | 20x | Optimal |
| Llama | 65B | 1.4T | 21.5x | Optimal |
| Llama 2 | 70B | 2T | 28.6x | Over-trained (intentionally) |
| Llama 3 | 8B | 15T | 1875x | Heavily over-trained |
| Llama 3 | 405B | 15T | 37x | Over-trained |

**Why over-train?** The Chinchilla-optimal point minimizes training compute, but in deployment, INFERENCE cost dominates. A smaller model trained on more data can match a larger Chinchilla-optimal model while being cheaper to serve. This is the "inference-optimal" paradigm.

**Scaling Laws for Downstream Tasks:**
- Performance on benchmarks (MMLU, GSM8K) also follows smooth power laws with scale
- BUT some tasks show "phase transitions" (emergent abilities, see below)

---

## 5.3 Emergent Abilities

### Q: What are emergent abilities in LLMs? Are they real?

**Answer:**

**Definition (Wei et al., 2022):** Emergent abilities are capabilities that are not present in smaller models but appear in larger models -- they are not predicted by simple extrapolation of scaling curves.

**Claimed Examples:**
- **Arithmetic**: Small models fail at multi-digit addition; large models (>100B) can do it
- **Chain-of-thought reasoning**: Only works in models above ~60B parameters
- **Word unscrambling**: Models suddenly gain this ability at a certain scale
- **Multi-step reasoning**: Compound tasks that require intermediate steps

**The Controversy (Schaeffer et al., 2023 - "Are Emergent Abilities a Mirage?"):**

This influential paper argues that emergent abilities are largely an artifact of the evaluation metric, not the model:

- When using **non-linear/discontinuous metrics** (like exact match accuracy), performance appears to suddenly jump
- When using **linear/continuous metrics** (like token-level edit distance or log-likelihood), performance improves smoothly and predictably

**Example:**
```
Task: Multi-digit arithmetic "What is 1234 + 5678?"
Correct answer: "6912"

Exact match metric:
- Model outputs "6913" -> Score: 0 (wrong!)
- Model outputs "6912" -> Score: 1 (right!)
- Appears as sudden jump from 0 to 1

Token-level metric:
- Model outputs "6913" -> Score: 0.75 (3/4 digits correct)
- Model outputs "6912" -> Score: 1.0
- Smooth improvement visible
```

**Current consensus:** Whether "emergence" is real depends on your definition. Performance does improve with scale. Some capabilities genuinely require a minimum scale to be useful. But the dramatic "phase transition" narrative is partially an artifact of how we measure.

**Practical takeaway for interviews:** Acknowledge both perspectives. Say that certain capabilities become practically useful above a threshold of scale, but the underlying capability often improves continuously.

---

## 5.4 Chain-of-Thought (CoT) Reasoning

### Q: Explain Chain-of-Thought prompting and its variants.

**Answer:**

**Chain-of-Thought (Wei et al., 2022)** is a prompting technique where the model is encouraged to generate intermediate reasoning steps before producing a final answer.

**Standard prompting vs CoT:**
```
Standard:
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many total?
A: 11

CoT:
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many total?
A: Roger starts with 5 balls. He buys 2 cans of 3 balls each, so 2 * 3 = 6 new balls.
   Total = 5 + 6 = 11. The answer is 11.
```

**Why does CoT work?**
1. Breaks complex problems into simpler sub-problems
2. Provides intermediate "scratch space" for computation
3. Reduces the number of reasoning steps per token generated
4. Allows error checking at intermediate steps

**CoT Variants:**

**1. Few-Shot CoT (original):**
Provide examples with reasoning chains in the prompt.

**2. Zero-Shot CoT (Kojima et al., 2022):**
Simply add "Let's think step by step" to the prompt.
```
Q: [problem]
A: Let's think step by step.
```
Surprisingly effective! This single phrase improves math accuracy on GSM8K from ~18% to ~41% for some models.

**3. Self-Consistency (Wang et al., 2022):**
Sample multiple CoT reasoning paths and take the majority vote on the final answer.
```
Path 1: ... -> Answer: 42
Path 2: ... -> Answer: 42
Path 3: ... -> Answer: 38
Majority vote: 42
```

**4. Tree of Thought (Yao et al., 2023):**
Explore multiple reasoning branches at each step, evaluate them, and backtrack if needed.
```
Problem -> [Thought 1a] -> [Thought 2a] -> [Thought 3a] -> Answer A
                        -> [Thought 2b] -> (dead end, backtrack)
        -> [Thought 1b] -> [Thought 2c] -> [Thought 3b] -> Answer B
Evaluate: Answer A is better
```

**5. ReAct (Reason + Act) (Yao et al., 2022):**
Interleave reasoning with actions (tool calls, searches).
```
Thought: I need to find when the Eiffel Tower was built.
Action: search("Eiffel Tower construction date")
Observation: Construction began in 1887 and was completed in 1889.
Thought: So it was completed in 1889. Let me calculate the age.
Action: calculate(2025 - 1889)
Observation: 136
Answer: The Eiffel Tower is 136 years old.
```

**6. Program-of-Thought (PoT):**
Generate code instead of natural language reasoning:
```python
# Q: If a train travels at 60 mph for 2.5 hours, how far does it go?
speed = 60  # mph
time = 2.5  # hours
distance = speed * time
print(distance)  # 150 miles
```

---

## 5.5 In-Context Learning (ICL)

### Q: What is in-context learning and why is it remarkable?

**Answer:**

**Definition:** In-context learning is the ability of LLMs to learn new tasks from examples provided in the prompt, WITHOUT any gradient updates or fine-tuning.

```
# In-context learning example (few-shot)
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivree
plush giraffe => girafe en peluche
cheese => fromage
```

**Why it is remarkable:**
- The model was never explicitly trained on this translation format
- It "learns" the task pattern from just a few examples
- No weight updates occur -- the model uses only its forward pass
- Performance improves with more examples (up to context window limits)

**Types of ICL:**

| Type | # Examples | Description |
|------|-----------|-------------|
| Zero-shot | 0 | Only task instruction, no examples |
| One-shot | 1 | One example provided |
| Few-shot | 2-32+ | Multiple examples provided |

**Theoretical explanations (active research area):**

1. **Implicit Bayesian Inference**: The model performs approximate Bayesian inference over possible tasks given the examples
2. **Gradient Descent Analogy (Akyurek et al., 2022)**: Transformers implement something equivalent to gradient descent in their forward pass -- attention layers can simulate learning algorithms
3. **Task Location**: Pre-training exposed the model to similar patterns; ICL is retrieving and adapting a learned pattern
4. **Induction Heads (Olsson et al., 2022)**: Specific attention head circuits that implement copying/pattern-matching

**Best practices for ICL:**
1. Use diverse, representative examples
2. Order matters: put easier examples first or use random ordering
3. Format consistency: keep input-output format identical across examples
4. Label balance: equal representation of each class
5. More examples generally helps (diminishing returns after ~8-16)
6. Examples closest to the query in distribution tend to work best

---

---

---

<a id="section-6"></a>
# SECTION 6: COMMON INTERVIEW QUESTIONS WITH DETAILED ANSWERS

## Category A: Foundational Understanding

### Q1: What makes LLMs different from traditional NLP models?

**Answer:**
Traditional NLP models (pre-2017) were task-specific: you trained a separate model for sentiment analysis, NER, translation, etc. Each required labeled data and custom architectures. LLMs differ in fundamental ways:

1. **Scale**: Billions of parameters vs millions
2. **Pre-training**: Learned from trillions of tokens of unsupervised text
3. **Generality**: One model handles many tasks without task-specific training
4. **In-context learning**: Can adapt to new tasks via prompting alone
5. **Emergent capabilities**: Abilities that arise from scale (reasoning, code generation, math)
6. **Foundation model paradigm**: Pre-train once, fine-tune or prompt for many downstream tasks

**Timeline of the shift:**
```
Word2Vec (2013)     -> Static word embeddings
ELMo (2018)         -> Contextualized embeddings
BERT (2018)         -> Pre-train + fine-tune paradigm
GPT-2 (2019)        -> Showed in-context learning potential
GPT-3 (2020)        -> Few-shot learning via prompting
ChatGPT (2022)      -> RLHF-aligned conversational AI
GPT-4 (2023)        -> Multimodal, near-human on many benchmarks
GPT-4o/Claude 3.5/  -> Reasoning, agentic, long-context
  Gemini 1.5 (2024)
```

---

### Q2: Explain the difference between autoregressive and autoencoding language models.

**Answer:**

| Feature | Autoregressive (AR) | Autoencoding (AE) |
|---------|--------------------|--------------------|
| Direction | Left-to-right (causal) | Bidirectional |
| Objective | Next-token prediction | Masked token prediction |
| Architecture | Decoder-only | Encoder-only |
| Examples | GPT, Llama, Claude, Mistral | BERT, RoBERTa, DeBERTa |
| Strengths | Text generation | Understanding, classification |
| Inference | Sequential (slow for long text) | Parallel (full sequence at once) |
| Attention | Causal mask (can only see past) | Full attention (sees all tokens) |

**AR models** generate one token at a time, left to right:
```
P(x1, x2, ..., xn) = P(x1) * P(x2|x1) * P(x3|x1,x2) * ... * P(xn|x1,...,xn-1)
```

**AE models** reconstruct masked tokens using full context:
```
Input:  "The [MASK] sat on the [MASK]"
Output: "The [cat]  sat on the [mat]"
```

**Why AR dominates for LLMs:** Generation is the core capability. Bidirectional models cannot generate text autoregressively (they need the full sequence). Also, the next-token prediction objective turns out to be a remarkably powerful learning signal at scale.

---

### Q3: What is the KV Cache and why is it critical for LLM inference?

**Answer:**

During autoregressive generation, at each step the model computes attention over all previous tokens. Without caching, this means recomputing K and V for all previous tokens at every step -- O(n^2) total computation for generating n tokens.

**KV Cache** stores the Key and Value projections from previous steps, so each new step only computes K,V for the NEW token and reuses cached values.

```
Without KV Cache (generating 4 tokens):
Step 1: Compute K,V for [t1]                  -> 1 computation
Step 2: Compute K,V for [t1, t2]              -> 2 computations
Step 3: Compute K,V for [t1, t2, t3]          -> 3 computations
Step 4: Compute K,V for [t1, t2, t3, t4]      -> 4 computations
Total: 1+2+3+4 = 10 computations

With KV Cache:
Step 1: Compute K,V for [t1], cache them        -> 1 computation
Step 2: Compute K,V for [t2], append to cache    -> 1 computation
Step 3: Compute K,V for [t3], append to cache    -> 1 computation
Step 4: Compute K,V for [t4], append to cache    -> 1 computation
Total: 4 computations
```

**Memory cost of KV Cache:**
```
KV Cache Size = 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * bytes_per_element

Example (Llama 3 70B, bf16):
= 2 * 80 * 8 * 128 * 8192 * 1 * 2 bytes
= ~2.68 GB per sequence at 8K context

At 128K context: ~42.9 GB per sequence!
```

**Optimization techniques:**
- **GQA/MQA**: Reduce n_kv_heads (Llama 2 70B: 8 KV heads vs 64 query heads)
- **PagedAttention (vLLM)**: Memory management like OS virtual memory, reduces fragmentation
- **Quantized KV Cache**: Store KV in int8 or even int4
- **Sliding Window**: Only cache last W tokens (Mistral)

---

### Q4: Explain the concept of "attention sinks" and how they affect LLM behavior.

**Answer:**

**Attention sinks** (Xiao et al., 2023) refer to the phenomenon where LLMs allocate disproportionately high attention to the first few tokens (especially the BOS/first token) regardless of their semantic relevance.

**Why it happens:**
- Softmax requires attention weights to sum to 1
- When no token is particularly relevant, the model "dumps" excess attention on the first token as a no-op
- The first token acts as a "sink" for unnecessary attention mass
- This is a learned behavior, not a bug

**Practical implication -- Streaming LLM:**
- If you use a sliding window and evict the first tokens, performance degrades catastrophically
- Solution: Always keep the first few "sink" tokens in the KV cache, even with a sliding window

```
Standard sliding window (window=4):
Tokens: [t1, t2, t3, t4, t5, t6, t7, t8]
Cache at step 8: [t5, t6, t7, t8]  <- Lost t1, performance drops!

StreamingLLM approach:
Cache at step 8: [t1, t2, t6, t7, t8]  <- Keep sink tokens + recent window
```

---

### Q5: What is Flash Attention and why is it important?

**Answer:**

**Flash Attention** (Dao et al., 2022) is an IO-aware exact attention algorithm that makes attention both faster and more memory-efficient.

**The problem:** Standard attention materializes the full N x N attention matrix in GPU HBM (High Bandwidth Memory). This is:
- Memory: O(N^2) -- for N=128K, the matrix is 128K x 128K = 16 billion elements
- Slow: Bottleneck is memory bandwidth, not compute

**Flash Attention solution:**
- Uses **tiling**: breaks Q, K, V into blocks that fit in SRAM (fast on-chip memory)
- Computes attention block-by-block, never materializing the full N x N matrix
- Uses online softmax trick to compute exact attention without storing intermediate results
- Memory: O(N) instead of O(N^2)
- Speed: 2-4x faster than standard attention

```
Standard Attention:
HBM -> load Q,K,V -> compute S=QK^T (write N^2 to HBM) -> softmax (read/write N^2)
-> compute P*V (read N^2) -> write output to HBM

Flash Attention:
HBM -> load Q,K,V blocks to SRAM -> compute everything in SRAM -> write only final
output to HBM (never write N^2 matrix)
```

**Flash Attention 2** improvements: Better parallelism across sequence length and batch dimensions. 2x faster than Flash Attention 1.

**Flash Attention 3** (2024): Leverages Hopper GPU features (TMA, FP8), async operations.

```python
# Using Flash Attention in PyTorch (built-in since 2.0)
import torch.nn.functional as F

# Automatically uses Flash Attention when available
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    is_causal=True,  # for decoder models
    dropout_p=0.0
)
```

---

### Q6: Explain model quantization and its impact on LLM deployment.

**Answer:**

**Quantization** reduces the precision of model weights (and sometimes activations) from higher-precision formats to lower ones, reducing memory footprint and increasing inference speed.

**Precision formats:**
```
FP32 (32-bit float):    1 sign + 8 exponent + 23 mantissa  = 4 bytes
FP16 (16-bit float):    1 sign + 5 exponent + 10 mantissa  = 2 bytes
BF16 (bfloat16):        1 sign + 8 exponent + 7 mantissa   = 2 bytes
INT8 (8-bit integer):   Range: -128 to 127                  = 1 byte
INT4 (4-bit integer):   Range: -8 to 7                      = 0.5 bytes
```

**Memory savings:**
```
Llama 3 70B:
FP32: 280 GB  (70B * 4 bytes)
FP16: 140 GB  (70B * 2 bytes)
INT8:  70 GB  (70B * 1 byte)
INT4:  35 GB  (70B * 0.5 bytes)
```

**Quantization techniques:**

1. **Post-Training Quantization (PTQ)**: Quantize after training
   - **GPTQ**: Layer-by-layer quantization with calibration data (most popular for 4-bit)
   - **AWQ (Activation-Aware Weight Quantization)**: Protects salient weights based on activation magnitudes
   - **GGUF/GGML**: Quantization format for CPU inference (used by llama.cpp)
   - **SqueezeLLM**: Non-uniform quantization preserving outlier weights

2. **Quantization-Aware Training (QAT)**: Train with simulated quantization
   - Higher quality but requires full training run
   - BitNet: 1-bit weights from scratch

3. **NF4 (Normal Float 4-bit)**: Data type used in QLoRA
   - Optimally quantizes normally distributed weights
   - Better than uniform INT4 for neural network weights

**Quality impact:**
```
Llama 2 70B on MMLU:
FP16:  69.8%
INT8:  69.5% (-0.3%)
INT4 (GPTQ): 68.9% (-0.9%)
INT4 (AWQ):  69.2% (-0.6%)
INT3:  66.1% (-3.7%)  -- significant degradation
```

---

### Q7: What is speculative decoding and how does it speed up LLM inference?

**Answer:**

**Speculative decoding** uses a small, fast "draft" model to generate candidate tokens, then verifies them in parallel with the large "target" model. Since verification is parallelizable (single forward pass), this can give 2-3x speedup.

```
Traditional autoregressive (target model only):
Step 1: Generate token 1  [slow]
Step 2: Generate token 2  [slow]
Step 3: Generate token 3  [slow]
...

Speculative decoding:
Step 1: Draft model generates tokens [1,2,3,4,5]         [fast]
Step 2: Target model verifies all 5 in ONE forward pass   [one slow step]
Step 3: Accept tokens 1,2,3 (match), reject 4,5
Step 4: Draft model generates new candidates from token 3  [fast]
...
```

**Key property:** The output distribution is IDENTICAL to the target model alone. Speculative decoding is lossless -- it is a pure speedup technique.

```python
def speculative_decode(target_model, draft_model, prompt, gamma=5):
    """
    gamma: number of tokens to draft at each step
    """
    tokens = prompt

    while not done:
        # Draft phase: generate gamma tokens with small model
        draft_tokens = []
        draft_probs = []
        for _ in range(gamma):
            p = draft_model(tokens + draft_tokens)
            t = sample(p)
            draft_tokens.append(t)
            draft_probs.append(p[t])

        # Verify phase: run target model on all draft tokens at once
        target_probs = target_model(tokens + draft_tokens)  # single forward pass

        # Accept/reject each draft token
        accepted = 0
        for i in range(gamma):
            r = random.random()
            if r < target_probs[i][draft_tokens[i]] / draft_probs[i]:
                accepted += 1  # accept this token
            else:
                # Reject: sample from adjusted distribution
                resample from (target_probs[i] - draft_probs[i]).clamp(min=0)
                break

        tokens.extend(draft_tokens[:accepted] + [resampled_token])

    return tokens
```

---

## Category B: Applied / System Design Questions

### Q8: How would you design a production RAG system?

**Answer:**

**RAG Architecture:**

```
User Query
    |
    v
[Query Processing]
    |  - Query rewriting / expansion
    |  - HyDE (Hypothetical Document Embedding)
    v
[Retrieval]
    |  - Embedding model encodes query
    |  - Vector similarity search (ANN)
    |  - Optional: hybrid search (dense + sparse)
    |  - Optional: reranking
    v
[Retrieved Documents] (top-k)
    |
    v
[Context Assembly]
    |  - Chunk selection & ordering
    |  - Deduplication
    |  - Context window management
    v
[LLM Generation]
    |  - System prompt with instructions
    |  - Retrieved context + user query
    |  - Citation generation
    v
[Post-processing]
    |  - Citation verification
    |  - Hallucination detection
    |  - Response formatting
    v
Final Response
```

**Key components in detail:**

**1. Document Processing Pipeline:**
```python
# Chunking strategies
# Fixed-size chunks (simple but may break context)
chunks = split_text(doc, chunk_size=512, overlap=50)

# Semantic chunking (split at paragraph/section boundaries)
chunks = semantic_split(doc, max_chunk_size=512)

# Recursive character text splitting (LangChain default)
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)
```

**2. Embedding Models:**
```python
# Popular embedding models (2025)
# OpenAI: text-embedding-3-large (3072 dim)
# Cohere: embed-v3 (1024 dim)
# Open source: BGE-M3, E5-Mistral, GTE-Qwen2
# Specialized: Nomic-embed, Jina-embeddings-v3

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
embeddings = model.encode(chunks)
```

**3. Vector Store:**
```python
# Options: Pinecone, Weaviate, Qdrant, Milvus, ChromaDB, pgvector
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "doc1.pdf", "page": i} for i in range(len(chunks))]
)

results = collection.query(query_embeddings=[query_embedding], n_results=5)
```

**4. Advanced RAG Techniques:**
- **HyDE**: Generate a hypothetical answer, embed it, search for similar real documents
- **Multi-query**: Generate multiple query variations, retrieve for each, merge results
- **Reranking**: Use a cross-encoder to rerank retrieved chunks (Cohere Rerank, BGE-Reranker)
- **Contextual compression**: Summarize/filter retrieved chunks to fit context window
- **Parent document retriever**: Embed small chunks, but retrieve the parent document for context
- **RAPTOR**: Recursively summarize clusters of chunks, creating a tree of summaries

---

### Q9: What are LLM Agents and how do they work?

**Answer:**

LLM Agents are systems where an LLM acts as a reasoning engine that can plan, use tools, and take actions to accomplish goals.

**Core components:**
1. **Planning**: Decompose complex tasks into sub-tasks
2. **Memory**: Short-term (conversation context) and long-term (vector store, database)
3. **Tool Use**: Call APIs, execute code, search the web
4. **Reflection**: Evaluate progress and adjust plans

**Agent architectures:**

**ReAct Pattern:**
```
Thought: I need to find the current stock price of Apple.
Action: search_web("Apple stock price today")
Observation: AAPL is trading at $195.23 as of market close.
Thought: Now I have the price. Let me calculate the market cap.
Action: calculator("195.23 * 15.4e9")
Observation: 3,006,542,000,000
Thought: Apple's market cap is approximately $3.01 trillion.
Answer: Apple's current stock price is $195.23 with a market cap of ~$3.01 trillion.
```

**Function Calling (Tool Use):**
```python
# OpenAI function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)
```

**Multi-Agent Systems:**
- Multiple specialized LLM agents collaborate
- Examples: CrewAI, AutoGen, LangGraph
- Roles: Researcher, Writer, Reviewer, Coder, etc.

---

### Q10: Explain the concept of model distillation and how it applies to LLMs.

**Answer:**

**Knowledge Distillation** transfers knowledge from a large "teacher" model to a smaller "student" model, producing a compact model that retains much of the teacher's capability.

**Classic distillation loss (Hinton et al., 2015):**
```
L = alpha * L_CE(student_logits, hard_labels) +
    (1-alpha) * T^2 * KL(softmax(student_logits/T) || softmax(teacher_logits/T))
```

Where T is the temperature (higher T produces softer probability distributions that carry more information about the teacher's learned similarities).

**LLM-specific distillation approaches:**

1. **Logit Distillation**: Student matches teacher's output distribution
   - Requires white-box access to teacher logits
   - Most faithful transfer but expensive

2. **Data Distillation**: Teacher generates training data for the student
   - Alpaca: GPT-3.5 generates 52K instruction-response pairs, used to train Llama 7B
   - Vicuna: Train on ShareGPT conversations (GPT-4 outputs)
   - Orca: Train on detailed reasoning traces from GPT-4
   - Phi: Use GPT-4 to generate "textbook quality" training data

3. **Step-by-step Distillation**: Teacher provides reasoning chains, student learns both the answer and the reasoning process

```python
# Data distillation example
def distill_from_teacher(teacher_model, prompts, student_model):
    # Step 1: Generate training data from teacher
    training_data = []
    for prompt in prompts:
        response = teacher_model.generate(prompt, temperature=0.7)
        training_data.append({"input": prompt, "output": response})

    # Step 2: Fine-tune student on teacher-generated data
    student_model.fine_tune(training_data)
```

**Notable distilled models:**
- Alpaca (7B, distilled from GPT-3.5)
- Vicuna (13B, distilled from GPT-4 via ShareGPT)
- Phi-1/2/3 (1.3B-14B, distilled with synthetic data)
- Orca 2 (7B/13B, progressive distillation from GPT-4)
- Gemma (2B/7B, distilled from Gemini)
- OpenELM (Apple, distilled for on-device)

---

## Category C: Architecture Deep-Dive Questions

### Q11: What is the Feed-Forward Network in a Transformer and how has it evolved?

**Answer:**

The FFN processes each position independently (unlike attention which mixes positions). It typically constitutes ~2/3 of the model's parameters.

**Original FFN:**
```
FFN(x) = ReLU(xW1 + b1)W2 + b2
```
- W1: (d_model, d_ff) where d_ff = 4 * d_model typically
- W2: (d_ff, d_model)

**Modern FFN variants:**

**SwiGLU (used in Llama, Mistral, most modern LLMs):**
```
FFN_SwiGLU(x) = (Swish(xW_gate) * xW_up) W_down

Where Swish(x) = x * sigmoid(x)
```
- Three weight matrices instead of two
- d_ff is typically 8/3 * d_model (to keep parameter count similar)
- Empirically outperforms ReLU FFN

**GeGLU:**
```
FFN_GeGLU(x) = (GELU(xW_gate) * xW_up) W_down
```

**Why GLU variants work better:** The gating mechanism allows the network to more precisely control information flow, effectively implementing a form of multiplicative interaction.

### Q12: What are the key architectural differences between GPT, Llama, and Mistral?

**Answer:**

| Feature | GPT-3 | Llama 2 | Llama 3 | Mistral 7B |
|---------|-------|---------|---------|------------|
| LayerNorm | Post-LN | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| Positional Enc | Learned | RoPE | RoPE | RoPE |
| Attention | MHA | GQA (70B) / MHA (7/13B) | GQA (all sizes) | GQA |
| FFN | ReLU | SwiGLU | SwiGLU | SwiGLU |
| Context Length | 2048 | 4096 | 8192 (128K extended) | 8192 (32K sliding window) |
| Sliding Window | No | No | No | Yes (4096) |
| Vocab Size | 50,257 | 32,000 | 128,256 | 32,000 |
| Bias terms | Yes | No | No | No |

**RMSNorm vs LayerNorm:**
```python
# LayerNorm
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta

# RMSNorm (simpler, no mean subtraction, no beta)
def rms_norm(x, gamma, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return gamma * x / rms
```
RMSNorm is ~10-15% faster than LayerNorm with negligible quality difference.

---

---

---

<a id="section-7"></a>
# SECTION 7: LATEST TRENDS IN LLMs (2025-2026)

## 7.1 Mixture of Experts (MoE)

### Q: Explain Mixture of Experts architecture. How does it differ from dense models?

**Answer:**

**Mixture of Experts (MoE)** replaces the single FFN block in each transformer layer with multiple parallel "expert" FFN blocks, and a routing mechanism (gate) that selects which experts process each token.

**Architecture:**
```
Input token embedding
    |
    v
[Self-Attention]
    |
    v
[Router / Gate Network]
    |  Computes routing scores for each expert
    |  Selects top-k experts (typically k=1 or k=2)
    v
[Expert 1] [Expert 2] [Expert 3] ... [Expert N]
    |          |                         |
    +---------+----------  ... ---------+
    |
    v (weighted sum of selected expert outputs)
[Output]
```

**Key concepts:**

1. **Router/Gate**: A small network (often a single linear layer + softmax) that produces probabilities for each expert
```python
class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        return top_k_probs, top_k_indices
```

2. **Sparse activation**: Only top-k experts are activated per token. Total params are large, but active params per token are small.

3. **Load balancing loss**: Ensures all experts are used roughly equally (prevents expert collapse)
```
L_balance = alpha * N * sum_i(f_i * P_i)
Where:
  f_i = fraction of tokens routed to expert i
  P_i = average routing probability for expert i
```

**Key MoE Models:**

| Model | Total Params | Active Params | Experts | Top-k | Release |
|-------|-------------|---------------|---------|-------|---------|
| Switch Transformer | 1.6T | ~same as T5-Base | 128 | 1 | 2022 |
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 | Dec 2023 |
| Mixtral 8x22B | 176B | 39B | 8 | 2 | Apr 2024 |
| DBRX | 132B | 36B | 16 | 4 | Mar 2024 |
| Grok-1 | 314B | ~86B | 8 | 2 | Mar 2024 |
| DeepSeek V2 | 236B | 21B | 160 | 6 | May 2024 |
| DeepSeek V3 | 671B | 37B | 256 | 8 | Dec 2024 |
| Llama 4 Scout | 109B (17B active) | 16 | 1 | Apr 2025 |
| Llama 4 Maverick | 400B (17B active) | 128 | 1 | Apr 2025 |
| Llama 4 Behemoth | ~2T (288B active) | 128+ | TBD | 2025 (training) |
| Qwen 2.5-MoE | 57.4B (14B active) | 64 | 8 | 2025 |

**Llama 4 -- Meta's MoE Pivot (April 2025):**
Meta made a major architectural shift to MoE with Llama 4:
- **Scout (109B total, 17B active)**: 16 experts, 10M token context window (longest open model). Uses interleaved attention-MoE with iRoPE (interleaved RoPE) for extreme context extension.
- **Maverick (400B total, 17B active)**: 128 experts with only 17B active params. Competitive with GPT-4o and Claude 3.5 Sonnet on many benchmarks while being far cheaper to serve.
- **Behemoth (~2T total, 288B active)**: The largest open model ever attempted. In training at release; targeting frontier-level performance.
- All models use native multimodal "early fusion" architecture trained on text, images, and video from the start.
- Trained with a massive synthetic data pipeline.

**DeepSeek's innovations in MoE:**
- **DeepSeekMoE**: Fine-grained experts (more, smaller experts) + shared experts
- Shared experts: Some experts are always activated (handle common patterns)
- Routed experts: Specialized experts activated by the router
- This outperforms standard MoE with the same compute budget

**Advantages of MoE:**
- More parameters (knowledge capacity) for the same compute budget
- Sub-linear scaling: 8x more params does not mean 8x more compute
- Specialization: different experts can specialize for different types of input

**Disadvantages of MoE:**
- Higher memory footprint (all expert weights must be in memory)
- Communication overhead in distributed training
- Load balancing is tricky
- Expert collapse risk
- More complex to serve (need all experts loaded)

---

## 7.2 Long-Context Models

### Q: What are the latest advances in long-context LLMs?

**Answer:**

Long-context capability has been one of the major battlegrounds in 2024-2025:

**Context Length Timeline:**
```
2020: GPT-3              2,048 tokens (~1.5K words)
2022: GPT-3.5            4,096 tokens
2023: GPT-4              32K/128K tokens
2023: Claude 2           100K tokens
2024: Claude 3           200K tokens (~150K words, ~500 pages)
2024: Gemini 1.5 Pro     1M tokens (2M in preview)
2024: Llama 3.1          128K tokens
2025: Gemini 2.0 Flash   1M tokens
2025: Gemini 2.5 Pro     1M tokens
2025: Claude 3.7 Sonnet  200K tokens (with extended thinking)
2025: Llama 4 Maverick   1M tokens
2025: Llama 4 Scout      10M tokens (!!) -- longest context of any model
2025: GPT-5/5.2          400K tokens
2025: Gemini 3 Pro        1M tokens (at frontier quality)
2026: Claude Opus 4.6    200K tokens
```

**Llama 4 Scout's 10M Token Context (April 2025):**
Meta achieved an extraordinary 10 million token context window using:
- **iRoPE (interleaved RoPE)**: A novel positional encoding scheme that interleaves layers with and without positional encodings, enabling extreme length generalization
- **Specialized training curriculum**: Progressive context extension during training
- This is ~50x longer than GPT-4's context and ~5x longer than Gemini 2.0's

**Key techniques enabling long context:**

1. **RoPE Extensions:**
   - **Position Interpolation (PI)**: Scale positions to fit more in the trained range
   - **NTK-aware scaling**: Modify RoPE's base frequency
   - **YaRN (Yet another RoPE extensioN)**: Combines NTK scaling with attention scaling
   - **Dynamic NTK**: Adjust scaling factor based on sequence length at inference time

2. **Ring Attention (Liu et al., 2023):**
   - Distributes sequence across multiple devices in a ring topology
   - Each device computes attention for its chunk while passing KV blocks around the ring
   - Enables virtually unlimited context by adding more devices

3. **Infini-Attention (Google, 2024):**
   - Combines local attention with compressive memory
   - Maintains a compressed representation of the full past context
   - Bounded memory with unbounded context

**Evaluation: "Needle in a Haystack" test:**
```
Insert a specific fact ("The best thing to do in San Francisco is eat
a sandwich and sit in Dolores Park") at various positions within a
long document, and test if the model can retrieve it.

Results (128K context):
- GPT-4-128K: ~97% accuracy across all positions
- Claude 3: ~98% accuracy
- Llama 3.1 128K: ~95% accuracy
- Gemini 1.5 Pro (1M): ~99.7% across 1M tokens
```

**Practical considerations:**
- Long context != good long context utilization
- "Lost in the middle" problem persists even in long-context models
- Cost scales linearly (or worse) with context length
- RAG is often more cost-effective than stuffing everything in context
- Best practice: combine long context WITH RAG for best results

---

## 7.3 Multimodal LLMs

### Q: How do multimodal LLMs process both text and images?

**Answer:**

Multimodal LLMs can process and reason across multiple modalities: text, images, audio, video.

**Architecture patterns:**

**Pattern 1: Vision Encoder + Projection + LLM (LLaVA-style)**
```
Image -> [Vision Encoder (ViT/CLIP)] -> Visual tokens
                                            |
                                      [Projection Layer (MLP)]
                                            |
                                            v
Text tokens -> [Tokenizer] ---------> [LLM Decoder] -> Output
                                      (both visual and text tokens)
```

**Pattern 2: Early Fusion (Gemini, GPT-4V/o)**
```
Image patches -> [Patch Embedding] --\
                                      --> [Unified Transformer] --> Output
Text tokens -> [Token Embedding] ----/
```
- All modalities share the same transformer
- Most flexible but requires training from scratch or extensive continued pre-training

**Pattern 3: Cross-Attention (Flamingo, IDEFICS)**
```
Image -> [Vision Encoder] -> Visual features
                                |
                          [Cross-Attention layers]
                                |
Text -> [LLM Decoder] ---------+---------> Output
         (with interleaved cross-attention)
```

**Major multimodal models (2024-2025):**

| Model | Modalities | Architecture | Key Capability |
|-------|-----------|--------------|----------------|
| GPT-4V/4o | Text, Image, Audio | Early fusion | General multimodal |
| Claude 3/3.5 | Text, Image | Vision encoder + LLM | Strong image understanding |
| Gemini 1.5/2.0 | Text, Image, Audio, Video | Native multimodal | Long video understanding |
| Llama 3.2 Vision | Text, Image | Cross-attention adapter | Open-source multimodal |
| LLaVA-1.6 | Text, Image | CLIP + MLP + Mistral | Open-source research |
| Qwen-VL 2.5 | Text, Image, Video | Dynamic resolution | Competitive open source |
| Pixtral (Mistral) | Text, Image | Native multimodal | European AI |

**Image tokenization approaches:**
1. **Patch-based**: Divide image into fixed-size patches (16x16 or 14x14), each becomes a token
2. **Dynamic resolution**: Divide image into tiles based on aspect ratio, process each tile
3. **Perceiver/Q-Former**: Use a fixed number of learned queries to compress visual information

```python
# Typical vision processing (LLaVA-style)
from transformers import CLIPVisionModel

vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

# Image -> patches -> visual tokens
image_features = vision_encoder(pixel_values)  # (batch, 576, 1024)

# Project to LLM embedding dimension
projection = nn.Linear(1024, 4096)  # CLIP dim -> LLM dim
visual_tokens = projection(image_features)  # (batch, 576, 4096)

# Concatenate with text tokens and feed to LLM
combined = torch.cat([visual_tokens, text_embeddings], dim=1)
output = llm(inputs_embeds=combined)
```

---

## 7.4 Reasoning Models (o1, o3, DeepSeek R1)

### Q: What are "reasoning models" and how do they differ from standard LLMs?

**Answer:**

Reasoning models are a new class of LLMs specifically trained to perform extended step-by-step reasoning before providing an answer. They represent a paradigm shift from "System 1" (fast, intuitive) to "System 2" (slow, deliberate) thinking.

**Key reasoning models:**

**OpenAI o1 (September 2024):**
- Uses internal "chain of thought" that is not shown to the user
- Spends more "thinking time" on harder problems
- Dramatically better at math, coding, and scientific reasoning
- Trades off latency for accuracy
- o1-preview and o1-mini variants

**OpenAI o3 (January 2025 release):**
- Successor to o1 with dramatically improved reasoning
- Scored 87.5% on ARC-AGI benchmark (vs o1's 32%)
- Introduces "compute dial" -- adjustable reasoning effort (low/medium/high)
- 200K context window
- Significantly better at math, science, and coding than o1

**OpenAI o4-mini (April 2025):**
- Successor to o3-mini with improved efficiency
- Strong reasoning at a lower cost tier
- Tool use support (function calling, web search, code execution during reasoning)
- 200K context window
- Represents OpenAI's push toward affordable reasoning models

**Claude 3.7 Sonnet (February 2025):**
- Anthropic's first hybrid reasoning model
- Supports "extended thinking" mode with visible chain-of-thought
- User-controllable thinking budget (like o3's compute dial)
- Can toggle between fast (standard) and deep (extended thinking) modes
- 200K context window maintained
- Competitive with o3 on math and coding benchmarks

**Claude Opus 4 and Claude Sonnet 4 (May 2025):**
- Anthropic's latest flagship models
- Claude Opus 4: Best-in-class for complex reasoning, coding, and agentic tasks
- Claude Sonnet 4: Strong balance of speed and intelligence, successor to 3.7 Sonnet
- Both support extended thinking with improved reasoning chains
- Opus 4 demonstrated state-of-the-art performance on SWE-bench (real-world software engineering)
- Both models excel at agentic, multi-step tool use workflows

**Google Gemini 2.5 Pro (March 2025):**
- Google's reasoning model entry
- Native "thinking" mode built into Gemini 2.5 architecture
- 1M token context window with reasoning
- Strong at code, math, and multimodal reasoning
- Competitive with o3 and Claude 3.7 on major benchmarks

**GPT-4.5 (February 2025):**
- OpenAI's largest dense (non-reasoning) model
- Focused on improved "EQ" -- emotional intelligence, nuance, creative writing
- Better at reducing hallucinations through scale
- Very expensive ($75/$150 per 1M input/output tokens)
- 128K context window

**DeepSeek R1 (January 2025):**
- Open-source reasoning model
- Key innovation: Trained primarily with RL (GRPO), minimal SFT
- Showed that reasoning can EMERGE from RL without explicit CoT training
- Shows explicit "thinking" tokens visible to the user
- Competitive with o1 on many benchmarks
- Available in multiple sizes (1.5B, 7B, 8B, 14B, 32B, 70B, 671B)

**How reasoning models work:**

```
Standard LLM:
User: What is 27 * 43?
Assistant: 1161

Reasoning Model:
User: What is 27 * 43?
[Internal thinking - hidden or shown]:
<think>
Let me multiply 27 by 43.
27 * 43 = 27 * 40 + 27 * 3
27 * 40 = 1080
27 * 3 = 81
1080 + 81 = 1161
</think>
Assistant: 1161
```

**Training approach (DeepSeek R1 revealed the process):**

1. **Cold Start**: Small amount of long-CoT SFT data to teach format
2. **RL Training (GRPO)**: Reward correct final answers; the model naturally develops reasoning strategies
3. **Rejection Sampling**: Generate many solutions, keep correct ones, use as SFT data
4. **Final SFT + RL**: Polish the model with curated data

**Remarkable finding from DeepSeek R1:**
During RL training without explicit CoT examples, the model spontaneously developed:
- Step-by-step reasoning
- Self-verification ("Let me check: ...")
- Backtracking ("Wait, that's wrong. Let me reconsider...")
- Breaking problems into sub-problems

**Test-Time Compute Scaling:**
Reasoning models introduce a new scaling axis: instead of only scaling training compute, you can also scale INFERENCE compute.

```
Traditional scaling: Better model = more training compute
Reasoning scaling:   Better answer = more inference compute (more thinking time)

This means a smaller reasoning model with more thinking time can outperform
a larger standard model:
  o1-mini (small) with extended thinking > GPT-4 (large) for math
```

**Benchmark comparison (reasoning models, as of mid-2025):**
| Benchmark | GPT-4o | o1 | o3 | o4-mini | DeepSeek R1 | Claude 3.7 (thinking) | Gemini 2.5 Pro |
|-----------|--------|-----|-----|---------|-------------|----------------------|----------------|
| MATH | 76.6% | 96.4% | 96.7% | 93.4% | 97.3% | ~95% | ~95% |
| AIME 2024 | 9/30 | 13/30 | 18/30 | 14/30 | 13/30 | ~15/30 | ~16/30 |
| ARC-AGI | ~5% | 32% | 87.5% | ~70% | ~15% | N/A | N/A |
| Codeforces | 11th %ile | 89th %ile | 99th+ %ile | 93rd %ile | 96.3rd %ile | ~90th %ile | ~88th %ile |
| GPQA Diamond | 53.6% | 78.0% | 83.3% | 74.1% | 71.5% | ~78% | ~80% |
| SWE-bench | ~33% | ~41% | ~69% | ~66% | ~49% | ~62% | ~64% |

**Note:** SWE-bench (verified) has emerged as a key benchmark for agentic coding ability in 2025. Claude Opus 4 achieved state-of-the-art results on SWE-bench verified at launch.

---

## 7.5 Small Language Models (SLMs) and On-Device LLMs

### Q: What is the trend toward smaller, more efficient language models?

**Answer:**

There is a strong counter-trend to the "bigger is better" narrative: small language models (1B-7B parameters) that can run on phones, laptops, and edge devices.

**Why small models matter:**
1. **Privacy**: Data never leaves the device
2. **Latency**: No network round trip
3. **Cost**: No API fees, no GPU servers
4. **Availability**: Works offline
5. **Customization**: Easier to fine-tune for specific domains

**Notable small models (2024-2025):**

| Model | Size | Developer | Key Feature |
|-------|------|-----------|-------------|
| Phi-3 Mini | 3.8B | Microsoft | Textbook-quality training data |
| Phi-3.5 Mini | 3.8B | Microsoft | Improved, multimodal |
| Gemma 2 | 2B, 9B | Google | Distilled from Gemini |
| Llama 3.2 | 1B, 3B | Meta | Optimized for on-device |
| Mistral 7B | 7.3B | Mistral | Best open 7B at release |
| Qwen 2.5 | 0.5B-72B | Alibaba | Full size range |
| SmolLM2 | 135M-1.7B | HuggingFace | Tiny but capable |
| Apple OpenELM | 270M-3B | Apple | On-device optimized |
| DeepSeek R1 Distill | 1.5B-70B | DeepSeek | Reasoning at small scale |
| Gemma 3 | 1B-27B | Google | Mar 2025, multimodal, strong quality |
| Phi-4 | 14B | Microsoft | Jan 2025, best-in-class at 14B |
| Mistral Small 3 | 24B | Mistral | Jan 2025, Apache 2.0 license |
| Qwen 2.5 Coder | 0.5B-32B | Alibaba | Code-specialized, competes with larger models |

**On-Device Inference Frameworks:**

| Framework | Platform | Key Feature |
|-----------|----------|-------------|
| llama.cpp | CPU/GPU (any) | C/C++, highly optimized, GGUF format |
| MLX | Apple Silicon | Apple's ML framework for M-series chips |
| MLC LLM | Mobile/Desktop | Compilation-based, Vulkan/Metal/CUDA |
| ONNX Runtime | Cross-platform | Microsoft, broad hardware support |
| MediaPipe LLM | Android/iOS | Google, mobile-optimized |
| ExecuTorch | Mobile | Meta's mobile inference, powers Llama on-device |
| CoreML | Apple | Native Apple ecosystem |

```python
# Running a model locally with llama.cpp (Python bindings)
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama-3.2-1b-instruct-q4_k_m.gguf",
    n_ctx=2048,        # context length
    n_threads=4,       # CPU threads
    n_gpu_layers=0,    # 0 = CPU only
)

output = llm(
    "What is machine learning?",
    max_tokens=256,
    temperature=0.7,
)
```

**Techniques for making small models effective:**
1. **Distillation**: Train on outputs from larger models
2. **Pruning**: Remove less important weights/heads/layers
3. **Quantization**: Run in INT4/INT8 precision
4. **Architecture search**: Optimize architecture for the target hardware
5. **Data quality over quantity**: Phi-3 showed that carefully curated training data matters more than scale for small models

---

## 7.6 Other Important 2025 Trends

### Structured Outputs / JSON Mode
Models can now reliably output valid JSON, function calls, and structured data:
```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_schema", "json_schema": schema},
    messages=[...]
)
```

### Agentic AI
- LLMs as autonomous agents that can plan, execute, and iterate
- Computer use capabilities (Claude Computer Use, OpenAI Operator)
- Multi-step task completion
- Coding agents (Devin, Claude Code, GitHub Copilot Workspace, Cursor, Windsurf)
- Multi-agent orchestration (AutoGen, LangGraph, CrewAI) -- see Section 9
- Model Context Protocol (MCP) by Anthropic for standardized tool integration -- see Section 10

> **YOUR EXPERIENCE**: Based on your work at RavianAI, you have direct experience building agentic AI systems using AutoGen and LangGraph for multi-step orchestration, tool use, and autonomous workflows. At MathCo, you built multi-agent analytics pipelines. Be prepared to discuss specific architectural decisions and challenges.

### Synthetic Data for Training
- Using LLMs to generate training data for other models
- Validated by: Phi series, Orca, Alpaca, UltraChat
- Self-improvement loops: model generates data, trains on it, generates better data

### Test-Time Compute
- Spending more compute at inference for better answers
- Reasoning models (o1, o3, R1)
- Search-based approaches (Tree of Thought, MCTS for LLMs)
- Emerging as an alternative scaling dimension to training compute

### Retrieval-Augmented Everything
- RAG has become standard practice, not optional
- GraphRAG: Combining knowledge graphs with RAG
- Agentic RAG: Agents that decide when and what to retrieve
- Corrective RAG: Verify retrieval quality before generation

> **YOUR EXPERIENCE**: Based on your work at MathCo, you built production RAG systems for analytics use cases. At RavianAI, you implemented agentic RAG patterns where LLM agents decide when to retrieve vs. reason from context. Discuss your experience with chunking strategies, reranking, and hybrid search.

---

## 7.7 Alternative Architectures: State Space Models and Beyond

### Q: What are State Space Models (SSMs) and how do they compare to Transformers?

**Answer:**

State Space Models (SSMs) are an emerging class of sequence models that challenge the Transformer's dominance by offering **linear** (O(n)) complexity instead of quadratic (O(n^2)) attention.

**Key SSM-based architectures:**

**Mamba (Gu & Dao, December 2023):**
- Based on Structured State Space Models (S4) with a key innovation: **selective state spaces**
- The selection mechanism allows the model to filter information based on input content (similar to attention's content-based routing)
- Linear time complexity: O(n) for sequence length n
- No KV cache needed -- constant memory per token during inference
- Hardware-aware implementation with efficient CUDA kernels

```
Transformer attention: O(n^2) time, O(n) memory (with Flash Attention)
Mamba:                 O(n) time, O(1) per-step memory

For sequence length 1M:
  Transformer: ~1,000,000^2 = 1 trillion operations
  Mamba:       ~1,000,000 operations (1 million x faster theoretically)
```

**Core SSM equation:**
```
h(t) = A * h(t-1) + B * x(t)    (hidden state update)
y(t) = C * h(t) + D * x(t)       (output)

Where:
  h(t) = hidden state at time t
  x(t) = input at time t
  A, B, C, D = learned parameters (in Mamba, B and C are input-dependent)
```

**Mamba-2 (Dao & Gu, May 2024):**
- Shows theoretical connection between SSMs and attention (Structured State Space Duality)
- 2-8x faster than Mamba-1
- Better scaling properties
- Can be viewed as a special form of linear attention

**Jamba (AI21 Labs, March 2024):**
- First production hybrid Transformer-Mamba model
- Interleaves Mamba layers with Transformer attention layers and MoE
- 256K context window
- 52B total params (12B active) with MoE
- Shows that hybrid architectures can get the best of both worlds

**RWKV (v5/v6, 2024-2025):**
- "Receptance Weighted Key Value" -- RNN-like architecture with Transformer-level performance
- O(n) training, O(1) inference per token (true constant memory)
- Can scale to very long contexts efficiently
- RWKV-6 (2024) matches Transformer quality at comparable scales
- Fully open source with active community

**Comparison:**
| Feature | Transformer | Mamba/SSM | RWKV | Hybrid (Jamba) |
|---------|------------|-----------|------|---------------|
| Training complexity | O(n^2) | O(n) | O(n) | O(n) to O(n^2) |
| Inference per token | O(n) with KV cache | O(1) | O(1) | Mixed |
| Long context | Expensive | Efficient | Efficient | Balanced |
| Quality at scale | Best | Near-Transformer | Near-Transformer | Competitive |
| Ecosystem/tooling | Mature | Growing | Niche | New |
| Key limitation | Memory for long seq | Recall on long contexts | Less proven at frontier | Complexity |

**2025 Status:**
- Transformers still dominate at frontier scale (GPT-4, Claude, Gemini all use Transformers)
- SSMs are gaining traction for efficiency-critical applications (on-device, long context)
- Hybrid models (mixing attention + SSM layers) appear most promising
- NVIDIA's research suggests hybrid architectures will likely be the future
- No pure SSM model has yet matched frontier Transformers on all benchmarks

**Interview insight:** SSMs are a hot topic. Know the O(n) vs O(n^2) tradeoff, understand why attention's recall advantage matters, and mention hybrid architectures as the likely future.

---

## 7.8 Extended Thinking / Reasoning-as-a-Feature (2025)

### Q: How has "thinking" become a standard feature across LLM providers?

**Answer:**

In 2025, extended thinking / reasoning became a mainstream feature rather than a niche capability:

**The convergence:**
| Provider | Model | Thinking Feature | Released |
|----------|-------|-----------------|----------|
| OpenAI | o1, o3, o4-mini | Hidden chain-of-thought, compute dial | 2024-2025 |
| Anthropic | Claude 3.7 Sonnet, Opus 4, Sonnet 4 | Extended thinking (visible, budget-controlled) | Feb-May 2025 |
| Google | Gemini 2.5 Pro | Built-in thinking mode | Mar 2025 |
| DeepSeek | R1 | Visible `<think>` tokens | Jan 2025 |

**Key design decisions across providers:**
1. **Visible vs Hidden thinking**: DeepSeek R1 and Claude show thinking; OpenAI o-series hides it
2. **Budget control**: Claude and o3 let users control how much thinking to do
3. **Streaming thinking**: Claude streams thinking tokens in real-time
4. **Cost model**: Thinking tokens are billed (input pricing) -- important for cost optimization

**Extended thinking pattern:**
```python
# Anthropic Claude 3.7 Sonnet with extended thinking
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # max tokens for thinking
    },
    messages=[{"role": "user", "content": "Solve this complex math problem..."}]
)

# Response contains both thinking and answer
for block in response.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking}")
    elif block.type == "text":
        print(f"Answer: {block.text}")
```

---

---

---

<a id="section-8"></a>
# SECTION 8: COMPARISON OF MAJOR LLM PROVIDERS

## 8.1 Provider Overview

### OpenAI

**Models:** GPT-4o, GPT-4o mini, GPT-4.5, GPT-5, GPT-5.2, o1, o3, o3-mini, o4-mini

**Key facts:**
- Pioneer of the GPT series and the modern LLM era
- First to achieve product-market fit with ChatGPT (Nov 2022)
- Architecture: Believed to be MoE for GPT-4 (unconfirmed, ~8 experts, ~220B active out of ~1.8T total -- per leaked reports)
- Strongest brand recognition and largest user base
- Most mature API ecosystem with function calling, JSON mode, assistants API, file search, Responses API
- Released GPT-4.5 (Feb 2025) as their largest dense model
- Reasoning model lineup expanded: o3 (Jan 2025), o4-mini (Apr 2025)
- **GPT-5 (mid-2025)**: Major leap with 400K context window, improved reasoning
- **GPT-5.2 (late 2025)**: Latest flagship -- 100% on AIME 2025 math, ~54% on ARC-AGI-2, 74.9% SWE-bench

**Model lineup (2025-2026):**

| Model | Context | Key Strength | Pricing (input/output per 1M tokens) |
|-------|---------|-------------|--------------------------------------|
| GPT-4o | 128K | Strong all-around, multimodal | $2.50 / $10.00 |
| GPT-4o mini | 128K | Fast, cheap, good quality | $0.15 / $0.60 |
| GPT-4.5 | 128K | Largest dense model, best EQ/creativity | $75.00 / $150.00 |
| **GPT-5.2** | **400K** | **Best reasoning, math, fastest** | **~$10-15 / $40-60** |
| o3 | 200K | Strongest reasoning | $10.00 / $40.00 |
| o3-mini | 200K | Efficient reasoning | $1.10 / $4.40 |
| o4-mini | 200K | Latest efficient reasoning + tools | $1.10 / $4.40 |

**Strengths:**
- Most polished API and developer experience
- Function calling and structured outputs
- Fine-tuning API for GPT-4o mini and GPT-4o
- Broadest ecosystem (plugins, GPTs, assistants, Operator for computer use)
- Vision, audio, image generation (DALL-E 3, GPT-4o native image gen), embeddings, moderation all in one platform
- Reasoning models with tool use (o4-mini can call tools mid-reasoning)
- New Responses API (2025) -- streamlined replacement for Chat Completions
- GPT-5.2 dominates in math/reasoning benchmarks (100% AIME 2025)

**Weaknesses:**
- Closed source / proprietary
- Less transparent about architecture and training data
- Premium models extremely expensive
- Rate limits can be restrictive

---

### Anthropic (Claude)

**Models:** Claude 3.5 Haiku, Claude 3.5 Sonnet, Claude 3.7 Sonnet, Claude Opus 4, Claude Sonnet 4, Claude Haiku 4.5, Claude Sonnet 4.5, Claude Opus 4.5/4.6

**Key facts:**
- Founded by former OpenAI research leaders (Dario and Daniela Amodei)
- Strong focus on AI safety and alignment
- Developed Constitutional AI approach
- Known for longer, more detailed responses and strong instruction following
- 200K context window across all models
- Pioneered Model Context Protocol (MCP) as an open standard (Nov 2024)
- Claude 3.7 Sonnet (Feb 2025): First hybrid model with extended thinking
- Claude Opus 4 & Sonnet 4 (May 2025): Flagships with state-of-the-art agentic performance
- **Claude Opus 4.5 (late 2025)**: Best-in-class coding (80.9% SWE-bench Verified), strongest agentic model
- **Claude Opus 4.6 (early 2026)**: Latest flagship with improved capabilities

**Model lineup (2025-2026):**

| Model | Context | Key Strength | Pricing (input/output per 1M tokens) |
|-------|---------|-------------|--------------------------------------|
| Claude 3.5 Haiku | 200K | Fast, efficient, good quality | $0.80 / $4.00 |
| Claude Sonnet 4 | 200K | Improved Sonnet, strong all-around | $3.00 / $15.00 |
| Claude Sonnet 4.5 | 200K | Latest mid-tier, excellent value | $3.00 / $15.00 |
| Claude Opus 4 | 200K | Best agentic, complex tasks | $15.00 / $75.00 |
| **Claude Opus 4.5** | **200K** | **SWE-bench SOTA (80.9%), best coding** | **$15.00 / $75.00** |
| **Claude Opus 4.6** | **200K** | **Latest flagship, improved overall** | **$15.00 / $75.00** |

**Strengths:**
- Consistently best for coding tasks (SWE-bench SOTA: Opus 4.5 at 80.9%, surpassing GPT-5.2 at 74.9%)
- Excellent instruction following and long-form content
- 200K context window standard across all models
- Strong safety properties (Constitutional AI)
- Computer use capability (Claude can control a desktop)
- Extended thinking (controllable reasoning budget)
- Artifacts feature for rich content generation
- Very strong at document analysis and summarization
- Claude Code -- dedicated CLI for agentic coding
- MCP ecosystem for standardized tool integration
- Best-in-class for agentic, multi-step workflows

**Weaknesses:**
- Closed source
- Ecosystem growing but still smaller than OpenAI
- Sometimes overly cautious (safety vs helpfulness tradeoff)
- Limited fine-tuning options (enterprise only)
- No native image generation
- Opus pricing is steep for high-volume use
- 200K context window smaller than GPT-5.2 (400K) and Gemini 3 Pro (1M)

---

### Google (Gemini)

**Models:** Gemini 1.5 (Flash, Pro), Gemini 2.0 (Flash, Flash-Lite), Gemini 2.5 (Pro, Flash), Gemini 3 Pro

**Key facts:**
- Natively multimodal (trained on text, images, audio, video from the start)
- Industry-leading context windows (1M-2M tokens)
- Integrated into Google ecosystem (Search, Android, Workspace)
- Uses MoE architecture
- Powers Google AI Studio and Vertex AI
- Gemini 2.5 Pro (Mar 2025): Google's first reasoning model with built-in thinking
- Gemini 2.5 Flash (May 2025): Fast reasoning model, excellent price-performance
- **Gemini 3 Pro (late 2025)**: Major leap -- 1M context, 76.8% SWE-bench, ~45% ARC-AGI-2, massive multimodal improvements

**Model lineup (2025-2026):**

| Model | Context | Key Strength | Pricing (input/output per 1M tokens) |
|-------|---------|-------------|--------------------------------------|
| Gemini 2.5 Flash | 1M | Fast reasoning, best price-performance | $0.15 / $0.60 |
| Gemini 2.0 Flash | 1M | Fast, multimodal, long context | $0.10 / $0.40 |
| Gemini 2.5 Pro | 1M | Reasoning + long context, top benchmarks | $1.25 / $10.00 |
| **Gemini 3 Pro** | **1M** | **Best multimodal, largest context at frontier** | **~$3-10 / $10-15** |

**Strengths:**
- Largest context windows in the industry (1M tokens at frontier level)
- Native multimodal (text + image + audio + video in one model)
- Excellent at video understanding (can process hours of video)
- Very competitive pricing, especially Flash models
- Free tier generous for developers (even for 2.5 Pro in AI Studio)
- Strong integration with Google Cloud, Search, and Android
- Gemini 3 Pro competitive with GPT-5.2 and Claude Opus 4.5 across benchmarks
- Gemini 2.5 Flash: reasoning-capable at Flash pricing

**Weaknesses:**
- Developer API experience improving but still behind OpenAI in maturity
- Enterprise adoption lags behind OpenAI/Anthropic
- Less popular in open-source/developer community compared to Claude/OpenAI
- Agentic capabilities less proven than Claude or OpenAI

---

### Meta (Llama)

**Models:** Llama 3.1, Llama 3.2, Llama 3.3, Llama 4 Scout, Llama 4 Maverick, Llama 4 Behemoth (training)

**Key facts:**
- OPEN WEIGHT models (not fully open source -- license restrictions exist)
- Most influential open model family
- Drives the open-source LLM ecosystem
- Llama 4 (April 2025): Major shift to MoE architecture + native multimodal
- Llama 4 Scout: 10M token context window (longest of any model)
- Community fine-tunes dominate: Vicuna, WizardLM, Hermes, etc.

**Model lineup (2025):**

| Model | Architecture | Sizes | Context | Key Feature |
|-------|-------------|-------|---------|-------------|
| Llama 3.1 | Dense | 8B, 70B, 405B | 128K | Strong general-purpose open model |
| Llama 3.2 | Dense | 1B, 3B, 11B, 90B | 128K | Multimodal (11B, 90B), On-device (1B, 3B) |
| Llama 3.3 | Dense | 70B | 128K | Quality of 405B at 70B cost |
| Llama 4 Scout | MoE (16 experts) | 109B (17B active) | 10M | Longest context of any model |
| Llama 4 Maverick | MoE (128 experts) | 400B (17B active) | 1M | Competitive with GPT-4o class |
| Llama 4 Behemoth | MoE (128+ experts) | ~2T (288B active) | TBD | In training, targeting frontier |

**Llama 4 Key Innovations:**
- **MoE architecture**: First Llama generation to use Mixture of Experts
- **Native multimodal**: Early fusion (text + image + video from pre-training)
- **iRoPE**: Interleaved RoPE positional encoding enabling 10M context
- Only 17B active params in Scout/Maverick = very efficient to serve despite large total params

**Strengths:**
- Open weights: full control, no API dependency
- Can fine-tune, deploy anywhere, modify freely
- Massive community and ecosystem
- Llama 4 MoE is extremely efficient (17B active params for GPT-4o-level quality)
- Drives innovation in open-source AI
- No per-token API costs (you pay for compute only)
- 10M token context (Scout) is unmatched

**Weaknesses:**
- Requires GPU infrastructure to serve (all expert weights in memory for MoE)
- License restrictions (no use for >700M MAU apps without permission)
- Not truly open source (no training data, limited training details)
- Llama 4 early benchmarks showed mixed reception (some benchmarks below expectations)
- MoE models harder to fine-tune than dense models

---

### Mistral

**Models:** Mistral 7B, Mixtral 8x7B, Mixtral 8x22B, Mistral Large, Mistral Small, Codestral, Pixtral

**Key facts:**
- French AI company, founded by ex-DeepMind and ex-Meta researchers
- Known for exceptional efficiency and innovation
- Pioneered several architectural innovations (sliding window attention, GQA in small models)
- Strong European AI competitor
- Both open and commercial models

**Model lineup (2024-2025):**

| Model | Size | Architecture | Key Feature |
|-------|------|-------------|-------------|
| Mistral 7B | 7.3B | Dense, SWA | Punched above weight class |
| Mixtral 8x7B | 46.7B (12.9B active) | MoE | First popular open MoE |
| Mixtral 8x22B | 176B (39B active) | MoE | Strong reasoning |
| Mistral Large 2 | ~123B | Dense | Competitive with GPT-4 |
| Mistral Small | ~22B | Dense | Efficient mid-tier |
| Codestral | 22B | Dense | Code-specialized |
| Pixtral Large | 124B | Multimodal | Vision + text |

**Strengths:**
- Excellent efficiency (best performance per parameter)
- Architectural innovations
- Strong open-source contributions
- Competitive pricing on API
- Good multilingual performance (especially European languages)

**Weaknesses:**
- Smaller company, less infrastructure
- Ecosystem not as developed
- Brand recognition lower outside AI community

---

### DeepSeek

**Models:** DeepSeek V2, DeepSeek V3 (0324 update), DeepSeek Coder V2, DeepSeek R1, DeepSeek R1 Distill series

**Key facts:**
- Chinese AI lab that shocked the industry in January 2025 with R1
- Achieved frontier performance at a fraction of the cost
- Open-weight releases with detailed technical reports
- Pioneered innovations in MoE efficiency and RL-based reasoning
- R1 release (Jan 2025) sent shockwaves through the AI industry and markets
- Demonstrated that reasoning can emerge from RL training alone (GRPO)

**Model lineup (2025):**

| Model | Size | Architecture | Key Feature |
|-------|------|-------------|-------------|
| DeepSeek V3 | 671B (37B active) | MoE (256 experts) | GPT-4 level at 1/10th cost |
| DeepSeek V3-0324 | 671B (37B active) | MoE (256 experts) | Updated V3 with improved reasoning |
| DeepSeek R1 | 671B (37B active) | MoE + RL reasoning | Competes with o1, open weights |
| DeepSeek R1 Distill | 1.5B-70B | Dense (distilled) | Reasoning at small scale |
| DeepSeek R1-0528 | 671B (37B active) | MoE + improved RL | Enhanced reasoning, May 2025 update |

**The DeepSeek R1 Moment (January 2025):**
DeepSeek R1 was one of the most significant AI releases of 2025:
- Open-weight 671B MoE reasoning model competitive with OpenAI o1
- Trained primarily with GRPO (RL without a critic model) -- showed reasoning EMERGES from RL
- Training cost estimated at a fraction of comparable Western models
- Triggered a stock market selloff in AI chip companies
- Proved that frontier reasoning does not require massive budgets
- R1 distill models (1.5B-70B) brought reasoning to small/medium scale

**Strengths:**
- Remarkable cost efficiency: V3 trained for ~$5.5M (vs estimated >$100M for GPT-4)
- Open weights and detailed technical reports
- Innovations: DeepSeekMoE, Multi-head Latent Attention (MLA), GRPO
- R1 showed reasoning can emerge from RL alone
- Very competitive on coding and math
- API pricing dramatically lower than competitors

**Weaknesses:**
- Geopolitical concerns (Chinese company, potential data sovereignty issues)
- API availability may be restricted in some regions
- Potential censorship on sensitive topics (political, historical)
- Less established enterprise support
- US export controls may limit future GPU access for training

**DeepSeek's Key Innovations:**

1. **Multi-head Latent Attention (MLA)**: Compresses KV cache using low-rank projections. Reduces KV cache by 93.3% compared to standard MHA.

2. **DeepSeekMoE**: Fine-grained experts (more smaller experts) + shared experts that are always active.

3. **Auxiliary-loss-free load balancing**: Novel approach to balance expert loads without auxiliary losses.

4. **FP8 mixed-precision training**: First to successfully train a 671B model in FP8 precision.

---

## 8.2 Head-to-Head Comparison

### Benchmark Comparison (approximate, as of mid-2025)

**Non-reasoning models:**

| Benchmark | GPT-4o | Claude 3.5 Sonnet | Gemini 2.0 Flash | Llama 4 Maverick | DeepSeek V3 |
|-----------|--------|-------------------|------------------|------------------|-------------|
| MMLU | 88.7% | 88.7% | 87.0% | ~88% | 88.5% |
| HumanEval | 90.2% | 92.0% | 89.1% | ~89% | 89.6% |
| MATH | 76.6% | 78.3% | 73.0% | ~77% | 84.6% |
| GSM8K | 95.8% | 96.4% | 93.7% | ~96% | 96.7% |
| GPQA | 53.6% | 59.4% | 50.2% | ~55% | 59.1% |

**Reasoning models (with thinking enabled):**

| Benchmark | o3 | o4-mini | Claude Opus 4 | Claude 3.7 Sonnet | Gemini 2.5 Pro | DeepSeek R1 |
|-----------|-----|---------|---------------|-------------------|----------------|-------------|
| MATH | 96.7% | 93.4% | ~96% | ~95% | ~95% | 97.3% |
| GPQA Diamond | 83.3% | 74.1% | ~80% | ~78% | ~80% | 71.5% |
| SWE-bench verified | ~69% | ~66% | ~72% (SOTA) | ~62% | ~64% | ~49% |
| AIME 2025 | ~80% | ~70% | N/A | N/A | ~73% | ~70% |
| HumanEval | 97.0% | 94.5% | ~96% | ~94% | ~95% | 92.3% |

**Note:** Benchmarks are imperfect and rapidly saturating. Real-world performance depends on use case, prompting, and evaluation methodology. SWE-bench (verified) has emerged as the key agentic coding benchmark in 2025. The Chatbot Arena Elo rankings (based on human preferences) are often more informative.

### Chatbot Arena Elo Rankings (approximate, mid-2025)

```
Standard (non-reasoning) models:
1. Claude Sonnet 4 / GPT-4o          (~1290+ Elo)
2. Gemini 2.5 Flash                  (~1280 Elo)
3. Claude 3.5 Sonnet                 (~1280 Elo)
4. DeepSeek V3                       (~1270 Elo)
5. Llama 4 Maverick                  (~1265 Elo)
6. Gemini 2.0 Flash                  (~1260 Elo)
7. GPT-4o mini                       (~1220 Elo)

Reasoning models (with thinking):
1. Claude Opus 4 (thinking)          (~1380+ Elo)
2. o3                                (~1370 Elo)
3. Gemini 2.5 Pro (thinking)         (~1360 Elo)
4. Claude 3.7 Sonnet (thinking)      (~1350 Elo)
5. DeepSeek R1                       (~1330 Elo)
6. o4-mini                           (~1320 Elo)
```

### Use Case Recommendations (Updated Mid-2025)

| Use Case | Best Choice | Runner Up | Why |
|----------|------------|-----------|-----|
| **General Chat** | Claude Sonnet 4 / GPT-4o | Gemini 2.5 Flash | Best overall quality |
| **Coding (agentic)** | Claude Opus 4 | o3 / Claude Sonnet 4 | SWE-bench SOTA, best with large codebases |
| **Coding (fast)** | Claude Sonnet 4 | GPT-4o / DeepSeek V3 | Fast + high quality for code |
| **Math/Reasoning** | o3 / DeepSeek R1 | Gemini 2.5 Pro (thinking) | Purpose-built for reasoning |
| **Long Documents** | Llama 4 Scout (10M) | Gemini 1.5 Pro (2M) | Longest context windows |
| **Cost-Sensitive** | GPT-4o mini / DeepSeek V3 | Gemini 2.5 Flash | Best quality per dollar |
| **Cost-Sensitive Reasoning** | DeepSeek R1 / o4-mini | Gemini 2.5 Flash | Cheapest reasoning-capable |
| **Privacy/On-Prem** | Llama 4 Maverick / Llama 3.1 | Mistral / DeepSeek V3 | Open weights, self-hosted |
| **Multimodal (Video)** | Gemini 2.5 Pro | GPT-4o | Native video understanding |
| **Enterprise** | Claude / GPT-4o | Gemini | Mature APIs, compliance, MCP |
| **Open Source Research** | Llama 4 Maverick | Qwen 2.5 / Mistral | Full model access |
| **Small/Edge** | Llama 3.2 1B/3B | Gemma 3 / Phi-4 | Optimized for on-device |
| **Agentic Workflows** | Claude Opus 4 | o3 + tools | Best multi-step agent performance |

> **YOUR EXPERIENCE**: Based on your work at RavianAI building agentic AI systems with AutoGen and LangGraph, you can speak to the practical differences between these models for agent orchestration. Claude's extended thinking and tool use, combined with MCP, makes it particularly strong for the agentic workflows you've built.

---

## 8.3 Pricing Comparison (per 1M tokens, as of mid-2025)

| Provider | Model | Input | Output | Notes |
|----------|-------|-------|--------|-------|
| **OpenAI** | GPT-4o | $2.50 | $10.00 | |
| | GPT-4o mini | $0.15 | $0.60 | Best budget option |
| | GPT-4.5 | $75.00 | $150.00 | Very expensive dense model |
| | o3 | $10.00 | $40.00 | Strong reasoning |
| | o3-mini | $1.10 | $4.40 | Efficient reasoning |
| | o4-mini | $1.10 | $4.40 | Latest efficient reasoning |
| **Anthropic** | Claude 3.5 Haiku | $0.80 | $4.00 | Fast, efficient |
| | Claude 3.7 Sonnet | $3.00 | $15.00 | Hybrid reasoning |
| | Claude Sonnet 4 | $3.00 | $15.00 | Latest Sonnet |
| | Claude Opus 4 | $15.00 | $75.00 | Best agentic model |
| **Google** | Gemini 2.0 Flash | $0.10 | $0.40 | Cheapest major model |
| | Gemini 2.5 Flash | $0.15 | $0.60 | Fast reasoning |
| | Gemini 2.5 Pro | $1.25 | $10.00 | Reasoning + long context |
| **Mistral** | Mistral Large | $2.00 | $6.00 | |
| | Mistral Small 3 | $0.10 | $0.30 | Very competitive |
| **DeepSeek** | DeepSeek V3 | $0.27 | $1.10 | Extremely cheap |
| | DeepSeek R1 | $0.55 | $2.19 | Cheapest reasoning model |
| **Meta** | Llama 4 Maverick | FREE | FREE | Self-hosted (you pay for GPU) |
| | Llama 3.1 | FREE | FREE | Self-hosted |

---

## 8.4 API Feature Comparison (Mid-2025)

| Feature | OpenAI | Anthropic | Google | Mistral | DeepSeek |
|---------|--------|-----------|--------|---------|----------|
| Function Calling | Yes | Yes | Yes | Yes | Yes |
| JSON Mode | Yes | Yes | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes | Yes | Yes |
| Vision | Yes | Yes | Yes | Yes (Pixtral) | No (V3) |
| Audio | Yes | No | Yes | No | No |
| Fine-tuning | Yes | Limited | Yes | Yes | No |
| Embeddings | Yes | No (use Voyage) | Yes | Yes | No |
| Batch API | Yes | Yes | Yes | Yes | No |
| Prompt Caching | Yes | Yes | Yes | Yes | Yes |
| Computer Use | Yes (Operator) | Yes | No | No | No |
| Extended Thinking | Yes (o-series) | Yes (3.7, Opus 4, Sonnet 4) | Yes (2.5 Pro/Flash) | No | Yes (R1) |
| MCP Support | Partial | Yes (creator) | Partial | No | No |
| Code Execution | Yes (sandbox) | Yes (tool use) | Yes | No | No |
| Responses/Messages API | Yes (new) | Yes (Messages) | Yes | Yes | Yes |

---

<a id="section-9"></a>
# SECTION 9: AI AGENT FRAMEWORKS LANDSCAPE 2025-2026

## 9.1 Overview: The Rise of Agentic AI

### Q: What is the current landscape of AI agent frameworks? How do they differ?

**Answer:**

2024-2025 saw an explosion of AI agent frameworks, each with different philosophies about how to orchestrate LLM-powered agents. Understanding these frameworks is critical for GenAI engineers.

**Core Agent Paradigms:**

```
1. Single Agent + Tools (simplest)
   LLM -> [Tool A, Tool B, Tool C] -> Output
   Examples: OpenAI Assistants, basic LangChain agents

2. Multi-Agent Conversation (collaborative)
   Agent A <-> Agent B <-> Agent C (converse to solve tasks)
   Examples: AutoGen, CrewAI

3. Graph-Based Orchestration (most flexible)
   Nodes (agents/functions) connected by edges (conditional routing)
   Examples: LangGraph, custom state machines

4. Hierarchical / Manager Pattern
   Manager Agent -> [Worker Agent 1, Worker Agent 2, ...]
   Examples: CrewAI hierarchical, AutoGen nested chats
```

## 9.2 Major Agent Frameworks

### LangChain / LangGraph

**LangGraph** (by LangChain team) has become the most popular production agent framework:

```python
# LangGraph agent example
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list, "add"]

# Define the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", call_model)       # LLM reasoning
graph.add_node("tools", ToolNode(tools))   # Tool execution

# Add edges with conditional routing
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,  # Function that checks if tools need to be called
    {"continue": "tools", "end": END}
)
graph.add_edge("tools", "agent")  # After tools, go back to agent

# Compile and run
app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Research and summarize...")]})
```

**Key features:**
- Graph-based state machine for agent logic
- Built-in persistence (checkpointing) for long-running agents
- Human-in-the-loop support
- Streaming support
- Deployment via LangGraph Cloud / LangGraph Platform
- Best for: Complex workflows requiring precise control over agent behavior

> **YOUR EXPERIENCE**: Based on your work at RavianAI, you have hands-on experience building production LangGraph agents with custom state management, conditional routing, and tool integration. Discuss your architectural patterns, how you handled state persistence, and error recovery in agent workflows.

### AutoGen (Microsoft)

**AutoGen** enables multi-agent conversations where agents collaborate through natural language:

```python
# AutoGen multi-agent example (v0.4+ / AG2)
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Define agents with different roles
researcher = AssistantAgent(
    name="Researcher",
    system_message="You research topics thoroughly using available tools.",
    llm_config=llm_config
)

analyst = AssistantAgent(
    name="Analyst",
    system_message="You analyze data and provide insights.",
    llm_config=llm_config
)

coder = AssistantAgent(
    name="Coder",
    system_message="You write and execute Python code.",
    llm_config=llm_config,
    code_execution_config={"work_dir": "coding"}
)

# Group chat for multi-agent collaboration
group_chat = GroupChat(
    agents=[researcher, analyst, coder],
    messages=[],
    max_round=20,
    speaker_selection_method="auto"  # LLM decides who speaks next
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# Start the conversation
researcher.initiate_chat(
    manager,
    message="Analyze the latest trends in AI agent frameworks."
)
```

**Key features:**
- Conversational multi-agent pattern (agents chat with each other)
- Built-in code execution
- Flexible speaker selection (round-robin, auto, custom)
- Nested conversations and teachable agents
- AutoGen v0.4 rewrite (AG2) with improved architecture
- Best for: Research tasks, data analysis, collaborative problem-solving

> **YOUR EXPERIENCE**: Based on your work at RavianAI with AutoGen, you can discuss the practical trade-offs of conversational multi-agent patterns vs. graph-based orchestration. Share your experience with speaker selection strategies, managing agent context, and when AutoGen excels vs. when LangGraph is more appropriate.

### CrewAI

**CrewAI** uses a role-based approach inspired by real-world team structures:

```python
from crewai import Agent, Task, Crew, Process

# Define agents with roles
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="Expert researcher with 10 years in AI/ML...",
    tools=[search_tool, web_scraper],
    llm="gpt-4o"
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content about AI developments",
    backstory="Renowned content strategist...",
    llm="claude-3-5-sonnet"
)

# Define tasks
research_task = Task(
    description="Research the latest AI agent frameworks...",
    agent=researcher,
    expected_output="Comprehensive research report"
)

writing_task = Task(
    description="Write an article based on the research...",
    agent=writer,
    expected_output="Published article draft",
    context=[research_task]  # Depends on research
)

# Create and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True
)

result = crew.kickoff()
```

**Key features:**
- Role-based agent design (intuitive for non-engineers)
- Sequential and hierarchical process modes
- Task dependency management
- Built-in memory (short-term, long-term, entity)
- Best for: Content creation, research pipelines, business workflows

### Other Notable Frameworks (2025)

| Framework | Developer | Key Differentiator | Best For |
|-----------|-----------|-------------------|----------|
| **OpenAI Assistants API** | OpenAI | First-party, built-in tools (code interpreter, file search) | Simple agent tasks with OpenAI models |
| **Claude Code** | Anthropic | CLI-based coding agent, MCP integration | Software engineering, codebase-level tasks |
| **Amazon Bedrock Agents** | AWS | Managed, enterprise-grade, AWS integration | Enterprise deployments |
| **Semantic Kernel** | Microsoft | .NET/Python, enterprise-focused, Azure integration | Enterprise .NET apps |
| **Haystack** | deepset | Pipeline-based, RAG-focused agents | RAG + agent hybrid systems |
| **DSPy** | Stanford | Programming (not prompting) LLMs, auto-optimization | Research, systematic prompt optimization |
| **Pydantic AI** | Pydantic | Type-safe agents with Pydantic validation | Production Python APIs |
| **OpenAI Agents SDK** | OpenAI | First-party SDK, Swarm successor, built-in tools + tracing | Production multi-agent with OpenAI models |
| **smolagents** | HuggingFace | Lightweight code agents, open-source models | Open-source agent building |
| **AG2 (AutoGen v2)** | Microsoft | Rewrite of AutoGen with improved architecture | Next-gen multi-agent |
| **Pydantic AI** | Pydantic | Type-safe agents with Pydantic validation | Production Python APIs |

**CrewAI Agent Operations Platform (AOP) (Late 2025):**
CrewAI launched its Agent Operations Platform, adding a control plane for deploying, monitoring, and governing agent teams in production. This addresses the "last mile" challenge of taking multi-agent systems from development to production.

**OpenAI Agents SDK (2025):**
Successor to the experimental Swarm framework. A production-grade SDK that allows building custom agents with OpenAI models. Uses a routine-based model where agents are defined through prompts and function docstrings with built-in tool usage, function calling, guardrails, and tracing.

## 9.3 Agent Design Patterns

### Q: What are the common design patterns for building LLM agents?

**Answer:**

**Pattern 1: ReAct (Reason + Act)**
```
Think -> Act -> Observe -> Think -> Act -> Observe -> ... -> Answer
```
Most fundamental pattern. LLM alternates between reasoning and tool use.

**Pattern 2: Plan-and-Execute**
```
Plan (decompose into subtasks) -> Execute each subtask -> Synthesize
```
Better for complex, multi-step tasks. Separates planning from execution.

**Pattern 3: Reflection / Self-Critique**
```
Generate -> Critique -> Revise -> Critique -> Finalize
```
Agent evaluates its own output and iteratively improves it.

**Pattern 4: Multi-Agent Debate**
```
Agent A proposes -> Agent B critiques -> Agent A revises -> Consensus
```
Multiple agents with different perspectives improve quality.

**Pattern 5: Supervisor / Router**
```
Supervisor receives task -> Routes to specialized agent -> Collects results
```
One agent orchestrates others, each with specialized capabilities.

**Pattern 6: Human-in-the-Loop**
```
Agent works -> Reaches checkpoint -> Human approves/redirects -> Agent continues
```
Critical for production systems where full autonomy is risky.

> **YOUR EXPERIENCE**: Based on your work at MathCo building multi-agent analytics systems, you can discuss how you implemented supervisor/router patterns for routing analytics queries to specialized agents (data retrieval, statistical analysis, visualization). Discuss the challenges of agent coordination and context management in production.

## 9.4 Agent Evaluation and Challenges

### Q: What are the key challenges in building production agent systems?

**Answer:**

**Key Challenges:**
1. **Reliability**: Agents can fail unpredictably (wrong tool calls, infinite loops, hallucinated actions)
2. **Cost**: Agentic loops consume many tokens (each reasoning step is an LLM call)
3. **Latency**: Multi-step agents can take minutes for complex tasks
4. **Observability**: Debugging agent behavior across multiple steps is hard
5. **Safety**: Agents with real-world tools (code execution, web access) need guardrails
6. **Evaluation**: No standard benchmarks for agent quality (SWE-bench is closest for coding)

**Production Best Practices:**
- Always implement maximum iteration limits
- Add human-in-the-loop checkpoints for high-stakes actions
- Use structured outputs for tool calls (JSON schemas)
- Implement comprehensive logging and tracing (LangSmith, Phoenix, Braintrust)
- Start with simple single-agent patterns; add complexity only when needed
- Use prompt caching aggressively to reduce cost
- Test with diverse scenarios including adversarial inputs

---

---

---

<a id="section-10"></a>
# SECTION 10: MODEL CONTEXT PROTOCOL (MCP) BY ANTHROPIC

## 10.1 What is MCP?

### Q: Explain the Model Context Protocol (MCP). Why was it created and how does it work?

**Answer:**

**Model Context Protocol (MCP)** is an open standard created by Anthropic (released November 2024) that provides a universal, standardized way for LLM applications to connect to external data sources and tools. It has rapidly become one of the most important infrastructure standards in the AI ecosystem.

**The Problem MCP Solves:**

Before MCP, every LLM application had to build custom integrations:
```
Without MCP (N x M integration problem):
  App 1 --custom code--> Tool A
  App 1 --custom code--> Tool B
  App 2 --custom code--> Tool A  (duplicate work!)
  App 2 --custom code--> Tool C
  App 3 --custom code--> Tool B  (duplicate work!)
  ...
  N apps x M tools = N*M custom integrations

With MCP (N + M):
  App 1 --MCP--> [MCP Server A (Tool A)]
  App 1 --MCP--> [MCP Server B (Tool B)]
  App 2 --MCP--> [MCP Server A (Tool A)]  (reuse!)
  App 2 --MCP--> [MCP Server C (Tool C)]
  ...
  N apps + M servers = N+M integrations (each built once)
```

**Analogy:** MCP is to AI applications what USB is to peripherals -- a universal connector standard.

## 10.2 MCP Architecture

### Q: Describe the MCP architecture in detail.

**Answer:**

**Core Components:**

```
+------------------+     +------------------+     +------------------+
|   MCP Host       |     |   MCP Client     |     |   MCP Server     |
|  (Application)   |<--->|  (Protocol Layer) |<--->|  (Integration)   |
|                  |     |                  |     |                  |
| Claude Desktop   |     | Manages 1:many   |     | Exposes tools,   |
| Claude Code      |     | server           |     | resources,       |
| IDE Plugin       |     | connections      |     | prompts          |
| Custom App       |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
                                                         |
                                                         v
                                                  +------------------+
                                                  | External System  |
                                                  | (Database, API,  |
                                                  |  File System,    |
                                                  |  SaaS tool, etc) |
                                                  +------------------+
```

**MCP exposes three core primitives:**

1. **Tools**: Functions that the LLM can call (similar to function calling)
   ```json
   {
     "name": "query_database",
     "description": "Execute a SQL query against the analytics database",
     "inputSchema": {
       "type": "object",
       "properties": {
         "query": {"type": "string", "description": "SQL query to execute"}
       },
       "required": ["query"]
     }
   }
   ```

2. **Resources**: Data that can be read by the LLM (like files, database records)
   ```json
   {
     "uri": "file:///project/src/main.py",
     "name": "Main application file",
     "mimeType": "text/x-python"
   }
   ```

3. **Prompts**: Reusable prompt templates
   ```json
   {
     "name": "code_review",
     "description": "Review code for bugs and improvements",
     "arguments": [
       {"name": "code", "description": "The code to review", "required": true}
     ]
   }
   ```

**Transport Layer:**
- **stdio**: Local communication (process stdin/stdout) -- most common for local tools
- **HTTP with SSE**: Remote communication via Server-Sent Events -- for remote/cloud servers
- **Streamable HTTP**: Newer transport option for better streaming support

## 10.3 Building MCP Servers

### Q: How do you build an MCP server?

**Answer:**

```python
# Example MCP Server in Python (using official SDK)
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Create server
server = Server("analytics-server")

# Define available tools
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="run_sql_query",
            description="Execute a SQL query against the analytics database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    },
                    "database": {
                        "type": "string",
                        "enum": ["analytics", "users", "products"],
                        "description": "Which database to query"
                    }
                },
                "required": ["query", "database"]
            }
        ),
        Tool(
            name="create_chart",
            description="Create a visualization from data",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "JSON data to visualize"},
                    "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter"]}
                },
                "required": ["data", "chart_type"]
            }
        )
    ]

# Handle tool calls
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "run_sql_query":
        result = execute_sql(arguments["query"], arguments["database"])
        return [TextContent(type="text", text=str(result))]
    elif name == "create_chart":
        chart_url = create_visualization(arguments["data"], arguments["chart_type"])
        return [TextContent(type="text", text=f"Chart created: {chart_url}")]

# Run the server
async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

import asyncio
asyncio.run(main())
```

**Configuration (Claude Desktop example):**
```json
{
  "mcpServers": {
    "analytics": {
      "command": "python",
      "args": ["/path/to/analytics_server.py"],
      "env": {
        "DB_CONNECTION_STRING": "postgresql://..."
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_..."
      }
    }
  }
}
```

## 10.4 MCP Ecosystem (2025)

### Q: What is the current MCP ecosystem?

**Answer:**

**Official/Popular MCP Servers:**

| Server | Purpose | Developer |
|--------|---------|-----------|
| **Filesystem** | Read/write local files | Anthropic |
| **GitHub** | Repos, issues, PRs | Anthropic |
| **GitLab** | GitLab integration | Community |
| **PostgreSQL** | Database queries | Anthropic |
| **Slack** | Channel messaging, search | Anthropic |
| **Google Drive** | Document access | Anthropic |
| **Puppeteer** | Browser automation | Anthropic |
| **Brave Search** | Web search | Community |
| **Figma** | Design file access | Figma |
| **Linear** | Issue tracking | Community |
| **Sentry** | Error tracking | Sentry |
| **Docker** | Container management | Community |
| **Kubernetes** | Cluster management | Community |

**MCP-Compatible Hosts (Applications):**
- Claude Desktop (first MCP host)
- Claude Code (CLI)
- Cursor (IDE)
- Windsurf (IDE)
- VS Code + Continue (IDE extension)
- Zed Editor
- Cline (VS Code agent)
- Various custom applications

**Why MCP Matters for Interviews:**
1. **Standardization**: First widely-adopted standard for LLM tool integration
2. **Ecosystem effect**: Growing rapidly, similar to how REST APIs standardized web services
3. **Production relevance**: Being adopted by major tools and platforms
4. **Architecture question**: Understanding MCP shows you think about system design, not just model APIs
5. **Open standard**: Anyone can build MCP servers and clients

> **YOUR EXPERIENCE**: Based on your work at RavianAI building agentic AI systems, MCP is directly relevant to the tool integration challenges you solved. You can discuss how MCP would simplify the custom tool integrations you built, and compare it to the ad-hoc function calling approaches you used with AutoGen and LangGraph. At MathCo, MCP could standardize how analytics agents connect to databases, visualization tools, and data sources.

## 10.5 MCP vs. Function Calling

### Q: How does MCP compare to standard function calling?

**Answer:**

| Aspect | Standard Function Calling | MCP |
|--------|--------------------------|-----|
| **Scope** | Per-API-call tool definitions | Persistent server with tools, resources, prompts |
| **Discovery** | Tools defined in each request | Tools discovered dynamically from server |
| **State** | Stateless (per call) | Stateful (server maintains connections) |
| **Reusability** | Tools redefined each call | Servers reused across applications |
| **Ecosystem** | Provider-specific (OpenAI, Anthropic formats) | Universal standard |
| **Resources** | No concept | Can expose data/files as resources |
| **Transport** | HTTP API | stdio, HTTP+SSE, Streamable HTTP |
| **Who builds** | App developer defines tools | Server developer publishes once, all apps consume |

**Key insight**: Function calling tells the LLM *what tools exist*. MCP provides a standard way to *discover, connect to, and use* those tools across any application.

```python
# Function calling: tools defined per request
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[
        {"name": "search_db", "description": "...", "input_schema": {...}}
    ],
    messages=[...]
)

# MCP: tools discovered from servers, available persistently
# The host (Claude Desktop, Claude Code, etc.) connects to MCP servers
# and automatically makes their tools available to the model
# No per-request tool definitions needed
```

---

## 10.6 Late 2025 - Early 2026 Frontier Model Benchmarks

### Q: Compare the latest frontier models (GPT-5.2 vs Claude Opus 4.5 vs Gemini 3 Pro) on key benchmarks.

**Answer:**

| Benchmark | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro | What It Measures |
|-----------|---------|-----------------|--------------|-----------------|
| **SWE-bench Verified** | 74.9% | **80.9%** | 76.8% | Real-world coding (GitHub issues) |
| **ARC-AGI-2** | **~54%** | ~37% | ~45% | Genuine novel reasoning |
| **AIME 2025** | **100%** | ~85% | ~90% | Competition-level math |
| **MMLU-Pro** | ~92% | ~91% | ~93% | Knowledge breadth |
| **Context Window** | 400K | 200K | **1M** | Maximum context |

**Key Takeaways for Interviews:**
- **No single winner**: Each model excels in different areas
- **Claude Opus 4.5**: Best for coding/agentic tasks (SWE-bench SOTA at 80.9%)
- **GPT-5.2**: Best for math/reasoning (perfect AIME score, highest ARC-AGI-2)
- **Gemini 3 Pro**: Best for multimodal + long context (1M tokens)
- **Smart routing**: Production systems increasingly route tasks to the best model for each task type
- **Cost matters**: Mid-tier models (Sonnet, Flash) handle 80%+ of production traffic

**The Frontier Model Race (Late 2025 Timeline):**
```
Mid 2025:  GPT-5 released (400K context, major reasoning leap)
Late 2025: Claude Opus 4.5 (SWE-bench record, best agentic model)
Late 2025: Gemini 3 Pro (1M context, strongest multimodal)
Late 2025: GPT-5.2 (refinement, 100% AIME math)
Late 2025: DeepSeek V3.2 (open-source competitor, strong reasoning)
Early 2026: Claude Opus 4.6 (latest Anthropic flagship)
Early 2026: Qwen3 (Alibaba, strong open-source alternative)
```

> **YOUR EXPERIENCE**: Understanding the strengths and weaknesses of each model family is critical for AI Engineers. At RavianAI, you likely need to make model selection decisions for different use cases -- use Claude for coding agents, GPT for reasoning-heavy tasks, and Gemini for multimodal/long-context needs.

---

---

# APPENDIX: QUICK REFERENCE FORMULAS AND CONCEPTS

## Key Formulas to Remember

```
1. Self-Attention:
   Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

2. Cross-Entropy Loss:
   L = -sum(y_true * log(y_pred))

3. Perplexity:
   PPL = exp(-1/N * sum(log P(token_i)))

4. BLEU:
   BLEU = BP * exp(sum(w_n * log(p_n)))

5. DPO Loss:
   L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

6. RLHF Objective:
   J = E[r(x,y)] - beta * KL(pi || pi_ref)

7. Scaling Law (Chinchilla):
   Optimal tokens ~ 20 * parameters

8. KV Cache Size:
   2 * layers * kv_heads * head_dim * seq_len * bytes

9. Model Memory (inference):
   Parameters * bytes_per_param + KV_cache

10. Temperature Scaling:
    P(token_i) = exp(logit_i / T) / sum(exp(logit_j / T))
```

## Acronym Reference

| Acronym | Full Form |
|---------|-----------|
| LLM | Large Language Model |
| NLP | Natural Language Processing |
| MHA | Multi-Head Attention |
| GQA | Grouped Query Attention |
| MQA | Multi-Query Attention |
| RoPE | Rotary Position Embedding |
| ALiBi | Attention with Linear Biases |
| FFN | Feed-Forward Network |
| SwiGLU | Swish-Gated Linear Unit |
| RMSNorm | Root Mean Square Normalization |
| BPE | Byte Pair Encoding |
| SFT | Supervised Fine-Tuning |
| RLHF | Reinforcement Learning from Human Feedback |
| DPO | Direct Preference Optimization |
| PPO | Proximal Policy Optimization |
| GRPO | Group Relative Policy Optimization |
| CAI | Constitutional AI |
| RLAIF | RL from AI Feedback |
| LoRA | Low-Rank Adaptation |
| QLoRA | Quantized LoRA |
| RAG | Retrieval-Augmented Generation |
| CoT | Chain of Thought |
| ToT | Tree of Thought |
| ICL | In-Context Learning |
| MoE | Mixture of Experts |
| KV Cache | Key-Value Cache |
| PPL | Perplexity |
| BLEU | Bilingual Evaluation Understudy |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation |
| MMLU | Massive Multitask Language Understanding |
| MLA | Multi-head Latent Attention |
| GPTQ | GPT Quantization |
| AWQ | Activation-aware Weight Quantization |
| PTQ | Post-Training Quantization |
| QAT | Quantization-Aware Training |

---

*[END OF GUIDE]*

*Guide compiled: February 2025, updated February 2026*
*Covers material relevant for AI Engineer interviews through 2025-2026*
