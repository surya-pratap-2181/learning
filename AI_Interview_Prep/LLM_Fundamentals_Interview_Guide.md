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

---

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

**Context window sizes across models (as of 2025):**
| Model | Context Window |
|-------|---------------|
| GPT-3 | 2,048 tokens |
| GPT-3.5 Turbo | 4,096 / 16,384 |
| GPT-4 | 8,192 / 32,768 / 128,000 |
| GPT-4o | 128,000 |
| Claude 3.5 Sonnet | 200,000 |
| Claude 3 Opus | 200,000 |
| Gemini 1.5 Pro | 1,000,000 / 2,000,000 |
| Llama 3.1 | 128,000 |
| Mistral Large | 128,000 |
| DeepSeek V3 | 128,000 |

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

*[End of Section 6]*
