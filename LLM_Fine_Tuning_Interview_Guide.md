---
title: "Fine-Tuning Deep Dive"
layout: default
parent: "LLM & AI Fundamentals"
nav_order: 3
---

# Comprehensive Guide: Fine-Tuning LLMs -- Interview Questions for AI Engineers (2025-2026)

---

## TABLE OF CONTENTS

1. [Fine-Tuning Fundamentals](#1-fine-tuning-fundamentals)
2. [PEFT Techniques Deep Dive](#2-peft-techniques-deep-dive)
3. [Training Data Preparation](#3-training-data-preparation)
4. [RLHF -- Reinforcement Learning from Human Feedback](#4-rlhf)
5. [DPO -- Direct Preference Optimization](#5-dpo)
6. [Quantization](#6-quantization)
7. [Training Infrastructure](#7-training-infrastructure)
8. [Evaluation of Fine-Tuned Models](#8-evaluation)
9. [Tools and Frameworks](#9-tools-and-frameworks)
10. [Common Interview Questions with Detailed Answers and Code](#10-interview-questions)

---

## 1. FINE-TUNING FUNDAMENTALS

### 1.1 What Is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained foundation model (e.g., LLaMA, Mistral, GPT) and continuing training on a smaller, task-specific or domain-specific dataset. The goal is to adapt the model's behavior, knowledge, or style for a particular use case.

**The LLM Adaptation Spectrum (from cheapest to most expensive):**

```
Prompt Engineering --> Few-Shot Prompting --> RAG --> Soft Prompt Tuning --> PEFT (LoRA) --> Full Fine-Tuning --> Pre-Training from Scratch
```

### 1.2 Full Fine-Tuning vs Parameter-Efficient Fine-Tuning (PEFT)

#### Full Fine-Tuning
- **What:** Updates ALL parameters of the model during training.
- **Parameters updated:** 100% of model weights (e.g., all 7B parameters for LLaMA-7B).
- **Memory requirement:** Extremely high. For a 7B model in fp16: ~14GB for weights + ~14GB for gradients + ~56GB for Adam optimizer states = ~84GB+ GPU memory.
- **When to use:**
  - You have massive compute resources.
  - You need maximum performance on a specific task.
  - You are building a foundation model variant.
  - The domain is very different from pre-training data (e.g., medical, legal).
- **Risks:** Catastrophic forgetting (model loses general capabilities), overfitting on small datasets.

#### Parameter-Efficient Fine-Tuning (PEFT)
- **What:** Only updates a small subset of parameters (typically 0.1%-5% of total).
- **Memory requirement:** Drastically reduced. A 7B model with LoRA rank 16 adds ~4M trainable parameters (~0.06%).
- **Methods:** LoRA, QLoRA, Prefix Tuning, P-Tuning, Adapter Layers, IA3.
- **When to use:**
  - Limited GPU resources (single GPU fine-tuning).
  - You want to maintain general capabilities while specializing.
  - You need to serve multiple fine-tuned variants efficiently (swap LoRA adapters).
  - Most practical production scenarios.

#### Decision Matrix: Full Fine-Tuning vs PEFT

| Factor | Full Fine-Tuning | PEFT (LoRA/QLoRA) |
|--------|-----------------|-------------------|
| GPU Memory | 4-8x model size | 1-2x model size |
| Training Speed | Slower | Faster |
| Risk of Catastrophic Forgetting | Higher | Lower |
| Performance Ceiling | Higher | Slightly lower (but often negligible) |
| Multi-Task Serving | Separate model copies | Swap adapters on base model |
| Typical Trainable Params | 100% | 0.01% - 5% |

### 1.3 When to Fine-Tune vs RAG vs Prompt Engineering

This is one of the MOST commonly asked interview questions.

#### Prompt Engineering (Use First)
- **When:** Task can be expressed with clear instructions and examples.
- **Best for:** General tasks, rapid prototyping, when data is limited.
- **Limitations:** Context window limits, inconsistent outputs, no new knowledge injection, no behavioral changes.
- **Cost:** Lowest (no training, only inference cost).

#### RAG -- Retrieval-Augmented Generation (Use Second)
- **When:** You need the model to use external, up-to-date, or proprietary knowledge.
- **Best for:** Question answering over documents, knowledge-intensive tasks, reducing hallucinations with grounding.
- **How:** Retrieve relevant documents from a vector store, inject into the prompt as context.
- **Limitations:** Retrieval quality bottleneck, increased latency, context window limits, cannot change model behavior/style/format.
- **Cost:** Medium (embedding + vector DB + retrieval infrastructure).

#### Fine-Tuning (Use When Others Fail)
- **When:**
  - You need to change the model's behavior, tone, or output format consistently.
  - You need the model to learn a specialized skill (e.g., code generation for a specific framework).
  - RAG context windows are insufficient.
  - You need consistent structured output (JSON, SQL, etc.).
  - Domain-specific terminology and reasoning patterns.
  - Latency requirements prohibit large prompts.
- **Limitations:** Requires curated training data, compute resources, risk of overfitting.
- **Cost:** Highest (GPU training + data curation).

#### Hybrid Approach (Most Common in Production)
Fine-tuned model + RAG is the most powerful production pattern:
- Fine-tune for behavior/format/style.
- RAG for up-to-date knowledge and grounding.

```
Interview Answer Framework:
"Start with prompt engineering. If the model lacks knowledge, add RAG.
If the model lacks behavioral alignment, fine-tune. In production,
combine fine-tuning (for behavior) with RAG (for knowledge)."
```

### 1.4 Types of Fine-Tuning

1. **Supervised Fine-Tuning (SFT):** Train on input-output pairs (instruction, response).
2. **Instruction Tuning:** SFT specifically on instruction-following datasets (FLAN, Alpaca).
3. **Alignment Tuning:** RLHF or DPO to align model with human preferences.
4. **Continual Pre-Training:** Continue unsupervised training on domain-specific corpus (medical, legal, code).
5. **Task-Specific Fine-Tuning:** Narrow fine-tuning for classification, NER, summarization, etc.

### 1.5 The Modern LLM Training Pipeline

```
Stage 1: Pre-Training (Unsupervised, trillions of tokens)
    --> Base Model (e.g., LLaMA-3 base)
Stage 2: Supervised Fine-Tuning (SFT on instruction data)
    --> Instruction-Tuned Model (e.g., LLaMA-3-Instruct)
Stage 3: Alignment (RLHF / DPO / KTO)
    --> Aligned Model (e.g., LLaMA-3-Chat)
Stage 4: Domain/Task Adaptation (Optional)
    --> Specialized Model
```

---

## 2. PEFT TECHNIQUES DEEP DIVE

### 2.1 LoRA (Low-Rank Adaptation)

**Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

#### How It Works

The core insight: weight updates during fine-tuning have a low intrinsic rank. Instead of updating the full weight matrix W (d x d), decompose the update into two low-rank matrices.

**Mathematical Formulation:**

```
Original: h = W * x            (where W is d x d)
LoRA:     h = W * x + (B * A) * x    (where B is d x r, A is r x d, r << d)

W_new = W_frozen + delta_W = W_frozen + B * A
```

- **W** is the original pre-trained weight matrix (FROZEN, not updated).
- **A** is initialized with random Gaussian values.
- **B** is initialized with zeros (so delta_W = 0 at start, preserving pre-trained behavior).
- **r** (rank) is the key hyperparameter (typically 4, 8, 16, 32, 64).
- **alpha** (scaling factor): The update is scaled by `alpha/r`, controlling the magnitude of LoRA updates.

**Parameter Savings Example:**
- Original weight matrix: 4096 x 4096 = 16,777,216 parameters
- LoRA with rank 16: (4096 x 16) + (16 x 4096) = 131,072 parameters
- **Savings: 99.2% fewer trainable parameters**

#### Which Layers to Apply LoRA

In a Transformer, LoRA is typically applied to:
- **Query (Wq) and Value (Wv) projections:** Most common, recommended by original paper.
- **All attention projections (Wq, Wk, Wv, Wo):** Better performance, slightly more parameters.
- **Attention + MLP layers:** Best performance, most parameters.

```python
# Typical LoRA configuration for LLaMA
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Alpha (scaling = alpha/r = 2)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

#### Key Hyperparameters

| Parameter | Typical Values | Effect |
|-----------|---------------|--------|
| `r` (rank) | 4, 8, 16, 32, 64 | Higher = more capacity, more parameters |
| `lora_alpha` | Usually 2*r (16, 32, 64) | Scaling factor; higher = stronger updates |
| `lora_dropout` | 0.0 - 0.1 | Regularization to prevent overfitting |
| `target_modules` | Attention layers, or all linear layers | More modules = more capacity |

#### Pros and Cons

**Pros:**
- Reduces trainable parameters by 90-99%.
- No additional inference latency (merge weights after training).
- Multiple LoRA adapters can share one base model.
- Minimal catastrophic forgetting.

**Cons:**
- Slightly lower ceiling than full fine-tuning on some tasks.
- Rank selection requires experimentation.
- Not suitable for very large domain shifts requiring fundamental changes.

### 2.2 QLoRA (Quantized LoRA)

**Paper:** "QLoRA: Efficient Finetuning of Quantized Language Models" (Dettmers et al., 2023)

#### How It Works

QLoRA combines 4-bit quantization of the base model with LoRA fine-tuning. Three key innovations:

1. **4-bit NormalFloat (NF4):** A new data type optimized for normally distributed neural network weights. Better than standard INT4 because neural network weights follow a normal distribution.

2. **Double Quantization:** Quantizes the quantization constants themselves, saving ~0.37 bits per parameter (~3GB for a 65B model).

3. **Paged Optimizers:** Uses NVIDIA unified memory to handle memory spikes during gradient checkpointing by automatically paging optimizer states to CPU RAM.

**Memory Comparison for LLaMA-7B:**

```
Full Fine-Tuning (fp16):   ~84 GB GPU memory
LoRA (fp16 base):          ~18 GB GPU memory
QLoRA (4-bit base + LoRA): ~6 GB GPU memory   <-- Can fit on single consumer GPU!
```

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # Double quantization
    bnb_4bit_quant_type="nf4",           # NormalFloat4 data type
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute in bf16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)  # Prepare for training

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

#### QLoRA vs LoRA Performance

Research shows QLoRA achieves ~99% of full fine-tuning performance while using ~1/14th the memory. The quality difference between LoRA (fp16) and QLoRA (4-bit) is typically negligible.

### 2.3 Prefix Tuning

**Paper:** "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (Li & Liang, 2021)

#### How It Works

- Prepends a sequence of learnable continuous vectors ("virtual tokens") to the key and value of every attention layer.
- These prefix vectors are trained while the entire model is frozen.
- Unlike discrete prompts, these are continuous embeddings optimized via backpropagation.

```
Original Attention:   Attention(Q, K, V)
Prefix Tuning:        Attention(Q, [P_k; K], [P_v; V])

Where P_k, P_v are learnable prefix matrices of shape (prefix_length, d_model)
```

**Trainable Parameters:** `num_layers * 2 * prefix_length * d_model`

**Pros:** Very parameter-efficient, no changes to model architecture.
**Cons:** Reduces effective context length, performance can be sensitive to prefix length, generally underperforms LoRA.

### 2.4 P-Tuning and P-Tuning v2

#### P-Tuning v1
- Adds learnable continuous embeddings ONLY at the input embedding layer.
- Uses a small LSTM or MLP to generate the soft prompt embeddings (reparameterization trick for stable training).
- Limited to input layer, so it has less expressive power.

#### P-Tuning v2
- Extends soft prompts to EVERY layer of the transformer (similar to prefix tuning).
- Removes the reparameterization network.
- Achieves comparable performance to full fine-tuning on many NLU benchmarks.

```python
from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,          # Number of prefix tokens
    prefix_projection=True,          # Use MLP to project prefix embeddings
    encoder_hidden_size=1024
)
```

### 2.5 Adapter Layers

**Paper:** "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019)

#### How It Works

- Inserts small bottleneck layers (adapters) between existing transformer layers.
- Each adapter: `Linear_down(d, r) -> NonLinearity -> Linear_up(r, d) + Residual`
- Only adapter parameters are trained; original model is frozen.

```
Adapter Architecture:
    Input (d) --> Down-project (d -> r) --> Activation (ReLU/GELU)
              --> Up-project (r -> d) --> + Residual Connection --> Output (d)
```

**Key difference from LoRA:** Adapters add sequential bottleneck layers (adds latency), while LoRA modifies existing weights via low-rank updates (no added latency when merged).

**Pros:** Well-studied, modular, composable.
**Cons:** Adds inference latency (cannot be merged like LoRA), less popular in 2025 practice.

### 2.6 IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**Paper:** "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (Liu et al., 2022)

#### How It Works

- Learns three rescaling vectors (NOT matrices) that multiply:
  1. Keys in self-attention (element-wise)
  2. Values in self-attention (element-wise)
  3. Intermediate activations in feed-forward layers (element-wise)

```
Standard:    k = W_k * x
IA3:         k = l_k * (W_k * x)     # l_k is a learned vector

Standard:    v = W_v * x
IA3:         v = l_v * (W_v * x)     # l_v is a learned vector

Standard:    ff = W_up * activation(W_down * x)
IA3:         ff = l_ff * (W_up * activation(W_down * x))
```

**Parameters:** Only 3 vectors per layer. For a model with d=4096 and 32 layers: 3 * 4096 * 32 = 393,216 parameters (vs. millions for LoRA).

**Pros:** Even fewer parameters than LoRA, no added latency, good for few-shot settings.
**Cons:** Lower capacity than LoRA, less widely adopted, can underperform on complex tasks.

### 2.7 Comparison Table: All PEFT Methods

| Method | Trainable Params | Inference Latency | Performance | Popularity (2025) |
|--------|-----------------|-------------------|-------------|-------------------|
| LoRA | ~0.1-1% | None (merge weights) | Excellent | Very High |
| QLoRA | ~0.1-1% | None (merge weights) | Excellent | Very High |
| Prefix Tuning | ~0.1% | Slight (longer sequence) | Good | Low |
| P-Tuning v2 | ~0.1-1% | Slight (longer sequence) | Good | Low |
| Adapter Layers | ~1-5% | Moderate (extra layers) | Good | Medium |
| IA3 | ~0.01% | None | Good (few-shot) | Low |
| Full Fine-Tuning | 100% | None | Best | Medium (resource-dependent) |

---

## 3. TRAINING DATA PREPARATION

### 3.1 Instruction Tuning Datasets

#### What Is Instruction Tuning?
Training a model to follow instructions by providing (instruction, input, output) triplets or (instruction, output) pairs.

#### Key Datasets

| Dataset | Size | Description | Quality |
|---------|------|-------------|---------|
| Alpaca (Stanford) | 52K | GPT-3.5-generated instruction data | Medium |
| Dolly (Databricks) | 15K | Human-written, commercially licensed | High |
| OpenAssistant (OASST) | 161K | Multi-turn conversation trees with human rankings | High |
| ShareGPT | ~90K | Real conversations shared by ChatGPT users | High (diverse) |
| FLAN Collection | ~15M | Massive multi-task instruction dataset | High |
| UltraChat | 1.5M | GPT-4 generated multi-turn conversations | Medium-High |
| Orca | ~1M | Complex reasoning with GPT-4 explanations | High |
| WizardLM (Evol-Instruct) | 250K | Evolved instructions increasing complexity | High |
| SlimOrca | 518K | Cleaned subset of OpenOrca | High |
| Capybara | 16K | Multi-turn, high quality | Very High |
| Deita | 6K-10K | Data-efficient instruction tuning | Very High |

### 3.2 Data Formats

#### Alpaca Format
```json
{
    "instruction": "Summarize the following text.",
    "input": "The quick brown fox jumped over the lazy dog...",
    "output": "A fox jumped over a dog."
}
```

Converted to prompt:
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Summarize the following text.

### Input:
The quick brown fox jumped over the lazy dog...

### Response:
A fox jumped over a dog.
```

#### ShareGPT Format (Multi-turn Conversations)
```json
{
    "conversations": [
        {"from": "human", "value": "What is machine learning?"},
        {"from": "gpt", "value": "Machine learning is a subset of AI..."},
        {"from": "human", "value": "Can you give an example?"},
        {"from": "gpt", "value": "Sure! Consider email spam filtering..."}
    ]
}
```

#### ChatML Format (Used by OpenAI, Mistral, many others)
```
<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
What is machine learning?<|im_end|>
<|im_start|>assistant
Machine learning is a subset of artificial intelligence...<|im_end|>
```

#### LLaMA-3 Chat Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is machine learning?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Machine learning is...<|eot_id|>
```

### 3.3 RLHF Data (Preference Data)

For reward model training:
```json
{
    "prompt": "Explain quantum computing",
    "chosen": "Quantum computing uses quantum bits (qubits) that can exist in superposition...",
    "rejected": "Quantum computing is when computers use quantum physics to be faster..."
}
```

### 3.4 DPO Data

Same format as RLHF preference data (chosen/rejected pairs), but used directly for DPO training without a separate reward model:

```json
{
    "prompt": "Write a poem about the ocean",
    "chosen": "Beneath the azure sky so wide,\nThe ocean breathes with every tide...",
    "rejected": "The ocean is big and blue. It has water and fish."
}
```

### 3.5 Data Quality and Curation Best Practices

**Key Principles (Critical for Interviews):**

1. **Quality > Quantity:** 1,000 high-quality examples often beat 100,000 low-quality ones. Research (LIMA paper) showed that 1,000 carefully curated examples can match larger datasets.

2. **Diversity Matters:** Cover the full range of desired behaviors, topics, complexities, and edge cases.

3. **Deduplication:** Remove near-duplicates using MinHash, SimHash, or embedding-based similarity.

4. **Data Cleaning Checklist:**
   - Remove incomplete or truncated responses.
   - Filter out toxic/harmful content (unless training for safety).
   - Ensure consistent formatting.
   - Remove personally identifiable information (PII).
   - Validate JSON/structured outputs.
   - Check for data contamination (test set leakage).

5. **Instruction Complexity Distribution:**
   - Include a range from simple to complex instructions.
   - WizardLM's Evol-Instruct methodology: iteratively rewrite instructions to increase complexity.

6. **Data Contamination:** Ensure your training data doesn't include examples from benchmarks you'll evaluate on.

```python
# Example: Data quality filtering pipeline
from datasets import load_dataset
import hashlib

def filter_dataset(dataset):
    seen_hashes = set()
    filtered = []

    for example in dataset:
        # Length filter
        if len(example["output"]) < 10:
            continue
        if len(example["output"]) > 4096:
            continue

        # Deduplication
        text_hash = hashlib.md5(example["output"].encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        # Quality heuristics
        if example["output"].count("\n") > 100:  # Likely garbage
            continue

        filtered.append(example)

    return filtered
```

---

## 4. RLHF -- REINFORCEMENT LEARNING FROM HUMAN FEEDBACK

### 4.1 Overview

RLHF is the process that transformed base LLMs into helpful, harmless, and honest assistants (the "ChatGPT moment"). It aligns model outputs with human preferences.

### 4.2 The Three Stages of RLHF

```
Stage 1: Supervised Fine-Tuning (SFT)
    Pre-trained Model + Instruction Data --> SFT Model

Stage 2: Reward Model Training
    SFT Model generates responses --> Humans rank them --> Train Reward Model

Stage 3: RL Optimization (PPO)
    SFT Model optimized against Reward Model using PPO --> Aligned Model
```

### 4.3 Stage 1: Supervised Fine-Tuning (SFT)

- Standard instruction tuning on high-quality demonstration data.
- Creates a model that can follow instructions but may not consistently produce preferred outputs.
- This model serves as the initialization for RL training AND as the reference model for KL penalty.

### 4.4 Stage 2: Reward Model (RM) Training

**Purpose:** Learn a scalar score function R(prompt, response) that predicts human preferences.

**Data Collection:**
1. Sample multiple responses from the SFT model for the same prompt.
2. Human annotators rank the responses (or choose between pairs).
3. Convert rankings to pairwise comparisons.

**Architecture:**
- Take the SFT model, replace the language modeling head with a scalar output head.
- Input: (prompt, response) --> Output: scalar reward score.

**Loss Function (Bradley-Terry Model):**

```
L(theta) = -E_{(x, y_w, y_l)} [ log( sigma( R_theta(x, y_w) - R_theta(x, y_l) ) ) ]

Where:
- x = prompt
- y_w = preferred (winning) response
- y_l = rejected (losing) response
- sigma = sigmoid function
- R_theta = reward model parameterized by theta
```

The reward model learns to assign higher scores to preferred responses.

```python
# Reward Model Training with TRL
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct", num_labels=1
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

training_args = RewardConfig(
    output_dir="./reward_model",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    max_length=512,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=preference_dataset,  # Must have 'chosen' and 'rejected' columns
)

trainer.train()
```

### 4.5 Stage 3: PPO (Proximal Policy Optimization)

**Purpose:** Optimize the SFT model to maximize the reward model score while staying close to the original SFT model behavior.

**PPO Objective:**

```
maximize  E_{x~D, y~pi_theta(y|x)} [ R_phi(x, y) - beta * KL(pi_theta(y|x) || pi_ref(y|x)) ]

Where:
- pi_theta = current policy (model being optimized)
- pi_ref = reference policy (frozen SFT model)
- R_phi = reward model
- beta = KL penalty coefficient
- KL divergence prevents the model from diverging too far from the SFT model
```

**Why KL Penalty?**
Without it, the model would "hack" the reward model by generating adversarial outputs that score high on the reward model but are actually degenerate (reward hacking).

**PPO Training Loop:**

```
For each batch:
    1. Sample prompts from dataset
    2. Generate responses using current policy (pi_theta)
    3. Score responses with reward model R_phi
    4. Compute KL divergence between pi_theta and pi_ref
    5. Compute PPO loss with clipped surrogate objective
    6. Update policy parameters theta
```

```python
# PPO Training with TRL
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# Load models
model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model-path")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model-path")
tokenizer = AutoTokenizer.from_pretrained("sft-model-path")
reward_model = ...  # Load trained reward model

ppo_config = PPOConfig(
    model_name="sft-model",
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=8,
    ppo_epochs=4,               # PPO update epochs per batch
    kl_penalty="kl",
    init_kl_coef=0.2,          # Beta for KL penalty
    target_kl=6.0,             # Target KL divergence
    gamma=1.0,
    lam=0.95,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop
for batch in dataloader:
    query_tensors = tokenizer(batch["prompt"], return_tensors="pt")

    # Generate responses
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)

    # Get rewards from reward model
    rewards = reward_model(query_tensors, response_tensors)

    # PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

### 4.6 Challenges of RLHF

1. **Complexity:** Three separate models (SFT, Reward, Policy) -- high engineering overhead.
2. **Reward Hacking:** Model exploits reward model weaknesses.
3. **Training Instability:** PPO is notoriously unstable; requires careful hyperparameter tuning.
4. **Reward Model Quality:** Ceiling is limited by reward model accuracy.
5. **Human Annotation Cost:** Expensive and time-consuming to collect preference data.
6. **Reproducibility:** High variance in training outcomes.
7. **Memory:** Need to keep policy, reference, reward model, and value head in memory simultaneously.

---

## 5. DPO -- DIRECT PREFERENCE OPTIMIZATION

### 5.1 Motivation

DPO was introduced to address RLHF's complexity. Key insight: **you can optimize for human preferences directly from preference data without training a separate reward model or using RL.**

### 5.2 Mathematical Formulation

**Key Insight:**
The optimal solution to the RLHF problem (maximize reward under KL constraint) has a closed-form relationship between the reward function and the optimal policy:

```
r(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) ) + beta * log Z(x)

Where Z(x) is the partition function (intractable but cancels out in the loss)
```

By substituting this into the Bradley-Terry preference model, we get the **DPO Loss**:

```
L_DPO(theta) = -E_{(x, y_w, y_l)} [ log sigma(
    beta * log(pi_theta(y_w|x) / pi_ref(y_w|x)) - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x))
)]

Simplified:
L_DPO = -E [ log sigma( beta * (log_ratio_chosen - log_ratio_rejected) ) ]

Where log_ratio = log(pi_theta(y|x)) - log(pi_ref(y|x))
```

### 5.3 How DPO Simplifies RLHF

```
RLHF Pipeline:                    DPO Pipeline:
1. Train SFT model                1. Train SFT model
2. Collect preference data        2. Collect preference data
3. Train reward model             3. Train with DPO loss directly
4. Run PPO training
(4 steps, 3 models in memory)     (2 steps, 2 models in memory)
```

**DPO eliminates:**
- Reward model training
- PPO optimization (and its instability)
- Value head training
- Complex RL infrastructure

### 5.4 DPO Implementation

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset

# Load model and reference
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Optional: Use LoRA for efficiency
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Load preference dataset (must have 'prompt', 'chosen', 'rejected')
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

# DPO training configuration
training_args = DPOConfig(
    output_dir="./dpo_model",
    beta=0.1,                        # Temperature parameter (key hyperparameter)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,              # Lower LR than SFT
    num_train_epochs=1,
    max_length=1024,
    max_prompt_length=512,
    warmup_ratio=0.1,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
)

# Initialize DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,                 # When using LoRA, ref_model=None uses the base model
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    peft_config=peft_config,
)

dpo_trainer.train()
```

### 5.5 DPO Variants (2024-2025)

| Variant | Key Idea |
|---------|----------|
| **IPO (Identity Preference Optimization)** | Removes the sigmoid, addresses overfitting to preference data |
| **KTO (Kahneman-Tversky Optimization)** | Works with binary feedback (good/bad) instead of paired preferences |
| **ORPO (Odds Ratio Preference Optimization)** | Combines SFT and alignment in a single step |
| **SimPO (Simple Preference Optimization)** | Reference-free DPO variant, uses length-normalized reward |
| **CPO (Contrastive Preference Optimization)** | More stable training than DPO |
| **SPIN (Self-Play Fine-Tuning)** | Uses model's own generations as rejected examples |

### 5.6 DPO vs RLHF Comparison

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Complexity | Very High | Low |
| Training Stability | Unstable | Stable |
| Memory (models in GPU) | 3-4 models | 2 models |
| Separate Reward Model | Yes | No |
| Hyperparameter Sensitivity | High | Low-Medium |
| Performance | Gold Standard | Comparable (sometimes better) |
| Online Data Generation | Yes (samples during training) | No (offline, fixed dataset) |
| Reward Hacking Risk | Higher | Lower |
| Industry Adoption (2025) | Declining | Dominant |

---

## 6. QUANTIZATION

### 6.1 What Is Quantization?

Reducing the numerical precision of model weights (and optionally activations) from higher precision (FP32/FP16) to lower precision (INT8/INT4) to reduce memory footprint and increase inference speed.

```
FP32 (32 bits) --> FP16/BF16 (16 bits) --> INT8 (8 bits) --> INT4 (4 bits)
Memory per param: 4 bytes    2 bytes         1 byte         0.5 bytes

7B model sizes:
FP32: 28 GB | FP16: 14 GB | INT8: 7 GB | INT4: 3.5 GB
```

### 6.2 Quantization Types

#### Post-Training Quantization (PTQ)
Quantize after training. No retraining required.
- **Weight-only quantization:** Only quantize weights (activations stay in FP16).
- **Weight + Activation quantization:** Quantize both (harder, more speedup).

#### Quantization-Aware Training (QAT)
Simulate quantization during training. Model learns to be robust to quantization noise. Better quality but requires full training.

### 6.3 Key Quantization Methods

#### INT8 (LLM.int8() -- bitsandbytes)
**Paper:** Dettmers et al., 2022

- Mixed-precision decomposition: identifies outlier features in activations.
- Outlier features (>6 standard deviations) stay in FP16.
- Non-outlier features quantized to INT8.
- **Memory:** ~50% reduction from FP16.
- **Quality:** Minimal degradation.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    load_in_8bit=True,        # INT8 quantization
    device_map="auto"
)
```

#### INT4 / NF4 (bitsandbytes)
- 4-bit quantization using NormalFloat (NF4) data type.
- Optimal for normally distributed weights.
- Used by QLoRA for training.
- **Memory:** ~75% reduction from FP16.
- **Quality:** Small degradation, excellent for fine-tuning with QLoRA.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

#### GPTQ (GPT-Quantization)
**Paper:** Frantar et al., 2022

- **Method:** One-shot weight quantization using approximate second-order information (Hessian-based).
- **Process:**
  1. Quantize weights column by column.
  2. For each column, minimize quantization error by adjusting remaining (not yet quantized) columns using Hessian information.
  3. Requires a calibration dataset (128-256 samples).
- **Formats:** INT4, INT3, INT2 (with grouping).
- **Speed:** Very fast inference via GPU kernels (ExLlama, Marlin).
- **Quality:** Good at INT4, degrades at INT3/INT2.

```python
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    group_size=128,              # Quantize in groups of 128
    dataset="c4",                # Calibration dataset
    desc_act=True,               # Activation order (better quality)
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=gptq_config,
    device_map="auto"
)
```

#### AWQ (Activation-Aware Weight Quantization)
**Paper:** Lin et al., 2023

- **Key Insight:** Not all weights are equally important. Weights corresponding to large activation magnitudes are more critical.
- **Method:**
  1. Identify salient weight channels by looking at activation distributions.
  2. Protect salient channels by scaling them up before quantization (and scaling activations down).
  3. Per-channel scaling factors are determined by activation statistics.
- **Advantages over GPTQ:**
  - Often better quality at the same bit-width.
  - Faster quantization process.
  - Better support for batched inference.
- **Speed:** Excellent with kernels (AutoAWQ, vLLM).

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("llama-3-8b-awq")
```

#### GGUF (GPT-Generated Unified Format)
- **Created by:** llama.cpp project (Georgi Gerganov).
- **Purpose:** CPU-friendly quantization format for running LLMs on consumer hardware.
- **Key Features:**
  - Single-file format containing model weights + metadata + tokenizer.
  - Supports multiple quantization levels: Q2_K, Q3_K, Q4_K_M, Q5_K_M, Q6_K, Q8_0.
  - Optimized for CPU inference with optional GPU offloading.
  - Used by llama.cpp, Ollama, LM Studio, GPT4All.
- **Naming Convention:** `Q{bits}_K_{size}` where K=k-quant method, size=S(small)/M(medium)/L(large).

```
Quantization Level  | Bits | Quality    | Size (7B model)
--------------------|------|------------|----------------
Q2_K                | 2    | Poor       | ~2.5 GB
Q3_K_M              | 3    | Fair       | ~3.3 GB
Q4_K_M              | 4    | Good       | ~4.1 GB
Q5_K_M              | 5    | Very Good  | ~4.8 GB
Q6_K                | 6    | Excellent  | ~5.5 GB
Q8_0                | 8    | Near FP16  | ~7.2 GB
```

### 6.4 When to Quantize

| Scenario | Recommended Method |
|----------|-------------------|
| Fine-tuning on limited GPU | QLoRA (NF4 + LoRA) |
| Fast GPU inference (production) | AWQ or GPTQ (INT4) |
| CPU inference (edge/local) | GGUF (Q4_K_M or Q5_K_M) |
| Minimal quality loss needed | INT8 (bitsandbytes) |
| Maximum compression | GPTQ INT3 or Q2_K (with quality trade-off) |
| vLLM / TGI serving | AWQ or GPTQ |

### 6.5 Quantization Quality Hierarchy

```
FP32 >= BF16 >= FP16 > INT8 > NF4 (QLoRA) > AWQ-INT4 >= GPTQ-INT4 > Q5_K_M > Q4_K_M > Q3_K > Q2_K
(Best quality)                                                                              (Most compressed)
```

---

## 7. TRAINING INFRASTRUCTURE

### 7.1 Distributed Training Strategies

#### Data Parallelism (DP)
- Replicate the entire model on each GPU.
- Split training data across GPUs.
- Each GPU computes gradients independently.
- Gradients are synchronized (all-reduce) after each step.
- **Limitation:** Each GPU must hold the full model.

#### Distributed Data Parallelism (DDP -- PyTorch)
- Improved data parallelism with efficient gradient communication.
- Uses NCCL backend for GPU-to-GPU communication.
- Overlaps communication with computation (bucketed all-reduce).

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])
```

#### Model Parallelism
- **Tensor Parallelism:** Split individual weight matrices across GPUs (column/row splitting).
- **Pipeline Parallelism:** Assign different layers to different GPUs. Process micro-batches in pipeline fashion.

### 7.2 DeepSpeed (Microsoft)

A comprehensive distributed training library with three ZeRO (Zero Redundancy Optimizer) stages:

#### ZeRO Stages

```
ZeRO-1: Partition optimizer states across GPUs
    Memory per GPU: Model + Gradients + (Optimizer States / N)
    Memory savings: ~4x reduction in optimizer memory

ZeRO-2: Partition optimizer states + gradients across GPUs
    Memory per GPU: Model + (Gradients / N) + (Optimizer States / N)
    Memory savings: ~8x reduction

ZeRO-3: Partition optimizer states + gradients + parameters across GPUs
    Memory per GPU: (Model / N) + (Gradients / N) + (Optimizer States / N)
    Memory savings: Linear scaling -- can train models larger than single GPU memory

ZeRO-Infinity: ZeRO-3 + offload to CPU/NVMe
    Can train trillion-parameter models on limited GPUs
```

```json
// DeepSpeed ZeRO-3 config (ds_config.json)
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

```python
# Using DeepSpeed with Hugging Face Trainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",    # DeepSpeed config
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=True,
    # ...
)
```

### 7.3 FSDP (Fully Sharded Data Parallelism -- PyTorch Native)

PyTorch's native alternative to DeepSpeed ZeRO-3.

- **How It Works:**
  - Shards model parameters, gradients, and optimizer states across GPUs.
  - Before a forward/backward pass on a layer, gather the full parameters (all-gather).
  - After computing gradients, scatter them (reduce-scatter).
  - Very similar to ZeRO-3 but integrated into PyTorch.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_offload_params": False,
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
    },
    bf16=True,
)
```

**DeepSpeed ZeRO-3 vs FSDP:**

| Feature | DeepSpeed ZeRO-3 | FSDP |
|---------|-----------------|------|
| Framework | Microsoft library | PyTorch native |
| CPU Offloading | Excellent (ZeRO-Infinity) | Supported |
| NVMe Offloading | Supported | Not supported |
| Ease of Use | Requires config file | More Pythonic |
| Community | Large | Growing |
| Integration | HF, custom | HF, native PyTorch |

### 7.4 Gradient Checkpointing

**Problem:** Storing all intermediate activations for backpropagation consumes enormous memory.

**Solution:** Only store activations at checkpoint boundaries. Recompute intermediate activations during backward pass.

**Trade-off:** ~30% slower training but ~60-70% memory reduction for activations.

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or in TrainingArguments
training_args = TrainingArguments(
    gradient_checkpointing=True,
    # ...
)
```

### 7.5 Mixed Precision Training

#### FP16 (Float16)
- Range: 6.1e-5 to 65504
- Common on older NVIDIA GPUs (V100).
- Risk: overflow/underflow due to limited range.
- Requires loss scaling to prevent underflow.

#### BF16 (BFloat16) -- Preferred in 2025
- Same range as FP32 but with FP16 memory.
- Lower precision than FP16 but much better numerical stability.
- Supported on A100, H100, A10G, and newer GPUs.
- **No loss scaling needed.**

```python
training_args = TrainingArguments(
    bf16=True,                    # Use BF16 mixed precision
    # OR
    fp16=True,                    # Use FP16 mixed precision
    tf32=True,                    # Enable TF32 on Ampere+ GPUs
)
```

#### Flash Attention 2
Not strictly mixed precision, but critical for training efficiency:
- Fuses attention operations to reduce memory I/O.
- Reduces attention memory from O(N^2) to O(N).
- 2-4x speedup on attention computation.

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

### 7.6 GPU Memory Breakdown for Training

For a model with P parameters in fp16:

```
Component                    | Memory
-----------------------------|------------------
Model Weights (fp16)         | 2P bytes
Gradients (fp16)             | 2P bytes
Adam Optimizer States         | 8P bytes (fp32 momentum + fp32 variance + fp32 master copy)
Activations                  | Varies (depends on batch size, sequence length)
-----------------------------|------------------
Total (without activations)  | 12P bytes

Example: 7B model
Weights: 14 GB | Gradients: 14 GB | Optimizer: 56 GB | Total: ~84 GB+

With QLoRA:
Base model (4-bit): 3.5 GB | LoRA params (fp16): ~0.1 GB |
LoRA optimizer (fp32): ~0.4 GB | Activations: ~2 GB | Total: ~6 GB
```

---

## 8. EVALUATION OF FINE-TUNED MODELS

### 8.1 Automated Benchmarks

| Benchmark | What It Measures | Format |
|-----------|-----------------|--------|
| **MMLU** | Massive Multitask Language Understanding (57 tasks) | Multiple choice |
| **HellaSwag** | Common sense reasoning, sentence completion | Multiple choice |
| **ARC** | AI2 Reasoning Challenge (grade-school science) | Multiple choice |
| **TruthfulQA** | Factual accuracy, resistance to misconceptions | Multiple choice / generation |
| **Winogrande** | Common sense reasoning (pronoun resolution) | Binary choice |
| **GSM8K** | Grade school math word problems | Generation + exact match |
| **HumanEval** | Python code generation | Pass@k |
| **MBPP** | Mostly Basic Python Problems | Pass@k |
| **MT-Bench** | Multi-turn conversation quality (GPT-4 judge) | LLM-as-judge scoring |
| **AlpacaEval** | Instruction following (GPT-4 judge) | Win-rate vs reference |
| **IFEval** | Instruction following (format compliance) | Exact format matching |
| **BBH** (Big Bench Hard) | Challenging reasoning tasks | Generation |
| **MATH** | Competition math problems | Exact match |

### 8.2 LLM-as-a-Judge Evaluation

Using a stronger model (GPT-4, Claude) to evaluate outputs:

```python
# Example: MT-Bench style evaluation
evaluation_prompt = """
Please act as an impartial judge and evaluate the quality of the response
provided by an AI assistant to the user question displayed below.

[Question]
{question}

[Assistant's Answer]
{answer}

Rate the response on a scale of 1-10, considering:
- Helpfulness
- Accuracy
- Depth
- Creativity
- Level of detail

Score:
Explanation:
"""
```

**Pros:** Scalable, consistent, correlates with human preferences (~80%+ agreement).
**Cons:** Judge model bias, cost, cannot catch subtle errors.

### 8.3 Human Evaluation

**Methods:**
- **Side-by-side comparison (A/B testing):** Show outputs from two models, human picks winner.
- **Likert scale rating:** Rate individual outputs on 1-5 scale for specific criteria.
- **Elo rating:** Tournament-style comparison across multiple models (used by Chatbot Arena).

**Criteria:**
- Helpfulness, Harmlessness, Honesty (HHH framework by Anthropic)
- Accuracy, Relevance, Completeness, Coherence, Fluency

### 8.4 Overfitting Detection

**Symptoms of Overfitting:**
1. Training loss continues decreasing, validation loss increases.
2. Model outputs become repetitive or memorize training examples.
3. Performance on benchmarks degrades compared to base model.
4. Model becomes overly verbose or uses specific phrases from training data excessively.

**Prevention and Detection:**

```python
from transformers import TrainingArguments, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=3,        # Keep low (1-3 epochs for LLMs)
    warmup_ratio=0.1,
    weight_decay=0.01,
    # ...
)

# Add early stopping
from transformers import EarlyStoppingCallback
trainer = Trainer(
    # ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

**Key Rules for LLM Fine-Tuning:**
- Use **1-3 epochs** (unlike CV models that train for hundreds).
- Monitor eval loss religiously.
- Use a held-out validation set (10-20%).
- After training, run benchmark comparisons against the base model.
- Check for "mode collapse" (reduced output diversity).

### 8.5 Evaluation Framework: lm-evaluation-harness

```bash
# Eleuther AI's lm-evaluation-harness
pip install lm-eval

# Run evaluation
lm_eval --model hf \
    --model_args pretrained=./my-finetuned-model \
    --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2,winogrande,gsm8k \
    --batch_size 8 \
    --num_fewshot 5 \
    --output_path ./eval_results
```

---

## 9. TOOLS AND FRAMEWORKS

### 9.1 Hugging Face Transformers + PEFT

The foundational ecosystem for LLM fine-tuning.

```python
# Complete SFT Pipeline with LoRA
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 1. Load model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# 3. Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 5. Format dataset
def format_instruction(example):
    if example.get("input"):
        text = f"""### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"""
    else:
        text = f"""### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"""
    return {"text": text}

dataset = dataset.map(format_instruction)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    group_by_length=True,
)

# 7. Initialize SFT Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    dataset_text_field="text",
)

# 8. Train
trainer.train()

# 9. Save LoRA adapter
trainer.save_model("./llama3-lora-adapter")

# 10. Merge and save full model (optional)
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
merged_model = PeftModel.from_pretrained(base_model, "./llama3-lora-adapter")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./llama3-merged")
```

### 9.2 TRL (Transformer Reinforcement Learning)

HuggingFace library for alignment training.

**Key Trainers:**

| Trainer | Purpose |
|---------|---------|
| `SFTTrainer` | Supervised fine-tuning |
| `DPOTrainer` | Direct Preference Optimization |
| `PPOTrainer` | Proximal Policy Optimization (RLHF) |
| `KTOTrainer` | Kahneman-Tversky Optimization |
| `ORPOTrainer` | Odds Ratio Preference Optimization |
| `RewardTrainer` | Reward model training |
| `CPOTrainer` | Contrastive Preference Optimization |

### 9.3 Axolotl

A high-level, configuration-driven fine-tuning framework.

```yaml
# axolotl config.yml
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
adapter: qlora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 4096
sample_packing: true           # Pack multiple samples into one sequence
pad_to_sequence_len: true

micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 1
learning_rate: 2e-4
optimizer: paged_adamw_8bit
lr_scheduler: cosine
warmup_ratio: 0.1

bf16: auto
flash_attention: true
gradient_checkpointing: true

val_set_size: 0.05
eval_steps: 100
save_steps: 100
logging_steps: 10

output_dir: ./output
```

```bash
# Run training
accelerate launch -m axolotl.cli.train config.yml
```

**Key Features:**
- YAML-based configuration (no code needed).
- Supports: LoRA, QLoRA, full fine-tuning, GPTQ-LoRA.
- Sample packing for efficient training.
- Multi-dataset mixing.
- Built-in support for many chat formats.
- Very popular in the open-source fine-tuning community.

### 9.4 Unsloth

**Purpose:** 2x faster and 60% less memory for LoRA/QLoRA fine-tuning.

**How:** Custom CUDA kernels for key operations, fused optimizers, manual autograd.

```python
from unsloth import FastLanguageModel

# Load model with Unsloth (2x faster, 60% less memory)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
)

# Use with HuggingFace SFTTrainer as normal
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # ... same config as before
)
trainer.train()

# Save and export
model.save_pretrained_gguf("model-gguf", tokenizer, quantization_method="q4_k_m")
model.save_pretrained_merged("model-merged", tokenizer, save_method="merged_16bit")
```

**Key Features:**
- 2x faster training, 60% less memory.
- Free and open source (Apache 2.0).
- Direct GGUF export.
- Supports LLaMA, Mistral, Phi, Gemma, Qwen, and more.
- Drop-in replacement for HuggingFace.

### 9.5 LitGPT (Lightning AI)

```python
# LitGPT fine-tuning via CLI
# litgpt finetune --help

# LoRA fine-tuning
litgpt finetune lora \
    --checkpoint_dir meta-llama/Llama-3-8B-Instruct \
    --data Alpaca \
    --train.epochs 1 \
    --train.lr 2e-4
```

### 9.6 OpenAI Fine-Tuning API

```python
from openai import OpenAI
client = OpenAI()

# 1. Prepare data in JSONL format
# Each line: {"messages": [{"role": "system", "content": "..."},
#                           {"role": "user", "content": "..."},
#                           {"role": "assistant", "content": "..."}]}

# 2. Upload training file
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# 3. Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",   # Base model
    hyperparameters={
        "n_epochs": 3,
        "batch_size": "auto",
        "learning_rate_multiplier": "auto"
    },
    suffix="my-custom-model"            # Custom model name suffix
)

# 4. Monitor training
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id)

# 5. Use fine-tuned model
completion = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:org::job-id",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**OpenAI Fine-Tuning Specifics:**
- Models available: GPT-4o, GPT-4o-mini, GPT-3.5-turbo.
- Minimum 10 examples (recommended 50-100+).
- Supports function calling fine-tuning.
- Automatic hyperparameter selection.
- Cannot access model weights (black box).
- Priced per training token.

---

## 10. COMMON INTERVIEW QUESTIONS WITH DETAILED ANSWERS

### Q1: Explain the difference between LoRA and full fine-tuning. When would you choose each?

**Answer:**

Full fine-tuning updates all model parameters (100%), requiring memory for weights, gradients, and optimizer states (~12x model size for Adam in fp16). LoRA freezes the original weights and adds low-rank decomposition matrices (A and B) to selected layers, typically updating only 0.1-1% of parameters.

**Mathematical difference:**
- Full: W_new = W_old + gradient_updates (all params updated)
- LoRA: W_new = W_frozen + B*A (only B and A are learned, rank r << d)

**Choose full fine-tuning when:**
- You have ample compute (multiple A100/H100 GPUs)
- Maximum performance is critical
- The domain shift is very large (e.g., English model to medical Japanese)
- You're creating a new base model variant

**Choose LoRA/QLoRA when:**
- Limited GPU resources (single GPU)
- You want to serve multiple task-specific adapters on one base model
- You want to minimize catastrophic forgetting
- Most practical production scenarios (this is the default in 2025)

**In practice:** QLoRA on a single 24GB GPU (3090/4090) can match 90-99% of full fine-tuning performance.

---

### Q2: How does QLoRA reduce memory requirements? Walk through the math.

**Answer:**

QLoRA combines three innovations to reduce memory:

**1. NF4 Quantization:** Base model weights stored in 4-bit NormalFloat format instead of 16-bit.
- 7B model: 14GB (fp16) --> 3.5GB (4-bit)

**2. Double Quantization:** The quantization constants (scales and zero-points, one per block of 64 weights) are themselves quantized from FP32 to FP8.
- Saves ~0.37 bits/param = ~0.33GB for 7B model

**3. Paged Optimizers:** When GPU memory fills up during gradient checkpointing spikes, optimizer states are automatically paged to CPU RAM using CUDA unified memory.

**Memory breakdown for 7B model:**
```
Full fine-tuning (fp16):
  Weights: 14GB + Gradients: 14GB + Adam states: 56GB = ~84GB

QLoRA:
  Base weights (NF4): 3.5GB
  LoRA weights (fp16): ~0.1GB (rank 16, all attention + MLP)
  LoRA gradients: ~0.1GB
  LoRA optimizer states: ~0.4GB
  Activations + overhead: ~2GB
  Total: ~6GB <-- fits on a single consumer GPU
```

---

### Q3: Explain DPO and how it differs from RLHF. Write the loss function.

**Answer:**

**RLHF** is a 3-stage process: (1) SFT, (2) train a reward model on preference data, (3) use PPO to optimize the policy against the reward model with a KL constraint.

**DPO** collapses stages 2 and 3 into one. The key mathematical insight is that the optimal policy under the RLHF objective has a closed-form solution:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y) / beta)
```

Rearranging: `r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta*log(Z(x))`

Substituting into Bradley-Terry preference model, the partition function Z(x) cancels:

```python
# DPO Loss (PyTorch pseudocode)
def dpo_loss(pi_logprobs_chosen, pi_logprobs_rejected,
             ref_logprobs_chosen, ref_logprobs_rejected, beta=0.1):

    log_ratio_chosen = pi_logprobs_chosen - ref_logprobs_chosen
    log_ratio_rejected = pi_logprobs_rejected - ref_logprobs_rejected

    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -F.logsigmoid(logits).mean()

    return loss
```

**Key differences:**
| RLHF | DPO |
|------|-----|
| Needs reward model | No reward model |
| Uses PPO (unstable) | Simple cross-entropy-like loss (stable) |
| Online (generates during training) | Offline (fixed dataset) |
| 3-4 models in memory | 2 models in memory |
| Higher engineering complexity | Simple implementation |

**Beta parameter:** Controls how much the model can deviate from the reference. Lower beta = closer to reference (more conservative). Higher beta = more divergence allowed.

---

### Q4: What is the training pipeline for building a ChatGPT-like model from scratch?

**Answer:**

```
Stage 1: Pre-Training (Unsupervised)
  - Data: Trillions of tokens from web, books, code
  - Objective: Next token prediction (causal LM)
  - Compute: Thousands of GPUs, weeks-months
  - Result: Base model that can complete text but doesn't follow instructions

Stage 2: Supervised Fine-Tuning (SFT)
  - Data: 10K-1M (instruction, response) pairs
  - Objective: Next token prediction on instruction-response pairs
  - Compute: 8-64 GPUs, hours-days
  - Result: Instruction-following model

Stage 3: Alignment (RLHF or DPO)
  - Data: Preference pairs (chosen, rejected) for the same prompts
  - Objective: Maximize reward / optimize preferences
  - Compute: 8-64 GPUs, hours-days
  - Result: Aligned model (helpful, harmless, honest)

Optional Stage 4: Domain Adaptation
  - Continual pre-training on domain corpus
  - Task-specific fine-tuning
```

---

### Q5: How do you decide between prompt engineering, RAG, and fine-tuning?

**Answer (Decision Framework):**

```
START
  |
  v
Can prompt engineering solve it? --> YES --> Use prompt engineering
  |                                          (cheapest, fastest)
  NO
  |
  v
Does the model lack knowledge? --> YES --> Use RAG
  |                                         (retrieval over your docs)
  NO
  |
  v
Does the model lack behavior/style/format? --> YES --> Fine-tune
  |                                                     (change behavior)
  BOTH
  |
  v
Fine-tune (for behavior) + RAG (for knowledge) --> Hybrid approach
```

**Specific Signals for Each:**

| Signal | Solution |
|--------|----------|
| Model gives correct but poorly formatted answers | Prompt Engineering or Fine-tuning |
| Model lacks domain-specific knowledge | RAG |
| Model needs to consistently output JSON/SQL | Fine-tuning |
| Knowledge changes frequently | RAG (not fine-tuning) |
| Need model to adopt a specific persona/tone | Fine-tuning |
| Model hallucinates facts | RAG (for grounding) |
| Latency is critical (can't afford long prompts) | Fine-tuning |
| Small dataset (< 100 examples) | Prompt Engineering (few-shot) |

---

### Q6: Explain the difference between GPTQ, AWQ, and GGUF. When to use each?

**Answer:**

| Aspect | GPTQ | AWQ | GGUF |
|--------|------|-----|------|
| Method | Hessian-based weight quantization | Activation-aware scaling + quantization | Multiple k-quant methods |
| Target | GPU inference | GPU inference | CPU inference (+ GPU offload) |
| Key Insight | Minimize quantization error using 2nd-order info | Protect salient channels based on activation magnitudes | Flexible multi-level quantization |
| Calibration | Required (128+ samples) | Required | Required |
| Speed | Fast (ExLlama kernels) | Very Fast (Marlin kernels) | Optimized for CPU |
| Ecosystem | AutoGPTQ, vLLM, TGI | AutoAWQ, vLLM, TGI | llama.cpp, Ollama, LM Studio |
| Bit widths | 2, 3, 4, 8 | 4 | 2, 3, 4, 5, 6, 8 |

**When to use:**
- **GPTQ:** GPU serving when you need flexibility in bit-widths.
- **AWQ:** Production GPU serving (best speed/quality trade-off, preferred by vLLM).
- **GGUF:** Local/edge deployment, CPU inference, consumer hardware.
- **bitsandbytes (NF4):** Training with QLoRA.

---

### Q7: What is catastrophic forgetting and how do you prevent it during fine-tuning?

**Answer:**

**Catastrophic forgetting** is when fine-tuning causes the model to lose general capabilities it learned during pre-training. For example, a model fine-tuned on medical data might forget how to write code.

**Prevention strategies:**

1. **Use PEFT (LoRA/QLoRA):** By keeping base weights frozen and only training low-rank adapters, you preserve most original capabilities.

2. **Regularization:**
   - Weight decay (L2 regularization)
   - Low learning rate (1e-5 to 5e-5 for full fine-tuning)
   - KL divergence penalty against the base model

3. **Data mixing:** Include general-purpose data alongside task-specific data.
   ```python
   # Example: Mix 80% domain data + 20% general data
   from datasets import concatenate_datasets
   mixed = concatenate_datasets([domain_data, general_data.select(range(len(domain_data)//4))])
   ```

4. **Short training:** Use 1-3 epochs maximum. LLMs overfit quickly.

5. **Elastic Weight Consolidation (EWC):** Add a penalty for changing weights that are important for previous tasks (less common in LLM era).

6. **Evaluation monitoring:** Track both task-specific and general benchmarks during training.

---

### Q8: Explain DeepSpeed ZeRO stages and when you would use each.

**Answer:**

DeepSpeed ZeRO partitions model states across GPUs to reduce per-GPU memory:

```
Without ZeRO (standard DDP):
  Each GPU holds: Full Weights + Full Gradients + Full Optimizer States
  Memory per GPU for 7B model: ~84 GB

ZeRO Stage 1 (Optimizer State Partitioning):
  Each GPU holds: Full Weights + Full Gradients + (Optimizer States / N)
  N=4 GPUs, 7B model: 14 + 14 + 14 = 42 GB per GPU
  Use when: Optimizer states are the bottleneck

ZeRO Stage 2 (+ Gradient Partitioning):
  Each GPU holds: Full Weights + (Gradients / N) + (Optimizer States / N)
  N=4 GPUs, 7B model: 14 + 3.5 + 3.5 = 21 GB per GPU
  Use when: Default recommendation for most multi-GPU training

ZeRO Stage 3 (+ Parameter Partitioning):
  Each GPU holds: (Weights / N) + (Gradients / N) + (Optimizer States / N)
  N=4 GPUs, 7B model: 3.5 + 3.5 + 3.5 = 10.5 GB per GPU
  Use when: Model doesn't fit on a single GPU, need to train very large models

ZeRO-Infinity (Stage 3 + CPU/NVMe offloading):
  Offloads to CPU RAM and NVMe SSD
  Use when: Extreme memory constraints, very large models
```

**Rule of thumb:** Start with Stage 2. If out of memory, try Stage 3. If still OOM, try ZeRO-Infinity with offloading.

---

### Q9: How do you prepare a dataset for instruction fine-tuning? Walk through the complete pipeline.

**Answer:**

```python
# Complete data preparation pipeline

# Step 1: Collect raw data
# Sources: existing datasets, manual creation, synthetic generation

# Step 2: Convert to standard format
def convert_to_chat_format(examples):
    """Convert to ChatML/messages format"""
    formatted = []
    for ex in examples:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]}
        ]
        formatted.append({"messages": messages})
    return formatted

# Step 3: Quality filtering
def quality_filter(example):
    response = example["messages"][-1]["content"]
    # Length checks
    if len(response) < 20 or len(response) > 8000:
        return False
    # Check for empty/placeholder content
    if response.strip() in ["", "N/A", "I don't know"]:
        return False
    # Language detection (optional)
    # Toxicity filtering (optional)
    return True

# Step 4: Deduplication
from datasketch import MinHash, MinHashLSH

def deduplicate(dataset, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_indices = []
    for i, example in enumerate(dataset):
        text = example["messages"][-1]["content"]
        mh = MinHash(num_perm=128)
        for word in text.split():
            mh.update(word.encode('utf8'))
        if not lsh.query(mh):
            lsh.insert(str(i), mh)
            unique_indices.append(i)
    return dataset.select(unique_indices)

# Step 5: Tokenize and check lengths
def check_token_lengths(dataset, tokenizer, max_length=2048):
    lengths = []
    for example in dataset:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
        tokens = tokenizer(text)
        lengths.append(len(tokens["input_ids"]))
    # Filter examples that exceed max_length
    valid_indices = [i for i, l in enumerate(lengths) if l <= max_length]
    return dataset.select(valid_indices)

# Step 6: Train/validation split
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Step 7: Apply chat template
def apply_template(example, tokenizer):
    example["text"] = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return example
```

---

### Q10: What is sample packing and why is it important?

**Answer:**

**Problem:** LLM training pads all sequences to the same length. Short examples waste compute on padding tokens.

```
Without packing (padded to max_length=2048):
  Sequence 1: [tokens...200 tokens...|PAD PAD PAD ... 1848 padding tokens]
  Sequence 2: [tokens...500 tokens...|PAD PAD PAD ... 1548 padding tokens]
  Sequence 3: [tokens...100 tokens...|PAD PAD PAD ... 1948 padding tokens]
  Efficiency: (200+500+100) / (2048*3) = 13% useful tokens!
```

**Sample packing** concatenates multiple examples into a single sequence:

```
With packing:
  Sequence 1: [Example1...200|Example2...500|Example3...100|Example4...|...]
  Total: ~2048 tokens of useful content per sequence
  Efficiency: ~100% useful tokens!
```

**Implementation considerations:**
- Need attention masking to prevent cross-contamination between packed examples.
- Axolotl and TRL support this natively.
- Can lead to 3-5x training speedup.

```python
# In TRL SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    packing=True,            # Enable sample packing
    max_seq_length=2048,
)
```

---

### Q11: What is the difference between SFT, RLHF, DPO, KTO, and ORPO?

**Answer:**

| Method | Training Signal | Data Format | Key Idea |
|--------|----------------|-------------|----------|
| **SFT** | Demonstration | (prompt, response) pairs | Learn to imitate good responses |
| **RLHF** | Human preferences | (prompt, chosen, rejected) | Train reward model, then optimize with PPO |
| **DPO** | Human preferences | (prompt, chosen, rejected) | Directly optimize preferences without reward model |
| **KTO** | Binary feedback | (prompt, response, good/bad label) | Works with just thumbs up/down (no pairs needed) |
| **ORPO** | Combined SFT + preferences | (prompt, chosen, rejected) | Single training stage combining SFT and alignment |
| **SimPO** | Preferences | (prompt, chosen, rejected) | Reference-free DPO with length normalization |

**KTO advantage:** In practice, it's much easier to collect binary feedback ("was this response good?") than paired preferences ("which of these two is better?").

**ORPO advantage:** Combines SFT and alignment into one step, reducing training from 2+ stages to 1.

---

### Q12: You fine-tuned a model and it's worse than the base model. What went wrong?

**Answer (Debugging Checklist):**

1. **Data Issues (Most Common):**
   - Low-quality training data
   - Data contamination or mislabeled examples
   - Insufficient data diversity
   - Wrong chat template / tokenizer formatting

2. **Hyperparameter Issues:**
   - Learning rate too high (catastrophic forgetting) or too low (no learning)
   - Too many epochs (overfitting)
   - Batch size too small (noisy gradients)

3. **Evaluation Issues:**
   - Wrong chat template during evaluation
   - Wrong generation parameters (temperature, top_p)
   - Evaluating with wrong tokenizer

4. **Technical Issues:**
   - LoRA targets wrong modules
   - Gradient checkpointing issues with some architectures
   - Padding side issues (should be "right" for training, "left" for generation)
   - Special tokens not properly handled

```python
# Common debugging code
# 1. Check training loss curve
# - Should decrease smoothly
# - If flat: learning rate too low, LoRA not applied correctly
# - If spikes: learning rate too high, data issues

# 2. Verify chat template
print(tokenizer.apply_chat_template(messages, tokenize=False))
# Make sure it matches what the model expects

# 3. Check LoRA is applied
model.print_trainable_parameters()
# Should show ~0.1-1% trainable params

# 4. Verify padding
assert tokenizer.padding_side == "right"  # For training
assert tokenizer.pad_token is not None

# 5. Test generation with base model first
# Ensure your evaluation pipeline works with the base model
```

---

### Q13: Explain Flash Attention and why it's important for fine-tuning.

**Answer:**

**Standard Attention:** O(N^2) memory and compute for sequence length N.
```
Q, K, V each: (batch, heads, seq_len, head_dim)
Attention matrix: (batch, heads, seq_len, seq_len) <-- O(N^2) memory!
For seq_len=4096: attention matrix = 4096*4096*4 bytes * heads = massive
```

**Flash Attention:** Computes exact attention without materializing the full N x N matrix.

**Key Insight:** Uses tiling and recomputation. Processes attention in blocks, keeping data in fast SRAM (on-chip memory) rather than slow HBM (GPU DRAM).

```
Standard Attention Memory: O(N^2)
Flash Attention Memory: O(N) -- only stores output, not full attention matrix

For seq_len=4096, 32 heads, fp16:
Standard: 4096 * 4096 * 32 * 2 bytes = 1 GB just for attention
Flash Attention: ~0 extra memory (computed in blocks)
```

**Benefits:**
- 2-4x faster attention computation
- O(N) memory instead of O(N^2)
- Enables much longer sequence lengths
- Exact computation (not an approximation)

```python
# Enable Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",  # Just one line
    torch_dtype=torch.bfloat16,
)
```

---

### Q14: How would you fine-tune a model for structured output (JSON)?

**Answer:**

```python
# Approach 1: SFT on structured output examples
training_data = [
    {
        "messages": [
            {"role": "system", "content": "You always respond in valid JSON format."},
            {"role": "user", "content": "Extract entities from: 'John works at Google in NYC'"},
            {"role": "assistant", "content": '{"entities": [{"name": "John", "type": "PERSON"}, {"name": "Google", "type": "ORG"}, {"name": "NYC", "type": "LOCATION"}]}'}
        ]
    },
    # ... hundreds more examples
]

# Approach 2: Use constrained decoding during inference
# (complement fine-tuning with grammar constraints)
# Libraries: guidance, outlines, jsonformer

# Approach 3: Fine-tune with schema in system prompt
training_data = [
    {
        "messages": [
            {"role": "system", "content": """You extract entities from text.
Response schema:
{"entities": [{"name": str, "type": "PERSON"|"ORG"|"LOCATION"}]}"""},
            {"role": "user", "content": "John works at Google in NYC"},
            {"role": "assistant", "content": '{"entities": [{"name": "John", "type": "PERSON"}, {"name": "Google", "type": "ORG"}, {"name": "NYC", "type": "LOCATION"}]}'}
        ]
    }
]

# Key tips:
# 1. Include diverse examples with varying complexity
# 2. Include edge cases (empty arrays, null values, nested objects)
# 3. Use consistent formatting (indented vs. compact JSON)
# 4. Validate all training data JSON is parseable
# 5. Consider constrained decoding at inference time as a safety net
```

---

### Q15: Explain the concept of "merging" LoRA adapters. What are the trade-offs?

**Answer:**

**Merging:** Combining the LoRA adapter weights back into the base model weights.

```python
# Mathematical merge operation:
# W_merged = W_base + (alpha/r) * B @ A

# Code:
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("base-model")
peft_model = PeftModel.from_pretrained(base_model, "lora-adapter-path")

# Merge adapter into base weights
merged_model = peft_model.merge_and_unload()

# Save the full merged model
merged_model.save_pretrained("merged-model-path")
```

**Advantages of merging:**
- Zero inference latency overhead (no adapter computation).
- Simpler deployment (single model file).
- Can be quantized normally (GPTQ, AWQ, GGUF).

**Advantages of NOT merging (keeping adapters separate):**
- Hot-swap adapters at runtime for different tasks.
- Smaller storage (share one base model + small adapter files).
- Combine multiple adapters (multi-LoRA serving with frameworks like LoRAX, S-LoRA).
- vLLM supports multi-LoRA serving with a single base model.

**LoRA Adapter Composition:**
```python
# Stack multiple adapters (additive)
# W_final = W_base + B1@A1 + B2@A2

# Or weighted merging
# W_final = W_base + w1*(B1@A1) + w2*(B2@A2)
```

---

### Q16: What are the key hyperparameters for LLM fine-tuning and how do you tune them?

**Answer:**

| Hyperparameter | Typical Range | Notes |
|---------------|---------------|-------|
| **Learning Rate** | 1e-5 to 3e-4 (LoRA), 1e-6 to 5e-5 (full FT) | Most important HP. Start with 2e-4 for LoRA, 2e-5 for full FT |
| **Epochs** | 1-3 | LLMs overfit fast. 1 epoch is often enough |
| **Batch Size (effective)** | 32-128 | Use gradient accumulation to achieve larger batch sizes |
| **LoRA Rank (r)** | 8, 16, 32, 64 | Higher = more capacity. 16 is a good default |
| **LoRA Alpha** | 2*r | Scaling factor. alpha/r ratio matters, not absolute value |
| **LoRA Dropout** | 0.0 - 0.1 | Regularization. 0.05 is common |
| **Warmup** | 5-10% of steps | Prevents early divergence |
| **Weight Decay** | 0.0 - 0.1 | L2 regularization. 0.01 is common |
| **Max Sequence Length** | 512-8192 | Depends on data. Longer = more memory |
| **LR Scheduler** | Cosine | Most popular for LLM fine-tuning |
| **Optimizer** | AdamW, paged_adamw_8bit | 8bit saves memory with minimal quality loss |

**Tuning Strategy:**
1. Start with community-recommended defaults.
2. First tune learning rate (most impactful).
3. Then tune LoRA rank if using PEFT.
4. Monitor validation loss to detect overfitting.
5. Keep training short (1-3 epochs).

---

### Q17: How does model merging work (e.g., TIES, DARE, Model Soups)?

**Answer:**

Model merging combines weights from multiple fine-tuned models WITHOUT additional training.

**Methods:**

1. **Linear Merging (Model Soups):**
   ```
   W_merged = alpha * W_model1 + (1-alpha) * W_model2
   ```

2. **SLERP (Spherical Linear Interpolation):**
   - Interpolates on a hypersphere for better quality than linear.

3. **TIES-Merging:**
   - Trim small-magnitude deltas (noise reduction).
   - Resolve sign conflicts (majority vote).
   - Scale remaining deltas.

4. **DARE (Drop And REscale):**
   - Randomly drop a fraction of delta parameters.
   - Rescale remaining to preserve magnitude.

```python
# Using mergekit (most popular merging library)
# merge_config.yml
models:
  - model: model_a
    parameters:
      weight: 0.5
  - model: model_b
    parameters:
      weight: 0.5
merge_method: ties
base_model: base_model
parameters:
  density: 0.5
  normalize: true
dtype: bfloat16
```

```bash
mergekit-yaml merge_config.yml ./merged-output
```

---

### Q18: What is Reward Hacking and how do you mitigate it?

**Answer:**

**Reward Hacking** occurs during RLHF when the policy model learns to exploit weaknesses in the reward model to achieve high reward scores without actually producing better outputs.

**Examples:**
- Generating excessively long responses (if RM rewards length).
- Using flattery or agreeable language (if RM rewards agreeableness).
- Generating repetitive high-scoring phrases.
- Producing outputs that are adversarial to the reward model.

**Mitigation Strategies:**

1. **KL Divergence Penalty:** Constrain the policy to stay close to the SFT model.
   ```
   objective = reward - beta * KL(pi_theta || pi_ref)
   ```

2. **Reward Model Ensemble:** Use multiple reward models and average their scores.

3. **Length Normalization:** Normalize rewards by response length.

4. **Reward Model Calibration:** Regularly update the reward model with new data.

5. **Use DPO Instead:** DPO is inherently less susceptible to reward hacking since there's no explicit reward model to exploit.

6. **Constitutional AI:** Use AI feedback with principles to catch degenerate outputs.

---

### Q19: Explain the concept of "Mixture of Experts" (MoE) and its implications for fine-tuning.

**Answer:**

**MoE Architecture:**
- Instead of one large feed-forward network, use N "expert" FFN networks.
- A gating/router network selects top-k experts for each token.
- Only the selected experts are activated (sparse computation).

```
Standard FFN:     Input --> FFN --> Output           (all params active)
MoE:              Input --> Router --> Top-K Experts --> Output  (only K of N active)

Example: Mixtral 8x7B
- 8 experts, 2 active per token
- Total params: ~47B, Active params per token: ~13B
- Performance of a ~47B model at the cost of a ~13B model
```

**Fine-tuning Implications:**
- Full fine-tuning requires loading ALL expert weights (high memory).
- LoRA can be applied to all experts or only active experts.
- Need to consider load balancing loss to prevent expert collapse.
- QLoRA works well for MoE models (quantize all experts, LoRA on active ones).

---

### Q20: How do you serve and deploy a fine-tuned model in production?

**Answer:**

**Serving Options (2025):**

| Framework | Key Features | Best For |
|-----------|-------------|----------|
| **vLLM** | PagedAttention, continuous batching, multi-LoRA | High-throughput GPU serving |
| **TGI (HuggingFace)** | Production-ready, easy deployment | Quick deployment, HF ecosystem |
| **Ollama** | Local deployment, GGUF support | Local/edge, development |
| **TensorRT-LLM** | NVIDIA optimization, fastest | Maximum performance on NVIDIA GPUs |
| **llama.cpp** | CPU inference, GGUF | Edge deployment, CPU only |
| **SGLang** | RadixAttention, fast structured output | Complex prompting, structured output |

```python
# vLLM deployment example
from vllm import LLM, SamplingParams

# Load merged model
llm = LLM(
    model="./merged-model",
    tensor_parallel_size=2,      # Use 2 GPUs
    max_model_len=4096,
    quantization="awq",          # Serve AWQ quantized model
)

# Or serve with LoRA adapters
llm = LLM(
    model="base-model",
    enable_lora=True,
    max_lora_rank=64,
)

# Serve with specific adapter
outputs = llm.generate(
    prompts,
    SamplingParams(temperature=0.7, max_tokens=512),
    lora_request=LoRARequest("adapter1", 1, "./lora-adapter-path")
)
```

```bash
# vLLM as OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model ./merged-model \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --port 8000
```

---

### BONUS Q21: What is Continual Pre-Training vs Fine-Tuning?

**Answer:**

| Aspect | Continual Pre-Training | Supervised Fine-Tuning |
|--------|----------------------|----------------------|
| **Objective** | Next token prediction (unsupervised) | Next token prediction on instruction pairs |
| **Data** | Raw domain text (books, papers, code) | (instruction, response) pairs |
| **Purpose** | Inject new domain knowledge | Teach new behavior/skills |
| **Data Size** | Millions to billions of tokens | Thousands to millions of examples |
| **Learning Rate** | Lower (~1e-5) | Higher (~1e-4 to 3e-4 for LoRA) |
| **Use Case** | "Make the model know about medicine" | "Make the model answer medical questions" |

**Typical Pipeline for Domain Specialization:**
```
Base Model
  --> Continual Pre-Training on domain corpus
    --> SFT on domain instruction data
      --> DPO/RLHF on domain preferences
        --> Production Model
```

---

### BONUS Q22: What is the role of the "system prompt" during fine-tuning?

**Answer:**

During fine-tuning, system prompts serve to:

1. **Define behavior:** "You are a medical assistant that only provides evidence-based advice."
2. **Set constraints:** "Always respond in JSON format."
3. **Control persona:** "You are a friendly tutor who explains concepts simply."

**Best Practices:**
- Include system prompts in your training data if you'll use them in production.
- Vary system prompts during training to prevent overfitting to one prompt.
- If you don't include system prompts during training, the model may ignore them at inference.
- Some models (LLaMA-3) are pre-trained with specific system prompt handling -- respect this format.

```python
# Training data with system prompt
{"messages": [
    {"role": "system", "content": "You are a SQL expert. Convert natural language to SQL."},
    {"role": "user", "content": "Show me all users who signed up last month"},
    {"role": "assistant", "content": "SELECT * FROM users WHERE signup_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND signup_date < DATE_TRUNC('month', CURRENT_DATE);"}
]}
```

---

## QUICK REFERENCE: KEY FORMULAS

```
LoRA:           W_new = W_frozen + (alpha/r) * B @ A
                B: (d, r), A: (r, d), r << d

QLoRA Memory:   ~(P * 0.5) + LoRA_params * 16 bytes (optimizer)

DPO Loss:       L = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x))
                                        - log(pi(y_l|x)/pi_ref(y_l|x)))))

RLHF Objective: max E[R(x,y) - beta * KL(pi_theta || pi_ref)]

PPO Clip:       L = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
                where r_t = pi_theta(a|s) / pi_old(a|s)

Bradley-Terry:  P(y_w > y_l) = sigmoid(R(x, y_w) - R(x, y_l))

KL Divergence:  KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))
```

---

## QUICK REFERENCE: COMMON TRAINING RECIPES

### Recipe 1: QLoRA Fine-Tuning (Single GPU, 24GB)
```
Model: LLaMA-3-8B | Quantization: NF4 | LoRA rank: 16 | Alpha: 32
LR: 2e-4 | Epochs: 1-3 | Batch: 4 (x4 accum = 16 effective)
Optimizer: paged_adamw_8bit | Scheduler: cosine | Warmup: 10%
Memory: ~6-8 GB | Time: ~2-4 hours for 10K examples
```

### Recipe 2: Full Fine-Tuning with DeepSpeed (8x A100)
```
Model: LLaMA-3-70B | DeepSpeed ZeRO-3 | BF16
LR: 2e-5 | Epochs: 1-2 | Batch: 128 effective
Optimizer: AdamW | Scheduler: cosine | Warmup: 5%
Memory: ~40 GB per GPU | Time: ~12-24 hours for 100K examples
```

### Recipe 3: DPO Alignment (After SFT)
```
Model: SFT-trained model | LoRA rank: 16 | Beta: 0.1
LR: 5e-7 (very low!) | Epochs: 1 | Batch: 8 effective
Note: DPO learning rate should be MUCH lower than SFT
Memory: ~8-12 GB | Data: 5K-50K preference pairs
```

---

*This guide covers the essential topics for AI Engineer interviews focused on LLM fine-tuning as of 2025-2026. The field evolves rapidly -- always check for the latest research papers and library updates.*
