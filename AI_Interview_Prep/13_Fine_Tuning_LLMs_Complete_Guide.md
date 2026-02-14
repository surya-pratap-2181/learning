---
title: "Fine-Tuning LLMs"
layout: default
parent: "LLM & AI Fundamentals"
nav_order: 2
---

# Fine-Tuning LLMs: Complete Interview Guide (2025-2026)
## LoRA, QLoRA, PEFT, RLHF, and Production Fine-Tuning

---

# TABLE OF CONTENTS
1. What is Fine-Tuning?
2. Explaining to a Layman
3. Pre-training vs Fine-tuning vs RAG vs Prompt Engineering
4. Full Fine-Tuning
5. LoRA (Low-Rank Adaptation)
6. QLoRA (Quantized LoRA)
7. Other PEFT Methods
8. Instruction Tuning
9. RLHF & DPO
10. Dataset Preparation
11. Training Infrastructure
12. Evaluation & Benchmarks
13. Common Pitfalls
14. Interview Questions (25+)
15. Code Examples

---

# SECTION 1: WHAT IS FINE-TUNING?

Fine-tuning adapts a pre-trained LLM to a specific task, domain, or behavior by training it on a curated dataset. It modifies the model's weights (all or some) to specialize its capabilities.

**Why fine-tune?**
- Teach the model a specific output format/style
- Adapt to domain-specific language (medical, legal, financial)
- Improve performance on specific tasks
- Reduce prompt length (behavior is learned, not prompted)
- Distill knowledge from larger models

**The Training Spectrum:**
```
Pre-training          Fine-tuning          Prompt Engineering
(Trillions of tokens) (Thousands-Millions)  (Zero training)
(Months, $$$$$)       (Hours-Days, $$)      (Minutes, $)
(General knowledge)   (Specialized skills)  (Task instructions)
```

> 🔵 **YOUR EXPERIENCE**: Your resume lists Fine-tuning as a professional skill. At MathCo and RavianAI, your work with LLMs gives you practical context for when fine-tuning is and isn't appropriate.

---

# SECTION 2: EXPLAINING TO A LAYMAN

> Imagine you hire a brilliant general-purpose doctor (the pre-trained LLM). They know a lot about everything. Now you want them to become a cardiologist. You don't send them back to medical school (pre-training) - instead, you give them 6 months of specialized cardiology training (fine-tuning). They keep all their general medical knowledge but become much better at heart-related tasks.
>
> LoRA is like giving them a small pocket reference guide instead of rewriting their entire brain - much cheaper and almost as effective.

---

# SECTION 3: DECISION MATRIX

| Dimension | Prompt Engineering | RAG | Fine-Tuning |
|-----------|-------------------|-----|-------------|
| **When** | Always try first | Need external data | Need new behaviors |
| **Cost** | $ (tokens only) | $$ (infra + tokens) | $$$ (GPU compute) |
| **Time** | Minutes | Hours-Days | Days-Weeks |
| **Data needed** | 0 examples | Documents | 1K-100K examples |
| **Knowledge** | In-model only | External docs | Learned from data |
| **Behavior change** | Limited | No | Yes |
| **Best for** | Formatting, simple tasks | QA, search, citations | Style, domain, format |

**Decision Flowchart:**
```
Can prompt engineering solve it?
├── Yes → Use prompt engineering
└── No → Do you need external knowledge?
    ├── Yes → Use RAG
    └── No → Do you need new behaviors/style?
        ├── Yes → Fine-tune
        └── No → Use RAG + better prompts
```

---

# SECTION 4: FULL FINE-TUNING

Updates ALL model parameters. Requires:
- Full model in memory (FP16: 2 bytes per param)
- Optimizer states (Adam: 8 bytes per param)
- Gradients (4 bytes per param)

| Model Size | VRAM Needed (Full FT) | GPU Requirement |
|-----------|----------------------|-----------------|
| 7B | ~112 GB | 2x A100 80GB |
| 13B | ~208 GB | 4x A100 80GB |
| 70B | ~1120 GB | 16x A100 80GB |

**When to use:** Training your own base model, maximum quality needed, have compute budget.

---

# SECTION 5: LoRA (LOW-RANK ADAPTATION)

## 5.1 How LoRA Works

Instead of updating all weights, LoRA adds small trainable low-rank matrices to specific layers:

```
Original weight: W (d × d matrix, e.g., 4096 × 4096 = 16M params)
LoRA: W + ΔW where ΔW = A × B
  A: (d × r) matrix  (e.g., 4096 × 16 = 65K params)
  B: (r × d) matrix  (e.g., 16 × 4096 = 65K params)

Total trainable params: 130K vs 16M (0.8% of original!)
```

```
┌──────────────────────────────────────┐
│          LoRA Architecture            │
│                                       │
│  Input ──→ [Frozen W] ──→ Output     │
│    │                        ↑         │
│    └──→ [A] ──→ [B] ──→ + ┘         │
│         (r×d)  (d×r)                  │
│       Trainable (tiny!)               │
└──────────────────────────────────────┘
```

## 5.2 Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| **r (rank)** | Rank of decomposition | 8-64 (higher = more capacity) |
| **alpha** | Scaling factor | Usually 2×r |
| **target_modules** | Which layers to adapt | q_proj, v_proj (attention layers) |
| **dropout** | LoRA dropout | 0.05-0.1 |

## 5.3 LoRA Code Example

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                           # rank
    lora_alpha=32,                  # scaling (usually 2*r)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # attention layers
    lora_dropout=0.05,
    bias="none"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 3,213,893,632 || trainable%: 0.13%
```

---

# SECTION 6: QLoRA (QUANTIZED LoRA)

QLoRA = LoRA + 4-bit quantization. Dramatically reduces memory:

| Model | Full FT VRAM | LoRA VRAM | QLoRA VRAM |
|-------|-------------|-----------|------------|
| 7B | 112 GB | 16 GB | 6 GB |
| 13B | 208 GB | 28 GB | 10 GB |
| 70B | 1120 GB | 160 GB | 48 GB |

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True     # Double quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA on top
model = get_peft_model(model, lora_config)
# Now fine-tuning a 3B model on a single consumer GPU!
```

---

# SECTION 7: OTHER PEFT METHODS

| Method | How it Works | Trainable Params | Best For |
|--------|-------------|-----------------|---------|
| **LoRA** | Low-rank matrices on attention | 0.1-1% | Most common, versatile |
| **QLoRA** | LoRA + 4-bit quant | 0.1-1% | Memory-constrained |
| **Prefix Tuning** | Trainable prefix tokens | <0.1% | Generation tasks |
| **P-tuning v2** | Trainable soft prompts at each layer | <0.1% | Classification |
| **Adapters** | Small bottleneck layers | 1-5% | Multi-task |
| **IA3** | Learned scaling vectors | <0.01% | Fastest training |

---

# SECTION 8: INSTRUCTION TUNING

Training models to follow instructions. Dataset format:

```json
{"instruction": "Summarize the following text", "input": "Long text here...", "output": "Summary here..."}
{"instruction": "Translate to French", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"}
```

**Chat format (recommended 2025+):**
```json
{"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this: ..."},
    {"role": "assistant", "content": "Here is the summary: ..."}
]}
```

---

# SECTION 9: RLHF & DPO

## RLHF (Reinforcement Learning from Human Feedback)
```
SFT Model → Generate responses → Humans rank → Train Reward Model → RL (PPO) → Aligned Model
```

## DPO (Direct Preference Optimization)
Simpler alternative - no separate reward model needed:
```
Preference pairs: (chosen response, rejected response)
Train directly on preferences using a modified loss function
```

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-7,
    beta=0.1  # KL penalty coefficient
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

---

# SECTION 10: DATASET PREPARATION

**Quality > Quantity.** Key principles:
- Diverse, representative examples
- Clean, consistent formatting
- Balanced across categories
- No data contamination with eval sets

| Dataset Size | Model Size | Expected Outcome |
|-------------|-----------|-----------------|
| 100-1K examples | Any | Formatting/style change |
| 1K-10K examples | 7B-13B | Domain adaptation |
| 10K-100K examples | 7B-70B | Significant behavior change |
| 100K+ examples | Any | Maximum quality |

---

# SECTION 11: TRAINING INFRASTRUCTURE

| Hardware | VRAM | Best For | Cost (cloud) |
|----------|------|----------|-------------|
| RTX 4090 | 24 GB | QLoRA 7B-13B | $0.50/hr |
| A100 40GB | 40 GB | LoRA 7B-13B | $2-3/hr |
| A100 80GB | 80 GB | LoRA 70B, Full FT 7B | $4-6/hr |
| H100 80GB | 80 GB | Full FT 13B+ | $8-12/hr |
| 8x H100 | 640 GB | Full FT 70B | $80-100/hr |

**Distributed Training:**
- **DeepSpeed ZeRO**: Stages 1-3, progressive memory optimization
- **FSDP**: PyTorch's Fully Sharded Data Parallel

---

# SECTION 12: EVALUATION

| Benchmark | What it Tests | Metric |
|-----------|-------------|--------|
| **MMLU** | General knowledge (57 subjects) | Accuracy % |
| **HumanEval** | Code generation | Pass@1 |
| **MT-Bench** | Multi-turn conversation quality | Score 1-10 |
| **GSM8K** | Math reasoning | Accuracy % |
| **TruthfulQA** | Truthfulness | % truthful |
| **Custom eval** | Your specific task | Your metrics |

---

# SECTION 13: COMMON PITFALLS

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **Catastrophic forgetting** | Model loses general capabilities | Lower learning rate, keep some general data |
| **Overfitting** | Great on train, bad on eval | More data, regularization, early stopping |
| **Data contamination** | Artificially high eval scores | Separate train/eval strictly |
| **Wrong format** | Model outputs garbage | Match training format to inference format |
| **Too high learning rate** | Training diverges | Start with 1e-5 to 5e-5 for LoRA |

---

# SECTION 14: INTERVIEW QUESTIONS (25+)

**Q1: What is fine-tuning and when should you use it?**
Fine-tuning adapts a pre-trained LLM by training on task-specific data. Use when: need new behaviors/style, domain adaptation, reducing prompt length. Don't use when: prompt engineering or RAG can solve it.

**Q2: Explain LoRA - how does it work mathematically?**
LoRA decomposes weight updates into low-rank matrices: ΔW = A × B where A is (d×r) and B is (r×d). Only A and B are trained. With r=16 and d=4096, trainable params go from 16M to 130K (0.8%).

**Q3: What is QLoRA and how does it reduce memory?**
QLoRA = LoRA + 4-bit NormalFloat quantization. Quantizes base model to 4-bit (4x memory reduction), applies LoRA adapters in higher precision. Enables fine-tuning 70B models on single GPU.

**Q4: Fine-tuning vs RAG - when to use which?**
RAG for external knowledge, citations, dynamic data. Fine-tuning for behavior change, style, domain language. Often combined: fine-tune for domain + RAG for specific knowledge.

**Q5: What are the key hyperparameters for LoRA?**
r (rank): 8-64, higher = more capacity. alpha: scaling factor, usually 2*r. target_modules: which layers (attention projections). dropout: 0.05-0.1.

**Q6: What is catastrophic forgetting and how do you prevent it?**
Model loses general capabilities after fine-tuning on narrow data. Prevent: lower learning rate, mix general data with task data, use LoRA (preserves base weights), early stopping.

**Q7: Explain RLHF and DPO.**
RLHF: Train reward model on human preferences, use RL (PPO) to optimize. Complex, unstable. DPO: Train directly on preference pairs without reward model. Simpler, more stable, increasingly preferred.

**Q8: How do you prepare a fine-tuning dataset?**
Clean, diverse, consistently formatted. Chat format with system/user/assistant. Balance across categories. Separate train/eval/test. Quality > quantity.

**Q9: What VRAM do you need for different model sizes?**
Full FT: ~14x model params in bytes. LoRA: ~2x model size. QLoRA: ~0.5x model size. 7B QLoRA fits on 24GB GPU.

**Q10: Compare PEFT methods: LoRA, Prefix Tuning, Adapters.**
LoRA: most versatile, 0.1-1% params. Prefix: prepend trainable tokens, <0.1% params. Adapters: bottleneck layers, 1-5% params. LoRA is the industry standard for 2025.

**Q11: How do you evaluate a fine-tuned model?**
Task-specific metrics + general benchmarks (MMLU, HumanEval). MT-Bench for chat quality. Human evaluation for subjective quality. A/B testing vs base model.

**Q12: What is instruction tuning?**
Training on instruction-response pairs to make models follow instructions. Examples: FLAN, Alpaca. Enables zero-shot task performance.

**Q13: How does SFT (Supervised Fine-Tuning) differ from RLHF?**
SFT trains on correct examples (imitation learning). RLHF optimizes for human preferences (can discover better behaviors than the training data). SFT first, then RLHF for alignment.

**Q14: What is the typical fine-tuning pipeline?**
Data preparation → choose base model → configure PEFT → train → evaluate → merge adapters → deploy. Monitor loss, eval metrics, and sample outputs.

**Q15: How do you deploy a LoRA fine-tuned model?**
Option 1: Merge LoRA weights into base model (merge_and_unload()). Option 2: Serve base model + load LoRA at runtime (vLLM supports this). Option 3: Multiple LoRA adapters on one base model (multi-tenant).

**Q16: What is the role of learning rate in fine-tuning?**
Critical parameter. Too high: training diverges/catastrophic forgetting. Too low: slow convergence. Typical: 1e-4 to 2e-4 for LoRA, 1e-5 to 5e-5 for full FT. Use warmup + cosine decay.

**Q17: How do you handle different chat templates?**
Each model has its own chat template (ChatML, Llama format, etc.). Use tokenizer.apply_chat_template(). Mismatched templates cause poor performance.

**Q18: What is model merging and when is it useful?**
Combining weights of multiple fine-tuned models. Methods: linear interpolation, TIES, DARE. Useful for combining domain expertise without multi-task training.

**Q19: Explain DeepSpeed ZeRO stages.**
ZeRO-1: Partition optimizer states. ZeRO-2: + partition gradients. ZeRO-3: + partition parameters. Each stage reduces per-GPU memory.

**Q20: What are synthetic datasets and how are they used?**
Generated by larger models (GPT-4) for fine-tuning smaller models. Examples: Alpaca, WizardLM. Risk: model collapse from training on model outputs.

---

# SECTION 15: CODE EXAMPLES

## Complete QLoRA Fine-Tuning Pipeline

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 1. Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 2. Load model + tokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 3. Prepare for training
model = prepare_model_for_kbit_training(model)

# 4. LoRA config
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. Load dataset
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# 6. Training args
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

# 7. Train
trainer = SFTTrainer(
    model=model, args=training_args, train_dataset=dataset,
    tokenizer=tokenizer, max_seq_length=2048,
)
trainer.train()

# 8. Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

---

## Sources
- [Analytics Vidhya: Fine-tuning Interview Questions](https://www.analyticsvidhya.com/blog/2024/04/fine-tuning-interview-questions-and-answers/)
- [Medium: 17 LoRA/QLoRA Interview Questions](https://medium.com/theultimateinterviewhack/17-must-know-lora-qlora-interview-questions-beginner-to-advanced-with-answers-part-4-1d76bae05f0e)
- [LLM Interview Questions Hub](https://github.com/KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub)
- [Fine-Tuning with LoRA: 2025 Guide](https://amirteymoori.com/fine-tuning-llms-with-lora-a-practical-guide-for-2025/)
