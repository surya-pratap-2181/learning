---
title: "Cost Optimization"
layout: default
parent: "DevOps & Cloud Infrastructure"
nav_order: 8
---

SECTION 6: COST OPTIMIZATION FOR AI SYSTEMS
AI System Design Interview Guide for AI/ML Engineers (2025-2026)
## 

6.1 MODEL SELECTION BASED ON COST

MODEL COST TIERS (2025 pricing per 1M tokens):

```
TIER 1 - PREMIUM (Complex reasoning, high-stakes):
  GPT-4o:              $2.50 input / $10.00 output
  Claude 3.5 Sonnet:   $3.00 input / $15.00 output
  Gemini 1.5 Pro:      $3.50 input / $10.50 output

TIER 2 - BALANCED (Good quality, moderate cost):
  GPT-4o-mini:         $0.15 input / $0.60 output
  Claude 3.5 Haiku:    $0.25 input / $1.25 output
  Gemini 1.5 Flash:    $0.075 input / $0.30 output

TIER 3 - BUDGET (Simple tasks, high volume):
  GPT-3.5 Turbo:       $0.50 input / $1.50 output
  Open source (hosted): ~$0.10 - $0.50 per 1M tokens
  Self-hosted (amortized): $0.02 - $0.20 per 1M tokens
```

MODEL SELECTION STRATEGY:

```python
class ModelRouter:
    """Route requests to appropriate model tier based on task complexity."""

    def __init__(self):
        self.models = {
            "tier1": "gpt-4o",           # Complex reasoning
            "tier2": "gpt-4o-mini",      # General tasks
            "tier3": "gpt-3.5-turbo",    # Simple tasks
        }

    def classify_and_route(self, request):
        complexity = self.estimate_complexity(request)

        if complexity == "high":
            # Complex: multi-step reasoning, code generation, analysis
            return self.models["tier1"]
        elif complexity == "medium":
            # Medium: summarization, Q&A, classification
            return self.models["tier2"]
        else:
            # Simple: formatting, extraction, simple classification
            return self.models["tier3"]

    def estimate_complexity(self, request):
        # Heuristics:
        # - Long input + complex instructions = high
        # - Classification / extraction = low
        # - Creative / reasoning = high
        # Could also use a small classifier model for this
        task_type = request.get("task_type")
        if task_type in ["reasoning", "code_generation", "analysis"]:
            return "high"
        elif task_type in ["summarization", "qa", "translation"]:
            return "medium"
        else:
            return "low"
```

COST IMPACT EXAMPLE:
```
Scenario: 1M requests/day, average 500 input + 200 output tokens

All GPT-4o:
  Input:  1M * 500/1M * $2.50 = $1,250/day
  Output: 1M * 200/1M * $10.00 = $2,000/day
  Total: $3,250/day = $97,500/month

Smart routing (60% tier3, 30% tier2, 10% tier1):
  Tier 1: 100K * (500*$2.50 + 200*$10.00)/1M = $325/day
  Tier 2: 300K * (500*$0.15 + 200*$0.60)/1M = $58.50/day
  Tier 3: 600K * (500*$0.50 + 200*$1.50)/1M = $330/day
  Total: $713.50/day = $21,405/month

SAVINGS: ~78% cost reduction with smart routing
```


## 6.2 CACHING FOR COST REDUCTION


CACHING IMPACT ANALYSIS:

```
Without caching:
  1M requests/day * $0.003/request = $3,000/day

With 30% exact cache hit rate:
  700K LLM calls * $0.003 = $2,100/day
  300K cache hits * $0.0001 (Redis) = $30/day
  Total: $2,130/day -> 29% savings

With 30% exact + 20% semantic cache:
  500K LLM calls * $0.003 = $1,500/day
  300K exact hits * $0.0001 = $30/day
  200K semantic hits * $0.0003 (embedding + vector search) = $60/day
  Total: $1,590/day -> 47% savings
```

CACHE WARMING STRATEGIES:
1. Pre-compute answers for top 1000 frequent queries
2. Cache system prompt KV-cache (vLLM prefix caching)
3. Cache embedding computations for static documents
4. Cache RAG retrieval results for popular queries

CACHE INVALIDATION:
- Time-based TTL (1 hour for dynamic content, 24 hours for static)
- Event-based (new document added -> invalidate related caches)
- Version-based (new model deployed -> flush all caches)


## 6.3 BATCHING FOR COST REDUCTION


A) REQUEST BATCHING:
   ```
   Instead of 100 individual API calls:
     100 calls * overhead_per_call = high cost

   Batch into single call:
     1 call with 100 items = lower cost (shared overhead)
   ```

   OpenAI Batch API: 50% discount for non-real-time batch processing
   ```python
   # OpenAI Batch API example
   batch = client.batches.create(
       input_file_id="file-abc123",  # JSONL file with requests
       endpoint="/v1/chat/completions",
       completion_window="24h"  # 50% discount for 24h window
   )
   ```

B) PROMPT BATCHING:
   ```
   Instead of:
     Call 1: "Classify this text: [text1]"
     Call 2: "Classify this text: [text2]"
     Call 3: "Classify this text: [text3]"

   Batch into:
     Call 1: "Classify these texts:
              1. [text1]
              2. [text2]
              3. [text3]
              Return results as JSON array."

   Saves: ~60% tokens (shared system prompt and instructions)
   ```

C) EMBEDDING BATCHING:
   ```python
   # Instead of one-at-a-time (slow, expensive)
   for text in texts:
       embedding = embed(text)  # One API call each

   # Batch embedding (fast, cheaper)
   embeddings = embed_batch(texts[:2048])  # Up to 2048 in one call
   ```


## 6.4 TOKEN OPTIMIZATION


A) PROMPT COMPRESSION:
   ```
   Before optimization (150 tokens):
   "You are a helpful AI assistant that specializes in customer support.
    Your role is to help customers with their questions and issues.
    Please provide detailed, accurate, and helpful responses.
    Always be polite and professional in your responses."

   After optimization (40 tokens):
   "You're a customer support AI. Be helpful, accurate, polite."

   Savings: 73% fewer system prompt tokens
   At scale (1M requests): saves ~$330/day with GPT-4o
   ```

B) CONTEXT WINDOW OPTIMIZATION:
   ```python
   def optimize_rag_context(documents, max_tokens=2000):
       """Include only the most relevant parts of retrieved documents."""
       optimized = []
       total_tokens = 0

       for doc in documents:
           # Extract only relevant paragraphs
           relevant_paragraphs = extract_relevant(doc, query)

           for para in relevant_paragraphs:
               para_tokens = count_tokens(para)
               if total_tokens + para_tokens <= max_tokens:
                   optimized.append(para)
                   total_tokens += para_tokens
               else:
                   break

       return optimized
   ```

C) OUTPUT TOKEN CONTROL:
   - Set max_tokens appropriately (don't default to 4096)
   - Use structured output (JSON mode) for predictable length
   - Use classification instead of generation where possible
   ```
   Wasteful: "Explain whether this text is positive or negative and why"
   Efficient: "Classify as POSITIVE or NEGATIVE. One word only."
   ```

D) FEW-SHOT OPTIMIZATION:
   ```
   5 few-shot examples = ~500 extra tokens per request
   At 1M requests/day with GPT-4o: $1,250/day just for examples

   Alternatives:
   - Fine-tune: eliminate few-shot examples entirely
   - Dynamic few-shot: select 1-2 most relevant examples
   - Zero-shot with good instructions (often sufficient)
   ```


## 6.5 FALLBACK CHAINS (CASCADE PATTERN)


PATTERN: Try expensive model, fall back to cheaper on failure or for simple queries.

```python
class FallbackChain:
    """Cascade from expensive to cheap models with quality checks."""

    def __init__(self):
        self.chain = [
            {
                "model": "gpt-4o-mini",      # Try cheapest first
                "cost_per_1k": 0.00015,
                "quality_threshold": 0.8,
            },
            {
                "model": "gpt-4o",            # Escalate if needed
                "cost_per_1k": 0.0025,
                "quality_threshold": 0.9,
            },
            {
                "model": "claude-3.5-sonnet",  # Final fallback
                "cost_per_1k": 0.003,
                "quality_threshold": 0.95,
            },
        ]

    async def generate(self, prompt, quality_required=0.8):
        for tier in self.chain:
            try:
                response = await call_model(tier["model"], prompt)

                # Check quality (confidence, format, relevance)
                quality = self.assess_quality(response)

                if quality >= quality_required:
                    return response

                # If quality too low, try next tier
                continue

            except (RateLimitError, TimeoutError):
                # On error, try next tier
                continue

        # All tiers failed
        return self.fallback_response()

    def assess_quality(self, response):
        """Quick quality check without another LLM call."""
        checks = {
            "not_empty": len(response.content) > 0,
            "not_refusal": "I cannot" not in response.content,
            "valid_format": self.check_format(response),
            "reasonable_length": 10 < len(response.content) < 10000,
        }
        return sum(checks.values()) / len(checks)
```

REVERSE CASCADE (Start cheap, escalate):
```
[Request]
    |
[GPT-4o-mini] --> Is response confident and well-formed?
    |                    |
    |               Yes: Return (cheap!)
    |                    |
    No/Low quality       |
    |
[GPT-4o] --> Is response good?
    |              |
    |         Yes: Return
    |              |
    No             |
    |
[Claude 3.5 Sonnet] --> Return (expensive, last resort)
```

COST SAVINGS WITH CASCADE:
```
If 70% of queries handled by tier 3 (cheapest):
  70% * $0.15/1M tokens = $0.105
If 25% escalated to tier 2:
  25% * $2.50/1M tokens = $0.625
If 5% escalated to tier 1:
  5% * $3.00/1M tokens = $0.15

Blended rate: $0.88/1M tokens
vs. all tier 1: $3.00/1M tokens
Savings: 71%
```


## 6.6 INFRASTRUCTURE COST OPTIMIZATION


A) SPOT/PREEMPTIBLE INSTANCES:
   - 60-70% cheaper than on-demand
   - Use for: batch processing, training, non-critical inference
   - Not for: real-time serving (can be interrupted)
   - Strategy: mix spot + on-demand (base on-demand, burst with spot)

B) RESERVED INSTANCES / COMMITTED USE:
   - 30-60% cheaper than on-demand for 1-3 year commitment
   - Use for: predictable baseline GPU usage
   - Calculate: if running > 60% of time, reserved is cheaper

C) RIGHT-SIZING:
   ```
   If model needs 15GB GPU memory:
     A100 80GB: $3.50/hr (53% waste)
     A10G 24GB: $1.50/hr (37.5% waste)
     T4 16GB:   $0.50/hr (6% waste) -- if model fits with quantization

   INT4 quantized model (3.75GB):
     T4 16GB:   $0.50/hr -- best cost-performance
   ```

D) SERVERLESS INFERENCE:
   - Pay only when inference is running
   - No GPU cost during idle time
   - AWS SageMaker Serverless, Google Cloud Run GPU, Modal, Replicate
   - Best for: low/variable traffic, experimentation
   - Trade-off: cold start latency (10-60 seconds)

E) SELF-HOSTED VS API:
   ```
   Break-even analysis:

   API (GPT-4o-mini): $0.60/1M output tokens
   Self-hosted (Llama 3 8B on A10G):
     GPU cost: $1.50/hr
     Throughput: ~3000 tokens/sec = 10.8M tokens/hr
     Cost: $1.50 / 10.8M = $0.14/1M tokens

   Self-hosted is cheaper if:
     - Volume > 18M tokens/hr (to justify always-on GPU)
     - Can accept open-source model quality
     - Have team to manage infrastructure
   ```

## END OF SECTION 6

