---
title: "AI System Design Patterns"
layout: default
parent: "System Design & Architecture"
nav_order: 1
---

SECTION 1: AI SYSTEM DESIGN PATTERNS
AI System Design Interview Guide for AI/ML Engineers (2025-2026)
## 

1.1 SERVING LLMs AT SCALE

KEY CHALLENGES:
- LLMs are memory-intensive (GPT-3 175B params = ~350GB in FP16)
- Inference is compute-bound (autoregressive generation is sequential)
- Latency requirements vary (real-time chat vs batch processing)
- Cost of GPU infrastructure is high ($2-8/hr per A100)

CORE SERVING PATTERNS:

A) SINGLE-MODEL SERVING
   - One model loaded per GPU/instance
   - Simplest deployment pattern
   - Use when: small-medium models (7B-13B), low traffic
   - Tools: vLLM, TGI (Text Generation Inference), Triton Inference Server

   Architecture:
   [Client] --> [Load Balancer] --> [GPU Instance with Model]
                                --> [GPU Instance with Model]
                                --> [GPU Instance with Model]

B) MODEL PARALLELISM (Tensor Parallelism + Pipeline Parallelism)

   Tensor Parallelism:
   - Split individual layers across multiple GPUs
   - Each GPU holds a slice of every layer
   - Reduces per-GPU memory, enables larger models
   - Best for: models too large for single GPU
   - Latency: adds inter-GPU communication overhead

   Pipeline Parallelism:
   - Split model layers sequentially across GPUs
   - GPU 1 handles layers 1-20, GPU 2 handles layers 21-40, etc.
   - Enables micro-batching for better throughput
   - Best for: maximizing throughput with large models

   Combined approach (used by Megatron-LM, DeepSpeed):
   [GPU 0: Layers 0-10, Slice A] [GPU 1: Layers 0-10, Slice B]
   [GPU 2: Layers 11-20, Slice A] [GPU 3: Layers 11-20, Slice B]

C) CONTINUOUS BATCHING (PagedAttention / vLLM pattern)
   - Traditional batching: wait for batch to fill, process together
   - Continuous batching: dynamically add/remove requests from batch
   - Requests join mid-batch when slots free up
   - 2-5x throughput improvement over static batching
   - vLLM's PagedAttention: manages KV-cache like virtual memory pages

   How it works:
   1. Request A arrives -> starts generating tokens
   2. Request B arrives -> joins the batch immediately
   3. Request A finishes -> slot freed, Request C joins
   4. No waiting for entire batch to complete

D) SPECULATIVE DECODING
   - Use a small "draft" model to generate N candidate tokens
   - Use the large model to verify all N tokens in one forward pass
   - Accept tokens that match, reject and regenerate from divergence
   - 2-3x speedup with no quality loss
   - Example: 7B draft model + 70B target model

E) QUANTIZATION SERVING
   - FP32 -> FP16 -> INT8 -> INT4 progression
   - GPTQ, AWQ, GGUF formats
   - 2-4x memory reduction, 1.5-3x speedup
   - Minimal quality loss for INT8, slight loss for INT4
   - Enables serving 70B models on single A100 (80GB) with INT4


## 1.2 MODEL SERVING ARCHITECTURES


A) DIRECT SERVING ARCHITECTURE
   [Client] --> [Model Server (vLLM/TGI)] --> [GPU]

   Pros: Simple, low latency
   Cons: No abstraction, tight coupling, hard to scale

B) GATEWAY + MODEL SERVER ARCHITECTURE (RECOMMENDED)
   [Client] --> [API Gateway] --> [Router/Orchestrator] --> [Model Server Pool]

   Components:
   - API Gateway: Auth, rate limiting, request validation
   - Router: Model selection, load balancing, failover
   - Model Server Pool: Multiple instances of vLLM/TGI/Triton

   Pros: Decoupled, scalable, supports multiple models
   Cons: More infrastructure complexity

C) MODEL MESH / MODEL REGISTRY PATTERN
   [Client] --> [Gateway] --> [Model Mesh Controller]
                                 |
                    [Model Registry (MLflow, Weights & Biases)]
                                 |
                    [Dynamic Model Loading/Unloading]
                                 |
              [GPU Pool: Model A, Model B, Model C...]

   - Models loaded/unloaded based on demand
   - Efficient GPU utilization
   - KServe ModelMesh, Triton Model Repository
   - Best for: serving many models with varying traffic

D) SIDECAR PATTERN FOR AI
   [Pod]
     [Main Container: Application Logic]
     [Sidecar Container: Model Inference]
     [Sidecar Container: Feature Store Client]
     [Sidecar Container: Monitoring Agent]

   - Each pod has model inference as a sidecar
   - Useful for edge deployment, embedding models


## 1.3 API GATEWAY PATTERNS FOR AI


AI-SPECIFIC GATEWAY REQUIREMENTS:
- Streaming support (SSE/WebSocket for token-by-token generation)
- Token counting and billing
- Model routing (route to different models based on request)
- Prompt validation and sanitization
- Response filtering (content safety)
- Usage tracking per API key

ARCHITECTURE:

[Client Request]
      |
[API Gateway Layer]
   |-- Authentication & API Key validation
   |-- Rate limiting (token-based, not just request-based)
   |-- Request validation (max tokens, valid model name)
   |-- Prompt safety check (injection detection)
   |-- Request logging
      |
[Router Layer]
   |-- Model selection (based on request params or routing rules)
   |-- A/B testing routing (10% to new model, 90% to stable)
   |-- Fallback routing (if primary model unavailable, use backup)
   |-- Priority queuing (premium users get priority)
      |
[Model Serving Layer]
   |-- Model Instance Pool
   |-- Response streaming back to client
      |
[Post-Processing Layer]
   |-- Content filtering
   |-- PII redaction
   |-- Response logging
   |-- Token counting for billing

KEY TOOLS:
- Kong Gateway with AI plugins
- LiteLLM (unified API for 100+ LLM providers)
- Portkey AI Gateway
- Custom FastAPI gateway
- AWS API Gateway + Lambda for serverless

EXAMPLE: LiteLLM Proxy Pattern
```python
# LiteLLM acts as unified gateway to multiple LLM providers
from litellm import completion

# Same API, different models
response = completion(
    model="gpt-4",           # Routes to OpenAI
    messages=[{"role": "user", "content": "Hello"}]
)

response = completion(
    model="claude-3-opus",    # Routes to Anthropic
    messages=[{"role": "user", "content": "Hello"}]
)

response = completion(
    model="ollama/llama3",    # Routes to local Ollama
    messages=[{"role": "user", "content": "Hello"}]
)
```


## 1.4 LOAD BALANCING FOR AI SERVICES


CHALLENGES UNIQUE TO AI:
- Requests have highly variable processing times (10 tokens vs 4000 tokens)
- GPU memory is the bottleneck, not CPU
- Streaming responses maintain long-lived connections
- Some requests are 100x more expensive than others

LOAD BALANCING STRATEGIES:

A) ROUND ROBIN (Basic)
   - Distribute requests evenly across instances
   - Problem: doesn't account for request complexity
   - Use when: requests are roughly uniform size

B) LEAST CONNECTIONS
   - Route to instance with fewest active requests
   - Better for variable-length requests
   - Problem: doesn't account for request size

C) LEAST TOKENS / WEIGHTED ROUTING (AI-Specific)
   - Estimate token count from request
   - Route to instance with lowest total pending tokens
   - Best for LLM serving

   Algorithm:
   1. Parse incoming request, estimate output tokens
   2. For each instance, calculate: pending_tokens = sum(estimated_tokens for active requests)
   3. Route to instance with lowest pending_tokens

D) GPU-UTILIZATION-AWARE ROUTING
   - Monitor GPU memory and compute utilization per instance
   - Route to instance with most available GPU resources
   - Requires metrics collection (DCGM, nvidia-smi)

   Metrics to monitor:
   - GPU Memory Used / Total
   - GPU Compute Utilization %
   - KV-Cache Utilization (for vLLM)
   - Queue depth per instance

E) CONSISTENT HASHING (for cached models)
   - Hash request to specific instance
   - Enables better KV-cache reuse
   - Good for: repeated similar prompts (chatbots with system prompts)

F) PRIORITY-BASED QUEUING
   [Incoming Requests]
         |
   [Priority Queue]
   |-- P0: Real-time (chat) -> dedicated GPU pool
   |-- P1: Near-real-time (API) -> shared GPU pool
   |-- P2: Batch (analytics) -> spot instances
   |-- P3: Background (fine-tuning) -> preemptible


## 1.5 CACHING STRATEGIES FOR AI


A) EXACT CACHE (Deterministic Cache)

   How it works:
   - Hash the exact input (prompt + parameters)
   - If hash matches, return cached response
   - Cache key = hash(model + prompt + temperature + max_tokens + ...)

   When temperature=0 (deterministic):
   - Same input always produces same output
   - Perfect for caching

   Implementation:
   ```python
   import hashlib
   import redis

   def get_cache_key(model, messages, temperature, max_tokens):
       content = f"{model}:{json.dumps(messages)}:{temperature}:{max_tokens}"
       return hashlib.sha256(content.encode()).hexdigest()

   def cached_completion(model, messages, temperature=0, max_tokens=1000):
       cache_key = get_cache_key(model, messages, temperature, max_tokens)

       # Check cache
       cached = redis_client.get(cache_key)
       if cached:
           return json.loads(cached)

       # Call LLM
       response = llm_client.chat.completions.create(
           model=model, messages=messages,
           temperature=temperature, max_tokens=max_tokens
       )

       # Store in cache with TTL
       redis_client.setex(cache_key, 3600, json.dumps(response))
       return response
   ```

   Pros: 100% accurate, very fast (sub-ms), simple
   Cons: Only works for exact matches, low hit rate for varied prompts
   Best for: API endpoints with repeated queries, static system prompts

B) SEMANTIC CACHE

   How it works:
   - Embed the input prompt using an embedding model
   - Search for similar embeddings in vector store
   - If similarity > threshold, return cached response
   - Otherwise, call LLM and cache the new result

   Architecture:
   [Input Prompt] --> [Embedding Model] --> [Vector Search in Cache]
                                                  |
                                     [Similar? (cosine > 0.95)]
                                        /              \
                                      Yes               No
                                       |                 |
                              [Return Cached]     [Call LLM]
                                                       |
                                              [Cache Response + Embedding]

   Implementation:
   ```python
   from openai import OpenAI
   import numpy as np
   from qdrant_client import QdrantClient

   class SemanticCache:
       def __init__(self, similarity_threshold=0.95):
           self.embedding_client = OpenAI()
           self.vector_db = QdrantClient(":memory:")
           self.threshold = similarity_threshold

       def get_embedding(self, text):
           response = self.embedding_client.embeddings.create(
               model="text-embedding-3-small", input=text
           )
           return response.data[0].embedding

       def lookup(self, prompt):
           embedding = self.get_embedding(prompt)
           results = self.vector_db.search(
               collection_name="cache",
               query_vector=embedding,
               limit=1
           )
           if results and results[0].score >= self.threshold:
               return results[0].payload["response"]
           return None

       def store(self, prompt, response):
           embedding = self.get_embedding(prompt)
           self.vector_db.upsert(
               collection_name="cache",
               points=[{
                   "id": uuid4().hex,
                   "vector": embedding,
                   "payload": {"prompt": prompt, "response": response}
               }]
           )
   ```

   Pros: Catches paraphrased queries, higher hit rate
   Cons: Embedding cost, possible false positives, latency for embedding
   Best for: Customer support bots, FAQ systems, search queries

   Tools: GPTCache, LangChain CacheBackedEmbeddings, Redis Vector Search

C) KV-CACHE (Internal to Model Serving)
   - Caches key-value attention tensors during generation
   - Prefix caching: if same system prompt, reuse KV cache
   - vLLM automatic prefix caching: shared system prompts cached
   - Huge speedup for chatbots with long system prompts

   Example: System prompt of 2000 tokens
   - Without prefix cache: compute 2000 tokens every request
   - With prefix cache: compute only user message tokens
   - Speedup: 2-10x for long system prompts

D) MULTI-LEVEL CACHE STRATEGY

   [Request] --> [L1: Exact Cache (Redis, <1ms)]
                     |miss
                 [L2: Semantic Cache (Vector DB, ~50ms)]
                     |miss
                 [L3: KV-Prefix Cache (vLLM, saves compute)]
                     |miss
                 [L4: Full LLM Inference (~500ms-5s)]
                     |
                 [Cache response at L1 + L2]


## 1.6 RATE LIMITING FOR AI SERVICES


AI-SPECIFIC RATE LIMITING DIMENSIONS:
1. Requests per minute (RPM) - standard
2. Tokens per minute (TPM) - AI-specific, critical
3. Tokens per day (TPD) - cost control
4. Concurrent requests - prevent GPU overload
5. Per-model limits - expensive models get tighter limits

ALGORITHMS:

A) TOKEN BUCKET (Most Common)
   - Bucket fills with tokens at constant rate
   - Each request consumes tokens based on estimated size
   - If bucket empty, request is queued or rejected

   ```python
   class TokenBucketRateLimiter:
       def __init__(self, tokens_per_minute, burst_size):
           self.rate = tokens_per_minute / 60  # tokens per second
           self.burst = burst_size
           self.tokens = burst_size
           self.last_refill = time.time()

       def allow_request(self, estimated_tokens):
           self._refill()
           if self.tokens >= estimated_tokens:
               self.tokens -= estimated_tokens
               return True
           return False

       def _refill(self):
           now = time.time()
           elapsed = now - self.last_refill
           self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
           self.last_refill = now
   ```

B) SLIDING WINDOW WITH TOKEN COUNTING
   - Track total tokens used in sliding window
   - More accurate than fixed windows
   - Prevents burst at window boundaries

C) TIERED RATE LIMITING
   ```
   Free Tier:     10 RPM,    10,000 TPM,   100,000 TPD
   Basic Tier:   100 RPM,   100,000 TPM, 1,000,000 TPD
   Pro Tier:     500 RPM,   500,000 TPM, 5,000,000 TPD
   Enterprise:  Custom RPM,  Custom TPM,   Custom TPD
   ```

D) ADAPTIVE RATE LIMITING
   - Adjust limits based on current system load
   - If GPU utilization > 80%, reduce limits
   - If queue depth > threshold, start rejecting low-priority requests
   - Protects system during traffic spikes

IMPLEMENTATION WITH REDIS:
```python
import redis
import time

class AIRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def check_rate_limit(self, api_key, estimated_tokens):
        pipe = self.redis.pipeline()
        now = int(time.time())
        window_key = f"ratelimit:{api_key}:{now // 60}"

        pipe.incrby(window_key, estimated_tokens)
        pipe.expire(window_key, 120)
        results = pipe.execute()

        current_usage = results[0]
        tier_limit = self.get_tier_limit(api_key)

        return current_usage <= tier_limit
```

## 1.8 AI GATEWAY PATTERNS (2025-2026)


AI Gateways have emerged as a critical infrastructure layer for production
LLM applications. They sit between clients and LLM providers, providing:

CORE CAPABILITIES:
- Model Routing: Route requests to optimal model based on task type, cost,
  latency requirements. E.g., route simple queries to GPT-4o mini, complex
  to Claude Opus 4.5, multimodal to Gemini 3 Pro.
- Fallback & Retry: Automatic failover between providers if one is down or
  rate-limited. Circuit breaker patterns for provider health.
- Cost Management: Track token usage per user/team, enforce budgets, optimize
  model selection for cost vs quality tradeoffs.
- Caching: Semantic caching -- use embedding similarity to match new queries
  to cached responses (threshold ~0.95). Reduces costs by 30-60%.
- Observability: Log all requests/responses, latency tracking, token usage
  analytics, quality metrics.
- Security: PII redaction before sending to providers, content filtering,
  prompt injection detection, audit logging.

SEMANTIC CACHING PATTERN:
```
User Query → Embed Query → Search Cache (cosine similarity)
                                    |
                        Similarity > 0.95? ─── Yes ──→ Return Cached
                                    |
                                   No
                                    |
                           Call LLM Provider
                                    |
                         Cache Response + Embedding
                                    |
                          Return Response
```

MODEL ROUTING STRATEGIES:
```
Simple Classification Router:
  - Input analysis (token count, has images, complexity score)
  - Route to appropriate model tier
  - Example: length < 100 tokens → mini model
             has_image → multimodal model
             complexity_score > 0.8 → frontier model

Cost-Optimized Router:
  - Start with cheapest model
  - Evaluate response quality (confidence score, self-check)
  - If quality insufficient, escalate to more expensive model
  - "Cascade" pattern: mini → standard → frontier
```

POPULAR AI GATEWAY TOOLS (2025):
- LiteLLM: Open-source, unified API for 100+ LLM providers
- Portkey: Commercial AI gateway with observability
- Helicone: Open-source LLM observability platform
- LangSmith: LangChain's observability and testing platform
- Braintrust: LLM eval and observability platform

LLM OBSERVABILITY (2025-2026):
Key metrics to track in production LLM systems:
- Latency: p50, p95, p99 response times, time-to-first-token (TTFT)
- Quality: User satisfaction, thumbs up/down, automated eval scores
- Cost: Tokens per request, cost per user, cost per feature
- Reliability: Error rates, timeout rates, provider availability
- Safety: Content filter trigger rates, prompt injection attempts

> YOUR EXPERIENCE: At RavianAI, building a production agentic AI platform
> requires all these patterns: model routing for cost optimization, semantic
> caching for repeated queries, observability for debugging agent workflows,
> and gateway-level security. This is directly relevant to your platform
> architecture work.

## END OF SECTION 1 (Updated February 2026)

