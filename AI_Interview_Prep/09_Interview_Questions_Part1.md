================================================================================
SECTION 9A: INTERVIEW QUESTIONS WITH DETAILED ANSWERS (Part 1)
AI System Design Interview Guide for AI/ML Engineers (2025-2026)
================================================================================

========================================
Q1: "How would you design a system to serve an LLM to 1 million users?"
========================================

FRAMEWORK FOR ANSWERING (Use this for ALL system design questions):
1. Clarify requirements (2 min)
2. High-level architecture (5 min)
3. Deep dive into components (15 min)
4. Scaling & trade-offs (5 min)
5. Monitoring & operations (3 min)

ANSWER:

Step 1 - Clarify:
- What kind of LLM? (chatbot, API, specific task?)
- Latency requirements? (real-time < 500ms TTFT? or async?)
- What model size? (7B, 70B, 175B?)
- Multi-turn or single-turn?
- Budget constraints?

Assume: Customer support chatbot, 1M DAU, multi-turn, TTFT < 500ms,
use 70B model, $100K/month budget.

Step 2 - High-level:
```
[Clients] -> [CDN] -> [API Gateway] -> [Chat Orchestrator]
                                            |
                    +------+------+------+------+
                    |      |      |      |      |
               [Session] [Guard] [RAG]  [LLM]  [Cache]
               [Store]   [rails] [Svc]  [Svc]  [Layer]
```

Step 3 - Deep dive:

A) Model Serving:
   - 70B model in FP16 = ~140GB VRAM
   - Use 2x A100-80GB per instance (tensor parallel)
   - vLLM with continuous batching
   - Speculative decoding with 7B draft model
   - 15-20 instances for 1M DAU (assuming 10 messages/user/day = 10M messages)
   - Throughput per instance: ~500 req/min -> need 15 instances for peak

B) Caching:
   - L1: Exact cache in Redis (repeated questions) -> 15% hit rate
   - L2: KV-prefix cache in vLLM (shared system prompt) -> 3x speedup
   - Estimated 30% total cost reduction from caching

C) Context Management:
   - Redis for session state (TTL 30 min)
   - Sliding window + summary for long conversations
   - Max 8K context window per request

D) RAG Integration:
   - Company knowledge base in Qdrant
   - Embedding service (separate GPU pool, A10G)
   - Hybrid search (BM25 + vector)
   - Reranker for top results

Step 4 - Scaling:
   - HPA based on queue depth and GPU utilization
   - Min 10 instances, max 30 instances
   - Warm pool of 3 standby instances
   - Multi-region for latency (US-East, US-West, EU)

Step 5 - Cost estimate:
   - 15 instances * 2 A100 * $3.50/hr = $105/hr = $75,600/month
   - With spot/reserved: ~$50K/month
   - Plus infrastructure (Redis, Qdrant, networking): ~$10K/month
   - Total: ~$60K/month (within $100K budget)


========================================
Q2: "Design a RAG system that handles 10,000 documents
     and serves 1,000 queries per second."
========================================

ANSWER:

Step 1 - Clarify:
- Document types? (assume: PDFs, web pages, text docs)
- Average document size? (assume: 10 pages / 5000 words each)
- Update frequency? (assume: 100 new docs/day)
- Accuracy requirements? (assume: 90%+ relevance)
- Latency? (assume: < 2 seconds end-to-end)

Step 2 - Back-of-envelope:
- 10,000 docs * 5000 words / 300 words per chunk = ~167,000 chunks
- Each chunk = 1536-dim embedding = ~6KB per vector
- Total vector storage: 167K * 6KB = ~1GB (easily fits in memory)
- 1000 QPS * 2s processing = 2000 concurrent requests

Step 3 - Ingestion Pipeline:
```
[Documents] -> [Parser (Unstructured.io)] -> [Chunker (512 tokens, 50 overlap)]
                                                    |
                                        [Embedding Service (batch)]
                                                    |
                                        [Qdrant (HNSW index)]
                                        [Elasticsearch (BM25)]
```
- Ingestion is offline, runs on schedule or event-driven
- GPU cluster for batch embedding (A10G, can process all 167K chunks in ~5 min)

Step 4 - Query Pipeline:
```
[Query] -> [Query Expansion (LLM generates 3 variations)]
              |
        [Hybrid Search]
           |-- Vector search (Qdrant, top 20)
           |-- BM25 search (Elasticsearch, top 20)
           |-- RRF fusion -> top 20 combined
              |
        [Cross-encoder Reranker] -> top 5
              |
        [LLM Generation with citations]
```

Step 5 - Scaling for 1000 QPS:
- Qdrant: 167K vectors easily in single instance, replicate 3x for read throughput
  Each replica handles ~500 QPS -> 3 replicas = 1500 QPS capacity
- Elasticsearch: 3-node cluster (more than enough for 10K docs)
- Embedding service: pre-compute query embeddings
  Embedding takes ~20ms per query, each GPU handles ~50 concurrent
  Need: 1000 / 50 = 20 embedding instances
- Reranker: ~50ms per query, each GPU handles 20 concurrent
  Need: 1000 / 20 = 50 reranker instances (THIS is the bottleneck)
  Optimization: batch requests, use lighter reranker, skip for high-confidence results
- LLM: ~1s per generation, each instance handles 10 concurrent (with batching)
  Need: 1000 / 10 = 100 LLM instances (MAJOR bottleneck)

Step 6 - Cost Optimization:
- Cache frequent queries (semantic cache): reduce LLM calls by 40%
  Now need: 600 QPS * LLM = 60 instances
- Use smaller model (GPT-4o-mini equivalent) for most queries
  Cascade: simple queries -> small model, complex -> large model
- Batch reranking: process 5 queries together
  Reduce reranker instances to ~15

KEY TRADE-OFFS TO DISCUSS:
- Accuracy vs latency: more retrieval stages = better accuracy but higher latency
- Cost vs quality: larger LLM = better answers but 10x cost
- Freshness vs performance: real-time indexing vs batch (5-min delay)


========================================
Q3: "How would you handle model versioning and deployment
     in a production ML system?"
========================================

ANSWER:

Architecture:
```
[Data Scientists]
      |
[Experiment Tracking (W&B / MLflow)]
      |
[Model Registry]
   |-- v1.0 (production, 90% traffic)
   |-- v1.1 (canary, 10% traffic)
   |-- v0.9 (rollback candidate)
      |
[CI/CD Pipeline (GitHub Actions)]
   |-- Automated testing
   |-- Performance benchmarking
   |-- Security scanning
   |-- Deployment automation
      |
[Model Serving (Kubernetes)]
   |-- Blue-green deployment
   |-- Canary releases
   |-- Automatic rollback
```

DETAILED WORKFLOW:

1. MODEL DEVELOPMENT:
   - Data scientist trains model, logs to experiment tracker
   - Evaluates on standard test set + held-out validation set
   - If metrics meet threshold, registers in model registry

2. AUTOMATED EVALUATION PIPELINE:
   ```python
   def evaluate_model(model_version):
       test_cases = load_test_suite()  # 500+ diverse test cases
       results = {
           "accuracy": run_accuracy_test(model_version, test_cases),
           "latency_p95": run_latency_test(model_version),
           "toxicity_rate": run_safety_test(model_version),
           "regression_tests": run_regression_tests(model_version),
           "cost_per_request": estimate_cost(model_version),
       }

       # Gates
       assert results["accuracy"] > 0.85, "Accuracy below threshold"
       assert results["latency_p95"] < 500, "Latency above SLA"
       assert results["toxicity_rate"] < 0.01, "Safety check failed"
       assert results["regression_tests"] == "PASS", "Regression detected"

       return results
   ```

3. CANARY DEPLOYMENT:
   - Deploy new version to 5% of traffic
   - Monitor for 24 hours:
     - Error rate (should not increase)
     - Latency (should not increase > 10%)
     - User feedback (should not decrease)
     - Business metrics (conversion, engagement)
   - If all good: increase to 25% -> 50% -> 100%
   - If bad: auto-rollback to previous version

4. ROLLBACK STRATEGY:
   - Keep previous 3 versions deployable
   - Rollback takes < 5 minutes (model already loaded in standby)
   - Trigger: automated (error rate > threshold) or manual

5. A/B TESTING:
   ```python
   class ModelRouter:
       def route(self, request, user):
           experiment = self.get_experiment(user.id)

           if experiment == "A":
               return self.models["v1.0"]  # control
           elif experiment == "B":
               return self.models["v1.1"]  # treatment

           # Default to production
           return self.models["production"]
   ```


========================================
Q4: "Design a system for real-time content moderation
     at social media scale (1M posts/minute)."
========================================

ANSWER:

Step 1 - Clarify:
- Content types: text, images, video, links
- Languages: 50+ languages
- Actions: approve, reject, escalate to human review
- Latency: < 500ms for text, < 2s for images, < 30s for video
- False positive rate: < 0.1% (don't wrongly remove content)
- False negative rate: < 1% (don't miss harmful content)

Step 2 - Architecture:
See Section 8.5 (Real-time Content Moderation) for full architecture.

Step 3 - Scale calculations:
- 1M posts/min = ~16,700 posts/second
- Text: ~12,000/sec (72% of content)
- Images: ~3,300/sec (20% of content)
- Video: ~830/sec (5% of content)
- Links: ~500/sec (3% of content)

Tier 1 (Rules, CPU): handles 60%
- Need: 10,000 req/sec throughput
- Each CPU instance: 2,000 req/sec
- Need: 5 instances (cheap, < $500/month)

Tier 2 (ML, GPU): handles 35%
- Need: 5,850 req/sec
- Text classifier (GPU): 500 req/sec per A10G -> need 8 GPUs for text
- Image classifier (GPU): 200 req/sec per A10G -> need 7 GPUs for images
- Video (frame sample + image classifier): 50 req/sec per GPU -> need 10 GPUs
- Total: ~25 GPUs ($18,000/month)

Tier 3 (LLM): handles 5%
- Need: ~835 req/sec
- LLM at 10 req/sec per instance -> need 84 instances
- TOO EXPENSIVE. Optimization:
  - Batch requests: 5 posts per LLM call -> 17 req/sec -> 50 instances
  - Use smaller model (GPT-4o-mini equivalent) -> cheaper
  - Cache common violation patterns -> reduce by 40% -> 30 instances
  - Cost: ~$50,000/month

Human review: 1-2%
- ~167-334 posts/minute for human review
- Average review time: 30 seconds
- Need: ~100-170 reviewers on duty

Step 4 - Key design decisions:
- Multi-language: fine-tune classifiers per language family
- Evolving threats: retrain weekly with new labeled data
- Appeals: separate pipeline, different reviewers
- Legal compliance: different policies per country
- Transparency: provide reasons for content removal


========================================
Q5: "How do you optimize the cost of an LLM-based application
     that's spending $500K/month on API calls?"
========================================

ANSWER:

Step 1 - Analyze current spending:
```
Current breakdown ($500K/month):
- GPT-4 API calls: $350K (70%)
  - 50M requests/month
  - Average 800 input + 400 output tokens
- GPT-3.5 API calls: $80K (16%)
  - 200M requests/month
- Embedding API: $50K (10%)
  - 500M embeddings/month
- Infrastructure: $20K (4%)
```

Step 2 - Optimization strategies (ranked by impact):

A) MODEL ROUTING ($350K -> $100K, save $250K):
   - Classify queries by complexity
   - Route 70% to GPT-4o-mini instead of GPT-4
   - Only 10% truly need GPT-4 quality
   - Savings: $250K/month

B) CACHING ($100K -> $70K, save $30K):
   - Exact cache: 20% hit rate for repeated queries
   - Semantic cache: 15% additional hit rate
   - KV-prefix cache for system prompts
   - Savings: $30K/month

C) PROMPT OPTIMIZATION ($70K -> $55K, save $15K):
   - Compress system prompts (200 tokens -> 50 tokens)
   - Reduce few-shot examples (5 -> 2 or zero-shot)
   - Use structured output (shorter responses)
   - Savings: $15K/month

D) BATCHING ($55K -> $45K, save $10K):
   - Use OpenAI Batch API (50% discount) for non-real-time
   - Batch embedding requests
   - Savings: $10K/month

E) SELF-HOSTING FOR EMBEDDINGS ($50K -> $5K, save $45K):
   - Host open-source embedding model (e5-large, BGE)
   - 5x A10G instances handle 500M embeddings/month
   - Cost: ~$5K/month vs $50K API
   - Savings: $45K/month

F) SELF-HOSTING FOR SIMPLE TASKS ($45K -> $35K, save $10K):
   - Host Llama 3 8B for simple classification/extraction
   - Replace GPT-3.5 calls for simple tasks
   - Savings: $10K/month

TOTAL SAVINGS: $360K/month (72% reduction)
NEW MONTHLY COST: ~$140K/month

Step 3 - Implementation priority:
1. Model routing (highest impact, 1 week to implement)
2. Self-host embeddings (high impact, 2 weeks)
3. Caching layer (medium impact, 1 week)
4. Prompt optimization (medium impact, ongoing)
5. Batching (lower impact, 1 week)
6. Self-host simple model (lower impact, 2 weeks)


========================================
Q6: "Design a multi-agent system for automated code review."
========================================

ANSWER:

Architecture:
```
[Pull Request Webhook]
      |
[Orchestrator Agent]
   |-- Analyzes PR: files changed, diff size, languages
   |-- Creates execution plan
   |-- Assigns specialist agents
      |
[Parallel Agent Execution]
   |
   +-- [Code Quality Agent]
   |   - Style guide compliance
   |   - Code complexity analysis
   |   - Best practices check
   |   - DRY principle violations
   |
   +-- [Security Agent]
   |   - Vulnerability scanning (OWASP Top 10)
   |   - Hardcoded secrets detection
   |   - Dependency vulnerability check
   |   - SQL injection / XSS detection
   |
   +-- [Performance Agent]
   |   - Big-O complexity analysis
   |   - N+1 query detection
   |   - Memory leak patterns
   |   - Unnecessary computation
   |
   +-- [Test Coverage Agent]
   |   - Missing test cases
   |   - Edge cases not covered
   |   - Test quality assessment
   |
   +-- [Documentation Agent]
       - Missing docstrings
       - Outdated comments
       - README updates needed
      |
[Aggregator Agent]
   |-- Collects all agent outputs
   |-- Deduplicates findings
   |-- Prioritizes (critical -> warning -> suggestion)
   |-- Generates unified review
      |
[PR Comment Generator]
   |-- Posts inline comments on specific lines
   |-- Posts summary comment
   |-- Requests changes if critical issues found
```

KEY DESIGN DECISIONS:
1. Each agent has limited scope (single responsibility)
2. Agents run in parallel (latency = max agent time, not sum)
3. Shared context: full PR diff, file contents, repo structure
4. Cost control: budget per PR ($0.50 max), skip review for trivial changes
5. Learning: track which suggestions developers accept/reject

COST ESTIMATE:
- Average PR: 5 files, 200 lines changed
- Each agent: ~2000 input tokens + ~500 output tokens
- 5 agents * $0.003/request = $0.015 per PR
- Aggregator: $0.005
- Total: ~$0.02 per PR ($600/month for 1000 PRs/day)


========================================
Q7: "What is the difference between synchronous and asynchronous
     inference, and when would you use each?"
========================================

ANSWER:

SYNCHRONOUS INFERENCE:
```
Client sends request -> waits -> receives response

Timeline:
Client: |--send--|------wait------|--receive--|
Server: |--------process----------|
```
- Client blocks until response ready
- Latency-sensitive (user waiting)
- Simple programming model
- Resource allocation: always-on GPU

Use when:
- User is waiting (chatbot, search, recommendations)
- Response needed in < 5 seconds
- Simple request-response pattern
- Moderate traffic that justifies always-on resources

ASYNCHRONOUS INFERENCE:
```
Client sends request -> gets job ID -> polls/webhook -> receives response

Timeline:
Client: |--send--|--get job ID--|...do other stuff...|--receive--|
Server: |--queue--|.....wait.....|--process--|--notify--|
```
- Client doesn't block
- Higher throughput (can batch process)
- Better resource utilization
- More complex programming model

Use when:
- Batch processing (1000s of items)
- Long-running tasks (video processing, document analysis)
- Cost optimization (can use spot instances, batch API discounts)
- User doesn't need immediate response (email, notifications)

HYBRID (STREAMING):
```
Client sends request -> receives first token quickly -> streams remaining

Timeline:
Client: |--send--|--first token--|--stream tokens...--|--done--|
Server: |--------|-generate token by token-----------|
```
- Best of both: responsive + can handle long generation
- User sees progress immediately
- Most common for LLM chat interfaces

========================================
END OF SECTION 9A
========================================
