================================================================================
SECTION 5: MONITORING AND OBSERVABILITY FOR AI SYSTEMS
AI System Design Interview Guide for AI/ML Engineers (2025-2026)
================================================================================

========================================
5.1 LLM OBSERVABILITY TOOLS AND FRAMEWORKS
========================================

A) LANGSMITH (by LangChain):
   Purpose: Tracing, debugging, and monitoring LLM applications

   Key Features:
   - Full trace of LLM chains (every step visualized)
   - Input/output logging for each chain step
   - Latency breakdown per step
   - Token usage tracking
   - Dataset management for evaluation
   - Human feedback collection
   - Comparison of runs across model versions

   Architecture:
   ```
   [Your LLM App (LangChain)]
        |
   [LangSmith SDK (traces)]
        |
   [LangSmith Cloud / Self-hosted]
        |-- Trace viewer
        |-- Run analytics
        |-- Dataset & evaluation
        |-- Feedback dashboard
   ```

   What to monitor:
   - Chain execution traces (every retrieval, LLM call, tool use)
   - Intermediate outputs (what did RAG retrieve? what did LLM generate?)
   - Latency per step (was retrieval slow? was generation slow?)
   - Error rates and error types
   - Token usage per chain execution

B) PHOENIX (by Arize AI):
   Purpose: Open-source LLM observability and evaluation

   Key Features:
   - Trace visualization (OpenTelemetry-based)
   - Embedding drift detection
   - Retrieval quality analysis (for RAG)
   - LLM evaluation (hallucination detection, relevance scoring)
   - Prompt template tracking
   - Open-source, self-hostable

   Architecture:
   ```
   [Your LLM App]
        |
   [OpenInference SDK (OpenTelemetry spans)]
        |
   [Phoenix Server (local or hosted)]
        |-- Trace explorer
        |-- Embedding visualizer
        |-- Evaluation dashboard
        |-- Retrieval metrics
   ```

   Unique strengths:
   - Embedding analysis: UMAP projections of query embeddings
   - Retrieval analysis: nDCG, precision@k for RAG pipelines
   - Drift detection: are query patterns changing over time?

C) WEIGHTS & BIASES (W&B):
   Purpose: Full ML lifecycle - training, evaluation, monitoring

   Key Features:
   - Experiment tracking (training runs)
   - Model registry and versioning
   - LLM trace logging (W&B Weave)
   - Prompt management
   - Evaluation tables
   - Artifact versioning
   - Team collaboration

   W&B Weave (LLM-specific):
   ```
   [Your LLM App]
        |
   [Weave SDK (@weave.op() decorator)]
        |
   [W&B Platform]
        |-- Call traces
        |-- Cost tracking
        |-- Evaluation scoring
        |-- Prompt versioning
        |-- A/B comparison
   ```

   ```python
   import weave

   @weave.op()
   def my_llm_pipeline(query: str) -> str:
       # All inputs, outputs, latency auto-logged
       context = retrieve_documents(query)
       response = generate_answer(query, context)
       return response
   ```

D) COMPARISON TABLE:
```
| Feature              | LangSmith    | Phoenix      | W&B Weave    |
|----------------------|--------------|--------------|--------------|
| Tracing              | Excellent    | Excellent    | Good         |
| Open Source           | No           | Yes          | Partial      |
| RAG Evaluation       | Good         | Excellent    | Good         |
| Training Tracking    | No           | No           | Excellent    |
| Model Registry       | No           | No           | Yes          |
| Embedding Analysis   | No           | Excellent    | Basic        |
| Self-hosted          | Enterprise   | Yes (free)   | Enterprise   |
| LangChain Integration| Native       | Via OTEL     | Via Weave    |
| Cost                 | Free tier +  | Free (OSS)   | Free tier +  |
| Best for             | LangChain    | Open-source  | Full ML      |
|                      | heavy teams  | RAG-focused  | lifecycle    |
```


========================================
5.2 PROMPT MONITORING
========================================

WHAT TO MONITOR:

1. PROMPT TEMPLATE VERSIONING:
   ```
   Prompt Registry:
     system_prompt_v1.0: "You are a helpful assistant..."
     system_prompt_v1.1: "You are a helpful assistant. Be concise..."
     system_prompt_v2.0: "You are an expert AI assistant..."

   Track:
     - Which version is active in production
     - Performance metrics per version
     - A/B test results between versions
   ```

2. INPUT MONITORING:
   - Token count distribution (are users sending longer prompts?)
   - Language distribution (unexpected languages?)
   - Topic distribution (shifting user needs?)
   - Injection attempt detection
   - PII detection in prompts

3. OUTPUT MONITORING:
   - Response length distribution
   - Refusal rate (how often model says "I can't help with that")
   - Hallucination detection (factual consistency)
   - Tone/sentiment of responses
   - Format compliance (JSON, structured output)

4. PROMPT-RESPONSE QUALITY METRICS:
   ```python
   class PromptMonitor:
       def log_interaction(self, prompt, response, metadata):
           metrics = {
               "input_tokens": count_tokens(prompt),
               "output_tokens": count_tokens(response),
               "latency_ms": metadata["latency"],
               "model": metadata["model"],
               "prompt_template_version": metadata["template_version"],
               "contains_pii": detect_pii(prompt),
               "response_format_valid": validate_format(response),
               "user_rating": metadata.get("user_rating"),
           }
           self.metrics_store.log(metrics)

           # Alert on anomalies
           if metrics["input_tokens"] > 10000:
               alert("Unusually long prompt detected")
           if metrics["contains_pii"]:
               alert("PII detected in prompt")
   ```


========================================
5.3 DRIFT DETECTION
========================================

TYPES OF DRIFT IN AI SYSTEMS:

A) DATA DRIFT (Input Drift):
   - Distribution of user inputs changes over time
   - Example: Chatbot trained on English, starts getting Spanish queries
   - Detection: Compare embedding distributions over time windows

   ```python
   from scipy.stats import ks_2samp
   import numpy as np

   class DriftDetector:
       def __init__(self, reference_embeddings):
           self.reference = reference_embeddings  # from training/baseline

       def detect_drift(self, current_embeddings, threshold=0.05):
           # Kolmogorov-Smirnov test per dimension
           drift_dimensions = 0
           for dim in range(self.reference.shape[1]):
               stat, p_value = ks_2samp(
                   self.reference[:, dim],
                   current_embeddings[:, dim]
               )
               if p_value < threshold:
                   drift_dimensions += 1

           drift_ratio = drift_dimensions / self.reference.shape[1]
           return {
               "drift_detected": drift_ratio > 0.1,  # >10% dimensions drifted
               "drift_ratio": drift_ratio,
               "severity": "high" if drift_ratio > 0.3 else "medium" if drift_ratio > 0.1 else "low"
           }
   ```

B) CONCEPT DRIFT:
   - Relationship between inputs and desired outputs changes
   - Example: Sentiment around a product changes after a controversy
   - Detection: Monitor prediction quality metrics over time

C) MODEL PERFORMANCE DRIFT:
   - Model accuracy degrades without input distribution changing
   - Caused by: changing user expectations, world knowledge becoming stale
   - Detection: Track user feedback, automated evaluation scores

D) RETRIEVAL DRIFT (RAG-specific):
   - Quality of retrieved documents degrades
   - Causes: new documents not indexed, embedding model mismatch
   - Detection: Monitor retrieval relevance scores

MONITORING DASHBOARD FOR DRIFT:
```
+--------------------------------------------------+
| DRIFT MONITORING DASHBOARD                        |
+--------------------------------------------------+
| Input Distribution    | [=========>  ] Normal     |
| Embedding Centroid    | [=====>      ] DRIFT!     |
| Response Quality      | [========>   ] Normal     |
| Retrieval Relevance   | [======>     ] Warning    |
| Token Usage Pattern   | [=========>  ] Normal     |
+--------------------------------------------------+
| Alerts:                                           |
| - Embedding drift detected: 2025-01-15 14:30     |
|   Query cluster shifted toward medical domain     |
| - Retrieval relevance dropped 15% this week       |
+--------------------------------------------------+
```


========================================
5.4 COST TRACKING
========================================

WHAT TO TRACK:

1. PER-REQUEST COST:
   ```python
   MODEL_PRICING = {
       "gpt-4-turbo": {"input": 0.01, "output": 0.03},     # per 1K tokens
       "gpt-4o": {"input": 0.0025, "output": 0.01},
       "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
       "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
       "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
   }

   def calculate_request_cost(model, input_tokens, output_tokens):
       pricing = MODEL_PRICING[model]
       cost = (input_tokens / 1000 * pricing["input"] +
               output_tokens / 1000 * pricing["output"])
       return cost
   ```

2. AGGREGATED COST VIEWS:
   - Cost per user / per API key
   - Cost per feature (chatbot vs search vs summarization)
   - Cost per model
   - Daily / weekly / monthly cost trends
   - Cost per conversation (multi-turn)

3. INFRASTRUCTURE COSTS:
   - GPU instance costs (on-demand vs spot vs reserved)
   - Storage costs (vector DB, model artifacts)
   - Network costs (especially cross-region)
   - Embedding computation costs

4. COST ALLOCATION TAGS:
   ```python
   @track_cost(
       team="search",
       feature="semantic_search",
       environment="production",
       priority="p1"
   )
   def search_query(query):
       embedding = embed(query)         # Cost: embedding model
       results = vector_search(embedding)  # Cost: vector DB query
       answer = llm_generate(query, results)  # Cost: LLM tokens
       return answer
   ```

COST ALERTING:
```
Rules:
- Alert if daily cost > $500 (budget threshold)
- Alert if single request cost > $1 (anomaly)
- Alert if cost/request increases > 20% week-over-week
- Alert if a user's daily cost exceeds their tier limit
```


========================================
5.5 LATENCY MONITORING
========================================

LATENCY BREAKDOWN FOR LLM SYSTEM:
```
Total Latency (e.g., 2500ms)
|
|-- Network latency: 20ms
|-- API Gateway processing: 10ms
|-- Authentication: 5ms
|-- Preprocessing: 30ms
|-- Feature/context retrieval: 150ms
|     |-- Embedding generation: 50ms
|     |-- Vector search: 80ms
|     |-- Document fetch: 20ms
|-- Queue wait time: 100ms
|-- Model inference: 2000ms
|     |-- Time to first token (TTFT): 200ms
|     |-- Token generation: 1800ms (at 50 tokens/sec)
|-- Postprocessing: 30ms
|-- Response serialization: 10ms
|-- Network return: 20ms
```

KEY LATENCY METRICS:

1. Time to First Token (TTFT):
   - Critical for streaming UX
   - User sees response beginning quickly
   - Target: < 500ms for interactive chat
   - Affected by: queue depth, model load, context length

2. Inter-Token Latency (ITL):
   - Time between consecutive tokens
   - Affects perceived streaming speed
   - Target: < 50ms for smooth reading experience
   - Affected by: batch size, model size, GPU speed

3. Total Generation Time:
   - TTFT + (num_output_tokens * ITL)
   - For 200-token response: 200ms + 200 * 20ms = 4.2s

4. End-to-End Latency:
   - From client request to complete response
   - Includes all preprocessing, retrieval, generation, postprocessing

PERCENTILE MONITORING:
```
Track P50, P90, P95, P99 for each:

| Metric          | P50    | P90    | P95    | P99    | SLA    |
|-----------------|--------|--------|--------|--------|--------|
| TTFT            | 150ms  | 300ms  | 500ms  | 1000ms | < 500ms|
| ITL             | 20ms   | 30ms   | 40ms   | 80ms   | < 50ms |
| Total latency   | 1.5s   | 3s     | 5s     | 10s    | < 5s   |
| Retrieval       | 50ms   | 100ms  | 150ms  | 300ms  | < 200ms|
| Embedding       | 30ms   | 50ms   | 80ms   | 150ms  | < 100ms|
```

PROMETHEUS METRICS EXAMPLE:
```python
from prometheus_client import Histogram, Counter

INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Time for model inference',
    ['model_name', 'request_type'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

TTFT_LATENCY = Histogram(
    'time_to_first_token_seconds',
    'Time to first token',
    ['model_name'],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

TOKEN_COUNT = Counter(
    'tokens_processed_total',
    'Total tokens processed',
    ['model_name', 'direction']  # direction: input/output
)

REQUEST_COUNT = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['model_name', 'status']  # status: success/error/timeout
)
```

OBSERVABILITY STACK:
```
[Application] --> [OpenTelemetry SDK]
                       |
              +--------+--------+
              |        |        |
         [Traces]  [Metrics]  [Logs]
              |        |        |
         [Jaeger/   [Prometheus] [Loki/
          Tempo]        |       Elasticsearch]
              |        |        |
              +--------+--------+
                       |
                  [Grafana Dashboard]
                       |
                  [Alert Manager -> PagerDuty/Slack]
```

========================================
5.6 LLM-SPECIFIC OBSERVABILITY (2025-2026)
========================================

New tools and standards have emerged specifically for LLM application
monitoring, beyond traditional APM.

LLM OBSERVABILITY PLATFORMS:

| Tool | Type | Key Feature | Best For |
|------|------|-------------|----------|
| LangSmith | Commercial | LangChain ecosystem, tracing | LangChain/LangGraph apps |
| Langfuse | Open Source | Self-hostable, OpenTelemetry | Privacy-conscious, self-hosted |
| Phoenix (Arize) | Open Source | Embeddings analysis, evals | ML teams, experimentation |
| Helicone | Open Source | Gateway-based, one-line setup | Quick start, cost tracking |
| Braintrust | Commercial | Eval-first, CI/CD integration | Prompt regression testing |
| Portkey | Commercial | AI gateway + observability | Multi-provider routing |

OPENTELEMETRY FOR LLM APPLICATIONS:

Two emerging standards for GenAI semantic conventions:
- OpenLLMetry (by Traceloop): OTel instrumentation for LLM frameworks
- OpenInference (by Arize): Semantic conventions for ML observability

```python
# Example: OpenTelemetry tracing for LLM calls
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer("llm-app")

@tracer.start_as_current_span("llm_call")
async def call_llm(prompt: str, model: str):
    span = trace.get_current_span()
    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.prompt_tokens", token_count)

    response = await client.chat.completions.create(...)

    span.set_attribute("gen_ai.usage.completion_tokens", resp_tokens)
    span.set_attribute("gen_ai.response.finish_reason", "stop")
    return response
```

KEY LLM METRICS TO TRACK:
- Time-to-First-Token (TTFT): Critical for streaming UX
- Tokens per Second (TPS): Generation throughput
- Cost per Request: Track across different model tiers
- Cache Hit Rate: For semantic caching effectiveness
- Tool Call Success Rate: For agentic applications
- Evaluation Scores: Automated quality metrics per request

AGENT WORKFLOW TRACING:
For multi-step agent systems, trace the full execution:
```
Trace: "Research and summarize AI trends"
  ├── Span: Planning (120ms, 450 tokens)
  ├── Span: Tool Call - web_search (2.1s)
  ├── Span: Tool Call - web_search (1.8s)
  ├── Span: Analysis (3.2s, 2100 tokens)
  ├── Span: Tool Call - write_document (0.5s)
  └── Span: Final Response (1.1s, 890 tokens)
  Total: 8.8s, 3440 tokens, $0.034
```

> YOUR EXPERIENCE: At RavianAI, monitoring agentic AI workflows requires
> this level of observability -- tracing each agent step, tracking tool call
> success rates, and monitoring costs across different model providers.

========================================
END OF SECTION 5 (Updated February 2026)
========================================
