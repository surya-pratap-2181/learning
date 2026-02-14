---
title: "Microservices for AI"
layout: default
parent: "System Design & Architecture"
nav_order: 3
---

SECTION 3: MICROSERVICES FOR AI
AI System Design Interview Guide for AI/ML Engineers (2025-2026)
## 

3.1 WHEN TO USE MICROSERVICES VS MONOLITH FOR AI

MONOLITH APPROACH (When to choose):

Good for:
- Small team (< 5 engineers)
- Single model serving (one model, one purpose)
- Prototype / MVP stage
- Low traffic (< 1000 req/day)
- Tight latency requirements (no network hops)
- Simple pipeline (input -> model -> output)

Architecture:
```
[Single Service]
   |-- API endpoint
   |-- Preprocessing
   |-- Model inference
   |-- Postprocessing
   |-- All in one container
```

Pros:
- Simple deployment
- No network overhead between components
- Easy debugging
- Single codebase
- Lower infrastructure cost initially

Cons:
- Cannot scale components independently
- Single point of failure
- Hard to use different languages/frameworks per component
- Deployment of entire system for any change
- GPU wasted if mostly doing CPU preprocessing

------

MICROSERVICES APPROACH (When to choose):

Good for:
- Multiple models in production
- Team > 5 engineers with different specializations
- Components need different scaling (GPU vs CPU)
- Components have different update frequencies
- Need to support multiple clients/interfaces
- Production system with high availability requirements

Architecture:
```
[API Gateway]
      |
   +--+--+--+--+
   |     |     |     |
[Pre-    [Model   [Post-   [Feature
 process  Serving  process  Store
 Service] Service] Service] Service]
   CPU     GPU      CPU      CPU
   3 pods  10 pods  3 pods   2 pods
```

Pros:
- Independent scaling (GPU for model, CPU for pre/post processing)
- Independent deployment (update model without touching preprocessing)
- Technology diversity (Python for ML, Go for API gateway)
- Fault isolation (preprocessing failure doesn't crash model server)
- Team autonomy (different teams own different services)

Cons:
- Network latency between services
- Distributed system complexity
- More infrastructure to manage
- Debugging across services is harder
- Data consistency challenges

------

HYBRID APPROACH (RECOMMENDED for most AI systems):

```
[API Gateway (Separate Service)]
      |
[AI Core Service (Monolith-ish)]
   |-- Preprocessing
   |-- Model Inference
   |-- Postprocessing
   (Tightly coupled, deployed together)
      |
[Supporting Microservices]
   |-- Feature Store Service
   |-- Embedding Service
   |-- Cache Service
   |-- Monitoring Service
   |-- Data Pipeline Service
```

Rationale:
- Keep the hot path (preprocess -> infer -> postprocess) together to minimize latency
- Extract supporting services that have different scaling/update needs
- Best of both worlds


## 3.2 SERVICE COMMUNICATION PATTERNS


A) SYNCHRONOUS (Request-Response):

   REST/HTTP:
   - Most common for real-time inference
   - Simple, well-understood
   - Higher latency due to serialization overhead
   ```
   POST /v1/predict
   {
     "input": "text to analyze",
     "model": "sentiment-v2",
     "parameters": {"threshold": 0.8}
   }
   ```

   gRPC:
   - Binary protocol (Protocol Buffers), lower latency
   - Strong typing with .proto definitions
   - Bidirectional streaming support
   - 2-10x faster than REST for model serving
   - Used by TensorFlow Serving, Triton Inference Server

   ```protobuf
   service ModelService {
     rpc Predict(PredictRequest) returns (PredictResponse);
     rpc StreamPredict(PredictRequest) returns (stream TokenResponse);
   }

   message PredictRequest {
     string model_name = 1;
     string input_text = 2;
     InferenceParams params = 3;
   }
   ```

   When to use:
   - REST: external APIs, simple integrations, debugging ease
   - gRPC: internal service communication, high throughput, streaming

B) ASYNCHRONOUS (Message Queue):

   Pattern:
   ```
   [Producer] --> [Message Queue] --> [Consumer (GPU Worker)]
                  (Kafka/SQS/          |
                   RabbitMQ)      [Result Store (Redis)]
                                       |
   [Client polls or receives webhook] <-+
   ```

   When to use:
   - Batch processing (document pipeline, bulk inference)
   - Long-running tasks (video processing, fine-tuning)
   - Decoupling producers from consumers
   - Smoothing traffic spikes (queue absorbs burst)

   Implementation:
   ```python
   # Producer
   async def submit_job(request):
       job_id = str(uuid4())
       await queue.send_message({
           "job_id": job_id,
           "input": request.input,
           "model": request.model,
           "callback_url": request.callback_url
       })
       return {"job_id": job_id, "status": "queued"}

   # Consumer (GPU Worker)
   async def process_job(message):
       result = model.predict(message["input"])
       await result_store.set(message["job_id"], result)
       if message.get("callback_url"):
           await http_client.post(message["callback_url"], json=result)
   ```

C) EVENT-DRIVEN:

   Pattern:
   ```
   [Model Serving] --publishes--> [Event Bus]
                                    |
                          +---------+---------+
                          |         |         |
                   [Monitoring] [Logging] [Alerting]
                   [Service]    [Service] [Service]
   ```

   Use for: Loose coupling, audit trails, analytics, monitoring
   Not for: Hot path inference (too much latency)

D) SIDECAR PATTERN:

   ```
   [Pod]
     |-- [Main: Application Logic]
     |-- [Sidecar: Model Inference (Triton)]
     |-- [Sidecar: Feature Fetcher]
     |-- [Sidecar: Metrics Exporter]
   ```

   Communication via localhost (no network hop)
   Managed by service mesh (Istio, Linkerd)


## 3.3 API DESIGN FOR AI SERVICES


STANDARD AI API DESIGN PATTERNS:

A) PREDICTION ENDPOINT:
```
POST /v1/models/{model_id}/predict
Request:
{
  "inputs": [
    {"text": "This movie was great!"},
    {"text": "Terrible experience."}
  ],
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 0.9
  }
}

Response:
{
  "id": "pred_abc123",
  "model": "sentiment-v2",
  "created": 1706000000,
  "results": [
    {"label": "positive", "score": 0.95},
    {"label": "negative", "score": 0.92}
  ],
  "usage": {
    "input_tokens": 15,
    "output_tokens": 0,
    "total_tokens": 15
  }
}
```

B) STREAMING ENDPOINT (for LLMs):
```
POST /v1/chat/completions
Request:
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}

Response (SSE):
data: {"choices": [{"delta": {"content": "Hello"}}]}
data: {"choices": [{"delta": {"content": "!"}}]}
data: {"choices": [{"delta": {"content": " How"}}]}
data: [DONE]
```

C) ASYNC/BATCH ENDPOINT:
```
POST /v1/batch/predict
Request:
{
  "inputs": [...1000 items...],
  "callback_url": "https://myapp.com/webhook/results",
  "priority": "low"
}

Response:
{
  "batch_id": "batch_xyz789",
  "status": "queued",
  "estimated_completion": "2025-01-15T10:30:00Z"
}

GET /v1/batch/batch_xyz789/status
Response:
{
  "batch_id": "batch_xyz789",
  "status": "processing",
  "progress": {"completed": 450, "total": 1000}
}
```

D) MODEL METADATA ENDPOINT:
```
GET /v1/models/{model_id}
Response:
{
  "id": "sentiment-v2",
  "version": "2.1.0",
  "created": "2025-01-10",
  "input_schema": {...},
  "output_schema": {...},
  "max_tokens": 4096,
  "supported_languages": ["en", "es", "fr"],
  "pricing": {"input": 0.001, "output": 0.002}
}
```

API DESIGN BEST PRACTICES FOR AI:
1. Always return request ID for traceability
2. Include usage/token counts in response
3. Support both sync and async modes
4. Version your API (v1, v2) in URL path
5. Return confidence scores alongside predictions
6. Implement proper error codes:
   - 400: Invalid input (bad prompt, too many tokens)
   - 413: Input too large
   - 429: Rate limit exceeded (include retry-after header)
   - 503: Model loading / GPU unavailable
   - 504: Inference timeout


## 3.4 VERSIONING ML MODELS IN PRODUCTION


CHALLENGES:
- Models change independently from code
- Need to support multiple model versions simultaneously
- A/B testing between model versions
- Rollback if new model performs worse
- Model + code + config all need to be versioned together

MODEL VERSIONING STRATEGIES:

A) URL-BASED VERSIONING:
```
POST /v1/models/sentiment-v2/predict    # API version 1, Model version 2
POST /v2/models/sentiment-v3/predict    # API version 2, Model version 3
```

B) HEADER-BASED VERSIONING:
```
POST /predict
Headers:
  X-Model-Version: sentiment-v2.1.0
  X-API-Version: 2024-01-15
```

C) MODEL REGISTRY PATTERN:
```
[Model Registry (MLflow / Weights & Biases / Vertex AI)]
   |
   |-- Model: sentiment-classifier
   |     |-- v1.0.0 (staging)
   |     |-- v2.0.0 (production, 90% traffic)
   |     |-- v2.1.0 (canary, 10% traffic)
   |     |-- v1.9.0 (archived)
   |
   |-- Each version includes:
         |-- Model weights (S3/GCS path)
         |-- Training config
         |-- Evaluation metrics
         |-- Input/output schema
         |-- Dependencies (Python packages)
         |-- Training data hash
```

D) MODEL DEPLOYMENT STRATEGIES:

   Blue-Green Deployment:
   ```
   [Load Balancer]
        |
   [Blue: v2.0 (current production)]  <-- 100% traffic
   [Green: v2.1 (new version)]        <-- 0% traffic (testing)

   After validation:
   [Blue: v2.0]   <-- 0% traffic
   [Green: v2.1]  <-- 100% traffic (now production)
   ```

   Canary Deployment:
   ```
   [Load Balancer]
        |
   [v2.0] <-- 95% traffic
   [v2.1] <-- 5% traffic (canary)

   Monitor metrics for canary...
   If good: gradually increase to 100%
   If bad: rollback to 0%
   ```

   Shadow Deployment:
   ```
   [Load Balancer]
        |
   [v2.0] <-- 100% traffic (serves response)
   [v2.1] <-- 100% traffic (shadow, response discarded)

   Compare outputs of v2.0 and v2.1 offline
   No risk to users
   ```

E) ARTIFACT VERSIONING:
```
model-artifacts/
  sentiment-classifier/
    v2.1.0/
      model.onnx              # Model weights
      tokenizer/               # Tokenizer files
      config.yaml              # Model config
      requirements.txt         # Dependencies
      evaluation_report.json   # Metrics on test set
      training_config.yaml     # Hyperparameters
      data_manifest.json       # Training data reference
```

BEST PRACTICES:
1. Use semantic versioning: MAJOR.MINOR.PATCH
   - MAJOR: Breaking schema change
   - MINOR: New capabilities, backward compatible
   - PATCH: Bug fix, retraining with same architecture
2. Always keep at least 2 previous versions deployable
3. Automate rollback triggers (if accuracy drops > 5%, auto-rollback)
4. Tag models with training data version
5. Use model cards for documentation

## END OF SECTION 3

