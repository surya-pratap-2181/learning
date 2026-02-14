---
title: "Scalability"
layout: default
parent: "System Design & Architecture"
nav_order: 4
---

SECTION 4: SCALABILITY FOR AI SYSTEMS
AI System Design Interview Guide for AI/ML Engineers (2025-2026)
## 

4.1 HORIZONTAL SCALING

DEFINITION: Adding more machines/instances to handle more load.

FOR AI SERVICES:

A) STATELESS INFERENCE SERVERS:
   ```
   [Load Balancer]
        |
   [Instance 1: vLLM + Model A]
   [Instance 2: vLLM + Model A]
   [Instance 3: vLLM + Model A]
   ... add more instances as traffic grows
   ```

   Requirements for horizontal scaling:
   - Inference servers must be stateless
   - Session state stored externally (Redis)
   - Model weights loaded from shared storage (S3/NFS)
   - Health checks for instance readiness
   - Graceful shutdown (finish in-flight requests)

B) DATABASE SCALING:
   Vector Database (Pinecone, Weaviate, Qdrant):
   - Shard by collection or partition key
   - Replicate for read throughput
   - Example: 100M vectors -> 10 shards of 10M each

   Feature Store:
   - Redis Cluster for online features
   - Partitioned by entity ID
   - Read replicas for high-read workloads

C) EMBEDDING SERVICE SCALING:
   ```
   [Embedding Requests]
         |
   [Batch Aggregator (collects requests for 10ms)]
         |
   [GPU Instance 1: batch of 64 texts]
   [GPU Instance 2: batch of 64 texts]
   [GPU Instance 3: batch of 64 texts]
   ```
   - Micro-batching for GPU efficiency
   - Scale instances based on queue depth


## 4.2 VERTICAL SCALING


DEFINITION: Using more powerful hardware (bigger GPU, more RAM).

FOR AI SERVICES:

GPU HIERARCHY (2025):
```
Edge/Dev:         T4 (16GB)         - Small models, INT8
Mid-range:        L4 (24GB)         - Medium models, good throughput
Production:       A10G (24GB)       - Good balance cost/performance
High-end:         A100 (40/80GB)    - Large models, training + inference
Ultra:            H100 (80GB)       - Largest models, highest throughput
Cutting edge:     H200 (141GB)      - Maximum memory for huge models
Multi-GPU:        8x H100 (640GB)   - 70B+ models with tensor parallelism
```

WHEN TO VERTICALLY SCALE:
- Model doesn't fit in current GPU memory
- Need lower latency (faster GPU = faster inference)
- Memory-bound workloads (large batch sizes)
- Before horizontal scaling (simpler, if cost-effective)

VERTICAL SCALING LIMITS:
- Single GPU maxes out at ~141GB (H200)
- For models > 141GB, must go to multi-GPU (tensor parallelism)
- Cost grows faster than linearly
- Single point of failure

DECISION FRAMEWORK:
```
Model Size:
  < 7B params:    Single T4/L4 (with quantization)
  7B-13B:         Single A10G/A100-40GB
  13B-30B:        Single A100-80GB or 2x A10G (tensor parallel)
  30B-70B:        2-4x A100-80GB (tensor parallel)
  70B-180B:       4-8x A100/H100 (tensor + pipeline parallel)
  180B+:          8+ H100s across multiple nodes
```


## 4.3 AUTO-SCALING BASED ON INFERENCE LOAD


METRICS FOR AUTO-SCALING DECISIONS:

1. GPU Utilization:
   - Scale up when avg GPU util > 70% for 5 min
   - Scale down when avg GPU util < 30% for 15 min
   - Collect via DCGM Exporter -> Prometheus

2. Request Queue Depth:
   - Scale up when queue depth > 100 requests
   - More responsive than GPU utilization
   - Direct measure of user-facing impact

3. Inference Latency (P95):
   - Scale up when P95 latency > SLA threshold
   - Scale down when P95 latency < 50% of threshold
   - Best user-experience-driven metric

4. Tokens Per Second (TPS):
   - Throughput metric
   - Scale based on TPS approaching capacity

AUTO-SCALING ARCHITECTURE:
```
[Prometheus / CloudWatch]
   |-- GPU utilization metrics
   |-- Queue depth metrics
   |-- Latency metrics
        |
[Custom Metrics Adapter / KEDA]
   |-- Evaluates scaling rules
   |-- Makes scaling decisions
        |
[Kubernetes HPA / Cloud Auto-Scaler]
   |-- Scales pod replicas
   |-- Requests new GPU nodes if needed
        |
[Cluster Auto-Scaler]
   |-- Provisions new GPU nodes
   |-- 2-10 minute delay for new nodes
        |
[Warm Pool Strategy]
   |-- Keep N standby instances with model loaded
   |-- Instant scale-up (no model loading delay)
   |-- Trade-off: cost of idle instances vs response time
```

KUBERNETES HPA EXAMPLE:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-serving
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"
  - type: Pods
    pods:
      metric:
        name: request_queue_depth
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 4
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

KEDA (Kubernetes Event-Driven Autoscaling):
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-serving-scaler
spec:
  scaleTargetRef:
    name: llm-serving
  minReplicaCount: 2
  maxReplicaCount: 20
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: avg_inference_latency_p95
      threshold: "500"  # ms
      query: |
        histogram_quantile(0.95,
          rate(inference_duration_seconds_bucket[5m]))
```

CHALLENGES:
- GPU node provisioning is slow (3-10 minutes)
- Model loading adds delay (30s - 5min depending on model size)
- Solution: warm pools, pre-loaded standby instances
- Predictive scaling: use traffic patterns to pre-scale before peak


## 4.4 GPU ALLOCATION STRATEGIES


A) DEDICATED GPU ALLOCATION:
   - Each model gets dedicated GPU(s)
   - No interference between models
   - Wasteful if models have variable traffic
   ```
   GPU 0: Model A (100% allocated)
   GPU 1: Model B (100% allocated)
   GPU 2: Model C (100% allocated)
   ```

B) GPU SHARING (MIG - Multi-Instance GPU):
   - NVIDIA A100/H100 can be partitioned into isolated instances
   - Each partition has dedicated memory and compute
   - Good for running multiple small models on one GPU
   ```
   A100 (80GB) partitioned:
     MIG Instance 1: 40GB - Large model
     MIG Instance 2: 20GB - Medium model
     MIG Instance 3: 20GB - Small model
   ```

C) GPU TIME-SHARING (MPS - Multi-Process Service):
   - Multiple processes share GPU concurrently
   - No memory isolation (risk of OOM)
   - Good for small models that don't fully utilize GPU
   ```
   GPU 0: [Model A: 30% compute] [Model B: 30% compute] [Model C: 40% compute]
   ```

D) DYNAMIC GPU ALLOCATION:
   - KServe ModelMesh pattern
   - Models loaded/unloaded based on demand
   - LRU eviction when GPU memory full
   ```
   [Model Registry: 100 models available]
        |
   [Model Mesh Controller]
        |
   [GPU Pool: 10 GPUs]
     Currently loaded:
     GPU 0: Model A (hot, serving traffic)
     GPU 1: Model B (hot, serving traffic)
     GPU 2: Model C (warm, loaded but idle)
     GPU 3: Model D (loading...)
     ...
   Models not in GPU are cold (need loading time)
   ```

E) FRACTIONAL GPU WITH KUBERNETES:
   ```yaml
   # Request a fraction of GPU
   resources:
     limits:
       nvidia.com/gpu: 1    # Full GPU
     # OR with time-slicing:
       nvidia.com/gpu: "0.5"  # Half a GPU (NVIDIA GPU Operator)
   ```

GPU ALLOCATION DECISION MATRIX:
```
| Scenario                    | Strategy                  |
|-----------------------------|---------------------------|
| Single large model          | Dedicated, multi-GPU      |
| Many small models           | MIG or time-sharing       |
| Variable traffic per model  | Dynamic (ModelMesh)       |
| Latency-critical            | Dedicated (no contention) |
| Cost-optimized              | Sharing + time-slicing    |
| Dev/test                    | Time-sharing              |
```


## 4.5 BATCH INFERENCE VS REAL-TIME INFERENCE


REAL-TIME INFERENCE:
```
[Single Request] --> [Model] --> [Single Response]
Latency: 50ms - 5s
Use cases: Chatbots, search, real-time recommendations
```

Characteristics:
- Low latency requirement (< 500ms for most, < 2s for LLMs)
- Single or small batch requests
- GPU always warm (model loaded in memory)
- Higher cost per prediction
- Auto-scaling based on request rate

BATCH INFERENCE:
```
[1000s of Requests] --> [Batch Queue] --> [Model (high throughput)] --> [Results Store]
Latency: minutes to hours
Use cases: Email classification, content moderation backlog, analytics
```

Characteristics:
- High throughput requirement
- Can use spot/preemptible instances (70% cheaper)
- Optimal GPU utilization (large batch sizes)
- Lower cost per prediction
- Can run during off-peak hours

MICRO-BATCHING (Hybrid):
```
[Individual Requests] --> [Aggregator (10-50ms window)]
                              |
                         [Batch of N requests]
                              |
                         [Model (batch inference)]
                              |
                         [Scatter results back to individual requests]
```

- Combines low latency with better GPU utilization
- Aggregates requests over short window (10-100ms)
- Processes as batch on GPU
- Scatters results back to individual callers
- Used by vLLM continuous batching

COMPARISON TABLE:
```
| Aspect              | Real-Time      | Batch           | Micro-Batch    |
|---------------------|----------------|-----------------|----------------|
| Latency             | < 500ms        | Minutes-Hours   | 50-200ms       |
| Throughput          | Low-Medium     | Very High       | High           |
| GPU Utilization     | 20-60%         | 80-95%          | 60-85%         |
| Cost per prediction | High           | Very Low        | Medium         |
| Infrastructure      | Always-on      | Spot instances   | Always-on      |
| Use case            | User-facing    | Background jobs  | User-facing    |
|                     | APIs           | Analytics        | high-throughput |
```

WHEN TO USE EACH:
- Real-time: User is waiting for response (chat, search, API)
- Batch: User doesn't need immediate result (reports, bulk processing)
- Micro-batch: High-throughput API with flexible latency (50-200ms OK)

COST COMPARISON EXAMPLE (processing 1M predictions):
```
Real-time (A100 on-demand):
  - 1M predictions / 100 predictions per second = 10,000 seconds
  - A100 cost: $3.50/hr = ~$10 for the job
  - But GPU idle 50% of time between requests = ~$20 effective

Batch (A100 spot instance):
  - 1M predictions / 500 predictions per second (large batch) = 2,000 seconds
  - Spot price: $1.20/hr = ~$0.67 for the job
  - 30x cheaper than real-time
```

## END OF SECTION 4

