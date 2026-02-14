================================================================================
  SECTION 8: COMMON INTERVIEW QUESTIONS WITH DETAILED ANSWERS (2025-2026)
================================================================================

These are scenario-based and system design questions frequently asked in
Cloud/DevOps for AI Engineer interviews.

##############################################################################
# QUESTION 1: SYSTEM DESIGN
##############################################################################

Q: Design a production ML inference system that serves a large language model
   (LLM) with <200ms latency, handles 10,000 requests per minute, and costs
   under $50,000/month.

Answer:

Architecture:
  Client -> CloudFront (CDN, cache identical prompts)
         -> API Gateway (throttling, auth, WebSocket for streaming)
         -> Network Load Balancer
         -> EKS Cluster with GPU nodes
             -> vLLM / TGI serving framework
             -> Model: Llama 3.1 8B (quantized INT8)
         -> Redis (semantic cache for similar queries)
         -> S3 (request/response logging)
         -> CloudWatch + Prometheus/Grafana (monitoring)

Key design decisions:

1. Model choice: Llama 3.1 8B (quantized INT8) instead of 70B.
   - 8B parameter model fits in single GPU (8GB in INT8).
   - INT8 quantization: Minimal quality loss, 4x memory reduction.
   - Latency: ~50-100ms for short completions on A10G GPU.

2. Serving framework: vLLM with PagedAttention.
   - Continuous batching: Process new requests without waiting.
   - PagedAttention: Efficient KV-cache management.
   - Throughput: 3-5x higher than naive serving.

3. Infrastructure:
   - 8x g5.xlarge instances (1x A10G each): 8 * $1.006/hr = $8.05/hr.
   - With Savings Plan (1-year): ~$4.83/hr = $3,478/month.
   - Reserve 4 instances (base), auto-scale 4-12 (on-demand/spot).
   - Estimated total: ~$8,000-15,000/month (well under $50K).

4. Caching:
   - Exact match cache (Redis): Cache identical prompts.
   - Semantic cache: Embed queries, cache similar queries.
   - Expected cache hit rate: 20-40% for many applications.

5. Auto-scaling:
   - HPA based on GPU utilization (target: 70%) and queue depth.
   - Scale-out: 30 second stabilization.
   - Scale-in: 5 minute stabilization (avoid thrashing).

6. Latency breakdown:
   - Network (client -> API GW -> NLB): ~10ms
   - Queue wait: ~5-20ms
   - Inference (prefill + decode): ~50-150ms
   - Total: ~70-180ms (within SLA)

7. Availability:
   - Multi-AZ deployment.
   - Min 4 pods (survive 1-2 pod failures).
   - Rolling updates for model changes.
   - Health checks every 10 seconds.

##############################################################################
# QUESTION 2: MLOPS PIPELINE
##############################################################################

Q: Describe how you would build an end-to-end MLOps pipeline for a
   recommendation model that retrains daily on new user interaction data.

Answer:

Pipeline overview:
  1. Data Collection -> 2. Feature Engineering -> 3. Training ->
  4. Evaluation -> 5. Registration -> 6. Deployment -> 7. Monitoring

Detailed steps:

1. Data Collection (Event-driven):
   - User interactions captured via Kinesis Data Streams.
   - Kinesis Firehose delivers to S3 (raw data lake) in Parquet format.
   - Partitioned by date: s3://data/interactions/dt=2025-03-15/
   - Schema validation via AWS Glue Schema Registry.

2. Feature Engineering (Scheduled):
   - Daily Airflow DAG triggers at 2 AM.
   - Spark job on EMR Serverless computes features:
     * User features: purchase history, browse patterns, demographics.
     * Item features: popularity, category, price, availability.
     * Interaction features: click-through rate, time spent, recency.
   - Features materialized to Feast feature store:
     * Online store (DynamoDB): Latest features for serving.
     * Offline store (S3): Historical features for training.

3. Training (Triggered by feature pipeline):
   - SageMaker Training Job with spot instances.
   - Model: Two-tower retrieval model + ranking model.
   - Hyperparameters versioned in Git, logged to MLflow.
   - Checkpointing to S3 every 1000 steps.
   - Training data: Point-in-time join from Feast offline store.
   - Duration: ~2-4 hours on ml.p4d.24xlarge spot.

4. Evaluation (Automated gate):
   - Offline metrics: AUC, NDCG@10, MAP@10.
   - Comparison against current production model.
   - Gate: New model must improve NDCG@10 by >= 0.5%.
   - Fairness check: Performance across user demographics.
   - Latency check: Inference p99 <= 50ms on target hardware.

5. Registration:
   - If evaluation passes: Register in MLflow Model Registry.
   - Auto-transition to "Staging" stage.
   - Model card generated with metrics, data summary, bias report.
   - Slack notification to ML team for review.

6. Deployment (Canary):
   - Deploy to staging endpoint, run integration tests.
   - Canary deployment to production:
     Step 1: 5% traffic for 1 hour.
     Step 2: 25% traffic for 2 hours.
     Step 3: 100% traffic.
   - Automated rollback if error rate > 1% or latency p99 > 100ms.
   - SageMaker deployment guardrails with CloudWatch alarms.

7. Monitoring (Continuous):
   - Data drift: Compare daily feature distributions (Evidently AI).
   - Model quality: Track NDCG@10 with delayed ground truth (1-7 day lag).
   - Operational: Latency, throughput, error rates (CloudWatch).
   - Business metrics: Click-through rate, revenue per user (Amplitude/Mixpanel).
   - Alert thresholds:
     * Data drift PSI > 0.2 -> Investigate.
     * NDCG@10 drops > 3% -> Trigger emergency retraining.
     * Error rate > 0.5% -> Page on-call engineer.

Tools stack:
  - Orchestration: Apache Airflow (MWAA).
  - Feature Store: Feast.
  - Experiment Tracking: MLflow.
  - Training: SageMaker + Spot.
  - Serving: SageMaker Real-time Endpoints.
  - Monitoring: CloudWatch + Evidently AI + Grafana.
  - IaC: Terraform.
  - CI/CD: GitHub Actions.

##############################################################################
# QUESTION 3: CONTAINERIZATION & DEPLOYMENT
##############################################################################

Q: How would you containerize and deploy a computer vision model that
   processes images in real-time on Kubernetes?

Answer:

1. Model preparation:
   - Convert PyTorch model to TensorRT for GPU-optimized inference.
   - Input: 224x224 RGB image.
   - Output: Class probabilities (1000 classes).
   - TensorRT engine built for specific GPU (e.g., A10G).

2. Dockerfile (multi-stage):
   # Build stage: Convert model
   FROM nvcr.io/nvidia/tensorrt:24.01-py3 AS builder
   COPY model/resnet50.onnx /workspace/
   RUN trtexec --onnx=/workspace/resnet50.onnx \
       --saveEngine=/workspace/model.trt \
       --fp16 --workspace=4096

   # Runtime stage: Serve model
   FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS runtime
   COPY --from=builder /workspace/model.trt /models/resnet50/1/model.plan
   COPY config/model_config.pbtxt /models/resnet50/config.pbtxt

   EXPOSE 8000 8001 8002
   CMD ["tritonserver", "--model-repository=/models"]

3. Kubernetes deployment:
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: vision-model
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: vision-model
     template:
       metadata:
         labels:
           app: vision-model
       spec:
         tolerations:
         - key: nvidia.com/gpu
           operator: Exists
           effect: NoSchedule
         containers:
         - name: triton
           image: 123456789.dkr.ecr.us-east-1.amazonaws.com/vision:v1.0
           ports:
           - containerPort: 8000  # HTTP
           - containerPort: 8001  # gRPC
           - containerPort: 8002  # Metrics
           resources:
             limits:
               nvidia.com/gpu: 1
               memory: "8Gi"
             requests:
               nvidia.com/gpu: 1
               memory: "4Gi"
           readinessProbe:
             httpGet:
               path: /v2/health/ready
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           livenessProbe:
             httpGet:
               path: /v2/health/live
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 30

4. Service + Ingress:
   apiVersion: v1
   kind: Service
   metadata:
     name: vision-model-svc
   spec:
     selector:
       app: vision-model
     ports:
     - name: http
       port: 8000
       targetPort: 8000
     - name: grpc
       port: 8001
       targetPort: 8001

5. Auto-scaling:
   - HPA on custom metric: inference requests per second per pod.
   - Target: 100 req/sec/pod (Triton can handle 500+ with batching).
   - Min replicas: 2 (availability).
   - Max replicas: 10.

##############################################################################
# QUESTION 4: SECURITY
##############################################################################

Q: How do you secure an AI/ML pipeline end-to-end on AWS?

Answer:

1. Network security:
   - VPC with private subnets for all ML resources.
   - VPC endpoints for S3, SageMaker, ECR, CloudWatch (no internet).
   - SageMaker network isolation: Training containers can't access internet.
   - Security groups: Restrict traffic to known ports/sources.
   - NACLs: Additional subnet-level controls.

2. Identity and access:
   - IAM roles with least privilege (not IAM users).
   - SageMaker execution roles scoped to specific S3 buckets.
   - Service control policies: Prevent launching p5 instances in dev accounts.
   - MFA for console access.
   - OIDC federation for CI/CD (GitHub Actions -> AWS).

3. Data protection:
   - Encryption at rest: KMS (customer-managed keys) for S3, EBS, SageMaker.
   - Encryption in transit: TLS 1.2+ for all API calls.
   - S3 bucket policies: Deny unencrypted uploads.
   - Data classification: Tag datasets with sensitivity level.

4. Model security:
   - Model artifact encryption (KMS).
   - ECR image scanning for vulnerabilities.
   - Container image signing (AWS Signer).
   - Input validation: Sanitize inference requests.
   - Adversarial input detection: Monitor for adversarial attacks.

5. Secrets management:
   - AWS Secrets Manager for API keys, database credentials.
   - SSM Parameter Store for configuration.
   - Never hardcode secrets in code, containers, or environment variables.
   - Rotate secrets automatically.

6. Logging and auditing:
   - CloudTrail: API activity logging.
   - S3 access logs: Track data access.
   - SageMaker Experiments: Track who trained what, when.
   - CloudWatch Logs: Application-level logging.
   - GuardDuty: Threat detection.

7. Compliance:
   - SageMaker Model Cards: Document model purpose, limitations, bias.
   - SageMaker Clarify: Bias detection and fairness reports.
   - Data lineage tracking: Know which data trained which model.
   - Audit trail: Complete history of model versions and deployments.

##############################################################################
# QUESTION 5: TROUBLESHOOTING
##############################################################################

Q: Your production ML model's latency has increased from 100ms to 500ms
   over the past week. How do you diagnose and fix this?

Answer:

Investigation steps:

1. Check operational metrics (immediate):
   - CloudWatch: CPU, GPU, memory utilization.
   - Is GPU utilization at 100%? -> Scaling issue.
   - Is memory usage high? -> Memory leak or increased payload size.
   - Has request volume increased? -> Need to scale out.
   - Network I/O? -> Downstream dependency slowness.

2. Check for changes (last 7 days):
   - Was there a model update? -> Model performance regression.
   - Was there a code deployment? -> Check diff for issues.
   - Was there an infrastructure change? -> Terraform/CloudFormation changes.
   - Was there a data drift? -> Input features changed distribution.

3. Analyze request patterns:
   - Are all requests slow or specific ones? -> If specific, check input size.
   - Is latency gradual increase or sudden? -> Gradual = drift, sudden = change.
   - Time-of-day pattern? -> Traffic-correlated.

4. Specific ML-related causes:
   a. Input data changes:
      - Feature values outside training distribution.
      - Larger input payloads (longer text, higher-res images).
      -> Fix: Add input validation, truncation, or resize.

   b. Model issues:
      - KV-cache growing (LLMs with longer conversations).
      - Memory fragmentation.
      -> Fix: Restart pods periodically, implement KV-cache limits.

   c. Resource contention:
      - Other pods on same GPU node competing for resources.
      - CPU preprocessing bottleneck.
      -> Fix: Dedicated node pools, resource limits.

   d. Dependency degradation:
      - Feature store (Feast/DynamoDB) latency increased.
      - S3 model loading slower.
      -> Fix: Check dependency metrics, add caching.

5. Immediate mitigations:
   - Scale out: Increase instance count.
   - Restart: Rolling restart of inference pods (clears memory).
   - Rollback: Revert to previous model/code version if recent change.
   - Cache: Enable or increase caching for frequent queries.

6. Long-term fixes:
   - Implement auto-scaling based on latency metrics.
   - Add circuit breakers for dependency calls.
   - Set up latency alarms at 150ms (before SLA breach).
   - Regular load testing in staging.
   - Model optimization (quantization, pruning).

##############################################################################
# QUESTION 6: COST SCENARIO
##############################################################################

Q: Your team is spending $200,000/month on AI infrastructure. The CEO wants
   to cut costs by 40% without impacting service quality. What do you do?

Answer:

Phase 1 - Quick wins (Week 1-2, target 15-20% savings):
  1. Identify and terminate unused resources:
     - Dev/staging endpoints left running. Expected savings: $10-20K.
     - Oversized notebook instances. Expected savings: $2-5K.
     - Unused EBS volumes and old snapshots. Expected savings: $1-3K.

  2. Right-size instances:
     - Review GPU utilization. If <30%, downsize.
     - g5.4xlarge -> g5.xlarge if single GPU sufficient. Expected: $5-10K.

  3. Schedule non-production resources:
     - Auto-stop dev endpoints after business hours (save 60% of dev costs).
     - Expected savings: $5-10K.

Phase 2 - Medium-term (Week 3-6, target 15-20% savings):
  4. Reserved instances / Savings Plans:
     - Commit to 1-year Savings Plan for production endpoints.
     - Production spend ~$80K/month -> $48K with 40% savings.
     - Expected savings: $32K.

  5. Spot instances for training:
     - Move all training to spot instances.
     - Training spend ~$30K/month -> $9K with 70% savings.
     - Expected savings: $21K.

  6. Model optimization:
     - Quantize production models (FP16 -> INT8).
     - Can serve on smaller instances or fewer instances.
     - Expected savings: $5-10K.

Phase 3 - Long-term (Week 6-12, target remaining savings):
  7. Multi-model endpoints:
     - Consolidate 20 small models onto 3 multi-model endpoints.
     - Expected savings: $5-10K.

  8. Caching layer:
     - Add Redis cache for frequent predictions.
     - Reduce inference calls by 30%.
     - Expected savings: $3-5K.

  9. Serverless for low-traffic models:
     - Move 10 low-traffic models to Lambda/SageMaker Serverless.
     - Expected savings: $5-8K.

  10. Data storage optimization:
      - Lifecycle policies on S3 (move old data to Glacier).
      - Expected savings: $2-5K.

Total expected savings: $80-100K/month (40-50% reduction).

##############################################################################
# QUESTION 7: AWS vs AZURE DECISION
##############################################################################

Q: Your company is choosing between AWS and Azure for a new AI platform.
   What factors would you consider?

Answer:

1. Model requirements:
   - Need OpenAI models (GPT-4, o1)? -> Azure (exclusive partnership).
   - Need multi-provider choice (Claude, Llama, Mistral)? -> AWS Bedrock.
   - Need custom training at scale? -> Both capable; AWS has Trainium.

2. Existing ecosystem:
   - Already on Azure (AD, Office 365)? -> Azure reduces integration effort.
   - Already on AWS? -> SageMaker ecosystem is mature.
   - Multi-cloud strategy? -> Both + Terraform for portability.

3. Team expertise:
   - Team knows AWS? -> Lower learning curve, faster delivery.
   - Team knows Kubernetes? -> Both have managed K8s (EKS/AKS).

4. Compliance requirements:
   - Government/FedRAMP? -> Both have GovCloud.
   - EU data residency? -> Both have EU regions.
   - Healthcare (HIPAA)? -> Both compliant.

5. Cost comparison (get quotes for specific workloads):
   - GPU pricing varies by region and availability.
   - Compare: Savings Plans, Reserved Instances, spot pricing.
   - Consider egress costs for data-heavy workloads.

6. Specific service comparison:
   - ML Platform: SageMaker (more mature) vs Azure ML (improving fast).
   - LLM APIs: Bedrock (multi-provider) vs Azure OpenAI (GPT-exclusive).
   - Container: EKS (more adoption) vs AKS (easier Azure AD integration).
   - Serverless: Lambda (more mature) vs Functions (good enough).

##############################################################################
# QUESTION 8: KUBERNETES FOR ML
##############################################################################

Q: How do you set up Kubernetes for ML model serving with proper resource
   management, scaling, and monitoring?

Answer:

1. Cluster setup:
   - Separate node pools: CPU (general), GPU-inference, GPU-training.
   - GPU nodes: Taints to prevent non-GPU workloads from scheduling.
   - NVIDIA device plugin for GPU visibility.
   - Container runtime: containerd with NVIDIA runtime hook.

2. Resource management:
   - Resource requests and limits for every pod.
   - GPU: nvidia.com/gpu: 1 in limits (guaranteed allocation).
   - Memory: Set based on model size + overhead.
   - CPU: Set for preprocessing workload.
   - Pod Priority Classes: Production > staging > batch.

   apiVersion: scheduling.k8s.io/v1
   kind: PriorityClass
   metadata:
     name: production-inference
   value: 1000000
   globalDefault: false
   description: "Priority for production inference workloads"

3. Serving framework options:
   - Triton Inference Server: Multi-framework, dynamic batching, ensemble.
   - TorchServe: PyTorch-native, easy setup.
   - TensorFlow Serving: TF-native, gRPC + REST.
   - vLLM: Optimized for LLM serving (PagedAttention).
   - KServe: K8s-native model serving (CRDs for InferenceService).
   - Seldon Core: Advanced ML deployment (A/B testing, canary built-in).
   - Ray Serve: Scalable, Python-native serving.

4. KServe example:
   apiVersion: serving.kserve.io/v1beta1
   kind: InferenceService
   metadata:
     name: sentiment-model
   spec:
     predictor:
       model:
         modelFormat:
           name: pytorch
         storageUri: "s3://models/sentiment/v2/"
         resources:
           limits:
             nvidia.com/gpu: 1
             memory: 8Gi
           requests:
             nvidia.com/gpu: 1
             memory: 4Gi
       minReplicas: 2
       maxReplicas: 10
     transformer:
       containers:
       - name: preprocessor
         image: my-preprocessor:v1
         resources:
           limits:
             memory: 2Gi
             cpu: "2"

5. Monitoring stack:
   - Prometheus: Scrape metrics from model servers.
   - Grafana: Dashboards for latency, throughput, GPU util.
   - Custom metrics: Prediction distribution, confidence scores.
   - DCGM Exporter: NVIDIA GPU metrics (temperature, utilization, memory).

   # ServiceMonitor for Prometheus
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: model-server-monitor
   spec:
     selector:
       matchLabels:
         app: vision-model
     endpoints:
     - port: metrics
       interval: 15s

##############################################################################
# QUESTION 9: RAG SYSTEM DESIGN
##############################################################################

Q: Design a production RAG (Retrieval-Augmented Generation) system for a
   company's internal knowledge base with 10 million documents.

Answer:

Architecture:
  User Query
    -> API Gateway (auth, rate limiting)
    -> Query Processing Service
        -> Query rewriting (LLM reformulates ambiguous queries)
        -> Embedding generation (text-embedding-3-large)
    -> Retrieval Layer
        -> Hybrid search (vector + keyword) on OpenSearch
        -> Reranking (cross-encoder reranker)
        -> Top-K context selection (5-10 chunks)
    -> Generation Layer
        -> LLM (GPT-4o or Claude via Bedrock)
        -> System prompt + retrieved context + user query
        -> Response with inline citations
    -> Response to user

Components:

1. Document ingestion pipeline:
   - Source connectors: SharePoint, Confluence, S3, databases.
   - Document processing: Extract text (Textract/Document Intelligence).
   - Chunking strategy: Semantic chunking (500-1000 tokens, 10% overlap).
   - Metadata extraction: Title, author, date, department, tags.
   - Embedding: text-embedding-3-large (3072 dimensions).
   - Storage: OpenSearch with vector + keyword indices.
   - Schedule: Incremental sync every 4 hours.

2. Vector database sizing:
   - 10M documents * ~20 chunks/doc = 200M vectors.
   - 200M * 3072 dims * 4 bytes (FP32) = ~2.3 TB vector storage.
   - Use HNSW index for ANN search.
   - OpenSearch cluster: 6x r6g.4xlarge.search instances.

3. Retrieval optimization:
   - Hybrid search: BM25 keyword + cosine similarity vector.
   - Reciprocal Rank Fusion (RRF) to combine results.
   - Cross-encoder reranker: Rerank top-50 to select top-5.
   - Metadata filtering: Filter by department, date range, document type.

4. Generation:
   - Context window management: Fit top chunks within context limit.
   - Prompt engineering: System prompt with citation format instructions.
   - Streaming: Return tokens as generated for perceived speed.
   - Guardrails: Content safety filters, PII redaction.

5. Evaluation:
   - Retrieval quality: MRR@10, Recall@10, NDCG.
   - Answer quality: Groundedness, relevance, coherence, faithfulness.
   - Automated eval: LLM-as-judge (GPT-4 rates answers).
   - Human eval: Weekly sampling of 100 Q&A pairs.

6. Cost estimate:
   - OpenSearch cluster: ~$8,000/month.
   - Embedding generation (initial): ~$2,000 one-time.
   - LLM inference (10K queries/day): ~$3,000-5,000/month.
   - Compute (API servers, reranker): ~$2,000/month.
   - Total: ~$15,000/month.

##############################################################################
# QUESTION 10: BEHAVIORAL / SCENARIO QUESTIONS
##############################################################################

Q: Tell me about a time you reduced ML infrastructure costs significantly.

Answer framework (STAR method):
- Situation: "Our ML team was spending $150K/month on AWS infrastructure for
  serving 15 production models. The budget was growing 20% quarter-over-quarter."
- Task: "I was tasked with reducing costs by 30% without impacting SLAs."
- Action: "I conducted a comprehensive audit and implemented:
  1. Consolidated 15 endpoints into 3 multi-model endpoints (saved $40K).
  2. Switched training from on-demand to managed spot (saved $15K).
  3. Implemented auto-scaling with scale-to-zero for 5 low-traffic models.
  4. Quantized 3 large models from FP32 to INT8 (smaller instances needed).
  5. Added Redis caching for top 1000 most common queries (30% cache hit rate)."
- Result: "Reduced monthly costs from $150K to $95K (37% reduction) while
  maintaining p99 latency within SLA. Also improved deployment velocity by
  standardizing on multi-model endpoints."

---

Q: A model in production is returning biased predictions for a certain
   demographic group. How do you handle this?

Answer:
1. Immediate: Assess severity. If harmful, consider taking model offline or
   falling back to rule-based system.
2. Investigate: Use SageMaker Clarify or Fairlearn to quantify bias.
   Check: Disparate impact ratio, equal opportunity difference.
3. Root cause: Was training data imbalanced? Are proxy features leaking
   protected attributes? Is the label itself biased?
4. Fix:
   - Rebalance training data (oversampling, SMOTE).
   - Remove or modify proxy features.
   - Apply fairness constraints during training (adversarial debiasing).
   - Post-processing calibration per demographic group.
5. Validate: Re-evaluate with fairness metrics on held-out data.
6. Deploy: Canary deployment with bias monitoring.
7. Prevent: Add bias checks to CI/CD pipeline as a gate.
   Add ongoing bias monitoring to production model monitor.

---

Q: How do you handle ML model versioning across multiple environments?

Answer:
Use a promotion-based workflow:
  Development -> Staging -> Production

- Code: Git branches (feature -> develop -> main).
- Model artifacts: Model registry with stages (Dev -> Staging -> Production).
- Config: Environment-specific configs (dev.yaml, staging.yaml, prod.yaml).
- Infrastructure: Terraform workspaces or separate state per environment.
- Data: Same data pipeline code, different data sources per environment
  (sampled data in dev, full data in staging/prod).

Promotion flow:
1. ML engineer trains model, logs to MLflow.
2. GitHub Actions evaluates model, registers if metrics pass.
3. Model auto-promoted to Staging, deployed to staging endpoint.
4. Integration tests run against staging.
5. Manual approval gate for production.
6. Canary deployment to production with auto-rollback.

Key principle: The container image and model artifact promoted to production
should be EXACTLY what was tested in staging. Never rebuild for production.

##############################################################################
# QUICK REFERENCE: TOP 30 RAPID-FIRE QUESTIONS
##############################################################################

1. What is SageMaker? -> Fully managed ML platform for build, train, deploy.
2. What is Bedrock? -> Managed service for foundation model APIs.
3. SageMaker vs Bedrock? -> SageMaker=custom ML, Bedrock=pre-built LLM APIs.
4. What is a SageMaker Endpoint? -> Hosted HTTPS inference endpoint.
5. What is Azure OpenAI Service? -> Azure-hosted OpenAI models with enterprise features.
6. ECS vs EKS? -> ECS=simpler/AWS-native, EKS=Kubernetes/portable.
7. What is Docker multi-stage build? -> Multiple FROM statements, only final stage ships.
8. How do GPUs work in Docker? -> NVIDIA Container Toolkit, --gpus flag.
9. What is CI/CD for ML? -> Automated pipeline: code->train->evaluate->deploy.
10. What is a canary deployment? -> Gradually shift traffic to new model.
11. What is blue-green deployment? -> Two identical envs, instant traffic switch.
12. Terraform vs CloudFormation? -> Terraform=multi-cloud, CFN=AWS-native.
13. What is a model registry? -> Central catalog for versioned ML models.
14. What is experiment tracking? -> Recording all details of ML experiments.
15. What is data drift? -> Input feature distributions change from training.
16. What is concept drift? -> Relationship between features and target changes.
17. What is a feature store? -> Centralized system for managing ML features.
18. Online vs offline store? -> Online=low-latency serving, offline=training.
19. What is DVC? -> Git-like version control for data and ML models.
20. Spot vs on-demand? -> Spot=60-90% cheaper but can be interrupted.
21. What is model quantization? -> Reducing precision (FP32->INT8) for efficiency.
22. What is RAG? -> Retrieval-Augmented Generation: retrieve context + generate.
23. What is MLOps? -> DevOps practices applied to ML lifecycle.
24. SageMaker Pipelines? -> ML workflow orchestration with step-based DAGs.
25. What is KServe? -> K8s-native model serving framework (CRDs).
26. What is vLLM? -> High-throughput LLM serving with PagedAttention.
27. Lambda for ML? -> CPU-only, small models, max 15min/10GB.
28. What is Triton? -> NVIDIA's multi-framework inference server.
29. How to reduce ML Docker image? -> Multi-stage, slim base, no-cache, quantize.
30. What is SageMaker Model Monitor? -> Detects data/model drift in production.
