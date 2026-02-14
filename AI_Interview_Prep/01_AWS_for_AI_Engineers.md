================================================================================
  SECTION 1: AWS FOR AI ENGINEERS - INTERVIEW QUESTIONS & ANSWERS (2025-2026)
================================================================================

##############################################################################
# 1.1  AMAZON SAGEMAKER
##############################################################################

Q1: What is Amazon SageMaker and what are its core components?

Answer:
Amazon SageMaker is a fully managed ML platform that covers the entire ML
lifecycle. Core components include:

- SageMaker Studio: IDE for ML development (JupyterLab-based).
- SageMaker Notebooks: Managed Jupyter notebook instances.
- SageMaker Training: Managed training jobs with built-in algorithms or custom
  containers. Supports distributed training across multiple GPUs/instances.
- SageMaker Processing: For data preprocessing and post-processing jobs.
- SageMaker Endpoints: Real-time inference hosting with auto-scaling.
- SageMaker Batch Transform: Offline batch inference on large datasets.
- SageMaker Pipelines: MLOps workflow orchestration (CI/CD for ML).
- SageMaker Feature Store: Centralized feature management (online + offline).
- SageMaker Model Registry: Versioned model catalog with approval workflows.
- SageMaker Clarify: Bias detection and model explainability.
- SageMaker Ground Truth: Data labeling service.
- SageMaker Canvas: No-code ML for business analysts.
- SageMaker JumpStart: Pre-trained model hub (foundation models).

-------------------------------------------------------------------------------

Q2: Explain SageMaker Endpoints and auto-scaling for real-time inference.

Answer:
SageMaker real-time endpoints host models behind an HTTPS endpoint. Key details:

- Endpoint Configuration: Specifies instance type (e.g., ml.g5.xlarge for GPU),
  instance count, model artifact location (S3), and container image.
- Multi-Model Endpoints (MME): Host multiple models on a single endpoint to
  reduce costs. Models are loaded/unloaded dynamically from S3.
- Multi-Container Endpoints: Run multiple containers (different frameworks)
  behind one endpoint using direct invocation or serial inference pipeline.
- Auto-scaling: Uses Application Auto Scaling with target tracking policies.
  Common metric: SageMakerVariantInvocationsPerInstance.

  Example policy:
    Target value: 70 invocations per instance per minute
    Scale-in cooldown: 300 seconds
    Scale-out cooldown: 60 seconds
    Min instances: 1, Max instances: 10

- Shadow Testing: Route production traffic to a shadow variant to test new
  models without impacting users.
- Inference Recommender: Benchmarks different instance types to find
  cost-optimal configuration.

-------------------------------------------------------------------------------

Q3: What is SageMaker Pipelines and how does it enable MLOps?

Answer:
SageMaker Pipelines is a purpose-built CI/CD service for ML. Key concepts:

- Pipeline Steps:
  * ProcessingStep: Data preprocessing/validation
  * TrainingStep: Model training
  * TuningStep: Hyperparameter optimization
  * TransformStep: Batch inference
  * RegisterModel: Register in Model Registry
  * ConditionStep: Branching logic (e.g., only register if accuracy > threshold)
  * CallbackStep: Integration with external systems
  * QualityCheckStep: Data/model quality baselines
  * LambdaStep: Run AWS Lambda functions

- Pipeline parameters are typed (String, Integer, Float, Boolean) and can be
  overridden at execution time.
- Integrates with SageMaker Model Registry for model approval workflows
  (Pending -> Approved -> Rejected).
- Supports caching of step outputs to avoid redundant computation.
- Pipeline execution history provides full lineage tracking.

Example skeleton:
  preprocessing -> training -> evaluation -> condition(accuracy>0.9)
    -> if true: register_model -> deploy
    -> if false: notify_failure

-------------------------------------------------------------------------------

Q4: How does SageMaker distributed training work?

Answer:
SageMaker supports two distributed training strategies:

1. Data Parallelism (SageMaker Distributed Data Parallel - SDDP):
   - Splits training data across GPUs; each GPU has a full model copy.
   - AllReduce for gradient synchronization.
   - Optimized for bandwidth with techniques like gradient compression.
   - Up to 40% faster than standard Horovod on AWS infrastructure.

2. Model Parallelism (SageMaker Distributed Model Parallel - SDMP):
   - Splits the model across GPUs when it doesn't fit in single GPU memory.
   - Supports pipeline parallelism and tensor parallelism.
   - Auto-partitioning: SageMaker can automatically decide how to split layers.
   - Used for training large language models (billions of parameters).

Configuration example:
  distribution = {
      "smdistributed": {
          "dataparallel": {"enabled": True}
      }
  }
  # or for model parallelism:
  distribution = {
      "smdistributed": {
          "modelparallel": {
              "enabled": True,
              "parameters": {
                  "partitions": 4,
                  "pipeline_parallel_degree": 2,
                  "tensor_parallel_degree": 2
              }
          }
      }
  }

Instance types for distributed training: ml.p4d.24xlarge (8x A100 GPUs),
ml.p5.48xlarge (8x H100 GPUs).

-------------------------------------------------------------------------------

Q5: Explain SageMaker Feature Store.

Answer:
SageMaker Feature Store provides a centralized repository for ML features with
two storage modes:

- Online Store: Low-latency reads (<10ms) for real-time inference. Backed by
  an in-memory store. Stores the latest feature values.
- Offline Store: Historical feature data stored in S3 (Parquet format) for
  training. Supports time-travel queries.

Key concepts:
- Feature Group: A collection of features (like a database table).
- Record Identifier: Unique key for each record.
- Event Time: Timestamp for point-in-time correctness.
- Ingestion: Supports PutRecord (online) and batch ingestion.

Use cases:
- Prevents training-serving skew by using same feature definitions.
- Feature reuse across teams (feature discovery).
- Point-in-time joins for training data generation.

-------------------------------------------------------------------------------

Q6: What is SageMaker Model Monitor?

Answer:
SageMaker Model Monitor detects drift and quality issues in deployed models:

- Data Quality Monitor: Detects data drift by comparing inference input
  distributions against a baseline (training data statistics).
- Model Quality Monitor: Tracks model performance metrics (accuracy, F1, etc.)
  by comparing predictions against ground truth labels.
- Bias Drift Monitor: Detects changes in bias metrics over time.
- Feature Attribution Drift: Detects changes in feature importance using SHAP.

Setup:
1. Enable data capture on the endpoint (captures request/response payloads).
2. Create a baseline (statistical properties of training data).
3. Schedule monitoring jobs (hourly, daily, custom).
4. Configure CloudWatch alarms for violations.

Output: Violation reports in S3, CloudWatch metrics, SNS notifications.

##############################################################################
# 1.2  AMAZON BEDROCK
##############################################################################

Q7: What is Amazon Bedrock and how is it used?

Answer:
Amazon Bedrock is a fully managed service for accessing foundation models (FMs)
from Amazon (Titan) and third-party providers (Anthropic Claude, Meta Llama,
Mistral, Cohere, Stability AI, AI21 Labs) via API.

Key capabilities:
- Model Access: Single API for multiple model providers. No infrastructure
  management. Pay-per-token pricing.
- Knowledge Bases: RAG (Retrieval-Augmented Generation) with automatic
  document chunking, embedding, and vector storage (OpenSearch Serverless,
  Pinecone, Redis Enterprise Cloud, or Amazon Aurora PostgreSQL with pgvector).
- Agents: Build autonomous AI agents that can break down tasks, call APIs,
  and execute multi-step workflows using function calling.
- Guardrails: Content filtering, PII redaction, topic denial, word filters,
  and contextual grounding checks.
- Fine-tuning: Customize models with your data (continued pre-training or
  instruction fine-tuning). Supported for select models (Titan, Llama, etc.).
- Model Evaluation: Compare model outputs on your data with built-in metrics
  or human evaluation.
- Provisioned Throughput: Reserved model capacity for consistent performance.

-------------------------------------------------------------------------------

Q8: How do you implement RAG with Bedrock Knowledge Bases?

Answer:
Steps to implement RAG:

1. Create a data source: Point to S3 bucket containing documents (PDF, TXT,
   HTML, MD, CSV, DOC, XLS).

2. Choose an embedding model: Amazon Titan Embeddings V2 or Cohere Embed.

3. Select a vector store:
   - Amazon OpenSearch Serverless (default, fully managed)
   - Amazon Aurora PostgreSQL (pgvector)
   - Pinecone
   - Redis Enterprise Cloud
   - MongoDB Atlas

4. Sync data source: Bedrock automatically chunks documents, generates
   embeddings, and stores them in the vector database.

5. Query: Use RetrieveAndGenerate API which:
   a. Converts user query to embedding
   b. Performs similarity search in vector store
   c. Retrieves top-k relevant chunks
   d. Passes chunks + query to the FM for answer generation
   e. Returns response with source citations

Configuration options:
- Chunking strategy: Fixed size, no chunking, or semantic chunking
- Chunk size: 100-500 tokens typical
- Overlap: 10-20% for context continuity
- Metadata filtering: Filter retrieval by document metadata
- Hybrid search: Combine semantic + keyword search

-------------------------------------------------------------------------------

Q9: What are Bedrock Agents and how do they work?

Answer:
Bedrock Agents enable building autonomous AI applications:

Architecture:
- Agent receives user input
- Agent uses FM to understand intent and create an execution plan
- Agent calls action groups (Lambda functions) or Knowledge Bases
- Agent orchestrates multi-step reasoning (ReAct-style)
- Agent returns synthesized response

Components:
- Instructions: System prompt defining agent behavior
- Action Groups: Lambda functions the agent can invoke (defined via OpenAPI schema)
- Knowledge Bases: RAG sources for information retrieval
- Guardrails: Safety controls applied to agent I/O

Advanced features:
- Return of Control: Agent asks user for confirmation before executing actions
- Session management: Multi-turn conversations with memory
- Prompt template customization: Override default orchestration prompts
- Tracing: Full visibility into agent reasoning steps

##############################################################################
# 1.3  EC2 FOR GPU INFERENCE
##############################################################################

Q10: Which EC2 instance types are used for AI/ML workloads?

Answer:
GPU instances:
- P5 (ml.p5.48xlarge): 8x NVIDIA H100 (80GB each), 640GB GPU memory.
  Best for: Large model training (LLMs, diffusion models).
- P4d (ml.p4d.24xlarge): 8x NVIDIA A100 (40GB each), 320GB GPU memory.
  Best for: Distributed training, large-scale inference.
- P4de: 8x A100 (80GB each). Extended memory variant.
- G5: NVIDIA A10G GPUs. Best for: Inference, graphics-intensive AI.
  g5.xlarge (1 GPU, 24GB), g5.48xlarge (8 GPUs, 192GB).
- G6: NVIDIA L4 GPUs. Newer generation for cost-effective inference.
- Inf2: AWS Inferentia2 chips. Best for: Cost-optimized inference for
  transformer models. Up to 4x better price-performance than GPU instances.
- Trn1: AWS Trainium chips. Best for: Cost-optimized training.
  trn1.32xlarge: 16 Trainium chips.

Accelerated computing instances:
- DL1: Habana Gaudi accelerators (for training).

Key considerations:
- EFA (Elastic Fabric Adapter): High-bandwidth, low-latency networking for
  distributed training. Available on P4d, P5, Trn1.
- Instance Store: NVMe SSD for high-speed local storage (training data cache).
- Placement Groups: Cluster placement for minimizing network latency.

-------------------------------------------------------------------------------

Q11: How do you optimize EC2 GPU instances for ML inference?

Answer:
Optimization strategies:

1. Right-sizing: Use SageMaker Inference Recommender or manual benchmarking
   to find the optimal instance type for your model.

2. Model optimization:
   - Quantization: FP32 -> FP16 -> INT8 -> INT4 (reduces memory, increases
     throughput). Tools: TensorRT, ONNX Runtime, bitsandbytes.
   - Pruning: Remove unnecessary weights.
   - Distillation: Train smaller model from larger one.
   - Compilation: Use AWS Neuron SDK (for Inferentia/Trainium), TensorRT,
     or ONNX Runtime for hardware-specific optimization.

3. Batching:
   - Dynamic batching: Accumulate requests and process in batches.
   - Continuous batching (for LLMs): Process tokens as they arrive without
     waiting for full sequences to complete. Supported by vLLM, TGI.

4. Multi-model serving:
   - Use Triton Inference Server or TorchServe to serve multiple models
     on the same GPU.

5. GPU memory optimization:
   - KV-cache optimization for LLMs (PagedAttention via vLLM).
   - Model sharding across multiple GPUs using tensor parallelism.

##############################################################################
# 1.4  S3 FOR MODEL STORAGE
##############################################################################

Q12: How do you use S3 for ML model storage and data management?

Answer:
S3 usage patterns in ML:

Model Artifacts:
- Store trained model files (model.tar.gz for SageMaker).
- Versioning: Enable S3 versioning for model artifact tracking.
- Lifecycle policies: Move old model versions to S3 Glacier after N days.
- Cross-region replication for disaster recovery.

Training Data:
- S3 as primary data lake for training datasets.
- SageMaker Pipe Mode: Streams data directly from S3 (no disk download).
- S3 Select: Query subsets of data using SQL (CSV, JSON, Parquet).
- Partitioning: Organize by date/feature for efficient access.

Best practices:
- Use S3 Intelligent-Tiering for datasets with unpredictable access patterns.
- Enable S3 Transfer Acceleration for large model uploads from distant regions.
- Use multipart upload for files >100MB.
- Use S3 Access Points for fine-grained access control per team.
- VPC endpoints (Gateway endpoint for S3) to keep traffic within AWS network.
- Server-side encryption (SSE-S3 or SSE-KMS) for all model artifacts.
- Object Lock for compliance (immutable model artifacts).

Storage classes for ML:
- S3 Standard: Active training data, current model artifacts.
- S3 Standard-IA: Previous model versions accessed occasionally.
- S3 Glacier Instant Retrieval: Archived models needing immediate access.
- S3 Glacier Deep Archive: Long-term compliance storage.

##############################################################################
# 1.5  AWS LAMBDA FOR SERVERLESS AI
##############################################################################

Q13: How can AWS Lambda be used for AI/ML workloads?

Answer:
Lambda use cases in AI:

1. Lightweight inference:
   - Deploy small models (<10GB with container images, <250MB with zip).
   - Lambda container images: Package ML framework + model in Docker image.
   - Example: Sentiment analysis, text classification, small NLP models.
   - Cold start mitigation: Use Provisioned Concurrency.

2. Pre/post-processing:
   - Image resizing before sending to SageMaker endpoint.
   - Response formatting and business logic after inference.
   - Feature engineering for real-time predictions.

3. Event-driven ML pipelines:
   - S3 trigger: New data arrives -> Lambda -> Start SageMaker training job.
   - API Gateway + Lambda: REST API for model inference.
   - EventBridge: Schedule retraining pipelines.
   - SNS/SQS: Process inference requests asynchronously.

4. Orchestration:
   - Step Functions + Lambda for complex ML workflows.

Limitations for ML:
- Memory: Up to 10GB RAM.
- Timeout: Maximum 15 minutes.
- No GPU support (use for CPU-based inference only).
- Package size: 250MB (zip) or 10GB (container image).
- Ephemeral storage: Up to 10GB in /tmp.
- Cold starts: Can be 5-15 seconds for ML container images.

Optimization:
- Use Lambda SnapStart (Java) or Provisioned Concurrency to reduce cold starts.
- Use ONNX Runtime for optimized CPU inference.
- Use Lambda Layers for shared ML dependencies.
- Use ARM64 (Graviton2) for better price-performance on CPU inference.

-------------------------------------------------------------------------------

Q14: How do you deploy a lightweight ML model on Lambda?

Answer:
Two approaches:

Approach 1: Lambda Container Image
  # Dockerfile
  FROM public.ecr.aws/lambda/python:3.11
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY model/ ./model/
  COPY app.py .
  CMD ["app.handler"]

  # app.py
  import json, onnxruntime as ort
  session = ort.InferenceSession("model/model.onnx")  # loaded at init

  def handler(event, context):
      input_data = json.loads(event["body"])
      result = session.run(None, {"input": input_data["features"]})
      return {"statusCode": 200, "body": json.dumps({"prediction": result})}

Approach 2: Lambda Layer + ZIP
  - Package model + dependencies in a Lambda Layer.
  - Keep handler code in the deployment package.
  - Total unzipped size must be <250MB.

Key best practices:
- Load model outside the handler function (reused across invocations).
- Use ONNX format for fast CPU inference.
- Quantize model to INT8 to reduce size and improve speed.
- Set memory to 3-6GB for ML workloads (CPU scales with memory).

##############################################################################
# 1.6  ECS/EKS FOR CONTAINER ORCHESTRATION
##############################################################################

Q15: Compare ECS vs EKS for AI/ML workloads.

Answer:
Amazon ECS (Elastic Container Service):
- AWS-native container orchestration.
- Simpler to set up and manage.
- Tight integration with AWS services (IAM roles for tasks, CloudWatch).
- Launch types: EC2 (manage instances) or Fargate (serverless).
- GPU support: EC2 launch type with GPU-optimized AMI (p3, p4, g4, g5).
- Fargate does NOT support GPUs (as of early 2025).
- Best for: Simpler AI serving pipelines, teams already invested in AWS.

Amazon EKS (Elastic Kubernetes Service):
- Managed Kubernetes.
- More complex but more flexible and portable.
- GPU support: GPU node groups with NVIDIA device plugin.
- Supports Karpenter for intelligent auto-scaling (can mix GPU instance types).
- Ecosystem: Helm charts, Kubeflow, KServe, Seldon Core, Ray, etc.
- Best for: Complex ML platforms, multi-cloud strategy, teams with K8s expertise.

GPU on EKS setup:
1. Create a managed node group with GPU instances (g5.xlarge, p4d.24xlarge).
2. Install NVIDIA device plugin DaemonSet:
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/
     k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
3. In pod spec, request GPU resources:
   resources:
     limits:
       nvidia.com/gpu: 1

ECS GPU task definition:
  "resourceRequirements": [
      {"type": "GPU", "value": "1"}
  ]

Scaling strategies:
- ECS: Application Auto Scaling on CPU/memory/custom metrics.
- EKS: Horizontal Pod Autoscaler (HPA) + Karpenter/Cluster Autoscaler.
- KEDA (Kubernetes Event Driven Autoscaling): Scale based on queue depth,
  HTTP requests, custom metrics.

##############################################################################
# 1.7  API GATEWAY FOR AI SERVICES
##############################################################################

Q16: How do you use API Gateway to expose AI/ML models?

Answer:
API Gateway patterns for ML:

1. API Gateway + Lambda:
   - Synchronous inference for lightweight models.
   - Timeout limit: 29 seconds (API Gateway limit).
   - Good for: Small models, low latency requirements.

2. API Gateway + SageMaker Endpoint (direct integration):
   - No Lambda needed; API Gateway calls SageMaker directly.
   - Uses AWS service integration with IAM role.
   - Mapping templates transform request/response formats.
   - Reduces latency by eliminating Lambda cold starts.

3. API Gateway + ECS/EKS (via ALB/NLB):
   - API Gateway routes to Application Load Balancer.
   - ALB distributes to ECS tasks or EKS pods running model servers.

4. API Gateway + Step Functions (async):
   - For long-running inference (>29 seconds).
   - Client polls for results or receives callback.

Key configurations:
- Throttling: Rate limit API calls (e.g., 1000 req/sec per API key).
- Usage Plans + API Keys: Control access for different consumers.
- Request validation: Validate input schema before forwarding to model.
- Caching: Cache inference results for repeated inputs (up to 300 seconds TTL).
- WAF integration: Protect against abuse.
- Custom authorizers: JWT/OAuth2 validation via Lambda authorizer.
- WebSocket API: For streaming inference results (e.g., LLM token streaming).

##############################################################################
# 1.8  CLOUDWATCH FOR AI MONITORING
##############################################################################

Q17: How do you monitor AI/ML workloads with CloudWatch?

Answer:
CloudWatch monitoring for ML:

SageMaker Metrics:
- Training: train:loss, validation:accuracy (emitted to CloudWatch).
- Endpoint: Invocations, InvocationModelErrors, ModelLatency,
  OverheadLatency, Invocation4XXErrors, Invocation5XXErrors,
  GPUMemoryUtilization, GPUUtilization, CPUUtilization, MemoryUtilization.

Custom Metrics for ML:
- Prediction distribution (detect output drift).
- Feature value distributions.
- Inference latency percentiles (p50, p95, p99).
- Model confidence scores.
- Token usage for LLM endpoints.

CloudWatch Alarms:
- Latency alarm: ModelLatency > 500ms for 5 consecutive minutes.
- Error rate: Invocation5XXErrors > 1% of total invocations.
- GPU utilization: < 20% (over-provisioned) or > 90% (under-provisioned).

CloudWatch Logs Insights:
  # Query SageMaker endpoint logs for errors
  fields @timestamp, @message
  | filter @message like /ERROR/
  | sort @timestamp desc
  | limit 50

CloudWatch Dashboards:
- Create unified dashboards showing: endpoint latency, error rates,
  GPU utilization, invocation counts, auto-scaling activity.

CloudWatch Anomaly Detection:
- ML-based anomaly detection on any metric (e.g., detect unusual
  inference latency patterns automatically).

Integration with other services:
- SNS: Send alerts to Slack/PagerDuty.
- EventBridge: Trigger automated remediation (e.g., scale up, retrain).
- X-Ray: Distributed tracing for complex inference pipelines.

##############################################################################
# 1.9  IAM FOR AI SERVICES
##############################################################################

Q18: How do you configure IAM for AI/ML workloads on AWS?

Answer:
IAM best practices for ML:

SageMaker Execution Role:
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject", "s3:PutObject", "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::my-ml-bucket/*",
          "arn:aws:s3:::my-ml-bucket"
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:GetAuthorizationToken"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": ["logs:CreateLogGroup", "logs:CreateLogStream",
                    "logs:PutLogEvents"],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": ["cloudwatch:PutMetricData"],
        "Resource": "*"
      }
    ]
  }

Bedrock permissions:
- bedrock:InvokeModel - Call foundation models.
- bedrock:InvokeModelWithResponseStream - Streaming inference.
- bedrock:CreateKnowledgeBase - Create RAG knowledge bases.
- bedrock:Retrieve - Query knowledge bases.

Principle of least privilege for ML:
- Data scientists: SageMaker notebook + training permissions, read-only S3.
- ML engineers: Full SageMaker + deploy permissions, ECR push, S3 read/write.
- Inference services: Only invoke endpoint, no training permissions.
- Use SageMaker Studio roles with domain-level and user-level separation.

Cross-service access:
- Lambda execution role needs sagemaker:InvokeEndpoint.
- ECS task role needs s3:GetObject for model artifacts.
- API Gateway needs sagemaker:InvokeEndpoint (service role).

Security best practices:
- VPC endpoints for SageMaker, S3, ECR (no internet exposure).
- KMS encryption for model artifacts, training data, and endpoint storage.
- SageMaker network isolation: prevent training containers from internet access.
- Service Control Policies (SCPs): Restrict GPU instance types to prevent cost
  overruns (e.g., deny p5 instances in non-production accounts).

##############################################################################
# 1.8  LATEST AWS AI SERVICES UPDATE (Late 2025 - Early 2026)
##############################################################################

Q20: What is Amazon Bedrock AgentCore and what are its new capabilities?

Answer:
Amazon Bedrock AgentCore (announced at re:Invent 2025) is a comprehensive
platform for building, deploying, and governing AI agents at scale.

Key Components:
- AgentCore Gateway: Intercepts every tool call in real time, ensuring agents
  stay within defined boundaries.
- AgentCore Policy (preview): Create policies using natural language that
  automatically convert to Cedar (AWS open-source policy language). Enforces
  guardrails on what agents can and cannot do.
- AgentCore Evaluations (preview): 13 built-in evaluators for common quality
  dimensions including helpfulness, tool selection accuracy, and correctness.
- AgentCore Memory: Now includes episodic memory, enabling agents to learn
  and adapt from past interactions.
- AgentCore Runtime: Supports bidirectional streaming for natural conversations
  where agents simultaneously listen and respond.

Q21: What is Amazon Nova and the latest model expansion?

Answer:
Amazon Nova 2 (announced re:Invent 2025) is Amazon's next-generation foundation
model family, trained by Amazon:
- Nova 2 Pro: Competitive with GPT-4o class models
- Nova 2 Lite: Fast and cost-effective for production workloads
- Nova models are available exclusively through Bedrock

Bedrock also added 18 fully managed open weight models (Dec 2025), the largest
expansion of models to date, including Llama 4, Mistral, and community models.

Q22: What are the latest Bedrock features for AI agents?

Answer:
Key 2025-2026 updates:
- Responses API: Server-side tool use -- agents can perform web search, code
  execution, and database updates within AWS security boundaries.
- Prompt Caching: 1-hour TTL option for caching to reduce costs and improve
  performance for long-running, multi-turn agent workflows.
- Reinforcement Fine-tuning: Delivers 66% accuracy gains on average over base
  models. Available for select models through Bedrock.
- Agent Workflows: Enhanced orchestration for complex multi-step agent tasks.
- Multi-Agent Collaboration: Support for supervisor-worker patterns natively.

> YOUR EXPERIENCE: At RavianAI, building agentic AI platforms on AWS means
> leveraging Bedrock AgentCore for production-grade agent deployment with
> built-in policy controls and evaluation. At MathCo, you used AWS services
> (EMR, Lambda, S3) for data engineering -- Bedrock Knowledge Bases would
> naturally extend this to RAG-powered analytics.
