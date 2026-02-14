================================================================================
  SECTION 7: COST OPTIMIZATION FOR AI - QUESTIONS & ANSWERS (2025-2026)
================================================================================

##############################################################################
# 7.1  SPOT INSTANCES FOR TRAINING
##############################################################################

Q1: How do you use spot instances to reduce ML training costs?

Answer:
Spot instances provide 60-90% cost savings over on-demand for GPU instances.

AWS Spot pricing examples (approximate):
  | Instance       | On-Demand/hr | Spot/hr   | Savings |
  |----------------|-------------|-----------|---------|
  | p4d.24xlarge   | $32.77      | $9.83     | 70%     |
  | g5.xlarge      | $1.006      | $0.30     | 70%     |
  | p3.16xlarge    | $24.48      | $7.34     | 70%     |
  | trn1.32xlarge  | $21.50      | $6.45     | 70%     |

Spot interruption handling for training:
- Spot instances can be reclaimed with 2-minute warning.
- Mitigation strategies:

1. Checkpointing:
   - Save model checkpoints every N steps to S3/persistent storage.
   - Resume from latest checkpoint on new instance.

   # PyTorch checkpointing example
   if step % checkpoint_interval == 0:
       torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'loss': loss,
           'step': global_step,
       }, f's3://checkpoints/model_step_{global_step}.pt')

2. SageMaker Managed Spot Training:
   from sagemaker.pytorch import PyTorch
   estimator = PyTorch(
       entry_point='train.py',
       instance_type='ml.p4d.24xlarge',
       instance_count=1,
       use_spot_instances=True,
       max_wait=72*3600,      # Max time including wait for spot
       max_run=48*3600,       # Max training time
       checkpoint_s3_uri='s3://checkpoints/my-training/',
   )

3. Spot Fleet diversification:
   - Request multiple instance types to reduce interruption probability.
   - "capacity-optimized" allocation strategy chooses pools with most
     available capacity.

   spot_fleet_config = {
       "LaunchTemplateConfigs": [
           {"LaunchTemplateSpecification": {"LaunchTemplateId": "lt-xxx"},
            "Overrides": [
                {"InstanceType": "p4d.24xlarge"},
                {"InstanceType": "p3.16xlarge"},
                {"InstanceType": "g5.48xlarge"}
            ]}
       ],
       "AllocationStrategy": "capacity-optimized"
   }

4. Kubernetes spot node groups (Karpenter):
   apiVersion: karpenter.sh/v1beta1
   kind: NodePool
   spec:
     template:
       spec:
         requirements:
           - key: karpenter.sh/capacity-type
             operator: In
             values: ["spot"]
           - key: node.kubernetes.io/instance-type
             operator: In
             values: ["g5.xlarge", "g5.2xlarge", "g5.4xlarge"]
         nodeClassRef:
           name: gpu-node-class
     disruption:
       consolidationPolicy: WhenUnderutilized

Azure Spot VMs:
  - Similar concept, up to 90% savings.
  - Eviction policy: Stop/Deallocate or Delete.
  - Use with Azure ML compute clusters:
    az ml compute create --name gpu-spot-cluster \
      --type AmlCompute \
      --size Standard_NC24ads_A100_v4 \
      --min-instances 0 \
      --max-instances 4 \
      --tier low_priority  # Spot equivalent in Azure ML

##############################################################################
# 7.2  RESERVED INSTANCES FOR INFERENCE
##############################################################################

Q2: How do you use reserved instances / savings plans for ML inference?

Answer:
Always-on inference endpoints benefit from commitment-based pricing.

AWS options:

1. SageMaker Savings Plans:
   - 1-year or 3-year commitment.
   - 25-64% savings over on-demand.
   - Covers SageMaker ML instances (training, inference, notebooks).
   - Flexible: Applies to any instance family, size, or region.
   - Example: Commit to $10/hr of SageMaker usage -> get discounted rate.

2. EC2 Reserved Instances (for self-managed inference):
   - Standard RI: Up to 72% savings, locked to instance type + region.
   - Convertible RI: Up to 66% savings, can change instance type.
   - 1-year or 3-year term.
   - Payment options: All upfront (max savings), partial upfront, no upfront.

3. EC2 Savings Plans:
   - Compute Savings Plans: Flexible across instance types, 66% savings.
   - EC2 Instance Savings Plans: Locked to instance family, 72% savings.

Azure options:
  - Azure Reservations: 1-year or 3-year, up to 72% savings.
  - Azure Savings Plans for Compute: Flexible across VM sizes.

Decision framework:
  | Workload Pattern          | Recommendation                    | Savings |
  |---------------------------|-----------------------------------|---------|
  | 24/7 production endpoint  | Reserved Instances / Savings Plan | 40-72%  |
  | Business hours only       | Savings Plan + Auto-scaling to 0  | 25-50%  |
  | Burst traffic             | Savings Plan base + On-demand     | 20-40%  |
  | Experimental/dev          | On-demand or Spot                 | 0/60-90%|

Sizing strategy:
  1. Reserve capacity for baseline (minimum consistent load).
  2. Use on-demand for expected peaks.
  3. Use spot for burst capacity if latency-tolerant.
  4. Review utilization quarterly and adjust reservations.

##############################################################################
# 7.3  AUTO-SCALING GPU CLUSTERS
##############################################################################

Q3: How do you auto-scale GPU clusters for AI workloads?

Answer:
Auto-scaling GPU clusters is critical for cost optimization. GPUs are
expensive ($1-35/hr per instance), so over-provisioning is costly.

SageMaker Endpoint Auto-scaling:
  - Target tracking: Scale based on invocations per instance.
  - Step scaling: Define thresholds for scale-out/in.
  - Scheduled scaling: Pre-scale for known traffic patterns.

  # Scale to 0 during off-hours (scheduled scaling)
  aws application-autoscaling put-scheduled-action \
    --service-namespace sagemaker \
    --resource-id endpoint/my-endpoint/variant/primary \
    --scheduled-action-name scale-down-night \
    --schedule "cron(0 22 * * ? *)" \
    --scalable-target-action MinCapacity=0,MaxCapacity=0

  # Scale up for business hours
  aws application-autoscaling put-scheduled-action \
    --service-namespace sagemaker \
    --resource-id endpoint/my-endpoint/variant/primary \
    --scheduled-action-name scale-up-morning \
    --schedule "cron(0 6 * * ? *)" \
    --scalable-target-action MinCapacity=2,MaxCapacity=10

Kubernetes GPU auto-scaling:

1. Horizontal Pod Autoscaler (HPA):
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: model-server-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: model-server
     minReplicas: 1
     maxReplicas: 10
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
           name: inference_queue_depth
         target:
           type: AverageValue
           averageValue: "5"
     behavior:
       scaleDown:
         stabilizationWindowSeconds: 300
         policies:
         - type: Pods
           value: 1
           periodSeconds: 60
       scaleUp:
         stabilizationWindowSeconds: 30
         policies:
         - type: Pods
           value: 2
           periodSeconds: 60

2. KEDA (Kubernetes Event Driven Autoscaling):
   - Scale based on queue depth (SQS, Kafka, etc.).
   - Scale to zero when no requests.

   apiVersion: keda.sh/v1alpha1
   kind: ScaledObject
   metadata:
     name: model-server-scaler
   spec:
     scaleTargetRef:
       name: model-server
     minReplicaCount: 0    # Scale to zero!
     maxReplicaCount: 10
     triggers:
     - type: aws-sqs-queue
       metadata:
         queueURL: https://sqs.us-east-1.amazonaws.com/123/inference-queue
         queueLength: "5"
         awsRegion: us-east-1

3. Karpenter (node-level scaling):
   - Automatically provisions right-sized GPU nodes.
   - Consolidation: Moves pods to fewer nodes when load decreases.
   - Interruption handling: Gracefully drains spot nodes.

Scaling challenges for GPUs:
- Cold start: GPU model loading takes 30-120 seconds.
  Mitigation: Keep warm pool, use readiness probes.
- Scaling granularity: GPUs can't be partially allocated (easily).
  Mitigation: GPU time-slicing, MIG (Multi-Instance GPU).
- Cost of idle GPUs: $1-35/hr wasted if not utilized.
  Mitigation: Scale to zero with KEDA, aggressive scale-down policies.

##############################################################################
# 7.4  SERVERLESS VS ALWAYS-ON TRADE-OFFS
##############################################################################

Q4: When should you use serverless vs always-on for AI workloads?

Answer:
Decision framework:

USE SERVERLESS (Lambda/Azure Functions/SageMaker Serverless) when:
- Traffic is sporadic or unpredictable (< 100 req/min average).
- Acceptable cold start latency (1-15 seconds).
- Model is small enough for CPU inference (< 10GB).
- Cost-sensitive with low traffic volumes.
- Event-driven workloads (S3 triggers, queue processing).
- No GPU required.

USE ALWAYS-ON (EC2/ECS/EKS/SageMaker real-time endpoints) when:
- Consistent high traffic (> 100 req/min sustained).
- Strict latency SLA (< 100ms p99).
- GPU required for inference.
- Large models that take minutes to load.
- Stateful inference (conversation context, session management).
- Streaming inference (LLM token streaming).

SageMaker Serverless Inference:
  - Provisions compute on demand, scales to zero.
  - Max memory: 6GB. Max concurrency: 200.
  - Cold start: 1-2 minutes for first request.
  - No GPU support.
  - Best for: Infrequent inference on small models.
  - Pricing: Per ms of compute + per GB-second of memory.

Cost comparison example:
  Scenario: Image classification model, 1000 requests/day, 500ms avg latency.

  SageMaker Real-time (ml.g5.xlarge):
    $1.006/hr * 24hr * 30 days = $724/month (always on)

  SageMaker Serverless (4GB memory):
    1000 req * 0.5 sec * 30 days * $0.00003/sec = $0.45/month
    (But add cold starts: ~$5-10/month for occasional cold starts)
    Total: ~$10/month

  Lambda (3GB memory, CPU inference, ONNX Runtime):
    1000 req * 0.5 sec * 30 days * $0.0000166667/GB-sec * 3GB = $0.75/month
    Total: ~$1/month

  Savings: 99% with serverless at this traffic level.

  But at 100,000 requests/day:
  Serverless: ~$100/month (plus cold starts become frequent)
  Always-on: $724/month (but better latency and throughput)
  -> Serverless still cheaper but latency degrades.

  At 1,000,000 requests/day:
  Serverless: ~$1,000+/month (concurrent limit may throttle)
  Always-on with auto-scaling: $724-2,000/month (better value)
  -> Always-on becomes more cost-effective.

Hybrid approach:
  - Base load: Always-on instances (reserved pricing).
  - Burst overflow: Serverless or spot instances.
  - Non-critical: Async queue + serverless processing.

##############################################################################
# 7.5  ADDITIONAL COST OPTIMIZATION STRATEGIES
##############################################################################

Q5: What are other key cost optimization strategies for AI workloads?

Answer:

1. Model optimization (reduce compute requirements):
   - Quantization: FP32 -> INT8 = 75% memory reduction, ~2x inference speed.
   - Distillation: Train smaller student model from large teacher.
   - Pruning: Remove redundant weights (30-50% reduction possible).
   - Architecture search: Use smaller architectures for simpler tasks.
   - Example: GPT-4 for prototyping, distilled small model for production.

2. Inference optimization:
   - Batching: Process multiple requests together (higher throughput).
   - Caching: Cache frequent predictions (Redis/ElastiCache).
     20-50% of requests may be cacheable.
   - Model compilation: TensorRT, ONNX Runtime, Neuron SDK.
     Typical 2-5x speedup = 50-80% cost reduction.

3. Data storage optimization:
   - S3 Intelligent Tiering: Automatic cost optimization.
   - Lifecycle policies: Archive old training data.
   - Compression: Use Parquet (vs CSV) for 50-75% storage savings.
   - Deduplication: Remove duplicate training samples.

4. Right-sizing:
   - Don't use GPU when CPU suffices (many classical ML models).
   - Don't use A100 when T4 is sufficient.
   - Use Inferentia2 for transformer inference (up to 4x better
     price-performance than GPU).
   - AWS Compute Optimizer recommendations.

5. Multi-tenancy:
   - SageMaker Multi-Model Endpoints: Host 100s of models on one endpoint.
   - Triton Inference Server: Serve multiple models per GPU.
   - GPU sharing: Time-slicing or MIG for underutilized GPUs.

6. Training cost reduction:
   - Spot instances: 60-90% savings (see Q1).
   - Mixed precision training: FP16/BF16 = 2x speed, same quality.
   - Gradient accumulation: Use smaller instances with accumulated gradients.
   - Early stopping: Stop training when validation loss plateaus.
   - Learning rate scheduling: Reach convergence faster.
   - Transfer learning: Fine-tune pre-trained models (less training needed).

7. Monitoring and governance:
   - AWS Cost Explorer with ML-specific tags.
   - Budget alerts: Warn at 50%, 80%, 100% of budget.
   - Service Control Policies: Prevent launching expensive instances.
   - Kubecost (Kubernetes): Per-pod, per-namespace cost tracking.
   - Spot instance advisor: Monitor spot pricing trends.

Cost optimization checklist:
  [ ] Tag all ML resources (project, team, environment, model).
  [ ] Use spot for training, reserved for inference.
  [ ] Right-size instances (benchmark before committing).
  [ ] Scale to zero when not in use (dev/staging endpoints).
  [ ] Optimize models (quantization, pruning, distillation).
  [ ] Cache frequent predictions.
  [ ] Use cheaper storage tiers for old data.
  [ ] Review costs monthly with ML team leads.
  [ ] Set budget alerts and spending limits.
  [ ] Consider serverless for low-traffic models.
