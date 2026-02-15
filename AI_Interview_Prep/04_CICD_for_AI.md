---
title: "CI/CD for AI"
layout: default
parent: "DevOps & Cloud Infrastructure"
nav_order: 4
render_with_liquid: false
---
{% raw %}

## SECTION 4: CI/CD FOR AI/ML - INTERVIEW QUESTIONS & ANSWERS (2025-2026)


## 4.1  GITHUB ACTIONS FOR ML PIPELINES


Q1: How do you design CI/CD pipelines for ML with GitHub Actions?

Answer:
ML CI/CD differs from traditional software CI/CD because it must handle
data, model artifacts, and model quality in addition to code.

ML CI/CD pipeline stages:
1. Code Quality: Lint, type check, unit tests (same as software).
2. Data Validation: Schema checks, distribution checks, data quality.
3. Training: Train model on new data or code changes.
4. Evaluation: Compare new model against baseline on held-out test set.
5. Registration: Register model artifact with version in model registry.
6. Deployment: Deploy to staging, run integration tests, promote to prod.
7. Monitoring: Post-deployment validation and ongoing monitoring.

Example GitHub Actions workflow:

  # .github/workflows/ml-pipeline.yml
  name: ML Pipeline

  on:
    push:
      branches: [main]
      paths:
        - 'src/**'
        - 'data/**'
        - 'configs/**'
    workflow_dispatch:
      inputs:
        force_retrain:
          description: 'Force model retraining'
          type: boolean
          default: false

  env:
    AWS_REGION: us-east-1
    ECR_REPOSITORY: ml-models
    MODEL_NAME: sentiment-classifier

  jobs:
    code-quality:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: '3.11'
        - name: Install dependencies
          run: pip install -r requirements-dev.txt
        - name: Lint
          run: ruff check src/
        - name: Type check
          run: mypy src/
        - name: Unit tests
          run: pytest tests/unit/ -v --cov=src/

    data-validation:
      runs-on: ubuntu-latest
      needs: code-quality
      steps:
        - uses: actions/checkout@v4
        - name: Configure AWS credentials
          uses: aws-actions/configure-aws-credentials@v4
          with:
            role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
            aws-region: ${{ env.AWS_REGION }}
        - name: Validate data
          run: |
            python scripts/validate_data.py \
              --data-path s3://ml-data/training/ \
              --schema configs/data_schema.json \
              --output reports/data_validation.json
        - name: Upload validation report
          uses: actions/upload-artifact@v4
          with:
            name: data-validation
            path: reports/data_validation.json

    train:
      runs-on: ubuntu-latest
      needs: data-validation
      steps:
        - uses: actions/checkout@v4
        - name: Configure AWS credentials
          uses: aws-actions/configure-aws-credentials@v4
          with:
            role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
            aws-region: ${{ env.AWS_REGION }}
        - name: Start SageMaker training job
          run: |
            python scripts/train.py \
              --config configs/training_config.yml \
              --experiment-name ${{ env.MODEL_NAME }} \
              --run-id ${{ github.sha }}
        - name: Wait for training completion
          run: python scripts/wait_for_training.py --job-name ${{ github.sha }}

    evaluate:
      runs-on: ubuntu-latest
      needs: train
      outputs:
        should_deploy: ${{ steps.compare.outputs.should_deploy }}
      steps:
        - uses: actions/checkout@v4
        - name: Evaluate model
          run: |
            python scripts/evaluate.py \
              --model-artifact s3://ml-models/${{ github.sha }}/model.tar.gz \
              --test-data s3://ml-data/test/ \
              --output reports/evaluation.json
        - name: Compare with baseline
          id: compare
          run: |
            python scripts/compare_models.py \
              --new-model reports/evaluation.json \
              --baseline-model s3://ml-models/production/metrics.json \
              --threshold 0.01
            # Sets output: should_deploy=true if new model is better

    deploy-staging:
      runs-on: ubuntu-latest
      needs: evaluate
      if: needs.evaluate.outputs.should_deploy == 'true'
      environment: staging
      steps:
        - uses: actions/checkout@v4
        - name: Deploy to staging
          run: |
            python scripts/deploy.py \
              --endpoint ${{ env.MODEL_NAME }}-staging \
              --model-artifact s3://ml-models/${{ github.sha }}/model.tar.gz \
              --instance-type ml.g5.xlarge \
              --instance-count 1

    integration-tests:
      runs-on: ubuntu-latest
      needs: deploy-staging
      steps:
        - name: Run integration tests
          run: |
            pytest tests/integration/ -v \
              --endpoint-url ${{ vars.STAGING_ENDPOINT_URL }}

    deploy-production:
      runs-on: ubuntu-latest
      needs: integration-tests
      environment: production
      steps:
        - name: Deploy canary (10% traffic)
          run: |
            python scripts/deploy.py \
              --endpoint ${{ env.MODEL_NAME }}-prod \
              --model-artifact s3://ml-models/${{ github.sha }}/model.tar.gz \
              --traffic-percentage 10 \
              --deployment-type canary
        - name: Monitor canary (15 min)
          run: python scripts/monitor_canary.py --duration 900
        - name: Promote to full traffic
          run: |
            python scripts/deploy.py \
              --endpoint ${{ env.MODEL_NAME }}-prod \
              --traffic-percentage 100

-------------------------------------------------------------------------------

Q2: How do you handle secrets and GPU runners in GitHub Actions for ML?

Answer:
Secrets management:
- GitHub Secrets: Store AWS credentials, API keys, model registry tokens.
- OIDC (OpenID Connect): Preferred for AWS. No long-lived credentials.
  Configure AWS IAM Identity Provider for GitHub Actions.
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::123456789:role/GitHubActionsML
    aws-region: us-east-1
- Environment secrets: Separate secrets for staging vs production.
- Environment protection rules: Require approval for production deployments.

GPU runners:
- GitHub-hosted runners do NOT have GPUs.
- Options for GPU workloads:
  1. Self-hosted runners on GPU machines:
     - Set up EC2 instances (g5.xlarge) as self-hosted runners.
     - Use runs-on: [self-hosted, gpu] label.
     - Ephemeral runners: Launch on demand, terminate after job.
  2. Offload to cloud:
     - Trigger SageMaker training jobs from GitHub Actions.
     - Trigger Azure ML jobs.
     - Use modal.com or cloud-based training services.
  3. GitHub-hosted larger runners (Team/Enterprise):
     - GPU runners available in beta/preview.

Self-hosted GPU runner setup:
  # On GPU machine:
  ./config.sh --url https://github.com/org/repo \
    --token XXXXX \
    --labels gpu,cuda12 \
    --ephemeral
  ./run.sh

  # In workflow:
  jobs:
    train:
      runs-on: [self-hosted, gpu, cuda12]

## 4.2  MODEL VERSIONING


Q3: How do you implement model versioning in ML projects?

Answer:
Model versioning tracks the complete lineage: code + data + config + model.

Versioning strategies:

1. Git-based (code + config):
   - Version training code, hyperparameters, configs in Git.
   - Tag releases: git tag model-v1.2.3
   - Use git hash as model identifier.

2. Model Registry (model artifacts):
   - MLflow Model Registry: Stage transitions (None -> Staging -> Production).
   - SageMaker Model Registry: Approval workflows + model packages.
   - Azure ML Model Registry: Versioned models with tags.
   - Weights & Biases: Model artifacts linked to experiments.

3. Data versioning (training data):
   - DVC (Data Version Control): Git-like versioning for large files.
   - LakeFS: Git-like branching for data lakes.
   - Delta Lake: Time-travel queries on data.

4. Container versioning (deployable artifact):
   - Tag images: mymodel:v1.2.3, mymodel:git-abc123, mymodel:latest
   - ECR/ACR immutable tags to prevent overwrites.

Comprehensive versioning schema:
  model_version:
    model_id: "sentiment-v2.1.0"
    git_commit: "abc123def456"
    training_data:
      dataset: "s3://data/training/v3/"
      dvc_hash: "md5:abc123"
      rows: 1_000_000
      date_range: "2024-01-01 to 2024-12-31"
    hyperparameters:
      learning_rate: 0.001
      epochs: 50
      batch_size: 32
    metrics:
      accuracy: 0.945
      f1_score: 0.932
      latency_p95_ms: 45
    environment:
      python: "3.11"
      pytorch: "2.2.0"
      cuda: "12.1"
    container_image: "123456789.dkr.ecr.us-east-1.amazonaws.com/models:v2.1.0"
    trained_by: "github-actions-run-456"
    approved_by: "jane.doe@company.com"
    deployment_date: "2025-03-15"

DVC example:
  # Track large model file with DVC
  dvc add models/large_model.pt
  git add models/large_model.pt.dvc models/.gitkeep
  git commit -m "Add model v2.1.0"
  dvc push  # Uploads to S3/GCS/Azure Blob

  # Reproduce training
  dvc repro  # Runs pipeline defined in dvc.yaml

## 4.3  A/B TESTING DEPLOYMENTS


Q4: How do you implement A/B testing for ML models in production?

Answer:
A/B testing compares two model versions with real user traffic.

Architecture:
  User Request -> Load Balancer / Router -> Model A (control, 50%)
                                         -> Model B (treatment, 50%)
  -> Log predictions + user outcomes -> Statistical analysis -> Decision

Implementation approaches:

1. SageMaker Production Variants:
   endpoint_config = {
       "ProductionVariants": [
           {
               "VariantName": "ModelA",
               "ModelName": "model-a",
               "InstanceType": "ml.g5.xlarge",
               "InitialInstanceCount": 1,
               "InitialVariantWeight": 0.5  # 50% traffic
           },
           {
               "VariantName": "ModelB",
               "ModelName": "model-b",
               "InstanceType": "ml.g5.xlarge",
               "InitialInstanceCount": 1,
               "InitialVariantWeight": 0.5  # 50% traffic
           }
       ]
   }

2. Kubernetes (Istio/Envoy):
   apiVersion: networking.istio.io/v1beta1
   kind: VirtualService
   spec:
     http:
     - route:
       - destination:
           host: model-service
           subset: model-a
         weight: 50
       - destination:
           host: model-service
           subset: model-b
         weight: 50

3. Feature flags (LaunchDarkly, Unleash, custom):
   if feature_flag.get_variant("model_version", user_id) == "B":
       prediction = model_b.predict(features)
   else:
       prediction = model_a.predict(features)

Statistical rigor:
- Define hypothesis: "Model B improves conversion rate by >= 2%."
- Sample size calculation: Use power analysis before starting.
- Duration: Run until statistical significance (typically 1-4 weeks).
- Metrics: Primary (business KPI) + guardrail metrics (latency, errors).
- Analysis: Use Bayesian or frequentist methods. Account for multiple testing.
- Segmentation: Check performance across user segments (mobile/desktop, etc.).

Logging requirements:
- Log: request_id, timestamp, model_version, user_id, features,
  prediction, confidence, ground_truth (when available), latency.

## 4.4  CANARY DEPLOYMENTS


Q5: What is a canary deployment for ML models and how do you implement it?

Answer:
Canary deployment gradually shifts traffic from old model to new model while
monitoring for errors/degradation. Unlike A/B testing, the goal is safe
rollout, not experimentation.

Canary stages:
  Stage 1: 5% traffic to new model, 95% to old  (30 min monitoring)
  Stage 2: 25% traffic to new model              (1 hour monitoring)
  Stage 3: 50% traffic to new model              (2 hours monitoring)
  Stage 4: 100% traffic to new model             (old model retired)
  Rollback: If any stage fails, immediately revert to 0% new model.

Monitoring during canary:
- Error rate: New model errors should be <= old model + threshold.
- Latency: p50, p95, p99 should not degrade beyond acceptable limits.
- Prediction distribution: Output distribution should be similar to old model
  (using KL divergence or KS test).
- Business metrics: Click-through rate, conversion rate (if measurable quickly).

SageMaker canary deployment:
  from sagemaker.model_monitor import DataCaptureConfig

  # Update endpoint with new model at 10% traffic
  predictor.update_endpoint(
      initial_instance_count=1,
      instance_type="ml.g5.xlarge",
      model_name="new-model-v2",
      wait=False
  )

  # SageMaker deployment guardrails
  deployment_config = {
      "BlueGreenUpdatePolicy": {
          "TrafficRoutingConfiguration": {
              "Type": "CANARY",
              "CanarySize": {
                  "Type": "INSTANCE_COUNT",
                  "Value": 1
              },
              "WaitIntervalInSeconds": 600  # 10 min between steps
          },
          "TerminationWaitInSeconds": 300,
          "MaximumExecutionTimeoutInSeconds": 3600
      },
      "AutoRollbackConfiguration": {
          "Alarms": [
              {"AlarmName": "HighErrorRate"},
              {"AlarmName": "HighLatency"}
          ]
      }
  }

Kubernetes canary with Argo Rollouts:
  apiVersion: argoproj.io/v1alpha1
  kind: Rollout
  spec:
    strategy:
      canary:
        steps:
        - setWeight: 5
        - pause: {duration: 30m}
        - analysis:
            templates:
            - templateName: model-quality-check
        - setWeight: 25
        - pause: {duration: 1h}
        - setWeight: 50
        - pause: {duration: 2h}
        - setWeight: 100

  # Analysis template
  apiVersion: argoproj.io/v1alpha1
  kind: AnalysisTemplate
  metadata:
    name: model-quality-check
  spec:
    metrics:
    - name: error-rate
      provider:
        prometheus:
          query: |
            sum(rate(model_errors_total{version="canary"}[5m])) /
            sum(rate(model_requests_total{version="canary"}[5m]))
      successCondition: result[0] < 0.05
      interval: 5m
      count: 6

## 4.5  BLUE-GREEN DEPLOYMENTS


Q6: How does blue-green deployment work for ML models?

Answer:
Blue-green maintains two identical production environments. Only one serves
live traffic at a time. Switch is instantaneous.

Architecture:
  Production DNS/LB
       |
  [Switch] --- Blue Environment (current production - Model v1)
       |
       +------ Green Environment (new version - Model v2, pre-validated)

  Steps:
  1. Blue is serving production traffic.
  2. Deploy Model v2 to Green environment.
  3. Run smoke tests and load tests against Green.
  4. Switch traffic from Blue to Green (DNS update or LB change).
  5. Monitor Green. If problems, switch back to Blue instantly.
  6. Once stable, Blue becomes the new staging environment.

SageMaker Blue-Green:
  deployment_config = {
      "BlueGreenUpdatePolicy": {
          "TrafficRoutingConfiguration": {
              "Type": "ALL_AT_ONCE",
              "WaitIntervalInSeconds": 0
          },
          "TerminationWaitInSeconds": 600,  # Keep blue alive 10 min
          "MaximumExecutionTimeoutInSeconds": 1800
      },
      "AutoRollbackConfiguration": {
          "Alarms": [
              {"AlarmName": "HighErrorRate"},
              {"AlarmName": "HighLatency"}
          ]
      }
  }

Azure ML Blue-Green with traffic split:
  # Deploy new version as "green" deployment
  az ml online-deployment create --name green \
    --endpoint-name my-endpoint \
    --model azureml:my-model:2 \
    --instance-type Standard_NC6s_v3

  # Test green directly (without affecting production)
  az ml online-endpoint invoke --name my-endpoint \
    --deployment-name green --request-file request.json

  # Switch 100% traffic to green
  az ml online-endpoint update --name my-endpoint \
    --traffic "blue=0 green=100"

  # If issues, rollback
  az ml online-endpoint update --name my-endpoint \
    --traffic "blue=100 green=0"

  # Once stable, delete blue
  az ml online-deployment delete --name blue --endpoint-name my-endpoint

Comparison of deployment strategies:
  | Strategy    | Risk    | Rollback Speed | Cost      | Use Case              |
  |-------------|---------|----------------|-----------|------------------------|
  | Blue-Green  | Low     | Instant        | 2x infra  | Critical models        |
  | Canary      | V. Low  | Fast           | ~1.05x    | Gradual validation     |
  | A/B Testing | Medium  | Fast           | ~1.5x     | Experimentation        |
  | Rolling     | Medium  | Slow           | 1x        | Non-critical updates   |
  | Shadow      | None    | N/A            | 2x compute| Pre-launch validation  |

Shadow deployment (shadow testing):
- Route 100% of production traffic to BOTH old and new model.
- Old model serves actual responses.
- New model processes same requests but responses are discarded (logged only).
- Compare predictions offline.
- Zero risk: Users never see new model's output.
{% endraw %}
