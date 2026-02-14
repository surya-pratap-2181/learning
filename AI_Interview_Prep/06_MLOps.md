================================================================================
  SECTION 6: MLOps - INTERVIEW QUESTIONS & ANSWERS (2025-2026)
================================================================================

##############################################################################
# 6.1  MODEL REGISTRY
##############################################################################

Q1: What is a model registry and why is it essential for MLOps?

Answer:
A model registry is a centralized catalog for managing ML model lifecycle.
It serves as the "source of truth" for all trained models.

Key capabilities:
- Version management: Track all model versions with metadata.
- Stage transitions: None -> Staging -> Production -> Archived.
- Approval workflows: Require human approval before production deployment.
- Metadata storage: Metrics, hyperparameters, training data lineage, author.
- Artifact management: Links to stored model files (S3, Blob, GCS).
- Model lineage: Track which data, code, and config produced each model.

Major model registries:

1. MLflow Model Registry:
   - Open source, widely adopted.
   - Stage transitions: None, Staging, Production, Archived.
   - REST API for programmatic access.
   - Integration with MLflow tracking for experiment lineage.

   import mlflow
   # Log and register model
   with mlflow.start_run():
       mlflow.log_params({"lr": 0.001, "epochs": 50})
       mlflow.log_metrics({"accuracy": 0.95, "f1": 0.93})
       mlflow.pytorch.log_model(model, "model",
           registered_model_name="sentiment-classifier")

   # Transition stage
   client = mlflow.tracking.MlflowClient()
   client.transition_model_version_stage(
       name="sentiment-classifier",
       version=3,
       stage="Production"
   )

2. SageMaker Model Registry:
   - Model packages and model package groups.
   - Approval status: PendingManualApproval, Approved, Rejected.
   - Integrated with SageMaker Pipelines.

3. Azure ML Model Registry:
   - Versioned models with tags and properties.
   - Integration with Azure ML endpoints.

4. Weights & Biases Model Registry:
   - Linked to W&B experiment tracking.
   - Model cards, lineage graphs.

5. Neptune.ai:
   - Model metadata management.

6. DVC with Git:
   - Lightweight, Git-based model tracking.

-------------------------------------------------------------------------------

Q2: How do you implement model approval workflows?

Answer:
Model approval ensures only validated models reach production.

Workflow:
  1. Training pipeline produces model artifact.
  2. Evaluation pipeline computes metrics on test set.
  3. Model registered with status "Pending".
  4. Automated checks:
     - Accuracy >= baseline + threshold
     - Latency p95 <= SLA requirement
     - No bias drift detected
     - Model size within deployment constraints
  5. If automated checks pass -> status = "Staging".
  6. Human review: ML engineer reviews metrics, model card.
  7. Staging deployment: Deploy to staging, run integration tests.
  8. Human approval: Senior ML engineer or product owner approves.
  9. Status = "Production" -> automated deployment triggered.

SageMaker Pipelines approval:
  from sagemaker.workflow.conditions import ConditionGreaterThan
  from sagemaker.workflow.condition_step import ConditionStep

  condition = ConditionGreaterThan(
      left=JsonGet(step_name="evaluate", property_file=eval_report,
                   json_path="metrics.accuracy"),
      right=0.90
  )

  condition_step = ConditionStep(
      name="CheckAccuracy",
      conditions=[condition],
      if_steps=[register_step, deploy_step],
      else_steps=[notify_failure_step]
  )

##############################################################################
# 6.2  EXPERIMENT TRACKING
##############################################################################

Q3: What is experiment tracking and which tools are commonly used?

Answer:
Experiment tracking records all details of ML experiments to enable
reproducibility, comparison, and collaboration.

What to track:
- Code version (git commit hash).
- Data version (DVC hash or dataset version).
- Hyperparameters (learning rate, batch size, architecture, etc.).
- Metrics (loss, accuracy, F1, AUC, latency, etc.) at each epoch/step.
- Artifacts (model checkpoints, plots, confusion matrices).
- Environment (Python version, package versions, hardware).
- Duration and resource utilization.

Major tools:

1. MLflow Tracking:
   - Open source, self-hosted or managed (Databricks).
   - Concepts: Experiments, Runs, Parameters, Metrics, Artifacts.
   - UI for comparing runs, visualizing metrics.

   import mlflow
   mlflow.set_tracking_uri("http://mlflow-server:5000")
   mlflow.set_experiment("sentiment-analysis")

   with mlflow.start_run(run_name="bert-finetune-v3"):
       mlflow.log_param("model_type", "bert-base-uncased")
       mlflow.log_param("learning_rate", 0.00002)
       mlflow.log_param("batch_size", 32)
       mlflow.log_param("epochs", 5)

       for epoch in range(5):
           train_loss, val_acc = train_epoch(model, data)
           mlflow.log_metrics({
               "train_loss": train_loss,
               "val_accuracy": val_acc
           }, step=epoch)

       mlflow.log_metric("test_accuracy", 0.945)
       mlflow.log_artifact("confusion_matrix.png")
       mlflow.pytorch.log_model(model, "model")

2. Weights & Biases (W&B):
   - Cloud-hosted (free tier available), popular in research.
   - Rich visualization, hyperparameter sweeps, model lineage.
   - Collaborative features (reports, team dashboards).

   import wandb
   wandb.init(project="sentiment-analysis", config={
       "model": "bert-base", "lr": 2e-5, "epochs": 5
   })
   for epoch in range(5):
       wandb.log({"train_loss": loss, "val_acc": acc, "epoch": epoch})
   wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(...)})
   wandb.finish()

3. SageMaker Experiments:
   - Native AWS integration.
   - Tracks training jobs, processing jobs, transform jobs automatically.

4. Azure ML Experiments:
   - Native Azure integration.
   - mlflow.set_tracking_uri(azureml_mlflow_uri) for MLflow compatibility.

5. Neptune.ai:
   - Enterprise experiment tracking.
   - Strong metadata management.

6. CometML:
   - Cloud + self-hosted.
   - Code comparison, diff visualization.

Best practices:
- Log EVERYTHING: Better to over-log than under-log.
- Use consistent naming conventions across experiments.
- Tag experiments with purpose (exploration, ablation, production-candidate).
- Auto-log when possible (mlflow.autolog()).
- Link experiments to Git commits and data versions.

##############################################################################
# 6.3  MODEL MONITORING
##############################################################################

Q4: How do you monitor ML models in production?

Answer:
ML monitoring goes beyond traditional software monitoring because model
performance can degrade silently without any system errors.

Types of ML monitoring:

1. Data Drift:
   - Input feature distributions shift from training data.
   - Detection methods:
     * Population Stability Index (PSI)
     * Kolmogorov-Smirnov (KS) test
     * Jensen-Shannon divergence
     * Chi-squared test (categorical features)
   - Example: Training data had age range 18-65, production sees 13-80.

2. Concept Drift:
   - Relationship between features and target changes.
   - Example: Customer behavior changes after market shift.
   - Detection: Monitor prediction accuracy over time (requires ground truth).
   - Types:
     * Sudden drift: Abrupt change (new regulation).
     * Gradual drift: Slow change over weeks/months.
     * Seasonal drift: Cyclical patterns (holiday shopping).
     * Recurring drift: Patterns that come and go.

3. Model Quality / Performance:
   - Track accuracy, precision, recall, F1, AUC over time.
   - Requires ground truth labels (often delayed).
   - Proxy metrics: Prediction confidence distribution, output entropy.

4. Prediction Drift:
   - Output distribution changes even if inputs look similar.
   - Monitor: Mean prediction, prediction distribution, class ratios.

5. Operational Monitoring:
   - Latency (p50, p95, p99).
   - Throughput (requests per second).
   - Error rates (4xx, 5xx).
   - Resource utilization (GPU, memory, CPU).
   - Queue depth (for async inference).

Tools for ML monitoring:
- Evidently AI: Open source, data drift + model quality reports.
- WhyLabs / whylogs: Data profiling and drift detection.
- NannyML: Performance estimation without ground truth.
- Arize AI: Commercial ML observability platform.
- SageMaker Model Monitor: Native AWS.
- Azure ML Model Monitoring: Native Azure.
- Prometheus + Grafana: Custom metrics dashboards.
- Fiddler AI: Model monitoring and explainability.

Example monitoring setup with Evidently:
  from evidently.report import Report
  from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

  report = Report(metrics=[
      DataDriftPreset(),
      TargetDriftPreset()
  ])
  report.run(
      reference_data=training_df,
      current_data=production_df
  )
  report.save_html("drift_report.html")

Alert thresholds:
  - PSI > 0.2: Significant drift, investigate immediately.
  - PSI 0.1-0.2: Moderate drift, monitor closely.
  - PSI < 0.1: No significant drift.
  - Accuracy drop > 5%: Trigger retraining pipeline.
  - Latency p99 > SLA: Scale up or optimize.

##############################################################################
# 6.4  DATA VERSIONING
##############################################################################

Q5: How do you implement data versioning for ML?

Answer:
Data versioning ensures reproducibility by tracking exact datasets used
for training, evaluation, and testing.

Tools and approaches:

1. DVC (Data Version Control):
   - Git-like commands for large data files.
   - Stores metadata in Git, data in remote storage (S3, GCS, Azure).
   - Supports pipelines (dvc.yaml) for reproducible workflows.

   # Initialize DVC in Git repo
   dvc init
   dvc remote add -d myremote s3://my-bucket/dvc-storage

   # Track a dataset
   dvc add data/training_data.parquet
   git add data/training_data.parquet.dvc data/.gitkeep
   git commit -m "Add training data v1"
   dvc push

   # Pipeline definition (dvc.yaml)
   stages:
     preprocess:
       cmd: python src/preprocess.py
       deps:
         - src/preprocess.py
         - data/raw/
       outs:
         - data/processed/
     train:
       cmd: python src/train.py
       deps:
         - src/train.py
         - data/processed/
       outs:
         - models/model.pt
       metrics:
         - metrics.json:
             cache: false

   # Reproduce pipeline
   dvc repro

   # Switch to different data version
   git checkout v1.0
   dvc checkout

2. LakeFS:
   - Git-like branching for data lakes (S3-compatible API).
   - Create branches for experiments, merge data changes.
   - Zero-copy branching (metadata only, no data duplication).

   lakectl branch create lakefs://repo/experiment-1 \
     --source lakefs://repo/main
   # Experiment with data modifications
   lakectl commit lakefs://repo/experiment-1 -m "Add augmented data"
   lakectl merge lakefs://repo/experiment-1 lakefs://repo/main

3. Delta Lake:
   - ACID transactions on data lakes.
   - Time travel: Query data at any previous version.
   - Schema enforcement and evolution.

   # Read data at a specific version
   df = spark.read.format("delta").option("versionAsOf", 5).load(path)
   # Or by timestamp
   df = spark.read.format("delta").option("timestampAsOf",
       "2025-01-15").load(path)

4. Dataset versioning in ML platforms:
   - SageMaker Data Assets: Versioned references to S3 data.
   - Azure ML Data Assets: URI file/folder/MLTable with versions.
   - Hugging Face Datasets: Version-controlled dataset library.

Best practices:
- Never modify data in place; create new versions.
- Store data schema alongside data (schema versioning).
- Track data lineage: raw -> preprocessed -> feature-engineered.
- Use immutable storage (S3 Object Lock) for regulatory compliance.
- Hash data files for integrity verification.
- Document data changes in data cards (what changed, why, impact).

##############################################################################
# 6.5  FEATURE STORES
##############################################################################

Q6: What is a feature store and why is it important for MLOps?

Answer:
A feature store is a centralized system for managing, storing, and serving
ML features. It solves the problem of feature engineering being duplicated,
inconsistent, and siloed across teams.

Core problems solved:
1. Training-serving skew: Features computed differently in training vs serving.
2. Feature duplication: Multiple teams compute the same features independently.
3. Feature discovery: Hard to find existing features across the organization.
4. Point-in-time correctness: Prevent data leakage in training by ensuring
   features reflect the state at prediction time.

Architecture:
  Feature Engineering -> Feature Store (Online + Offline)
                              |
            +-----------------+-----------------+
            |                                   |
    Online Store (low latency)        Offline Store (batch)
    Redis, DynamoDB, etc.             S3, BigQuery, etc.
            |                                   |
    Real-time Inference              Model Training
    (< 10ms lookups)                 (historical features)

Feature store components:
- Feature definitions: Schema, transformations, metadata.
- Online store: Low-latency key-value lookups for serving.
- Offline store: Historical feature data for training.
- Feature pipelines: Batch and streaming feature computation.
- Feature registry: Catalog of all available features.
- Point-in-time joins: Correct temporal joins for training data.

Major feature stores:

1. Feast (open source):
   # feature_store.yaml
   project: my_ml_project
   registry: s3://feast-registry/registry.db
   provider: aws
   online_store:
     type: dynamodb
     region: us-east-1
   offline_store:
     type: file  # or redshift, bigquery, snowflake

   # feature definitions
   from feast import Entity, Feature, FeatureView, FileSource, ValueType

   customer = Entity(name="customer_id", value_type=ValueType.INT64)
   customer_features = FeatureView(
       name="customer_features",
       entities=[customer],
       ttl=timedelta(days=1),
       schema=[
           Field(name="total_purchases", dtype=Float64),
           Field(name="avg_order_value", dtype=Float64),
           Field(name="days_since_last_order", dtype=Int64),
       ],
       source=FileSource(path="s3://data/customer_features.parquet",
                         timestamp_field="event_timestamp")
   )

   # Materialize to online store
   feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)

   # Get features for inference
   feature_vector = store.get_online_features(
       features=["customer_features:total_purchases",
                  "customer_features:avg_order_value"],
       entity_rows=[{"customer_id": 12345}]
   ).to_dict()

   # Get training data with point-in-time join
   training_df = store.get_historical_features(
       entity_df=entity_df_with_timestamps,
       features=["customer_features:total_purchases",
                  "customer_features:avg_order_value"]
   ).to_df()

2. SageMaker Feature Store:
   - Managed AWS service.
   - Online (DynamoDB-backed) + Offline (S3 Parquet) stores.

3. Tecton:
   - Enterprise feature platform built on Feast concepts.
   - Real-time feature engineering, streaming features.

4. Databricks Feature Store:
   - Integrated with Delta Lake and MLflow.
   - Unity Catalog for feature discovery.

5. Vertex AI Feature Store (GCP):
   - Managed feature store on Google Cloud.

Feature pipeline types:
- Batch: Run on schedule (hourly/daily). Most common.
  Tools: Spark, dbt, Airflow.
- Streaming: Real-time feature computation from event streams.
  Tools: Kafka + Flink, Spark Structured Streaming.
- On-demand: Compute at request time (e.g., current time features).
  Tools: Custom code in serving layer.
