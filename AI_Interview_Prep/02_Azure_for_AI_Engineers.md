================================================================================
  SECTION 2: AZURE FOR AI ENGINEERS - INTERVIEW QUESTIONS & ANSWERS (2025-2026)
================================================================================

##############################################################################
# 2.1  AZURE OPENAI SERVICE
##############################################################################

Q1: What is Azure OpenAI Service and how does it differ from using OpenAI directly?

Answer:
Azure OpenAI Service provides REST API access to OpenAI's models (GPT-4,
GPT-4o, GPT-4 Turbo, o1, o3-mini, DALL-E 3, Whisper, text-embedding-ada-002,
text-embedding-3-large) through Azure's infrastructure.

Key differences from OpenAI direct:
- Enterprise security: Azure AD authentication, RBAC, private endpoints,
  managed identity, VNET integration.
- Data privacy: Data is NOT used for model training. Azure processes data
  within your selected region.
- Compliance: SOC 2, HIPAA, GDPR, FedRAMP certifications.
- Content filtering: Built-in content safety system (configurable severity
  levels for hate, violence, sexual, self-harm categories).
- Provisioned Throughput Units (PTUs): Reserved capacity with guaranteed
  latency (unlike OpenAI's shared capacity).
- Regional deployment: Choose specific Azure regions for data residency.
- SLA: 99.9% uptime SLA (OpenAI has no formal SLA).

Deployment models:
- Standard: Pay-per-token, shared capacity, best-effort latency.
- Provisioned (PTU): Reserved capacity, consistent latency, monthly billing.
- Global: Routes to nearest available region automatically.

API compatibility:
- Uses OpenAI-compatible API format but with Azure-specific auth.
- Endpoint format: https://{resource-name}.openai.azure.com/
- Auth: API key or Azure AD token (Bearer token).
- Each model must be explicitly deployed with a deployment name.

-------------------------------------------------------------------------------

Q2: How do you implement RAG with Azure OpenAI?

Answer:
Azure provides "On Your Data" feature for built-in RAG:

Architecture:
  User Query -> Azure OpenAI -> Azure AI Search (retrieval) -> Azure OpenAI
  (generation with context) -> Response with citations

Setup steps:
1. Upload documents to Azure Blob Storage.
2. Create Azure AI Search index:
   - Configure indexer to process documents from Blob Storage.
   - Enable semantic search for improved relevance ranking.
   - Configure vector search (integrated vectorization).
3. Connect Azure OpenAI to AI Search:
   - Use the "On Your Data" feature in Azure OpenAI Studio.
   - Or programmatically via the API with data_sources parameter.

API call with data source:
  POST /openai/deployments/{deployment}/chat/completions
  {
    "messages": [{"role": "user", "content": "What is our refund policy?"}],
    "data_sources": [{
      "type": "azure_search",
      "parameters": {
        "endpoint": "https://my-search.search.windows.net",
        "index_name": "my-index",
        "authentication": {"type": "api_key", "key": "<key>"},
        "query_type": "vector_semantic_hybrid",
        "embedding_dependency": {
          "type": "deployment_name",
          "deployment_name": "text-embedding-ada-002"
        },
        "strictness": 3,
        "top_n_documents": 5
      }
    }]
  }

Query types:
- simple: Keyword matching.
- semantic: AI-powered semantic ranking.
- vector: Embedding-based similarity search.
- vector_simple_hybrid: Vector + keyword.
- vector_semantic_hybrid: Vector + keyword + semantic ranking (best quality).

-------------------------------------------------------------------------------

Q3: Explain Azure OpenAI content filtering and responsible AI.

Answer:
Azure OpenAI includes a multi-layered content filtering system:

Layers:
1. Input filter: Analyzes user prompt before sending to model.
2. Model processing: Model generates response.
3. Output filter: Analyzes model response before returning to user.

Categories:
- Hate and fairness
- Sexual content
- Violence
- Self-harm
- Jailbreak detection (prompt injection protection)
- Protected material (copyrighted content detection)

Severity levels (per category):
- Safe, Low, Medium, High
- Default: Block Medium and High across all categories.
- Configurable: Can adjust thresholds per category per deployment.

Custom content filters:
- Blocklists: Define custom blocked terms/phrases.
- Annotations: Get filter results in API response metadata for logging.

##############################################################################
# 2.2  AZURE AI STUDIO
##############################################################################

Q4: What is Azure AI Studio and its key capabilities?

Answer:
Azure AI Studio (now part of Azure AI Foundry) is a unified platform for
building generative AI applications. It consolidates multiple Azure AI services.

Key capabilities:
- Model Catalog: Browse and deploy models from OpenAI, Meta (Llama), Mistral,
  Microsoft (Phi), Hugging Face, NVIDIA, and others.
- Prompt Flow: Visual workflow builder for LLM applications. Chain prompts,
  tools, and code. Supports evaluation and deployment.
- Playground: Interactive environment to test models, system prompts, and
  parameters (temperature, top_p, max_tokens, etc.).
- Evaluation: Built-in evaluation metrics for LLM apps:
  * Groundedness: Is the response grounded in the provided context?
  * Relevance: Is the response relevant to the question?
  * Coherence: Is the response logically coherent?
  * Fluency: Is the response grammatically correct?
  * GPT Similarity: Semantic similarity to reference answers.
  * Custom metrics: Define your own evaluation criteria.
- Fine-tuning: Fine-tune models directly from the studio.
- Deployment: One-click deployment to managed endpoints.
- Monitoring: Track deployed model performance, token usage, latency.

Project structure:
- AI Hub: Shared resource for organization (compute, storage, connections).
- AI Project: Individual workspace within a hub for building AI apps.
- Connections: Configured links to Azure services (Search, Storage, etc.).

Prompt Flow components:
- LLM nodes: Call Azure OpenAI or other model endpoints.
- Python nodes: Custom code execution.
- Tool nodes: Pre-built integrations (search, embeddings, etc.).
- Prompt nodes: Templated prompts with variable substitution.
- Variants: A/B test different prompt versions.

##############################################################################
# 2.3  AZURE FUNCTIONS
##############################################################################

Q5: How do you use Azure Functions for AI workloads?

Answer:
Azure Functions is Azure's serverless compute for event-driven AI workloads.

Use cases for AI:
1. Lightweight inference: Deploy small models for real-time predictions.
2. Pre/post processing: Transform data before/after model inference.
3. Event-driven triggers: Process new data uploads, queue messages, HTTP calls.
4. Orchestration: Durable Functions for complex ML workflows.

Hosting plans:
- Consumption: Auto-scale, pay per execution, cold starts.
  Max timeout: 5 min (default), 10 min (configurable).
- Premium: Pre-warmed instances (no cold start), VNET integration.
  Max timeout: unlimited. Supports larger packages.
- Dedicated (App Service Plan): Full control, always running.

Key triggers for AI:
- HTTP Trigger: REST API for model inference.
- Blob Trigger: Process new files (images, documents) for AI analysis.
- Queue Trigger: Process inference requests asynchronously.
- Timer Trigger: Scheduled model retraining or evaluation.
- Event Grid Trigger: React to Azure resource events.
- Cosmos DB Trigger: Process new/updated documents.

Limitations for ML:
- No GPU support.
- Memory limits: 1.5GB (Consumption), 14GB (Premium).
- Package size: 1GB max.
- Cold starts: 1-10 seconds (Consumption plan).

Example - Azure Function calling Azure OpenAI:

  import azure.functions as func
  from openai import AzureOpenAI
  import json

  client = AzureOpenAI(
      azure_endpoint="https://myresource.openai.azure.com/",
      api_key=os.environ["AZURE_OPENAI_KEY"],
      api_version="2024-02-01"
  )

  def main(req: func.HttpRequest) -> func.HttpResponse:
      body = req.get_json()
      response = client.chat.completions.create(
          model="gpt-4o",
          messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": body["prompt"]}
          ]
      )
      return func.HttpResponse(
          json.dumps({"response": response.choices[0].message.content}),
          mimetype="application/json"
      )

Durable Functions for ML pipelines:
- Fan-out/fan-in: Process multiple data chunks in parallel.
- Chaining: Sequential steps (preprocess -> infer -> postprocess).
- Human interaction: Wait for human approval in model deployment.
- Monitor: Long-running model training with status polling.

##############################################################################
# 2.4  AZURE BLOB STORAGE
##############################################################################

Q6: How is Azure Blob Storage used in AI/ML workflows?

Answer:
Azure Blob Storage serves as the primary data lake for ML workloads.

Storage tiers:
- Hot: Frequently accessed training data, current model artifacts.
- Cool: Previous model versions, older datasets (lower storage cost, higher
  access cost).
- Cold: Rarely accessed data, minimum 90-day retention.
- Archive: Long-term compliance storage, minimum 180-day retention.

ML-specific patterns:
1. Model artifact storage:
   - Store serialized models (ONNX, SavedModel, .pt, .pkl, .safetensors).
   - Use blob versioning for model version tracking.
   - Immutability policies for regulatory compliance.

2. Training data management:
   - Azure Data Lake Storage Gen2 (ADLS Gen2) = Blob Storage + hierarchical
     namespace. Better for structured data operations.
   - Supports Parquet, CSV, JSON, TFRecord, images, etc.
   - Azure ML can mount blob containers as datasets directly.

3. Integration with Azure ML:
   - Datastores: Register blob containers as Azure ML datastores.
   - Datasets: Create versioned dataset references.
   - Data assets: Track data lineage and provenance.

4. Integration with AI Search (for RAG):
   - Blob indexer crawls containers and extracts text from documents.
   - Built-in AI enrichment: OCR, key phrases, entity recognition.

Security:
- Azure AD + RBAC for access control.
- Managed identities (no API keys in code).
- Private endpoints + VNET integration.
- Encryption at rest (Microsoft-managed or customer-managed keys).
- SAS tokens for time-limited access.

Performance:
- Premium Blob Storage: SSD-backed, low latency.
- AzCopy: High-performance CLI for bulk data transfer.
- Blob NFS v3: Mount as NFS for Linux-based training workloads.

##############################################################################
# 2.5  AZURE MACHINE LEARNING
##############################################################################

Q7: What are the key components of Azure Machine Learning?

Answer:
Azure ML is the enterprise MLOps platform. Key components:

Workspace Resources:
- Compute Instances: Dev VMs for data scientists (with GPU options).
- Compute Clusters: Auto-scaling clusters for training jobs.
  Instance types: Standard_NC24ads_A100_v4 (A100), Standard_ND96amsr_A100_v4
  (8x A100).
- Inference Clusters: AKS-based for real-time serving.
- Managed Endpoints: Simplified deployment (online and batch).
- Serverless Compute: On-demand, no cluster management.

Data Management:
- Datastores: Connections to Blob, ADLS, SQL, etc.
- Data Assets: Versioned references to datasets (URI file, URI folder, MLTable).
- MLTable: Tabular data abstraction with schema.

Training:
- Jobs: Submit training scripts to managed compute.
  Types: Command jobs, Sweep jobs (hyperparameter tuning), Pipeline jobs.
- Environments: Docker-based environments (curated or custom).
  Curated environments: AzureML-sklearn, AzureML-pytorch, AzureML-tensorflow.
- Components: Reusable pipeline building blocks.

MLOps:
- Model Registry: Central model catalog with versioning, tagging, staging.
- Managed Endpoints:
  * Online endpoints: Real-time inference with auto-scaling.
    - Managed (Azure manages infrastructure).
    - Kubernetes (bring your own AKS cluster).
  * Batch endpoints: Score large datasets in batch jobs.
- Pipelines: Multi-step ML workflows with scheduling.
- Model Monitoring: Data drift, prediction drift, data quality.

CLI v2 and SDK v2:
  # CLI v2 - submit a training job
  az ml job create --file train-job.yml --workspace-name myws

  # SDK v2 - create and submit
  from azure.ai.ml import MLClient, command
  ml_client = MLClient(credential, subscription_id, rg, ws)
  job = command(
      code="./src",
      command="python train.py --lr 0.01",
      environment="AzureML-pytorch-2.0@latest",
      compute="gpu-cluster",
  )
  ml_client.jobs.create_or_update(job)

-------------------------------------------------------------------------------

Q8: How do you deploy models with Azure ML Managed Endpoints?

Answer:
Managed Online Endpoints deployment:

1. Register the model:
  az ml model create --name my-model --version 1 \
    --path ./model --type custom_model

2. Create endpoint:
  az ml online-endpoint create --name my-endpoint

3. Create deployment:
  # deployment.yml
  $schema: https://azuremlschemas.azureedge.net/latest/
           managedOnlineDeployment.schema.json
  name: blue
  endpoint_name: my-endpoint
  model: azureml:my-model:1
  code_configuration:
    code: ./src
    scoring_script: score.py
  environment: azureml:AzureML-pytorch-2.0@latest
  instance_type: Standard_NC6s_v3
  instance_count: 1
  request_settings:
    request_timeout_ms: 90000
    max_concurrent_requests_per_instance: 10
  liveness_probe:
    initial_delay: 300
  readiness_probe:
    initial_delay: 300
  scale_settings:
    type: target_utilization
    min_instances: 1
    max_instances: 5
    target_utilization_percentage: 70
    polling_interval: 30

4. Scoring script (score.py):
  import json, torch, os
  def init():
      global model
      model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pt")
      model = torch.load(model_path)
      model.eval()

  def run(raw_data):
      data = json.loads(raw_data)
      tensor = torch.tensor(data["input"])
      with torch.no_grad():
          prediction = model(tensor)
      return prediction.tolist()

Traffic splitting (Blue-Green):
  az ml online-endpoint update --name my-endpoint \
    --traffic "blue=90 green=10"

##############################################################################
# 2.6  AZURE COGNITIVE SERVICES (NOW AZURE AI SERVICES)
##############################################################################

Q9: What are Azure AI Services and their ML engineering use cases?

Answer:
Azure AI Services (formerly Cognitive Services) provide pre-built AI APIs:

Vision:
- Computer Vision: Image analysis, OCR, spatial analysis.
- Custom Vision: Train custom image classifiers/detectors with few samples.
- Face API: Face detection, recognition, verification.
- Document Intelligence (Form Recognizer): Extract structured data from
  documents (invoices, receipts, IDs).

Language:
- Language Service: Sentiment analysis, NER, key phrase extraction,
  summarization, PII detection, custom text classification.
- Translator: Real-time text translation (100+ languages).
- Immersive Reader: Text-to-speech, reading assistance.

Speech:
- Speech to Text: Real-time and batch transcription.
- Text to Speech: Neural TTS with custom voice.
- Speech Translation: Real-time speech translation.
- Speaker Recognition: Voice identification and verification.

Decision:
- Content Safety: Detect harmful content in text and images.
- Personalizer: Real-time personalization with reinforcement learning.

For AI Engineers, key integration patterns:
1. Container deployment: Most services available as Docker containers for
   on-premises or edge deployment. Run disconnected from cloud.
2. Custom models: Fine-tune pre-built models with your data (Custom Vision,
   Custom Speech, Custom Language).
3. Multi-service resource: Single endpoint for multiple AI services.
4. Managed identity auth: No API keys in code.

Q10: How do you compare AWS vs Azure for AI workloads?

Answer:
Comparison matrix:

| Category            | AWS                          | Azure                          |
|---------------------|------------------------------|--------------------------------|
| ML Platform         | SageMaker                    | Azure ML                       |
| LLM API Service     | Bedrock                      | Azure OpenAI Service           |
| Model Hub           | SageMaker JumpStart          | Azure AI Model Catalog         |
| GPU Instances       | P5 (H100), P4d (A100)       | ND H100, NC A100               |
| Custom AI Chips     | Inferentia2, Trainium        | Maia 100 (2024+)               |
| Serverless          | Lambda                       | Azure Functions                |
| Container Orch.     | ECS, EKS                    | ACI, AKS                       |
| Object Storage      | S3                           | Blob Storage / ADLS Gen2       |
| AI Search / RAG     | Kendra, OpenSearch           | Azure AI Search                |
| Pre-built AI APIs   | Rekognition, Comprehend,     | Azure AI Services              |
|                     | Textract, Polly, Transcribe  |                                 |
| Experiment Tracking | SageMaker Experiments        | Azure ML Experiments           |
| Feature Store       | SageMaker Feature Store      | Azure ML Feature Store (preview)|
| Model Monitor       | SageMaker Model Monitor      | Azure ML Model Monitoring      |
| API Management      | API Gateway                  | Azure API Management           |
| IaC                 | CloudFormation, CDK          | ARM Templates, Bicep           |
| IAM                 | IAM (policies + roles)       | Azure AD + RBAC                |

Key differentiator: Azure has exclusive access to OpenAI models (GPT-4, GPT-5,
o-series) with enterprise features. AWS Bedrock offers multi-provider choice
(Claude, Llama, Mistral, Nova, etc.).

##############################################################################
# 2.8  LATEST AZURE AI UPDATES (Late 2025 - Early 2026)
##############################################################################

Q11: What is Microsoft Foundry and how does it replace Azure AI Studio?

Answer:
Microsoft Foundry (formerly Azure AI Studio / Azure AI Foundry) is the unified
platform for building enterprise AI applications. Rebranded in late 2025.

Foundry comprises natively integrated services:
- Foundry Models: Model catalog and deployment (OpenAI, Meta, Mistral, Phi)
- Foundry Agent Service: Build and deploy AI agents with tools and memory
- Foundry Tools: Pre-built tools for RAG, code execution, web search
- Foundry IQ: Evolution of Azure AI Search -- semantic search and retrieval
- Foundry Control Plane: Governance, monitoring, and policy management
- Foundry Local: Run AI locally for development and edge scenarios
- Azure Machine Learning: Full ML lifecycle management

Key additions include the Foundry Agent Service which allows building agents
that can use tools, access databases, and orchestrate multi-step workflows
natively within the Azure ecosystem.

Q12: What are the latest Azure Cosmos DB vector search improvements?

Answer:
Azure Cosmos DB received major AI-focused updates at Ignite 2025:

1. Float16 Vector Embeddings: Cuts storage by 50%, 30% faster vector
   ingestion, 300% lower P99 latency.

2. Full-Text Search GA: Fuzzy matching, new language support, enabling
   hybrid search (vector + keyword) in a single database.

3. Cosmos DB MCP Toolkit (public preview): Integrates the NoSQL API with
   Microsoft's Foundry Agent Service, allowing AI agents to perform vector
   searches and data operations against live Cosmos DB data.

4. Semantic Reranking (private preview): Applied to any query type, uses
   Azure AI Search's reranking model to reorder results by semantic meaning.

5. Foundry Connection: Direct integration between Cosmos DB and Foundry
   for seamless RAG pipelines.

Q13: What is Foundry Agent Service and how do you build agents?

Answer:
Foundry Agent Service allows building production AI agents that can:
- Use tools (web search, code execution, database queries)
- Access Cosmos DB for vector search and data operations via MCP
- Leverage Azure AI Search (now Foundry IQ) for RAG
- Maintain conversation memory across sessions
- Follow governance policies from Foundry Control Plane

Agents can be built using:
- Azure SDK for Python/JavaScript
- REST API
- Foundry Studio (visual builder)
- Integration with LangChain/LangGraph via Azure connectors

> YOUR EXPERIENCE: At RavianAI, you can leverage Azure's Foundry platform
> for enterprise-grade AI agent deployment. The Cosmos DB vector search and
> MCP Toolkit align with building RAG-powered agents. Azure Functions for
> serverless AI processing connects to your experience at Fealty Technologies
> building web applications on Azure.
