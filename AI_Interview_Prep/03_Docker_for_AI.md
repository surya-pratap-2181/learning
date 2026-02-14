================================================================================
  SECTION 3: DOCKER FOR AI/ML - INTERVIEW QUESTIONS & ANSWERS (2025-2026)
================================================================================

##############################################################################
# 3.1  CONTAINERIZING ML MODELS
##############################################################################

Q1: Why containerize ML models and what are the key benefits?

Answer:
Containerization solves critical ML deployment challenges:

1. Reproducibility: Same container runs identically in dev, staging, prod.
   Eliminates "it works on my machine" for ML (CUDA versions, Python deps,
   framework versions).

2. Dependency isolation: ML frameworks have complex, conflicting dependencies.
   PyTorch + CUDA 12.1 vs TensorFlow + CUDA 11.8 can coexist in separate
   containers.

3. Portability: Deploy the same container to ECS, EKS, Azure AKS, GCP GKE,
   or on-premises GPU servers.

4. Scaling: Container orchestrators (Kubernetes, ECS) can auto-scale inference
   containers based on load.

5. CI/CD integration: Containers are immutable artifacts that flow through
   testing, staging, and production pipelines.

6. Model versioning: Tag containers with model version + code version.
   Easy rollback by deploying previous container image.

-------------------------------------------------------------------------------

Q2: Write a production Dockerfile for serving a PyTorch model.

Answer:
  # Production Dockerfile for PyTorch inference
  FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

  # Prevent interactive prompts during build
  ENV DEBIAN_FRONTEND=noninteractive
  ENV PYTHONUNBUFFERED=1
  ENV PYTHONDONTWRITEBYTECODE=1

  # Install Python and system dependencies
  RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      libgomp1 \
      && rm -rf /var/lib/apt/lists/*

  # Create non-root user for security
  RUN useradd --create-home --shell /bin/bash modeluser
  WORKDIR /app

  # Install Python dependencies (separate layer for caching)
  COPY requirements.txt .
  RUN pip3 install --no-cache-dir -r requirements.txt

  # Copy model artifacts
  COPY model/ ./model/

  # Copy application code
  COPY src/ ./src/

  # Switch to non-root user
  USER modeluser

  # Health check
  HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
      CMD curl -f http://localhost:8080/health || exit 1

  EXPOSE 8080

  # Use gunicorn for production
  CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", \
       "--threads", "4", "--timeout", "120", "src.app:app"]

Key considerations:
- Use --no-cache-dir with pip to reduce image size.
- Non-root user for security (required by many K8s clusters).
- HEALTHCHECK for orchestrator liveness probes.
- Single worker for GPU (GPU doesn't benefit from multiple workers).
- Multiple threads for I/O-bound preprocessing.

##############################################################################
# 3.2  MULTI-STAGE BUILDS FOR ML
##############################################################################

Q3: How do multi-stage builds optimize ML Docker images?

Answer:
Multi-stage builds separate build-time dependencies from runtime, dramatically
reducing final image size.

Example - Multi-stage build for ONNX Runtime inference:

  # Stage 1: Builder - compile dependencies
  FROM python:3.11-slim AS builder

  WORKDIR /build

  # Install build tools (only needed for compilation)
  RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ cmake \
      && rm -rf /var/lib/apt/lists/*

  COPY requirements.txt .
  RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

  # Stage 2: Model optimizer - convert model format
  FROM builder AS optimizer

  COPY model/pytorch_model.pt ./model/
  COPY scripts/convert_to_onnx.py .
  RUN python convert_to_onnx.py \
      --input model/pytorch_model.pt \
      --output model/optimized_model.onnx \
      --quantize int8

  # Stage 3: Runtime - minimal production image
  FROM python:3.11-slim AS runtime

  ENV PYTHONUNBUFFERED=1
  WORKDIR /app

  # Copy only installed packages from builder
  COPY --from=builder /install /usr/local

  # Copy only the optimized model from optimizer
  COPY --from=optimizer /build/model/optimized_model.onnx ./model/

  # Copy application code
  COPY src/ ./src/

  RUN useradd --create-home modeluser
  USER modeluser

  EXPOSE 8080
  CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]

Size comparison:
- Single-stage with build tools: ~5-8 GB
- Multi-stage (runtime only):    ~1-2 GB
- With model quantization (INT8): ~0.5-1 GB

Benefits for ML:
- Build tools (gcc, cmake) NOT in final image.
- Model conversion happens in build, only optimized model ships.
- Smaller images = faster pulls, faster scaling, lower registry costs.

##############################################################################
# 3.3  GPU SUPPORT IN DOCKER
##############################################################################

Q4: How does GPU support work in Docker?

Answer:
GPU access in Docker requires the NVIDIA Container Toolkit (formerly
nvidia-docker2).

Architecture:
  Container -> libnvidia-container -> NVIDIA Driver (host) -> GPU Hardware

Setup:
  # 1. Install NVIDIA driver on host
  # 2. Install NVIDIA Container Toolkit
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/\
      libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

  # 3. Run container with GPU access
  docker run --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
  # Or specify GPU count/IDs:
  docker run --gpus '"device=0,1"' my-model:latest
  docker run --gpus 2 my-model:latest

NVIDIA base images hierarchy:
- nvidia/cuda:12.1.1-base-ubuntu22.04        (~120MB) - CUDA runtime only
- nvidia/cuda:12.1.1-runtime-ubuntu22.04     (~800MB) - + cuDNN, cuBLAS
- nvidia/cuda:12.1.1-devel-ubuntu22.04       (~3.5GB) - + compilers, headers

Choose wisely:
- For inference: Use "runtime" (has cuDNN but not compilation tools).
- For training/building: Use "devel" (has nvcc compiler).
- For custom builds: Use "base" + install only what you need.

GPU memory management in containers:
- By default, container sees all GPU memory.
- Use NVIDIA_VISIBLE_DEVICES to limit GPU visibility.
- Use CUDA_MEM_FRACTION or framework-specific limits.
- MPS (Multi-Process Service): Share GPU across containers.
- MIG (Multi-Instance GPU): Partition A100/H100 into isolated instances.

Kubernetes GPU scheduling:
  resources:
    limits:
      nvidia.com/gpu: 1
  # With time-slicing (share GPU across pods):
  nvidia.com/gpu: 1  # Each pod gets time-sliced access

-------------------------------------------------------------------------------

Q5: How do you handle CUDA version compatibility in Docker?

Answer:
CUDA compatibility is a common pain point. Rules:

1. Forward compatibility: NVIDIA driver supports CUDA versions <= driver's
   CUDA version. E.g., Driver 535 (CUDA 12.2) supports CUDA 12.1, 12.0,
   11.8, etc. in containers.

2. Container CUDA version must be <= host driver CUDA version.
   The container does NOT need its own driver; it uses the host driver.

3. Framework CUDA compatibility:
   - PyTorch 2.2+: CUDA 11.8 or 12.1
   - TensorFlow 2.15+: CUDA 12.2
   - Check framework docs for exact CUDA + cuDNN versions.

Best practice:
- Pin CUDA version in Dockerfile:
  FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
- Pin framework version with CUDA variant:
  pip install torch==2.2.0+cu121 --index-url \
    https://download.pytorch.org/whl/cu121
- Document minimum driver version in deployment docs.
- Use nvidia-smi in health checks to verify GPU access.

##############################################################################
# 3.4  DOCKER COMPOSE FOR AI STACKS
##############################################################################

Q6: How do you use Docker Compose for AI/ML development stacks?

Answer:
Docker Compose orchestrates multi-container AI applications locally.

Example - Full AI inference stack:

  # docker-compose.yml
  version: "3.8"

  services:
    # Vector database for RAG
    vectordb:
      image: qdrant/qdrant:v1.7.4
      ports:
        - "6333:6333"
      volumes:
        - qdrant_data:/qdrant/storage
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
        interval: 10s
        timeout: 5s
        retries: 5

    # Redis for caching inference results
    redis:
      image: redis:7-alpine
      ports:
        - "6379:6379"
      volumes:
        - redis_data:/data

    # ML model server (GPU-enabled)
    model-server:
      build:
        context: ./model-server
        dockerfile: Dockerfile
      ports:
        - "8080:8080"
      environment:
        - MODEL_PATH=/models/model.onnx
        - CUDA_VISIBLE_DEVICES=0
        - REDIS_URL=redis://redis:6379
      volumes:
        - ./models:/models:ro
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
      depends_on:
        vectordb:
          condition: service_healthy
        redis:
          condition: service_started
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
        interval: 30s
        timeout: 10s
        start_period: 60s

    # API gateway / application
    api:
      build:
        context: ./api
        dockerfile: Dockerfile
      ports:
        - "3000:3000"
      environment:
        - MODEL_SERVER_URL=http://model-server:8080
        - VECTORDB_URL=http://vectordb:6333
        - REDIS_URL=redis://redis:6379
      depends_on:
        model-server:
          condition: service_healthy

    # Monitoring
    prometheus:
      image: prom/prometheus:v2.48.0
      ports:
        - "9090:9090"
      volumes:
        - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

    grafana:
      image: grafana/grafana:10.2.0
      ports:
        - "3001:3000"
      environment:
        - GF_SECURITY_ADMIN_PASSWORD=admin
      volumes:
        - grafana_data:/var/lib/grafana

  volumes:
    qdrant_data:
    redis_data:
    grafana_data:

Key GPU syntax in Compose:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all    # or specific number
            capabilities: [gpu]

Note: GPU support in Docker Compose requires Docker Compose v2 (the Go
rewrite) and the NVIDIA Container Toolkit.

##############################################################################
# 3.5  OPTIMIZING IMAGE SIZES FOR ML
##############################################################################

Q7: How do you optimize Docker image sizes for ML workloads?

Answer:
ML images are notoriously large (5-20GB+). Optimization strategies:

1. Choose minimal base images:
   - python:3.11-slim (~120MB) instead of python:3.11 (~900MB).
   - nvidia/cuda:12.1.1-runtime (~800MB) instead of devel (~3.5GB).
   - distroless images for maximum minimalism.

2. Multi-stage builds (see Q3 above):
   - Build dependencies in one stage, copy artifacts to slim runtime stage.
   - Typical savings: 50-70% size reduction.

3. Reduce Python package footprint:
   - pip install --no-cache-dir (prevents pip cache in image).
   - Install CPU-only PyTorch if no GPU needed:
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     (2GB vs 4.5GB for CUDA version).
   - Use ONNX Runtime (~200MB) instead of PyTorch (~2GB) for inference.
   - Remove test files: find /usr/local -name "tests" -type d -exec rm -rf {} +

4. Optimize model artifacts:
   - Quantize models: FP32 (4 bytes/param) -> FP16 (2) -> INT8 (1) -> INT4.
   - Use model-specific formats: GGUF for llama.cpp, TensorRT engines.
   - Store models externally (S3/Blob) and download at startup instead of
     baking into image.

5. Layer optimization:
   - Combine RUN commands to reduce layers:
     RUN apt-get update && apt-get install -y pkg1 pkg2 && rm -rf /var/lib/apt/lists/*
   - Order layers by change frequency (deps before code).
   - Use .dockerignore to exclude:
     __pycache__/
     *.pyc
     .git/
     data/
     notebooks/
     *.pt   # Don't copy training checkpoints
     wandb/

6. Use BuildKit features:
   - DOCKER_BUILDKIT=1 for parallel layer building.
   - Cache mounts for pip:
     RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
   - Secret mounts for private package installation:
     RUN --mount=type=secret,id=pip_conf,target=/etc/pip.conf pip install private-pkg

Size comparison table:
  | Configuration                          | Approx Size |
  |----------------------------------------|-------------|
  | Full Ubuntu + PyTorch + CUDA (devel)   | 15-20 GB    |
  | CUDA runtime + PyTorch                 | 5-8 GB      |
  | Multi-stage + CUDA runtime + PyTorch   | 3-5 GB      |
  | Slim + ONNX Runtime (CPU)              | 0.5-1 GB    |
  | Slim + ONNX Runtime + INT8 model       | 200-500 MB  |
  | Distroless + compiled binary           | 50-200 MB   |

-------------------------------------------------------------------------------

Q8: How do you handle model loading in Docker containers efficiently?

Answer:
Model loading strategies:

1. Bake model into image:
   COPY model/ /app/model/
   Pros: Self-contained, no external dependencies at runtime.
   Cons: Large images, slow pulls, rebuild needed for model updates.

2. Download at startup:
   # entrypoint.sh
   aws s3 cp s3://models/v1/model.pt /app/model/model.pt
   exec python serve.py
   Pros: Small images, easy model updates.
   Cons: Longer startup time, requires network access, S3 availability.

3. Shared volume mount:
   docker run -v /host/models:/app/model:ro my-server
   Pros: Fast startup, models shared across containers.
   Cons: Host dependency, doesn't work well with some orchestrators.

4. Init container (Kubernetes):
   initContainers:
   - name: model-downloader
     image: amazon/aws-cli
     command: ["aws", "s3", "cp", "s3://models/v1/", "/models/", "--recursive"]
     volumeMounts:
     - name: model-volume
       mountPath: /models
   containers:
   - name: inference
     image: my-inference:latest
     volumeMounts:
     - name: model-volume
       mountPath: /app/model
       readOnly: true

5. Lazy loading / memory-mapped:
   - Use mmap for large models (loads pages on demand).
   - GGUF format supports memory-mapped loading natively.
   - safetensors format supports mmap + lazy loading.

Best practice: Use init containers or download-at-startup for production.
Bake into image for edge deployments without network access.

##############################################################################
# 3.8  DOCKER FOR LLM INFERENCE SERVERS (2025-2026)
##############################################################################

Q9: How do you containerize LLM inference servers for production?

Answer:
Production Dockerfile for vLLM:
```dockerfile
FROM vllm/vllm-openai:latest
# Or pin version: vllm/vllm-openai:v0.7.0

ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ENV MAX_MODEL_LEN=8192
ENV GPU_MEMORY_UTILIZATION=0.90
ENV TENSOR_PARALLEL_SIZE=1

# Download model at build time (optional, for air-gapped)
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('${MODEL_NAME}')"

EXPOSE 8000
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server", \
    "--model", "${MODEL_NAME}", \
    "--max-model-len", "${MAX_MODEL_LEN}", \
    "--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION}", \
    "--tensor-parallel-size", "${TENSOR_PARALLEL_SIZE}"]
```

Multi-GPU Docker Compose:
```yaml
version: "3.8"
services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2  # Number of GPUs
              capabilities: [gpu]
    environment:
      - MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
      - TENSOR_PARALLEL_SIZE=2
    ports:
      - "8000:8000"
    volumes:
      - model-cache:/root/.cache/huggingface
    shm_size: "16gb"  # Required for tensor parallelism

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vllm

volumes:
  model-cache:
```

Key Docker settings for LLM inference:
- shm_size: "16gb" or higher for multi-GPU tensor parallelism
- NVIDIA Container Toolkit with CDI support (2025)
- GPU MIG partitioning for H100/A100 (share single GPU)
- GPU time-slicing in Kubernetes for cost optimization

> YOUR EXPERIENCE: At RavianAI, you managed Docker orchestration with Nginx
> load balancing for the AI platform. Containerizing LLM inference servers
> with proper GPU configuration, health checks, and scaling is directly
> relevant to your production deployment experience.
