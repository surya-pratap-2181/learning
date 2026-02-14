---
title: "FastAPI Backend Deep Dive"
layout: default
parent: "AI Frameworks & Agents"
nav_order: 8
---

# FastAPI Backend Development for AI Applications -- Comprehensive Interview Guide (2025-2026)

---

## TABLE OF CONTENTS

1. [FastAPI Fundamentals for AI](#1-fastapi-fundamentals-for-ai)
2. [Building AI APIs](#2-building-ai-apis)
3. [WebSocket Patterns for AI](#3-websocket-patterns-for-ai)
4. [Performance Optimization](#4-performance-optimization)
5. [Authentication & Security](#5-authentication--security)
6. [Deployment](#6-deployment)
7. [Database Patterns](#7-database-patterns)
8. [Django vs Flask vs FastAPI for AI](#8-django-vs-flask-vs-fastapi-comparison-for-ai)
9. [Common Interview Questions with Code Examples](#9-common-interview-questions-with-code-examples)

---

## 1. FASTAPI FUNDAMENTALS FOR AI

### 1.1 Why FastAPI for AI Applications?

FastAPI is the dominant choice for AI/ML backend services because:
- **Async-first**: Native `async/await` support critical for I/O-bound AI workloads (API calls to LLM providers, database queries, file I/O)
- **Type safety**: Pydantic models enforce strict request/response validation -- essential for AI API contracts
- **Auto-generated OpenAPI docs**: Swagger UI for testing AI endpoints interactively
- **StreamingResponse**: Native support for streaming LLM token-by-token output
- **WebSocket support**: Built-in support for real-time bidirectional AI chat
- **Performance**: One of the fastest Python frameworks (on par with Node.js and Go for I/O tasks)
- **Dependency injection**: Clean separation of concerns for model loading, auth, DB connections

### 1.2 Async Endpoints

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

# Synchronous endpoint -- blocks the event loop (BAD for I/O)
@app.get("/sync-inference")
def sync_inference():
    # This runs in a threadpool -- acceptable but not optimal
    result = expensive_cpu_computation()
    return {"result": result}

# Asynchronous endpoint -- non-blocking (GOOD for I/O)
@app.get("/async-inference")
async def async_inference():
    async with httpx.AsyncClient() as client:
        # Non-blocking HTTP call to an LLM API
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer sk-..."}
        )
    return response.json()
```

**Key Interview Points:**
- `async def` endpoints run on the main event loop -- use for I/O-bound operations
- `def` (sync) endpoints run in a threadpool automatically -- use for CPU-bound operations
- NEVER use blocking I/O (e.g., `requests.get()`, `time.sleep()`) inside `async def` -- it blocks the entire event loop
- Use `httpx.AsyncClient` instead of `requests` for async HTTP calls
- For CPU-bound AI inference (e.g., running a local model), use `def` or offload to a thread/process pool:

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)

@app.get("/cpu-inference")
async def cpu_bound_inference(text: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, run_model_inference, text)
    return {"result": result}

def run_model_inference(text: str):
    """CPU-intensive model inference -- runs in a separate process."""
    # Load and run PyTorch/TF model
    return model.predict(text)
```

### 1.3 Dependency Injection

Dependency injection is FastAPI's core pattern for managing shared resources like ML models, database connections, and API clients.

```python
from typing import Annotated
from fastapi import Depends, FastAPI
from functools import lru_cache

app = FastAPI()

# --- Model Loading as a Dependency ---
class MLModelService:
    def __init__(self):
        self.model = None

    def load_model(self):
        import torch
        self.model = torch.load("model.pt")
        self.model.eval()
        return self.model

    def predict(self, text: str):
        if self.model is None:
            self.load_model()
        return self.model(text)

# Singleton pattern using lru_cache
@lru_cache()
def get_model_service() -> MLModelService:
    service = MLModelService()
    service.load_model()
    return service

# Type alias for reuse
ModelDep = Annotated[MLModelService, Depends(get_model_service)]

@app.post("/predict")
async def predict(text: str, model_service: ModelDep):
    result = model_service.predict(text)
    return {"prediction": result}

# --- Chained Dependencies ---
async def get_db_session():
    session = AsyncSession()
    try:
        yield session  # yield-based dependency with cleanup
    finally:
        await session.close()

async def get_user_service(db: Annotated[AsyncSession, Depends(get_db_session)]):
    return UserService(db)

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    service: Annotated[UserService, Depends(get_user_service)]
):
    return await service.get(user_id)
```

**Key Interview Points:**
- Dependencies can be `async def` or `def` -- FastAPI handles both
- Use `yield` for dependencies that need cleanup (DB sessions, file handles)
- `Depends()` takes the function itself (not its return value)
- Dependencies form a DAG (directed acyclic graph) -- resolved automatically
- Use `@lru_cache()` for singleton dependencies (e.g., ML model loading)
- Dependencies are resolved per-request by default
- Sub-dependencies are cached within a single request

### 1.4 Middleware

```python
import time
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

# --- Method 1: Decorator-based middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- Method 2: Class-based middleware ---
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request
        print(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        print(f"Response status: {response.status_code}")
        return response

app.add_middleware(RequestLoggingMiddleware)

# --- AI-specific: Token counting middleware ---
class TokenCountingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # After response, log token usage for billing
        token_count = response.headers.get("X-Token-Count", "0")
        user_id = request.headers.get("X-User-ID")
        if user_id and token_count:
            await log_token_usage(user_id, int(token_count))
        return response

# --- CORS middleware (essential for AI web apps) ---
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myaiapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Middleware Execution Order:**
- Last added middleware is the outermost (first to process request, last to process response)
- Request flow: Middleware B -> Middleware A -> Route handler
- Response flow: Route handler -> Middleware A -> Middleware B

### 1.5 Background Tasks

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def log_inference_result(user_id: str, prompt: str, result: str, latency: float):
    """Runs AFTER response is sent to client."""
    with open("inference_log.jsonl", "a") as f:
        import json
        f.write(json.dumps({
            "user_id": user_id,
            "prompt": prompt,
            "result": result,
            "latency_ms": latency
        }) + "\n")

@app.post("/inference")
async def run_inference(
    prompt: str,
    user_id: str,
    background_tasks: BackgroundTasks
):
    start = time.perf_counter()
    result = await call_llm(prompt)
    latency = (time.perf_counter() - start) * 1000

    # Schedule background work AFTER response
    background_tasks.add_task(
        log_inference_result, user_id, prompt, result, latency
    )
    return {"result": result}
```

**When to use BackgroundTasks vs Celery:**
| Feature | BackgroundTasks | Celery/Redis Queue |
|---|---|---|
| Complexity | Simple | Complex setup |
| Use case | Small tasks (logging, email) | Heavy computation, retries |
| Process | Same process | Separate workers |
| Memory | Shares app memory | Independent |
| Retries | No built-in retry | Built-in retry/backoff |
| AI use case | Log inference, notify | Fine-tuning, batch processing |

### 1.6 StreamingResponse and Server-Sent Events (SSE)

This is THE most important pattern for AI applications -- streaming LLM responses token by token.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

# --- Basic StreamingResponse ---
async def generate_tokens():
    """Simulates streaming LLM output token by token."""
    tokens = ["Hello", " there", "!", " How", " can", " I", " help", " you", "?"]
    for token in tokens:
        yield token
        await asyncio.sleep(0.05)  # Simulates model generation time

@app.get("/stream")
async def stream_response():
    return StreamingResponse(
        generate_tokens(),
        media_type="text/plain"
    )

# --- Server-Sent Events (SSE) for AI Chat ---
async def sse_generator(prompt: str):
    """SSE format: each event is 'data: ...\n\n'"""
    async for token in call_llm_streaming(prompt):
        # SSE format requires 'data: ' prefix and double newline
        yield f"data: {json.dumps({'token': token})}\n\n"
    # Signal completion
    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/stream")
async def chat_stream(prompt: str):
    return StreamingResponse(
        sse_generator(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )

# --- Streaming OpenAI Responses (Production Pattern) ---
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_openai_response(messages: list[dict]):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True,
    )
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            yield f"data: {json.dumps({'content': content})}\n\n"

    yield f"data: [DONE]\n\n"

@app.post("/v1/chat/completions/stream")
async def stream_chat(messages: list[dict]):
    return StreamingResponse(
        stream_openai_response(messages),
        media_type="text/event-stream"
    )
```

**Frontend JavaScript for consuming SSE:**
```javascript
// Using EventSource (GET only)
const eventSource = new EventSource('/stream');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.done) {
        eventSource.close();
        return;
    }
    document.getElementById('output').textContent += data.token;
};

// Using fetch for POST requests (more common for AI chat)
async function streamChat(prompt) {
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                document.getElementById('output').textContent += data.content;
            }
        }
    }
}
```

---

## 2. BUILDING AI APIs

### 2.1 Streaming LLM Responses (Complete Production Pattern)

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import AsyncGenerator
import json
import time
from openai import AsyncOpenAI

app = FastAPI()
openai_client = AsyncOpenAI()

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "gpt-4"
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1024, ge=1, le=4096)
    stream: bool = True

class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    usage: dict

async def stream_llm_tokens(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Production-grade streaming with error handling."""
    try:
        response = await openai_client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )

        total_tokens = 0
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                total_tokens += 1
                yield f"data: {json.dumps({'content': content, 'type': 'token'})}\n\n"

        # Send final metadata
        yield f"data: {json.dumps({'type': 'done', 'total_tokens': total_tokens})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

@app.post("/v1/chat")
async def chat_endpoint(request: ChatRequest):
    if request.stream:
        return StreamingResponse(
            stream_llm_tokens(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # Non-streaming response
        response = await openai_client.chat.completions.create(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ChatResponse(
            id=response.id,
            content=response.choices[0].message.content,
            model=response.model,
            usage=response.usage.model_dump(),
        )
```

### 2.2 Handling Long-Running Inference

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid
import asyncio

app = FastAPI()

# In-memory job store (use Redis in production)
jobs: dict[str, dict] = {}

class InferenceRequest(BaseModel):
    text: str
    model: str = "large-model"

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result: dict | None = None
    progress: float = 0.0

async def run_long_inference(job_id: str, request: InferenceRequest):
    """Long-running task executed in background."""
    jobs[job_id]["status"] = "processing"
    try:
        # Simulate long inference
        for i in range(10):
            await asyncio.sleep(1)
            jobs[job_id]["progress"] = (i + 1) / 10

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {"output": f"Processed: {request.text}"}
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

# POST to submit job
@app.post("/inference/submit", response_model=JobStatus)
async def submit_inference(
    request: InferenceRequest,
    background_tasks: BackgroundTasks
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0.0, "result": None}
    background_tasks.add_task(run_long_inference, job_id, request)
    return JobStatus(job_id=job_id, status="pending")

# GET to poll status
@app.get("/inference/status/{job_id}", response_model=JobStatus)
async def get_inference_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        progress=job.get("progress", 0.0),
    )
```

**Production Enhancement -- Using Celery + Redis:**
```python
from celery import Celery

celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1")

@celery_app.task(bind=True)
def run_inference_task(self, text: str, model: str):
    self.update_state(state="PROCESSING", meta={"progress": 0})
    result = model_inference(text)
    return {"output": result}

@app.post("/inference/submit")
async def submit(request: InferenceRequest):
    task = run_inference_task.delay(request.text, request.model)
    return {"job_id": task.id}

@app.get("/inference/status/{job_id}")
async def status(job_id: str):
    task = celery_app.AsyncResult(job_id)
    return {"status": task.state, "result": task.result}
```

### 2.3 File Upload for Document Processing

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Annotated
import hashlib

app = FastAPI()

ALLOWED_TYPES = {"application/pdf", "text/plain", "text/csv", "application/json"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.post("/upload/document")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not allowed"
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # Generate document hash for deduplication
    doc_hash = hashlib.sha256(content).hexdigest()

    # Schedule background processing
    background_tasks.add_task(process_document, content, file.filename, doc_hash)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "hash": doc_hash,
        "status": "processing"
    }

# Multiple file upload for batch document processing
@app.post("/upload/batch")
async def upload_batch(
    files: list[UploadFile],
    background_tasks: BackgroundTasks
):
    results = []
    for file in files:
        content = await file.read()
        doc_id = str(uuid.uuid4())
        background_tasks.add_task(process_document, content, file.filename, doc_id)
        results.append({"filename": file.filename, "doc_id": doc_id})
    return {"documents": results, "count": len(results)}

# File upload with form data (multipart)
from fastapi import Form

@app.post("/upload/with-metadata")
async def upload_with_metadata(
    file: UploadFile,
    description: Annotated[str, Form()],
    collection_name: Annotated[str, Form()],
    chunk_size: Annotated[int, Form()] = 500,
):
    content = await file.read()
    # Process document: chunk -> embed -> store in vector DB
    chunks = chunk_document(content.decode(), chunk_size)
    embeddings = await generate_embeddings(chunks)
    await store_in_vector_db(collection_name, chunks, embeddings)
    return {"chunks_processed": len(chunks), "collection": collection_name}
```

### 2.4 Batch Inference Endpoints

```python
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

class BatchRequest(BaseModel):
    texts: list[str]
    model: str = "text-embedding-ada-002"

class BatchResponse(BaseModel):
    embeddings: list[list[float]]
    total_tokens: int
    processing_time_ms: float

@app.post("/v1/embeddings/batch", response_model=BatchResponse)
async def batch_embeddings(request: BatchRequest):
    """Process multiple embedding requests concurrently."""
    start = time.perf_counter()

    # Process in batches to avoid rate limits
    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(request.texts), BATCH_SIZE):
        batch = request.texts[i:i + BATCH_SIZE]
        response = await openai_client.embeddings.create(
            model=request.model,
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])

    elapsed = (time.perf_counter() - start) * 1000
    return BatchResponse(
        embeddings=all_embeddings,
        total_tokens=sum(len(t.split()) for t in request.texts),
        processing_time_ms=elapsed,
    )

# Concurrent batch inference with semaphore for rate limiting
@app.post("/v1/inference/batch")
async def batch_inference(requests: list[dict]):
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

    async def process_one(item: dict):
        async with semaphore:
            return await call_llm(item["prompt"])

    results = await asyncio.gather(
        *[process_one(req) for req in requests],
        return_exceptions=True
    )
    return {"results": results}
```

---

## 3. WEBSOCKET PATTERNS FOR AI

### 3.1 Real-Time Chat with LLMs

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json
import uuid

app = FastAPI()

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> list[dict]:
        return self.messages[-20:]  # Keep last 20 messages for context

# Session store (use Redis in production)
sessions: dict[str, ChatSession] = {}

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = ChatSession(session_id)
    session = sessions[session_id]

    try:
        while True:
            # Receive user message
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            # Add to conversation history
            session.add_message("user", user_message)

            # Send acknowledgment
            await websocket.send_json({
                "type": "ack",
                "message_id": str(uuid.uuid4()),
            })

            # Stream LLM response
            full_response = ""
            async for token in stream_llm(session.get_context()):
                full_response += token
                await websocket.send_json({
                    "type": "token",
                    "content": token,
                })

            # Add assistant response to history
            session.add_message("assistant", full_response)

            # Send completion signal
            await websocket.send_json({
                "type": "done",
                "full_content": full_response,
            })

    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
```

### 3.2 Connection Manager for Multi-User AI Chat

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json

class AIConnectionManager:
    def __init__(self):
        # user_id -> set of WebSocket connections (multi-device)
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # session_id -> conversation history
        self.sessions: Dict[str, list] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def send_to_user(self, user_id: str, message: dict):
        """Send to all connections of a specific user."""
        if user_id in self.active_connections:
            disconnected = []
            for ws in self.active_connections[user_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                self.active_connections[user_id].discard(ws)

    async def broadcast(self, message: dict):
        """Broadcast to all connected users."""
        for user_id in self.active_connections:
            await self.send_to_user(user_id, message)

manager = AIConnectionManager()

@app.websocket("/ws/ai-chat/{user_id}")
async def ai_chat_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "chat":
                # Stream AI response back to the specific user
                async for token in stream_llm(data["messages"]):
                    await manager.send_to_user(user_id, {
                        "type": "token",
                        "content": token
                    })
                await manager.send_to_user(user_id, {"type": "done"})

            elif action == "cancel":
                # Cancel ongoing generation
                await manager.send_to_user(user_id, {"type": "cancelled"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
```

### 3.3 Agent Communication over WebSocket

```python
from fastapi import FastAPI, WebSocket
import json

app = FastAPI()

class AgentOrchestrator:
    """Multi-agent system communicating over WebSocket."""

    def __init__(self):
        self.agents = {}

    async def route_message(self, websocket: WebSocket, message: dict):
        agent_type = message.get("agent", "general")
        action = message.get("action")

        if action == "research":
            # Research agent
            async for update in self.research_agent(message["query"]):
                await websocket.send_json({
                    "type": "agent_update",
                    "agent": "research",
                    "content": update
                })

        elif action == "code":
            # Code generation agent
            async for update in self.code_agent(message["spec"]):
                await websocket.send_json({
                    "type": "agent_update",
                    "agent": "code",
                    "content": update
                })

        elif action == "multi_agent":
            # Orchestrate multiple agents
            await self.run_pipeline(websocket, message)

    async def run_pipeline(self, websocket: WebSocket, message: dict):
        """Run a multi-agent pipeline with status updates."""
        # Step 1: Planning agent
        await websocket.send_json({"type": "status", "step": "planning"})
        plan = await self.planning_agent(message["task"])

        # Step 2: Execute each step
        for i, step in enumerate(plan["steps"]):
            await websocket.send_json({
                "type": "status",
                "step": f"executing_step_{i+1}",
                "description": step["description"]
            })
            result = await self.execute_step(step)
            await websocket.send_json({
                "type": "step_result",
                "step": i + 1,
                "result": result
            })

        # Step 3: Synthesis
        await websocket.send_json({"type": "status", "step": "synthesizing"})
        final = await self.synthesis_agent(plan)
        await websocket.send_json({"type": "final_result", "content": final})

orchestrator = AgentOrchestrator()

@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            await orchestrator.route_message(websocket, data)
    except WebSocketDisconnect:
        pass
```

### 3.4 Heartbeat and Reconnection Strategy

```python
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

HEARTBEAT_INTERVAL = 30  # seconds
HEARTBEAT_TIMEOUT = 10   # seconds

@app.websocket("/ws/chat-robust/{session_id}")
async def robust_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Send session info for client reconnection
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "heartbeat_interval": HEARTBEAT_INTERVAL
    })

    last_pong = asyncio.get_event_loop().time()

    async def heartbeat():
        """Server-side heartbeat to detect dead connections."""
        nonlocal last_pong
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await websocket.send_json({"type": "ping"})
                # Check if we received a pong recently
                if asyncio.get_event_loop().time() - last_pong > HEARTBEAT_TIMEOUT + HEARTBEAT_INTERVAL:
                    print(f"Client {session_id} timed out")
                    await websocket.close()
                    return
            except Exception:
                return

    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "pong":
                last_pong = asyncio.get_event_loop().time()
                continue

            if data.get("type") == "reconnect":
                # Client reconnecting -- restore session state
                session = sessions.get(session_id)
                if session:
                    await websocket.send_json({
                        "type": "session_restored",
                        "history": session.messages[-10:]
                    })
                continue

            # Normal message processing
            await process_message(websocket, session_id, data)

    except WebSocketDisconnect:
        heartbeat_task.cancel()
        print(f"Client {session_id} disconnected")
```

**Client-side reconnection (JavaScript):**
```javascript
class RobustWebSocket {
    constructor(url) {
        this.url = url;
        this.maxRetries = 5;
        this.retryCount = 0;
        this.baseDelay = 1000;  // 1 second
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('Connected');
            this.retryCount = 0;
            // Start heartbeat response
            this.startHeartbeat();
        };

        this.ws.onclose = (event) => {
            if (!event.wasClean && this.retryCount < this.maxRetries) {
                // Exponential backoff with jitter
                const delay = this.baseDelay * Math.pow(2, this.retryCount)
                    + Math.random() * 1000;
                this.retryCount++;
                console.log(`Reconnecting in ${delay}ms (attempt ${this.retryCount})`);
                setTimeout(() => this.connect(), delay);
            }
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'ping') {
                this.ws.send(JSON.stringify({ type: 'pong' }));
                return;
            }
            this.onMessage(data);
        };
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'pong' }));
            }
        }, 25000);
    }
}
```

---

## 4. PERFORMANCE OPTIMIZATION

### 4.1 Async/Await Patterns

```python
import asyncio
from typing import Any

# PATTERN 1: Concurrent I/O operations
async def get_ai_response_with_context(query: str):
    """Run multiple async operations concurrently."""
    # These run in PARALLEL -- not sequentially
    embedding, relevant_docs, user_history = await asyncio.gather(
        generate_embedding(query),
        search_vector_db(query),
        get_user_history(user_id),
    )
    # Then use the results
    return await call_llm(query, context=relevant_docs, history=user_history)

# PATTERN 2: Semaphore for rate limiting
class RateLimitedLLMClient:
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def call(self, prompt: str) -> str:
        async with self.semaphore:
            return await self._make_api_call(prompt)

# PATTERN 3: Timeout handling
async def inference_with_timeout(prompt: str, timeout: float = 30.0):
    try:
        result = await asyncio.wait_for(
            call_llm(prompt),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        return {"error": "Inference timed out"}

# PATTERN 4: Task cancellation
@app.post("/generate")
async def generate(request: Request, prompt: str):
    task = asyncio.create_task(call_llm(prompt))
    try:
        # Check if client disconnected while we wait
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                return {"status": "cancelled"}
            await asyncio.sleep(0.1)
        return {"result": task.result()}
    except asyncio.CancelledError:
        return {"status": "cancelled"}
```

### 4.2 Connection Pooling

```python
import httpx
from contextlib import asynccontextmanager

# --- httpx connection pooling ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan -- setup and teardown."""
    # Startup: create connection pool
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30,
        ),
        timeout=httpx.Timeout(30.0, connect=10.0),
    )
    yield
    # Shutdown: close connection pool
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/inference")
async def inference(request: Request, prompt: str):
    client = request.app.state.http_client
    response = await client.post(
        "https://api.openai.com/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": "Bearer sk-..."}
    )
    return response.json()

# --- Database connection pooling with SQLAlchemy ---
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/dbname",
    pool_size=20,           # Number of persistent connections
    max_overflow=10,        # Extra connections allowed beyond pool_size
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True,     # Check connection health before use
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

### 4.3 Caching with Redis

```python
import redis.asyncio as redis
import json
import hashlib
from functools import wraps

# Redis client setup
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# PATTERN 1: Cache LLM responses
async def get_cached_or_compute(prompt: str, model: str = "gpt-4") -> str:
    # Create cache key from prompt hash
    cache_key = f"llm:{model}:{hashlib.sha256(prompt.encode()).hexdigest()}"

    # Try cache first
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Cache miss -- compute
    result = await call_llm(prompt, model)

    # Store in cache with TTL (1 hour)
    await redis_client.setex(cache_key, 3600, json.dumps(result))

    return result

# PATTERN 2: Cache embeddings
async def get_embedding_cached(text: str) -> list[float]:
    cache_key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    embedding = await generate_embedding(text)
    await redis_client.setex(cache_key, 86400, json.dumps(embedding))  # 24h TTL
    return embedding

# PATTERN 3: Decorator-based caching
def cache_response(ttl: int = 3600, prefix: str = "api"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = f"{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"

            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            result = await func(*args, **kwargs)
            await redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@app.get("/embeddings/{text}")
@cache_response(ttl=86400, prefix="embeddings")
async def get_embeddings(text: str):
    return await generate_embedding(text)

# PATTERN 4: Session caching for conversation history
class ConversationCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600 * 24  # 24 hours

    async def get_history(self, session_id: str) -> list[dict]:
        history = await self.redis.lrange(f"conv:{session_id}", 0, -1)
        return [json.loads(msg) for msg in history]

    async def add_message(self, session_id: str, role: str, content: str):
        message = json.dumps({"role": role, "content": content})
        await self.redis.rpush(f"conv:{session_id}", message)
        await self.redis.expire(f"conv:{session_id}", self.ttl)

    async def clear_history(self, session_id: str):
        await self.redis.delete(f"conv:{session_id}")
```

### 4.4 Rate Limiting

```python
import time
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis

redis_client = redis.Redis(host="localhost", port=6379)

# --- Token Bucket Rate Limiter ---
class RateLimiter:
    def __init__(self, redis_client, requests_per_minute: int = 60):
        self.redis = redis_client
        self.rate = requests_per_minute
        self.window = 60  # seconds

    async def is_allowed(self, key: str) -> tuple[bool, dict]:
        now = time.time()
        window_key = f"ratelimit:{key}:{int(now // self.window)}"

        current = await self.redis.incr(window_key)
        if current == 1:
            await self.redis.expire(window_key, self.window)

        remaining = max(0, self.rate - current)
        reset_time = (int(now // self.window) + 1) * self.window

        headers = {
            "X-RateLimit-Limit": str(self.rate),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(reset_time)),
        }

        return current <= self.rate, headers

rate_limiter = RateLimiter(redis_client, requests_per_minute=60)

# As middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract user identifier
        api_key = request.headers.get("X-API-Key", "anonymous")

        allowed, headers = await rate_limiter.is_allowed(api_key)
        if not allowed:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers=headers,
            )

        response = await call_next(request)
        for key, value in headers.items():
            response.headers[key] = value
        return response

app.add_middleware(RateLimitMiddleware)

# --- As a dependency (per-endpoint rate limiting) ---
from fastapi import Depends

async def rate_limit_dependency(request: Request):
    api_key = request.headers.get("X-API-Key", "anonymous")
    allowed, headers = await rate_limiter.is_allowed(api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return headers

@app.post("/v1/chat")
async def chat(
    prompt: str,
    rate_limit_headers: dict = Depends(rate_limit_dependency)
):
    return await call_llm(prompt)

# --- Tiered rate limiting for AI (by plan) ---
TIER_LIMITS = {
    "free": {"requests_per_minute": 10, "tokens_per_day": 10000},
    "pro": {"requests_per_minute": 60, "tokens_per_day": 100000},
    "enterprise": {"requests_per_minute": 1000, "tokens_per_day": 10000000},
}
```

### 4.5 Request Queuing and Load Balancing

```python
import asyncio
from collections import deque

class InferenceQueue:
    """Priority queue for inference requests."""

    def __init__(self, max_concurrent: int = 5):
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.processing = False

    async def enqueue(self, priority: int, request_data: dict) -> dict:
        """Add request to queue and wait for result."""
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((priority, request_data, future))

        if not self.processing:
            asyncio.create_task(self._process_queue())

        return await future

    async def _process_queue(self):
        self.processing = True
        while not self.queue.empty():
            async with self.semaphore:
                priority, data, future = await self.queue.get()
                try:
                    result = await call_llm(data["prompt"])
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
        self.processing = False

inference_queue = InferenceQueue(max_concurrent=5)

@app.post("/inference/queued")
async def queued_inference(prompt: str, priority: int = 5):
    result = await inference_queue.enqueue(priority, {"prompt": prompt})
    return {"result": result}
```

---

## 5. AUTHENTICATION & SECURITY

### 5.1 OAuth 2.0 with JWT (Complete Implementation)

```python
from datetime import datetime, timedelta, timezone
from typing import Annotated
import jwt
from fastapi import Depends, FastAPI, HTTPException, status, Security
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    APIKeyHeader,
    HTTPBearer,
)
from pydantic import BaseModel
from pwdlib import PasswordHash

# --- Configuration ---
SECRET_KEY = "your-secret-key-from-environment"  # Use env var in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# --- Models ---
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    sub: str          # user_id
    scopes: list[str] = []
    tier: str = "free"

class User(BaseModel):
    id: str
    username: str
    email: str
    tier: str = "free"  # free, pro, enterprise
    disabled: bool = False

# --- Password Hashing ---
password_hash = PasswordHash.recommended()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return password_hash.hash(password)

# --- JWT Token Creation ---
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- Dependencies ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(**payload)
    except jwt.InvalidTokenError:
        raise credentials_exception

    user = await get_user_from_db(user_id)
    if user is None or user.disabled:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- Endpoints ---
app = FastAPI()

@app.post("/token", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.id, "tier": user.tier},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(data={"sub": user.id})
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

@app.post("/token/refresh")
async def refresh_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload.get("sub")
        new_access = create_access_token(data={"sub": user_id})
        return {"access_token": new_access, "token_type": "bearer"}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@app.get("/protected/chat")
async def protected_chat(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return {"message": f"Hello {current_user.username}, you are on the {current_user.tier} plan"}
```

### 5.2 API Key Authentication for AI Services

```python
from fastapi import Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> dict:
    """Validate API key and return associated metadata."""
    # Look up API key in database
    key_data = await db.api_keys.find_one({"key": api_key, "active": True})
    if not key_data:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key_data

@app.post("/v1/completions")
async def completions(
    request: ChatRequest,
    api_key_data: dict = Depends(verify_api_key)
):
    # Check tier-based access
    if request.model == "gpt-4" and api_key_data["tier"] == "free":
        raise HTTPException(status_code=403, detail="GPT-4 requires a Pro plan")
    return await process_request(request)
```

### 5.3 Token Counting and Billing

```python
import tiktoken

class TokenCounter:
    def __init__(self):
        self.encoders = {}

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        if model not in self.encoders:
            self.encoders[model] = tiktoken.encoding_for_model(model)
        return len(self.encoders[model].encode(text))

    def count_messages(self, messages: list[dict], model: str = "gpt-4") -> int:
        total = 0
        for message in messages:
            total += 4  # Every message has overhead tokens
            for key, value in message.items():
                total += self.count_tokens(value, model)
        total += 2  # Reply priming
        return total

token_counter = TokenCounter()

# Billing middleware
class BillingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Extract token usage from response headers
        input_tokens = int(response.headers.get("X-Input-Tokens", 0))
        output_tokens = int(response.headers.get("X-Output-Tokens", 0))

        if input_tokens or output_tokens:
            user_id = request.state.user_id
            # Log to billing system
            await record_usage(
                user_id=user_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=request.headers.get("X-Model", "gpt-4"),
                cost=calculate_cost(input_tokens, output_tokens, "gpt-4"),
            )
        return response

# Cost calculation
PRICING = {
    "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
    "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
    "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
}

def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = PRICING.get(model, PRICING["gpt-4"])
    return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
```

---

## 6. DEPLOYMENT

### 6.1 Docker Containerization

**Production Dockerfile:**
```dockerfile
# Multi-stage build for smaller image
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Production stage ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY ./app /app/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run with exec form for graceful shutdown
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Docker Compose for AI stack:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/aiapp
    depends_on:
      redis:
        condition: service_healthy
      db:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    volumes:
      - redis_data:/data

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: aiapp
    healthcheck:
      test: ["CMD-LINE", "pg_isready", "-U", "user"]
      interval: 10s
      timeout: 5s
      retries: 5
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

volumes:
  redis_data:
  postgres_data:
```

### 6.2 Nginx Reverse Proxy Configuration

```nginx
upstream fastapi_backend {
    least_conn;
    server api:8000;
    # For multiple instances:
    # server api_1:8000;
    # server api_2:8000;
}

server {
    listen 80;
    server_name api.myaiapp.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.myaiapp.com;

    ssl_certificate /etc/ssl/certs/fullchain.pem;
    ssl_certificate_key /etc/ssl/private/privkey.pem;

    # SSE / Streaming -- critical for AI
    proxy_buffering off;
    proxy_cache off;

    # Regular API endpoints
    location /api/ {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long inference
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
    }

    # SSE streaming endpoints
    location /api/stream/ {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
        proxy_buffering off;
        proxy_cache off;

        # Longer timeout for streaming
        proxy_read_timeout 600s;
    }

    # WebSocket endpoints
    location /ws/ {
        proxy_pass http://fastapi_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # WebSocket timeout
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }

    # Health check (no auth needed)
    location /health {
        proxy_pass http://fastapi_backend;
    }
}
```

### 6.3 Gunicorn/Uvicorn Workers

```python
# gunicorn.conf.py
import multiprocessing

# Worker configuration
workers = multiprocessing.cpu_count() * 2 + 1  # Rule of thumb
worker_class = "uvicorn.workers.UvicornWorker"   # ASGI worker
bind = "0.0.0.0:8000"

# Timeouts
timeout = 120           # Worker timeout (increase for long inference)
graceful_timeout = 30   # Time to finish current requests on shutdown
keepalive = 5           # Keep connections alive

# Logging
accesslog = "-"         # stdout
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "ai-api"

# Preloading (loads app before forking workers -- saves memory for ML models)
preload_app = True

# Server hooks
def on_starting(server):
    """Called before the master process is initialized."""
    pass

def pre_fork(server, worker):
    """Called before a worker is forked."""
    pass

def post_fork(server, worker):
    """Called after a worker is forked."""
    pass

def on_exit(server):
    """Called before exiting Gunicorn."""
    pass
```

**Run command:**
```bash
# With gunicorn (production)
gunicorn app.main:app -c gunicorn.conf.py

# With uvicorn directly (development/simple production)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# With fastapi CLI
fastapi run app/main.py --port 8000 --workers 4
```

### 6.4 Health Checks and Graceful Shutdown

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import signal

# --- Lifespan for startup/shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === STARTUP ===
    print("Loading ML models...")
    app.state.model = load_model()
    app.state.redis = await redis.Redis.from_url("redis://localhost")
    app.state.db_engine = create_async_engine(DATABASE_URL)

    print("Application ready")
    yield

    # === SHUTDOWN (graceful) ===
    print("Shutting down gracefully...")
    await app.state.redis.close()
    await app.state.db_engine.dispose()
    print("Cleanup complete")

app = FastAPI(lifespan=lifespan)

# --- Health check endpoints ---
@app.get("/health")
async def health_check():
    """Basic liveness check."""
    return {"status": "healthy"}

@app.get("/health/ready")
async def readiness_check():
    """Readiness check -- verify all dependencies are available."""
    checks = {}

    # Check Redis
    try:
        await app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "error"

    # Check Database
    try:
        async with app.state.db_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"

    # Check model loaded
    checks["model"] = "ok" if hasattr(app.state, 'model') and app.state.model else "error"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
    }

@app.get("/health/startup")
async def startup_check():
    """Kubernetes startup probe -- allows slow model loading."""
    if hasattr(app.state, 'model') and app.state.model is not None:
        return {"status": "started"}
    raise HTTPException(status_code=503, detail="Still loading")
```

---

## 7. DATABASE PATTERNS

### 7.1 SQLAlchemy Async with FastAPI

```python
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, DateTime, ForeignKey, JSON
from datetime import datetime

# --- Database Setup ---
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:5432/ai_app"

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

# --- Models ---
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(100), unique=True)
    api_key: Mapped[str] = mapped_column(String(64), unique=True)
    tier: Mapped[str] = mapped_column(String(20), default="free")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(200))
    model: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String(20))  # user, assistant, system
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(default=0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

# --- Dependency ---
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# --- CRUD Operations ---
from sqlalchemy import select
from typing import Annotated
from fastapi import Depends

DBDep = Annotated[AsyncSession, Depends(get_db)]

@app.post("/conversations")
async def create_conversation(user_id: int, title: str, db: DBDep):
    conversation = Conversation(user_id=user_id, title=title, model="gpt-4")
    db.add(conversation)
    await db.flush()
    return {"id": conversation.id, "title": conversation.title}

@app.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: int, db: DBDep):
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in messages]

@app.post("/conversations/{conv_id}/chat")
async def chat_in_conversation(conv_id: int, user_message: str, db: DBDep):
    # Save user message
    user_msg = Message(
        conversation_id=conv_id,
        role="user",
        content=user_message,
        token_count=token_counter.count_tokens(user_message),
    )
    db.add(user_msg)

    # Get conversation history
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
    )
    history = [{"role": m.role, "content": m.content} for m in result.scalars().all()]

    # Call LLM
    response = await call_llm(history)

    # Save assistant message
    assistant_msg = Message(
        conversation_id=conv_id,
        role="assistant",
        content=response,
        token_count=token_counter.count_tokens(response),
    )
    db.add(assistant_msg)

    return {"response": response}
```

### 7.2 MongoDB with Motor

```python
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime

# --- Setup ---
MONGO_URL = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URL)
db = client.ai_application

# --- Collections ---
conversations_collection = db.conversations
messages_collection = db.messages
embeddings_collection = db.embeddings

# --- Dependency ---
async def get_mongo_db():
    return db

# --- Conversation Storage ---
@app.post("/conversations")
async def create_conversation(user_id: str, title: str):
    conversation = {
        "user_id": user_id,
        "title": title,
        "model": "gpt-4",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "message_count": 0,
        "total_tokens": 0,
    }
    result = await conversations_collection.insert_one(conversation)
    return {"id": str(result.inserted_id), "title": title}

@app.post("/conversations/{conv_id}/messages")
async def add_message(conv_id: str, role: str, content: str):
    message = {
        "conversation_id": ObjectId(conv_id),
        "role": role,
        "content": content,
        "token_count": token_counter.count_tokens(content),
        "created_at": datetime.utcnow(),
    }
    await messages_collection.insert_one(message)

    # Update conversation metadata
    await conversations_collection.update_one(
        {"_id": ObjectId(conv_id)},
        {
            "$inc": {"message_count": 1, "total_tokens": message["token_count"]},
            "$set": {"updated_at": datetime.utcnow()},
        }
    )
    return {"status": "ok"}

@app.get("/conversations/{conv_id}/history")
async def get_history(conv_id: str, limit: int = 50):
    cursor = messages_collection.find(
        {"conversation_id": ObjectId(conv_id)}
    ).sort("created_at", 1).limit(limit)

    messages = []
    async for msg in cursor:
        messages.append({
            "id": str(msg["_id"]),
            "role": msg["role"],
            "content": msg["content"],
            "created_at": msg["created_at"].isoformat(),
        })
    return {"messages": messages}

# --- Indexes (run at startup) ---
async def create_indexes():
    await messages_collection.create_index(
        [("conversation_id", 1), ("created_at", 1)]
    )
    await conversations_collection.create_index([("user_id", 1)])
    await embeddings_collection.create_index([("embedding", "2dsphere")])
```

### 7.3 Vector Store Integration

```python
# --- ChromaDB Integration ---
import chromadb

chroma_client = chromadb.HttpClient(host="localhost", port=8000)

class VectorStoreService:
    def __init__(self):
        self.collection = chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(
        self, texts: list[str], metadatas: list[dict], ids: list[str]
    ):
        embeddings = await generate_embeddings(texts)
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    async def search(self, query: str, n_results: int = 5) -> list[dict]:
        query_embedding = await generate_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results

# --- Pgvector (PostgreSQL) Integration ---
# Uses SQLAlchemy with pgvector extension

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    embedding = Column(Vector(1536))  # OpenAI embedding dimension
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=True)
    collection: Mapped[str] = mapped_column(String(100))

@app.post("/rag/search")
async def rag_search(query: str, collection: str, top_k: int = 5, db: DBDep = None):
    """RAG retrieval endpoint."""
    query_embedding = await generate_embedding(query)

    # Cosine similarity search using pgvector
    result = await db.execute(
        select(DocumentEmbedding)
        .where(DocumentEmbedding.collection == collection)
        .order_by(DocumentEmbedding.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    documents = result.scalars().all()

    # Build context for LLM
    context = "\n\n".join([doc.content for doc in documents])
    response = await call_llm_with_context(query, context)

    return {
        "answer": response,
        "sources": [{"content": doc.content, "metadata": doc.metadata_} for doc in documents],
    }
```

---

## 8. DJANGO VS FLASK VS FASTAPI COMPARISON FOR AI

### 8.1 Feature Comparison Table

| Feature | FastAPI | Flask | Django |
|---|---|---|---|
| **Async Support** | Native (built on ASGI) | Limited (Flask 2.0+ with async views) | Partial (Django 4.1+ async views) |
| **Type Hints** | Core feature, auto-validation | Optional, no auto-validation | Optional, no auto-validation |
| **Auto API Docs** | Built-in (Swagger + ReDoc) | Requires flask-restx/apispec | Requires DRF + drf-spectacular |
| **Performance** | High (on par with Node.js) | Moderate | Moderate |
| **WebSocket** | Built-in | Requires flask-socketio | Requires Django Channels |
| **Streaming** | Native StreamingResponse | Possible but awkward | Possible with StreamingHttpResponse |
| **Dependency Injection** | Built-in, powerful | No built-in DI | No built-in DI (has middleware) |
| **ORM** | Any (SQLAlchemy, Tortoise) | Any (SQLAlchemy common) | Built-in Django ORM |
| **Admin Panel** | No built-in | No built-in | Built-in admin |
| **Learning Curve** | Low-Medium | Low | High |
| **Community/Ecosystem** | Growing rapidly | Mature | Very mature |
| **Best For AI** | API-first AI services | Simple ML APIs | Full-stack AI web apps |

### 8.2 When to Choose Each Framework

**Choose FastAPI when:**
- Building pure AI/ML API services
- Streaming LLM responses is required
- WebSocket real-time AI chat is needed
- High concurrency with async I/O-bound workloads
- You want auto-generated API documentation
- Type safety and validation are important
- Microservices architecture

**Choose Flask when:**
- Simple ML model serving (single endpoint)
- Quick prototyping / proof of concept
- Team already knows Flask well
- Minimal framework overhead needed
- Simple synchronous ML inference

**Choose Django when:**
- Full-stack web application with AI features
- Need admin panel for managing AI models/data
- Complex data models with relationships (Django ORM)
- Content management alongside AI features
- Authentication/authorization out of the box
- Team building a traditional web app that incorporates AI

### 8.3 Code Comparison -- Streaming LLM Response

**FastAPI (natural fit):**
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def stream_chat(prompt: str):
    async def generate():
        async for token in call_llm_streaming(prompt):
            yield f"data: {token}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Flask (requires workarounds):**
```python
from flask import Flask, Response, stream_with_context

app = Flask(__name__)

@app.route("/chat/stream", methods=["POST"])
def stream_chat():
    prompt = request.json["prompt"]

    def generate():
        for token in call_llm_streaming_sync(prompt):  # Sync only!
            yield f"data: {token}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream"
    )
# Note: True async streaming in Flask is complex and not native
```

**Django (requires extra setup):**
```python
from django.http import StreamingHttpResponse
from django.views import View

class ChatStreamView(View):
    def post(self, request):
        import json
        body = json.loads(request.body)
        prompt = body["prompt"]

        def generate():
            for token in call_llm_streaming_sync(prompt):
                yield f"data: {token}\n\n"

        response = StreamingHttpResponse(
            generate(),
            content_type="text/event-stream"
        )
        response["Cache-Control"] = "no-cache"
        return response
# Note: Django async views still have limitations with streaming
```

### 8.4 WebSocket Comparison

**FastAPI:**
```python
@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Echo: {data}")
```

**Flask (requires flask-socketio):**
```python
from flask_socketio import SocketIO, emit

socketio = SocketIO(app)

@socketio.on("message")
def handle_message(data):
    emit("response", {"data": f"Echo: {data}"})
```

**Django (requires Django Channels + ASGI):**
```python
# consumers.py
from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def receive(self, text_data=None):
        await self.send(text_data=json.dumps({
            "message": f"Echo: {text_data}"
        }))
```

---

## 9. COMMON INTERVIEW QUESTIONS WITH CODE EXAMPLES

### Q1: What is FastAPI and why is it preferred for AI applications?

**Answer:** FastAPI is a modern, high-performance Python web framework built on Starlette (ASGI) and Pydantic. It is preferred for AI applications because:

1. **Native async support** -- essential for non-blocking I/O when calling LLM APIs
2. **StreamingResponse** -- enables token-by-token streaming from LLMs
3. **WebSocket support** -- real-time bidirectional AI chat
4. **Pydantic validation** -- strict type checking for AI API contracts
5. **Dependency injection** -- clean model loading, auth, DB management
6. **Auto OpenAPI docs** -- interactive testing of AI endpoints
7. **Performance** -- one of the fastest Python frameworks

---

### Q2: Explain the difference between `def` and `async def` in FastAPI. When would you use each for AI workloads?

**Answer:**

```python
# async def -- runs on the event loop (non-blocking)
# USE FOR: I/O-bound operations (API calls to LLM providers, DB queries, file I/O)
@app.post("/llm-call")
async def call_llm_endpoint(prompt: str):
    # This does NOT block the event loop
    result = await openai_client.chat.completions.create(...)
    return result

# def -- runs in a thread pool (blocking but isolated)
# USE FOR: CPU-bound operations (local model inference, data processing)
@app.post("/local-inference")
def local_inference(text: str):
    # This runs in a separate thread, so it does not block other requests
    result = model.predict(text)  # CPU-intensive
    return {"result": result}
```

**Critical rule:** NEVER use blocking operations (like `requests.get()` or `time.sleep()`) inside `async def` -- it will block the entire event loop and all other requests.

---

### Q3: How would you implement streaming LLM responses in FastAPI?

**Answer:**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json

app = FastAPI()
client = AsyncOpenAI()

async def stream_tokens(messages: list[dict]):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True,
    )
    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/stream")
async def chat_stream(messages: list[dict]):
    return StreamingResponse(
        stream_tokens(messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Prevents Nginx from buffering
        }
    )
```

Key points:
- Use `async generator` with `yield`
- SSE format: `data: ...\n\n`
- Set `media_type="text/event-stream"`
- Disable Nginx buffering with `X-Accel-Buffering: no`
- Send `[DONE]` signal at the end

---

### Q4: How do you handle WebSocket authentication in FastAPI?

**Answer:**

```python
from fastapi import WebSocket, WebSocketException, Depends, Query, status
import jwt

async def authenticate_websocket(
    websocket: WebSocket,
    token: str = Query(None),
):
    """WebSocket auth via query parameter (cannot use headers in browser WebSocket)."""
    if token is None:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

@app.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    user_data: dict = Depends(authenticate_websocket),
):
    await websocket.accept()
    user_id = user_data["sub"]
    # Now we have an authenticated WebSocket connection
    # ...
```

Note: Browser WebSocket API does not support custom headers, so authentication is typically done via:
1. Query parameter (`?token=xxx`)
2. First message after connection
3. Cookie-based auth

---

### Q5: How would you implement a RAG (Retrieval Augmented Generation) endpoint?

**Answer:**

```python
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

app = FastAPI()

class RAGQuery(BaseModel):
    question: str
    collection: str = "default"
    top_k: int = 5

class RAGResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float

# Step 1: Ingest documents
@app.post("/rag/ingest")
async def ingest_document(file: UploadFile, collection: str = "default"):
    content = await file.read()
    text = extract_text(content, file.content_type)

    # Chunk the document
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    # Generate embeddings
    embeddings = await generate_embeddings(chunks)

    # Store in vector DB
    await vector_store.add(
        collection=collection,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": file.filename, "chunk_idx": i} for i in range(len(chunks))],
    )
    return {"chunks_stored": len(chunks)}

# Step 2: Query with RAG
@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(query: RAGQuery):
    # Retrieve relevant documents
    results = await vector_store.search(
        query=query.question,
        collection=query.collection,
        top_k=query.top_k,
    )

    # Build context
    context = "\n\n---\n\n".join([doc["content"] for doc in results])

    # Generate answer with context
    messages = [
        {"role": "system", "content": f"Answer based on this context:\n{context}"},
        {"role": "user", "content": query.question},
    ]
    answer = await call_llm(messages)

    return RAGResponse(
        answer=answer,
        sources=results,
        confidence=results[0]["score"] if results else 0.0,
    )
```

---

### Q6: How do you handle graceful shutdown in FastAPI?

**Answer:**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    app.state.model = load_ml_model()
    app.state.redis = await aioredis.from_url("redis://localhost")
    app.state.http_client = httpx.AsyncClient()

    print("Application started")
    yield  # Application runs here

    # SHUTDOWN (guaranteed to run)
    print("Shutting down...")
    await app.state.redis.close()
    await app.state.http_client.aclose()
    # Model cleanup
    del app.state.model
    print("Shutdown complete")

app = FastAPI(lifespan=lifespan)
```

Key points:
- Use `lifespan` context manager (replaces deprecated `@app.on_event("startup")`)
- Everything before `yield` runs at startup
- Everything after `yield` runs at shutdown
- Docker: use exec form `CMD ["uvicorn", ...]` (not shell form) for proper signal handling
- Uvicorn handles SIGTERM/SIGINT gracefully, allowing in-flight requests to complete

---

### Q7: How do you implement rate limiting per user for an AI API?

**Answer:**

```python
import redis.asyncio as redis
import time
from fastapi import Depends, HTTPException

redis_client = redis.Redis(host="localhost", port=6379)

# Sliding window rate limiter
async def check_rate_limit(
    user_id: str,
    tier: str,
    endpoint: str
):
    limits = {
        "free": {"rpm": 10, "rpd": 100, "tpd": 10000},
        "pro": {"rpm": 60, "rpd": 1000, "tpd": 100000},
        "enterprise": {"rpm": 600, "rpd": 100000, "tpd": 10000000},
    }

    user_limits = limits.get(tier, limits["free"])

    # Check requests per minute
    minute_key = f"rate:{user_id}:{endpoint}:rpm:{int(time.time() // 60)}"
    count = await redis_client.incr(minute_key)
    await redis_client.expire(minute_key, 60)

    if count > user_limits["rpm"]:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )

    # Check daily token limit
    day_key = f"tokens:{user_id}:{time.strftime('%Y-%m-%d')}"
    tokens_used = int(await redis_client.get(day_key) or 0)
    if tokens_used > user_limits["tpd"]:
        raise HTTPException(
            status_code=429,
            detail="Daily token limit exceeded"
        )
```

---

### Q8: Explain FastAPI's dependency injection and how you would use it for ML model management.

**Answer:**

```python
from functools import lru_cache
from typing import Annotated, Protocol

# 1. Define a protocol for model services (for testing/swapping)
class ModelServiceProtocol(Protocol):
    async def predict(self, text: str) -> dict: ...
    async def embed(self, text: str) -> list[float]: ...

# 2. Concrete implementation
class OpenAIModelService:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def predict(self, text: str) -> dict:
        response = await self.client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": text}]
        )
        return {"content": response.choices[0].message.content}

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

# 3. Factory with caching (singleton pattern)
@lru_cache()
def get_model_service() -> OpenAIModelService:
    return OpenAIModelService()

# 4. Type alias for clean injection
ModelDep = Annotated[OpenAIModelService, Depends(get_model_service)]

# 5. Use in endpoints
@app.post("/predict")
async def predict(text: str, model: ModelDep):
    return await model.predict(text)

@app.post("/embed")
async def embed(text: str, model: ModelDep):
    return {"embedding": await model.embed(text)}

# 6. Override for testing
def get_mock_model_service():
    return MockModelService()

app.dependency_overrides[get_model_service] = get_mock_model_service
```

---

### Q9: What is the difference between SSE and WebSocket? When would you use each for AI?

**Answer:**

| Feature | SSE (Server-Sent Events) | WebSocket |
|---|---|---|
| Direction | Server -> Client (unidirectional) | Bidirectional |
| Protocol | HTTP/1.1 | ws:// or wss:// |
| Reconnection | Built-in auto-reconnect | Manual implementation needed |
| Binary data | Text only | Text + Binary |
| Browser support | Excellent (EventSource API) | Excellent |
| HTTP/2 multiplexing | Yes | No |
| Load balancer friendly | Yes (standard HTTP) | Needs special config |
| Complexity | Simple | More complex |

**Use SSE for:**
- Streaming LLM responses (most common AI use case)
- One-way notifications (job completion, progress updates)
- When you need HTTP middleware compatibility (auth, logging, caching)

**Use WebSocket for:**
- Real-time interactive AI chat (user can cancel, type indicators)
- Multi-agent communication requiring bidirectional messages
- Collaborative AI workspaces (multiple users)
- When you need to send binary data (audio, images)

---

### Q10: How do you test FastAPI AI endpoints?

**Answer:**

```python
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app

# --- Sync testing with TestClient ---
def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# --- Async testing ---
@pytest.mark.anyio
async def test_chat_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False}
        )
        assert response.status_code == 200
        assert "content" in response.json()

# --- Mocking LLM calls ---
@pytest.mark.anyio
async def test_chat_with_mock():
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(message=AsyncMock(content="Mocked response"))
    ]

    with patch("app.main.openai_client.chat.completions.create", return_value=mock_response):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "Test"}], "stream": False}
            )
            assert response.status_code == 200

# --- Testing streaming responses ---
@pytest.mark.anyio
async def test_streaming():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        async with client.stream("POST", "/v1/chat/stream", json={"prompt": "Hi"}) as response:
            assert response.status_code == 200
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line)
            assert len(chunks) > 0

# --- Testing WebSocket ---
def test_websocket():
    client = TestClient(app)
    with client.websocket_connect("/ws/chat/test-session") as websocket:
        websocket.send_json({"message": "Hello"})
        data = websocket.receive_json()
        assert data["type"] == "ack"

# --- Dependency override for testing ---
def test_with_mock_model():
    mock_service = MockModelService()

    app.dependency_overrides[get_model_service] = lambda: mock_service

    client = TestClient(app)
    response = client.post("/predict", params={"text": "test"})
    assert response.status_code == 200

    app.dependency_overrides.clear()
```

---

### Q11: How do you handle CORS for AI web applications?

**Answer:**

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-RateLimit-Remaining"],
)

# Production -- be specific
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myaiapp.com", "https://www.myaiapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600,  # Cache preflight requests for 1 hour
)
```

---

### Q12: How would you structure a production FastAPI AI application?

**Answer:**

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app creation, lifespan, middleware
│   ├── config.py             # Settings with Pydantic BaseSettings
│   ├── dependencies.py       # Shared dependencies (DB, model, auth)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py     # API router aggregation
│   │   │   ├── chat.py       # Chat endpoints
│   │   │   ├── embeddings.py # Embedding endpoints
│   │   │   ├── documents.py  # Document upload/RAG
│   │   │   └── websocket.py  # WebSocket endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py       # JWT, API keys, auth
│   │   ├── rate_limiter.py   # Rate limiting logic
│   │   └── middleware.py      # Custom middleware
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py       # SQLAlchemy models
│   │   └── schemas.py        # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py    # LLM API client
│   │   ├── embedding_service.py
│   │   ├── vector_store.py   # Vector DB operations
│   │   └── cache_service.py  # Redis caching
│   └── utils/
│       ├── __init__.py
│       └── token_counter.py
├── tests/
│   ├── conftest.py
│   ├── test_chat.py
│   ├── test_websocket.py
│   └── test_rag.py
├── alembic/                  # Database migrations
├── Dockerfile
├── docker-compose.yml
├── gunicorn.conf.py
├── requirements.txt
└── .env
```

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "AI API"
    debug: bool = False
    openai_api_key: str
    database_url: str
    redis_url: str = "redis://localhost:6379"
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"

# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import Settings
from app.api.v1.router import api_router

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.settings = settings
    yield
    # Shutdown

app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
)
app.include_router(api_router, prefix="/api/v1")
```

---

### Q13: What is Pydantic and why is it critical for AI APIs?

**Answer:** Pydantic provides data validation using Python type hints. It is critical for AI APIs because:

```python
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., min_length=1)
    model: str = Field("gpt-4", pattern="^(gpt-4|gpt-3.5-turbo|claude-3)$")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=128000)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        valid_roles = {"system", "user", "assistant"}
        for msg in v:
            if msg.get("role") not in valid_roles:
                raise ValueError(f"Invalid role: {msg.get('role')}")
            if not msg.get("content"):
                raise ValueError("Message content cannot be empty")
        return v

# Automatic validation -- invalid requests return 422 with detailed errors
# No manual validation code needed in your endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    return await process_chat(request)
```

Benefits:
1. **Automatic request validation** -- invalid data returns 422 before your code runs
2. **Serialization** -- `model_dump()` and `model_dump_json()` for responses
3. **OpenAPI schema generation** -- docs auto-generated from Pydantic models
4. **Type safety** -- IDE autocompletion and error detection
5. **Settings management** -- `BaseSettings` for environment variable parsing

---

### Q14: How do you handle errors in AI API endpoints?

**Answer:**

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Custom exception classes
class AIServiceError(Exception):
    def __init__(self, message: str, status_code: int = 500, error_type: str = "ai_error"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type

class RateLimitError(AIServiceError):
    def __init__(self, retry_after: int = 60):
        super().__init__("Rate limit exceeded", 429, "rate_limit")
        self.retry_after = retry_after

class TokenLimitError(AIServiceError):
    def __init__(self, max_tokens: int):
        super().__init__(f"Input exceeds maximum token limit of {max_tokens}", 400, "token_limit")

# Global exception handlers
@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request: Request, exc: AIServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the error (never expose internal errors to client)
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "An internal error occurred",
            }
        }
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    # Token validation
    token_count = count_tokens(request.messages)
    if token_count > 128000:
        raise TokenLimitError(128000)

    try:
        return await call_llm(request)
    except openai.RateLimitError:
        raise RateLimitError(retry_after=60)
    except openai.APIError as e:
        raise AIServiceError(f"LLM provider error: {str(e)}", 502, "upstream_error")
```

---

### Q15: How do you implement a complete AI chat application with conversation persistence?

**Answer:** This is a comprehensive question that tests multiple concepts:

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
import json, uuid

app = FastAPI()

# --- Pydantic Models ---
class CreateConversation(BaseModel):
    title: str = "New Chat"
    model: str = "gpt-4"
    system_prompt: str = "You are a helpful assistant."

class SendMessage(BaseModel):
    content: str
    stream: bool = True

# --- REST Endpoints ---
@app.post("/conversations")
async def create_conversation(req: CreateConversation, db: DBDep, user: UserDep):
    conv = Conversation(
        user_id=user.id, title=req.title, model=req.model
    )
    db.add(conv)
    await db.flush()

    # Add system message
    system_msg = Message(
        conversation_id=conv.id, role="system", content=req.system_prompt
    )
    db.add(system_msg)
    return {"conversation_id": conv.id}

@app.post("/conversations/{conv_id}/messages")
async def send_message(
    conv_id: int,
    req: SendMessage,
    db: DBDep,
    user: UserDep,
):
    # Verify ownership
    conv = await db.get(Conversation, conv_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")

    # Save user message
    user_msg = Message(conversation_id=conv_id, role="user", content=req.content)
    db.add(user_msg)
    await db.flush()

    # Get history
    messages = await get_conversation_messages(db, conv_id)

    if req.stream:
        return StreamingResponse(
            stream_and_save(messages, conv_id, db),
            media_type="text/event-stream",
        )
    else:
        response = await call_llm(messages)
        assistant_msg = Message(
            conversation_id=conv_id, role="assistant", content=response
        )
        db.add(assistant_msg)
        return {"content": response}

async def stream_and_save(messages, conv_id, db):
    """Stream tokens and save the complete response."""
    full_response = ""
    async for token in call_llm_streaming(messages):
        full_response += token
        yield f"data: {json.dumps({'content': token})}\n\n"

    # Save complete response to DB
    assistant_msg = Message(
        conversation_id=conv_id, role="assistant", content=full_response
    )
    db.add(assistant_msg)
    await db.commit()
    yield f"data: {json.dumps({'done': True})}\n\n"

# --- WebSocket alternative ---
@app.websocket("/ws/conversations/{conv_id}")
async def ws_conversation(
    websocket: WebSocket,
    conv_id: int,
    user: dict = Depends(authenticate_websocket),
):
    await websocket.accept()
    db = AsyncSessionLocal()

    try:
        while True:
            data = await websocket.receive_json()

            if data["action"] == "send":
                # Save user message
                user_msg = Message(
                    conversation_id=conv_id, role="user", content=data["content"]
                )
                db.add(user_msg)
                await db.flush()

                # Get history and stream response
                messages = await get_conversation_messages(db, conv_id)
                full_response = ""

                async for token in call_llm_streaming(messages):
                    full_response += token
                    await websocket.send_json({"type": "token", "content": token})

                # Save assistant message
                db.add(Message(
                    conversation_id=conv_id, role="assistant", content=full_response
                ))
                await db.commit()
                await websocket.send_json({"type": "done"})

            elif data["action"] == "history":
                messages = await get_conversation_messages(db, conv_id)
                await websocket.send_json({"type": "history", "messages": messages})

    except WebSocketDisconnect:
        pass
    finally:
        await db.close()
```

---

### Q16: How does FastAPI handle concurrency and what are the threading/async models?

**Answer:**

FastAPI runs on **Uvicorn**, which is an ASGI server based on **uvloop** (a fast implementation of asyncio's event loop).

```
┌─────────────────────────────────────────┐
│  Uvicorn ASGI Server                    │
│                                         │
│  ┌──────────────────────────────┐       │
│  │  Event Loop (uvloop)         │       │
│  │                              │       │
│  │  async def endpoints ────────┼──→ Run directly on event loop  │
│  │  def endpoints ──────────────┼──→ Run in ThreadPoolExecutor   │
│  │                              │       │
│  └──────────────────────────────┘       │
│                                         │
│  ┌──────────────────────────────┐       │
│  │  ThreadPoolExecutor          │       │
│  │  (default 40 threads)        │       │
│  │                              │       │
│  │  Handles sync `def` endpoints│       │
│  └──────────────────────────────┘       │
└─────────────────────────────────────────┘
```

- **`async def`**: Runs directly on the event loop. Must use `await` for I/O.
- **`def`**: FastAPI automatically runs it in a thread pool. Safe for blocking operations but limited by thread count.
- **Workers**: Gunicorn can spawn multiple Uvicorn workers (separate processes) for CPU parallelism.
- **For local ML inference**: Use `ProcessPoolExecutor` to avoid GIL limitations:

```python
from concurrent.futures import ProcessPoolExecutor
import asyncio

process_pool = ProcessPoolExecutor(max_workers=4)

@app.post("/local-model")
async def local_model_inference(text: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(process_pool, model_predict, text)
    return {"result": result}
```

---

### Q17: What is the lifespan context manager and how does it replace startup/shutdown events?

**Answer:**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

# OLD WAY (deprecated)
# @app.on_event("startup")
# async def startup():
#     pass
#
# @app.on_event("shutdown")
# async def shutdown():
#     pass

# NEW WAY -- lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything BEFORE yield runs at STARTUP
    print("Loading resources...")
    app.state.model = load_model()
    app.state.redis = await init_redis()

    yield  # App runs between startup and shutdown

    # Everything AFTER yield runs at SHUTDOWN
    print("Cleaning up...")
    await app.state.redis.close()
    del app.state.model

app = FastAPI(lifespan=lifespan)
```

Advantages of lifespan over events:
1. **Single function** instead of separate startup/shutdown handlers
2. **Shared state** -- variables declared before `yield` are accessible after
3. **Guaranteed cleanup** -- `finally` block ensures cleanup even on crash
4. **Type-safe** -- app state is accessible through `app.state`

---

### Q18: How do you implement API versioning in FastAPI?

**Answer:**

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Version 1 router
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.post("/chat")
async def chat_v1(prompt: str):
    return {"version": "v1", "response": await call_llm_v1(prompt)}

# Version 2 router with breaking changes
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.post("/chat")
async def chat_v2(request: ChatRequestV2):
    # V2 uses structured messages instead of a single prompt
    return {"version": "v2", "response": await call_llm_v2(request)}

app.include_router(v1_router)
app.include_router(v2_router)
```

---

### Q19: How would you implement retry logic for LLM API calls?

**Answer:**

```python
import asyncio
from typing import TypeVar, Callable
import random

T = TypeVar("T")

async def retry_with_exponential_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (Exception,),
    **kwargs,
) -> T:
    """Retry async function with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            if attempt == max_retries:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter

            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.1f}s")
            await asyncio.sleep(total_delay)

# Usage with OpenAI
import openai

async def call_llm_with_retry(messages: list[dict]) -> str:
    return await retry_with_exponential_backoff(
        openai_client.chat.completions.create,
        model="gpt-4",
        messages=messages,
        max_retries=3,
        retryable_exceptions=(
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ),
    )
```

---

### Q20: Explain the difference between `HTTPException` and `WebSocketException` in FastAPI.

**Answer:**

```python
from fastapi import HTTPException, WebSocketException, status

# HTTPException -- for REST endpoints
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",             # Can be string or dict
            headers={"X-Error": "not_found"},    # Optional custom headers
        )
    return items[item_id]

# WebSocketException -- for WebSocket endpoints
@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    if room_id not in valid_rooms:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,  # WebSocket close code
            reason="Invalid room"                   # Close reason string
        )
    await websocket.accept()
    # ...
```

Key differences:
- `HTTPException` uses HTTP status codes (400, 401, 403, 404, 429, 500)
- `WebSocketException` uses WebSocket close codes (1000-4999)
- `HTTPException` returns JSON error body
- `WebSocketException` closes the connection with a code and reason
- You CANNOT use `HTTPException` inside a WebSocket handler

Common WebSocket close codes:
- 1000: Normal closure
- 1001: Going away
- 1003: Unsupported data
- 1008: Policy violation (use for auth failures)
- 1011: Internal error

---

## ADDITIONAL KEY TOPICS FOR INTERVIEWS

### Pydantic V2 Performance (relevant for 2025-2026)
- Pydantic V2 (used in FastAPI 0.100+) is written in Rust (pydantic-core)
- 5-50x faster than Pydantic V1
- Uses `model_dump()` instead of `.dict()` and `model_dump_json()` instead of `.json()`

### FastAPI Lifecycle Management
```python
# Startup order:
# 1. Lifespan startup code runs
# 2. Middleware is initialized
# 3. Dependencies are resolved per-request
# 4. Path operations handle requests

# Shutdown order:
# 1. Stop accepting new connections
# 2. Wait for in-flight requests (graceful_timeout)
# 3. Cancel remaining tasks
# 4. Lifespan shutdown code runs
# 5. Process exits
```

### Key Libraries in the FastAPI AI Ecosystem (2025-2026)
- **httpx**: Async HTTP client (replaces requests)
- **openai**: Official OpenAI Python SDK (async support)
- **anthropic**: Anthropic Python SDK (async support)
- **langchain/langgraph**: LLM orchestration frameworks
- **chromadb/pgvector/pinecone**: Vector stores
- **redis.asyncio**: Async Redis client
- **sqlalchemy[asyncio]**: Async ORM
- **motor**: Async MongoDB driver
- **celery**: Distributed task queue
- **pydantic-settings**: Environment variable management
- **python-jose/PyJWT**: JWT handling
- **tiktoken**: OpenAI token counting
- **uvicorn/gunicorn**: ASGI server / process manager

---

*This guide covers the major topics expected in FastAPI AI backend development interviews for 2025-2026. Each section includes production-ready code examples that can be discussed and adapted during interviews.*
