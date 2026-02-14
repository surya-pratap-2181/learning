# Section 2: Concurrency in Python -- Deep Dive for AI Engineers

---

## 2.1 THE GIL (GLOBAL INTERPRETER LOCK)

### Q: What is the GIL and why does Python have it?

**Answer:**
The Global Interpreter Lock (GIL) is a mutex in CPython that allows only ONE thread to execute Python bytecode at any given time. Even on multi-core machines, Python threads cannot achieve true CPU parallelism.

**Why it exists:**
- CPython's memory management (reference counting) is not thread-safe.
- The GIL simplifies the C extension API and prevents race conditions on Python object reference counts.
- Without it, every operation on a Python object would need fine-grained locks, adding massive overhead.

**Impact on AI workloads:**
```
Workload Type        | GIL Impact | Solution
---------------------|------------|----------------------------------
API calls to LLMs    | LOW        | asyncio or threading (I/O-bound)
Data preprocessing   | HIGH       | multiprocessing or C extensions
Model inference      | LOW*       | NumPy/PyTorch release GIL in C code
Training loops       | LOW*       | Framework handles parallelism
Image processing     | HIGH       | multiprocessing or OpenCV (C)
Text parsing (pure)  | HIGH       | multiprocessing
```

*NumPy, PyTorch, TensorFlow release the GIL during C/CUDA operations.

### Q: What about the "no-GIL" Python (PEP 703, Python 3.13+)?

**Answer:**
Python 3.13 introduced an **experimental free-threaded mode** (compiled with `--disable-gil`):
- True multi-threaded parallelism for CPU-bound Python code
- Experimental in 3.13, maturing in 3.14+
- Performance trade-off: single-threaded code may be slightly slower due to per-object locking
- Not all C extensions are compatible yet
- To use: `python3.13t` (free-threaded build) or `PYTHON_GIL=0` environment variable

```bash
# Check if free-threaded mode is available
python -c "import sys; print(sys._is_gil_enabled())"

# Run with GIL disabled
PYTHON_GIL=0 python my_script.py
```

**Interview perspective:** Know that this exists, know the trade-offs, but acknowledge that for production AI systems in 2025, the standard approach is still asyncio (I/O-bound) + multiprocessing (CPU-bound).

---

## 2.2 ASYNCIO IN DEPTH

### Q: Explain the asyncio event loop architecture.

**Answer:**
```
+-----------------------------------------------------------+
|                     EVENT LOOP                             |
|                                                            |
|  +--------+   +---------+   +----------+   +-----------+  |
|  | Ready  |   | Waiting |   | Scheduled|   | Callbacks |  |
|  | Queue  |   | (I/O)   |   | (timers) |   | Queue     |  |
|  +--------+   +---------+   +----------+   +-----------+  |
|       |            |              |               |        |
|       +------+-----+--------------+------+--------+        |
|              |                           |                 |
|         [ Selector ]              [ Execute next ]         |
|         (epoll/kqueue)             ready callback          |
+-----------------------------------------------------------+
```

**Event loop cycle (simplified):**
1. Check for ready callbacks and run them
2. Poll for I/O events (with timeout from scheduled callbacks)
3. Process completed I/O events, schedule their callbacks
4. Run scheduled callbacks whose time has arrived
5. Repeat

```python
import asyncio

# --- Core asyncio patterns for AI systems ---

# Pattern 1: Semaphore for rate limiting LLM API calls
async def rate_limited_llm_calls(prompts: list[str], max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def call_with_limit(prompt: str) -> str:
        async with semaphore:  # At most max_concurrent calls at once
            return await llm_client.complete(prompt)

    return await asyncio.gather(*[call_with_limit(p) for p in prompts])

# Pattern 2: Producer-Consumer with Queue
async def embedding_pipeline(texts: list[str], batch_size: int = 32):
    queue: asyncio.Queue[list[str]] = asyncio.Queue(maxsize=10)

    async def producer():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            await queue.put(batch)  # Blocks if queue full (backpressure)
        await queue.put(None)  # Sentinel

    async def consumer():
        results = []
        while True:
            batch = await queue.get()
            if batch is None:
                break
            embeddings = await embedding_model.embed(batch)
            results.extend(embeddings)
            queue.task_done()
        return results

    # Run producer and consumer concurrently
    _, results = await asyncio.gather(producer(), consumer())
    return results

# Pattern 3: Streaming with async generators
async def stream_and_process(prompt: str):
    full_response = []
    async for chunk in llm_client.stream(prompt):
        full_response.append(chunk)
        # Process each chunk in real-time (e.g., display to user)
        yield chunk
    # After streaming completes, log the full response
    await log_response("".join(full_response))

# Pattern 4: Timeout and fallback
async def llm_with_fallback(prompt: str) -> str:
    try:
        async with asyncio.timeout(10):
            return await primary_llm.complete(prompt)
    except (asyncio.TimeoutError, APIError):
        # Fall back to faster, cheaper model
        return await fallback_llm.complete(prompt)

# Pattern 5: Event for coordination
async def wait_for_model_ready():
    model_loaded = asyncio.Event()

    async def load_model():
        model = await download_and_load_model()
        model_loaded.set()  # Signal that model is ready
        return model

    async def process_requests():
        await model_loaded.wait()  # Block until model is loaded
        # Now process requests...

    model_task = asyncio.create_task(load_model())
    process_task = asyncio.create_task(process_requests())
    await asyncio.gather(model_task, process_task)
```

### Q: How do you run blocking (synchronous) code within an async application?

**Answer:**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Use run_in_executor for blocking I/O or CPU-bound work

# Option 1: ThreadPoolExecutor (for blocking I/O)
async def call_sync_api():
    loop = asyncio.get_event_loop()
    # Runs in a thread, doesn't block the event loop
    result = await loop.run_in_executor(
        None,  # Default ThreadPoolExecutor
        requests.get,  # Blocking function
        "https://api.example.com/data"  # Arguments
    )
    return result

# Option 2: ProcessPoolExecutor (for CPU-bound work)
process_pool = ProcessPoolExecutor(max_workers=4)

async def compute_embeddings_cpu(texts: list[str]):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        process_pool,
        cpu_intensive_embedding_function,
        texts
    )
    return result

# asyncio.to_thread (Python 3.9+) -- simpler syntax
async def simpler_blocking_call():
    result = await asyncio.to_thread(requests.get, "https://api.example.com/data")
    return result
```

---

## 2.3 THREADING

### Q: When should you use threading in AI applications?

**Answer:**
Use threading for **I/O-bound** operations where you cannot or do not want to use asyncio:
- Legacy synchronous codebases
- Database operations with sync drivers
- File I/O operations
- When mixing with libraries that are not async-compatible

```python
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Thread pool for parallel LLM calls (sync API) ---
def parallel_llm_calls(prompts: list[str], max_workers: int = 5) -> list[str]:
    results = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(call_llm_sync, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"Error: {e}"

    return results

# --- Thread-safe shared state ---
class ThreadSafeTokenCounter:
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def add(self, tokens: int):
        with self._lock:
            self._count += tokens

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

# --- Threading pitfalls ---
# 1. Race conditions (use locks, queues, or atomic operations)
# 2. Deadlocks (always acquire locks in consistent order)
# 3. Thread safety of shared data structures
#    - list.append() is thread-safe in CPython (due to GIL) but do NOT rely on this
#    - Use queue.Queue for thread-safe producer/consumer
#    - Use threading.local() for thread-local storage

# --- Thread-local storage for per-thread DB connections ---
import threading

thread_local = threading.local()

def get_db_connection():
    if not hasattr(thread_local, "connection"):
        thread_local.connection = create_connection()
    return thread_local.connection
```

---

## 2.4 MULTIPROCESSING

### Q: When and how do you use multiprocessing for AI workloads?

**Answer:**
Use multiprocessing for **CPU-bound** tasks that need true parallelism:
- Data preprocessing (tokenization, image augmentation)
- Feature engineering on large datasets
- Batch inference on CPU
- Any pure Python compute-intensive work

```python
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue
from concurrent.futures import ProcessPoolExecutor

# --- Pool for parallel data preprocessing ---
def preprocess_document(doc: dict) -> dict:
    """CPU-bound preprocessing -- runs in separate process."""
    text = doc["text"]
    tokens = tokenize(text)
    embeddings = compute_local_embeddings(tokens)
    return {"id": doc["id"], "tokens": tokens, "embeddings": embeddings}

def parallel_preprocess(documents: list[dict], num_workers: int = None) -> list[dict]:
    num_workers = num_workers or mp.cpu_count()
    with Pool(num_workers) as pool:
        results = pool.map(preprocess_document, documents)
        # Or for better load balancing:
        # results = pool.map(preprocess_document, documents, chunksize=100)
    return results

# --- ProcessPoolExecutor (higher-level API) ---
def parallel_preprocess_futures(documents: list[dict]) -> list[dict]:
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(preprocess_document, doc) for doc in documents]
        results = [f.result() for f in futures]
    return results

# --- Shared memory for large arrays (Python 3.8+) ---
from multiprocessing import shared_memory
import numpy as np

def create_shared_embeddings(embeddings: np.ndarray):
    """Share large numpy arrays between processes without copying."""
    shm = shared_memory.SharedMemory(create=True, size=embeddings.nbytes)
    shared_array = np.ndarray(embeddings.shape, dtype=embeddings.dtype, buffer=shm.buf)
    shared_array[:] = embeddings[:]
    return shm.name, embeddings.shape, embeddings.dtype

def worker_read_shared(shm_name, shape, dtype):
    """Worker process reads shared memory without copying."""
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    # Use array (read-only for safety)
    result = process_embeddings(array)
    existing_shm.close()
    return result

# --- Key multiprocessing gotchas ---
# 1. Serialization: Arguments/return values must be picklable
#    - Lambda functions are NOT picklable
#    - Local functions are NOT picklable
#    - Some objects (file handles, DB connections, GPU tensors) are NOT picklable
# 2. Memory: Each process gets its own copy of memory (unless using shared memory)
# 3. Startup cost: Process creation is expensive (~100ms) -- use pools
# 4. macOS: Use 'spawn' start method (default on macOS since 3.8)
#    mp.set_start_method('spawn')  # or 'fork' or 'forkserver'
```

---

## 2.5 CONCURRENT.FUTURES -- UNIFIED INTERFACE

### Q: Explain concurrent.futures and when to use it.

**Answer:**
`concurrent.futures` provides a uniform high-level interface for both threading and multiprocessing:

```python
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
    ALL_COMPLETED,
)

# --- Unified pattern: swap executor based on workload ---
def process_batch(items: list, func, use_processes: bool = False, max_workers: int = 4):
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in items}
        results = {}

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                results[item] = future.result(timeout=30)
            except TimeoutError:
                results[item] = None
                logging.warning(f"Timeout processing {item}")
            except Exception as e:
                results[item] = None
                logging.error(f"Error processing {item}: {e}")

        return results

# I/O-bound: use threads
api_results = process_batch(prompts, call_llm_api, use_processes=False)

# CPU-bound: use processes
preprocessed = process_batch(documents, preprocess_doc, use_processes=True)

# --- Future object methods ---
future = executor.submit(func, arg)
future.done()         # True if completed (success or error)
future.running()      # True if currently running
future.cancelled()    # True if cancelled
future.cancel()       # Attempt to cancel (returns True if successful)
future.result(timeout=10)  # Get result (blocks), raises TimeoutError
future.exception()    # Get exception (None if no error)
future.add_done_callback(lambda f: print(f.result()))

# --- map() for simple cases ---
with ThreadPoolExecutor(max_workers=10) as executor:
    # Returns iterator of results IN ORDER (unlike as_completed)
    results = list(executor.map(call_llm_api, prompts, timeout=60))
```

---

## 2.6 AIOHTTP FOR ASYNC HTTP

### Q: How do you use aiohttp for async LLM API calls?

**Answer:**
```python
import aiohttp
import asyncio
from typing import AsyncIterator

class AsyncLLMClient:
    def __init__(self, api_key: str, base_url: str, max_concurrent: int = 10):
        self.api_key = api_key
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            connector = aiohttp.TCPConnector(
                limit=100,           # Max total connections
                limit_per_host=20,   # Max per host
                ttl_dns_cache=300,   # DNS cache TTL
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def complete(self, prompt: str, model: str = "gpt-4") -> str:
        async with self.semaphore:  # Rate limiting
            session = await self._get_session()
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"]

    async def stream(self, prompt: str, model: str = "gpt-4") -> AsyncIterator[str]:
        """Stream tokens from the LLM."""
        async with self.semaphore:
            session = await self._get_session()
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]

    async def batch_complete(self, prompts: list[str]) -> list[str]:
        """Complete multiple prompts concurrently with rate limiting."""
        tasks = [self.complete(p) for p in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

# Usage
async def main():
    async with AsyncLLMClient(api_key="...", base_url="...") as client:
        # Single call
        response = await client.complete("Explain transformers")

        # Batch calls (concurrent with rate limiting)
        responses = await client.batch_complete([
            "Question 1", "Question 2", "Question 3"
        ])

        # Streaming
        async for token in client.stream("Tell me a story"):
            print(token, end="", flush=True)
```

---

## 2.7 DECISION MATRIX: WHICH CONCURRENCY APPROACH?

### Q: How do you decide which concurrency approach to use?

**Answer:**

```
+---------------------+------------------+------------------+-----------------+
| Scenario            | Best Approach    | Why              | Example         |
+---------------------+------------------+------------------+-----------------+
| Many LLM API calls  | asyncio          | I/O-bound,       | Batch inference |
|                     |                  | high concurrency | via API         |
+---------------------+------------------+------------------+-----------------+
| LLM API + sync libs | threading        | I/O-bound, but   | Using requests  |
|                     |                  | legacy sync code | library         |
+---------------------+------------------+------------------+-----------------+
| Data preprocessing  | multiprocessing  | CPU-bound, needs  | Tokenization,   |
| (pure Python)       |                  | true parallelism | text cleaning   |
+---------------------+------------------+------------------+-----------------+
| NumPy/Torch compute | As-is (no GIL   | These libs       | Matrix ops,     |
|                     | release)         | release GIL      | model inference |
+---------------------+------------------+------------------+-----------------+
| Mixed I/O + CPU     | asyncio +        | Async for I/O,   | Fetch data then |
|                     | ProcessPool      | processes for    | preprocess it   |
|                     | Executor         | CPU work         |                 |
+---------------------+------------------+------------------+-----------------+
| Web server          | asyncio (uvicorn | High concurrency | FastAPI app     |
|                     | + FastAPI)       | for requests     | serving AI      |
+---------------------+------------------+------------------+-----------------+
| Simple scripting    | concurrent.      | Easy API,        | Batch file      |
|                     | futures          | good enough      | processing      |
+---------------------+------------------+------------------+-----------------+
```

### Q: How would you architect an AI service that handles 1000 concurrent requests?

**Answer:**
```python
# Architecture: FastAPI + asyncio + multiprocessing

from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ProcessPoolExecutor

# Process pool for CPU-bound work (shared across requests)
process_pool = ProcessPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.llm_client = AsyncLLMClient(...)
    app.state.semaphore = asyncio.Semaphore(50)  # Max 50 concurrent LLM calls
    yield
    # Shutdown
    await app.state.llm_client.close()
    process_pool.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Step 1: CPU-bound preprocessing (run in process pool)
    loop = asyncio.get_event_loop()
    preprocessed = await loop.run_in_executor(
        process_pool,
        preprocess_input,
        request.text
    )

    # Step 2: Async LLM call (I/O-bound, with rate limiting)
    async with app.state.semaphore:
        result = await app.state.llm_client.complete(preprocessed)

    # Step 3: CPU-bound postprocessing
    final = await loop.run_in_executor(
        process_pool,
        postprocess_output,
        result
    )

    return {"result": final}
```

---

## 2.8 ADVANCED: ASYNCIO INTERNALS AND DEBUGGING

### Q: How do you debug asyncio code in production?

**Answer:**
```python
import asyncio
import logging

# Enable asyncio debug mode
asyncio.run(main(), debug=True)
# Or: PYTHONASYNCIODEBUG=1 python script.py

# This enables:
# - Warnings for coroutines that were never awaited
# - Warnings for callbacks taking > 100ms
# - Tracking of where tasks were created

# --- Custom exception handler ---
def exception_handler(loop, context):
    exception = context.get("exception")
    message = context.get("message", "")
    task = context.get("task")
    logging.error(f"Async error: {message}", exc_info=exception)
    if task:
        logging.error(f"Task: {task.get_name()}, created at: {task.get_coro()}")

loop = asyncio.get_event_loop()
loop.set_exception_handler(exception_handler)

# --- Monitor event loop health ---
async def monitor_event_loop():
    """Detect event loop blocking."""
    while True:
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)
        elapsed = asyncio.get_event_loop().time() - start
        if elapsed > 0.2:  # Should be ~0.1s
            logging.warning(f"Event loop blocked for {elapsed:.2f}s")
```
