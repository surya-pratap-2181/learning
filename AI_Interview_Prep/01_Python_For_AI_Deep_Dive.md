---
title: "Python for AI Deep Dive"
layout: default
parent: "Python & Coding"
nav_order: 1
---

# Section 1: Python for AI -- Deep Dive Interview Questions & Answers

---

## 1.1 ASYNC/AWAIT DEEP DIVE

### Q: What is the difference between concurrency, parallelism, and asynchronous execution in Python?

**Answer:**
- **Concurrency**: Multiple tasks make progress over the same time period (interleaved execution). They don't necessarily run at the same instant.
- **Parallelism**: Multiple tasks literally execute at the same instant on multiple CPU cores.
- **Asynchronous**: A programming model where tasks can start, pause (while waiting for I/O), and resume -- without blocking other tasks.

In Python:
- `asyncio` provides **concurrency** (single-threaded, cooperative multitasking)
- `threading` provides **concurrency** (preemptive, but limited by GIL for CPU work)
- `multiprocessing` provides true **parallelism** (separate processes, separate GIL)

### Q: Explain the async/await syntax in depth. What happens under the hood?

**Answer:**
```python
import asyncio

async def fetch_data(url: str) -> dict:
    """An async function (coroutine function) that returns a coroutine object when called."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def main():
    # These run concurrently, NOT in parallel
    results = await asyncio.gather(
        fetch_data("https://api.example.com/data1"),
        fetch_data("https://api.example.com/data2"),
        fetch_data("https://api.example.com/data3"),
    )
    return results

asyncio.run(main())
```

**Under the hood:**
1. `async def` creates a **coroutine function**. Calling it returns a **coroutine object** (not the result).
2. `await` suspends the coroutine and yields control back to the **event loop**.
3. The **event loop** is a single-threaded scheduler that manages which coroutine runs next.
4. When the awaited operation completes (e.g., network I/O), the event loop resumes the coroutine.
5. `asyncio.run()` creates a new event loop, runs the coroutine to completion, and closes the loop.

**Key concepts:**
- Coroutines are based on Python generators (they use `yield` internally via `__await__`).
- The event loop uses OS-level I/O multiplexing (`select`, `epoll`, `kqueue`) to efficiently wait for I/O.
- Only one coroutine runs at any given time (single-threaded).

### Q: What is the difference between `asyncio.gather()`, `asyncio.wait()`, and `asyncio.TaskGroup`?

**Answer:**
```python
import asyncio

# 1. asyncio.gather() -- run coroutines concurrently, return results in order
async def with_gather():
    results = await asyncio.gather(
        coro1(),
        coro2(),
        coro3(),
        return_exceptions=True  # Don't cancel others if one fails
    )
    # results is [result1, result2, result3] in ORDER

# 2. asyncio.wait() -- more control, returns sets of done/pending
async def with_wait():
    tasks = [asyncio.create_task(coro()) for coro in [coro1, coro2, coro3]]
    done, pending = await asyncio.wait(
        tasks,
        timeout=5.0,
        return_when=asyncio.FIRST_COMPLETED  # or ALL_COMPLETED, FIRST_EXCEPTION
    )
    for task in done:
        result = task.result()

# 3. asyncio.TaskGroup (Python 3.11+) -- structured concurrency
async def with_taskgroup():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(coro1())
        task2 = tg.create_task(coro2())
        task3 = tg.create_task(coro3())
    # All tasks guaranteed complete here
    # If ANY task raises, ALL others are cancelled and ExceptionGroup is raised
    results = [task1.result(), task2.result(), task3.result()]
```

**When to use which:**
- `gather()` -- simple fan-out, need ordered results, okay with partial failure handling
- `wait()` -- need timeout control, want to process results as they complete
- `TaskGroup` -- modern Python 3.11+, want structured concurrency with automatic cleanup

### Q: How do you handle cancellation and timeouts in asyncio?

**Answer:**
```python
import asyncio

# Timeout with asyncio.timeout (Python 3.11+)
async def with_timeout():
    async with asyncio.timeout(5.0):
        result = await long_running_operation()

# Timeout with asyncio.wait_for (older approach)
async def with_wait_for():
    try:
        result = await asyncio.wait_for(long_running_operation(), timeout=5.0)
    except asyncio.TimeoutError:
        print("Operation timed out")

# Manual cancellation
async def cancellation_example():
    task = asyncio.create_task(long_running_operation())
    await asyncio.sleep(2)
    task.cancel()  # Sends CancelledError to the task
    try:
        await task
    except asyncio.CancelledError:
        print("Task was cancelled")

# Handling cancellation inside a coroutine
async def cancellable_operation():
    try:
        while True:
            await asyncio.sleep(1)
            # do work
    except asyncio.CancelledError:
        # Cleanup resources
        print("Cleaning up...")
        raise  # Re-raise to propagate cancellation (best practice)
```

### Q: What are async generators and async context managers?

**Answer:**
```python
# Async Generator -- yields values asynchronously
async def stream_llm_response(prompt: str):
    """Async generator for streaming LLM tokens."""
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_URL, json={"prompt": prompt}) as resp:
            async for chunk in resp.content.iter_any():
                token = chunk.decode()
                yield token

# Consuming an async generator
async def process_stream():
    async for token in stream_llm_response("Explain transformers"):
        print(token, end="", flush=True)

# Async Context Manager
class AsyncDatabasePool:
    async def __aenter__(self):
        self.pool = await create_pool(dsn="postgresql://...")
        return self.pool

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.pool.close()
        return False  # Don't suppress exceptions

# Using contextlib
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_llm_client():
    client = AsyncLLMClient()
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()
```

### Q: What are common async pitfalls in production AI systems?

**Answer:**
1. **Blocking the event loop**: Calling synchronous I/O or CPU-heavy code in an async function.
   ```python
   # BAD -- blocks event loop
   async def bad_example():
       result = requests.get(url)  # Synchronous! Blocks everything.

   # GOOD -- use async library or run in executor
   async def good_example():
       result = await aiohttp_session.get(url)

   # GOOD -- for CPU-bound or unavoidable sync code
   async def good_cpu_example():
       loop = asyncio.get_event_loop()
       result = await loop.run_in_executor(None, cpu_intensive_function, args)
   ```

2. **Fire-and-forget tasks disappearing**: Tasks created but never awaited get garbage collected.
   ```python
   # BAD -- task may be garbage collected
   async def bad():
       asyncio.create_task(background_job())

   # GOOD -- keep a reference
   background_tasks = set()
   async def good():
       task = asyncio.create_task(background_job())
       background_tasks.add(task)
       task.add_done_callback(background_tasks.discard)
   ```

3. **Not handling backpressure**: Producer creates items faster than consumer processes.
   ```python
   # Use asyncio.Queue for backpressure
   async def producer(queue: asyncio.Queue):
       for item in data:
           await queue.put(item)  # Blocks if queue is full

   async def consumer(queue: asyncio.Queue):
       while True:
           item = await queue.get()
           await process(item)
           queue.task_done()
   ```

---

## 1.2 GENERATORS DEEP DIVE

### Q: Explain generators, generator expressions, and the `yield` keyword in depth.

**Answer:**
```python
# Basic generator function
def fibonacci():
    """Lazy infinite sequence -- only computes values on demand."""
    a, b = 0, 1
    while True:
        yield a  # Suspends execution, returns value
        a, b = b, a + b

# Generator expression (like list comprehension but lazy)
squares_gen = (x**2 for x in range(1_000_000))  # No memory allocation for all values
squares_list = [x**2 for x in range(1_000_000)]  # Allocates full list in memory

# Generator protocol
gen = fibonacci()
next(gen)  # 0 -- advances to first yield
next(gen)  # 1 -- advances to next yield
next(gen)  # 1
```

**How generators work internally:**
- A generator function returns a **generator object** that implements the **iterator protocol** (`__iter__` and `__next__`).
- Each `yield` suspends the function's execution and saves its **frame state** (local variables, instruction pointer).
- `next()` resumes execution from where it was suspended.
- When the function returns (or falls off the end), `StopIteration` is raised.

### Q: What is `yield from` and how does it enable coroutine delegation?

**Answer:**
```python
# yield from delegates to a sub-generator
def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)  # Delegates to recursive call
        else:
            yield item

list(flatten([1, [2, [3, 4], 5], 6]))  # [1, 2, 3, 4, 5, 6]

# yield from with send() -- bidirectional communication
def sub_generator():
    value = yield "from sub"
    return f"sub got: {value}"

def main_generator():
    result = yield from sub_generator()
    # result is the return value of sub_generator
    print(result)
```

**`yield from` provides:**
1. Automatic iteration of the sub-generator
2. Forwarding of `send()` values to the sub-generator
3. Forwarding of `throw()` exceptions to the sub-generator
4. Propagation of the sub-generator's return value

### Q: How are generators used in AI/ML pipelines?

**Answer:**
```python
# Data processing pipeline for training data
def read_jsonl(filepath: str):
    """Lazily read JSONL file -- handles files larger than RAM."""
    with open(filepath) as f:
        for line in f:
            yield json.loads(line)

def filter_quality(records):
    """Filter based on quality threshold."""
    for record in records:
        if record.get("quality_score", 0) > 0.8:
            yield record

def batch(iterable, size=32):
    """Create batches for model inference."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

def tokenize_batch(batches, tokenizer):
    for b in batches:
        yield tokenizer(
            [item["text"] for item in b],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

# Composable pipeline -- processes data lazily
pipeline = tokenize_batch(
    batch(
        filter_quality(
            read_jsonl("training_data.jsonl")
        ),
        size=64
    ),
    tokenizer=my_tokenizer
)

for token_batch in pipeline:
    model.train_step(token_batch)
```

### Q: What is `send()` and `throw()` on generators? When would you use them?

**Answer:**
```python
# send() -- inject values into a generator
def accumulator():
    total = 0
    while True:
        value = yield total  # yield current total, receive new value
        if value is None:
            break
        total += value

acc = accumulator()
next(acc)       # Prime the generator, returns 0
acc.send(10)    # Returns 10
acc.send(20)    # Returns 30
acc.send(5)     # Returns 35

# throw() -- inject exceptions into a generator
def resilient_processor():
    while True:
        try:
            data = yield
            process(data)
        except ValueError:
            log.error("Bad data, skipping")
            continue

proc = resilient_processor()
next(proc)
proc.send(good_data)
proc.throw(ValueError, "bad input")  # Generator handles it and continues
```

**AI use case -- a controllable streaming pipeline:**
```python
def streaming_inference(model):
    """Generator that accepts prompts via send() and yields responses."""
    while True:
        prompt = yield  # Receive prompt
        for token in model.generate_stream(prompt):
            yield token
        yield None  # Sentinel: generation complete
```

---

## 1.3 DECORATORS DEEP DIVE

### Q: Explain decorators, decorator factories, and class-based decorators with real AI use cases.

**Answer:**
```python
import functools
import time
import logging
from typing import Callable, Any

# --- Basic decorator ---
def log_llm_call(func: Callable) -> Callable:
    """Log every LLM API call with timing."""
    @functools.wraps(func)  # Preserves __name__, __doc__, etc.
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        logging.info(f"Calling {func.__name__} with model={kwargs.get('model', 'unknown')}")
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logging.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logging.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    return wrapper

@log_llm_call
def call_openai(prompt: str, model: str = "gpt-4") -> str:
    return openai.chat.completions.create(...)

# --- Decorator Factory (decorator with arguments) ---
def retry(max_retries: int = 3, backoff_factor: float = 2.0, exceptions=(Exception,)):
    """Retry decorator with exponential backoff -- essential for LLM API calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    wait = backoff_factor ** attempt
                    logging.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. "
                                    f"Retrying in {wait}s...")
                    time.sleep(wait)
            raise last_exception
        return wrapper
    return decorator

@retry(max_retries=5, backoff_factor=1.5, exceptions=(RateLimitError, TimeoutError))
def call_llm(prompt: str) -> str:
    ...

# --- Async decorator ---
def async_retry(max_retries: int = 3):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator

# --- Class-based decorator ---
class CacheEmbeddings:
    """Cache embedding results with TTL."""
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self.cache: dict[str, tuple[float, list[float]]] = {}

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(text: str, *args, **kwargs):
            now = time.time()
            if text in self.cache:
                timestamp, result = self.cache[text]
                if now - timestamp < self.ttl:
                    return result
            result = func(text, *args, **kwargs)
            self.cache[text] = (now, result)
            return result
        return wrapper

@CacheEmbeddings(ttl_seconds=7200)
def get_embedding(text: str) -> list[float]:
    return openai.embeddings.create(input=text, model="text-embedding-3-small")

# --- Stacking decorators ---
@log_llm_call
@retry(max_retries=3)
@validate_input
def robust_llm_call(prompt: str) -> str:
    ...
# Equivalent to: log_llm_call(retry(3)(validate_input(robust_llm_call)))
```

### Q: What does `functools.wraps` do and why is it important?

**Answer:**
Without `@functools.wraps(func)`, the wrapper function replaces the original function's metadata:
- `__name__` becomes `"wrapper"` instead of the original function name
- `__doc__` is lost
- `__module__` is wrong
- `__qualname__` is wrong
- `__annotations__` are lost

This breaks introspection, documentation tools, debugging, and frameworks like FastAPI/Flask that use function metadata for routing.

---

## 1.4 CONTEXT MANAGERS

### Q: Explain context managers, `__enter__`/`__exit__`, and `contextlib` patterns for AI systems.

**Answer:**
```python
# Class-based context manager
class ModelInferenceSession:
    """Manages GPU memory and model loading lifecycle."""
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None

    def __enter__(self):
        self.model = load_model(self.model_path).to(self.device)
        self.model.eval()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        if exc_type is OutOfMemoryError:
            logging.error("OOM during inference, freed GPU memory")
            return True  # Suppress the exception
        return False  # Don't suppress other exceptions

# Usage
with ModelInferenceSession("./models/bert-large", device="cuda:0") as model:
    outputs = model(inputs)

# contextlib.contextmanager -- simpler syntax
from contextlib import contextmanager

@contextmanager
def token_budget_tracker(max_tokens: int):
    """Track and enforce token budgets for LLM calls."""
    tracker = {"used": 0, "max": max_tokens}
    try:
        yield tracker
    finally:
        logging.info(f"Token usage: {tracker['used']}/{tracker['max']}")
        if tracker["used"] > tracker["max"]:
            logging.warning("Token budget exceeded!")

with token_budget_tracker(4096) as budget:
    response = call_llm(prompt)
    budget["used"] += response.usage.total_tokens

# Nested context managers with ExitStack
from contextlib import ExitStack

def load_ensemble_models(model_paths: list[str]):
    with ExitStack() as stack:
        models = [
            stack.enter_context(ModelInferenceSession(path))
            for path in model_paths
        ]
        # All models loaded, all will be cleaned up even if one fails
        return ensemble_predict(models, inputs)

# Reentrant vs Reusable context managers
# - Reentrant: can be used in nested `with` statements (e.g., redirect_stdout)
# - Reusable: can be used multiple times but not nested (most custom CMs)
# - Single-use: can only be used once (e.g., generators from @contextmanager)
```

### Q: When should `__exit__` return True vs False?

**Answer:**
- Return `True` to **suppress** the exception (it will not propagate).
- Return `False` (or `None`) to **propagate** the exception.
- Suppressing exceptions is rare and should be done very carefully. Common use case: handling expected failures gracefully (e.g., OOM errors, cleanup-on-failure patterns).

---

## 1.5 METACLASSES

### Q: What are metaclasses? When would you use them in an AI system?

**Answer:**
A metaclass is the "class of a class." Just as an object is an instance of a class, a class is an instance of a metaclass. The default metaclass is `type`.

```python
# The relationship:
# type -> MyClass -> my_instance
# metaclass -> class -> object

# How classes are created (simplified):
# MyClass = type('MyClass', (BaseClass,), {'method': method_func})

# Custom metaclass example: Auto-registering AI model plugins
class ModelRegistry(type):
    """Metaclass that automatically registers all model subclasses."""
    _registry: dict[str, type] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases:  # Don't register the base class itself
            model_name = namespace.get("model_name", name.lower())
            mcs._registry[model_name] = cls
        return cls

    @classmethod
    def get_model(mcs, name: str):
        return mcs._registry.get(name)

class BaseModel(metaclass=ModelRegistry):
    model_name: str = ""

    def predict(self, inputs):
        raise NotImplementedError

class GPT4Model(BaseModel):
    model_name = "gpt-4"
    def predict(self, inputs):
        return call_openai(inputs, model="gpt-4")

class ClaudeModel(BaseModel):
    model_name = "claude-3-opus"
    def predict(self, inputs):
        return call_anthropic(inputs, model="claude-3-opus-20240229")

# Now we can do:
model_cls = ModelRegistry.get_model("gpt-4")  # Returns GPT4Model class
model = model_cls()
```

**When to use metaclasses vs alternatives:**
- **Metaclasses**: Deep framework-level behavior, auto-registration, enforcing class invariants.
- **Class decorators**: Simpler, often sufficient. Prefer over metaclasses.
- **`__init_subclass__`** (Python 3.6+): Simpler alternative for subclass hooks.
- **ABCs**: For enforcing interface contracts.

```python
# __init_subclass__ -- usually preferred over metaclasses
class BaseModel:
    _registry = {}

    def __init_subclass__(cls, model_name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if model_name:
            BaseModel._registry[model_name] = cls

class GPT4Model(BaseModel, model_name="gpt-4"):
    ...
```

---

## 1.6 TYPE HINTS (Advanced)

### Q: Demonstrate advanced type hints relevant to AI codebases.

**Answer:**
```python
from typing import (
    TypeVar, Generic, Protocol, TypeAlias, Literal, TypeGuard,
    overload, TypedDict, Unpack, Self, Never, Annotated
)
from collections.abc import Callable, AsyncIterator, Sequence
import numpy as np
from numpy.typing import NDArray

# --- TypeVar and Generic for model wrappers ---
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class Pipeline(Generic[InputT, OutputT]):
    def __init__(self, steps: list[Callable]):
        self.steps = steps

    def run(self, input_data: InputT) -> OutputT:
        result = input_data
        for step in self.steps:
            result = step(result)
        return result

# --- Protocol for structural subtyping (duck typing with type checking) ---
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...
    @property
    def dimension(self) -> int: ...

# Any class with embed() and dimension works -- no inheritance needed
class OpenAIEmbedder:
    @property
    def dimension(self) -> int:
        return 1536
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        ...

def build_index(embedder: Embedder, documents: list[str]) -> None:
    vectors = embedder.embed(documents)  # Type-safe

# --- TypedDict for LLM response structures ---
class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class LLMResponse(TypedDict):
    id: str
    choices: list[dict]
    usage: "TokenUsage"

class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# --- TypeGuard for narrowing types ---
def is_valid_embedding(value: object) -> TypeGuard[list[float]]:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(v, float) for v in value)
    )

def process(data: object) -> None:
    if is_valid_embedding(data):
        # Type checker knows data is list[float] here
        vector = np.array(data)

# --- Overload for different return types ---
@overload
def get_completion(prompt: str, stream: Literal[True]) -> AsyncIterator[str]: ...
@overload
def get_completion(prompt: str, stream: Literal[False] = False) -> str: ...

def get_completion(prompt: str, stream: bool = False):
    if stream:
        return _stream_completion(prompt)
    return _sync_completion(prompt)

# --- Type aliases for complex types ---
EmbeddingVector: TypeAlias = NDArray[np.float32]
ConversationHistory: TypeAlias = list[LLMMessage]
ModelConfig: TypeAlias = dict[str, int | float | str | bool]

# --- ParamSpec for preserving function signatures in decorators ---
from typing import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def log_call(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logging.info(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# --- Annotated for validation metadata ---
from annotated_types import Gt, Le
Temperature = Annotated[float, Gt(0), Le(2.0)]
MaxTokens = Annotated[int, Gt(0), Le(128000)]
```

---

## 1.7 DATACLASSES

### Q: Explain dataclasses and their features. When would you use dataclasses vs Pydantic vs NamedTuple vs attrs?

**Answer:**
```python
from dataclasses import dataclass, field, asdict, astuple
from typing import Optional

# --- Basic dataclass ---
@dataclass
class ModelConfig:
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    stop_sequences: list[str] = field(default_factory=list)  # Mutable defaults need factory
    _internal: str = field(repr=False, compare=False, default="")

# --- Frozen (immutable) dataclass ---
@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    vector: tuple[float, ...]  # Use tuple for immutability
    model: str
    dimension: int

# --- Post-init processing ---
@dataclass
class TrainingRun:
    epochs: int
    learning_rate: float
    batch_size: int
    total_steps: int = field(init=False)  # Computed field

    def __post_init__(self):
        self.total_steps = self.epochs * (50000 // self.batch_size)

# --- Slots (Python 3.10+) ---
@dataclass(slots=True)  # Faster attribute access, less memory
class Token:
    text: str
    logprob: float
    index: int

# --- Serialization ---
config = ModelConfig(model_name="gpt-4", temperature=0.9)
config_dict = asdict(config)   # {'model_name': 'gpt-4', 'temperature': 0.9, ...}
config_tuple = astuple(config)
```

**Comparison table:**

| Feature            | dataclass | Pydantic (v2) | NamedTuple    | attrs         |
|--------------------|-----------|---------------|---------------|---------------|
| Validation         | Manual    | Automatic     | None          | Optional      |
| Immutability       | frozen=True | Configurable | Always frozen | Optional      |
| Performance        | Fast      | Fast (Rust)   | Fastest       | Fast          |
| Serialization      | asdict    | model_dump    | _asdict       | asdict        |
| JSON Schema        | No        | Yes (auto)    | No            | No (plugin)   |
| Inheritance        | Yes       | Yes           | Awkward       | Yes           |
| Slots support      | 3.10+     | Yes           | N/A           | Yes           |
| Best for           | Internal data | API/config  | Simple records | Power users  |

---

## 1.8 PYDANTIC DEEP DIVE

### Q: Demonstrate Pydantic v2 patterns used in production AI systems.

**Answer:**
```python
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic import SecretStr, HttpUrl, computed_field
from typing import Literal, Annotated
from datetime import datetime
from enum import Enum

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

# --- LLM Configuration with validation ---
class LLMConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,             # Immutable
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",          # No extra fields allowed
    )

    provider: ModelProvider
    model_name: str
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.7
    max_tokens: Annotated[int, Field(gt=0, le=128000)] = 4096
    api_key: SecretStr  # Automatically redacted in logs/repr
    base_url: HttpUrl | None = None
    stop_sequences: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str, info) -> str:
        provider = info.data.get("provider")
        valid_models = {
            ModelProvider.OPENAI: ["gpt-4", "gpt-4-turbo", "gpt-4o"],
            ModelProvider.ANTHROPIC: ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        }
        if provider in valid_models and v not in valid_models[provider]:
            raise ValueError(f"Invalid model {v} for provider {provider}")
        return v

# --- Structured Output from LLM ---
class ExtractedEntity(BaseModel):
    name: str
    entity_type: Literal["person", "organization", "location", "date"]
    confidence: float = Field(ge=0.0, le=1.0)
    context: str = ""

class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    raw_text: str
    model_used: str
    processing_time_ms: float

    @computed_field
    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @model_validator(mode="after")
    def check_entities_from_text(self) -> "ExtractionResult":
        for entity in self.entities:
            if entity.name.lower() not in self.raw_text.lower():
                raise ValueError(f"Entity '{entity.name}' not found in raw text")
        return self

# --- Discriminated unions for different tool outputs ---
class SearchResult(BaseModel):
    tool: Literal["search"] = "search"
    query: str
    results: list[str]

class CalculatorResult(BaseModel):
    tool: Literal["calculator"] = "calculator"
    expression: str
    result: float

class CodeResult(BaseModel):
    tool: Literal["code"] = "code"
    code: str
    output: str

ToolOutput = Annotated[
    SearchResult | CalculatorResult | CodeResult,
    Field(discriminator="tool")
]

# --- Parsing LLM JSON output with Pydantic ---
import json

def parse_llm_output(raw_output: str, schema: type[BaseModel]) -> BaseModel:
    """Robustly parse LLM JSON output."""
    # Strip markdown code fences if present
    cleaned = raw_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return schema.model_validate_json(cleaned.strip())
```

---

## 1.9 ABSTRACT BASE CLASSES (ABCs)

### Q: How do you use ABCs to define interfaces in AI systems?

**Answer:**
```python
from abc import ABC, abstractmethod
from typing import Any

class BaseLLMClient(ABC):
    """Abstract interface for LLM providers -- enables swappable backends."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt."""
        ...

    @abstractmethod
    async def acomplete(self, prompt: str, **kwargs) -> str:
        """Async version of complete."""
        ...

    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Stream a completion token by token."""
        ...

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Count tokens for the given text."""
        ...

    # Concrete method (shared implementation)
    def complete_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        for attempt in range(max_retries):
            try:
                return self.complete(prompt, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

class BaseVectorStore(ABC):
    """Abstract interface for vector databases."""

    @abstractmethod
    def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]) -> None: ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int = 10, filter: dict | None = None) -> list[dict]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

# Using register() for virtual subclasses (duck typing with ABC)
from collections.abc import Sized

class EmbeddingBatch:
    def __len__(self):
        return self._count

Sized.register(EmbeddingBatch)  # EmbeddingBatch is now a virtual subclass of Sized
isinstance(EmbeddingBatch(), Sized)  # True
```

**ABCs vs Protocols:**
- ABCs require **explicit inheritance** (`class Foo(MyABC)`)
- Protocols use **structural subtyping** (duck typing) -- no inheritance needed
- Use ABCs when you want to enforce that developers explicitly opt into an interface
- Use Protocols when you want flexibility and interoperability with third-party code

---

## 1.10 DESIGN PATTERNS IN PYTHON FOR AI

### Q: What design patterns are most relevant to AI systems?

**Answer:**

```python
# === 1. STRATEGY PATTERN -- swappable LLM providers ===
from dataclasses import dataclass

class LLMStrategy(Protocol):
    def generate(self, prompt: str) -> str: ...

class OpenAIStrategy:
    def generate(self, prompt: str) -> str:
        return openai_client.chat(prompt)

class AnthropicStrategy:
    def generate(self, prompt: str) -> str:
        return anthropic_client.messages(prompt)

class LocalModelStrategy:
    def generate(self, prompt: str) -> str:
        return local_model.generate(prompt)

@dataclass
class AIAgent:
    llm: LLMStrategy  # Inject any strategy
    def run(self, task: str) -> str:
        return self.llm.generate(task)

agent = AIAgent(llm=OpenAIStrategy())
agent = AIAgent(llm=AnthropicStrategy())  # Easy to swap

# === 2. CHAIN OF RESPONSIBILITY -- RAG pipeline stages ===
class PipelineStep(ABC):
    def __init__(self):
        self._next: PipelineStep | None = None

    def set_next(self, step: "PipelineStep") -> "PipelineStep":
        self._next = step
        return step

    def process(self, data: dict) -> dict:
        result = self.handle(data)
        if self._next:
            return self._next.process(result)
        return result

    @abstractmethod
    def handle(self, data: dict) -> dict: ...

class QueryRewriter(PipelineStep):
    def handle(self, data: dict) -> dict:
        data["query"] = rewrite_query(data["query"])
        return data

class Retriever(PipelineStep):
    def handle(self, data: dict) -> dict:
        data["documents"] = vector_store.query(data["query"])
        return data

class Reranker(PipelineStep):
    def handle(self, data: dict) -> dict:
        data["documents"] = rerank(data["documents"], data["query"])
        return data

class Generator(PipelineStep):
    def handle(self, data: dict) -> dict:
        data["response"] = llm.generate(data["query"], data["documents"])
        return data

# Build the chain
pipeline = QueryRewriter()
pipeline.set_next(Retriever()).set_next(Reranker()).set_next(Generator())
result = pipeline.process({"query": "What is attention?"})

# === 3. OBSERVER PATTERN -- monitoring AI pipeline events ===
class EventBus:
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}

    def subscribe(self, event: str, callback: Callable):
        self._listeners.setdefault(event, []).append(callback)

    def publish(self, event: str, data: Any):
        for callback in self._listeners.get(event, []):
            callback(data)

bus = EventBus()
bus.subscribe("llm.token_generated", lambda data: metrics.record_tokens(data))
bus.subscribe("llm.error", lambda data: alerting.send_alert(data))
bus.subscribe("llm.latency", lambda data: dashboard.update(data))

# === 4. FACTORY PATTERN -- creating model instances from config ===
class ModelFactory:
    _creators: dict[str, Callable] = {}

    @classmethod
    def register(cls, model_type: str):
        def decorator(creator_cls):
            cls._creators[model_type] = creator_cls
            return creator_cls
        return decorator

    @classmethod
    def create(cls, config: dict) -> BaseLLMClient:
        model_type = config["type"]
        if model_type not in cls._creators:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._creators[model_type](**config.get("params", {}))

@ModelFactory.register("openai")
class OpenAIClient(BaseLLMClient):
    ...

@ModelFactory.register("anthropic")
class AnthropicClient(BaseLLMClient):
    ...

# Create from config
client = ModelFactory.create({"type": "openai", "params": {"model": "gpt-4"}})

# === 5. SINGLETON -- shared resources ===
class EmbeddingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Better: use module-level instance or dependency injection

# === 6. BUILDER PATTERN -- constructing complex prompts ===
class PromptBuilder:
    def __init__(self):
        self._messages: list[dict] = []
        self._tools: list[dict] = []

    def system(self, content: str) -> "PromptBuilder":
        self._messages.append({"role": "system", "content": content})
        return self

    def user(self, content: str) -> "PromptBuilder":
        self._messages.append({"role": "user", "content": content})
        return self

    def assistant(self, content: str) -> "PromptBuilder":
        self._messages.append({"role": "assistant", "content": content})
        return self

    def with_tool(self, name: str, description: str, parameters: dict) -> "PromptBuilder":
        self._tools.append({"name": name, "description": description, "parameters": parameters})
        return self

    def build(self) -> dict:
        result = {"messages": self._messages}
        if self._tools:
            result["tools"] = self._tools
        return result

prompt = (
    PromptBuilder()
    .system("You are a helpful AI assistant.")
    .user("What is the capital of France?")
    .with_tool("search", "Search the web", {"query": {"type": "string"}})
    .build()
)
```
