# Section 3: Data Structures, Algorithms & Memory Management for AI Engineers

---

## 3.1 PYTHON DATA STRUCTURES FOR AI WORKLOADS

### Q: What are the most important data structure choices for AI systems?

**Answer:**

```python
# === DICTIONARIES: The workhorse of AI systems ===
# Python dicts are hash tables: O(1) average lookup, insert, delete
# Since Python 3.7+: insertion-ordered

# Use case: Feature stores, caches, configuration
feature_cache: dict[str, list[float]] = {}

# defaultdict for aggregation
from collections import defaultdict
token_counts: defaultdict[str, int] = defaultdict(int)
for token in document_tokens:
    token_counts[token] += 1

# OrderedDict for LRU-like behavior (move_to_end)
from collections import OrderedDict

class LRUCache:
    def __init__(self, maxsize: int = 1000):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # Remove oldest

# === DEQUE: Sliding windows, BFS, conversation history ===
from collections import deque

class ConversationBuffer:
    """Fixed-size conversation history using deque."""
    def __init__(self, max_turns: int = 10):
        self._messages: deque[dict] = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def token_count(self) -> int:
        return sum(count_tokens(m["content"]) for m in self._messages)

# Sliding window for streaming metrics
class SlidingWindowMetrics:
    def __init__(self, window_size: int = 100):
        self._latencies: deque[float] = deque(maxlen=window_size)

    def record(self, latency: float):
        self._latencies.append(latency)

    @property
    def p50(self) -> float:
        sorted_vals = sorted(self._latencies)
        return sorted_vals[len(sorted_vals) // 2]

    @property
    def p99(self) -> float:
        sorted_vals = sorted(self._latencies)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

# === HEAPQ: Priority queues for ranking, top-k retrieval ===
import heapq

def top_k_documents(scores: list[tuple[float, str]], k: int = 10) -> list[tuple[float, str]]:
    """Efficiently get top-k items. O(n log k) vs O(n log n) for full sort."""
    return heapq.nlargest(k, scores, key=lambda x: x[0])

# Min-heap for streaming top-k (memory efficient)
class StreamingTopK:
    def __init__(self, k: int = 10):
        self.k = k
        self._heap: list[tuple[float, str]] = []

    def add(self, score: float, doc_id: str):
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (score, doc_id))
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, (score, doc_id))

    def get_top_k(self) -> list[tuple[float, str]]:
        return sorted(self._heap, reverse=True)

# === SETS: Deduplication, membership testing ===
# O(1) membership testing, O(n) intersection/union

seen_documents: set[str] = set()
def deduplicate_stream(documents):
    for doc in documents:
        doc_hash = hash_document(doc)
        if doc_hash not in seen_documents:
            seen_documents.add(doc_hash)
            yield doc

# frozenset for hashable sets (can be dict keys)
feature_combinations = frozenset(["temperature", "max_tokens", "top_p"])

# === NAMEDTUPLE and DATACLASS for structured records ===
from typing import NamedTuple

class SearchResult(NamedTuple):
    """Immutable, lightweight, hashable."""
    doc_id: str
    score: float
    text: str

# === BISECT: Binary search for sorted data ===
import bisect

class ScoreThresholds:
    """Efficient threshold lookup for classification."""
    def __init__(self, thresholds: list[float], labels: list[str]):
        self.thresholds = sorted(thresholds)
        self.labels = labels

    def classify(self, score: float) -> str:
        idx = bisect.bisect_right(self.thresholds, score)
        return self.labels[idx]

thresholds = ScoreThresholds([0.3, 0.6, 0.9], ["low", "medium", "high", "very_high"])
thresholds.classify(0.75)  # "high"
```

---

## 3.2 EFFICIENT DATA PROCESSING PATTERNS

### Q: How do you process large datasets that don't fit in memory?

**Answer:**

```python
# === Pattern 1: Generator-based streaming ===
import json
from typing import Iterator

def stream_jsonl(filepath: str, chunk_size: int = 8192) -> Iterator[dict]:
    """Memory-efficient JSONL reading."""
    with open(filepath, "r") as f:
        for line in f:
            yield json.loads(line)

def process_in_chunks(filepath: str, chunk_size: int = 1000):
    """Process data in fixed-size chunks."""
    chunk = []
    for record in stream_jsonl(filepath):
        chunk.append(record)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# === Pattern 2: Memory-mapped files ===
import mmap
import numpy as np

def read_large_embeddings(filepath: str, dim: int = 1536):
    """Memory-map large embedding files for random access without loading all into RAM."""
    with open(filepath, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        # Read specific embedding by index
        embedding_size = dim * 4  # float32
        def get_embedding(idx: int) -> np.ndarray:
            offset = idx * embedding_size
            mm.seek(offset)
            data = mm.read(embedding_size)
            return np.frombuffer(data, dtype=np.float32)
        return get_embedding

# === Pattern 3: itertools for composable pipelines ===
import itertools

def efficient_pipeline(data_stream):
    # Take first 10,000 items
    limited = itertools.islice(data_stream, 10_000)

    # Filter
    filtered = filter(lambda x: x["quality"] > 0.8, limited)

    # Batch
    def batched(iterable, n):
        """itertools.batched in Python 3.12+"""
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, n))
            if not batch:
                break
            yield batch

    for batch in batched(filtered, 32):
        process_batch(batch)

# === Pattern 4: Polars for large dataframes (faster than pandas) ===
import polars as pl

def process_large_csv(filepath: str):
    # Lazy evaluation -- builds query plan, executes efficiently
    result = (
        pl.scan_csv(filepath)
        .filter(pl.col("score") > 0.8)
        .select(["id", "text", "score"])
        .sort("score", descending=True)
        .head(1000)
        .collect()  # Execute the plan
    )
    return result

# === Pattern 5: Apache Arrow for zero-copy data sharing ===
import pyarrow as pa
import pyarrow.parquet as pq

def process_parquet_streaming(filepath: str):
    """Process parquet file in batches without loading entire file."""
    parquet_file = pq.ParquetFile(filepath)
    for batch in parquet_file.iter_batches(batch_size=10_000):
        table = pa.Table.from_batches([batch])
        df = table.to_pandas()
        process_batch(df)
```

---

## 3.3 STREAMING DATA PATTERNS FOR AI

### Q: How do you implement streaming patterns for real-time AI processing?

**Answer:**

```python
import asyncio
from collections import deque
from typing import AsyncIterator

# === Pattern 1: Async streaming pipeline ===
async def streaming_rag_pipeline(query: str) -> AsyncIterator[str]:
    """Stream RAG results token by token."""
    # Step 1: Retrieve (fast)
    documents = await retrieve_documents(query)

    # Step 2: Stream LLM response
    context = format_context(documents)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    async for token in llm_client.stream(prompt):
        yield token

# === Pattern 2: Backpressure-aware streaming ===
class BackpressureStream:
    def __init__(self, max_buffer: int = 100):
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max_buffer)

    async def write(self, data: str):
        await self._queue.put(data)  # Blocks if buffer full

    async def close(self):
        await self._queue.put(None)

    async def __aiter__(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

# === Pattern 3: Fan-out/Fan-in for parallel retrieval ===
async def parallel_retrieval(query: str, sources: list[str]) -> list[dict]:
    """Query multiple vector stores in parallel, merge results."""
    async def query_source(source: str) -> list[dict]:
        async with get_vector_store(source) as store:
            return await store.query(query, top_k=10)

    all_results = await asyncio.gather(
        *[query_source(s) for s in sources],
        return_exceptions=True
    )

    # Merge and deduplicate
    merged = []
    seen_ids = set()
    for results in all_results:
        if isinstance(results, Exception):
            continue
        for result in results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                merged.append(result)

    # Re-rank merged results
    return sorted(merged, key=lambda x: x["score"], reverse=True)[:20]

# === Pattern 4: Windowed aggregation for metrics ===
class TimeWindowAggregator:
    """Aggregate metrics over time windows."""
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self.events: deque[tuple[float, float]] = deque()

    def add(self, value: float, timestamp: float):
        self.events.append((timestamp, value))
        self._cleanup(timestamp)

    def _cleanup(self, now: float):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def average(self, now: float) -> float:
        self._cleanup(now)
        if not self.events:
            return 0.0
        return sum(v for _, v in self.events) / len(self.events)

    def count(self, now: float) -> int:
        self._cleanup(now)
        return len(self.events)
```

---

## 3.4 MEMORY MANAGEMENT AND PROFILING

### Q: How does Python memory management work and how do you optimize it for AI workloads?

**Answer:**

**Python Memory Model:**
1. **Reference Counting**: Primary mechanism. Each object has a reference count. When it reaches 0, the object is immediately deallocated.
2. **Garbage Collector**: Handles cyclic references that reference counting cannot. Uses generational GC (3 generations).
3. **Memory Allocator**: CPython uses `pymalloc` for small objects (< 512 bytes) with memory pools/arenas.

```python
import sys
import gc
import tracemalloc
from memory_profiler import profile  # pip install memory-profiler

# === Check object size ===
obj = {"key": "value", "list": [1, 2, 3]}
sys.getsizeof(obj)         # Shallow size (just the dict, not contents)

# Deep size calculation
def deep_getsizeof(obj, seen=None):
    """Recursively calculate total memory of an object."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_getsizeof(item, seen) for item in obj)
    return size

# === tracemalloc for memory profiling ===
tracemalloc.start()

# ... your code ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

print("[ Top 10 memory allocations ]")
for stat in top_stats[:10]:
    print(stat)

# Compare snapshots to find leaks
snapshot1 = tracemalloc.take_snapshot()
# ... more code ...
snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, "lineno")
for stat in top_stats[:10]:
    print(stat)

# === Memory-efficient patterns ===

# 1. __slots__ -- eliminates __dict__ per instance (saves ~40 bytes/instance)
class Token:
    __slots__ = ("text", "logprob", "index")
    def __init__(self, text: str, logprob: float, index: int):
        self.text = text
        self.logprob = logprob
        self.index = index

# 2. Generators instead of lists for large sequences
# BAD: creates entire list in memory
all_embeddings = [compute_embedding(text) for text in million_texts]

# GOOD: generates one at a time
def embedding_generator(texts):
    for text in texts:
        yield compute_embedding(text)

# 3. numpy arrays instead of Python lists for numerical data
import numpy as np
# Python list of floats: ~28 bytes per float (object overhead)
# numpy array: 4 bytes per float32, 8 bytes per float64
python_list = [0.0] * 1_000_000   # ~28 MB
numpy_array = np.zeros(1_000_000, dtype=np.float32)  # ~4 MB

# 4. Weak references for caches
import weakref

class ModelCache:
    """Cache models without preventing garbage collection."""
    def __init__(self):
        self._cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def get_or_load(self, model_name: str):
        model = self._cache.get(model_name)
        if model is None:
            model = load_model(model_name)
            self._cache[model_name] = model
        return model

# 5. del and gc.collect() for explicit cleanup
def process_large_batch(data):
    embeddings = compute_embeddings(data)  # Large allocation
    results = process_embeddings(embeddings)
    del embeddings  # Release reference immediately
    gc.collect()    # Force GC cycle (usually not needed, but helpful for large objects)
    return results

# 6. Memory-efficient string handling
# Use intern() for repeated strings
import sys
model_name = sys.intern("gpt-4-turbo")  # Reuses existing string object

# === Profiling tools summary ===
# memory_profiler: Line-by-line memory usage (@profile decorator)
# tracemalloc: Built-in, tracks memory allocations by source
# objgraph: Visualize object reference graphs (find memory leaks)
# pympler: Detailed object memory tracking
# scalene: CPU + memory + GPU profiler (recommended for AI)
```

### Q: How do you profile Python code performance for AI workloads?

**Answer:**
```python
# === 1. cProfile -- function-level profiling ===
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
    return result

# Command line: python -m cProfile -s cumulative script.py

# === 2. line_profiler -- line-by-line timing ===
# pip install line_profiler
# @profile  (decorator from line_profiler)
# kernprof -l -v script.py

# === 3. time.perf_counter for micro-benchmarks ===
import time

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed:.4f}s")

with Timer("LLM call"):
    result = call_llm(prompt)

# === 4. Scalene -- best all-in-one profiler for AI ===
# pip install scalene
# scalene script.py
# Reports: CPU time (Python vs C), memory allocation, GPU usage

# === 5. py-spy -- sampling profiler (low overhead, production-safe) ===
# pip install py-spy
# py-spy record -o profile.svg -- python script.py
# Generates flame graph

# === 6. Custom metrics collection ===
from dataclasses import dataclass, field
from contextlib import contextmanager
import statistics

@dataclass
class PerformanceMetrics:
    latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @contextmanager
    def measure(self, operation: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.latencies[operation].append(elapsed)

    def report(self) -> dict:
        return {
            op: {
                "count": len(times),
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "p95": sorted(times)[int(len(times) * 0.95)] if times else 0,
                "p99": sorted(times)[int(len(times) * 0.99)] if times else 0,
            }
            for op, times in self.latencies.items()
        }

metrics = PerformanceMetrics()

with metrics.measure("embedding"):
    embeddings = model.embed(texts)

with metrics.measure("llm_call"):
    response = llm.complete(prompt)

print(metrics.report())
```

---

## 3.5 ALGORITHMS COMMONLY ASKED IN AI INTERVIEWS

### Q: What algorithms should AI engineers know well?

**Answer:**

**1. Similarity Search / Nearest Neighbors:**
```python
import numpy as np
from typing import List, Tuple

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Core operation in embedding-based retrieval."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def brute_force_knn(query: np.ndarray, vectors: np.ndarray, k: int = 10) -> list[int]:
    """O(n*d) brute force -- works for small datasets."""
    similarities = vectors @ query / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    )
    return np.argsort(similarities)[-k:][::-1].tolist()

# For large-scale: use FAISS, Annoy, ScaNN for approximate nearest neighbors
```

**2. Text Processing Algorithms:**
```python
# TF-IDF from scratch
from collections import Counter
import math

def compute_tf(document: list[str]) -> dict[str, float]:
    counts = Counter(document)
    total = len(document)
    return {word: count / total for word, count in counts.items()}

def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    n = len(documents)
    df = Counter()
    for doc in documents:
        df.update(set(doc))
    return {word: math.log(n / (count + 1)) for word, count in df.items()}

def compute_tfidf(document: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = compute_tf(document)
    return {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}

# BPE tokenization (simplified)
def byte_pair_encoding(text: str, num_merges: int = 100) -> list[str]:
    """Simplified BPE -- the algorithm behind GPT tokenization."""
    tokens = list(text)
    for _ in range(num_merges):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        if not pairs:
            break
        most_common = pairs.most_common(1)[0][0]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == most_common:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

**3. Graph Algorithms for Knowledge Graphs:**
```python
from collections import defaultdict, deque

class KnowledgeGraph:
    def __init__(self):
        self.graph: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add_edge(self, source: str, relation: str, target: str):
        self.graph[source].append((relation, target))

    def bfs_reachable(self, start: str, max_hops: int = 2) -> set[str]:
        """Find all entities reachable within max_hops -- useful for context expansion."""
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for relation, neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return visited

    def find_path(self, start: str, end: str) -> list[tuple[str, str, str]]:
        """Find reasoning path between entities."""
        visited = {start}
        queue = deque([(start, [])])
        while queue:
            node, path = queue.popleft()
            for relation, neighbor in self.graph[node]:
                if neighbor == end:
                    return path + [(node, relation, neighbor)]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(node, relation, neighbor)]))
        return []
```

**4. Caching with LRU:**
```python
from functools import lru_cache
from cachetools import TTLCache, LRUCache

# Built-in LRU cache
@lru_cache(maxsize=1024)
def get_embedding_cached(text: str) -> tuple[float, ...]:
    """Cache embeddings. Note: args must be hashable."""
    return tuple(compute_embedding(text))

# TTL cache for API responses
api_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL

def cached_llm_call(prompt: str) -> str:
    if prompt in api_cache:
        return api_cache[prompt]
    result = llm.complete(prompt)
    api_cache[prompt] = result
    return result
```
