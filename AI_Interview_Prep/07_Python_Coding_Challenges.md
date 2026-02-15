---
title: "Python Coding Challenges"
layout: default
parent: "Python & Coding"
nav_order: 4
render_with_liquid: false
---
{% raw %}

# Section 7: Python Coding Challenges for AI Engineer Roles

---

## 7.1 CHALLENGE: IMPLEMENT A RATE LIMITER FOR LLM API CALLS

**Difficulty: Medium | Topics: asyncio, data structures, time management**

```python
"""
Design a rate limiter that:
1. Limits requests to N per minute (sliding window)
2. Limits concurrent requests to M
3. Supports async operations
4. Has a queue for excess requests (with timeout)
"""

import asyncio
import time
from collections import deque
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")

class AsyncRateLimiter:
    def __init__(self, requests_per_minute: int, max_concurrent: int):
        self.rpm = requests_per_minute
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def _wait_for_slot(self):
        """Wait until we're under the rate limit."""
        while True:
            async with self._lock:
                now = time.monotonic()
                # Remove timestamps older than 60 seconds
                while self.request_times and self.request_times[0] < now - 60:
                    self.request_times.popleft()

                if len(self.request_times) < self.rpm:
                    self.request_times.append(now)
                    return

                # Calculate wait time
                wait_until = self.request_times[0] + 60
                wait_time = wait_until - now

            await asyncio.sleep(wait_time)

    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute a function with rate limiting."""
        await self._wait_for_slot()
        async with self.semaphore:
            return await func(*args, **kwargs)

# Usage
limiter = AsyncRateLimiter(requests_per_minute=60, max_concurrent=10)

async def call_llm(prompt: str) -> str:
    return await openai_client.complete(prompt)

# Rate-limited call
result = await limiter.execute(call_llm, "Hello")
```

**Follow-up questions:**
- How would you handle different rate limits for different API endpoints?
- How would you implement a token bucket algorithm instead?
- How would you distribute this across multiple worker processes?

---

## 7.2 CHALLENGE: IMPLEMENT A STREAMING TOKEN AGGREGATOR

**Difficulty: Medium | Topics: async generators, state management**

```python
"""
Implement a system that:
1. Receives streaming tokens from an LLM
2. Aggregates them into complete words/sentences
3. Detects JSON blocks and parses them
4. Handles incomplete streams (timeout, error)
"""

import asyncio
import json
import re
from typing import AsyncIterator
from dataclasses import dataclass, field

@dataclass
class StreamEvent:
    type: str  # "token", "word", "sentence", "json_block", "done", "error"
    content: str
    metadata: dict = field(default_factory=dict)

class StreamAggregator:
    def __init__(self):
        self.buffer = ""
        self.json_buffer = ""
        self.in_json = False
        self.brace_count = 0

    async def process_stream(
        self, token_stream: AsyncIterator[str]
    ) -> AsyncIterator[StreamEvent]:
        """Process a stream of tokens into higher-level events."""
        try:
            async for token in token_stream:
                self.buffer += token

                # Check for JSON block start
                if not self.in_json and "{" in token:
                    # Find where JSON starts in buffer
                    json_start = self.buffer.rfind("{")
                    pre_json = self.buffer[:json_start]
                    if pre_json.strip():
                        yield StreamEvent("token", pre_json)
                    self.json_buffer = self.buffer[json_start:]
                    self.brace_count = self.json_buffer.count("{") - self.json_buffer.count("}")
                    self.in_json = True
                    self.buffer = ""
                    continue

                if self.in_json:
                    self.json_buffer += token
                    self.brace_count += token.count("{") - token.count("}")

                    if self.brace_count == 0:
                        # Complete JSON block
                        try:
                            parsed = json.loads(self.json_buffer)
                            yield StreamEvent("json_block", self.json_buffer,
                                              {"parsed": parsed})
                        except json.JSONDecodeError:
                            yield StreamEvent("token", self.json_buffer)
                        self.json_buffer = ""
                        self.in_json = False
                    continue

                # Emit token events
                yield StreamEvent("token", token)

                # Check for sentence boundaries
                if any(token.endswith(p) for p in [".", "!", "?"]):
                    yield StreamEvent("sentence", self.buffer.strip())
                    self.buffer = ""

            # Flush remaining buffer
            remaining = self.buffer + self.json_buffer
            if remaining.strip():
                yield StreamEvent("token", remaining)

            yield StreamEvent("done", "")

        except asyncio.TimeoutError:
            yield StreamEvent("error", "Stream timed out",
                              {"partial": self.buffer})
        except Exception as e:
            yield StreamEvent("error", str(e),
                              {"partial": self.buffer})

# Usage
async def demo():
    aggregator = StreamAggregator()

    async def mock_stream():
        tokens = ["The answer is ", '{"result": ', '"42", "conf', 'idence": 0.95}', ". Done."]
        for t in tokens:
            yield t

    async for event in aggregator.process_stream(mock_stream()):
        if event.type == "json_block":
            print(f"Parsed JSON: {event.metadata['parsed']}")
        elif event.type == "sentence":
            print(f"Complete sentence: {event.content}")
        elif event.type == "token":
            print(f"Token: {event.content}")
```

---

## 7.3 CHALLENGE: IMPLEMENT A PROMPT TEMPLATE ENGINE

**Difficulty: Medium | Topics: string processing, validation, design patterns**

```python
"""
Build a prompt template engine that:
1. Supports variable substitution with {{variable}}
2. Supports conditional blocks {% if condition %}...{% endif %}
3. Supports loops {% for item in items %}...{% endfor %}
4. Validates all variables are provided
5. Tracks token count
"""

import re
from typing import Any
from dataclasses import dataclass

@dataclass
class RenderedPrompt:
    text: str
    variables_used: set[str]
    estimated_tokens: int

class PromptTemplate:
    VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")
    IF_PATTERN = re.compile(r"\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}", re.DOTALL)
    FOR_PATTERN = re.compile(
        r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}", re.DOTALL
    )

    def __init__(self, template: str):
        self.template = template
        self.required_variables = self._extract_variables()

    def _extract_variables(self) -> set[str]:
        """Extract all variable names from the template."""
        variables = set(self.VAR_PATTERN.findall(self.template))
        # Also extract from if/for conditions
        for match in self.IF_PATTERN.finditer(self.template):
            variables.add(match.group(1))
        for match in self.FOR_PATTERN.finditer(self.template):
            variables.add(match.group(2))
        return variables

    def render(self, **kwargs: Any) -> RenderedPrompt:
        """Render the template with given variables."""
        # Validate all required variables are provided
        missing = self.required_variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        text = self.template
        variables_used = set()

        # Process for loops first (they may contain variables and conditionals)
        def replace_for(match):
            item_name = match.group(1)
            collection_name = match.group(2)
            body = match.group(3)
            variables_used.add(collection_name)

            collection = kwargs.get(collection_name, [])
            if not isinstance(collection, (list, tuple)):
                raise TypeError(f"Variable '{collection_name}' must be iterable")

            result = []
            for item in collection:
                rendered_body = body.replace(f"{{{{{item_name}}}}}", str(item))
                # Handle item attributes: {{item.attr}}
                if isinstance(item, dict):
                    for key, value in item.items():
                        rendered_body = rendered_body.replace(
                            f"{{{{{item_name}.{key}}}}}", str(value)
                        )
                result.append(rendered_body)
            return "".join(result)

        text = self.FOR_PATTERN.sub(replace_for, text)

        # Process conditionals
        def replace_if(match):
            condition_var = match.group(1)
            body = match.group(2)
            variables_used.add(condition_var)
            if kwargs.get(condition_var):
                return body
            return ""

        text = self.IF_PATTERN.sub(replace_if, text)

        # Process simple variable substitution
        def replace_var(match):
            var_name = match.group(1)
            variables_used.add(var_name)
            value = kwargs.get(var_name, "")
            return str(value)

        text = self.VAR_PATTERN.sub(replace_var, text)

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return RenderedPrompt(
            text=text,
            variables_used=variables_used,
            estimated_tokens=len(text.split()) * 4 // 3,  # Rough estimate
        )

# Usage
template = PromptTemplate("""
You are a {{role}} assistant.

{% if use_context %}
Context:
{% for doc in documents %}
- {{doc.title}}: {{doc.content}}
{% endfor %}
{% endif %}

Question: {{question}}
Answer concisely.
""")

result = template.render(
    role="helpful",
    use_context=True,
    documents=[
        {"title": "Doc 1", "content": "Content 1"},
        {"title": "Doc 2", "content": "Content 2"},
    ],
    question="What is attention?",
)
print(result.text)
print(f"Estimated tokens: {result.estimated_tokens}")
```

---

## 7.4 CHALLENGE: IMPLEMENT A VECTOR SIMILARITY SEARCH ENGINE

**Difficulty: Medium-Hard | Topics: numpy, algorithms, data structures**

```python
"""
Implement a simple in-memory vector store that:
1. Stores vectors with metadata
2. Supports cosine and euclidean similarity search
3. Supports metadata filtering
4. Is efficient for moderate-sized datasets (up to 100K vectors)
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal

@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict

class SimpleVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.ids: list[str] = []
        self.metadata: list[dict] = []
        self.vectors: np.ndarray | None = None
        self._normalized: np.ndarray | None = None  # Cache for cosine similarity

    def upsert(self, ids: list[str], vectors: list[list[float]],
               metadata: list[dict] | None = None):
        """Add or update vectors."""
        new_vectors = np.array(vectors, dtype=np.float32)
        assert new_vectors.shape[1] == self.dimension

        if metadata is None:
            metadata = [{}] * len(ids)

        # Handle updates
        existing_idx = {id_: i for i, id_ in enumerate(self.ids)}
        new_ids, new_vecs, new_meta = [], [], []

        for id_, vec, meta in zip(ids, new_vectors, metadata):
            if id_ in existing_idx:
                idx = existing_idx[id_]
                self.vectors[idx] = vec
                self.metadata[idx] = meta
            else:
                new_ids.append(id_)
                new_vecs.append(vec)
                new_meta.append(meta)

        if new_ids:
            self.ids.extend(new_ids)
            self.metadata.extend(new_meta)
            new_array = np.array(new_vecs, dtype=np.float32)
            if self.vectors is None:
                self.vectors = new_array
            else:
                self.vectors = np.vstack([self.vectors, new_array])

        self._normalized = None  # Invalidate cache

    def _get_normalized(self) -> np.ndarray:
        """Cache normalized vectors for cosine similarity."""
        if self._normalized is None:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            self._normalized = self.vectors / norms
        return self._normalized

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        metric: Literal["cosine", "euclidean"] = "cosine",
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbors."""
        if self.vectors is None or len(self.ids) == 0:
            return []

        query_vec = np.array(vector, dtype=np.float32)

        # Apply metadata filter first
        if filter:
            mask = np.ones(len(self.ids), dtype=bool)
            for key, value in filter.items():
                for i, meta in enumerate(self.metadata):
                    if meta.get(key) != value:
                        mask[i] = False
            candidate_indices = np.where(mask)[0]
            if len(candidate_indices) == 0:
                return []
        else:
            candidate_indices = np.arange(len(self.ids))

        # Compute similarities
        if metric == "cosine":
            normalized = self._get_normalized()
            query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-10)
            scores = normalized[candidate_indices] @ query_norm
        elif metric == "euclidean":
            diffs = self.vectors[candidate_indices] - query_vec
            scores = -np.linalg.norm(diffs, axis=1)  # Negative distance (higher = closer)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Get top-k
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            original_idx = candidate_indices[idx]
            results.append(SearchResult(
                id=self.ids[original_idx],
                score=float(scores[idx]),
                metadata=self.metadata[original_idx],
            ))
        return results

    def delete(self, ids: list[str]):
        """Delete vectors by ID."""
        ids_to_delete = set(ids)
        keep_mask = [i for i, id_ in enumerate(self.ids) if id_ not in ids_to_delete]

        self.ids = [self.ids[i] for i in keep_mask]
        self.metadata = [self.metadata[i] for i in keep_mask]
        if self.vectors is not None:
            self.vectors = self.vectors[keep_mask]
        self._normalized = None

    def __len__(self) -> int:
        return len(self.ids)

# Usage
store = SimpleVectorStore(dimension=3)
store.upsert(
    ids=["doc1", "doc2", "doc3"],
    vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
    metadata=[
        {"category": "science"},
        {"category": "math"},
        {"category": "science"},
    ],
)

results = store.query(
    vector=[0.9, 0.1, 0.0],
    top_k=2,
    filter={"category": "science"},
)
```

---

## 7.5 CHALLENGE: IMPLEMENT A RETRY MECHANISM WITH EXPONENTIAL BACKOFF

**Difficulty: Easy-Medium | Topics: decorators, error handling, async**

```python
"""
Implement a production-quality retry decorator that:
1. Supports both sync and async functions
2. Uses exponential backoff with jitter
3. Has configurable retry conditions (which exceptions to retry)
4. Logs retry attempts
5. Has a maximum total timeout
"""

import asyncio
import functools
import logging
import random
import time
from typing import Callable, Type, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[Type[Exception], ...] = (Exception,),
    max_total_timeout: float | None = None,
):
    """Production retry decorator with exponential backoff and jitter."""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.monotonic()
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except retry_on as e:
                        last_exception = e
                        if attempt == max_retries:
                            logger.error(
                                f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                            )
                            raise

                        # Check total timeout
                        elapsed = time.monotonic() - start_time
                        if max_total_timeout and elapsed >= max_total_timeout:
                            logger.error(
                                f"{func.__name__} total timeout ({max_total_timeout}s) exceeded"
                            )
                            raise

                        # Calculate delay with exponential backoff
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        if jitter:
                            delay = delay * (0.5 + random.random())  # 50-150% of delay

                        # Don't wait longer than remaining timeout
                        if max_total_timeout:
                            remaining = max_total_timeout - elapsed
                            delay = min(delay, remaining)

                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {e}. Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)

                raise last_exception  # Should never reach here
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.monotonic()
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except retry_on as e:
                        last_exception = e
                        if attempt == max_retries:
                            raise

                        elapsed = time.monotonic() - start_time
                        if max_total_timeout and elapsed >= max_total_timeout:
                            raise

                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        if jitter:
                            delay = delay * (0.5 + random.random())

                        if max_total_timeout:
                            remaining = max_total_timeout - elapsed
                            delay = min(delay, remaining)

                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {e}. Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)

                raise last_exception
            return sync_wrapper
    return decorator

# Usage
@retry(
    max_retries=5,
    base_delay=1.0,
    retry_on=(RateLimitError, TimeoutError, ConnectionError),
    max_total_timeout=30.0,
)
async def call_llm(prompt: str) -> str:
    return await openai_client.chat(prompt)
```

---

## 7.6 CHALLENGE: IMPLEMENT A SIMPLE LRU CACHE WITH TTL

**Difficulty: Medium | Topics: data structures, threading, time**

```python
"""
Implement an LRU cache that:
1. Has a maximum size (evicts least recently used)
2. Has a TTL (time-to-live) for entries
3. Is thread-safe
4. Tracks hit/miss statistics
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, TypeVar, Generic, Hashable

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

@dataclass
class CacheEntry(Generic[V]):
    value: V
    expires_at: float

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class LRUTTLCache(Generic[K, V]):
    def __init__(self, maxsize: int = 1000, ttl_seconds: float = 3600):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: K) -> V | None:
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.monotonic() > entry.expires_at:
                del self._cache[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return entry.value

    def put(self, key: K, value: V, ttl: float | None = None):
        with self._lock:
            ttl = ttl or self.ttl
            entry = CacheEntry(
                value=value,
                expires_at=time.monotonic() + ttl,
            )

            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = entry
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)  # Remove LRU
                    self.stats.evictions += 1
                self._cache[key] = entry

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self):
        """Remove all expired entries. Call periodically."""
        with self._lock:
            now = time.monotonic()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry.expires_at
            ]
            for key in expired_keys:
                del self._cache[key]
                self.stats.expirations += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            return time.monotonic() <= self._cache[key].expires_at

# Usage
cache = LRUTTLCache[str, list[float]](maxsize=10000, ttl_seconds=3600)

def get_embedding_cached(text: str) -> list[float]:
    cached = cache.get(text)
    if cached is not None:
        return cached
    embedding = compute_embedding(text)
    cache.put(text, embedding)
    return embedding
```

---

## 7.7 CHALLENGE: IMPLEMENT A DOCUMENT CHUNKER FOR RAG

**Difficulty: Medium | Topics: string processing, algorithms, NLP**

```python
"""
Implement a document chunker that:
1. Splits text into chunks of approximately N tokens
2. Respects sentence boundaries (no mid-sentence splits)
3. Has configurable overlap between chunks
4. Preserves metadata (page numbers, section headers)
5. Handles multiple formats (plain text, markdown)
"""

import re
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict
    estimated_tokens: int

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 512,       # Target tokens per chunk
        chunk_overlap: int = 50,      # Overlap tokens between chunks
        min_chunk_size: int = 100,    # Minimum tokens per chunk
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token for English)."""
        return len(text) // 4

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, preserving the delimiter."""
        # Handle common abbreviations and edge cases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        """Split markdown text by headers, return (header, content) pairs."""
        sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)
        results = []
        current_header = ""
        for section in sections:
            if re.match(r'^#{1,6}\s+', section):
                current_header = section.strip()
            elif section.strip():
                results.append((current_header, section.strip()))
        if not results and text.strip():
            results = [("", text.strip())]
        return results

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Chunk plain text with sentence-aware splitting."""
        metadata = metadata or {}
        sentences = self._split_into_sentences(text)

        chunks = []
        current_sentences: list[str] = []
        current_tokens = 0
        current_start = 0
        char_pos = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata=metadata.copy(),
                    estimated_tokens=current_tokens,
                ))

                # Calculate overlap
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
                current_start = char_pos - sum(len(s) + 1 for s in overlap_sentences)

            current_sentences.append(sentence)
            current_tokens += sentence_tokens
            char_pos += len(sentence) + 1

        # Don't forget the last chunk
        if current_sentences and current_tokens >= self.min_chunk_size:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                index=len(chunks),
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                metadata=metadata.copy(),
                estimated_tokens=current_tokens,
            ))
        elif current_sentences and chunks:
            # Merge with previous chunk if too small
            prev = chunks[-1]
            merged_text = prev.text + " " + " ".join(current_sentences)
            chunks[-1] = Chunk(
                text=merged_text,
                index=prev.index,
                start_char=prev.start_char,
                end_char=prev.start_char + len(merged_text),
                metadata=prev.metadata,
                estimated_tokens=self._estimate_tokens(merged_text),
            )

        return chunks

    def chunk_markdown(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Chunk markdown, respecting section boundaries."""
        metadata = metadata or {}
        sections = self._split_by_headers(text)

        all_chunks = []
        for header, content in sections:
            section_meta = {**metadata, "section_header": header}
            section_chunks = self.chunk_text(content, metadata=section_meta)
            all_chunks.extend(section_chunks)

        # Re-index
        for i, chunk in enumerate(all_chunks):
            chunk.index = i

        return all_chunks

# Usage
chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_text(long_document, metadata={"source": "manual.pdf", "page": 5})
for chunk in chunks:
    print(f"Chunk {chunk.index}: {chunk.estimated_tokens} tokens")
    print(f"  {chunk.text[:100]}...")
```

---

## 7.8 CHALLENGE: IMPLEMENT A SIMPLE FUNCTION CALLING PARSER

**Difficulty: Medium | Topics: parsing, regex, JSON, error handling**

```python
"""
Parse LLM function call outputs in multiple formats:
1. JSON format: {"function": "search", "arguments": {"query": "..."}}
2. XML format: <function_call name="search"><arg name="query">...</arg></function_call>
3. Markdown format: ```tool_call\nsearch(query="...")\n```
Handle malformed outputs gracefully.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

@dataclass
class FunctionCall:
    name: str
    arguments: dict[str, Any]
    raw: str
    format_detected: str

class FunctionCallParser:
    def parse(self, text: str) -> list[FunctionCall]:
        """Try all formats, return parsed function calls."""
        results = []

        # Try JSON format
        results.extend(self._parse_json(text))
        if results:
            return results

        # Try XML format
        results.extend(self._parse_xml(text))
        if results:
            return results

        # Try markdown/code format
        results.extend(self._parse_markdown(text))
        return results

    def _parse_json(self, text: str) -> list[FunctionCall]:
        """Parse JSON function calls."""
        results = []
        # Find JSON objects in text
        json_pattern = re.compile(r'\{[^{}]*"function"[^{}]*\}', re.DOTALL)

        # Also try to find JSON in code blocks
        code_block_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
        for match in code_block_pattern.finditer(text):
            try:
                data = json.loads(match.group(1).strip())
                if isinstance(data, dict) and "function" in data:
                    results.append(FunctionCall(
                        name=data["function"],
                        arguments=data.get("arguments", data.get("args", {})),
                        raw=match.group(0),
                        format_detected="json_code_block",
                    ))
            except json.JSONDecodeError:
                continue

        if results:
            return results

        # Try finding JSON objects directly in text
        for match in json_pattern.finditer(text):
            try:
                data = json.loads(match.group())
                if "function" in data:
                    results.append(FunctionCall(
                        name=data["function"],
                        arguments=data.get("arguments", data.get("args", {})),
                        raw=match.group(),
                        format_detected="json_inline",
                    ))
            except json.JSONDecodeError:
                continue

        return results

    def _parse_xml(self, text: str) -> list[FunctionCall]:
        """Parse XML-style function calls."""
        results = []
        pattern = re.compile(
            r'<function_call\s+name="(\w+)">(.*?)</function_call>',
            re.DOTALL
        )
        arg_pattern = re.compile(r'<arg\s+name="(\w+)">(.*?)</arg>', re.DOTALL)

        for match in pattern.finditer(text):
            func_name = match.group(1)
            body = match.group(2)
            arguments = {}
            for arg_match in arg_pattern.finditer(body):
                arg_name = arg_match.group(1)
                arg_value = arg_match.group(2).strip()
                # Try to parse as JSON value
                try:
                    arguments[arg_name] = json.loads(arg_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[arg_name] = arg_value

            results.append(FunctionCall(
                name=func_name,
                arguments=arguments,
                raw=match.group(0),
                format_detected="xml",
            ))
        return results

    def _parse_markdown(self, text: str) -> list[FunctionCall]:
        """Parse function-call-style syntax in code blocks."""
        results = []
        pattern = re.compile(r'```tool_call\s*\n(\w+)\((.*?)\)\s*\n```', re.DOTALL)

        for match in pattern.finditer(text):
            func_name = match.group(1)
            args_str = match.group(2)

            # Parse keyword arguments
            arguments = {}
            arg_pattern = re.compile(r'(\w+)\s*=\s*(".*?"|\'.*?\'|\[.*?\]|\{.*?\}|[\w.]+)')
            for arg_match in arg_pattern.finditer(args_str):
                key = arg_match.group(1)
                value = arg_match.group(2).strip("\"'")
                try:
                    arguments[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = value

            results.append(FunctionCall(
                name=func_name,
                arguments=arguments,
                raw=match.group(0),
                format_detected="markdown",
            ))
        return results

# Usage
parser = FunctionCallParser()

# JSON format
result = parser.parse('I need to search. {"function": "search", "arguments": {"query": "Python"}}')

# XML format
result = parser.parse('<function_call name="search"><arg name="query">Python</arg></function_call>')

# Markdown format
result = parser.parse('```tool_call\nsearch(query="Python", limit=10)\n```')
```

---

## 7.9 CHALLENGE: IMPLEMENT A CONVERSATION MEMORY MANAGER

**Difficulty: Medium | Topics: data structures, token counting, algorithms**

```python
"""
Implement a conversation memory manager that:
1. Maintains a conversation history within a token budget
2. Supports multiple summarization strategies for old messages
3. Prioritizes recent messages and system prompts
4. Tracks token usage
"""

from dataclasses import dataclass, field
from typing import Literal
from abc import ABC, abstractmethod

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    tokens: int = 0
    pinned: bool = False  # Pinned messages are never evicted
    timestamp: float = 0.0

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = len(self.content) // 4  # Rough estimate

class EvictionStrategy(ABC):
    @abstractmethod
    def evict(self, messages: list[Message], target_tokens: int) -> list[Message]:
        """Remove messages until total tokens <= target_tokens."""
        ...

class FIFOEviction(EvictionStrategy):
    """Remove oldest non-pinned messages first."""
    def evict(self, messages: list[Message], target_tokens: int) -> list[Message]:
        total = sum(m.tokens for m in messages)
        result = list(messages)
        i = 0
        while total > target_tokens and i < len(result):
            if not result[i].pinned:
                total -= result[i].tokens
                result.pop(i)
            else:
                i += 1
        return result

class SlidingWindowEviction(EvictionStrategy):
    """Keep only the last N messages plus pinned ones."""
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def evict(self, messages: list[Message], target_tokens: int) -> list[Message]:
        pinned = [m for m in messages if m.pinned]
        unpinned = [m for m in messages if not m.pinned]

        # Keep last window_size unpinned messages
        kept = unpinned[-self.window_size:]

        # Merge pinned (at original positions) with kept
        result = pinned + kept
        result.sort(key=lambda m: m.timestamp)

        # Further trim if still over budget
        total = sum(m.tokens for m in result)
        while total > target_tokens and any(not m.pinned for m in result):
            for i, m in enumerate(result):
                if not m.pinned:
                    total -= m.tokens
                    result.pop(i)
                    break

        return result

class SummarizeEviction(EvictionStrategy):
    """Summarize old messages instead of removing them."""
    def __init__(self, summarizer):
        self.summarizer = summarizer  # A function that summarizes text

    def evict(self, messages: list[Message], target_tokens: int) -> list[Message]:
        total = sum(m.tokens for m in messages)
        if total <= target_tokens:
            return messages

        pinned = [m for m in messages if m.pinned]
        unpinned = [m for m in messages if not m.pinned]

        # Split into old and recent halves
        midpoint = len(unpinned) // 2
        old_messages = unpinned[:midpoint]
        recent_messages = unpinned[midpoint:]

        # Summarize old messages
        old_text = "\n".join(f"{m.role}: {m.content}" for m in old_messages)
        summary = self.summarizer(old_text)

        summary_message = Message(
            role="system",
            content=f"[Summary of earlier conversation]: {summary}",
            pinned=False,
            timestamp=old_messages[0].timestamp if old_messages else 0,
        )

        result = pinned + [summary_message] + recent_messages
        result.sort(key=lambda m: m.timestamp)
        return result

class ConversationMemory:
    def __init__(
        self,
        max_tokens: int = 4096,
        strategy: EvictionStrategy | None = None,
    ):
        self.max_tokens = max_tokens
        self.strategy = strategy or FIFOEviction()
        self.messages: list[Message] = []
        self._message_counter = 0.0

    def add(self, role: str, content: str, pinned: bool = False):
        self._message_counter += 1
        message = Message(
            role=role,
            content=content,
            pinned=pinned,
            timestamp=self._message_counter,
        )
        self.messages.append(message)
        self._enforce_budget()

    def add_system(self, content: str):
        """System messages are always pinned."""
        self.add("system", content, pinned=True)

    def _enforce_budget(self):
        total = sum(m.tokens for m in self.messages)
        if total > self.max_tokens:
            self.messages = self.strategy.evict(self.messages, self.max_tokens)

    def get_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages)

    @property
    def message_count(self) -> int:
        return len(self.messages)

# Usage
memory = ConversationMemory(
    max_tokens=4096,
    strategy=SlidingWindowEviction(window_size=10),
)

memory.add_system("You are a helpful assistant.")
memory.add("user", "What is machine learning?")
memory.add("assistant", "Machine learning is...")
# ... many more messages
# Old messages are automatically evicted to stay within budget
```

---

## 7.10 QUICK-FIRE CODING QUESTIONS

These are shorter questions that test fundamental Python skills:

```python
# 1. Flatten a nested dictionary (common for config management)
def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
# -> {"a.b": 1, "a.c.d": 2}

# 2. Merge two sorted lists of search results by score
def merge_sorted_results(list1: list[dict], list2: list[dict]) -> list[dict]:
    """Merge two sorted lists maintaining sort order. O(n+m)."""
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i]["score"] >= list2[j]["score"]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result

# 3. Implement a simple bloom filter for deduplication
class SimpleBloomFilter:
    def __init__(self, size: int = 10000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size

    def _hashes(self, item: str) -> list[int]:
        hashes = []
        for i in range(self.num_hashes):
            h = hash(f"{item}_{i}") % self.size
            hashes.append(h)
        return hashes

    def add(self, item: str):
        for h in self._hashes(item):
            self.bit_array[h] = True

    def might_contain(self, item: str) -> bool:
        return all(self.bit_array[h] for h in self._hashes(item))

# 4. Find the most frequent n-grams in a text (useful for prompt analysis)
from collections import Counter

def top_ngrams(text: str, n: int = 2, top_k: int = 10) -> list[tuple[tuple[str, ...], int]]:
    words = text.lower().split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return Counter(ngrams).most_common(top_k)

# 5. Implement exponential moving average for monitoring metrics
class EMA:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

# 6. Deduplicate while preserving order
def deduplicate_ordered(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        key = item if isinstance(item, str) else str(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
```
{% endraw %}
