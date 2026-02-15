---
title: "Testing AI Systems"
layout: default
parent: "DevOps & Cloud Infrastructure"
nav_order: 11
render_with_liquid: false
---
{% raw %}

# Section 4: Testing AI Systems -- pytest, Mocking, Pipelines, Snapshots

---

## 4.1 PYTEST FUNDAMENTALS FOR AI ENGINEERS

### Q: How do you structure tests for an AI application?

**Answer:**
```
project/
  src/
    llm/
      client.py
      prompts.py
    pipeline/
      rag.py
      preprocessing.py
    models/
      config.py
  tests/
    unit/
      test_prompts.py
      test_preprocessing.py
      test_config.py
    integration/
      test_rag_pipeline.py
      test_llm_client.py
    e2e/
      test_api_endpoints.py
    fixtures/
      sample_documents.json
      expected_outputs.json
    snapshots/
      test_prompts/
    conftest.py
```

```python
# === conftest.py -- shared fixtures ===
import pytest
from unittest.mock import AsyncMock, MagicMock
import json

@pytest.fixture
def sample_documents():
    return [
        {"id": "1", "text": "Transformers use self-attention mechanisms.", "score": 0.95},
        {"id": "2", "text": "BERT is a bidirectional encoder.", "score": 0.87},
    ]

@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns predictable responses."""
    client = AsyncMock()
    client.complete.return_value = "This is a mock LLM response."
    client.stream.return_value = async_generator(["This ", "is ", "streamed."])
    return client

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model with deterministic outputs."""
    model = MagicMock()
    model.embed.return_value = [[0.1, 0.2, 0.3] * 512]  # 1536-dim
    model.dimension = 1536
    return model

# Async generator helper
async def async_generator(items):
    for item in items:
        yield item

# Session-scoped fixture for expensive setup
@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("bert-base-uncased")

# Fixture with cleanup
@pytest.fixture
async def vector_store():
    store = await create_test_vector_store()
    yield store
    await store.cleanup()
```

### Q: How do you write effective pytest tests for AI components?

**Answer:**
```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call
import numpy as np

# === Basic test patterns ===

class TestPromptBuilder:
    def test_system_message_added(self):
        builder = PromptBuilder()
        result = builder.system("You are helpful.").build()
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}

    def test_conversation_history_ordering(self):
        builder = PromptBuilder()
        result = (
            builder
            .system("System prompt")
            .user("Hello")
            .assistant("Hi there!")
            .user("How are you?")
            .build()
        )
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant", "user"]

    def test_empty_prompt_raises(self):
        builder = PromptBuilder()
        with pytest.raises(ValueError, match="At least one message required"):
            builder.build()

# === Parametrized tests ===
class TestTokenCounter:
    @pytest.mark.parametrize("text,expected_range", [
        ("Hello world", (2, 4)),
        ("", (0, 0)),
        ("A " * 1000, (900, 1100)),
        ("Special chars: @#$%^&*()", (5, 15)),
    ])
    def test_token_count_ranges(self, text, expected_range):
        """Token counts may vary by tokenizer, so test ranges."""
        count = count_tokens(text)
        assert expected_range[0] <= count <= expected_range[1]

# === Async tests ===
class TestAsyncLLMClient:
    @pytest.mark.asyncio
    async def test_complete_returns_response(self, mock_llm_client):
        response = await mock_llm_client.complete("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, mock_llm_client):
        tokens = []
        async for token in mock_llm_client.stream("Test prompt"):
            tokens.append(token)
        assert len(tokens) > 0
        assert "".join(tokens) == "This is streamed."

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Verify semaphore limits concurrent calls."""
        call_times = []

        async def slow_complete(prompt):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)
            return "response"

        client = AsyncLLMClient(max_concurrent=2)
        client._complete = slow_complete

        # 4 calls with max 2 concurrent
        await asyncio.gather(*[client.complete(f"prompt {i}") for i in range(4)])

        # First 2 should start nearly simultaneously, next 2 after ~0.1s
        assert call_times[2] - call_times[0] >= 0.09

# === Testing with fixtures and markers ===
@pytest.mark.slow
@pytest.mark.integration
class TestRAGPipeline:
    @pytest.fixture
    def pipeline(self, mock_llm_client, mock_embedding_model):
        return RAGPipeline(
            llm_client=mock_llm_client,
            embedding_model=mock_embedding_model,
            vector_store=InMemoryVectorStore(),
        )

    def test_retrieval_returns_relevant_docs(self, pipeline, sample_documents):
        pipeline.ingest(sample_documents)
        results = pipeline.retrieve("attention mechanism")
        assert len(results) > 0
        assert any("attention" in doc["text"].lower() for doc in results)

    def test_end_to_end_query(self, pipeline, sample_documents):
        pipeline.ingest(sample_documents)
        response = pipeline.query("What is self-attention?")
        assert isinstance(response, str)
```

---

## 4.2 MOCKING LLM CALLS

### Q: What are the best practices for mocking LLM API calls in tests?

**Answer:**
```python
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock
import pytest

# === Strategy 1: Mock at the client level ===
class TestWithMockedLLM:
    @patch("myapp.llm.client.OpenAIClient.complete")
    def test_extraction_pipeline(self, mock_complete):
        mock_complete.return_value = json.dumps({
            "entities": [{"name": "Python", "type": "language"}]
        })

        result = extraction_pipeline("Python is a programming language.")
        assert len(result.entities) == 1
        assert result.entities[0].name == "Python"

        # Verify the prompt was constructed correctly
        called_prompt = mock_complete.call_args[0][0]
        assert "Python is a programming language" in called_prompt

    @patch("myapp.llm.client.OpenAIClient.complete")
    def test_handles_malformed_llm_output(self, mock_complete):
        """LLMs sometimes return invalid JSON."""
        mock_complete.return_value = "Not valid JSON {{"

        with pytest.raises(LLMOutputParsingError):
            extraction_pipeline("Some text")

    @patch("myapp.llm.client.OpenAIClient.complete")
    def test_retry_on_rate_limit(self, mock_complete):
        mock_complete.side_effect = [
            RateLimitError("Rate limited"),
            RateLimitError("Rate limited"),
            "Valid response",  # Third attempt succeeds
        ]

        result = call_with_retry("Test prompt")
        assert result == "Valid response"
        assert mock_complete.call_count == 3

# === Strategy 2: Dependency injection with fake client ===
class FakeLLMClient:
    """Deterministic fake for testing -- more maintainable than mocks."""
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []  # Record all calls for assertions
        self.default_response = "Default fake response"

    async def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        # Match response based on keywords in prompt
        for keyword, response in self.responses.items():
            if keyword in prompt:
                return response
        return self.default_response

    async def stream(self, prompt: str, **kwargs):
        response = await self.complete(prompt, **kwargs)
        for word in response.split():
            yield word + " "

class TestWithFakeLLM:
    @pytest.fixture
    def fake_llm(self):
        return FakeLLMClient(responses={
            "summarize": "This is a summary of the document.",
            "translate": "Ceci est une traduction.",
            "extract": '{"entities": []}',
        })

    @pytest.mark.asyncio
    async def test_summarization(self, fake_llm):
        pipeline = SummarizationPipeline(llm=fake_llm)
        result = await pipeline.run("Please summarize this document...")
        assert "summary" in result.lower()
        assert len(fake_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_multi_step_agent(self, fake_llm):
        agent = Agent(llm=fake_llm)
        await agent.run("Research and summarize topic X")
        # Verify the agent made multiple LLM calls
        assert len(fake_llm.calls) >= 2
        # Verify the sequence of calls makes sense
        assert "research" in fake_llm.calls[0]["prompt"].lower() or \
               "search" in fake_llm.calls[0]["prompt"].lower()

# === Strategy 3: Response fixtures from files ===
@pytest.fixture
def llm_response_fixtures():
    """Load pre-recorded LLM responses."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "llm_responses"
    responses = {}
    for file in fixtures_dir.glob("*.json"):
        responses[file.stem] = json.loads(file.read_text())
    return responses

# === Strategy 4: Mock streaming responses ===
class TestStreaming:
    @pytest.mark.asyncio
    async def test_streaming_aggregation(self):
        chunks = ["Hello", " ", "world", "!"]

        async def mock_stream(prompt):
            for chunk in chunks:
                yield chunk

        with patch.object(llm_client, "stream", side_effect=mock_stream):
            result = await aggregate_stream(llm_client, "test prompt")
            assert result == "Hello world!"
```

---

## 4.3 TESTING AI PIPELINES

### Q: How do you test complex multi-step AI pipelines?

**Answer:**
```python
# === Testing RAG Pipeline ===

class TestRAGPipeline:
    """Test each stage independently and then integration."""

    # --- Unit tests for each stage ---
    def test_query_rewriting(self):
        rewriter = QueryRewriter()
        result = rewriter.rewrite("what is attention")
        assert isinstance(result, str)
        assert len(result) > len("what is attention")  # Should be expanded

    def test_document_retrieval_relevance(self, mock_vector_store):
        retriever = Retriever(vector_store=mock_vector_store)
        docs = retriever.retrieve("transformer architecture", top_k=5)
        assert len(docs) <= 5
        assert all(doc.score >= 0.0 for doc in docs)
        # Verify docs are sorted by relevance
        scores = [doc.score for doc in docs]
        assert scores == sorted(scores, reverse=True)

    def test_context_window_fitting(self):
        """Verify context doesn't exceed model's token limit."""
        formatter = ContextFormatter(max_tokens=4096)
        long_docs = [{"text": "x " * 2000}] * 10  # Way too long

        context = formatter.format(long_docs)
        token_count = count_tokens(context)
        assert token_count <= 4096

    def test_answer_grounding(self, fake_llm):
        """Verify answer references source documents."""
        generator = AnswerGenerator(llm=fake_llm)
        docs = [
            {"text": "The Eiffel Tower is 330 meters tall.", "source": "wiki"},
        ]
        answer = generator.generate("How tall is the Eiffel Tower?", docs)
        # In a real test, you might check for citation markers
        assert isinstance(answer, str)

    # --- Integration test ---
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_pipeline(self, fake_llm, in_memory_vector_store):
        pipeline = RAGPipeline(
            llm=fake_llm,
            vector_store=in_memory_vector_store,
            reranker=SimpleReranker(),
        )

        # Ingest test documents
        await pipeline.ingest([
            {"id": "1", "text": "Python is a programming language."},
            {"id": "2", "text": "JavaScript runs in browsers."},
        ])

        # Query
        result = await pipeline.query("What is Python?")

        assert isinstance(result.answer, str)
        assert len(result.source_documents) > 0
        assert result.source_documents[0]["id"] == "1"  # Most relevant

# === Testing Agent/Tool-use Systems ===

class TestAgent:
    @pytest.fixture
    def agent(self, fake_llm):
        return Agent(
            llm=fake_llm,
            tools=[
                SearchTool(mock_search_engine()),
                CalculatorTool(),
                CodeExecutor(sandbox=True),
            ],
            max_iterations=5,
        )

    @pytest.mark.asyncio
    async def test_agent_uses_correct_tool(self, agent):
        result = await agent.run("What is 2 + 2?")
        # Verify calculator tool was used
        assert any(
            step.tool_name == "calculator"
            for step in result.steps
        )

    @pytest.mark.asyncio
    async def test_agent_respects_max_iterations(self, agent):
        """Agent should stop after max_iterations to prevent infinite loops."""
        result = await agent.run("An impossible task that requires many steps")
        assert len(result.steps) <= 5

    @pytest.mark.asyncio
    async def test_agent_handles_tool_failure(self, agent):
        """Agent should gracefully handle tool errors."""
        with patch.object(agent.tools[0], "execute", side_effect=ToolError("API down")):
            result = await agent.run("Search for something")
            # Agent should either retry or report the error gracefully
            assert result.status in ("completed_with_errors", "completed")

# === Testing Data Pipelines ===

class TestDataPipeline:
    def test_preprocessing_idempotent(self):
        """Running preprocessing twice should give same result."""
        raw_data = load_test_data()
        result1 = preprocess(raw_data)
        result2 = preprocess(raw_data)
        assert result1 == result2

    def test_preprocessing_handles_edge_cases(self):
        edge_cases = [
            "",                           # Empty string
            " " * 100,                    # Only whitespace
            "\x00\x01\x02",              # Binary data
            "a" * 1_000_000,             # Very long string
            "Hello\nWorld\n\n\n",        # Multiple newlines
            "<script>alert('xss')</script>",  # HTML/XSS
        ]
        for case in edge_cases:
            result = preprocess(case)
            assert isinstance(result, str)
            assert "\x00" not in result  # No null bytes

    def test_batch_processing_matches_individual(self):
        """Batch processing should give same results as individual processing."""
        items = [generate_test_item() for _ in range(100)]
        individual_results = [process_single(item) for item in items]
        batch_results = process_batch(items)
        assert individual_results == batch_results
```

---

## 4.4 SNAPSHOT TESTING FOR PROMPTS

### Q: How do you use snapshot testing for LLM prompts?

**Answer:**
Snapshot testing captures the expected output and compares future runs against it. This is critical for prompts because unintended prompt changes can drastically alter LLM behavior.

```python
# Using pytest-snapshot or syrupy
# pip install syrupy

from syrupy.assertion import SnapshotAssertion

class TestPromptTemplates:
    def test_system_prompt_snapshot(self, snapshot: SnapshotAssertion):
        """Ensures system prompt doesn't change accidentally."""
        prompt = build_system_prompt(
            role="helpful assistant",
            capabilities=["search", "calculate"],
            constraints=["be concise", "cite sources"],
        )
        assert prompt == snapshot

    def test_rag_prompt_snapshot(self, snapshot: SnapshotAssertion):
        """Ensures RAG prompt template is stable."""
        prompt = build_rag_prompt(
            query="What is attention?",
            context_docs=[
                {"text": "Attention is a mechanism...", "source": "paper.pdf"},
            ],
            instructions="Answer concisely.",
        )
        assert prompt == snapshot

    def test_tool_use_prompt_snapshot(self, snapshot: SnapshotAssertion):
        tools = [
            {"name": "search", "description": "Search the web"},
            {"name": "calculate", "description": "Perform math"},
        ]
        prompt = build_tool_use_prompt(tools=tools, query="What is 2+2?")
        assert prompt == snapshot

    @pytest.mark.parametrize("scenario", [
        "simple_question",
        "multi_turn_conversation",
        "with_system_prompt",
        "with_tools",
        "with_images",
    ])
    def test_prompt_scenarios_snapshot(self, scenario, snapshot: SnapshotAssertion):
        prompt = build_prompt_for_scenario(scenario)
        assert prompt == snapshot

# === Custom snapshot strategy for prompts ===
# You can also use inline snapshots or custom comparison

class TestPromptRegression:
    """Manual snapshot testing without framework."""

    SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"

    def _load_snapshot(self, name: str) -> str | None:
        path = self.SNAPSHOTS_DIR / f"{name}.txt"
        if path.exists():
            return path.read_text()
        return None

    def _save_snapshot(self, name: str, content: str):
        self.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        (self.SNAPSHOTS_DIR / f"{name}.txt").write_text(content)

    def test_prompt_stability(self):
        prompt = build_main_prompt(query="test", context="test context")

        snapshot = self._load_snapshot("main_prompt")
        if snapshot is None:
            self._save_snapshot("main_prompt", prompt)
            pytest.skip("Snapshot created, run again to verify")
        else:
            assert prompt == snapshot, (
                "Prompt has changed! If this is intentional, delete the snapshot "
                "file and re-run to update it."
            )

# === Testing prompt variations don't break parsing ===
class TestPromptParsing:
    """Ensure LLM output can be parsed regardless of prompt variations."""

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_output_parseable_at_different_temperatures(self, temperature, fake_llm):
        """Different temperatures shouldn't break output parsing."""
        fake_llm.responses["extract"] = '{"entities": [{"name": "Test", "type": "org"}]}'
        result = extraction_pipeline(
            "Test Corp is a company.",
            llm=fake_llm,
            temperature=temperature,
        )
        assert isinstance(result, ExtractionResult)

    def test_output_parsing_with_markdown_wrapping(self):
        """LLMs sometimes wrap JSON in markdown code blocks."""
        raw_outputs = [
            '{"key": "value"}',
            '```json\n{"key": "value"}\n```',
            '```\n{"key": "value"}\n```',
            'Here is the result:\n```json\n{"key": "value"}\n```\n',
        ]
        for raw in raw_outputs:
            result = parse_llm_json(raw)
            assert result == {"key": "value"}
```

---

## 4.5 ADVANCED TESTING PATTERNS

### Q: What advanced testing patterns are important for AI systems?

**Answer:**

```python
# === 1. Property-based testing with Hypothesis ===
from hypothesis import given, strategies as st, settings

@given(
    text=st.text(min_size=0, max_size=10000),
    max_tokens=st.integers(min_value=1, max_value=4096),
)
@settings(max_examples=200)
def test_tokenizer_never_exceeds_max(text, max_tokens):
    """Property: truncation should ALWAYS respect max_tokens."""
    result = truncate_to_tokens(text, max_tokens)
    assert count_tokens(result) <= max_tokens

@given(
    embeddings=st.lists(
        st.lists(st.floats(min_value=-1, max_value=1), min_size=3, max_size=3),
        min_size=1,
        max_size=100,
    )
)
def test_cosine_similarity_bounds(embeddings):
    """Property: cosine similarity is always in [-1, 1]."""
    import numpy as np
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            a = np.array(embeddings[i])
            b = np.array(embeddings[j])
            if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                sim = cosine_similarity(a, b)
                assert -1.0 <= sim <= 1.0

# === 2. Approval testing for LLM outputs ===
class TestLLMOutputApproval:
    """For outputs that are hard to assert on, use approval testing."""

    def test_summary_quality(self, fake_llm):
        summary = summarize("Long document text...", llm=fake_llm)

        # Structural assertions
        assert len(summary) > 0
        assert len(summary) < len("Long document text...")
        assert summary.endswith(".")

        # Quality assertions (heuristics)
        assert not summary.startswith("I ")  # Should not be first-person
        assert "```" not in summary  # Should not contain code blocks

# === 3. Deterministic tests with seed control ===
class TestWithSeeds:
    def test_sampling_reproducible(self):
        import random
        random.seed(42)
        result1 = stochastic_function()
        random.seed(42)
        result2 = stochastic_function()
        assert result1 == result2

# === 4. Performance regression tests ===
class TestPerformance:
    @pytest.mark.benchmark
    def test_embedding_latency(self, benchmark):
        """Ensure embedding computation stays under threshold."""
        result = benchmark(compute_embedding, "Test text for embedding")
        assert benchmark.stats["mean"] < 0.1  # Under 100ms

    def test_memory_usage(self):
        import tracemalloc
        tracemalloc.start()
        process_large_batch(generate_test_data(10_000))
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 500 * 1024 * 1024  # Under 500MB

# === 5. Contract testing for APIs ===
class TestAPIContract:
    """Verify API responses match expected schema."""

    def test_completion_response_schema(self):
        response = call_llm_api("test")
        # Validate with Pydantic
        validated = CompletionResponse.model_validate(response)
        assert validated.choices[0].message.content is not None
        assert validated.usage.total_tokens > 0
```

---

## 4.6 PYTEST CONFIGURATION AND BEST PRACTICES

```python
# === pytest.ini or pyproject.toml ===
# [tool.pytest.ini_options]
# testpaths = ["tests"]
# asyncio_mode = "auto"
# markers = [
#     "slow: marks tests as slow (deselect with '-m \"not slow\"')",
#     "integration: integration tests that need external services",
#     "e2e: end-to-end tests",
#     "benchmark: performance benchmark tests",
# ]
# filterwarnings = [
#     "ignore::DeprecationWarning",
# ]
# addopts = "-v --tb=short --strict-markers"

# === Useful pytest plugins for AI projects ===
# pytest-asyncio: async test support
# pytest-mock: enhanced mocking
# pytest-benchmark: performance benchmarks
# pytest-timeout: prevent hanging tests
# pytest-xdist: parallel test execution
# syrupy: snapshot testing
# hypothesis: property-based testing
# pytest-cov: coverage reporting
# pytest-env: environment variables for tests
# pytest-recording: VCR-style HTTP recording for API tests
```
{% endraw %}
