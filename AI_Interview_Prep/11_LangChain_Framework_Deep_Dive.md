---
title: "LangChain Framework"
layout: default
parent: "AI Frameworks & Agents"
nav_order: 2
---

# LangChain Framework: Complete Interview Guide (2025-2026)
## For AI Engineers Building Production LLM Applications

---

# TABLE OF CONTENTS
1. What is LangChain?
2. Explaining to a Layman
3. Core Concepts (Chains, Models, Prompts, Parsers)
4. LCEL (LangChain Expression Language)
5. Document Loaders & Text Splitters
6. Retrieval & Vector Stores
7. Agents & Tools
8. Memory Systems
9. LangSmith (Tracing & Evaluation)
10. LangChain vs LlamaIndex vs Haystack
11. Interview Questions (25+)
12. Code Examples
13. Production Best Practices

---

# SECTION 1: WHAT IS LANGCHAIN?

## 1.1 Definition

LangChain is an open-source framework for building applications powered by LLMs. It provides:
- **Abstractions** for common LLM patterns (RAG, agents, chains)
- **Integrations** with 700+ providers (LLMs, vector stores, tools)
- **Composability** via LCEL (LangChain Expression Language)

## 1.2 LangChain Ecosystem (2025)

```
┌─────────────────────────────────────────────────────────────┐
│                    LANGCHAIN ECOSYSTEM                        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ langchain-   │  │ langchain-   │  │ langchain-   │      │
│  │ core         │  │ community    │  │ openai/      │      │
│  │              │  │              │  │ anthropic/   │      │
│  │ • Runnables  │  │ • 700+       │  │ google/etc   │      │
│  │ • LCEL       │  │   integrations│  │              │      │
│  │ • Prompts    │  │ • Vector     │  │ • First-party│      │
│  │ • Messages   │  │   stores     │  │   integrations│     │
│  │ • Tools      │  │ • Loaders    │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │  LangGraph   │  │  LangSmith   │                         │
│  │              │  │              │                         │
│  │ • Stateful   │  │ • Tracing    │                         │
│  │   graphs     │  │ • Evaluation │                         │
│  │ • Multi-agent│  │ • Monitoring │                         │
│  │ • HITL       │  │ • Datasets   │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

> 🔵 **YOUR EXPERIENCE**: At Stellantis (MathCo), you built a LangChain-powered conversational AI chatbot for natural language querying and real-time insights.

---

# SECTION 2: EXPLAINING TO A LAYMAN

> LangChain is like a toolbox for building AI-powered applications. Just like a carpenter has a toolbox with hammers, saws, and drills, LangChain gives developers pre-built tools for connecting AI to documents, databases, and APIs. Instead of building everything from scratch, you pick the pieces you need and snap them together.

---

# SECTION 3: CORE CONCEPTS

## 3.1 Chat Models

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Anthropic
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

# Invoke
response = llm.invoke("What is RAG?")
print(response.content)
```

## 3.2 Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

# Simple
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")

# With system message
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor specializing in {subject}."),
    ("human", "{question}")
])

# Create messages
messages = prompt.invoke({"subject": "machine learning", "question": "What is RAG?"})
```

## 3.3 Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# String parser
parser = StrOutputParser()

# JSON parser with Pydantic model
class Answer(BaseModel):
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="Source references")

json_parser = JsonOutputParser(pydantic_object=Answer)
```

## 3.4 Chains (Composition)

```python
# LCEL chain: prompt | model | parser
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"subject": "AI", "question": "What is RAG?"})
```

---

# SECTION 4: LCEL (LANGCHAIN EXPRESSION LANGUAGE)

## 4.1 What is LCEL?

LCEL is LangChain's declarative way to compose chains using the **pipe operator** (`|`). Every component implements the `Runnable` interface.

## 4.2 The Runnable Interface

Every LCEL component has these methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| `invoke(input)` | Single input, single output | Standard call |
| `batch([inputs])` | Multiple inputs, parallel | Batch processing |
| `stream(input)` | Stream output chunks | Real-time chat |
| `ainvoke(input)` | Async invoke | FastAPI endpoints |
| `abatch([inputs])` | Async batch | High throughput |
| `astream(input)` | Async stream | WebSocket |

## 4.3 LCEL Patterns

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

# Basic chain
chain = prompt | llm | parser

# Parallel execution
chain = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
) | prompt | llm | parser

# Custom function in chain
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Branching (router)
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "code" in x["topic"], code_chain),
    (lambda x: "math" in x["topic"], math_chain),
    default_chain  # fallback
)

# With fallbacks
reliable_chain = primary_llm.with_fallbacks([backup_llm_1, backup_llm_2])

# With retry
chain = llm.with_retry(stop_after_attempt=3, wait_exponential_multiplier=1)
```

## 4.4 Streaming with LCEL

```python
# Stream tokens
for chunk in chain.stream({"question": "What is RAG?"}):
    print(chunk, end="", flush=True)

# Async streaming (for FastAPI)
async for chunk in chain.astream({"question": "What is RAG?"}):
    yield chunk
```

---

# SECTION 5: DOCUMENT LOADERS & TEXT SPLITTERS

## 5.1 Common Loaders

| Loader | Source | Example |
|--------|--------|---------|
| `PyPDFLoader` | PDF files | `PyPDFLoader("doc.pdf")` |
| `WebBaseLoader` | Web pages | `WebBaseLoader("https://...")` |
| `CSVLoader` | CSV files | `CSVLoader("data.csv")` |
| `DirectoryLoader` | Folder of files | `DirectoryLoader("./docs/")` |
| `UnstructuredLoader` | Any document | Complex PDFs, Word docs |
| `NotionDBLoader` | Notion | Notion databases |
| `ConfluenceLoader` | Confluence | Wiki pages |
| `S3FileLoader` | AWS S3 | Cloud storage |

## 5.2 Text Splitters

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Most common - recursive splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# For code
from langchain.text_splitter import Language
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)
```

---

# SECTION 6: RETRIEVAL & VECTOR STORES

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Pinecone

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# As retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # or "similarity"
    search_kwargs={"k": 5, "fetch_k": 20}
)

# Hybrid retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs, k=5)
vector = vectorstore.as_retriever(search_kwargs={"k": 5})
hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.4, 0.6])
```

---

# SECTION 7: AGENTS & TOOLS

## 7.1 Creating Tools

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return tavily_client.search(query)

@tool
def calculate(expression: str) -> float:
    """Calculate a math expression."""
    return eval(expression)  # Use safe_eval in production!
```

## 7.2 ReAct Agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search_web, calculate],
    state_modifier="You are a helpful research assistant."
)

result = agent.invoke({"messages": [HumanMessage(content="What is 2+2?")]})
```

## 7.3 Tool Calling Agent (Recommended for 2025+)

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "Search for LangChain latest features"})
```

---

# SECTION 8: MEMORY SYSTEMS

| Memory Type | Description | Use Case |
|-------------|-------------|----------|
| `ConversationBufferMemory` | Stores all messages | Short conversations |
| `ConversationSummaryMemory` | Summarizes older messages | Long conversations |
| `ConversationBufferWindowMemory` | Last K messages | Fixed context budget |
| `VectorStoreRetrieverMemory` | Embeds & retrieves relevant memories | Long-term recall |

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=2000,
    return_messages=True
)
```

**Note (2025):** For new projects, use **LangGraph's built-in persistence** (checkpointers) instead of LangChain memory classes. LangGraph state management is the recommended approach.

---

# SECTION 9: LANGSMITH

## 9.1 What is LangSmith?

LangSmith is LangChain's platform for **tracing, evaluating, and monitoring** LLM applications.

| Feature | Description |
|---------|-------------|
| **Tracing** | See every step of chain execution with latencies and token counts |
| **Evaluation** | Run datasets against your chain, score with custom evaluators |
| **Monitoring** | Production dashboards, error alerts, cost tracking |
| **Datasets** | Create test datasets, track regression |
| **Playground** | Test prompts interactively |

```python
# Enable tracing (just set env vars)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All chain invocations are now traced automatically
```

---

# SECTION 10: LANGCHAIN vs LLAMAINDEX vs HAYSTACK

| Feature | LangChain | LlamaIndex | Haystack |
|---------|-----------|------------|----------|
| **Focus** | General LLM app framework | Data framework for LLMs | NLP/RAG pipelines |
| **Strength** | Breadth of integrations, LCEL | Index structures, RAG | Production RAG |
| **Agents** | Strong (via LangGraph) | Basic | Limited |
| **RAG** | Good | Excellent (specialized) | Excellent |
| **Multi-Agent** | LangGraph (best in class) | Limited | Limited |
| **Learning Curve** | Medium | Medium | Low-Medium |
| **Community** | Largest | Large | Growing |
| **Monitoring** | LangSmith | LlamaTrace | Haystack Studio |
| **Best For** | Full-stack AI apps | RAG-focused apps | Production NLP |

---

# SECTION 11: INTERVIEW QUESTIONS (25+)

**Q1: What is LangChain and what problem does it solve?**
LangChain is a framework for building LLM applications that provides abstractions, integrations, and composability. Solves: boilerplate code, provider switching, common patterns (RAG, agents, chains).

**Q2: Explain LCEL and the Runnable interface.**
LCEL composes components with pipe operator. Every component is a Runnable with invoke/batch/stream/ainvoke methods. Enables: streaming, async, batch out of the box.

**Q3: What is the difference between a chain and an agent?**
Chain: predetermined sequence of steps. Agent: LLM decides what to do next dynamically. Agents use tools and can loop.

**Q4: How does RAG work with LangChain?**
Load docs → split → embed → store in vector DB → retrieve relevant chunks → augment prompt → generate. Using LCEL: retriever | format_docs | prompt | llm | parser.

**Q5: Explain the different memory types.**
Buffer (all messages), Summary (compressed), Window (last K), Vector (semantic retrieval). For new projects, use LangGraph persistence instead.

**Q6: How do you handle streaming in LangChain?**
chain.stream() for sync, chain.astream() for async. Works with FastAPI SSE or WebSocket. LCEL makes all components streamable automatically.

**Q7: What is LangSmith and why is it important?**
Tracing, evaluation, monitoring platform. See every step of execution, evaluate against test sets, monitor production. Critical for debugging and quality assurance.

**Q8: How do you choose between LangChain and LlamaIndex?**
LangChain for general-purpose LLM apps, agents, multi-agent. LlamaIndex for RAG-focused apps with complex index structures. Can use both together.

**Q9: What are the best practices for production LangChain?**
Use LCEL, add fallbacks, implement retries, use LangSmith tracing, cache responses, handle rate limits, use async for throughput.

**Q10: How do you test LangChain applications?**
Unit test individual components, integration test chains with mock LLMs, use LangSmith datasets for evaluation, implement regression testing.

**Q11: Explain the document loading and chunking pipeline.**
Loader extracts text from source → splitter breaks into chunks with overlap → metadata preserved → chunks embedded and stored.

**Q12: What is the difference between stuff, map_reduce, refine, and map_rerank chains?**
Stuff: all docs in one prompt. Map_reduce: summarize each, then combine. Refine: iteratively build answer. Map_rerank: score each, pick best.

**Q13: How do you handle rate limiting with LangChain?**
with_retry() for automatic retry, batch with max_concurrency, caching to avoid repeat calls, multiple model fallbacks.

**Q14: What are callbacks in LangChain?**
Hooks that run during chain execution: on_llm_start, on_chain_end, etc. Used for logging, streaming, monitoring. LangSmith uses callbacks internally.

**Q15: How do you switch between different LLM providers?**
LCEL makes models interchangeable. Use with_fallbacks for reliability. Provider-specific packages (langchain-openai, langchain-anthropic).

**Q16: Explain the tool calling vs ReAct agent approaches.**
Tool calling: LLM natively returns structured tool calls (GPT-4, Claude). ReAct: LLM uses reasoning + action text format. Tool calling is more reliable and recommended.

**Q17: How do you implement caching in LangChain?**
InMemoryCache, SQLiteCache, RedisCache. Set as global or per-chain. Semantic cache using embeddings for similar queries.

**Q18: What are RunnableParallel and RunnablePassthrough?**
RunnableParallel: run multiple chains simultaneously. RunnablePassthrough: pass input through unchanged (useful in LCEL for carrying data forward).

**Q19: How do you handle multi-modal content (images + text)?**
Use models with vision (GPT-4o, Claude). Pass images as base64 or URLs in HumanMessage. Multimodal document loaders for PDFs with images.

**Q20: What is the recommended architecture for a production LangChain app?**
FastAPI backend with LCEL chains, LangGraph for complex logic, LangSmith for monitoring, vector store for RAG, Redis for caching. Async everywhere.

---

# SECTION 12: PRODUCTION BEST PRACTICES

1. **Use LCEL everywhere** - Don't use legacy Chain classes
2. **Async by default** - Use ainvoke/astream for throughput
3. **Add fallbacks** - `model.with_fallbacks([backup])`
4. **Implement retries** - `model.with_retry(stop_after_attempt=3)`
5. **Cache aggressively** - Reduce costs and latency
6. **Trace with LangSmith** - Non-negotiable for production
7. **Use LangGraph for agents** - Don't use AgentExecutor for complex workflows
8. **Version your prompts** - Track prompt changes in LangSmith
9. **Stream responses** - Users expect real-time output
10. **Monitor costs** - Track token usage per chain/user

---

## Sources
- [LangChain Docs](https://python.langchain.com/docs/)
- [LCEL Concepts](https://python.langchain.com/docs/concepts/lcel/)
- [LangChain Blog: LCEL](https://blog.langchain.com/langchain-expression-language/)
- [Pinecone: LCEL Guide](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/)
