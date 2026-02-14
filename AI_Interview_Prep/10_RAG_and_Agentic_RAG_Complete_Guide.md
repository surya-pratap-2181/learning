# RAG & Agentic RAG: The Definitive Interview Guide (2025-2026)
## For AI Engineers with Production Experience

---

# TABLE OF CONTENTS

1. [What is RAG?](#section-1)
2. [Explaining RAG to a Layman](#section-2)
3. [RAG Architecture Deep Dive](#section-3)
4. [Chunking Strategies](#section-4)
5. [Embedding Models](#section-5)
6. [Vector Stores](#section-6)
7. [Retrieval Strategies](#section-7)
8. [Advanced RAG Techniques](#section-8)
9. [Agentic RAG](#section-9)
10. [RAG Evaluation](#section-10)
11. [Production RAG Challenges](#section-11)
12. [Frequently Asked Interview Questions (30+)](#section-12)
13. [Follow-up Questions Interviewers Ask](#section-13)
14. [Real-World Use Cases](#section-14)
15. [Complete Code Examples](#section-15)

---

<a name="section-1"></a>
# SECTION 1: WHAT IS RAG?

## 1.1 Definition

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by retrieving relevant external knowledge at inference time and injecting it into the prompt context before generation. It was introduced by Lewis et al. (2020) at Facebook AI Research (FAIR).

**The Core Problem RAG Solves:**
- LLMs have a **knowledge cutoff** (training data is static)
- LLMs **hallucinate** (generate plausible but incorrect facts)
- LLMs cannot access **private/proprietary data**
- Fine-tuning is **expensive** and doesn't solve freshness

**RAG Formula:**
```
Response = LLM(User Query + Retrieved Context)

Instead of:  LLM(User Query) → hallucination risk
RAG does:    Retrieve(Query) → Context → LLM(Query + Context) → grounded answer
```

## 1.2 How RAG Works - Step by Step

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE (End-to-End)                       │
├─────────────────────────────┬───────────────────────────────────────┤
│  OFFLINE (Indexing Phase)   │      ONLINE (Query Phase)            │
│                             │                                       │
│  ┌─────────────────┐       │      ┌─────────────────┐             │
│  │  Documents       │       │      │  User Query      │             │
│  │  (PDF, Web,      │       │      │  "What is our    │             │
│  │   DB, API)       │       │      │   refund policy?"│             │
│  └────────┬────────┘       │      └────────┬────────┘             │
│           │                 │               │                       │
│           ▼                 │               ▼                       │
│  ┌─────────────────┐       │      ┌─────────────────┐             │
│  │  Chunking        │       │      │  Query Embedding  │             │
│  │  (Split into     │       │      └────────┬────────┘             │
│  │   passages)      │       │               │                       │
│  └────────┬────────┘       │               ▼                       │
│           │                 │      ┌─────────────────┐             │
│           ▼                 │      │  Similarity       │             │
│  ┌─────────────────┐       │      │  Search in        │             │
│  │  Embedding       │       │      │  Vector DB        │             │
│  │  (text → vector) │       │      └────────┬────────┘             │
│  └────────┬────────┘       │               │                       │
│           │                 │               ▼                       │
│           ▼                 │      ┌─────────────────┐             │
│  ┌─────────────────┐       │      │  Top-K Chunks     │             │
│  │  Store in        │───────┼─────▶│  Retrieved        │             │
│  │  Vector DB       │       │      └────────┬────────┘             │
│  └─────────────────┘       │               │                       │
│                             │               ▼                       │
│                             │      ┌─────────────────┐             │
│                             │      │  Augmented Prompt │             │
│                             │      │  Query + Context  │             │
│                             │      └────────┬────────┘             │
│                             │               │                       │
│                             │               ▼                       │
│                             │      ┌─────────────────┐             │
│                             │      │  LLM Generation   │             │
│                             │      └────────┬────────┘             │
│                             │               │                       │
│                             │               ▼                       │
│                             │      ┌─────────────────┐             │
│                             │      │  Grounded Answer  │             │
│                             │      │  with Citations   │             │
│                             │      └─────────────────┘             │
└─────────────────────────────┴───────────────────────────────────────┘
```

## 1.3 The Five Pillars of RAG

| Pillar | Description | Key Decisions |
|--------|-------------|---------------|
| **1. Data Ingestion** | Loading documents from various sources | Connectors, parsers, metadata extraction |
| **2. Chunking** | Splitting documents into retrievable units | Chunk size, overlap, strategy |
| **3. Embedding** | Converting text to dense vectors | Model choice, dimensionality, fine-tuning |
| **4. Retrieval** | Finding relevant chunks for a query | Search algorithm, top-k, reranking |
| **5. Generation** | Producing answer from context + query | Prompt template, LLM choice, guardrails |

> 🔵 **YOUR EXPERIENCE**: Based on your work at Stellantis (MathCo), you built a full LangChain-powered RAG pipeline for natural language querying. At MARS (MathCo), you scaled this to a multi-agent analytics platform. You can speak to production experience across all five pillars.

## 1.4 RAG vs Fine-Tuning vs Prompt Engineering

| Dimension | RAG | Fine-Tuning | Prompt Engineering |
|-----------|-----|-------------|-------------------|
| **Knowledge freshness** | Real-time (retrieve latest) | Static (training snapshot) | Static (model knowledge) |
| **Cost** | Medium (embedding + retrieval) | High (GPU training) | Low (just prompts) |
| **Hallucination control** | Strong (grounded in sources) | Moderate | Weak |
| **Private data** | Excellent | Good (but expensive) | Poor |
| **Setup complexity** | Medium | High | Low |
| **Latency** | Higher (retrieval step adds ~200-500ms) | Same as base model | Same as base model |
| **Best for** | Dynamic knowledge, QA, citations | Style/format/domain adaptation | Simple tasks, formatting |
| **Update data** | Add docs anytime | Retrain needed | Limited by context window |

**Decision Matrix:**
```
Need fresh/private data?  ──────▶  RAG
Need new behavior/style?  ──────▶  Fine-Tuning
Need better formatting?   ──────▶  Prompt Engineering
Need all of the above?    ──────▶  RAG + Fine-Tuned model + Good Prompts
```

---

<a name="section-2"></a>
# SECTION 2: EXPLAINING RAG TO A LAYMAN

## 2.1 The Library Analogy

**Without RAG (Plain LLM):**
> Imagine asking a very smart person a question. They answer from memory. They graduated college 2 years ago and haven't read anything since. They might confidently give you outdated information or just make something up that *sounds* right.

**With RAG:**
> Now imagine that same smart person, but before answering your question, they quickly go to the library, find the 3 most relevant books, read the relevant pages, and THEN answer your question based on what they just read. They can even tell you which book and page they got the answer from.

## 2.2 The Open-Book Exam Analogy

```
Closed-Book Exam (Plain LLM):
  Student has memorized everything → answers from memory → might forget or confuse facts

Open-Book Exam (RAG):
  Student gets the question → looks up relevant notes → answers using notes → accurate!
```

## 2.3 The Google + ChatGPT Analogy

> "RAG is like giving ChatGPT the ability to Google your private documents before answering."

1. You ask a question
2. The system searches your company's documents (like Google searches the web)
3. It finds the most relevant paragraphs
4. It reads those paragraphs and answers your question using that information
5. It cites where it found the answer

## 2.4 For a Business Stakeholder

> "Instead of training a new AI model on our data (which costs $50K+ and takes weeks), RAG lets us plug our existing documents into an AI system. When someone asks a question, the AI first searches our documents, finds relevant information, and then generates an answer grounded in our actual data."

**Business selling points:**
- No expensive model training
- Data stays in our control
- Answers are traceable to source documents
- Knowledge updates are instant (just add/update documents)
- Reduces AI hallucination dramatically

---

<a name="section-3"></a>
# SECTION 3: RAG ARCHITECTURE DEEP DIVE

## 3.1 Evolution of RAG (2020-2026)

```
2020          2022           2023            2024-2025         2025-2026
 │              │              │                │                 │
 ▼              ▼              ▼                ▼                 ▼
Naive RAG → Advanced RAG → Modular RAG → Agentic RAG → Agentic + MCP/A2A
(Basic         (Optimized      (Composable     (Self-routing,    (Tool-using,
 pipeline)      pipeline)       modules)        self-correcting)  autonomous)
```

## 3.2 Naive RAG

The simplest implementation: Index → Retrieve → Generate.

```
┌─────────────────────────────────────┐
│           NAIVE RAG                  │
│                                      │
│  Query → Embed → Vector Search →    │
│           Top-K Chunks → LLM →      │
│           Response                   │
│                                      │
│  Problems:                           │
│  • Low retrieval precision           │
│  • Irrelevant chunks dilute context  │
│  • No query understanding            │
│  • No answer validation              │
│  • Chunk boundary issues             │
└─────────────────────────────────────┘
```

## 3.3 Advanced RAG

Adds pre-retrieval, retrieval, and post-retrieval optimizations:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ADVANCED RAG                                  │
│                                                                       │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │ PRE-RETRIEVAL   │   │ RETRIEVAL        │   │ POST-RETRIEVAL   │  │
│  │                 │   │                  │   │                  │  │
│  │ • Query         │   │ • Hybrid Search  │   │ • Reranking      │  │
│  │   Rewriting     │──▶│   (Dense+Sparse) │──▶│ • Compression    │  │
│  │ • Query         │   │ • Multi-index    │   │ • Filtering      │  │
│  │   Decomposition │   │   Retrieval      │   │ • Diversity      │  │
│  │ • HyDE          │   │ • Metadata       │   │   Selection      │  │
│  │ • Step-back     │   │   Filtering      │   │ • Lost-in-middle │  │
│  │   Prompting     │   │                  │   │   Reordering     │  │
│  └─────────────────┘   └──────────────────┘   └──────────────────┘  │
│                                                        │             │
│                                                        ▼             │
│                                              ┌──────────────────┐   │
│                                              │ GENERATION       │   │
│                                              │ • Prompt Eng.    │   │
│                                              │ • Citation       │   │
│                                              │ • Answer Valid.  │   │
│                                              └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Pre-Retrieval Optimizations

| Technique | How it Works | When to Use |
|-----------|-------------|-------------|
| **Query Rewriting** | LLM rephrases query for better retrieval | Conversational queries, vague questions |
| **Query Decomposition** | Break complex query into sub-queries | Multi-hop questions |
| **HyDE** | Generate hypothetical answer, use it to retrieve | When query-document gap is large |
| **Step-back Prompting** | Ask a more general question first | Specific questions needing broad context |
| **Query Expansion** | Add synonyms/related terms | Domain-specific jargon |

### Post-Retrieval Optimizations

| Technique | How it Works | When to Use |
|-----------|-------------|-------------|
| **Reranking** | Cross-encoder scores query-chunk pairs | Always (significant quality boost) |
| **Contextual Compression** | LLM extracts only relevant parts | Long chunks, cost optimization |
| **Diversity Selection** | MMR or similar to reduce redundancy | When chunks overlap |
| **Lost-in-middle Reorder** | Place most relevant at start/end | Long context windows |

## 3.4 Modular RAG

Treats RAG as composable modules that can be mixed, replaced, and extended:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Linear** | Modules in sequence | Retrieve → Rerank → Generate |
| **Conditional** | Route based on query type | If factual → RAG; if creative → direct LLM |
| **Branching** | Parallel retrieval, merge results | Search DB1 + DB2 → Fuse → Generate |
| **Loop** | Iterative refinement | Retrieve → Generate → Evaluate → Re-retrieve |
| **Adaptive** | Agent decides module order | LLM decides: need retrieval? which index? |

> 🔵 **YOUR EXPERIENCE**: At MathCo MARS project, you built a modular multi-agent architecture where different analytics queries were routed to different retrieval strategies - this maps directly to Modular RAG patterns.

---

<a name="section-4"></a>
# SECTION 4: CHUNKING STRATEGIES

## 4.1 Why Chunking Matters

Chunking directly impacts retrieval quality. Too large → irrelevant noise. Too small → missing context.

| Strategy | Chunk Size | Overlap | Best For |
|----------|-----------|---------|----------|
| **Fixed-size** | 256-1024 tokens | 10-20% | Simple docs, uniform content |
| **Recursive** | Variable | Yes | Most general-purpose use |
| **Semantic** | Variable | No | Topic-diverse documents |
| **Document-aware** | Varies by structure | Optional | PDFs, HTML, Markdown |
| **Sentence-based** | 3-5 sentences | 1 sentence | QA-style retrieval |

## 4.2 Detailed Strategies with Code

### Fixed-Size Chunking
```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="\n\n"
)
chunks = splitter.split_text(document)
```

### Recursive Character Splitting (Most Common)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try each in order
)
chunks = splitter.split_documents(documents)
```

### Semantic Chunking
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)
chunks = splitter.split_text(document)
```

### Parent-Document Retrieval Pattern
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Small chunks for precise retrieval, return parent for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

## 4.3 Chunking Decision Framework

```
Document Type?
├── Structured (HTML, Markdown, Code)
│   └── Use Document-Aware chunking (respect structure)
├── Conversational (Chat logs, Transcripts)
│   └── Use Semantic chunking (topic boundaries)
├── Technical (Research papers, Manuals)
│   └── Use Recursive with larger chunks (1000-1500)
└── Mixed / General
    └── Use Recursive Character (500-1000, 20% overlap)
```

---

<a name="section-5"></a>
# SECTION 5: EMBEDDING MODELS

## 5.1 What are Embeddings?

Embeddings convert text into dense numerical vectors that capture semantic meaning. Similar texts have vectors that are close together in vector space.

## 5.2 Embedding Models Comparison (2025-2026)

| Model | Dimensions | Max Tokens | MTEB Score | Cost | Open Source? |
|-------|-----------|------------|------------|------|-------------|
| **OpenAI text-embedding-3-large** | 3072 (configurable) | 8191 | 64.6 | $0.13/1M tokens | No |
| **OpenAI text-embedding-3-small** | 1536 (configurable) | 8191 | 62.3 | $0.02/1M tokens | No |
| **Cohere embed-v3** | 1024 | 512 | 64.5 | $0.10/1M tokens | No |
| **Voyage AI voyage-3** | 1024 | 32000 | 67.1 | $0.06/1M tokens | No |
| **BGE-M3** | 1024 | 8192 | 66.1 | Free | Yes |
| **GTE-Qwen2** | 1536-8192 | 131072 | 67.2 | Free | Yes |
| **E5-Mistral-7B** | 4096 | 32768 | 66.6 | Free | Yes |
| **Nomic embed-text-v1.5** | 768 | 8192 | 62.3 | Free | Yes |
| **all-MiniLM-L6-v2** | 384 | 512 | 56.3 | Free | Yes |

## 5.3 How to Choose

```
Budget constrained?
├── Yes → Open source (BGE-M3, GTE-Qwen2)
└── No → How important is quality?
    ├── Critical → Voyage AI voyage-3 or OpenAI large
    └── Good enough → OpenAI small or Cohere embed-v3

Long documents?
├── Yes → GTE-Qwen2 (128K ctx) or Voyage (32K ctx)
└── No → Any model works

Multilingual?
├── Yes → Cohere embed-v3 or BGE-M3
└── No → Any English model
```

---

<a name="section-6"></a>
# SECTION 6: VECTOR STORES

## 6.1 Comprehensive Comparison

| Database | Type | Managed? | Hybrid Search | Metadata Filter | Best For | Pricing |
|----------|------|----------|--------------|----------------|----------|---------|
| **Pinecone** | Managed | Yes (fully) | Yes | Yes | Production, scale | Free tier + $70/mo+ |
| **Weaviate** | Open Source + Cloud | Both | Yes | Yes (GraphQL) | Semantic + graph | Free self-hosted |
| **Qdrant** | Open Source + Cloud | Both | Yes | Yes (rich filters) | Complex filtering | Free self-hosted |
| **Milvus/Zilliz** | Open Source + Cloud | Both | Yes | Yes | Billion-scale | Free self-hosted |
| **Chroma** | Open Source | Self-hosted | No | Yes | Prototyping, local dev | Free |
| **FAISS** | Library | N/A | No | No | Research, benchmarks | Free |
| **pgvector** | Extension | N/A | No (add BM25 separately) | Yes (SQL) | Already using PostgreSQL | Free |

## 6.2 When to Use What

```
Prototyping / Local Dev?
├── Chroma (easiest setup, in-memory or persistent)
├── FAISS (fastest for benchmarks)
└── pgvector (if already using PostgreSQL)

Production - Small Scale (<1M vectors)?
├── pgvector (simplest if on PostgreSQL)
├── Qdrant (if need rich metadata filtering)
└── Pinecone Serverless (if want zero ops)

Production - Large Scale (>10M vectors)?
├── Pinecone (fully managed, battle-tested)
├── Milvus/Zilliz (billion-scale, open source)
└── Weaviate (if need graph + vector)
```

> 🔵 **YOUR EXPERIENCE**: At Stellantis, you integrated vector stores with LangChain for the conversational AI chatbot. Discuss your choice of vector store, how you handled updates, and scale considerations.

---

<a name="section-7"></a>
# SECTION 7: RETRIEVAL STRATEGIES

## 7.1 Dense Retrieval
Uses embedding similarity (cosine, dot product, Euclidean):
```python
# Dense retrieval with LangChain
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(docs, embedding_model)
results = vectorstore.similarity_search(query, k=5)
```

## 7.2 Sparse Retrieval (BM25)
Classic keyword-based retrieval:
```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5
results = bm25_retriever.invoke(query)
```

## 7.3 Hybrid Search (Dense + Sparse)
Combines both for best results:
```python
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # BM25 weight, Dense weight
)
results = ensemble.invoke(query)
```

## 7.4 Reranking
Second-pass scoring with cross-encoder:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-v3.5", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)
results = compression_retriever.invoke(query)
```

## 7.5 Retrieval Strategy Comparison

| Strategy | Precision | Recall | Latency | Cost | Best For |
|----------|----------|--------|---------|------|----------|
| Dense only | Medium | High | Low | Medium | General semantic search |
| Sparse (BM25) only | High (exact) | Low (semantic) | Very Low | Low | Keyword-heavy, exact match |
| Hybrid | High | High | Medium | Medium | Production (recommended) |
| Hybrid + Reranking | Very High | High | Higher | Higher | Quality-critical applications |

---

<a name="section-8"></a>
# SECTION 8: ADVANCED RAG TECHNIQUES

## 8.1 HyDE (Hypothetical Document Embedding)

Instead of embedding the query directly, generate a hypothetical answer and embed that:

```python
from langchain.chains import HypotheticalDocumentEmbedder

hyde = HypotheticalDocumentEmbedder.from_llm(
    llm=ChatOpenAI(model="gpt-4o"),
    base_embeddings=OpenAIEmbeddings(),
    prompt_key="web_search"  # or "sci_fact", etc.
)
# Query → LLM generates hypothetical answer → embed that → search
results = vectorstore.similarity_search_by_vector(hyde.embed_query(query))
```

**When to use:** Query and documents have very different language styles (e.g., user asks "why is my app slow?" but docs say "performance optimization techniques").

## 8.2 Multi-Query Retrieval

Generate multiple perspectives of the same query:

```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(model="gpt-4o-mini")
)
# Generates 3+ variations of the query, retrieves for each, deduplicates
results = retriever.invoke("What are the side effects of ibuprofen?")
# Generated queries:
# 1. "ibuprofen adverse reactions"
# 2. "negative effects of taking ibuprofen"
# 3. "ibuprofen safety concerns and risks"
```

## 8.3 Contextual Compression

Extract only the relevant portions from retrieved chunks:

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

## 8.4 Self-Query Retrieval

LLM parses the query into a structured filter + semantic query:

```python
from langchain.retrievers import SelfQueryRetriever

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Technical documentation about software products",
    metadata_field_info=[
        AttributeInfo(name="category", type="string", description="Document category"),
        AttributeInfo(name="date", type="date", description="Publication date"),
    ],
)
# "Find Python docs from 2024" → semantic: "Python" + filter: date >= 2024
```

## 8.5 Step-Back Prompting for RAG

Ask a broader question first, then use that context:

```
User: "What happens if I exceed 100 API calls per minute on the Pro plan?"

Step-back question: "What are the rate limits and policies for different plan tiers?"
→ Retrieves broader rate limit documentation
→ Then answers the specific question with full context
```

---

<a name="section-9"></a>
# SECTION 9: AGENTIC RAG

## 9.1 What is Agentic RAG?

Agentic RAG embeds **autonomous AI agents** into the RAG pipeline. Instead of a fixed retrieve→generate pipeline, an agent dynamically decides:
- **Whether** to retrieve at all
- **Where** to retrieve from (which index, which tool)
- **How** to process results (rerank, filter, re-query)
- **When** to stop (self-evaluate answer quality)

```
┌──────────────────────────────────────────────────────────────────┐
│                       AGENTIC RAG                                 │
│                                                                    │
│           ┌──────────────────────────┐                            │
│           │      AGENT (LLM)         │                            │
│           │  ┌────────────────────┐  │                            │
│           │  │ OBSERVE            │  │                            │
│           │  │ Read query + state │  │                            │
│           │  └────────┬───────────┘  │                            │
│           │           │              │                            │
│           │  ┌────────▼───────────┐  │                            │
│           │  │ THINK              │  │                            │
│           │  │ Plan next action   │  │                            │
│           │  └────────┬───────────┘  │                            │
│           │           │              │                            │
│           │  ┌────────▼───────────┐  │                            │
│           │  │ ACT                │  │                            │
│           │  │ Execute tool/search│  │                            │
│           │  └────────────────────┘  │                            │
│           └──────────┬───────────────┘                            │
│                      │                                             │
│      ┌───────────────┼───────────────┐                            │
│      ▼               ▼               ▼                            │
│  ┌────────┐   ┌──────────┐   ┌────────────┐                     │
│  │Vector  │   │SQL       │   │Web Search  │   ← TOOLS           │
│  │Search  │   │Database  │   │API Calls   │                     │
│  └────────┘   └──────────┘   └────────────┘                     │
│                                                                    │
│  Loop: Observe → Think → Act → Observe → ... → Final Answer      │
└──────────────────────────────────────────────────────────────────┘
```

## 9.2 Types of Agentic RAG

### Corrective RAG (CRAG)
Self-evaluates retrieval quality and corrects if needed:

```
Query → Retrieve → Grade Relevance →
  ├── Relevant → Generate answer
  ├── Ambiguous → Rewrite query → Re-retrieve
  └── Irrelevant → Web search fallback → Generate
```

```python
# CRAG with LangGraph
from langgraph.graph import StateGraph, END

def grade_documents(state):
    """Grade retrieved documents for relevance."""
    query = state["query"]
    documents = state["documents"]

    graded = []
    web_search_needed = False
    for doc in documents:
        score = grader_llm.invoke(
            f"Is this document relevant to '{query}'?\n{doc.page_content}"
        )
        if score == "relevant":
            graded.append(doc)
        else:
            web_search_needed = True

    return {"documents": graded, "web_search_needed": web_search_needed}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("grade", grade_documents)
workflow.add_node("web_search", web_search_fallback)
workflow.add_node("generate", generate_answer)

workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade",
    lambda s: "web_search" if s["web_search_needed"] else "generate")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
```

### Self-RAG
The model decides when to retrieve and self-evaluates its output:

```
Query → LLM decides: need retrieval?
  ├── No → Generate directly (simple factual or creative)
  └── Yes → Retrieve → Generate → Self-evaluate:
        ├── Answer supported by evidence? → Return
        └── Not supported → Re-retrieve with refined query
```

### Adaptive RAG
Routes queries to different strategies based on complexity:

```
Query → Complexity Classifier →
  ├── Simple (factual) → Direct LLM answer
  ├── Medium (single-hop) → Standard RAG
  └── Complex (multi-hop) → Agentic RAG with iterative retrieval
```

> 🔵 **YOUR EXPERIENCE**: At RavianAI, you designed the end-to-end agentic architecture from intent recognition to task execution. Your dynamic capability registry and tool discovery mechanism is directly applicable to Agentic RAG - agents reasoning about which tools/indices to use at runtime.

## 9.3 Agentic RAG with LangGraph (Production Pattern)

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "add"]
    query: str
    documents: List[str]
    generation: str
    retry_count: int

def route_query(state):
    """Route query to appropriate retrieval strategy."""
    query = state["query"]
    classification = classifier_llm.invoke(
        f"Classify this query as SIMPLE, FACTUAL, or COMPLEX: {query}"
    )
    if "SIMPLE" in classification:
        return "direct_answer"
    elif "FACTUAL" in classification:
        return "vector_search"
    else:
        return "multi_step_retrieval"

def vector_search(state):
    docs = vectorstore.similarity_search(state["query"], k=5)
    return {"documents": [d.page_content for d in docs]}

def multi_step_retrieval(state):
    # Decompose query → multiple searches → merge results
    sub_queries = decomposer_llm.invoke(f"Break this into sub-queries: {state['query']}")
    all_docs = []
    for sq in sub_queries:
        docs = vectorstore.similarity_search(sq, k=3)
        all_docs.extend(docs)
    return {"documents": list(set([d.page_content for d in all_docs]))}

def evaluate_answer(state):
    """Self-evaluate if answer is supported by evidence."""
    is_grounded = evaluator_llm.invoke(
        f"Is this answer grounded in the documents?\n"
        f"Answer: {state['generation']}\nDocs: {state['documents']}"
    )
    if "yes" in is_grounded.lower() or state["retry_count"] >= 2:
        return "done"
    return "retry"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("vector_search", vector_search)
graph.add_node("multi_step_retrieval", multi_step_retrieval)
graph.add_node("generate", generate_answer)
graph.add_node("evaluate", evaluate_answer)

graph.set_conditional_entry_point(route_query, {
    "direct_answer": "generate",
    "vector_search": "vector_search",
    "multi_step_retrieval": "multi_step_retrieval"
})
graph.add_edge("vector_search", "generate")
graph.add_edge("multi_step_retrieval", "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", evaluate_answer, {
    "done": END,
    "retry": "vector_search"
})
```

---

<a name="section-10"></a>
# SECTION 10: RAG EVALUATION

## 10.1 RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) is the standard evaluation framework:

| Metric | What it Measures | Range | Formula |
|--------|-----------------|-------|---------|
| **Faithfulness** | Is the answer grounded in context? | 0-1 | #supported_claims / #total_claims |
| **Answer Relevancy** | Does the answer address the question? | 0-1 | Cosine sim(generated_questions, original_question) |
| **Context Precision** | Are relevant chunks ranked higher? | 0-1 | Weighted precision of relevant contexts at each rank |
| **Context Recall** | Are all needed facts retrieved? | 0-1 | #attributable_sentences / #ground_truth_sentences |

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall
)
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": generated_answers,
    "contexts": retrieved_contexts,
    "ground_truth": reference_answers
})

result = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=eval_llm,
    embeddings=eval_embeddings
)
print(result)
# {'faithfulness': 0.87, 'answer_relevancy': 0.91,
#  'context_precision': 0.83, 'context_recall': 0.79}
```

## 10.2 Other Evaluation Approaches

| Tool | Type | Best For |
|------|------|----------|
| **RAGAS** | Automated (LLM-as-judge) | Standard RAG evaluation |
| **LangSmith** | Tracing + Evaluation | Production monitoring |
| **DeepEval** | Automated | Comprehensive metrics |
| **TruLens** | Automated | Feedback functions |
| **Human Evaluation** | Manual | Gold standard, expensive |

---

<a name="section-11"></a>
# SECTION 11: PRODUCTION RAG CHALLENGES

## 11.1 Common Challenges & Solutions

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Stale data** | Documents change but index doesn't | Incremental indexing, change detection |
| **Scale** | Millions of documents, high QPS | Sharding, caching, async retrieval |
| **Latency** | Retrieval adds 200-500ms | Semantic caching, pre-computed results |
| **Cost** | Embedding + LLM costs add up | Batch embedding, smaller models, caching |
| **Multi-tenancy** | Different users/orgs, data isolation | Namespace filtering, permission-aware retrieval |
| **Hallucination** | LLM ignores context, makes things up | Stronger prompts, faithfulness checks, citations |
| **Lost in middle** | LLM ignores middle of long context | Reorder chunks (relevant at start/end) |
| **Chunk boundaries** | Important info split across chunks | Overlapping chunks, parent-document retriever |

## 11.2 Semantic Caching

```python
from langchain.cache import RedisSemanticCache

# If a similar query was asked before, return cached result
cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95  # 95% similarity = cache hit
)
```

> 🔵 **YOUR EXPERIENCE**: At MathCo, you dealt with production RAG challenges at enterprise scale - multi-tenancy for different analytics teams, data freshness for uploaded datasets, and cost optimization. Discuss specific solutions you implemented.

---

<a name="section-12"></a>
# SECTION 12: FREQUENTLY ASKED INTERVIEW QUESTIONS (30+)

## Fundamentals

**Q1: What is RAG and why is it important?**
**A:** RAG (Retrieval-Augmented Generation) combines information retrieval with LLM generation. Instead of relying solely on model memory, it retrieves relevant documents and grounds the LLM's response in actual data. It's critical because it solves hallucination, enables access to private data, and keeps knowledge fresh without expensive retraining.

**Q2: Explain the complete RAG pipeline from document ingestion to response generation.**
**A:** The pipeline has two phases:
- **Offline (Indexing):** Load documents → chunk them → generate embeddings → store in vector database
- **Online (Query):** User query → embed query → similarity search in vector DB → retrieve top-k chunks → construct prompt (query + context) → LLM generates grounded answer

**Q3: What is the difference between Naive RAG, Advanced RAG, and Agentic RAG?**
**A:**
- **Naive RAG:** Simple retrieve-then-generate. No query optimization or result refinement.
- **Advanced RAG:** Adds pre-retrieval (query rewriting, HyDE), improved retrieval (hybrid search), and post-retrieval (reranking, compression) optimizations.
- **Agentic RAG:** An autonomous agent decides *whether*, *where*, and *how* to retrieve, can use multiple tools, self-evaluates answers, and iteratively refines until quality threshold is met.

**Q4: RAG vs Fine-tuning - when would you use each?**
**A:** RAG for dynamic/private knowledge, citations, and when data changes frequently. Fine-tuning for teaching new behaviors, styles, or domain-specific reasoning. In practice, you often use both: fine-tune for domain adaptation + RAG for specific knowledge retrieval.

## Chunking & Embedding

**Q5: How do you choose the right chunk size?**
**A:** It depends on the use case. Smaller chunks (256-512 tokens) give more precise retrieval but may lack context. Larger chunks (1000-1500 tokens) provide more context but may introduce noise. Best practice: experiment with multiple sizes and evaluate retrieval quality using metrics like context precision/recall. Use overlap (10-20%) to prevent information loss at boundaries.

**Q6: What is semantic chunking and when would you use it?**
**A:** Semantic chunking splits text at natural topic boundaries using embedding similarity. Adjacent sentences are compared, and when similarity drops below a threshold, a split occurs. Use it when documents cover multiple topics and you want each chunk to be a coherent semantic unit.

**Q7: How do embedding models work and how do you choose one?**
**A:** Embedding models convert text to dense vectors in a high-dimensional space where semantically similar texts are close together. Trained using contrastive learning (similar pairs close, dissimilar pairs far). Choose based on: MTEB benchmark scores, dimensionality (affects storage/speed), max context length, cost, and whether you need multilingual support.

## Retrieval

**Q8: Explain hybrid search and why it's better than pure vector search.**
**A:** Hybrid search combines dense retrieval (embeddings/vector similarity) with sparse retrieval (BM25/keyword matching). Dense catches semantic similarity ("car" ≈ "automobile") while sparse catches exact matches (product IDs, technical terms). Combined using Reciprocal Rank Fusion (RRF), it gives significantly better results than either alone.

**Q9: What is reranking and why is it important?**
**A:** Reranking uses a cross-encoder to re-score the top-k results from initial retrieval. Unlike bi-encoders (used for initial search, encode query and doc separately), cross-encoders process query+document together, enabling deeper understanding of relevance. Improves precision significantly, typically a 10-20% quality gain.

**Q10: How does Maximum Marginal Relevance (MMR) work?**
**A:** MMR balances relevance and diversity in results. Formula: MMR = argmax[λ * Sim(doc, query) - (1-λ) * max(Sim(doc, already_selected))]. λ controls the tradeoff: λ=1 is pure relevance, λ=0 is pure diversity. Prevents retrieving 5 nearly identical chunks.

## Advanced Techniques

**Q11: Explain HyDE (Hypothetical Document Embedding).**
**A:** HyDE generates a hypothetical answer to the query using the LLM (without retrieval), then embeds this hypothetical document to search for similar real documents. This bridges the query-document semantic gap because the hypothetical answer uses similar language to actual documents. Effective when user queries are very different from document language.

**Q12: What is query decomposition and when do you use it?**
**A:** Breaking a complex multi-hop question into simpler sub-questions, retrieving for each, then synthesizing. Example: "Compare the pricing of product A and B" → sub-query 1: "What is the pricing of product A?" + sub-query 2: "What is the pricing of product B?" Use for multi-hop, comparative, or complex analytical questions.

**Q13: Explain the parent-document retrieval strategy.**
**A:** Index small chunks for precise retrieval but return the parent (larger) chunk for context. You split documents into large parents (2000 tokens) and small children (400 tokens). Search against children for precision, but return the parent for the LLM to have enough context for good generation.

**Q14: What is Corrective RAG (CRAG)?**
**A:** CRAG adds a self-evaluation step after retrieval. A grader LLM assesses if retrieved documents are relevant. If relevant → proceed to generation. If ambiguous → rewrite query and re-retrieve. If irrelevant → fall back to web search. This self-corrective loop dramatically improves answer quality.

**Q15: What is Self-RAG?**
**A:** Self-RAG trains the model to decide when to retrieve, and then self-evaluate its outputs with special "reflection tokens." The model generates critique tokens like [IsRel] (is retrieval needed?), [IsSup] (is answer supported?), [IsUse] (is answer useful?). This makes retrieval adaptive rather than always-on.

## Production & Scale

**Q16: How do you handle document updates in a production RAG system?**
**A:** Implement incremental indexing: detect changed documents (hash comparison), re-chunk and re-embed only changed docs, update vector store entries (upsert). For deletions, remove corresponding vectors. Use metadata timestamps for freshness. Consider a dual-index approach during reindexing to avoid downtime.

**Q17: How do you evaluate a RAG system?**
**A:** Use RAGAS framework: Faithfulness (is answer grounded?), Answer Relevancy (does it address the question?), Context Precision (are relevant docs ranked high?), Context Recall (are all needed docs retrieved?). Also track: end-to-end latency, retrieval latency, token costs, user satisfaction, and hallucination rate.

**Q18: How do you handle multi-tenancy in RAG?**
**A:** Use namespace/partition-based isolation in the vector store. Each tenant gets their own namespace (Pinecone namespaces, Weaviate classes, metadata filters). Apply permissions at retrieval time. Never let one tenant's query retrieve another tenant's data.

**Q19: How do you reduce latency in RAG?**
**A:** Semantic caching (cache similar queries), pre-computed embeddings, streaming responses, async retrieval, use faster models for reranking, optimize chunk sizes, use connection pooling for vector DB, deploy embedding model closer to vector DB.

**Q20: How do you handle long documents that exceed context windows?**
**A:** Map-reduce (summarize chunks separately, then combine), refine (iteratively build answer), hierarchical summarization, recursive retrieval (retrieve summaries, then drill into details). For very long docs, use tree-based summarization.

## Agentic RAG

**Q21: What makes Agentic RAG different from standard RAG?**
**A:** Standard RAG has a fixed pipeline (always retrieve, always from the same source). Agentic RAG has an autonomous agent that: (1) decides IF retrieval is needed, (2) chooses WHICH source to query, (3) evaluates results and decides to re-query or use a different tool, (4) can use multiple retrieval steps with self-correction. It's more flexible but also more complex and expensive.

**Q22: How would you implement routing in Agentic RAG?**
**A:** Use a router/classifier LLM that categorizes the query and routes to the appropriate strategy. Can be implemented with LangGraph conditional edges or function calling. Routes might include: vector search, SQL query, web search, direct LLM answer, or multi-step retrieval.

**Q23: What is the Observe-Think-Act loop in Agentic RAG?**
**A:** The ReAct (Reasoning + Acting) pattern: OBSERVE the current state (query, retrieved docs, previous attempts), THINK about what to do next (reason about quality, decide next action), ACT (retrieve more, search differently, generate answer, or request clarification). This loop repeats until the agent is confident in its answer.

**Q24: How do you prevent infinite loops in Agentic RAG?**
**A:** Set maximum iteration limits, use diminishing returns detection (if re-retrieval doesn't improve relevance scores, stop), implement timeout mechanisms, track retry counts in state, and have a fallback "best effort" answer after N attempts.

**Q25: How does Agentic RAG handle multi-source retrieval?**
**A:** The agent has access to multiple tools/indices (vector DB, SQL database, web search, APIs). Based on query analysis, it selects the appropriate source(s). Can query multiple sources in parallel, merge and deduplicate results, then generate from combined context.

## Design & Architecture

**Q26: Design a RAG system for a customer support chatbot.**
**A:** Components: (1) Ingest support docs, FAQs, product manuals, (2) Chunk by section/topic, (3) Hybrid search (keyword for product IDs + semantic for descriptions), (4) Reranking for precision, (5) Guard rails for hallucination, (6) Citation links back to docs, (7) Escalation to human when confidence is low, (8) Feedback loop for continuous improvement.

**Q27: How would you handle structured and unstructured data in the same RAG system?**
**A:** Use a multi-index approach: vector index for unstructured text, SQL/graph database for structured data. Agent routes queries: factual/numerical → SQL, descriptive/conceptual → vector search. Merge results in the prompt.

**Q28: How do you implement citation/source attribution in RAG?**
**A:** Include source metadata (document name, page, URL) with each chunk. In the generation prompt, instruct the LLM to cite sources inline (e.g., [Source: doc_name, p.12]). Post-process to verify citations exist and link correctly. Some approaches use structured output to separate answer from citations.

**Q29: What's the role of guardrails in a RAG system?**
**A:** Input guardrails: detect prompt injection, off-topic queries. Output guardrails: check for hallucination (answer must be grounded in context), PII leakage, toxic content, and format compliance. Use tools like Guardrails AI, NeMo Guardrails, or custom LLM-based checks.

**Q30: How do you A/B test different RAG configurations?**
**A:** Create evaluation datasets with ground truth. Test variations: chunk sizes, embedding models, retrieval strategies, reranking models, number of retrieved docs, prompt templates. Use RAGAS metrics + latency + cost as evaluation criteria. Implement feature flags to route a percentage of traffic to new configurations.

---

<a name="section-13"></a>
# SECTION 13: FOLLOW-UP QUESTIONS INTERVIEWERS ASK

These are the tricky follow-ups after your initial answer:

| After You Say... | They Ask... | What They Want |
|-------------------|-------------|----------------|
| "We use cosine similarity" | "Why not dot product or Euclidean?" | Understanding of similarity metrics |
| "We chunk at 500 tokens" | "How did you arrive at that number?" | Evidence-based decision making |
| "We use hybrid search" | "What weights do you use for dense vs sparse?" | Practical tuning experience |
| "We rerank with Cohere" | "Have you tried cross-encoders? What's the latency impact?" | Cost-benefit analysis |
| "We use RAGAS for eval" | "What faithfulness score is acceptable?" | Production quality thresholds |
| "Our RAG reduces hallucination" | "By how much? How do you measure it?" | Quantitative thinking |
| "We cache similar queries" | "How do you invalidate the cache when docs update?" | System design thinking |
| "We use Agentic RAG" | "How do you prevent it from making too many LLM calls?" | Cost awareness |

---

<a name="section-14"></a>
# SECTION 14: REAL-WORLD USE CASES

## 14.1 Enterprise Knowledge Base (Your Stellantis Experience)

```
┌─────────────────────────────────────────────────────────────────┐
│  ENTERPRISE KNOWLEDGE BASE RAG                                   │
│                                                                   │
│  Data Sources:                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │Confluence │ │SharePoint│ │ Internal │ │ Product  │          │
│  │  Docs    │ │  Files   │ │  Wiki    │ │  Manuals │          │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘          │
│        └──────────────┴──────────────┴──────────────┘            │
│                           │                                       │
│                    ┌──────▼──────┐                                │
│                    │ Ingestion    │                                │
│                    │ Pipeline     │                                │
│                    └──────┬──────┘                                │
│                           │                                       │
│              ┌────────────┼────────────┐                         │
│              ▼            ▼            ▼                          │
│         ┌────────┐  ┌────────┐  ┌────────┐                     │
│         │Chunking│  │Metadata│  │Embedding│                     │
│         │        │  │Extract │  │         │                     │
│         └───┬────┘  └───┬────┘  └───┬────┘                     │
│             └────────────┴────────────┘                           │
│                          │                                        │
│                   ┌──────▼──────┐                                │
│                   │ Vector DB    │                                │
│                   │ (Pinecone)   │                                │
│                   └──────┬──────┘                                │
│                          │                                        │
│   User Query ─▶ Hybrid Search ─▶ Rerank ─▶ LLM ─▶ Answer      │
│                                                    + Citations   │
└─────────────────────────────────────────────────────────────────┘
```

> 🔵 **YOUR EXPERIENCE**: At Stellantis (MathCo), you built exactly this architecture - a LangChain-powered conversational AI for natural language querying across customer data. Discuss how you handled cross-region data, consent management, and real-time insights.

## 14.2 Multi-Agent Analytics Platform (Your MARS Experience)

```
User uploads dataset ──▶ EDA Agent ──▶ Feature Engineering Agent
                                              │
                              ┌────────────────┤
                              ▼                ▼
                     Insights Agent    Model Training Agent
                              │                │
                              ▼                ▼
                     RAG-powered         Evaluation Agent
                     Q&A on data               │
                              │                ▼
                              └────▶ Interactive Dashboard
```

> 🔵 **YOUR EXPERIENCE**: At MathCo MARS project, you built this exact multi-agent analytics platform with RAG capabilities, integrating multiple LLM APIs with a ReactJS frontend.

---

<a name="section-15"></a>
# SECTION 15: COMPLETE CODE EXAMPLES

## 15.1 Production RAG with LangChain (Full Example)

```python
"""
Complete Production RAG Pipeline
Includes: Hybrid search, reranking, streaming, evaluation
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load and chunk documents
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("company_docs.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)

# 2. Create vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 3. Set up hybrid retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(chunks, k=10)

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# 4. Add reranking
reranker = CohereRerank(model="rerank-v3.5", top_n=5)
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)

# 5. Create RAG chain
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Provide a clear, concise answer with citations to the source documents.
""")

llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, "
        f"Page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    )

rag_chain = (
    {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query with streaming
for chunk in rag_chain.stream("What is our refund policy?"):
    print(chunk, end="", flush=True)
```

## 15.2 Agentic RAG with LangGraph

```python
"""
Agentic RAG with self-correction using LangGraph
Implements: CRAG pattern (Corrective RAG)
"""
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class RAGState(TypedDict):
    query: str
    documents: List[str]
    generation: str
    relevance_score: float
    retry_count: int
    search_type: str

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def retrieve(state: RAGState) -> dict:
    """Retrieve documents from vector store."""
    docs = vectorstore.similarity_search(state["query"], k=5)
    return {"documents": [d.page_content for d in docs], "search_type": "vector"}

def grade_documents(state: RAGState) -> dict:
    """Grade retrieved documents for relevance."""
    relevant_docs = []
    for doc in state["documents"]:
        response = llm.invoke(
            f"Is this document relevant to the query '{state['query']}'? "
            f"Document: {doc}\nAnswer YES or NO only."
        )
        if "YES" in response.content.upper():
            relevant_docs.append(doc)

    score = len(relevant_docs) / max(len(state["documents"]), 1)
    return {"documents": relevant_docs, "relevance_score": score}

def decide_next_step(state: RAGState) -> Literal["generate", "web_search", "rewrite"]:
    """Route based on document relevance."""
    if state["relevance_score"] > 0.6:
        return "generate"
    elif state["retry_count"] < 2:
        return "rewrite"
    else:
        return "web_search"

def rewrite_query(state: RAGState) -> dict:
    """Rewrite query for better retrieval."""
    response = llm.invoke(
        f"Rewrite this query to get better search results: {state['query']}"
    )
    return {"query": response.content, "retry_count": state["retry_count"] + 1}

def web_search(state: RAGState) -> dict:
    """Fallback to web search."""
    from langchain_community.tools import TavilySearchResults
    search = TavilySearchResults(max_results=3)
    results = search.invoke(state["query"])
    docs = [r["content"] for r in results]
    return {"documents": docs, "search_type": "web"}

def generate(state: RAGState) -> dict:
    """Generate answer from documents."""
    context = "\n\n".join(state["documents"])
    response = llm.invoke(
        f"Answer based on context:\n{context}\n\nQuestion: {state['query']}"
    )
    return {"generation": response.content}

# Build the graph
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_next_step, {
    "generate": "generate",
    "rewrite": "rewrite",
    "web_search": "web_search"
})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# Run
result = app.invoke({
    "query": "What is our return policy for electronics?",
    "documents": [],
    "generation": "",
    "relevance_score": 0.0,
    "retry_count": 0,
    "search_type": ""
})
print(result["generation"])
```

---

## Sources & References

- [DataCamp: Top 30 RAG Interview Questions 2026](https://www.datacamp.com/blog/rag-interview-questions)
- [Analytics Vidhya: RAG Interview 40 Questions](https://www.analyticsvidhya.com/blog/2026/02/rag-interview-questions-and-answers/)
- [Humanloop: 8 RAG Architectures 2025](https://humanloop.com/blog/rag-architectures)
- [ArXiv: Agentic RAG Survey](https://arxiv.org/abs/2501.09136)
- [Aisera: Agentic RAG Complete Guide](https://aisera.com/blog/agentic-rag/)
- [SDH Global: RAG Architecture Diagrams](https://sdh.global/blog/development/8-rag-architecture-diagrams-you-need-to-master-in-2025/)
