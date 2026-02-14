---
title: "Part 1 - Architecture"
layout: default
parent: "RAG Systems"
nav_order: 2
---


# RAG (Retrieval Augmented Generation) - Comprehensive Interview Guide
# PART 1: ARCHITECTURE FUNDAMENTALS

---

## 1. What is RAG and Why Does It Exist?

**RAG (Retrieval Augmented Generation)** is a technique that enhances Large Language Model (LLM) outputs by retrieving relevant information from external knowledge sources before generating a response. It was introduced by Lewis et al. in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Facebook AI Research).

### The Core Problem RAG Solves:
- **Knowledge cutoff**: LLMs are trained on data up to a certain date
- **Hallucinations**: LLMs confidently generate incorrect information
- **No access to private/proprietary data**: LLMs cannot access your company's internal documents
- **Stale information**: The world changes after training
- **Lack of attribution**: No way to cite sources for generated answers

### RAG vs Fine-tuning vs Prompt Engineering:

| Aspect | RAG | Fine-tuning | Prompt Engineering |
|--------|-----|-------------|-------------------|
| Knowledge update | Real-time, no retraining | Requires retraining | Limited by context window |
| Cost | Moderate (infra + retrieval) | High (GPU, data prep) | Low |
| Hallucination control | High (grounded in docs) | Moderate | Low |
| Domain adaptation | Excellent | Excellent | Limited |
| Implementation complexity | Moderate | High | Low |
| Data privacy | Data stays in your infra | Data used in training | Data in prompts |
| Latency | Higher (retrieval step) | Same as base model | Same as base model |

---

## 2. RAG Architecture - End to End Pipeline

### The Two Main Phases:

#### Phase 1: Indexing (Offline / Ingestion Pipeline)
```
Documents --> Document Loading --> Chunking --> Embedding --> Vector Store
```

#### Phase 2: Retrieval & Generation (Online / Query Pipeline)
```
User Query --> Query Embedding --> Similarity Search --> Context Assembly --> LLM Generation --> Response
```

### Detailed Component Breakdown:

### 2.1 Document Loading

Document loaders ingest data from various sources into a unified format.

**Common Document Types and Loaders:**

```python
# LangChain Document Loading Examples

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
pages = loader.load()  # Returns list of Document objects

# Each Document has: page_content (str) + metadata (dict)
print(pages[0].page_content)
print(pages[0].metadata)  # {'source': 'document.pdf', 'page': 0}

# Web Page Loading
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# CSV Loading
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")
docs = loader.load()

# Directory Loading (multiple files)
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./documents/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Unstructured Loading (handles many formats)
from langchain_community.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader("mixed_content.docx", mode="elements")
docs = loader.load()

# Notion Database
from langchain_community.document_loaders import NotionDBLoader
loader = NotionDBLoader(
    integration_token="your_token",
    database_id="your_db_id"
)
docs = loader.load()

# SQL Database
from langchain_community.document_loaders import SQLDatabaseLoader
loader = SQLDatabaseLoader(
    query="SELECT * FROM articles",
    db=database_connection
)

# S3 Bucket
from langchain_community.document_loaders import S3FileLoader
loader = S3FileLoader(bucket="my-bucket", key="docs/file.pdf")
```

**Interview Question: "How do you handle different document formats in a RAG pipeline?"**

**Answer**: Use format-specific loaders that extract text while preserving structure. For PDFs, PyPDFLoader or Unstructured handles text extraction. For HTML, BeautifulSoup-based loaders strip tags. For complex documents (tables, images), use multimodal approaches - extract tables separately with tools like Camelot or Tabula, process images with vision models, and merge the extracted text. Always preserve metadata (source, page number, section) for citation and filtering.

---

### 2.2 Text Splitting / Chunking

Chunking breaks documents into smaller, semantically meaningful pieces that fit within embedding model context windows and retrieval granularity requirements.

**(Detailed chunking strategies covered in Part 3)**

Quick overview of strategies:
- **Fixed-size chunking**: Split by character/token count with overlap
- **Recursive character splitting**: Hierarchical splitting by separators
- **Semantic chunking**: Split based on embedding similarity
- **Document-specific**: Markdown headers, HTML tags, code structure
- **Agentic chunking**: LLM-driven chunking decisions

```python
# Basic RecursiveCharacterTextSplitter Example
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)
```

---

### 2.3 Embedding Models

Embeddings convert text into dense numerical vectors that capture semantic meaning. Similar texts produce vectors that are close together in vector space.

**How Embeddings Work:**
- Input text is tokenized and passed through a transformer encoder
- The model produces a fixed-dimensional vector (e.g., 768, 1024, 1536, 3072 dimensions)
- Vectors are normalized so cosine similarity can be used for comparison

**Popular Embedding Models (as of 2025):**

| Model | Dimensions | Context Length | MTEB Score | Provider |
|-------|-----------|---------------|------------|----------|
| text-embedding-3-large | 3072 | 8191 tokens | ~64.6 | OpenAI |
| text-embedding-3-small | 1536 | 8191 tokens | ~62.3 | OpenAI |
| text-embedding-ada-002 | 1536 | 8191 tokens | ~61.0 | OpenAI |
| Cohere embed-v3 | 1024 | 512 tokens | ~64.5 | Cohere |
| voyage-large-2 | 1536 | 16000 tokens | ~65.0 | Voyage AI |
| BGE-large-en-v1.5 | 1024 | 512 tokens | ~63.6 | BAAI (open source) |
| E5-mistral-7b-instruct | 4096 | 32768 tokens | ~66.6 | Microsoft (open source) |
| GTE-Qwen2-7B-instruct | 3584 | 32768 tokens | ~67.0+ | Alibaba (open source) |
| nomic-embed-text-v1.5 | 768 | 8192 tokens | ~62.3 | Nomic (open source) |
| jina-embeddings-v3 | 1024 | 8192 tokens | ~65.5 | Jina AI |
| all-MiniLM-L6-v2 | 384 | 256 tokens | ~56.3 | Sentence Transformers |

```python
# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector = embeddings.embed_query("What is machine learning?")
print(f"Vector dimension: {len(vector)}")  # 3072

# Dimensionality reduction with text-embedding-3-*
embeddings_small = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=256  # Reduce from 3072 to 256 with Matryoshka representation
)

# Open Source with HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# Using sentence-transformers directly
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["This is a sentence", "This is another sentence"]
embeddings = model.encode(sentences)

# Cohere Embeddings with input types
import cohere
co = cohere.Client('your-api-key')

# For documents being stored
doc_embeddings = co.embed(
    texts=["Document text here"],
    model="embed-english-v3.0",
    input_type="search_document"  # Important: specify input type
).embeddings

# For queries
query_embedding = co.embed(
    texts=["What is RAG?"],
    model="embed-english-v3.0",
    input_type="search_query"  # Different input type for queries
).embeddings
```

**Interview Question: "How do you choose the right embedding model?"**

**Answer**: Consider these factors:
1. **Task domain**: General-purpose vs domain-specific. For medical/legal, fine-tuned or domain-specific models perform better.
2. **Dimensions vs performance tradeoff**: Higher dimensions (3072) capture more nuance but cost more storage/compute. Lower dims (384) are faster but less accurate.
3. **Context length**: If chunks are long, need models that handle >512 tokens (e.g., E5-mistral at 32K, jina-v3 at 8K).
4. **Latency requirements**: Smaller models (MiniLM at 384d) are 5-10x faster than 7B parameter models.
5. **Cost**: Open source (BGE, E5) = free inference on your GPU. API-based (OpenAI, Cohere) = per-token cost.
6. **Matryoshka embeddings**: Models like text-embedding-3 support dimensionality reduction post-hoc.
7. **Benchmark performance**: Check MTEB leaderboard, but always evaluate on YOUR data.

**Interview Question: "What is the difference between symmetric and asymmetric embedding models?"**

**Answer**:
- **Symmetric**: Query and document are treated identically. Good when query and document are similar in form (e.g., sentence similarity). Example: `all-MiniLM-L6-v2`.
- **Asymmetric**: Query and document are encoded differently. The query is typically short, the document is longer. Models like Cohere embed-v3 use `input_type` parameter. BGE models use instruction prefixes like "Represent this sentence for searching relevant passages:". This typically performs better for retrieval tasks.

---

### 2.4 Vector Stores

Vector stores are specialized databases optimized for storing, indexing, and querying high-dimensional vectors.

**Core Concepts:**

**Similarity Metrics:**
- **Cosine Similarity**: Measures angle between vectors. Range [-1, 1]. Most common for normalized embeddings.
  ```
  cos(A, B) = (A . B) / (||A|| * ||B||)
  ```
- **Euclidean Distance (L2)**: Straight-line distance. Lower = more similar.
  ```
  d(A, B) = sqrt(sum((Ai - Bi)^2))
  ```
- **Dot Product (Inner Product)**: For non-normalized vectors. Higher = more similar.
  ```
  dot(A, B) = sum(Ai * Bi)
  ```
- **Manhattan Distance (L1)**: Sum of absolute differences.

**Indexing Algorithms:**
- **Flat (Brute Force)**: Exact search, O(n) per query. Perfect recall but slow at scale.
- **IVF (Inverted File Index)**: Clusters vectors with k-means, searches only nearby clusters. `nprobe` controls accuracy/speed tradeoff.
- **HNSW (Hierarchical Navigable Small World)**: Graph-based, builds multi-layer graph. Best recall/speed tradeoff. Memory intensive.
- **PQ (Product Quantization)**: Compresses vectors by splitting into subvectors and quantizing. Reduces memory 4-64x.
- **ScaNN (Scalable Nearest Neighbors)**: Google's algorithm, anisotropic vector quantization.
- **DiskANN**: Microsoft's disk-based ANN for billion-scale datasets.

```python
# ChromaDB (In-memory, great for prototyping)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)

# Query
results = vectorstore.similarity_search("What is RAG?", k=5)
results_with_scores = vectorstore.similarity_search_with_score("What is RAG?", k=5)

# FAISS (Facebook AI Similarity Search)
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Load later
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Pinecone (Managed cloud service)
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-index")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=embeddings
)

# Weaviate
from langchain_weaviate import WeaviateVectorStore
import weaviate

client = weaviate.connect_to_local()
vectorstore = WeaviateVectorStore(
    client=client,
    index_name="Documents",
    embedding=embeddings,
    text_key="text"
)

# pgvector (PostgreSQL extension)
from langchain_postgres import PGVector

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="my_collection",
    connection="postgresql+psycopg://user:pass@localhost:5432/vectordb",
)
```

**(Detailed vector database comparison in Part 7)**

---

### 2.5 Retrieval Methods

#### Basic Retrieval:

```python
# Simple similarity search
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
docs = retriever.invoke("What is RAG?")
```

#### Maximum Marginal Relevance (MMR):
Balances relevance with diversity to avoid redundant results.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           # Number of results to return
        "fetch_k": 20,    # Number of candidates to consider
        "lambda_mult": 0.7 # 0=max diversity, 1=max relevance
    }
)
```

**MMR Formula:**
```
MMR = argmax[lambda * Sim(doc, query) - (1-lambda) * max(Sim(doc, selected_docs))]
```

#### Similarity Score Threshold:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # Only return docs above this score
        "k": 10
    }
)
```

#### Hybrid Search (Dense + Sparse):
Combines semantic (dense) search with keyword (sparse/BM25) search.

```python
# Using Weaviate's built-in hybrid search
from langchain_weaviate import WeaviateVectorStore

vectorstore = WeaviateVectorStore(client=client, index_name="Docs", embedding=embeddings)
retriever = vectorstore.as_retriever(
    search_type="hybrid",
    search_kwargs={"alpha": 0.5}  # 0=pure BM25, 1=pure vector
)

# Manual hybrid search with BM25 + Vector using Ensemble Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(documents, k=5)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Weight towards semantic search
)
results = ensemble_retriever.invoke("What is RAG?")
```

**Interview Question: "When would you use hybrid search over pure vector search?"**

**Answer**: Use hybrid search when:
1. **Exact keyword matching matters**: Product IDs, technical terms, acronyms (e.g., "HNSW" might not have a semantically close embedding)
2. **Domain-specific vocabulary**: Medical/legal terms that general embeddings may not capture well
3. **Mixed query types**: Some users type keywords, others type natural language questions
4. **Proper nouns**: People names, company names, specific identifiers
5. **Short queries**: BM25 often outperforms dense retrieval on very short (1-2 word) queries
6. Research shows hybrid consistently outperforms either method alone. The alpha parameter lets you tune the balance.

#### Multi-Index Retrieval:
Query across multiple vector stores or collections.

```python
from langchain.retrievers import MergerRetriever

retriever1 = vectorstore1.as_retriever(search_kwargs={"k": 3})
retriever2 = vectorstore2.as_retriever(search_kwargs={"k": 3})

merger_retriever = MergerRetriever(retrievers=[retriever1, retriever2])
```

#### Metadata Filtering:

```python
# Filter by metadata before similarity search
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {
            "source": "annual_report_2024.pdf",
            "department": "engineering"
        }
    }
)
```

---

### 2.6 Generation with Context

The final step: inject retrieved context into the LLM prompt.

```python
# Basic RAG Chain with LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# RAG Prompt Template
template = """You are a helpful assistant. Answer the question based ONLY on the following context.
If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )

# RAG Chain using LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is retrieval augmented generation?")
print(response)
```

```python
# Full RAG Pipeline with Sources/Citations
from langchain_core.runnables import RunnableParallel

# Chain that returns both answer and source documents
rag_chain_with_sources = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(
    answer=lambda x: (
        prompt.invoke({"context": format_docs(x["context"]), "question": x["question"]})
        | llm
        | StrOutputParser()
    ).invoke(x)
)

# Alternative: Simple approach
def rag_with_sources(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    return {
        "answer": response.content,
        "sources": [doc.metadata for doc in docs]
    }
```

```python
# Using LlamaIndex for RAG
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# Load and index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"  # Options: refine, compact, tree_summarize, simple
)
response = query_engine.query("What is RAG?")
print(response)
print(response.source_nodes)  # Retrieved chunks with scores
```

**Interview Question: "What are the different response synthesis modes?"**

**Answer**:
- **Stuff/Simple**: Concatenate all retrieved chunks into one prompt. Fast but limited by context window.
- **Refine**: Iterate through chunks one by one, refining the answer with each. Better for many chunks but slow (multiple LLM calls).
- **Map-Reduce**: Generate answer from each chunk independently, then combine. Good for summarization.
- **Tree Summarize**: Recursively summarize chunks in a tree structure. Balanced approach.
- **Compact**: Like refine but stuffs as many chunks as possible per LLM call to minimize calls.

---

## 3. Key Architecture Interview Questions

### Q: "Walk me through designing a RAG system from scratch for a company's internal documentation."

**Answer**:
1. **Requirements gathering**: Volume of docs, update frequency, query patterns, latency requirements, security constraints
2. **Document ingestion pipeline**:
   - Set up document loaders for each format (PDF, Confluence, Notion, etc.)
   - Implement recursive character splitting with 512-1000 token chunks, 10-20% overlap
   - Choose embedding model (start with text-embedding-3-small for cost, upgrade if needed)
   - Store in vector database (Qdrant/Weaviate for production, Chroma for prototyping)
3. **Retrieval pipeline**:
   - Implement hybrid search (BM25 + dense) for best coverage
   - Add metadata filtering (department, date, document type)
   - Implement re-ranking with cross-encoder for precision
   - Set up MMR for diversity
4. **Generation**:
   - Design prompts with clear instructions to stay grounded in context
   - Include source citation requirements
   - Add guardrails for hallucination detection
5. **Evaluation**: Set up RAGAS metrics, human evaluation pipeline
6. **Production concerns**: Caching, rate limiting, monitoring, feedback loop

### Q: "What is the context window problem in RAG?"

**Answer**: Even with large context windows (128K+ tokens), challenges remain:
- **Lost in the middle**: LLMs perform worse on information in the middle of long contexts (Liu et al., 2023). Information at the beginning and end gets more attention.
- **Cost**: More tokens = higher API costs. 100K tokens per query is expensive at scale.
- **Latency**: Processing large contexts is slower.
- **Noise**: More retrieved chunks may include irrelevant information that confuses the model.
- **Solution**: Retrieve fewer, higher-quality chunks. Use re-ranking to put best chunks first. Use contextual compression to remove irrelevant parts.

### Q: "How does RAG handle real-time or frequently updated data?"

**Answer**:
- **Incremental indexing**: Only embed and store new/changed documents, not re-index everything
- **Document versioning**: Track document versions in metadata, filter for latest
- **TTL (Time-to-Live)**: Set expiration on vectors for time-sensitive data
- **Streaming ingestion**: Use message queues (Kafka, RabbitMQ) to process document updates
- **Cache invalidation**: Invalidate cached responses when source documents change
- **Hybrid approach**: Use RAG for stable knowledge, tool/API calls for real-time data (stock prices, weather)
