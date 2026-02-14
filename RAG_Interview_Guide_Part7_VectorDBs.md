---
title: "Part 7 - Vector DBs"
layout: default
parent: "RAG Systems"
nav_order: 8
---


# RAG Interview Guide
# PART 7: VECTOR DATABASE COMPARISON

---

## 1. Overview of Vector Databases

Vector databases are purpose-built to store, index, and query high-dimensional vector embeddings efficiently. They are the backbone of RAG retrieval.

**Key Requirements:**
- Fast approximate nearest neighbor (ANN) search
- Scalability to millions/billions of vectors
- Metadata filtering alongside vector search
- CRUD operations (add, update, delete vectors)
- Persistence and durability
- Optional: hybrid search (vector + keyword), multi-tenancy, access control

---

## 2. Detailed Comparison

### 2.1 FAISS (Facebook AI Similarity Search)

**Type**: Library (not a database)
**License**: MIT (open source)
**Language**: C++ with Python bindings

**Strengths:**
- Extremely fast - optimized C++ with GPU support
- Many index types (Flat, IVF, HNSW, PQ, and combinations)
- No server needed - runs in-process
- Best for benchmarking and research
- Supports billion-scale with IVF+PQ+OPQ

**Weaknesses:**
- Not a database: no built-in persistence, CRUD, filtering, replication
- No metadata filtering (must implement separately)
- No built-in API server
- Manual index management
- No multi-tenancy

**Best for**: Research, prototyping, offline batch processing, when you need raw speed and control.

```python
import faiss
import numpy as np

# Create index
dimension = 1536
index = faiss.IndexFlatL2(dimension)  # Exact search (brute force)

# IVF index for large scale
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(training_vectors)  # Must train IVF index
index.add(vectors)

# HNSW index
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64
index.add(vectors)

# Search
distances, indices = index.search(query_vector.reshape(1, -1), k=5)

# Save/Load
faiss.write_index(index, "my_index.faiss")
index = faiss.read_index("my_index.faiss")

# GPU acceleration
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

# With LangChain
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
results = vectorstore.similarity_search("query", k=5)
```

**Index Type Decision Tree:**
```
< 10K vectors: IndexFlatL2 (exact search)
10K - 1M vectors: IndexHNSWFlat (best recall/speed)
1M - 100M vectors: IndexIVFFlat or IndexIVF,PQ (with nprobe tuning)
> 100M vectors: IndexIVF,PQ with OPQ preprocessing
GPU available: Use GPU versions of above
```

---

### 2.2 Chroma

**Type**: Embedded database (client-server mode available)
**License**: Apache 2.0 (open source)
**Language**: Python

**Strengths:**
- Simplest API - get started in 5 lines of code
- Great for prototyping and development
- Built-in persistence
- Metadata filtering
- Automatic embedding (pass text, Chroma embeds it)
- Active open-source community

**Weaknesses:**
- Not designed for production scale (limited to ~1M vectors efficiently)
- No distributed mode / horizontal scaling
- Limited index type options
- No built-in hybrid search
- Limited access control

**Best for**: Prototyping, small-medium projects, learning RAG, local development.

```python
import chromadb

# Persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # Distance metric
)

# Add documents (Chroma can embed automatically)
collection.add(
    documents=["Document 1 text", "Document 2 text"],
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}],
    ids=["id1", "id2"]
)

# Or add pre-computed embeddings
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["Document 1", "Document 2"],
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}],
    ids=["id1", "id2"]
)

# Query with metadata filtering
results = collection.query(
    query_texts=["What is RAG?"],
    n_results=5,
    where={"source": "doc1.pdf"},
    where_document={"$contains": "retrieval"}  # Full-text filter
)

# Update
collection.update(ids=["id1"], documents=["Updated text"])

# Delete
collection.delete(ids=["id1"])

# With LangChain
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)
```

---

### 2.3 Pinecone

**Type**: Fully managed cloud service
**License**: Proprietary (cloud service)

**Strengths:**
- Fully managed - no infrastructure to manage
- Scales to billions of vectors
- Fast query latency (~50ms p95)
- Built-in metadata filtering
- Namespaces for multi-tenancy
- Serverless pricing option (pay per query)
- Built-in hybrid search (sparse-dense)
- SOC 2, HIPAA compliant
- Free tier available

**Weaknesses:**
- Vendor lock-in (proprietary)
- No self-hosted option
- Can be expensive at scale
- Limited customization of index parameters
- Data must be sent to Pinecone's cloud
- No built-in embedding (bring your own vectors)

**Best for**: Production deployments, teams that want managed infrastructure, enterprise with compliance needs.

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index("my-index")

# Upsert vectors
index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1, 0.2, ...],  # 1536-dim vector
            "metadata": {
                "source": "doc1.pdf",
                "page": 5,
                "category": "engineering"
            }
        },
        # ... more vectors
    ],
    namespace="my-namespace"
)

# Query with metadata filtering
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={
        "category": {"$eq": "engineering"},
        "page": {"$gte": 1, "$lte": 10}
    },
    include_metadata=True,
    namespace="my-namespace"
)

# Hybrid search (sparse + dense)
from pinecone_text.sparse import BM25Encoder

bm25 = BM25Encoder.default("english")
sparse_vector = bm25.encode_queries("What is RAG?")

results = index.query(
    vector=[0.1, 0.2, ...],           # Dense vector
    sparse_vector=sparse_vector,       # Sparse vector
    top_k=5,
    alpha=0.5  # Balance dense vs sparse
)

# With LangChain
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="my-namespace"
)
```

**Pricing (as of 2025):**
- Serverless: ~$0.01 per 1000 queries, storage based on vectors
- Pod-based: Starts ~$70/month for s1.x1 pod
- Free tier: 1 index, 100K vectors

---

### 2.4 Weaviate

**Type**: Open-source database with managed cloud option
**License**: BSD-3-Clause (open source)

**Strengths:**
- Native hybrid search (BM25 + vector) built in
- GraphQL API (flexible queries)
- Module system (plug in different embedding models, rerankers)
- Multi-tenancy support
- Built-in vectorization (can embed text automatically using modules)
- Generative search module (RAG built into the database)
- Supports named vectors (multiple vector spaces per object)
- Both self-hosted and managed cloud options

**Weaknesses:**
- More complex setup than Chroma/Pinecone
- GraphQL learning curve
- Resource intensive (memory)
- Managed cloud can be expensive

**Best for**: Production systems needing hybrid search, teams wanting open-source with managed option, multi-modal RAG.

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

# Connect to local instance
client = weaviate.connect_to_local()

# Or connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="your-cluster-url",
    auth_credentials=weaviate.auth.AuthApiKey("your-api-key")
)

# Create collection (class)
collection = client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small"
    ),
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="page", data_type=DataType.INT),
    ]
)

# Add objects (Weaviate auto-embeds using configured vectorizer)
collection = client.collections.get("Document")
collection.data.insert({
    "content": "RAG stands for Retrieval Augmented Generation...",
    "source": "rag_guide.pdf",
    "page": 1
})

# Vector search
response = collection.query.near_text(
    query="What is RAG?",
    limit=5,
    filters=weaviate.classes.query.Filter.by_property("source").equal("rag_guide.pdf")
)

# Hybrid search (BM25 + vector)
response = collection.query.hybrid(
    query="What is RAG?",
    alpha=0.5,  # 0 = pure BM25, 1 = pure vector
    limit=5
)

# With LangChain
from langchain_weaviate import WeaviateVectorStore

vectorstore = WeaviateVectorStore(
    client=client,
    index_name="Document",
    text_key="content",
    embedding=embeddings
)
```

---

### 2.5 Qdrant

**Type**: Open-source database with managed cloud option
**License**: Apache 2.0 (open source)

**Strengths:**
- Written in Rust (fast and memory efficient)
- Excellent filtering performance (payload indexing)
- Advanced filtering with nested conditions
- Multi-vector support (named vectors)
- Quantization built-in (scalar, binary, product)
- Batch and streaming APIs
- Distributed mode with sharding and replication
- On-disk storage for cost-effective large deployments
- Both gRPC and REST APIs
- Active development and community

**Weaknesses:**
- No built-in hybrid search (must implement BM25 separately)
- Newer than some competitors (smaller ecosystem)
- Self-hosted requires Rust compilation knowledge for custom builds

**Best for**: Production deployments needing high-performance filtering, cost-effective large-scale deployments, teams wanting open-source with good cloud option.

```python
from qdrant_client import QdrantClient, models

# Connect
client = QdrantClient(url="http://localhost:6333")  # Local
# client = QdrantClient(url="https://xyz.cloud.qdrant.io", api_key="key")  # Cloud

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE
    ),
    # Enable quantization for efficiency
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True
        )
    )
)

# Upsert points
client.upsert(
    collection_name="documents",
    points=[
        models.PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={
                "content": "RAG combines retrieval with generation...",
                "source": "doc1.pdf",
                "page": 5,
                "department": "engineering",
                "date": "2024-01-15"
            }
        ),
        # ... more points
    ]
)

# Search with filtering
results = client.query_points(
    collection_name="documents",
    query=[0.1, 0.2, ...],  # query vector
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="department",
                match=models.MatchValue(value="engineering")
            ),
            models.FieldCondition(
                key="date",
                range=models.Range(gte="2024-01-01")
            )
        ]
    ),
    limit=5
)

# Multi-vector (named vectors)
client.create_collection(
    collection_name="multimodal",
    vectors_config={
        "text": models.VectorParams(size=1536, distance=models.Distance.COSINE),
        "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
    }
)

# With LangChain
from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="documents",
    embedding=embeddings
)
```

**Pricing (Qdrant Cloud):**
- Free tier: 1GB storage, 1 node
- Starter: ~$25/month
- Pay per use based on RAM/storage/compute

---

### 2.6 Milvus / Zilliz

**Type**: Open-source distributed database (Zilliz = managed cloud)
**License**: Apache 2.0 (open source)

**Strengths:**
- Purpose-built for billion-scale vector search
- Cloud-native, distributed architecture
- Multiple index types (IVF, HNSW, DiskANN, GPU indexes)
- Hybrid search (sparse + dense)
- Attribute filtering
- Multi-vector and dynamic schema
- GPU-accelerated indexing and search
- Milvus Lite for embedded/edge use cases

**Weaknesses:**
- Complex deployment (etcd, MinIO, Pulsar dependencies for distributed mode)
- Heavier resource requirements than simpler options
- Steeper learning curve
- Over-engineered for small-scale use

**Best for**: Billion-scale deployments, organizations needing distributed vector search, GPU-accelerated workloads.

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Connect
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
]
schema = CollectionSchema(fields, description="Document collection")

# Create collection
collection = Collection("documents", schema)

# Insert data
collection.insert([
    ["Document 1 text", "Document 2 text"],  # content
    ["doc1.pdf", "doc2.pdf"],                 # source
    [[0.1, 0.2, ...], [0.3, 0.4, ...]],     # embeddings
])

# Create index
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index("embedding", index_params)

# Load to memory and search
collection.load()
results = collection.search(
    data=[[0.1, 0.2, ...]],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5,
    expr='source == "doc1.pdf"',  # Metadata filtering
    output_fields=["content", "source"]
)

# Milvus Lite (embedded mode for dev)
from pymilvus import MilvusClient
client = MilvusClient("./milvus.db")  # SQLite-like local file
```

---

### 2.7 pgvector (PostgreSQL Extension)

**Type**: Extension for PostgreSQL
**License**: PostgreSQL License (open source)

**Strengths:**
- Leverages existing PostgreSQL infrastructure and expertise
- Full SQL capabilities (joins, aggregations, transactions)
- ACID compliant
- Combine vector search with relational queries
- No additional infrastructure needed
- Support for IVFFlat and HNSW indexes
- Halfvec support (float16) for memory savings
- Great for teams already using PostgreSQL

**Weaknesses:**
- Not as fast as purpose-built vector databases for pure vector search
- Limited to single-node PostgreSQL performance
- No built-in distributed mode for vectors (though PG has Citus for distribution)
- Index build times can be slow for large datasets
- Memory management shared with all PostgreSQL operations

**Best for**: Teams already on PostgreSQL, applications needing ACID + vector search, when you want to avoid another database.

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    source VARCHAR(200),
    page INTEGER,
    department VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    embedding vector(1536)
);

-- Insert
INSERT INTO documents (content, source, embedding)
VALUES ('RAG combines retrieval with generation...', 'doc1.pdf', '[0.1, 0.2, ...]');

-- Create HNSW index (recommended for most cases)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Or IVFFlat index (faster build, slightly less accurate)
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Query: vector search + SQL filtering
SELECT content, source, page,
       1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
WHERE department = 'engineering'
  AND created_at > '2024-01-01'
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;

-- Hybrid: combine with full-text search
SELECT content, source,
       ts_rank(to_tsvector('english', content), plainto_tsquery('english', 'RAG retrieval')) AS text_score,
       1 - (embedding <=> query_embedding) AS vector_score
FROM documents,
     (SELECT '[0.1, 0.2, ...]'::vector AS query_embedding) q
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'RAG retrieval')
ORDER BY (0.5 * ts_rank(to_tsvector('english', content), plainto_tsquery('english', 'RAG retrieval'))
         + 0.5 * (1 - (embedding <=> query_embedding)))
        DESC
LIMIT 5;
```

```python
# With LangChain
from langchain_postgres import PGVector

CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/vectordb"

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="documents",
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# Add documents
vectorstore.add_documents(documents)

# Search with metadata filter
results = vectorstore.similarity_search(
    "What is RAG?",
    k=5,
    filter={"department": "engineering"}
)
```

---

## 3. Head-to-Head Comparison Table

| Feature | FAISS | Chroma | Pinecone | Weaviate | Qdrant | Milvus | pgvector |
|---------|-------|--------|----------|----------|--------|--------|----------|
| **Type** | Library | Embedded DB | Managed Cloud | Open Source + Cloud | Open Source + Cloud | Open Source + Cloud | PG Extension |
| **License** | MIT | Apache 2.0 | Proprietary | BSD-3 | Apache 2.0 | Apache 2.0 | PostgreSQL |
| **Language** | C++/Python | Python | - | Go | Rust | Go/C++ | C |
| **Max Scale** | Billions (manual) | ~1M easy | Billions | Billions | Billions | Billions | Millions |
| **Self-hosted** | Yes | Yes | No | Yes | Yes | Yes | Yes |
| **Managed Cloud** | No | No (limited) | Yes | Yes | Yes | Yes (Zilliz) | Many PG providers |
| **Hybrid Search** | No | No | Yes | Yes (native) | No (manual) | Yes | Yes (with FTS) |
| **Metadata Filter** | No | Yes | Yes | Yes | Yes (excellent) | Yes | Yes (full SQL) |
| **Multi-tenancy** | No | Limited | Namespaces | Yes | Collection-based | Yes | Schema/RLS |
| **ACID** | No | No | No | No | No | No | Yes |
| **GPU Support** | Yes (excellent) | No | N/A | No | No | Yes | No |
| **Quantization** | PQ, SQ, OPQ | No | Yes | BQ, PQ | Scalar, Binary, PQ | Yes (all types) | Halfvec |
| **Free Tier** | Free (OSS) | Free (OSS) | 100K vectors | Free (OSS) | 1GB cloud | Free (OSS) | Free (OSS) |
| **Ease of Setup** | Easy (pip) | Very Easy | Very Easy | Moderate | Easy | Complex | Easy (if PG exists) |
| **Production Ready** | Yes (as library) | Limited | Yes | Yes | Yes | Yes | Yes |

---

## 4. When to Choose What

### Decision Framework:

```
START
  |
  v
Do you already use PostgreSQL?
  |-- YES --> pgvector (simplest, leverages existing infra)
  |-- NO
       |
       v
  Is this a prototype/learning project?
  |-- YES --> Chroma (simplest API, zero config)
  |-- NO
       |
       v
  Do you need managed infrastructure?
  |-- YES --> Do you need hybrid search?
  |            |-- YES --> Pinecone or Weaviate Cloud
  |            |-- NO  --> Pinecone (simplest managed) or Qdrant Cloud
  |-- NO (self-hosted)
       |
       v
  What scale do you need?
  |-- < 10M vectors --> Qdrant or Weaviate (single node)
  |-- 10M - 1B vectors --> Qdrant cluster, Weaviate cluster, or Milvus
  |-- > 1B vectors --> Milvus (purpose-built for billion scale)
       |
       v
  Do you need hybrid search (BM25 + vector)?
  |-- YES --> Weaviate (built-in) or Qdrant + Elasticsearch
  |-- NO  --> Qdrant (fastest filtering) or Milvus (most scalable)
```

### Specific Recommendations by Use Case:

| Use Case | Recommendation | Why |
|----------|---------------|-----|
| Learning/Prototyping | Chroma | Simplest setup, good docs |
| Startup MVP | Pinecone (serverless) | No ops, pay-per-use, fast to market |
| Enterprise (existing PG) | pgvector | No new infra, ACID, familiar tooling |
| Enterprise (new infra) | Qdrant or Weaviate | Open source, production-grade, cloud options |
| Billion-scale | Milvus/Zilliz | Purpose-built for massive scale |
| Research/Benchmarking | FAISS | Raw performance, full control |
| Multi-modal RAG | Weaviate or Qdrant | Named vectors, module system |
| Hybrid search priority | Weaviate | Best native hybrid search |
| Cost-sensitive | Qdrant (self-hosted) | Rust efficiency, on-disk mode |

---

## 5. Interview Questions on Vector Databases

### Q: "How does HNSW indexing work and why is it the most popular index type?"

**Answer:**
HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm:

1. **Construction**: Build a multi-layer graph. Bottom layer contains ALL nodes. Higher layers are subsampled (fewer nodes, longer-range connections).
2. **Search**: Start at the top layer (few nodes, coarse), greedily navigate to the nearest node. Drop down to the next layer and repeat. At the bottom layer, do a more thorough local search.
3. **Key parameters**:
   - `M`: Number of connections per node (higher = better recall, more memory). Typical: 16-64
   - `efConstruction`: Beam width during index building (higher = better index quality, slower build). Typical: 100-200
   - `efSearch`: Beam width during query (higher = better recall, slower search). Typical: 64-128

**Why it's popular**: Best tradeoff between recall (>95% at 1ms latency), speed, and ease of use. No training step needed (unlike IVF). Works well up to ~100M vectors per node.

### Q: "What are the tradeoffs between exact search and approximate nearest neighbor (ANN) search?"

**Answer:**
- **Exact (flat/brute force)**: 100% recall, O(n) per query. Fine for <10K vectors. No index maintenance.
- **ANN (HNSW, IVF, etc.)**: 95-99% recall, O(log n) or better per query. Essential for >100K vectors.
- **Key insight**: For RAG, 95% recall is usually sufficient. The 5% missed results are unlikely to be the ONE critical document. Re-ranking and diverse retrieval strategies further mitigate missed results.

### Q: "How do you handle vector database migrations?"

**Answer:**
1. **Export vectors + metadata** from source DB
2. **Re-embed if necessary** (if changing embedding model)
3. **Batch import** to target DB
4. **Validate**: Run evaluation queries, compare results between old and new
5. **Blue-green deployment**: Run both databases in parallel, gradually shift traffic
6. **Keep embedding model version in metadata** to know which vectors need re-embedding

### Q: "How does metadata filtering interact with vector search performance?"

**Answer:**
Two approaches:
1. **Pre-filtering**: Filter by metadata first, then vector search on filtered set. Fast for selective filters but can miss results if filter is too aggressive.
2. **Post-filtering**: Vector search first (top-N), then filter by metadata. Guarantees vector search quality but may return fewer than k results if many are filtered out.
3. **Integrated filtering (Qdrant, Weaviate)**: Filter and vector search simultaneously using payload indexes. Best performance but database-specific optimization.

**Qdrant's approach is notably efficient** - it uses payload indexes alongside HNSW, filtering during graph traversal rather than before/after.
