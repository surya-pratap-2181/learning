---
title: "Part 3 - Database Comparison"
layout: default
parent: "Vector Databases & Embeddings"
nav_order: 4
---

# PART 3: VECTOR DATABASE COMPARISON

---

## 3.1 Overview of the Vector Database Landscape

Vector databases are purpose-built or adapted systems for storing, indexing, and querying high-dimensional vector embeddings. The market has matured significantly in 2024-2025.

**Categories:**
1. **Purpose-built vector databases:** Pinecone, Weaviate, Qdrant, Milvus, Chroma
2. **Vector-capable extensions:** pgvector (PostgreSQL), Elasticsearch kNN
3. **Vector libraries (not databases):** FAISS, Annoy, ScaNN, hnswlib

---

## 3.2 Pinecone

### Overview:
- Fully managed, cloud-native vector database
- Founded by Edo Liberty (former Amazon director of AI research)
- Serverless architecture launched in 2024

### Architecture:
- Serverless: Pay per usage (reads, writes, storage) rather than provisioned capacity
- Pod-based: Dedicated compute (legacy pricing model)
- Supports namespaces for multi-tenancy within an index

### Key Features:
- **Serverless indexes**: Auto-scaling, zero infrastructure management
- **Namespaces**: Logical partitions within an index
- **Sparse-dense vectors**: Native hybrid search support
- **Metadata filtering**: Rich filtering with multiple operators
- **Collections**: Snapshots/backups of indexes

### Pricing (2025):
- **Serverless**: ~$0.33/GB storage/month + $8.25/1M read units + $2/1M write units
- **Pod-based (Standard)**: From ~$70/month per pod
- **Free tier**: 1 index, 2GB storage, 100K vectors

### Pros:
- Easiest to get started (fully managed)
- Excellent documentation and developer experience
- Serverless pricing is cost-effective for variable workloads
- Strong enterprise features (SSO, encryption, audit logs)
- Sparse-dense vectors for hybrid search

### Cons:
- Vendor lock-in (proprietary, closed-source)
- Limited control over index configuration
- Can be expensive at scale vs self-hosted options
- No on-premises deployment option
- Limited to cloud regions offered by Pinecone

### Best For:
- Teams wanting zero operational overhead
- Startups and prototypes
- Applications with variable query patterns

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_KEY")

# Create serverless index
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("my-index")

# Upsert vectors with metadata
index.upsert(
    vectors=[
        {"id": "doc1", "values": [0.1, 0.2, ...], "metadata": {"genre": "sci-fi", "year": 2024}},
        {"id": "doc2", "values": [0.3, 0.4, ...], "metadata": {"genre": "drama", "year": 2023}},
    ],
    namespace="movies"
)

# Query with metadata filter
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    filter={"genre": {"$eq": "sci-fi"}, "year": {"$gte": 2023}},
    namespace="movies",
    include_metadata=True
)

# Hybrid search (sparse-dense)
from pinecone import SparseValues
results = index.query(
    vector=[0.1, 0.2, ...],          # Dense vector
    sparse_vector=SparseValues(
        indices=[10, 45, 128],         # Token positions
        values=[0.5, 0.3, 0.8]        # Token weights (BM25 or SPLADE)
    ),
    top_k=10
)
```

---

## 3.3 Weaviate

### Overview:
- Open-source vector database (BSL license, later Apache 2.0 for older versions)
- Written in Go
- Founded in Amsterdam (2019)
- Combines vector search with structured data capabilities

### Architecture:
- Modular vectorizer: Can use built-in (module) or external embedding models
- HNSW with custom optimizations + flat indexes
- GraphQL and REST API
- Multi-tenancy support

### Key Features:
- **Vectorizer modules**: Built-in integrations with OpenAI, Cohere, Hugging Face, etc. (auto-embeds text)
- **Hybrid search**: BM25 + vector search with configurable alpha
- **Generative search**: Built-in RAG modules (generative-openai, etc.)
- **Multi-tenancy**: First-class support for SaaS applications
- **Classification**: Built-in kNN and zero-shot classification
- **Schema-based**: Strong typing with class/property definitions

### Pricing (Weaviate Cloud):
- **Sandbox**: Free, 14-day expiry
- **Serverless**: Pay-per-use from ~$25/month
- **Enterprise**: Custom pricing
- **Self-hosted**: Free (open source), bring your own infrastructure

### Pros:
- Built-in vectorization (no separate embedding pipeline needed)
- Excellent hybrid search implementation
- Rich module ecosystem (generative, reranker, etc.)
- Active open-source community
- GraphQL API is powerful for complex queries
- Multi-tenancy designed for SaaS

### Cons:
- GraphQL learning curve
- Module system adds complexity
- Resource-heavy for small deployments
- Schema changes can be difficult
- Newer compared to traditional databases (less battle-tested)

### Best For:
- Applications needing built-in vectorization
- Hybrid search use cases
- Multi-tenant SaaS applications
- RAG applications (built-in generative modules)

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery, Filter

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://your-cluster.weaviate.network",
    auth_credentials=weaviate.auth.AuthApiKey("YOUR_KEY")
)

# Create collection with vectorizer module
collection = client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
    generative_config=Configure.Generative.openai(),
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="category", data_type=DataType.TEXT),
    ]
)

# Add objects (auto-vectorized)
collection = client.collections.get("Document")
collection.data.insert({"content": "Vector databases are great", "category": "tech"})

# Semantic search
results = collection.query.near_text(
    query="database for AI",
    limit=5,
    return_metadata=MetadataQuery(distance=True)
)

# Hybrid search (BM25 + vector)
results = collection.query.hybrid(
    query="vector database performance",
    alpha=0.5,  # 0 = pure BM25, 1 = pure vector
    limit=10,
    filters=Filter.by_property("category").equal("tech")
)

# Generative search (RAG)
results = collection.generate.near_text(
    query="vector databases",
    grouped_task="Summarize the key points about vector databases",
    limit=5
)
```

---

## 3.4 Chroma

### Overview:
- Open-source embedding database, Apache 2.0 license
- Python-first, designed for simplicity
- Founded 2022, popular in the LangChain/LlamaIndex ecosystem
- Positioning: "The AI-native open-source embedding database"

### Architecture:
- Embedded mode (in-process, like SQLite for vectors) or client-server
- Uses HNSW (hnswlib) under the hood
- SQLite for metadata storage
- Supports persistent storage

### Key Features:
- **Simplicity**: Minimal API surface, easy to get started
- **Embedded mode**: Runs in-process, no separate server needed
- **Built-in embedding functions**: OpenAI, Cohere, Sentence Transformers, etc.
- **Where filtering**: Simple metadata filtering syntax
- **Document storage**: Stores original text alongside embeddings

### Pricing:
- Fully open source (Apache 2.0)
- Chroma Cloud: Managed service (launched late 2024/2025)

### Pros:
- Simplest API of any vector database
- Perfect for prototyping and local development
- Embedded mode (no server needed)
- Strong Python ecosystem integration
- Lightweight, minimal dependencies

### Cons:
- Not designed for large-scale production (historically)
- Limited distributed capabilities
- Fewer features than Pinecone/Weaviate
- Single-node architecture (Chroma Cloud addresses this)
- Limited language support (Python-focused)
- No native hybrid search (keyword + vector)

### Best For:
- Prototyping and experimentation
- Local development and testing
- Small to medium-scale applications
- LangChain/LlamaIndex projects
- Educational projects

```python
import chromadb
from chromadb.utils import embedding_functions

# In-memory client
client = chromadb.Client()

# Persistent client
client = chromadb.PersistentClient(path="./chroma_data")

# With OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="YOUR_KEY",
    model_name="text-embedding-3-small"
)

# Create collection
collection = client.create_collection(
    name="documents",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}  # Distance metric
)

# Add documents (auto-embedded)
collection.add(
    documents=["Vector databases store embeddings",
               "HNSW is an efficient search algorithm",
               "RAG improves LLM accuracy"],
    metadatas=[{"topic": "databases"}, {"topic": "algorithms"}, {"topic": "llm"}],
    ids=["doc1", "doc2", "doc3"]
)

# Query
results = collection.query(
    query_texts=["How do vector databases work?"],
    n_results=2,
    where={"topic": "databases"}  # Metadata filter
)

# Query with embedding directly
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
    where={"$and": [
        {"topic": {"$eq": "databases"}},
        {"year": {"$gte": 2023}}
    ]}
)
```

---

## 3.5 Qdrant

### Overview:
- Open-source vector database (Apache 2.0)
- Written in Rust (high performance, memory safety)
- Founded in Berlin (2021)
- Strong focus on filtering and payload-based search

### Architecture:
- Written in Rust for maximum performance
- Custom HNSW implementation with payload filtering
- Supports on-disk storage with memory-mapped files
- gRPC + REST API

### Key Features:
- **Advanced filtering**: Filtering is integrated into the HNSW graph traversal (not post-filter)
- **Payload (metadata)**: Rich payload types and indexing
- **Quantization**: Scalar, product, and binary quantization built-in
- **On-disk storage**: Memory-mapped vectors for large datasets
- **Multivector support**: ColBERT-style late interaction models
- **Sparse vectors**: Native sparse vector support for hybrid search
- **Snapshot and replication**: Built-in distributed deployment

### Pricing:
- Open source: Free (self-hosted)
- Qdrant Cloud: From ~$25/month (managed)
- Free tier: 1GB free cluster

### Pros:
- Excellent filtering performance (filter-aware HNSW)
- Written in Rust (fast, safe, low memory overhead)
- Rich quantization options (reduce memory 4-32x)
- Good documentation and growing community
- Supports on-disk vectors (handle larger-than-RAM datasets)
- Native sparse vector and multivector support

### Cons:
- Smaller community than Weaviate/Milvus
- Fewer built-in integrations than Weaviate
- Newer product (less battle-tested at extreme scale)
- No built-in vectorization module

### Best For:
- Applications with heavy metadata filtering
- Performance-critical applications
- Teams comfortable with Rust ecosystem
- Cost-sensitive deployments (efficient memory usage)
- ColBERT-style retrieval

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    QuantizationConfig, ScalarQuantization, ScalarType
)

client = QdrantClient(url="http://localhost:6333")

# Create collection with quantization
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantization(type=ScalarType.INT8, quantile=0.99, always_ram=True)
    )
)

# Insert points
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1, vector=[0.1, 0.2, ...],
            payload={"title": "Vector DB Guide", "category": "tech", "rating": 4.5}
        ),
        PointStruct(
            id=2, vector=[0.3, 0.4, ...],
            payload={"title": "ML Interview Prep", "category": "career", "rating": 4.8}
        ),
    ]
)

# Search with filtering
results = client.query_points(
    collection_name="documents",
    query=[0.1, 0.2, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="tech")),
            FieldCondition(key="rating", range=Range(gte=4.0)),
        ]
    ),
    limit=10,
    with_payload=True
)

# Named vectors (multiple vectors per point)
client.create_collection(
    collection_name="multimodal",
    vectors_config={
        "text": VectorParams(size=768, distance=Distance.COSINE),
        "image": VectorParams(size=512, distance=Distance.COSINE),
    }
)
```

---

## 3.6 Milvus / Zilliz

### Overview:
- Open-source vector database (Apache 2.0)
- Written in Go + C++
- Originally from Zilliz (commercial managed version = Zilliz Cloud)
- Designed for billion-scale vector search

### Architecture:
- Cloud-native, distributed architecture
- Separates storage, computing, and coordination
- Uses etcd for metadata, MinIO/S3 for object storage, Pulsar/Kafka for log streaming
- Multiple index types: HNSW, IVF_FLAT, IVF_PQ, IVF_SQ8, DiskANN, GPU indexes

### Key Features:
- **Multi-index support**: Most index types of any vector database
- **GPU acceleration**: Native GPU support for index building and search
- **Schema enforcement**: Strong typing with primary keys
- **Partition key**: Automatic data partitioning for multi-tenancy
- **Dynamic schema**: Add fields without rebuilding
- **Consistency levels**: Strong, bounded staleness, session, eventual
- **CDC (Change Data Capture)**: Stream changes for downstream processing

### Pricing:
- Open source: Free (self-hosted)
- Milvus Lite: Embedded (like Chroma, for dev)
- Zilliz Cloud Serverless: From free tier to pay-per-use
- Zilliz Cloud Dedicated: From ~$65/month

### Pros:
- Most mature distributed vector database architecture
- Widest range of index types (including GPU indexes)
- Proven at billion-scale deployments
- Strong consistency guarantees
- Active large community (20K+ GitHub stars)
- Milvus Lite for embedded/development use

### Cons:
- Complex deployment (many components: etcd, MinIO, Pulsar)
- Steep learning curve
- Heavy resource requirements for distributed mode
- Schema changes require careful planning
- Overkill for small-scale use cases

### Best For:
- Billion-scale vector search
- Enterprise deployments requiring distributed architecture
- GPU-accelerated search
- Applications needing multiple index types
- Teams with infrastructure expertise

```python
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)

# Connect
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields, description="Document collection")

# Create collection
collection = Collection("documents", schema)

# Insert data
import numpy as np
data = [
    ["Vector DB Guide", "ML Interview Prep"],       # title
    ["tech", "career"],                               # category
    np.random.rand(2, 768).tolist(),                  # embedding
]
collection.insert(data)

# Build index
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 32, "efConstruction": 200}
}
collection.create_index("embedding", index_params)

# Load and search
collection.load()
search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr='category == "tech"',  # Metadata filter
    output_fields=["title", "category"]
)
```

---

## 3.7 pgvector (PostgreSQL Extension)

### Overview:
- Open-source PostgreSQL extension for vector similarity search
- Adds a `vector` data type and similarity search operators
- Keeps vectors alongside your existing relational data

### Key Features:
- **PostgreSQL native**: Full SQL support, ACID transactions, joins
- **Index types**: IVFFlat, HNSW (added in v0.5.0)
- **Distance functions**: L2 (<->), inner product (<#>), cosine (<=>)
- **Exact and approximate search**: Choose per query
- **Halfvec**: Half-precision (float16) vectors for 2x storage savings

### Pros:
- No new infrastructure (just a PostgreSQL extension)
- Full ACID transactions
- SQL joins with other tables
- Familiar PostgreSQL ecosystem (backup, replication, monitoring)
- Perfect for applications already using PostgreSQL
- Strong consistency guarantees

### Cons:
- Single-node PostgreSQL limits (vertical scaling only without Citus)
- Slower than purpose-built vector databases at scale
- HNSW memory usage can be high
- Limited to PostgreSQL-compatible features
- No built-in distributed vector search (without Citus/AlloyDB)
- Tuning requires PostgreSQL expertise

### Best For:
- Applications already using PostgreSQL
- Small to medium scale (< 10M vectors typically)
- When ACID transactions are needed for vectors + metadata
- When you want a single database for everything
- Rapid prototyping with existing infrastructure

```sql
-- Install extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    category TEXT,
    embedding VECTOR(1536)  -- 1536-dimensional vector
);

-- Insert data
INSERT INTO documents (title, content, category, embedding)
VALUES ('Vector DB Guide', 'Content here...', 'tech', '[0.1, 0.2, ...]');

-- Create HNSW index
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Similarity search with cosine distance
SELECT id, title, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
WHERE category = 'tech'
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- Set ef_search for HNSW
SET hnsw.ef_search = 100;

-- L2 distance search
SELECT * FROM documents
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- Inner product search (negate for ORDER BY since we want max)
SELECT * FROM documents
ORDER BY embedding <#> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

```python
# Python with psycopg2/SQLAlchemy
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)

cur = conn.cursor()

# Insert
cur.execute(
    "INSERT INTO documents (title, embedding) VALUES (%s, %s)",
    ("Vector DB Guide", np.array([0.1, 0.2, ...]))
)

# Search
cur.execute("""
    SELECT id, title, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 10
""", (query_embedding, query_embedding))

results = cur.fetchall()
```

---

## 3.8 FAISS (Facebook AI Similarity Search)

### Overview:
- **Library, NOT a database** (no persistence, no CRUD, no metadata)
- Written in C++ with Python bindings
- Created by Meta AI Research
- The de facto standard for vector search research and benchmarking

### Key Features:
- **Comprehensive index types**: Flat, IVF, HNSW, PQ, LSH, and combinations
- **GPU support**: First-class GPU acceleration
- **Composite indexes**: Combine approaches (e.g., IVF + PQ + Reranking)
- **Quantization**: PQ, OPQ, SQ (scalar), binary

### Pros:
- Fastest raw search performance (especially on GPU)
- Most flexible index configuration
- Excellent for research and benchmarking
- Very memory-efficient with quantization
- Mature and well-tested (used at Meta scale)

### Cons:
- **Not a database**: No metadata filtering, no CRUD, no persistence out of the box
- Requires custom infrastructure for production use
- No distributed search natively
- API can be complex
- No built-in replication or sharding

### Best For:
- Research and benchmarking
- Embedded in custom applications
- GPU-accelerated search
- When you need maximum control over index configuration
- Backend for other systems (many vector DBs use FAISS internally)

```python
import faiss
import numpy as np

d = 768          # Dimension
nb = 1000000     # Database size
nq = 100         # Number of queries

# Generate data
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# === Various Index Types ===

# 1. Flat (brute force)
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)

# 2. IVF + Flat
nlist = 1024
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(xb)
index_ivf.add(xb)
index_ivf.nprobe = 64

# 3. IVF + PQ
m = 96  # Sub-vectors
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index_ivfpq.train(xb)
index_ivfpq.add(xb)

# 4. HNSW
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32 = M parameter
index_hnsw.hnsw.efSearch = 128
index_hnsw.add(xb)

# 5. Composite index via factory string
# "IVF1024,PQ96" = IVF with 1024 centroids + PQ with 96 sub-vectors
index = faiss.index_factory(d, "IVF1024,PQ96", faiss.METRIC_L2)
index.train(xb)
index.add(xb)

# 6. GPU index
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)

# Search
D, I = index.search(xq, k=10)  # D=distances, I=indices

# Save/Load
faiss.write_index(index, "my_index.faiss")
index = faiss.read_index("my_index.faiss")
```

---

## 3.9 Elasticsearch with Vectors

### Overview:
- Elasticsearch added kNN search capabilities (v8.0+)
- Uses HNSW algorithm
- Combines traditional full-text search with vector search

### Key Features:
- **Hybrid search**: Native BM25 + kNN in a single query
- **Dense vector field type**: `dense_vector` with `dims` and `similarity`
- **Approximate and exact kNN**: HNSW-based ANN or brute-force script scoring
- **Existing ecosystem**: Kibana, Logstash, Beats, observability tools
- **ELSER**: Elastic's own sparse embedding model for semantic search

### Pros:
- Unified search platform (text + vector + structured)
- Massive existing ecosystem and community
- Battle-tested at enterprise scale
- Excellent full-text search capabilities
- Managed offerings (Elastic Cloud, AWS OpenSearch)
- Rich aggregation and analytics

### Cons:
- Not optimized primarily for vector search (bolted on)
- Higher memory/storage overhead than purpose-built vector DBs
- Vector search performance lags behind Qdrant, Milvus
- Complex cluster management
- Expensive at scale
- JVM-based (memory management challenges)

### Best For:
- Organizations already using Elasticsearch
- Hybrid text + vector search
- When you need full-text search features alongside vectors
- Log analytics + semantic search combination

```json
// Create index with dense vector field
PUT /documents
{
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "category": { "type": "keyword" },
      "embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "hnsw",
          "m": 16,
          "ef_construction": 200
        }
      }
    }
  }
}

// kNN search
POST /documents/_search
{
  "knn": {
    "field": "embedding",
    "query_vector": [0.1, 0.2, ...],
    "k": 10,
    "num_candidates": 100,
    "filter": {
      "term": { "category": "tech" }
    }
  }
}

// Hybrid search (BM25 + kNN)
POST /documents/_search
{
  "query": {
    "bool": {
      "should": [
        { "match": { "title": "vector database" } }
      ]
    }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [0.1, 0.2, ...],
    "k": 10,
    "num_candidates": 100
  },
  "rank": {
    "rrf": {}  // Reciprocal Rank Fusion to combine scores
  }
}
```

---

## 3.10 Comparison Matrix

| Feature | Pinecone | Weaviate | Chroma | Qdrant | Milvus | pgvector | FAISS | Elasticsearch |
|---------|----------|----------|--------|--------|--------|----------|-------|---------------|
| **Type** | Managed DB | Open-source DB | Open-source DB | Open-source DB | Open-source DB | PG Extension | Library | Search Engine |
| **License** | Proprietary | BSL/Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 | PostgreSQL | MIT | SSPL/Elastic |
| **Language** | - | Go | Python | Rust | Go/C++ | C | C++ | Java |
| **Self-hosted** | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Managed** | Yes | Yes | Coming | Yes | Yes (Zilliz) | Yes (various) | No | Yes |
| **Max Scale** | Billions | 100M+ | Millions | 100M+ | Billions | ~10M | Billions | 100M+ |
| **Hybrid Search** | Yes (sparse) | Yes (BM25) | No | Yes (sparse) | Yes | No (native) | No | Yes (BM25) |
| **Metadata Filter** | Yes | Yes | Yes | Yes (best) | Yes | Yes (SQL) | No | Yes |
| **GPU Support** | No | No | No | No | Yes | No | Yes | No |
| **ACID** | No | No | No | No | No | Yes | No | No |
| **Quantization** | Auto | PQ, BQ | No | Scalar, PQ, BQ | SQ8, PQ | halfvec | PQ, SQ, OPQ | No |
| **Multi-vector** | No | No | No | Yes | No | No | No | No |
| **Ease of Use** | 10/10 | 7/10 | 9/10 | 8/10 | 6/10 | 7/10 | 5/10 | 6/10 |
| **Community** | Medium | Large | Large | Growing | Very Large | Large | Very Large | Massive |
