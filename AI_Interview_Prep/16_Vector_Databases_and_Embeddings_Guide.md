---
title: "Vector DBs Overview"
layout: default
parent: "Vector Databases & Embeddings"
nav_order: 1
---

# Vector Databases & Embeddings: Complete Interview Guide (2025-2026)
## For AI Engineers Building RAG and Semantic Search Systems

---

# TABLE OF CONTENTS
1. What are Embeddings?
2. Explaining to a Layman
3. Embedding Models Comparison
4. Vector Similarity Metrics
5. ANN Algorithms (HNSW, IVF, PQ)
6. Vector Database Comparison
7. Hybrid Search & Reranking
8. Production Patterns
9. Interview Questions (25+)
10. Code Examples

---

# SECTION 1: WHAT ARE EMBEDDINGS?

Embeddings are dense numerical vectors that capture the semantic meaning of text (or images, audio). They map content into a high-dimensional space where similar items are close together.

```
"king" → [0.23, -0.45, 0.78, ..., 0.12]  (1536 dimensions)
"queen" → [0.22, -0.43, 0.76, ..., 0.11] (very close to "king"!)
"banana" → [-0.67, 0.32, -0.11, ..., 0.89] (far from "king")
```

**Key Property:**
```
cosine_similarity("king", "queen") ≈ 0.95  (very similar)
cosine_similarity("king", "banana") ≈ 0.12  (very different)
```

---

# SECTION 2: EXPLAINING TO A LAYMAN

> **GPS Coordinates for Meaning:**
> Just like GPS coordinates tell you where something is on Earth, embeddings tell you where a piece of text is in "meaning space." Texts about similar topics have coordinates that are close together. Paris and London are close on a map (both European capitals); similarly, "machine learning" and "neural networks" are close in embedding space.

---

# SECTION 3: EMBEDDING MODELS COMPARISON (2025-2026)

| Model | Provider | Dims | Max Tokens | MTEB | Cost/1M tokens | Open Source |
|-------|----------|------|------------|------|----------------|-------------|
| text-embedding-3-large | OpenAI | 3072 | 8191 | 64.6 | $0.13 | No |
| text-embedding-3-small | OpenAI | 1536 | 8191 | 62.3 | $0.02 | No |
| embed-v3 | Cohere | 1024 | 512 | 64.5 | $0.10 | No |
| voyage-3 | Voyage AI | 1024 | 32000 | 67.1 | $0.06 | No |
| BGE-M3 | BAAI | 1024 | 8192 | 66.1 | Free | Yes |
| GTE-Qwen2 | Alibaba | 1536+ | 131072 | 67.2 | Free | Yes |
| E5-Mistral-7B | Microsoft | 4096 | 32768 | 66.6 | Free | Yes |
| all-MiniLM-L6-v2 | SBERT | 384 | 512 | 56.3 | Free | Yes |

**Decision Guide:**
```
Production with budget?     → OpenAI text-embedding-3-small (best value)
Maximum quality?            → Voyage AI voyage-3 or GTE-Qwen2
Self-hosted / privacy?      → BGE-M3 or GTE-Qwen2
Multilingual?               → Cohere embed-v3 or BGE-M3
Long documents (>8K tokens)?→ GTE-Qwen2 (128K context!)
Prototyping / local dev?    → all-MiniLM-L6-v2 (tiny, fast)
```

---

# SECTION 4: VECTOR SIMILARITY METRICS

| Metric | Formula | Range | Best For | Normalized? |
|--------|---------|-------|----------|-------------|
| **Cosine Similarity** | dot(A,B) / (‖A‖·‖B‖) | [-1, 1] | Text similarity | Yes |
| **Dot Product** | Σ(Ai × Bi) | (-∞, ∞) | When magnitude matters | No |
| **Euclidean (L2)** | √Σ(Ai - Bi)² | [0, ∞) | Spatial distance | No |

**When to use which:**
- **Cosine:** Default for text embeddings (most models output normalized vectors)
- **Dot Product:** When vectors are already normalized (equivalent to cosine) or magnitude carries meaning
- **Euclidean:** When absolute distance matters, image embeddings

---

# SECTION 5: ANN ALGORITHMS

Exact nearest neighbor search is O(n) - too slow for millions of vectors. ANN (Approximate Nearest Neighbor) trades a tiny amount of accuracy for massive speed gains.

## 5.1 HNSW (Hierarchical Navigable Small World)

Most popular algorithm. Like a skip list for high-dimensional space:

```
Layer 3: [Node A] ─────────── [Node F]           (few connections, fast)
Layer 2: [Node A] ── [Node C] ── [Node F]
Layer 1: [Node A] ─ [Node B] ─ [Node C] ─ [Node D] ─ [Node F]
Layer 0: [A] [B] [C] [D] [E] [F] [G] [H] ...    (all nodes, precise)
```

| Parameter | Description | Tradeoff |
|-----------|-------------|----------|
| **M** | Connections per node | Higher = better recall, more memory |
| **efConstruction** | Beam width during build | Higher = better quality, slower build |
| **efSearch** | Beam width during query | Higher = better recall, slower query |

**Typical values:** M=16, efConstruction=200, efSearch=100

## 5.2 IVF (Inverted File Index)

Clusters vectors, searches only relevant clusters:

```
Step 1: Cluster all vectors into K centroids
Step 2: For query, find nearest nprobe centroids
Step 3: Search only vectors in those clusters
```

| Parameter | Description |
|-----------|-------------|
| **nlist** | Number of clusters (typically √n) |
| **nprobe** | Clusters to search (higher = better recall) |

## 5.3 Product Quantization (PQ)

Compresses vectors to reduce memory. Splits vector into sub-vectors, quantizes each:

```
Original: 1536 floats × 4 bytes = 6,144 bytes per vector
PQ (96 sub-vectors × 8 bits): 96 bytes per vector
Compression: 64x!
```

## 5.4 Algorithm Comparison

| Algorithm | Build Time | Query Time | Memory | Recall | Best For |
|-----------|-----------|-----------|--------|--------|----------|
| **HNSW** | Slow | Very Fast | High | 95-99% | Low latency |
| **IVF** | Medium | Medium | Medium | 90-98% | Large scale |
| **IVF-PQ** | Medium | Fast | Very Low | 85-95% | Billion scale |
| **Flat (exact)** | None | Slow (O(n)) | Baseline | 100% | Small datasets |

---

# SECTION 6: VECTOR DATABASE COMPARISON (2025)

| Database | Type | Language | Best For | Scale | Hybrid | Managed |
|----------|------|----------|----------|-------|--------|---------|
| **Pinecone** | Managed | - | Production, zero-ops | Billions | Yes | Yes only |
| **Weaviate** | Open Source | Go | Graph + vector | Millions | Yes | Both |
| **Qdrant** | Open Source | Rust | Complex filtering | Billions | Yes | Both |
| **Milvus** | Open Source | Go/C++ | Maximum scale | Billions | Yes | Both (Zilliz) |
| **Chroma** | Open Source | Python | Prototyping | Millions | No | Self only |
| **FAISS** | Library | C++/Python | Research | Billions | No | N/A |
| **pgvector** | Extension | C | PostgreSQL users | Millions | No* | N/A |

## Pinecone
```python
from pinecone import Pinecone
pc = Pinecone(api_key="...")
index = pc.Index("my-index")

# Upsert
index.upsert(vectors=[("id1", [0.1, 0.2, ...], {"text": "hello"})])

# Query with metadata filter
results = index.query(vector=[0.1, 0.2, ...], top_k=5,
                      filter={"category": {"$eq": "technical"}})
```

## Chroma (Local Dev)
```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("docs")

collection.add(documents=["Hello world"], ids=["1"], metadatas=[{"source": "web"}])
results = collection.query(query_texts=["greeting"], n_results=5)
```

## pgvector
```sql
CREATE EXTENSION vector;
CREATE TABLE documents (id serial, content text, embedding vector(1536));
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Query
SELECT content, 1 - (embedding <=> query_embedding) AS similarity
FROM documents ORDER BY embedding <=> query_embedding LIMIT 5;
```

---

# SECTION 7: HYBRID SEARCH & RERANKING

## Hybrid Search (Dense + Sparse)

```python
# Reciprocal Rank Fusion (RRF)
def rrf_score(dense_rank, sparse_rank, k=60):
    return 1/(k + dense_rank) + 1/(k + sparse_rank)
```

## Reranking with Cross-Encoders

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Score each query-document pair
pairs = [(query, doc) for doc in retrieved_docs]
scores = reranker.predict(pairs)
reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
```

| Reranker | Provider | Latency | Quality |
|----------|----------|---------|---------|
| rerank-v3.5 | Cohere | ~100ms | Best |
| cross-encoder/ms-marco | SBERT | ~50ms | Good |
| BGE-Reranker-v2 | BAAI | ~50ms | Good |
| ColBERT v2 | Stanford | ~30ms | Good (token-level) |

---

# SECTION 8: PRODUCTION PATTERNS

| Pattern | Description | When |
|---------|-------------|------|
| **Namespace isolation** | Separate namespaces per tenant | Multi-tenant RAG |
| **Metadata filtering** | Pre-filter by metadata before vector search | Permission-aware retrieval |
| **Incremental indexing** | Update only changed documents | Live data |
| **Embedding cache** | Cache embedding API calls | Repeated content |
| **Warm-up queries** | Pre-load popular queries | Low latency |

---

# SECTION 9: INTERVIEW QUESTIONS (25+)

**Q1: What are embeddings and how do they work?**
Dense numerical vectors capturing semantic meaning. Trained via contrastive learning - similar pairs close, dissimilar far. Each dimension captures a latent feature of meaning.

**Q2: Cosine similarity vs dot product vs Euclidean - when to use each?**
Cosine for normalized text embeddings (direction matters). Dot product when magnitude matters or vectors pre-normalized. Euclidean for spatial/image data.

**Q3: Explain HNSW algorithm.**
Multi-layer graph where each layer has progressively fewer nodes. Search starts at top (coarse), navigates down to bottom (fine). Parameters: M (connections), efConstruction (build quality), efSearch (query quality).

**Q4: Compare Pinecone vs Weaviate vs Qdrant vs pgvector.**
Pinecone: fully managed, zero ops, best for production. Weaviate: graph+vector, GraphQL. Qdrant: best filtering (Rust). pgvector: add vectors to existing Postgres. Choice depends on: ops budget, filtering needs, existing infra.

**Q5: How do you choose an embedding model?**
Consider: MTEB benchmark scores, dimensionality (storage/speed), max context length, multilingual needs, cost, latency. Test on your specific data.

**Q6: What is hybrid search?**
Combining dense (embedding) + sparse (BM25) retrieval. Dense catches semantic similarity, sparse catches exact matches. Combined via Reciprocal Rank Fusion. 10-20% quality improvement.

**Q7: What is reranking and why does it matter?**
Second-pass scoring with cross-encoder. Initial retrieval (bi-encoder) is fast but approximate. Reranker processes query+doc together for deeper understanding. Typically 10-20% precision improvement.

**Q8: How do you handle embedding model updates?**
Re-embed all documents with new model. Run old + new index in parallel during transition. A/B test quality. Never mix embeddings from different models.

**Q9: How do you scale to billions of vectors?**
Sharding across multiple nodes, approximate algorithms (IVF-PQ), tiered storage (hot/cold), batch operations, connection pooling.

**Q10: What is Product Quantization?**
Compress vectors by splitting into sub-vectors and quantizing each. 64x memory reduction with ~5% recall loss. Essential for billion-scale.

**Q11: How do you handle multi-tenant vector search?**
Namespaces/partitions per tenant, metadata-based filtering, row-level security. Never let tenants' data mix in search results.

**Q12: What is Maximum Marginal Relevance (MMR)?**
Diversity-aware retrieval. Balances relevance and diversity: picks docs that are relevant to query but dissimilar to already-selected docs.

**Q13: How do you evaluate embedding quality?**
MTEB benchmark suite, task-specific retrieval metrics (MRR, NDCG, recall@k), A/B testing on your data.

**Q14: Explain IVF index and its parameters.**
Cluster vectors into K centroids (nlist). At query time, search only nprobe nearest clusters. Tradeoff: more nprobe = better recall, slower.

**Q15: How do you handle document updates in vector DB?**
Upsert (update or insert) by document ID. For deletions, remove by ID. For bulk updates, consider reindexing. Track document versions.

**Q16: What is ColBERT and how does it differ?**
Late interaction model - embeds each token separately, scores at token level. Better for long documents. Faster than cross-encoders for reranking.

**Q17: How do you reduce embedding costs?**
Batch embedding API calls, cache embeddings, use cheaper models for non-critical data, reduce dimensions (OpenAI supports this), use open-source models.

**Q18: What is multi-modal embedding?**
Single vector space for text + images (CLIP, etc.). Enables cross-modal search: search images with text, search text with images.

**Q19: How do you monitor vector DB performance?**
Track: query latency (p50, p95, p99), recall@k, QPS, index size, memory usage, replication lag.

**Q20: What is semantic caching?**
Cache query results and use embedding similarity to match new queries to cached ones. If similarity > threshold (e.g., 0.95), return cached result.

---

# SECTION 10: CODE EXAMPLES

## Complete Embedding Pipeline

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load
docs = PyPDFLoader("document.pdf").load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed & Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

# Search
results = vectorstore.similarity_search_with_score("What is the refund policy?", k=5)
for doc, score in results:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")
```

---

## Sources
- [LiquidMetal AI: Vector DB Comparison 2025](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [Firecrawl: Best Vector Databases 2025](https://www.firecrawl.dev/blog/best-vector-databases-2025)
- [lakefs: Best Vector DBs 2026](https://lakefs.io/blog/best-vector-databases/)
- [GeeksforGeeks: Top Vector Databases 2025](https://www.geeksforgeeks.org/dbms/top-vector-databases/)
