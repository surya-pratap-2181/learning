# PART 8: COMMON INTERVIEW QUESTIONS WITH DETAILED ANSWERS & CODE

---

## Q1: "Explain the difference between sparse and dense embeddings. When would you use each?"

### Answer:

**Sparse embeddings** have mostly zero values with a few non-zero entries. The dimensionality equals the vocabulary size (30K-100K+).

Examples: TF-IDF, BM25 scores, SPLADE, one-hot encodings.

**Dense embeddings** have non-zero values in every dimension. Typical dimensionality: 256-3072.

Examples: Word2Vec, BERT, OpenAI embeddings, Sentence-Transformers.

| Aspect | Sparse | Dense |
|--------|--------|-------|
| Dimensionality | 30K-100K+ | 256-3072 |
| Values | Mostly zeros | All non-zero |
| Interpretability | High (each dim = a word) | Low (abstract features) |
| Exact matching | Excellent | Poor |
| Semantic matching | Poor | Excellent |
| Storage (per vector) | Low (only store non-zeros) | Fixed (dim * 4 bytes) |

**When to use sparse:** Exact term matching, when keywords matter (product search, error codes, legal citations), when interpretability is needed.

**When to use dense:** Semantic search, when meaning matters more than exact words, cross-lingual search, multimodal search.

**Best practice:** Use both (hybrid search) for production systems.

---

## Q2: "You have 100 million 768-dimensional vectors. Design a vector search system that can handle 1000 QPS with p99 latency under 50ms."

### Answer:

**Step 1: Calculate memory requirements**
```python
num_vectors = 100_000_000
dimensions = 768
bytes_per_float = 4

# Raw vector storage
raw_storage = num_vectors * dimensions * bytes_per_float
print(f"Raw vectors: {raw_storage / 1e9:.1f} GB")  # ~307.2 GB

# With HNSW overhead (M=16, ~128 bytes per vector for graph)
hnsw_overhead = num_vectors * 128
total_hnsw = raw_storage + hnsw_overhead
print(f"HNSW total: {total_hnsw / 1e9:.1f} GB")  # ~320 GB

# With scalar quantization (int8)
sq8_storage = num_vectors * dimensions * 1  # 1 byte per dim
total_sq8 = sq8_storage + hnsw_overhead
print(f"SQ8 + HNSW: {total_sq8 / 1e9:.1f} GB")  # ~89.6 GB

# With product quantization (m=96)
pq_storage = num_vectors * 96  # 96 bytes per vector
total_pq = pq_storage + hnsw_overhead
print(f"PQ96 + HNSW: {total_pq / 1e9:.1f} GB")  # ~22.4 GB
```

**Step 2: Architecture Design**

Option A: HNSW + Scalar Quantization (~90GB total)
- 3 nodes x 32GB RAM = 96GB (with replication factor 2, need 6 nodes)
- Use Qdrant or Milvus with SQ8 quantization
- ef_search tuned for <50ms latency

Option B: IVFPQ (memory-constrained)
- ~22GB total with PQ96
- 2 nodes x 16GB RAM with replication
- Lower recall (~95%) but much cheaper

**Step 3: QPS calculation**
- Single HNSW query at ef=100, 768d: ~1-5ms
- Single node can handle: ~200-1000 QPS
- For 1000 QPS with headroom: 3-5 query nodes
- Add replication for fault tolerance: 6-10 nodes total

**Step 4: Complete architecture**
```
                    Load Balancer
                    /     |      \
              Node 1   Node 2   Node 3  (Query routers)
              /    \   /    \   /    \
           Shard1  Shard2  Shard3  Shard4  (Data shards)
           Rep-A   Rep-A   Rep-A   Rep-A   (Primary)
           Rep-B   Rep-B   Rep-B   Rep-B   (Replica)
```

**Design decisions:**
- Shard by hash of vector ID (even distribution)
- 4 shards x 25M vectors each (~22GB per shard with SQ8)
- Replication factor 2 for availability
- Fan-out: Each query hits all 4 shards in parallel
- Merge top-K from each shard
- Total nodes: 8 (4 shards x 2 replicas)
- Each node: 32GB RAM, 8 cores

---

## Q3: "What is the curse of dimensionality and how does it affect vector search?"

### Answer:

The curse of dimensionality refers to phenomena that arise in high-dimensional spaces that are counterintuitive from low-dimensional experience.

**Key effects on vector search:**

1. **Distance concentration**: As dimensionality increases, the ratio of nearest to farthest neighbor approaches 1. All points become roughly equidistant.

```python
import numpy as np

def distance_concentration(n_points=1000, dims=[2, 10, 100, 768, 1536]):
    for d in dims:
        points = np.random.randn(n_points, d)
        query = np.random.randn(1, d)
        distances = np.linalg.norm(points - query, axis=1)
        ratio = distances.min() / distances.max()
        print(f"d={d:>5}: min/max ratio = {ratio:.4f}, "
              f"std/mean = {distances.std()/distances.mean():.4f}")

distance_concentration()
# d=    2: min/max ratio = 0.0312, std/mean = 0.3842
# d=   10: min/max ratio = 0.3501, std/mean = 0.1284
# d=  100: min/max ratio = 0.7123, std/mean = 0.0401
# d=  768: min/max ratio = 0.8812, std/mean = 0.0144
# d= 1536: min/max ratio = 0.9156, std/mean = 0.0102
```

2. **Volume concentration**: Most of the volume of a high-d hypersphere is concentrated near its surface. Points are spread "thin."

3. **Hubness problem**: Some points (hubs) appear as nearest neighbors of many query points, even if they are not truly similar. This distorts search results.

4. **Empty space**: The volume of space grows exponentially with dimensions, making data extremely sparse.

**Practical implications:**
- ANN algorithms become less effective (recall drops for the same speed)
- More data is needed to cover the space adequately
- Dimensionality reduction (PCA, Matryoshka) can help
- Distance metrics may need to be chosen carefully (cosine often better than L2 in high-d)

**Mitigation strategies:**
- Use embeddings with appropriate dimensionality (384-1024 usually sufficient)
- Apply dimensionality reduction if vectors are >2000d
- Use Matryoshka-trained models (truncate without retraining)
- Product quantization inherently works in lower-dimensional subspaces

---

## Q4: "Explain how HNSW works. Why is it the most popular ANN algorithm?"

### Answer:

*[See Part 2, Section 2.4 for full details. Summary answer below]*

HNSW builds a multi-layer navigable small world graph. Think of it like an express transportation system:

- **Top layers** = Express routes (few stops, long jumps) - like a highway
- **Bottom layers** = Local routes (many stops, short jumps) - like city streets

**Construction:**
1. Assign each vector a random maximum layer (exponential distribution, so most vectors only exist on lower layers)
2. Insert vectors top-down: at each layer, find nearest neighbors and connect

**Search:**
1. Enter at the top layer with a single entry point
2. Greedily navigate to the nearest node to the query
3. Drop down one layer, repeat
4. At the bottom layer, perform a thorough beam search

**Why it dominates:**
1. **No training phase**: Unlike IVF, you can add vectors one at a time
2. **Logarithmic query time**: O(log N) layers to traverse
3. **Excellent recall/speed trade-off**: >99% recall with sub-millisecond latency
4. **Tunable**: ef_search parameter allows runtime speed/accuracy trade-off
5. **Battle-tested**: Used by every major vector database

**Weakness:** High memory usage (entire graph in RAM). This is why quantization and DiskANN are important for billion-scale.

---

## Q5: "How would you build a RAG (Retrieval-Augmented Generation) system? What role do vector databases play?"

### Answer:

```python
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize
client = OpenAI()
qdrant = QdrantClient("http://localhost:6333")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

# === INDEXING PIPELINE ===

def chunk_document(text, chunk_size=512, overlap=50):
    """Split document into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def index_documents(documents):
    """Embed and store documents in vector database."""
    for doc in documents:
        chunks = chunk_document(doc["text"])
        embeddings = embedder.encode(chunks, normalize_embeddings=True)

        points = [
            PointStruct(
                id=f"{doc['id']}_chunk_{i}",
                vector=emb.tolist(),
                payload={
                    "text": chunk,
                    "doc_id": doc["id"],
                    "source": doc["source"],
                    "chunk_index": i
                }
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        qdrant.upsert(collection_name="documents", points=points)

# === RETRIEVAL PIPELINE ===

def retrieve(query, top_k=5, score_threshold=0.7):
    """Retrieve relevant chunks from vector database."""
    query_embedding = embedder.encode(query, normalize_embeddings=True)

    results = qdrant.query_points(
        collection_name="documents",
        query=query_embedding.tolist(),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True
    )

    return [
        {"text": hit.payload["text"], "score": hit.score, "source": hit.payload["source"]}
        for hit in results.points
    ]

# === GENERATION PIPELINE ===

def generate_answer(query, context_docs):
    """Generate answer using LLM with retrieved context."""
    context = "\n\n".join([
        f"[Source: {doc['source']}]\n{doc['text']}"
        for doc in context_docs
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are a helpful assistant.
Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so.
Cite your sources."""},
            {"role": "user", "content": f"""Context:
{context}

Question: {query}

Answer:"""}
        ],
        temperature=0
    )

    return response.choices[0].message.content

# === FULL RAG PIPELINE ===

def rag_query(query):
    """Complete RAG pipeline: retrieve + generate."""
    # Step 1: Retrieve relevant documents
    docs = retrieve(query, top_k=5)

    if not docs:
        return "I don't have enough information to answer this question."

    # Step 2: Generate answer with context
    answer = generate_answer(query, docs)

    return {
        "answer": answer,
        "sources": [{"source": d["source"], "score": d["score"]} for d in docs]
    }

# Usage
result = rag_query("How does HNSW algorithm work?")
print(result["answer"])
print(result["sources"])
```

**Key design decisions in a production RAG system:**
1. **Chunking strategy**: Size, overlap, respect document structure (headers, paragraphs)
2. **Embedding model**: Match to your domain (general vs. domain-specific)
3. **Retrieval**: Hybrid search (dense + sparse) for better recall
4. **Reranking**: Add a cross-encoder reranker (Cohere Rerank, BGE Reranker) for precision
5. **Context window management**: Fit the most relevant chunks within the LLM's context
6. **Evaluation**: Use RAGAS or similar frameworks to measure retrieval quality

---

## Q6: "What are the trade-offs between using a managed vector database (Pinecone) vs self-hosting (Qdrant/Milvus)?"

### Answer:

| Aspect | Managed (Pinecone) | Self-Hosted (Qdrant/Milvus) |
|--------|-------------------|---------------------------|
| Setup time | Minutes | Hours to days |
| Ops overhead | Zero | Significant (updates, monitoring, backups) |
| Cost (small scale) | Low ($25-100/mo) | Higher (minimum server costs) |
| Cost (large scale) | High ($1000s/mo) | Lower (own infrastructure) |
| Customization | Limited | Full control |
| Data sovereignty | Vendor cloud | Your infrastructure |
| Vendor lock-in | High | Low (open source) |
| SLA | Contractual | Self-managed |
| Scaling | Automatic | Manual (but Kubernetes helps) |
| Latency control | Limited | Full control (co-locate with app) |

**Choose managed when:** Small team, variable workloads, fast time-to-market, budget allows.
**Choose self-hosted when:** Large scale, data sensitivity, cost optimization, customization needs.

---

## Q7: "How do you evaluate the quality of embeddings for your specific use case?"

### Answer:

```python
import numpy as np
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer

def evaluate_embeddings(model_name, queries, corpus, relevance_labels):
    """
    Evaluate embedding model quality for retrieval.

    Args:
        queries: List of query strings
        corpus: List of document strings
        relevance_labels: Dict[query_idx] -> Dict[doc_idx] -> relevance_score
    """
    model = SentenceTransformer(model_name)

    # Encode
    query_embeddings = model.encode(queries, normalize_embeddings=True)
    corpus_embeddings = model.encode(corpus, normalize_embeddings=True)

    # Compute similarities
    similarities = query_embeddings @ corpus_embeddings.T

    # Metrics
    metrics = {"mrr": [], "ndcg@10": [], "recall@10": [], "precision@10": []}

    for q_idx in range(len(queries)):
        scores = similarities[q_idx]
        ranked_indices = np.argsort(-scores)

        # Get relevance for this query
        relevant = relevance_labels.get(q_idx, {})

        # MRR (Mean Reciprocal Rank)
        for rank, doc_idx in enumerate(ranked_indices, 1):
            if relevant.get(doc_idx, 0) > 0:
                metrics["mrr"].append(1.0 / rank)
                break
        else:
            metrics["mrr"].append(0.0)

        # nDCG@10
        true_relevance = np.array([relevant.get(idx, 0) for idx in range(len(corpus))])
        predicted_relevance = scores
        metrics["ndcg@10"].append(
            ndcg_score([true_relevance], [predicted_relevance], k=10)
        )

        # Recall@10
        top_10 = set(ranked_indices[:10])
        relevant_set = set(idx for idx, rel in relevant.items() if rel > 0)
        if relevant_set:
            metrics["recall@10"].append(len(top_10 & relevant_set) / len(relevant_set))
        else:
            metrics["recall@10"].append(0.0)

        # Precision@10
        relevant_in_top10 = sum(1 for idx in ranked_indices[:10] if relevant.get(idx, 0) > 0)
        metrics["precision@10"].append(relevant_in_top10 / 10)

    return {k: np.mean(v) for k, v in metrics.items()}

# Compare models
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
]

for model_name in models:
    results = evaluate_embeddings(model_name, queries, corpus, labels)
    print(f"\n{model_name}:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
```

**Key evaluation metrics:**
- **MRR (Mean Reciprocal Rank)**: How early does the first relevant result appear?
- **nDCG@K**: Considers both relevance grades and position
- **Recall@K**: What fraction of relevant documents are in the top K?
- **Precision@K**: What fraction of top K results are relevant?

---

## Q8: "Explain cosine similarity vs dot product vs Euclidean distance. When does it matter which you choose?"

### Answer:

**It matters when vectors are NOT normalized.** For normalized vectors (L2 norm = 1), all three produce equivalent rankings.

```python
import numpy as np

a = np.array([3.0, 4.0])   # norm = 5
b = np.array([1.0, 2.0])   # norm = sqrt(5)

# Not normalized: metrics disagree
cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
dot = np.dot(a, b)
euclidean = np.linalg.norm(a - b)
print(f"Cosine: {cosine:.4f}, Dot: {dot:.4f}, Euclidean: {euclidean:.4f}")

# Normalize
a_norm = a / np.linalg.norm(a)  # [0.6, 0.8]
b_norm = b / np.linalg.norm(b)  # [0.447, 0.894]

cosine_n = np.dot(a_norm, b_norm)  # Same as cosine above
dot_n = np.dot(a_norm, b_norm)     # Now equals cosine!
euclidean_n = np.linalg.norm(a_norm - b_norm)
# euclidean_n^2 = 2*(1-cosine_n)   # Algebraic relationship

print(f"Normalized - Cosine: {cosine_n:.4f}, Dot: {dot_n:.4f}")
```

**Practical guidance:**
- Most embedding APIs (OpenAI, Cohere) return normalized vectors -> All metrics equivalent
- If you're unsure, use cosine similarity (most robust)
- If vectors are normalized and you want speed, use dot product (no normalization step)
- Use Euclidean for clustering tasks where absolute position matters
- Use dot product when magnitude encodes importance (e.g., recommendation scores)

---

## Q9: "How would you handle embedding model updates in production? The new model produces different vectors."

### Answer:

This is a critical production concern. Embedding models get updated (e.g., OpenAI ada-002 -> text-embedding-3-small), and new vectors are NOT compatible with old ones.

**Strategy 1: Dual-Write + Cutover (Blue-Green)**
```python
# During migration period, write to both old and new indexes
def index_document(doc):
    old_embedding = old_model.encode(doc["text"])
    new_embedding = new_model.encode(doc["text"])

    # Write to old index (serving traffic)
    old_index.upsert(id=doc["id"], vector=old_embedding, metadata=doc["metadata"])

    # Write to new index (shadow)
    new_index.upsert(id=doc["id"], vector=new_embedding, metadata=doc["metadata"])

# Backfill: Re-embed all existing documents with new model
def backfill_new_index():
    for doc in iterate_all_documents():
        new_embedding = new_model.encode(doc["text"])
        new_index.upsert(id=doc["id"], vector=new_embedding, metadata=doc["metadata"])

# Once backfill complete + validated:
# Switch traffic from old_index to new_index
# Delete old_index after grace period
```

**Strategy 2: Version Field in Metadata**
```python
# Store embedding model version in metadata
def index_document(doc, model_version="v2"):
    embedding = models[model_version].encode(doc["text"])
    index.upsert(
        id=doc["id"],
        vector=embedding,
        metadata={**doc["metadata"], "embed_version": model_version}
    )

# Query only searches compatible vectors
results = index.query(
    vector=new_model.encode(query),
    filter={"embed_version": "v2"},
    top_k=10
)
```

**Strategy 3: Adapter/Projection Layer**
```python
import torch
import torch.nn as nn

# Learn a linear projection from old -> new embedding space
class EmbeddingAdapter(nn.Module):
    def __init__(self, old_dim, new_dim):
        super().__init__()
        self.projection = nn.Linear(old_dim, new_dim)

    def forward(self, old_embedding):
        return self.projection(old_embedding)

# Train on paired (old_embedding, new_embedding) data
adapter = EmbeddingAdapter(1536, 1024)
optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

for old_emb, new_emb in training_pairs:
    predicted = adapter(old_emb)
    loss = nn.MSELoss()(predicted, new_emb)
    loss.backward()
    optimizer.step()

# Use adapter to convert old vectors without re-embedding all documents
# (Quality will be lower than full re-embedding but avoids the cost)
```

**Best practice:** Plan for model migration from day one. Use the blue-green strategy for critical applications. Budget for periodic re-embedding as models improve.

---

## Q10: "What is the difference between a vector database and a vector library like FAISS?"

### Answer:

| Feature | Vector Library (FAISS) | Vector Database (Pinecone, Qdrant) |
|---------|----------------------|-----------------------------------|
| Persistence | No (in-memory only) | Yes (durable storage) |
| CRUD operations | Add only (no update/delete) | Full CRUD |
| Metadata | None | Rich metadata + filtering |
| Scalability | Single process | Distributed, sharded |
| Replication | None | Built-in |
| API | C++/Python function calls | REST/gRPC APIs |
| Consistency | N/A | Configurable |
| Backup/Recovery | Manual (save/load index) | Automatic |
| Monitoring | None | Built-in metrics |
| Access Control | None | Authentication, RBAC |
| Production readiness | Requires custom wrapping | Production-ready |

**Use FAISS when:** Prototyping, research, embedding in a larger application, need GPU search, maximum performance control.

**Use a vector database when:** Production applications, need persistence, metadata filtering, distributed search, team collaboration, operational features.

---

## Q11: "How do you chunk documents for a RAG system? What are the trade-offs?"

### Answer:

```python
# === Chunking Strategies ===

# 1. Fixed-size chunking (simplest)
def fixed_size_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# 2. Sentence-based chunking
import nltk
nltk.download('punkt')

def sentence_chunks(text, max_chunk_size=500):
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(sentence)
        current_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# 3. Recursive/hierarchical chunking (LangChain style)
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],  # Try largest separators first
    length_function=len
)
chunks = splitter.split_text(document_text)

# 4. Semantic chunking (split at topic boundaries)
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunks(text, threshold=0.5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = nltk.sent_tokenize(text)
    embeddings = model.encode(sentences)

    # Find breakpoints where consecutive sentences are dissimilar
    chunks, current_chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

**Trade-offs:**

| Chunk Size | Pros | Cons |
|-----------|------|------|
| Small (100-200 tokens) | Precise retrieval, less noise | May lose context, more chunks to store |
| Medium (300-500 tokens) | Good balance | Standard choice |
| Large (500-1000 tokens) | More context per chunk | May include irrelevant info, lower precision |

**Best practices:**
- Include overlap (10-20%) to avoid splitting important context
- Respect document structure (don't split mid-paragraph if possible)
- Include metadata (document title, section headers) in each chunk
- Use the "parent document retriever" pattern: retrieve small chunks, return larger parent

---

## Q12: "Explain what a reranker is and how it differs from an embedding model."

### Answer:

**Embedding model (bi-encoder):**
- Encodes query and documents INDEPENDENTLY
- Can pre-compute document embeddings
- Fast at search time: O(1) per document (after indexing)
- Lower accuracy for fine-grained relevance

**Reranker (cross-encoder):**
- Processes query AND document TOGETHER
- Cannot pre-compute: must process each (query, document) pair
- Slow: O(n) where n = number of candidates
- Higher accuracy for relevance scoring

**The two-stage pipeline:**
```
Query -> Embedding Model -> Vector DB (top 100) -> Reranker (top 10) -> LLM
         (fast, broad)      (ANN search)          (slow, precise)     (generation)
```

```python
# Two-stage retrieval with reranking
from sentence_transformers import SentenceTransformer, CrossEncoder

# Stage 1: Fast retrieval with bi-encoder
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
query_emb = bi_encoder.encode(query)
# Search vector DB -> get top 100 candidates
candidates = vector_db.search(query_emb, top_k=100)

# Stage 2: Precise reranking with cross-encoder
cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs = [(query, candidate["text"]) for candidate in candidates]
rerank_scores = cross_encoder.predict(pairs)

# Sort by reranker scores
reranked = sorted(
    zip(candidates, rerank_scores),
    key=lambda x: x[1],
    reverse=True
)[:10]

# Alternative: Cohere Rerank API
import cohere
co = cohere.Client("API_KEY")

reranked = co.rerank(
    query=query,
    documents=[c["text"] for c in candidates],
    top_n=10,
    model="rerank-english-v3.0"
)
```

---

## Q13: "What is Matryoshka Representation Learning and why is it important?"

### Answer:

Matryoshka Representation Learning (MRL) trains embedding models such that the first d dimensions of a higher-dimensional embedding are still useful on their own. Like Russian nesting dolls - each smaller subset is a valid embedding.

**Why it matters:**
- Reduce storage by 2-8x without retraining the model
- Adaptive: Use full dimensions for high-accuracy, reduced for cost-sensitive use
- OpenAI's text-embedding-3 models support this natively

```python
from openai import OpenAI

client = OpenAI()

# Full 3072 dimensions
full_emb = client.embeddings.create(
    model="text-embedding-3-large",
    input="Vector databases are important",
    dimensions=3072
).data[0].embedding

# Truncated to 256 dimensions (still useful!)
small_emb = client.embeddings.create(
    model="text-embedding-3-large",
    input="Vector databases are important",
    dimensions=256
).data[0].embedding

# Storage savings: 3072*4=12288 bytes vs 256*4=1024 bytes (12x reduction!)
# Quality: ~95% of full-dimension retrieval quality at 256d
```

---

## Q14: "How do you handle multi-tenancy in a vector database?"

### Answer:

**Approach 1: Metadata filtering (shared index)**
```python
# All tenants share one index, filter by tenant_id
index.upsert(vectors=[
    {"id": "doc1", "values": [...], "metadata": {"tenant_id": "tenant_A", ...}},
    {"id": "doc2", "values": [...], "metadata": {"tenant_id": "tenant_B", ...}},
])

results = index.query(
    vector=[...],
    filter={"tenant_id": "tenant_A"},
    top_k=10
)
# Pros: Simple, memory efficient
# Cons: Noisy neighbor, filter performance at high selectivity
```

**Approach 2: Namespaces/partitions (logical isolation)**
```python
# Pinecone namespaces
index.upsert(vectors=[...], namespace="tenant_A")
results = index.query(vector=[...], top_k=10, namespace="tenant_A")

# Milvus partition key
schema = CollectionSchema([
    FieldSchema("tenant_id", DataType.VARCHAR, is_partition_key=True, max_length=64),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
])
# Pros: Better isolation, efficient per-tenant queries
# Cons: Resource overhead per namespace
```

**Approach 3: Separate collections (physical isolation)**
```python
# One collection per tenant
client.create_collection(f"tenant_{tenant_id}", vectors_config=...)
# Pros: Full isolation, independent scaling, easy deletion
# Cons: Resource overhead, management complexity, cold starts
```

**Approach 4: Weaviate/Qdrant native multi-tenancy**
```python
# Qdrant: Shard-level multi-tenancy (planned/available in newer versions)
# Weaviate: First-class multi-tenancy
collection = client.collections.create(
    name="Documents",
    multi_tenancy_config=Configure.multi_tenancy(enabled=True),
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
)

# Activate tenant
collection.tenants.create([Tenant(name="tenantA"), Tenant(name="tenantB")])

# Operations are scoped to tenant
tenant_collection = collection.with_tenant("tenantA")
tenant_collection.data.insert({"content": "..."})
# Pros: Efficient resource usage, proper isolation, easy tenant management
# Cons: Database-specific feature
```

---

## Q15: "What are the key metrics you would monitor for a vector database in production?"

### Answer:

**Performance Metrics:**
- **QPS (Queries Per Second)**: Throughput
- **p50/p95/p99 latency**: Query response time distribution
- **Index build time**: How long it takes to index new vectors
- **Recall@K**: Accuracy of ANN results (sample and compare against brute force periodically)

**Resource Metrics:**
- **Memory usage**: RAM consumed by index + vectors + metadata
- **Disk usage**: Persistent storage consumption
- **CPU usage**: Per-query CPU cost
- **Network I/O**: For distributed deployments

**Operational Metrics:**
- **Index segment count**: Too many segments = degraded performance (needs compaction)
- **Write queue depth**: Backlog of pending writes
- **Replication lag**: How far behind replicas are
- **Error rate**: Failed queries, timeouts

**Business Metrics:**
- **Relevance (human evaluation)**: Are results actually useful?
- **Click-through rate**: For search applications
- **Cost per query**: Total infrastructure cost / total queries

```python
# Example monitoring setup with Prometheus + Grafana
# (conceptual - implementation varies by vector database)

import time
from prometheus_client import Histogram, Counter, Gauge

# Define metrics
QUERY_LATENCY = Histogram('vector_search_latency_seconds', 'Query latency',
                          buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
QUERY_COUNT = Counter('vector_search_total', 'Total queries')
RESULT_COUNT = Histogram('vector_search_results', 'Number of results returned')
INDEX_SIZE = Gauge('vector_index_size_vectors', 'Number of vectors in index')

def monitored_search(query_vector, top_k=10, **kwargs):
    QUERY_COUNT.inc()
    start = time.time()

    results = vector_db.search(query_vector, top_k=top_k, **kwargs)

    latency = time.time() - start
    QUERY_LATENCY.observe(latency)
    RESULT_COUNT.observe(len(results))

    return results
```
