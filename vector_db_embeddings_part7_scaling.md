# PART 7: SCALING VECTOR DATABASES

---

## 7.1 Scaling Challenges Unique to Vector Databases

Vector databases face unique scaling challenges compared to traditional databases:

1. **Memory-intensive**: HNSW indexes must typically fit in RAM
2. **CPU-intensive**: Distance computations are compute-heavy
3. **No natural partitioning key**: Unlike SQL databases, vectors don't have natural shard keys
4. **Index rebuild cost**: Changing index parameters often requires full rebuild
5. **Consistency vs performance**: Distributed ANN search introduces accuracy trade-offs

---

## 7.2 Sharding Strategies

### What Is Sharding?
Splitting data across multiple nodes (shards) to distribute storage and compute load.

### Strategy 1: Random/Hash-Based Sharding
- Vectors are distributed randomly (or by hash of ID) across shards
- **Query**: Must query ALL shards and merge results
- **Pros**: Even data distribution, simple implementation
- **Cons**: Every query hits every shard (fan-out), high network overhead

```
Query -> [Shard 1] -> top-K results
      -> [Shard 2] -> top-K results   -> Merge -> Global top-K
      -> [Shard 3] -> top-K results
```

### Strategy 2: Metadata-Based Sharding (Partition Key)
- Vectors are sharded by a metadata field (e.g., tenant_id, region, category)
- **Query**: Only need to query relevant shards (if filter includes partition key)
- **Pros**: Eliminates fan-out for filtered queries, natural multi-tenancy
- **Cons**: Potential for hot shards, uneven distribution

```python
# Milvus partition key example
schema = CollectionSchema([
    FieldSchema("id", DataType.INT64, is_primary=True),
    FieldSchema("tenant_id", DataType.VARCHAR, max_length=64, is_partition_key=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
])
# Milvus automatically partitions data by tenant_id
# Queries with tenant_id filter only search relevant partitions
```

### Strategy 3: Cluster-Based Sharding (IVF-like)
- Use coarse clustering (like IVF) to assign vectors to shards
- Each shard contains vectors from specific clusters
- **Query**: Route to shards containing nearest clusters
- **Pros**: Smart routing reduces fan-out
- **Cons**: Complex routing logic, cluster imbalance

### Strategy 4: Geographic Sharding
- Shard by data center / region
- Users query nearest region
- **Pros**: Low latency, data sovereignty compliance
- **Cons**: Cross-region queries are expensive

### Shard Sizing Guidelines:
- **Per-shard sweet spot**: 1M-50M vectors (depends on dimensionality)
- **Memory per shard**: Keep index + vectors in RAM if possible
- **Rule of thumb**: HNSW with 768d vectors needs ~4-6GB RAM per million vectors
- **With quantization**: 1-2GB per million vectors

---

## 7.3 Replication

### Why Replicate?
1. **High availability**: Survive node failures
2. **Read throughput**: Distribute queries across replicas
3. **Latency**: Place replicas closer to users

### Replication Topologies:

**Leader-Follower (Primary-Replica):**
- Writes go to leader, replicated to followers
- Reads can go to any replica
- Standard approach in Qdrant, Milvus, Weaviate

**Leaderless:**
- Any node can accept writes
- Quorum-based consistency (read R + write W > N)
- More complex but higher availability

### Consistency Levels:

| Level | Description | Use Case |
|-------|-------------|----------|
| Strong | Read sees all committed writes | Financial, critical data |
| Bounded Staleness | Read may be up to X seconds behind | Most applications |
| Session | Read-your-own-writes guarantee | User-facing applications |
| Eventual | Reads may return stale data | Analytics, logging |

```python
# Milvus consistency levels
collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    consistency_level="Bounded"  # or "Strong", "Session", "Eventually"
)

# Qdrant replication
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    replication_factor=3,    # 3 replicas
    write_consistency_factor=2  # Majority writes
)
```

---

## 7.4 Indexing Strategies for Scale

### Index Type Selection by Scale:

| Scale | Recommended Index | RAM per 1M (768d) | Recall |
|-------|-------------------|-------------------|--------|
| < 100K | Flat (brute force) | ~3 GB | 100% |
| 100K - 1M | HNSW | ~4-6 GB | 99%+ |
| 1M - 10M | HNSW + SQ8 | ~1.5-2 GB | 98%+ |
| 10M - 100M | IVFPQ or HNSW + PQ | ~0.3-0.5 GB | 95%+ |
| 100M - 1B | DiskANN or IVFPQ + Sharding | ~0.1-0.3 GB | 93%+ |
| > 1B | Distributed IVFPQ / DiskANN | Varies | 90%+ |

### Quantization Strategies for Memory Reduction:

**Scalar Quantization (SQ8):**
- Convert float32 to int8 (4x compression)
- Minimal recall loss (~0.5-1%)
- Great first step for memory optimization

**Product Quantization (PQ):**
- Compress to m bytes per vector (up to 32x compression)
- More recall loss (2-5% typically)
- Essential for billion-scale

**Binary Quantization (BQ):**
- Convert each dimension to 1 bit (32x compression)
- Significant recall loss (~5-15%), use with rescoring
- Fastest distance computation (Hamming distance with POPCNT)
- Works well with high-dimensional models (1536d+)

```python
# Qdrant: Scalar quantization
client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,       # Clip outliers
            always_ram=True      # Keep quantized vectors in RAM
        )
    ),
    # Keep original vectors on disk for rescoring
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=20000
    )
)

# Qdrant: Binary quantization (for OpenAI embeddings)
client.create_collection(
    collection_name="binary_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(always_ram=True)
    )
)

# Search with oversampling for rescoring
results = client.query_points(
    collection_name="binary_collection",
    query=[0.1, 0.2, ...],
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=True,        # Rescore with original vectors
            oversampling=3.0     # Fetch 3x candidates before rescoring
        )
    ),
    limit=10
)
```

---

## 7.5 Batch Operations

### Why Batching Matters:
- Network round-trips are expensive
- Amortize connection overhead
- Better GPU utilization for embedding generation
- Most vector databases are optimized for batch operations

### Batch Insertion:

```python
# Pinecone batch upsert
from itertools import islice

def chunks(iterable, batch_size=100):
    iterator = iter(iterable)
    while chunk := list(islice(iterator, batch_size)):
        yield chunk

vectors = [
    {"id": f"doc_{i}", "values": embeddings[i].tolist(), "metadata": metadata[i]}
    for i in range(len(embeddings))
]

# Upsert in batches of 100
for batch in chunks(vectors, batch_size=100):
    index.upsert(vectors=batch)

# Qdrant batch upsert with parallel workers
client.upload_points(
    collection_name="documents",
    points=[
        PointStruct(id=i, vector=emb.tolist(), payload=meta)
        for i, (emb, meta) in enumerate(zip(embeddings, metadata_list))
    ],
    batch_size=256,
    parallel=4  # Parallel upload workers
)

# Milvus batch insert
collection.insert([
    titles,        # List of titles
    categories,    # List of categories
    embeddings,    # List of embedding vectors
])
collection.flush()  # Ensure data is persisted
```

### Batch Search:

```python
# FAISS batch search (highly efficient)
D, I = index.search(query_batch, k=10)  # query_batch shape: (num_queries, d)

# Qdrant batch search
results = client.query_batch_points(
    collection_name="documents",
    requests=[
        QueryRequest(query=q, limit=10)
        for q in query_vectors
    ]
)

# Pinecone batch query (not natively supported - use asyncio)
import asyncio

async def batch_query(index, queries, top_k=10):
    tasks = [
        asyncio.to_thread(index.query, vector=q, top_k=top_k)
        for q in queries
    ]
    return await asyncio.gather(*tasks)
```

### Batch Embedding Generation:

```python
# OpenAI batch embeddings
from openai import OpenAI

client = OpenAI()

# Process in batches (API limit: 2048 items per request)
def batch_embed(texts, model="text-embedding-3-small", batch_size=2048):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
    return all_embeddings

# Sentence-Transformers batch encoding (GPU-optimized)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
embeddings = model.encode(
    texts,
    batch_size=256,      # GPU batch size
    show_progress_bar=True,
    normalize_embeddings=True  # L2 normalize
)
```

---

## 7.6 Performance Optimization Checklist

### Indexing Phase:
- [ ] Choose appropriate index type for your scale
- [ ] Tune construction parameters (M, ef_construction for HNSW)
- [ ] Apply quantization if memory is constrained
- [ ] Batch insert vectors (avoid one-by-one inserts)
- [ ] Create metadata/payload indexes for filter fields
- [ ] Pre-warm indexes after loading

### Query Phase:
- [ ] Tune search parameters (ef_search, nprobe)
- [ ] Use batch queries when possible
- [ ] Enable quantized search with rescoring for memory savings
- [ ] Set appropriate consistency level
- [ ] Monitor and optimize filter selectivity
- [ ] Use caching for repeated queries

### Infrastructure:
- [ ] Size RAM for index + vectors (or enable disk offloading)
- [ ] Use NVMe SSDs for disk-based indexes (DiskANN)
- [ ] Set appropriate replication factor (>= 2 for production)
- [ ] Configure connection pooling for clients
- [ ] Monitor: QPS, p99 latency, recall, memory usage, disk I/O
- [ ] Set up alerting on performance degradation

---

## 7.7 Cost Optimization Strategies

### 1. Reduce Dimensions
- Use Matryoshka models (OpenAI v3) and truncate dimensions
- 1536d -> 512d can save 66% storage with ~5% quality loss
- Always benchmark quality impact on your specific data

### 2. Quantization
- Scalar (int8): 4x savings, minimal quality loss
- Binary: 32x savings, use with oversampling/rescoring
- Product: 8-32x savings, moderate quality loss

### 3. Tiered Storage
- Hot data: RAM (HNSW)
- Warm data: SSD (DiskANN)
- Cold data: Object storage (archive, reindex on demand)

### 4. Right-Size Your Index
- Don't use HNSW for 10K vectors (flat index is fine and cheaper)
- Don't use float32 if int8 suffices
- Don't use 3072d if 768d gives acceptable quality

### 5. Multi-Tenancy Optimization
- Shared index with metadata filtering vs. separate indexes per tenant
- Shared: More memory-efficient, simpler management
- Separate: Better isolation, easier deletion, more predictable performance
- Qdrant/Weaviate have first-class multi-tenancy support
