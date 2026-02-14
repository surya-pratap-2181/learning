---
title: "Part 2 - Search Algorithms"
layout: default
parent: "Vector Databases & Embeddings"
nav_order: 3
---

# PART 2: VECTOR SEARCH ALGORITHMS

---

## 2.1 The Nearest Neighbor Problem

Given a query vector q and a dataset of N vectors, find the K vectors most similar to q.

**Exact (Brute Force):** O(N * d) per query where d = dimensionality
**Approximate (ANN):** Sub-linear time with controllable accuracy trade-off

---

## 2.2 Brute Force (Flat/Exact Search)

### How It Works:
- Compare the query vector against every vector in the dataset
- Compute similarity/distance for each pair
- Return the top-K results

### Complexity:
- Time: O(N * d) per query
- Space: O(N * d)
- 100% recall (exact results)

### When to Use:
- Small datasets (< 10,000-50,000 vectors)
- When perfect recall is required
- As a baseline to evaluate ANN algorithms
- When queries are infrequent

```python
import numpy as np

def brute_force_search(query, database, k=10):
    """
    query: (d,) vector
    database: (N, d) matrix
    returns: indices and distances of k nearest neighbors
    """
    # Compute all distances
    distances = np.linalg.norm(database - query, axis=1)  # Euclidean
    # Or for cosine similarity:
    # similarities = database @ query / (np.linalg.norm(database, axis=1) * np.linalg.norm(query))

    # Get top-K
    top_k_indices = np.argpartition(distances, k)[:k]
    top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
    return top_k_indices, distances[top_k_indices]

# FAISS flat index (optimized brute force)
import faiss

d = 768  # dimensionality
index = faiss.IndexFlatL2(d)       # L2 distance
# index = faiss.IndexFlatIP(d)     # Inner product (dot product)

index.add(database_vectors)         # Add vectors
distances, indices = index.search(query_vectors, k=10)  # Search
```

---

## 2.3 Inverted File Index (IVF)

### How It Works:
1. **Training Phase:** Cluster the dataset into `nlist` clusters using k-means
2. **Insertion:** Assign each vector to its nearest cluster centroid
3. **Search:**
   - Find the `nprobe` nearest cluster centroids to the query
   - Only search vectors within those clusters
   - Return top-K from the searched subset

### Key Parameters:
- **nlist**: Number of clusters (typically sqrt(N) to 4*sqrt(N))
- **nprobe**: Number of clusters to search at query time (controls speed/accuracy trade-off)

### Complexity:
- Training: O(N * nlist * d * iterations)
- Query: O(nprobe * N/nlist * d)
- Space: O(N * d + nlist * d)

### Trade-offs:
- nprobe = 1: Fastest, lowest recall
- nprobe = nlist: Equivalent to brute force
- Sweet spot: nprobe = 5-20% of nlist for >95% recall

```python
import faiss

d = 768
nlist = 100  # number of clusters

# Create IVF index with flat (exact) quantizer
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Must train before adding vectors
index.train(training_vectors)  # Learns cluster centroids
index.add(database_vectors)

# Search with nprobe
index.nprobe = 10  # Search 10 nearest clusters
distances, indices = index.search(query_vectors, k=10)
```

---

## 2.4 HNSW (Hierarchical Navigable Small World)

HNSW is the most widely used ANN algorithm in production vector databases. It builds a multi-layer graph structure for efficient search.

### How It Works:

**Based on two concepts:**
1. **Navigable Small World (NSW):** A graph where each node connects to several neighbors, allowing greedy traversal from any starting point to the nearest neighbor of a query
2. **Hierarchical structure:** Multiple layers, each being a subgraph of the layer below

**Construction:**
1. Each new vector is assigned a random maximum layer (exponential distribution)
2. Starting from the top layer, greedily find the nearest neighbor
3. At the insertion layer and below, connect to the M nearest neighbors
4. Use heuristic to select diverse neighbors (not just closest)

**Search:**
1. Start at the entry point in the top layer
2. Greedily navigate to the nearest node to the query
3. Move down one layer, using the current nearest node as the entry point
4. At the bottom layer (layer 0), perform a more thorough beam search with `ef_search` candidates

### Key Parameters:
| Parameter | Description | Typical Values | Effect |
|-----------|-------------|---------------|--------|
| M | Max connections per node per layer | 16-64 | Higher = better recall, more memory |
| ef_construction | Beam width during construction | 100-500 | Higher = better graph quality, slower build |
| ef_search | Beam width during search | 50-500 | Higher = better recall, slower search |

### Complexity:
- Construction: O(N * log(N) * M * ef_construction)
- Query: O(log(N) * ef_search * d)
- Space: O(N * M * layers * 4 bytes) for graph + O(N * d * 4 bytes) for vectors

### Memory Overhead:
- Graph structure adds ~(M * 2 * 4 bytes) per vector per layer
- With M=16, approximately 128 bytes per vector for the graph
- For 1M vectors with 768d: vectors = 3GB, graph overhead = ~200MB

### Strengths:
- Excellent recall/speed trade-off (often >99% recall at sub-millisecond latency)
- No training phase (vectors can be added incrementally)
- Works well for all distance metrics
- Production-proven at billion scale

### Weaknesses:
- High memory usage (entire graph must fit in RAM)
- Slow construction time for very large datasets
- Delete operations are expensive (requires graph reconnection)
- Not ideal for very high-dimensional data (>2000d)

```python
import faiss
import hnswlib

# FAISS HNSW
d = 768
M = 32
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 100
index.add(database_vectors)
distances, indices = index.search(query_vectors, k=10)

# hnswlib (dedicated HNSW library)
index = hnswlib.Index(space='cosine', dim=d)
index.init_index(max_elements=1000000, ef_construction=200, M=32)
index.add_items(database_vectors, ids=np.arange(len(database_vectors)))
index.set_ef(100)  # ef_search
labels, distances = index.knn_query(query_vectors, k=10)
```

---

## 2.5 Product Quantization (PQ)

PQ is a lossy compression technique that dramatically reduces memory usage at the cost of some accuracy.

### How It Works:
1. **Split** each d-dimensional vector into m sub-vectors of d/m dimensions each
2. **Train** a separate k-means codebook for each sub-vector space (typically k=256 = 8 bits)
3. **Encode** each sub-vector as its nearest centroid index (1 byte per sub-vector)
4. **Search** using precomputed distance tables between query sub-vectors and codebook centroids

### Compression:
- Original: d * 4 bytes (float32)
- Compressed: m bytes (one byte per sub-vector)
- Compression ratio: d * 4 / m
- Example: d=768, m=96 -> 3072 bytes -> 96 bytes (32x compression)

### Variants:
- **OPQ (Optimized PQ):** Applies rotation matrix before quantization for better results
- **IVFPQ:** Combines IVF with PQ - cluster first, then compress residuals
- **IVFPQR:** Adds re-ranking step with PQ for better accuracy

### When to Use:
- Very large datasets that don't fit in memory
- When memory is the bottleneck, not accuracy
- Combined with IVF for the best memory/speed/recall trade-off

```python
import faiss

d = 768
m = 96        # Number of sub-vectors (must divide d evenly)
nbits = 8     # Bits per sub-vector (256 centroids per codebook)

# Pure PQ index
index = faiss.IndexPQ(d, m, nbits)
index.train(training_vectors)
index.add(database_vectors)
distances, indices = index.search(query_vectors, k=10)

# IVF + PQ (most common production setup)
nlist = 1024
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(training_vectors)
index.add(database_vectors)
index.nprobe = 32
distances, indices = index.search(query_vectors, k=10)

# Memory comparison
print(f"Flat index: {d * 4 * len(database_vectors) / 1e9:.2f} GB")
print(f"PQ index: {m * len(database_vectors) / 1e9:.2f} GB")
```

---

## 2.6 ScaNN (Scalable Nearest Neighbors, Google)

Google's library that achieves state-of-the-art ANN performance.

### Key Innovation - Anisotropic Vector Quantization:
- Standard PQ minimizes reconstruction error uniformly
- ScaNN recognizes that for inner product search, the component of error parallel to the original vector matters more than the perpendicular component
- Anisotropic quantization assigns different weights to parallel vs perpendicular error

### Architecture:
1. **Partitioning** (like IVF): Divide vectors into clusters
2. **Scoring** with anisotropic quantization: Score candidates efficiently
3. **Rescoring** with exact distances: Re-rank top candidates for accuracy

```python
import scann

# Build ScaNN index
searcher = scann.scann_ops_pybind.builder(
    database_vectors, 10, "dot_product"
).tree(
    num_leaves=2000,            # Number of partitions
    num_leaves_to_search=100,   # Partitions to search
    training_sample_size=250000
).score_ah(
    2,                          # Dimensions per block
    anisotropic_quantization_threshold=0.2
).reorder(
    100                         # Reorder top 100 with exact distances
).build()

neighbors, distances = searcher.search(query, final_num_neighbors=10)

# Batch search
neighbors, distances = searcher.search_batched(query_batch)
```

---

## 2.7 Other ANN Algorithms

### LSH (Locality-Sensitive Hashing):
- Uses hash functions that map similar items to the same bucket
- Multiple hash tables increase recall
- Fast but lower recall than HNSW/IVF for same speed
- Good for: Very high-dimensional data, streaming scenarios

### Annoy (Approximate Nearest Neighbors Oh Yeah, Spotify):
- Builds a forest of random projection trees
- Each tree splits space with random hyperplanes
- Search queries multiple trees and merges results
- Good for: Read-heavy workloads, mmap support, static datasets

### VP-Tree (Vantage-Point Tree):
- Binary space partitioning using distance to a vantage point
- Good for metric spaces but not as fast as HNSW for high-d
- Mostly theoretical interest

### DiskANN (Microsoft):
- Graph-based (like HNSW) but designed for SSD storage
- Vamana graph: single-layer, bounded-degree graph
- Can index billions of vectors with minimal RAM
- Key for cost-effective billion-scale search

---

## 2.8 Algorithm Comparison Table

| Algorithm | Build Time | Query Time | Memory | Recall@10 | Dynamic |
|-----------|-----------|------------|--------|-----------|---------|
| Flat (Brute Force) | O(1) | O(N*d) | O(N*d) | 100% | Yes |
| IVF | O(N*k*d) | O(nprobe*N/nlist*d) | O(N*d) | 90-99% | Partial |
| HNSW | O(N*log(N)) | O(log(N)*ef*d) | O(N*(d+M)) | 95-99.9% | Yes |
| PQ | O(N*k*d/m) | O(N*m) or O(nprobe*N/nlist*m) | O(N*m) | 80-95% | No |
| IVFPQ | O(N*k*d) | O(nprobe*N/nlist*m) | O(N*m) | 85-98% | Partial |
| ScaNN | O(N*k*d) | varies | O(N*m) | 95-99% | Partial |
| LSH | O(N*L*K) | O(L*K) | O(N*L*K) | 70-95% | Yes |
| Annoy | O(N*T*log(N)) | O(T*log(N)) | O(N*T) | 85-95% | No |
| DiskANN | O(N*R*d) | O(R*log(N)*d) | O(graph) | 95-99% | Partial |

**Recommendation for interviews:**
> "For most production use cases, HNSW provides the best recall-speed trade-off. If memory is constrained, IVFPQ is the go-to choice. For billion-scale with limited RAM, DiskANN is worth considering. ScaNN offers state-of-the-art performance for inner product search. Always benchmark on your specific data distribution - ANN algorithm performance is data-dependent."

---

## 2.9 Key ANN Metrics

**Recall@K:** Fraction of true K nearest neighbors found by the ANN algorithm
```
Recall@10 = |ANN_results intersect True_10_NN| / 10
```

**QPS (Queries Per Second):** Throughput at a given recall level

**Latency:** p50, p95, p99 query latency

**Build time:** Time to construct the index

**Memory usage:** RAM/disk required for the index

The standard benchmark is the **ANN Benchmarks** project (ann-benchmarks.com) which evaluates algorithms across different datasets and metrics.
