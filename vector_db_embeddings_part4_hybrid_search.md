---
title: "Part 4 - Hybrid Search"
layout: default
parent: "Vector Databases & Embeddings"
nav_order: 5
---

# PART 4: HYBRID SEARCH

---

## 4.1 Why Hybrid Search?

Pure vector search (dense retrieval) has limitations:
- **Exact keyword matching**: Dense models may miss exact terms (e.g., product SKUs, error codes, proper nouns)
- **Rare/domain terms**: Embedding models may not have good representations for rare terminology
- **Lexical precision**: Sometimes you need exact string matching (e.g., "Python 3.11" vs "Python 3.12")

Pure keyword search (sparse retrieval) has limitations:
- **Vocabulary mismatch**: "car" won't match "automobile"
- **No semantic understanding**: Cannot understand meaning, only lexical overlap
- **Rigid**: Cannot handle paraphrases or conceptual similarity

**Hybrid search combines both** to get the best of both worlds.

---

## 4.2 BM25 (Best Matching 25)

BM25 is the standard keyword/sparse retrieval algorithm, an improvement over TF-IDF.

### Formula:
```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d|/avgdl))
```

Where:
- `f(t,d)` = term frequency of term t in document d
- `|d|` = document length
- `avgdl` = average document length
- `k1` = term frequency saturation parameter (typically 1.2-2.0)
- `b` = document length normalization (typically 0.75)
- `IDF(t)` = inverse document frequency of term t

### Key Insights for Interviews:
- **k1**: Controls term frequency saturation. Higher k1 = more weight to term frequency
- **b**: Controls length normalization. b=0 means no length normalization, b=1 means full normalization
- **IDF**: Rare terms get higher scores (discriminative power)
- **Sublinear TF scaling**: Unlike TF-IDF, BM25 saturates - the 10th occurrence of a term adds less than the 2nd

```python
# BM25 with rank_bm25 library
from rank_bm25 import BM25Okapi

corpus = [
    "vector databases store embeddings",
    "HNSW algorithm for approximate search",
    "embeddings represent semantic meaning",
    "databases can scale horizontally"
]

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "vector database embeddings"
tokenized_query = query.split()
scores = bm25.get_scores(tokenized_query)
top_docs = bm25.get_top_n(tokenized_query, corpus, n=3)
```

---

## 4.3 Learned Sparse Representations (SPLADE, etc.)

Modern alternative to BM25 that learns sparse representations using transformers.

### SPLADE (SParse Lexical AnD Expansion model):
- Uses a transformer (BERT) to predict term importance weights
- Can add terms NOT in the original text (query/document expansion)
- Output: sparse vector over vocabulary (30K-50K dimensions, mostly zeros)
- Typically outperforms BM25 while maintaining interpretability

### Other Learned Sparse Models:
- **SPLADE++**: Improved training with distillation
- **Elastic's ELSER**: Elastic Learned Sparse EncodeR
- **DeepImpact**: Learns term impact scores
- **uniCOIL**: Context-aware term weighting
- **SPLADE v3**: Latest version with improved efficiency

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# SPLADE model
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")

def get_splade_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    # SPLADE aggregation: log(1 + ReLU(logits)) * attention_mask
    sparse_vec = torch.max(
        torch.log(1 + torch.relu(output.logits)) * tokens["attention_mask"].unsqueeze(-1),
        dim=1
    ).values.squeeze()
    return sparse_vec

sparse = get_splade_embedding("What are vector databases?")
# Result: sparse vector with ~100-500 non-zero values out of 30522
non_zero = torch.nonzero(sparse).squeeze()
```

---

## 4.4 Dense + Sparse Combination Strategies

### Strategy 1: Score-Level Fusion (Late Fusion)

Run both retrievers independently, then combine their scores.

**Linear Combination:**
```
final_score = alpha * dense_score + (1 - alpha) * sparse_score
```
- alpha = 1.0: Pure dense retrieval
- alpha = 0.0: Pure sparse retrieval
- alpha = 0.5-0.7: Common sweet spot

**Challenge:** Scores from different systems are on different scales. Need normalization.

**Normalization Methods:**
1. **Min-max normalization**: Scale to [0, 1]
2. **Z-score normalization**: (score - mean) / std
3. **Rank-based**: Use ranks instead of raw scores (see RRF below)

```python
def linear_fusion(dense_results, sparse_results, alpha=0.6):
    """Combine dense and sparse results with score normalization."""
    # Normalize dense scores (min-max)
    dense_scores = {r['id']: r['score'] for r in dense_results}
    d_min, d_max = min(dense_scores.values()), max(dense_scores.values())
    dense_norm = {k: (v - d_min) / (d_max - d_min + 1e-8) for k, v in dense_scores.items()}

    # Normalize sparse scores
    sparse_scores = {r['id']: r['score'] for r in sparse_results}
    s_min, s_max = min(sparse_scores.values()), max(sparse_scores.values())
    sparse_norm = {k: (v - s_min) / (s_max - s_min + 1e-8) for k, v in sparse_scores.items()}

    # Combine
    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    combined = {}
    for doc_id in all_ids:
        d_score = dense_norm.get(doc_id, 0)
        s_score = sparse_norm.get(doc_id, 0)
        combined[doc_id] = alpha * d_score + (1 - alpha) * s_score

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

### Strategy 2: Reciprocal Rank Fusion (RRF)

RRF combines results based on their ranks rather than scores, avoiding normalization issues.

**Formula:**
```
RRF_score(d) = sum over retrievers r: 1 / (k + rank_r(d))
```
Where k is a constant (typically 60).

**Why k=60?** Research by Cormack et al. (2009) found this value performs robustly across diverse datasets. The constant prevents top-ranked documents from dominating too heavily.

```python
def reciprocal_rank_fusion(result_lists, k=60):
    """
    Combine multiple ranked lists using RRF.

    Args:
        result_lists: List of lists, each containing (doc_id, score) tuples
                      ordered by relevance (best first)
        k: Smoothing constant (default 60)

    Returns:
        Combined ranking as list of (doc_id, rrf_score) tuples
    """
    rrf_scores = {}

    for result_list in result_lists:
        for rank, (doc_id, _score) in enumerate(result_list, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# Example usage
dense_results = [("doc3", 0.95), ("doc1", 0.87), ("doc5", 0.82)]
sparse_results = [("doc1", 12.5), ("doc3", 10.2), ("doc7", 8.1)]

combined = reciprocal_rank_fusion([dense_results, sparse_results])
# doc3 and doc1 will rank highest (appear in both lists)
```

### Strategy 3: Retrieval-Level Fusion (Early Fusion)

Combine dense and sparse vectors into a single representation before search.

**Approach 1: Concatenation**
- Concatenate dense + sparse vectors
- Index as a single high-dimensional vector
- Challenge: Different scales, very high dimensionality

**Approach 2: Pinecone Sparse-Dense**
- Store both sparse and dense vectors per document
- Single query with both representations
- Pinecone handles the fusion internally

```python
# Pinecone hybrid search (sparse-dense)
from pinecone import Pinecone

pc = Pinecone(api_key="KEY")
index = pc.Index("hybrid-index")

# Query with both dense and sparse vectors
results = index.query(
    vector=dense_query_vector,       # Dense: [0.1, 0.2, ...]
    sparse_vector={
        "indices": [102, 3457, 9821],  # Vocabulary indices
        "values": [0.5, 1.2, 0.8]     # BM25/SPLADE weights
    },
    top_k=10
)
```

**Approach 3: ColBERT (Contextualized Late Interaction)**
- Generate multiple vectors per document (one per token)
- At query time, compute MaxSim between query token vectors and document token vectors
- Superior quality but higher storage/compute cost

---

## 4.5 Hybrid Search in Different Databases

### Weaviate Hybrid Search:
```python
# Weaviate hybrid search with alpha parameter
results = collection.query.hybrid(
    query="vector database performance",
    alpha=0.5,           # 0 = BM25 only, 1 = vector only
    limit=10,
    fusion_type="relative_score"  # or "ranked" for RRF
)
```

### Elasticsearch Hybrid Search:
```json
// Elasticsearch RRF hybrid
POST /my-index/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "match": { "content": "vector database" }
            }
          }
        },
        {
          "knn": {
            "field": "embedding",
            "query_vector": [0.1, 0.2, ...],
            "k": 10,
            "num_candidates": 100
          }
        }
      ],
      "rank_window_size": 100,
      "rank_constant": 60
    }
  }
}
```

### Qdrant Hybrid Search (with sparse vectors):
```python
from qdrant_client.models import SparseVector, NamedSparseVector, Prefetch, FusionQuery, Fusion

# Create collection with both dense and sparse vectors
client.create_collection(
    collection_name="hybrid",
    vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()}
)

# Hybrid query using RRF
results = client.query_points(
    collection_name="hybrid",
    prefetch=[
        Prefetch(query=dense_query_vector, using="dense", limit=20),
        Prefetch(
            query=SparseVector(indices=[1, 5, 100], values=[0.5, 0.8, 1.2]),
            using="sparse",
            limit=20
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10
)
```

---

## 4.6 When to Use Hybrid Search

| Scenario | Recommendation |
|----------|---------------|
| General semantic search | Dense only (simpler) |
| E-commerce product search | Hybrid (SKUs + semantic) |
| Legal/medical document search | Hybrid (exact terminology matters) |
| Code search | Hybrid (function names + semantic) |
| FAQ/knowledge base | Dense usually sufficient |
| Multi-language search | Dense (embeddings handle cross-lingual) |
| Log/error search | Sparse/keyword (exact matching critical) |

### Interview Answer:
> "Hybrid search combines the semantic understanding of dense retrieval with the lexical precision of sparse retrieval. The typical approach is to run both retrievers independently and combine results using Reciprocal Rank Fusion (RRF). RRF is preferred because it's rank-based, avoiding the need to normalize heterogeneous score distributions. In practice, I've found that hybrid search improves recall by 5-15% over dense-only retrieval, especially for queries containing domain-specific terms or proper nouns."
