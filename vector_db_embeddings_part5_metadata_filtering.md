---
title: "Part 5 - Metadata Filtering"
layout: default
parent: "Vector Databases & Embeddings"
nav_order: 6
---

# PART 5: METADATA FILTERING

---

## 5.1 What Is Metadata Filtering?

Metadata filtering narrows vector search results based on structured attributes (metadata/payload) attached to vectors. Examples:
- "Find similar products, but only in the 'Electronics' category"
- "Find relevant documents, but only those created after 2024-01-01"
- "Find similar users, but only in the 'US' region"

This is one of the most critical production features and a common interview topic because naive implementations can destroy performance.

---

## 5.2 Pre-Filtering vs Post-Filtering

### Post-Filtering (Filter After Search)

**How it works:**
1. Perform ANN search to find top-K nearest neighbors (ignoring filters)
2. Apply metadata filters to remove non-matching results
3. Return remaining results

**Problem - The Empty Result Set Issue:**
If the filter is highly selective (e.g., only 1% of vectors match), you might:
- Search for top-100 nearest neighbors
- Filter out 99 of them
- Return only 1 result (or zero!)
- User asked for 10 results but got 1

**Mitigation:** Over-fetch by a multiplier (e.g., fetch top-1000, filter to get 10). But this is wasteful and unpredictable.

```python
# Post-filtering pseudocode
def post_filter_search(query_vec, k, metadata_filter, overfetch=10):
    # Step 1: ANN search (fetch more than needed)
    candidates = ann_index.search(query_vec, k=k * overfetch)

    # Step 2: Filter
    filtered = [c for c in candidates if matches_filter(c, metadata_filter)]

    # Step 3: Return top-k
    return filtered[:k]
    # Problem: filtered might have < k results!
```

**When post-filtering works well:**
- Low selectivity filters (>50% of data matches)
- When you can afford to over-fetch
- When filter criteria are simple

### Pre-Filtering (Filter Before Search)

**How it works:**
1. First identify which vectors match the metadata filter
2. Then search only within the matching subset

**Challenge:** You need to intersect the filter results with the ANN index, which can be complex.

**Approaches:**
1. **Partition-based**: Create separate indexes per filter value (e.g., one HNSW graph per category)
2. **Bitmap intersection**: Use bitmap indexes for metadata, intersect with ANN candidate list
3. **Filter-aware graph traversal**: Modify HNSW traversal to skip non-matching nodes

**Problems with naive pre-filtering:**
- Separate indexes per filter value: Too many indexes, memory explosion
- Restricting graph traversal: Can disconnect the navigable small world graph, causing recall collapse

```python
# Pre-filtering approaches

# Approach 1: Partition indexes (simple but memory-heavy)
indexes = {}
for category in categories:
    mask = metadata['category'] == category
    indexes[category] = build_hnsw_index(vectors[mask])

def pre_filter_search_partitioned(query_vec, k, category):
    return indexes[category].search(query_vec, k=k)

# Approach 2: ID-based filtering (FAISS IDSelectorBatch)
import faiss

index = faiss.IndexHNSWFlat(768, 32)
index.add(all_vectors)

# Get IDs matching filter
matching_ids = np.array([i for i, m in enumerate(metadata) if m['category'] == 'tech'])

# Search with ID selector
selector = faiss.IDSelectorBatch(matching_ids)
params = faiss.SearchParametersHNSW()
params.sel = selector
D, I = index.search(query_vec.reshape(1, -1), k=10, params=params)
```

### In-Filtering (Filter During Search) - The Best Approach

**How it works:**
Integrate filtering directly into the ANN algorithm's traversal. During HNSW graph navigation, skip nodes that don't match the filter but continue navigating through them.

This is what Qdrant and other modern vector databases implement.

**Key insight:** The graph structure is preserved (no disconnection), but non-matching nodes are skipped when building the result set. The beam search continues through filtered-out nodes to maintain graph connectivity.

```
# Pseudocode for filter-aware HNSW search
def hnsw_search_with_filter(query, k, filter_fn, ef_search):
    candidates = priority_queue()
    results = priority_queue()
    visited = set()

    # Start from entry point
    candidates.push(entry_point)

    while not candidates.empty() and len(results) < ef_search:
        current = candidates.pop_nearest()

        if current in visited:
            continue
        visited.add(current)

        # KEY: Always explore neighbors for graph connectivity
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                candidates.push(neighbor)

        # But only add to results if filter matches
        if filter_fn(current.metadata):
            results.push(current)

    return results.top_k(k)
```

---

## 5.3 Performance Implications

### The Selectivity Problem:

| Filter Selectivity | Post-Filter | Pre-Filter | In-Filter |
|-------------------|-------------|------------|-----------|
| 90% match (low) | Good | Good | Good |
| 50% match (medium) | OK (2x overfetch) | OK | Good |
| 10% match (high) | Bad (10x overfetch) | OK if indexed | Good |
| 1% match (very high) | Terrible | OK if indexed | Moderate |
| 0.01% match (extreme) | Broken | Good if indexed | Can degrade |

### Qdrant's Approach (Best in Class):
- Uses payload indexes (keyword, integer, float, geo, datetime, full-text)
- Filter-aware HNSW: Integrated into graph traversal
- Automatically chooses strategy based on filter selectivity
- For very selective filters, falls back to filtered brute-force on matching vectors

### Weaviate's Approach:
- Roaring bitmaps for efficient filter intersection
- Allow-list strategy for pre-filtering
- ACORN: A filter-aware HNSW variant (added in 2024)

### Milvus's Approach:
- Partition keys for automatic data partitioning
- Expression-based filtering with operator pushdown
- Supports index creation on scalar fields

### Pinecone's Approach:
- Automatic optimization of filter execution
- Metadata indexes for common filter patterns
- Transparent to the user (managed service handles strategy)

---

## 5.4 Metadata Index Types

Different data types benefit from different index structures:

| Data Type | Index Type | Example Queries |
|-----------|-----------|----------------|
| Keyword/Tag | Inverted index / Bitmap | `category == "tech"` |
| Integer/Float | B-tree / Sorted array | `price >= 10 AND price <= 50` |
| DateTime | B-tree / Range index | `created_at > "2024-01-01"` |
| Geo | R-tree / Geohash | `location within 10km of (lat, lon)` |
| Boolean | Bitmap | `is_published == true` |
| Text | Full-text inverted index | `description contains "database"` |
| Array/List | Multi-value inverted index | `tags contains "ml"` |

---

## 5.5 Best Practices for Metadata Filtering

### 1. Index Your Filter Fields
```python
# Qdrant: Create payload index
client.create_payload_index(
    collection_name="documents",
    field_name="category",
    field_schema="keyword"  # or "integer", "float", "geo", "datetime", "text"
)
```

### 2. Use Partition Keys for High-Cardinality Filters
```python
# Milvus: Partition key
fields = [
    FieldSchema("tenant_id", DataType.VARCHAR, max_length=64, is_partition_key=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
]
# Data is automatically partitioned by tenant_id
```

### 3. Avoid Deeply Nested Filters
```python
# BAD: Deeply nested complex filter
filter = {
    "$and": [
        {"$or": [{"a": 1}, {"b": 2}]},
        {"$or": [{"c": 3}, {"$and": [{"d": 4}, {"e": 5}]}]}
    ]
}

# BETTER: Flatten when possible
# Use application logic to simplify filter expressions
```

### 4. Denormalize for Filter Performance
- Store frequently filtered fields directly in metadata
- Avoid needing joins or lookups during search
- Trade storage for query performance

### 5. Monitor Filter Selectivity
- Track what percentage of vectors pass each filter
- Alert on queries with < 1% selectivity (potential performance issue)
- Consider partition-based approaches for very selective filters

---

## 5.6 Interview Answer Template

> "Metadata filtering in vector databases can be implemented as post-filtering, pre-filtering, or in-filtering. Post-filtering is simplest but breaks down with selective filters because you may not get enough results. Pre-filtering can disconnect the HNSW graph, degrading recall. The best approach is in-filtering, where the filter is integrated into the graph traversal - nodes are explored for connectivity but only counted as results if they pass the filter. Qdrant does this particularly well with their filter-aware HNSW implementation. In production, I always create payload indexes on frequently filtered fields and monitor filter selectivity to catch performance issues early."
