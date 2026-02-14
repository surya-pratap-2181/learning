---
title: "Part 1 - Fundamentals"
layout: default
parent: "Vector Databases & Embeddings"
nav_order: 2
---

# Vector Databases & Embeddings - Complete Interview Guide for AI Engineers (2025-2026)

# PART 1: EMBEDDING FUNDAMENTALS

---

## 1.1 What Are Embeddings?

Embeddings are dense, fixed-length vector representations of data (text, images, audio, code) in a continuous vector space. They capture semantic meaning such that similar items are located near each other in the vector space.

**Key Properties:**
- **Dense**: Unlike sparse representations (bag-of-words, TF-IDF), every dimension carries information
- **Fixed-length**: Regardless of input length, the output vector has a consistent dimensionality
- **Learned**: The mapping is learned from data, not hand-engineered
- **Semantic**: Items with similar meanings have similar vector representations

**Interview Answer Template:**
> "An embedding is a learned mapping from high-dimensional, discrete data into a lower-dimensional, continuous vector space. The key insight is that the geometric relationships in this space (distances, directions) correspond to semantic relationships in the original data. For example, the embedding for 'king' minus 'man' plus 'woman' approximately equals the embedding for 'queen' in word2vec."

---

## 1.2 Word2Vec (2013, Mikolov et al.)

Word2Vec produces static word-level embeddings using shallow neural networks.

### Two Architectures:

**CBOW (Continuous Bag of Words):**
- Predicts the center word from surrounding context words
- Faster to train, works well for frequent words
- Input: context words -> Output: target word

**Skip-gram:**
- Predicts surrounding context words from the center word
- Works better for rare words, smaller datasets
- Input: target word -> Output: context words

### Training Details:
- Typical dimensions: 100-300
- Window size: usually 5-10 words
- Negative sampling: Instead of computing softmax over entire vocabulary, sample ~5-20 negative examples
- Subword information: Not captured (later addressed by FastText)

### Limitations:
- **Static embeddings**: One vector per word regardless of context ("bank" has the same vector whether it means "river bank" or "financial bank")
- **Out-of-vocabulary (OOV)**: Cannot handle words not seen during training
- **Word-level only**: No sentence or document embeddings natively

```python
# Word2Vec with Gensim
from gensim.models import Word2Vec

sentences = [["king", "is", "a", "male", "ruler"],
             ["queen", "is", "a", "female", "ruler"]]

model = Word2Vec(sentences, vector_size=100, window=5,
                 min_count=1, sg=1)  # sg=1 for skip-gram

# Get word vector
vector = model.wv['king']  # shape: (100,)

# Find similar words
similar = model.wv.most_similar('king', topn=5)

# Analogy: king - man + woman = ?
result = model.wv.most_similar(positive=['king', 'woman'],
                                negative=['man'], topn=1)
```

---

## 1.3 GloVe (Global Vectors, 2014, Pennington et al.)

GloVe combines the benefits of count-based methods (like LSA) with prediction-based methods (like Word2Vec).

### How It Works:
1. Build a global word-word co-occurrence matrix X from the corpus
2. The objective is to learn vectors such that their dot product equals the log of the co-occurrence probability:
   `w_i . w_j + b_i + b_j = log(X_ij)`
3. Uses a weighted least squares objective with a weighting function that caps the influence of very frequent co-occurrences

### Key Differences from Word2Vec:
| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| Training | Sliding window, local context | Global co-occurrence matrix |
| Approach | Predictive | Count-based + predictive hybrid |
| Memory | Lower (streaming) | Higher (full matrix needed) |
| Performance | Better on analogy tasks | Better on word similarity |

### Pre-trained Dimensions:
- GloVe 6B: 50d, 100d, 200d, 300d (trained on Wikipedia + Gigaword)
- GloVe 42B: 300d (Common Crawl)
- GloVe 840B: 300d (Common Crawl)

```python
# Loading GloVe embeddings
import numpy as np

def load_glove(path, dim=300):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove = load_glove('glove.6B.300d.txt')
king_vec = glove['king']  # shape: (300,)
```

---

## 1.4 Contextual Embeddings

### BERT Embeddings (2018, Devlin et al.)

BERT (Bidirectional Encoder Representations from Transformers) produces **contextual** embeddings - the same word gets different vectors based on context.

**Architecture:**
- Transformer encoder (12 layers for base, 24 for large)
- BERT-base: 768-dimensional embeddings
- BERT-large: 1024-dimensional embeddings
- Trained with Masked Language Model (MLM) + Next Sentence Prediction (NSP)

**How to extract embeddings:**
- **[CLS] token**: Often used as a sentence-level representation
- **Mean pooling**: Average all token embeddings (usually better for similarity)
- **Max pooling**: Take element-wise maximum across tokens
- **Layer selection**: Different layers capture different information (lower = syntax, higher = semantics)

**Limitation for search/retrieval:**
- BERT requires both texts to be processed together (cross-encoder) for best accuracy
- This makes it O(n) for retrieval - you cannot pre-compute embeddings independently
- Solution: Sentence-BERT (bi-encoder approach)

```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Vector databases store embeddings efficiently"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

# [CLS] token embedding
cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, 768)

# Mean pooling (better for similarity)
attention_mask = inputs['attention_mask']
token_embeddings = outputs.last_hidden_state
input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
mean_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
```

### Sentence-Transformers (2019, Reimers & Gurevych)

Sentence-Transformers (SBERT) fine-tune BERT-like models using siamese/triplet networks to produce semantically meaningful sentence embeddings that can be compared with cosine similarity.

**Key Innovation:**
- Bi-encoder architecture: Each sentence is encoded independently
- Embeddings can be pre-computed and stored in a vector database
- At query time, only the query needs encoding -> O(1) per comparison after indexing

**Popular Models (2025):**
| Model | Dimensions | Max Tokens | Use Case |
|-------|-----------|------------|----------|
| all-MiniLM-L6-v2 | 384 | 256 | Fast, general purpose |
| all-mpnet-base-v2 | 768 | 384 | Higher quality, general |
| multi-qa-mpnet-base-dot-v1 | 768 | 512 | Question-answering |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 128 | Multilingual |
| e5-large-v2 | 1024 | 512 | High quality retrieval |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Vector databases are optimized for similarity search",
    "Databases designed for vector similarity queries",
    "The weather is nice today"
]

# Encode sentences to embeddings
embeddings = model.encode(sentences)  # shape: (3, 384)

# Compute cosine similarity
from sentence_transformers.util import cos_sim
similarities = cos_sim(embeddings[0], embeddings[1:])
# Result: [[0.87, 0.12]] - first pair is similar, second is not
```

---

## 1.5 OpenAI Embeddings

### Models (as of 2025):
| Model | Dimensions | Max Tokens | Price (per 1M tokens) |
|-------|-----------|------------|----------------------|
| text-embedding-3-small | 1536 (default), supports 512 | 8191 | ~$0.02 |
| text-embedding-3-large | 3072 (default), supports 256-3072 | 8191 | ~$0.13 |
| text-embedding-ada-002 | 1536 | 8191 | ~$0.10 (legacy) |

### Key Features:
- **Matryoshka Representation Learning (MRL)**: text-embedding-3 models support dimension reduction while preserving quality. You can truncate vectors to 256d and still get reasonable performance
- **Normalized outputs**: Vectors have unit length (L2 norm = 1), so cosine similarity = dot product
- **Native shortening**: Pass `dimensions` parameter to get shorter vectors

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Vector databases store high-dimensional vectors",
    dimensions=512  # MRL: reduce from 1536 to 512
)

embedding = response.data[0].embedding  # list of 512 floats

# Batch embedding
texts = ["first document", "second document", "third document"]
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=texts
)
embeddings = [item.embedding for item in response.data]
```

---

## 1.6 Cohere Embeddings

### Models (as of 2025):
| Model | Dimensions | Max Tokens | Features |
|-------|-----------|------------|----------|
| embed-english-v3.0 | 1024 | 512 | Search, classification, clustering |
| embed-multilingual-v3.0 | 1024 | 512 | 100+ languages |
| embed-english-light-v3.0 | 384 | 512 | Faster, lighter |
| embed-multilingual-light-v3.0 | 384 | 512 | Lighter multilingual |

### Key Feature - Input Types:
Cohere v3 models accept an `input_type` parameter that optimizes embeddings for specific tasks:
- `search_document`: For documents to be stored and searched
- `search_query`: For search queries
- `classification`: For text classification
- `clustering`: For clustering tasks

This asymmetric embedding approach significantly improves retrieval quality.

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

# Embed documents (for storage)
docs = ["Vector databases enable semantic search",
        "HNSW is an efficient ANN algorithm"]
doc_embeddings = co.embed(
    texts=docs,
    model="embed-english-v3.0",
    input_type="search_document"
).embeddings

# Embed query (for searching)
query_embedding = co.embed(
    texts=["How do vector databases work?"],
    model="embed-english-v3.0",
    input_type="search_query"
).embeddings
```

---

## 1.7 Voyage AI Embeddings

Voyage AI (acquired by Anthropic in 2025) provides high-quality domain-specific embedding models.

### Models:
| Model | Dimensions | Max Tokens | Specialty |
|-------|-----------|------------|-----------|
| voyage-3 | 1024 | 32000 | General purpose, state-of-the-art |
| voyage-3-lite | 512 | 32000 | Lightweight general purpose |
| voyage-code-3 | 1024 | 32000 | Code retrieval |
| voyage-finance-2 | 1024 | 32000 | Financial domain |
| voyage-law-2 | 1024 | 16000 | Legal domain |
| voyage-multilingual-2 | 1024 | 32000 | Multilingual |

### Key Features:
- **Very long context**: Up to 32K tokens (much longer than most competitors)
- **Domain-specific models**: Fine-tuned for code, finance, law, healthcare
- **High MTEB scores**: Consistently top-ranking on embedding benchmarks

```python
import voyageai

vo = voyageai.Client()

# General embedding
result = vo.embed(
    ["Vector databases are essential for RAG"],
    model="voyage-3",
    input_type="document"
)
embedding = result.embeddings[0]  # 1024-dim vector

# Code embedding
code_result = vo.embed(
    ["def binary_search(arr, target): ..."],
    model="voyage-code-3",
    input_type="document"
)
```

---

## 1.8 Dimensionality

### What Dimensionality Means:
- The number of components in the embedding vector
- Higher dimensions can capture more nuanced relationships
- But higher dimensions also mean more storage, memory, and compute

### Common Dimensionalities:
| Range | Examples | Trade-off |
|-------|----------|-----------|
| 50-100 | GloVe-50d, GloVe-100d | Very fast, lower quality |
| 256-384 | MiniLM, Cohere-light | Good balance for production |
| 512-768 | BERT-base, MPNet | Standard quality |
| 1024 | Cohere v3, Voyage, E5-large | High quality |
| 1536-3072 | OpenAI ada-002, OpenAI v3-large | Very high quality |

### Dimensionality Reduction Techniques:
1. **PCA (Principal Component Analysis)**: Linear reduction, fast
2. **Matryoshka Representation Learning**: Trained to be truncatable
3. **Random Projection**: Simple but effective for high dimensions
4. **Autoencoders**: Non-linear reduction, trainable
5. **UMAP/t-SNE**: For visualization (2D/3D), not for search

### Storage Impact:
- Each float32 dimension = 4 bytes
- 1M vectors x 1536 dimensions x 4 bytes = ~5.7 GB
- 1M vectors x 384 dimensions x 4 bytes = ~1.4 GB
- Quantization (int8) reduces by 4x: 1M x 1536 x 1 byte = ~1.4 GB

---

## 1.9 Similarity Metrics

### Cosine Similarity

Measures the angle between two vectors. Range: [-1, 1] (or [0, 1] for positive embeddings).

**Formula:** `cos(A, B) = (A . B) / (||A|| * ||B||)`

**When to use:**
- When magnitude doesn't matter, only direction
- Most common default for text embeddings
- OpenAI and most embedding APIs return normalized vectors, so cosine = dot product

**Properties:**
- Scale-invariant (multiplying a vector by a constant doesn't change similarity)
- Bounded output [-1, 1]
- Most popular for text similarity

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# For normalized vectors (L2 norm = 1):
# cosine_similarity = dot_product
```

### Euclidean Distance (L2 Distance)

Measures the straight-line distance between two points.

**Formula:** `L2(A, B) = sqrt(sum((A_i - B_i)^2))`

**When to use:**
- When absolute positions in the vector space matter
- When vectors are NOT normalized
- Common in image embeddings and clustering

**Properties:**
- Range: [0, infinity)
- Affected by vector magnitude
- Lower distance = more similar

```python
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Squared Euclidean (faster, avoids sqrt):
def squared_euclidean(a, b):
    return np.sum((a - b) ** 2)
```

### Dot Product (Inner Product)

Measures both angle AND magnitude.

**Formula:** `dot(A, B) = sum(A_i * B_i)`

**When to use:**
- When vector magnitude carries meaning (e.g., "importance" or "relevance strength")
- Maximum Inner Product Search (MIPS) scenarios
- When vectors are normalized (equivalent to cosine similarity)

**Properties:**
- Range: (-infinity, +infinity)
- Affected by magnitude (longer vectors score higher)
- Fastest to compute

```python
def dot_product(a, b):
    return np.dot(a, b)
```

### Relationship Between Metrics (Critical Interview Knowledge):

For **normalized vectors** (||A|| = ||B|| = 1):
```
cosine_similarity(A, B) = dot_product(A, B)
euclidean_distance(A, B)^2 = 2 * (1 - cosine_similarity(A, B))
```

This means for normalized vectors, all three metrics produce equivalent rankings.

### Which Metric to Choose:

| Metric | Best For | Embedding Models |
|--------|----------|-----------------|
| Cosine | Text similarity, general purpose | OpenAI, Cohere, Sentence-Transformers |
| Euclidean | Clustering, image similarity | Raw BERT, custom models |
| Dot Product | MIPS, recommendation systems | Models trained with dot product loss |

### Manhattan Distance (L1 Distance):
- `L1(A, B) = sum(|A_i - B_i|)`
- More robust to outliers than L2
- Used less commonly in practice for embeddings

### Hamming Distance:
- For binary vectors only
- Counts the number of positions where bits differ
- Extremely fast (bitwise XOR + popcount)
- Used with binary quantization of embeddings
