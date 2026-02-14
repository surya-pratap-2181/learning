# PART 6: EMBEDDING MODEL SELECTION

---

## 6.1 MTEB Benchmark (Massive Text Embedding Benchmark)

MTEB is the standard benchmark for evaluating embedding models across multiple tasks.

### Tasks Covered:
1. **Classification**: Text classification accuracy
2. **Clustering**: Clustering quality (V-measure)
3. **Pair Classification**: Sentence pair classification (e.g., paraphrase detection)
4. **Reranking**: Re-ranking relevance (MAP)
5. **Retrieval**: Information retrieval (nDCG@10)
6. **Semantic Textual Similarity (STS)**: Sentence similarity correlation
7. **Summarization**: Summary quality assessment

### Top Models on MTEB (as of late 2024 / early 2025):

| Model | Avg Score | Retrieval | Dimensions | Max Tokens | Provider |
|-------|-----------|-----------|------------|------------|----------|
| voyage-3 | ~68-70 | ~60+ | 1024 | 32000 | Voyage AI |
| text-embedding-3-large | ~64-66 | ~58+ | 3072 | 8191 | OpenAI |
| Cohere embed-v3 | ~64-66 | ~58+ | 1024 | 512 | Cohere |
| e5-mistral-7b-instruct | ~66 | ~60+ | 4096 | 32768 | Microsoft |
| BGE-large-en-v1.5 | ~64 | ~55+ | 1024 | 512 | BAAI |
| GTE-large-en-v1.5 | ~65 | ~57+ | 1024 | 8192 | Alibaba |
| jina-embeddings-v3 | ~66 | ~59+ | 1024 | 8192 | Jina AI |
| NV-Embed-v2 | ~72 | ~62+ | 4096 | 32768 | NVIDIA |
| all-MiniLM-L6-v2 | ~56 | ~42 | 384 | 256 | Sentence-Transformers |
| text-embedding-3-small | ~62 | ~52+ | 1536 | 8191 | OpenAI |

**Important Interview Note:** MTEB scores should not be the sole selection criterion. Real-world performance depends on your specific data distribution, query patterns, and latency requirements.

---

## 6.2 Task-Specific Embeddings

Different tasks benefit from different embedding approaches:

### Retrieval/Search Embeddings:
- Optimized for query-document matching (asymmetric)
- Often use instruction prefixes: "query: " and "passage: "
- Examples: E5, BGE, GTE, Cohere (with input_type)

```python
# E5 models use instruction prefix
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")

# For queries
query_embedding = model.encode("query: What are vector databases?")

# For documents
doc_embedding = model.encode("passage: Vector databases store and index embeddings...")
```

### Symmetric vs Asymmetric Embeddings:
- **Symmetric**: Both inputs are similar in nature (e.g., sentence similarity, duplicate detection)
  - Example: "The cat sat on the mat" vs "A cat was sitting on a rug"
- **Asymmetric**: Inputs are different in nature (e.g., query vs document)
  - Example: Query: "vector database" vs Document: "Vector databases are specialized systems that store, index, and query high-dimensional embedding vectors..."

### Classification Embeddings:
- Capture features useful for downstream classifiers
- Often fine-tuned on labeled data
- SVM, logistic regression, or kNN on top of embeddings

### Clustering Embeddings:
- Produce well-separated clusters
- Good intra-cluster similarity, low inter-cluster similarity
- K-means, DBSCAN, or hierarchical clustering

---

## 6.3 Multilingual Embeddings

### Why Multilingual Embeddings Matter:
- Single model handles multiple languages
- Cross-lingual retrieval: Query in English, retrieve in French
- Reduces infrastructure complexity (one model vs one per language)

### Top Multilingual Models:

| Model | Languages | Dimensions | Notes |
|-------|-----------|------------|-------|
| multilingual-e5-large | 100+ | 1024 | Microsoft, instruction-based |
| paraphrase-multilingual-MiniLM-L12-v2 | 50+ | 384 | Fast, compact |
| Cohere embed-multilingual-v3.0 | 100+ | 1024 | Commercial API |
| jina-embeddings-v3 | 90+ | 1024 | Open weights |
| BGE-M3 | 100+ | 1024 | Dense + sparse + ColBERT |
| voyage-multilingual-2 | 100+ | 1024 | Voyage AI API |

### BGE-M3 (Multi-Functionality, Multi-Linguality, Multi-Granularity):
A standout model that produces three types of representations simultaneously:
1. **Dense embedding** (1024d): For standard vector search
2. **Sparse embedding** (lexical weights): For keyword-like matching
3. **ColBERT embedding** (token-level): For fine-grained late interaction

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences = ["What is a vector database?", "Qu'est-ce qu'une base de donnees vectorielle?"]

# Returns dense, sparse, and colbert embeddings
embeddings = model.encode(
    sentences,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True
)

dense_vecs = embeddings['dense_vecs']      # shape: (2, 1024)
sparse_vecs = embeddings['lexical_weights'] # list of dicts {token_id: weight}
colbert_vecs = embeddings['colbert_vecs']   # list of (seq_len, 1024) arrays
```

---

## 6.4 Code Embeddings

### Purpose:
- Code search: Find code snippets from natural language queries
- Code clone detection: Find similar/duplicate code
- Code documentation: Match code to documentation

### Top Code Embedding Models:

| Model | Dimensions | Languages | Notes |
|-------|-----------|-----------|-------|
| voyage-code-3 | 1024 | 20+ | Best overall, long context |
| CodeBERT | 768 | 6 | Microsoft, older but foundational |
| StarCoder Embeddings | 768 | 80+ | BigCode project |
| text-embedding-3-large | 3072 | Multi | General but works for code |
| Jina Code v2 | 768 | 30+ | Open source |
| Nomic Embed Code | 768 | Multi | Open weights |

```python
import voyageai

vo = voyageai.Client()

# Embed code
code_snippets = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "function quickSort(arr) { if (arr.length <= 1) return arr; ... }",
]

code_embeddings = vo.embed(code_snippets, model="voyage-code-3", input_type="document")

# Embed natural language query
query = "recursive function to calculate fibonacci numbers"
query_embedding = vo.embed([query], model="voyage-code-3", input_type="query")
```

---

## 6.5 Multimodal Embeddings

### What Are Multimodal Embeddings?
Embeddings that map different modalities (text, image, audio, video) into a shared vector space, enabling cross-modal search.

### Key Models:

**CLIP (OpenAI, 2021):**
- Maps images and text to the same 512/768d space
- Contrastive learning on 400M image-text pairs
- Can search images with text queries and vice versa

**SigLIP (Google, 2023):**
- Improved CLIP with sigmoid loss instead of softmax
- Better performance, especially at scale
- More efficient training

**ImageBind (Meta, 2023):**
- Maps 6 modalities to a shared space: image, text, audio, depth, thermal, IMU
- Zero-shot cross-modal retrieval

**Nomic Embed Vision (2024):**
- Aligned with Nomic Embed text model
- Shared space for text and images

**Jina CLIP v2 (2024):**
- Text-image alignment with multilingual support
- 1024 dimensions

```python
# CLIP for multimodal embeddings
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Embed image
image = Image.open("photo.jpg")
image_inputs = processor(images=image, return_tensors="pt")
image_embedding = model.get_image_features(**image_inputs)  # (1, 768)

# Embed text
text_inputs = processor(text=["a photo of a cat"], return_tensors="pt")
text_embedding = model.get_text_features(**text_inputs)  # (1, 768)

# Cross-modal similarity
similarity = torch.cosine_similarity(image_embedding, text_embedding)

# OpenAI multimodal (2024+)
# Note: OpenAI does not yet have a public multimodal embedding model
# but this is expected in the near future
```

---

## 6.6 Choosing the Right Embedding Model - Decision Framework

### Step 1: Define Your Task
```
Retrieval/Search -> Asymmetric models (E5, BGE, Cohere, Voyage)
Similarity/Matching -> Symmetric models (all-MiniLM, mpnet)
Classification -> Fine-tunable models (BERT, E5)
Clustering -> Models with good clustering scores on MTEB
Code Search -> Voyage-code-3, CodeBERT, StarCoder
Multilingual -> BGE-M3, multilingual-e5, Cohere multilingual
Multimodal -> CLIP, SigLIP, Jina CLIP v2
```

### Step 2: Consider Constraints
```
Latency requirements:
  < 5ms per query -> Small models (MiniLM-L6: 384d)
  < 20ms per query -> Medium models (mpnet: 768d)
  < 100ms per query -> Large models (e5-mistral: 4096d)
  API acceptable -> OpenAI, Cohere, Voyage

Memory/Storage budget:
  Tight -> 384d models + quantization
  Moderate -> 768-1024d models
  Generous -> 1536-3072d models

Privacy/On-premise:
  Required -> Open-source models (E5, BGE, GTE)
  Not needed -> API models (OpenAI, Cohere, Voyage)
```

### Step 3: Evaluate on YOUR Data
```python
# Always benchmark on your own data!
from sentence_transformers import SentenceTransformer, evaluation

# Load your evaluation dataset
# Format: (query, positive_doc, negative_doc) triples
evaluator = evaluation.InformationRetrievalEvaluator(
    queries=your_queries,
    corpus=your_corpus,
    relevant_docs=your_relevance_labels,
    name="my-domain-eval"
)

models_to_test = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
    "BAAI/bge-large-en-v1.5",
]

for model_name in models_to_test:
    model = SentenceTransformer(model_name)
    result = evaluator(model)
    print(f"{model_name}: nDCG@10={result['my-domain-eval_ndcg@10']:.4f}")
```

### Step 4: Consider Fine-tuning
If off-the-shelf models don't meet requirements, fine-tune:

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Training data: (anchor, positive, negative) triples
train_examples = [
    InputExample(texts=["query text", "relevant doc", "irrelevant doc"]),
    # ... more examples
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.TripletLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./fine-tuned-model"
)
```

---

## 6.7 Embedding Model Pricing Comparison (2025)

| Provider | Model | Price per 1M Tokens | Dimensions | Notes |
|----------|-------|--------------------:|-----------|-------|
| OpenAI | text-embedding-3-small | $0.02 | 1536 | Cheapest API |
| OpenAI | text-embedding-3-large | $0.13 | 3072 | Higher quality |
| Cohere | embed-english-v3.0 | $0.10 | 1024 | Input types |
| Cohere | embed-english-light-v3.0 | $0.01 | 384 | Budget option |
| Voyage AI | voyage-3 | $0.06 | 1024 | High quality |
| Voyage AI | voyage-3-lite | $0.02 | 512 | Budget |
| Jina AI | jina-embeddings-v3 | $0.02 | 1024 | Flexible |
| Self-hosted | any open-source | GPU cost only | varies | Full control |

**Self-hosting cost estimate:**
- Single A10G GPU (~$1/hr): Can serve ~500-2000 req/s for small models
- Monthly: ~$720 for 24/7 operation
- Break-even vs API: Typically at 5-10B tokens/month
