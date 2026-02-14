
# RAG Interview Guide
# PART 5: COMMON INTERVIEW QUESTIONS WITH DETAILED ANSWERS AND CODE

---

## SECTION A: FUNDAMENTAL QUESTIONS

### Q1: "Explain the complete RAG pipeline from document ingestion to response generation."

**Answer:**

**Ingestion Pipeline (Offline):**
1. **Document Loading**: Ingest documents from various sources (PDFs, databases, APIs, web) using format-specific loaders
2. **Pre-processing**: Clean text, remove boilerplate, extract metadata, handle special elements (tables, images)
3. **Chunking**: Split documents into smaller pieces using a strategy appropriate for the content type. Typically recursive character splitting with 512-1000 tokens and 10-20% overlap
4. **Embedding**: Convert each chunk into a dense vector using an embedding model (e.g., text-embedding-3-small). Both the text and metadata are stored
5. **Indexing**: Store vectors in a vector database with appropriate index type (HNSW for balanced speed/accuracy)

**Query Pipeline (Online):**
1. **Query Processing**: Optionally transform the query (rewrite, expand, decompose)
2. **Embedding**: Convert query to vector using the SAME embedding model used for documents
3. **Retrieval**: Find top-k most similar chunks using vector similarity search (+ optional keyword search for hybrid)
4. **Post-retrieval**: Re-rank results, compress context, filter by threshold
5. **Prompt Assembly**: Combine query + retrieved context + system instructions into a prompt
6. **Generation**: LLM generates answer grounded in the provided context
7. **Post-processing**: Extract citations, validate faithfulness, format response

```python
# Complete minimal RAG pipeline
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load
loader = PyPDFLoader("knowledge_base.pdf")
documents = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Embed and Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. Create RAG chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

If the context doesn't contain enough information, say "I don't have enough information."
Cite the source documents in your answer.
""")

def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query
answer = rag_chain.invoke("What is the company's PTO policy?")
```

---

### Q2: "How do you handle conversational RAG (multi-turn conversations)?"

**Answer:** The challenge is that follow-up questions may contain pronouns or implicit references to previous turns. "What about their revenue?" only makes sense with conversation history.

**Approach 1: Query Contextualization**
Rewrite the follow-up question to be standalone using conversation history.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Contextualization prompt: rewrite question using chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question which might reference
    context in the chat history, formulate a standalone question which can be understood
    without the chat history. Do NOT answer the question, just reformulate it if needed
    and otherwise return it as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the following context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Usage with chat history
chat_history = []

response1 = rag_chain.invoke({
    "input": "What is Apple's revenue?",
    "chat_history": chat_history
})
chat_history.extend([
    HumanMessage(content="What is Apple's revenue?"),
    AIMessage(content=response1["answer"])
])

# Follow-up question
response2 = rag_chain.invoke({
    "input": "How does that compare to last year?",  # "that" refers to Apple's revenue
    "chat_history": chat_history
})
# The contextualized query becomes: "How does Apple's revenue compare to last year?"
```

**Approach 2: Memory-Augmented Retrieval**
Include conversation summary in retrieval query.

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500,
    return_messages=True,
    memory_key="chat_history"
)
```

---

### Q3: "What is the difference between a vector store retriever and a knowledge graph retriever? When would you use each?"

**Answer:**

| Aspect | Vector Store | Knowledge Graph |
|--------|-------------|-----------------|
| Data representation | Unstructured text chunks as vectors | Structured entities and relationships as triples |
| Query type | Semantic similarity | Graph traversal, pattern matching |
| Strength | Finding similar passages | Multi-hop reasoning, relationship queries |
| Weakness | No understanding of relationships | Requires structured extraction, brittle to extraction errors |
| Scale | Billions of vectors | Millions of entities (extraction is bottleneck) |
| Update | Easy (add/remove vectors) | Complex (maintain consistency of graph) |
| Cost | Lower (embed once) | Higher (LLM-based entity extraction) |
| Best for | General QA, semantic search | Relationship queries, structured domains |

**Use vector store when:** General QA, document search, content recommendation, when documents are unstructured prose.

**Use knowledge graph when:** Questions about relationships ("Who reports to the CEO?"), multi-hop reasoning ("Which products were developed by teams in the NY office?"), when data is naturally structured (organizational charts, product catalogs with relationships).

**Best practice:** Use both. Vector search for general retrieval, knowledge graph for relationship queries. Route queries appropriately.

---

### Q4: "How do you handle multimodal RAG (text + images + tables)?"

**Answer:**

```python
# Approach 1: Convert everything to text
# - Use OCR for images (Tesseract, AWS Textract)
# - Use table extraction (Camelot, Tabula)
# - Use vision models to describe images

from langchain_openai import ChatOpenAI
import base64

def describe_image(image_path):
    """Use GPT-4V to convert image to text description."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke([
        {
            "type": "text",
            "text": "Describe this image in detail, including any text, numbers, charts, or diagrams."
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"}
        }
    ])
    return response.content

# Approach 2: Multi-vector retrieval
# Store text embeddings AND image embeddings, retrieve from both
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore

# Create summaries for different modalities
id_key = "doc_id"
store = InMemoryByteStore()

multi_vector_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# For each document:
# 1. Generate text summary -> embed summary -> store in vectorstore
# 2. Store original (text/image/table) in docstore
# 3. At retrieval: match on summary embedding, return original content

# Approach 3: Use multimodal embedding models
# Models like CLIP, SigLIP can embed both images and text into same vector space
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('clip-ViT-B-32')
text_embedding = model.encode("a cat sitting on a desk")
image_embedding = model.encode(Image.open("cat.jpg"))
# Both are in the same vector space - can compute cosine similarity
```

---

### Q5: "Explain the lost-in-the-middle problem and how to mitigate it."

**Answer:**

Liu et al. (2023) showed that LLMs have a U-shaped attention pattern over long contexts - they attend most to the beginning and end, with degraded performance for information in the middle.

**Impact on RAG:** If the most relevant chunk is ranked 3rd out of 5 (middle of context), the LLM may ignore or underweight it.

**Mitigation strategies:**
1. **Re-rank and place best chunk first**: After retrieval, put the most relevant chunk at the top of the context
2. **Reduce number of chunks**: Use fewer, higher-quality chunks instead of many noisy ones
3. **Reversed ordering**: Place best results both first and last ("sandwich" approach)
4. **Contextual compression**: Remove irrelevant parts so context is shorter and denser
5. **Map-reduce generation**: Process each chunk independently, then synthesize (avoids long context entirely)
6. **Use models trained for long context**: Claude, GPT-4o handle this better than older models but the effect still exists

```python
def mitigate_lost_in_middle(docs, query, reranker):
    """Place most relevant docs at beginning and end."""
    # Re-rank
    ranked_docs = reranker.rerank(query, docs)

    if len(ranked_docs) <= 2:
        return ranked_docs

    # Interleave: best at start and end
    n = len(ranked_docs)
    reordered = []
    for i in range(n):
        if i % 2 == 0:
            reordered.insert(0, ranked_docs[i])  # Even ranks at beginning
        else:
            reordered.append(ranked_docs[i])  # Odd ranks at end

    return reordered
```

---

### Q6: "How do you implement streaming RAG responses?"

**Answer:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streaming response
async def stream_rag_response(question):
    async for chunk in rag_chain.astream(question):
        print(chunk, end="", flush=True)  # Print each token as it arrives

# With FastAPI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/query")
async def query_rag(question: str):
    async def generate():
        async for chunk in rag_chain.astream(question):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## SECTION B: SYSTEM DESIGN QUESTIONS

### Q7: "Design a RAG system for a customer support chatbot that handles 10,000 queries/day."

**Answer:**

**Architecture:**

```
                    +------------------+
                    |   Load Balancer  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
         +----v----+   +----v----+   +----v----+
         | API #1  |   | API #2  |   | API #3  |
         +----+----+   +----+----+   +----+----+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v--------+
                    |  Redis Cache    |
                    | (Query Cache)   |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     | Vector DB       |          | BM25 Index      |
     | (Qdrant/Weaviate)|         | (Elasticsearch) |
     +--------+--------+          +--------+--------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v--------+
                    | Re-ranker       |
                    | (Cohere/Cross)  |
                    +--------+--------+
                             |
                    +--------v--------+
                    | LLM Generation  |
                    | (GPT-4o-mini)   |
                    +-----------------+
```

**Key Design Decisions:**

1. **Caching Layer (Redis)**:
   - Cache frequent queries and their responses (exact match + semantic similarity cache)
   - 10K queries/day, likely 30-40% are repeated or similar -> 3-4K cache hits
   - TTL: 24 hours for dynamic content, 7 days for static
   - Semantic cache: embed query, check if similar query exists in cache

```python
import redis
import hashlib
import json
import numpy as np

class SemanticCache:
    def __init__(self, embeddings, threshold=0.95):
        self.redis_client = redis.Redis()
        self.embeddings = embeddings
        self.threshold = threshold

    def get(self, query):
        # Try exact match first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached = self.redis_client.get(f"exact:{cache_key}")
        if cached:
            return json.loads(cached)

        # Try semantic match
        query_embedding = self.embeddings.embed_query(query)
        # Check against cached query embeddings
        # (In production, use vector DB for this)
        return None

    def set(self, query, response, ttl=86400):
        cache_key = hashlib.md5(query.encode()).hexdigest()
        self.redis_client.setex(
            f"exact:{cache_key}",
            ttl,
            json.dumps(response)
        )
```

2. **Model Selection**:
   - Embedding: text-embedding-3-small (fast, cheap at scale)
   - LLM: GPT-4o-mini for 90% of queries (fast, cheap), GPT-4o for escalated/complex queries
   - Re-ranker: Cohere Rerank (managed, fast)

3. **Latency Budget** (target: <3 seconds):
   - Query embedding: 50ms
   - Vector search: 20ms
   - BM25 search: 10ms (parallel with vector)
   - Re-ranking: 100ms
   - LLM generation: 1-2 seconds (streaming)
   - Total: ~2 seconds

4. **Scaling**:
   - Horizontal scaling of API servers (stateless)
   - Vector DB: Qdrant in distributed mode or Pinecone (managed)
   - Read replicas for high-read workloads

5. **Monitoring**:
   - Track: latency p50/p95/p99, cache hit rate, retrieval relevance scores, user feedback
   - Alert on: latency spike, low feedback scores, high error rate

---

### Q8: "How would you build a RAG system that needs to handle documents in 20+ languages?"

**Answer:**

1. **Multilingual Embedding Models**: Use models trained on multiple languages
   - `multilingual-e5-large` (supports 100+ languages)
   - `Cohere embed-multilingual-v3.0`
   - `paraphrase-multilingual-MiniLM-L12-v2`

2. **Language Detection and Routing**: Detect query language, optionally translate

3. **Cross-lingual retrieval**: Query in French, retrieve English docs (multilingual embeddings handle this)

4. **Translation-based approach**: Translate all docs to English, retrieve in English, translate answer back

```python
# Approach 1: Multilingual embeddings (recommended)
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={
        'normalize_embeddings': True,
        'prompt': "query: "  # E5 models use instruction prefix
    }
)

# Documents in any language are embedded into shared vector space
# Query in any language retrieves relevant docs regardless of language

# Approach 2: Language-specific collections + routing
from langdetect import detect

def multilingual_rag(query):
    query_lang = detect(query)

    if query_lang in language_specific_collections:
        # Search language-specific collection first
        primary_docs = lang_collections[query_lang].similarity_search(query, k=3)
        # Also search English collection (often the largest)
        english_docs = lang_collections["en"].similarity_search(query, k=2)
        docs = primary_docs + english_docs
    else:
        # Fall back to global multilingual index
        docs = global_collection.similarity_search(query, k=5)

    return generate_response(query, docs, target_language=query_lang)
```

---

## SECTION C: OPTIMIZATION QUESTIONS

### Q9: "How do you reduce latency in a RAG pipeline?"

**Answer:**

1. **Caching** (biggest impact):
   - Exact query cache
   - Semantic query cache
   - Embedding cache (avoid re-embedding same text)

2. **Smaller/faster models**:
   - Embedding: MiniLM (384d) instead of large models
   - LLM: GPT-4o-mini or Claude Haiku instead of flagship models
   - Re-ranker: MiniLM cross-encoder instead of large cross-encoder

3. **Parallel processing**:
   - Run BM25 and vector search in parallel
   - Embed query while loading retriever

4. **Reduce retrieval scope**:
   - Metadata pre-filtering narrows search space
   - Query routing to specific collections
   - Use IVF or HNSW with aggressive pruning

5. **Streaming**: Start generating as soon as first tokens are ready

6. **Quantization**: Quantize embedding vectors (float32 -> int8) for faster search

7. **Reduce context size**: Fewer, higher-quality chunks. Contextual compression.

8. **Pre-computation**: Pre-compute answers for common queries

```python
import asyncio
import time

async def optimized_rag(query):
    start = time.time()

    # Check cache first
    cached = cache.get(query)
    if cached:
        return cached  # ~5ms

    # Parallel retrieval
    query_embedding = embeddings.embed_query(query)

    vector_task = asyncio.create_task(
        vectorstore.asimilarity_search_by_vector(query_embedding, k=10)
    )
    bm25_task = asyncio.create_task(
        bm25_retriever.ainvoke(query)
    )

    vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

    # Fuse and re-rank
    combined = fuse_results(vector_results, bm25_results)
    reranked = reranker.rerank(query, combined, top_k=5)

    # Generate with streaming
    response = await llm.ainvoke(format_prompt(query, reranked))

    # Cache result
    cache.set(query, response)

    print(f"Total latency: {time.time() - start:.2f}s")
    return response
```

---

### Q10: "How do you reduce costs in a RAG pipeline?"

**Answer:**

| Optimization | Savings | Trade-off |
|-------------|---------|-----------|
| Use smaller embedding model | 5-10x on embedding costs | Slight quality drop |
| Reduce embedding dimensions (Matryoshka) | 2-4x storage savings | Minimal quality impact |
| Use GPT-4o-mini instead of GPT-4o | 10-15x on generation | Good enough for most queries |
| Caching (exact + semantic) | 30-50% query cost reduction | Cache staleness |
| Batch embeddings | 2-3x vs one-by-one | Slight latency for batch |
| Reduce chunk count (better top-k) | Less LLM input tokens | May miss relevant info |
| Open-source models (self-hosted) | No per-token costs | Infra management overhead |
| Quantized vectors (int8 vs float32) | 4x storage reduction | ~1% recall drop |
| Contextual compression | 30-60% fewer tokens to LLM | Extra LLM call for compression |
| Model routing (easy/hard queries) | 5-10x on easy queries | Routing accuracy |

```python
# Cost-aware model routing
def route_by_complexity(query, retriever, small_llm, large_llm):
    docs = retriever.invoke(query)

    # Estimate query complexity
    complexity_prompt = f"""Rate the complexity of answering this question (1=simple, 5=complex):
    Question: {query}
    Available context: {len(docs)} documents
    Complexity (1-5):"""

    complexity = int(small_llm.invoke(complexity_prompt).content.strip())

    if complexity <= 2:
        return small_llm.invoke(format_prompt(query, docs))  # GPT-4o-mini: $0.15/1M tokens
    else:
        return large_llm.invoke(format_prompt(query, docs))  # GPT-4o: $2.50/1M tokens
```

---

## SECTION D: TRICKY INTERVIEW QUESTIONS

### Q11: "The user asks a question and your RAG system retrieves relevant documents but the LLM still generates an incorrect answer. How do you debug this?"

**Answer:** Systematic debugging approach:

1. **Inspect retrieved documents**: Are they actually relevant? Score them manually.
2. **Check chunk quality**: Is the answer split across chunks? Is relevant info diluted by noise?
3. **Examine the prompt**: Is the instruction clear? Does it tell the LLM to ONLY use provided context?
4. **Test with perfect context**: Manually provide the perfect context - does the LLM answer correctly? If yes, it's a retrieval problem. If no, it's a generation problem.
5. **Check for conflicting information**: Do retrieved docs contain contradictory statements?
6. **Verify LLM follows instructions**: Is the LLM ignoring the context and using its parametric knowledge?
7. **Temperature**: Lower temperature for factual QA (0 or 0.1)
8. **Context ordering**: Move most relevant chunk to top (lost-in-middle)
9. **Add explicit instructions**: "Quote the exact text from the context that supports your answer"

### Q12: "How do you handle queries that your RAG system cannot answer?"

**Answer:**

```python
class RAGWithFallback:
    def __init__(self, retriever, llm, confidence_threshold=0.6):
        self.retriever = retriever
        self.llm = llm
        self.threshold = confidence_threshold

    def answer(self, query):
        docs = self.retriever.invoke(query)

        # Check retrieval quality
        if not docs:
            return self.no_results_response(query)

        top_score = docs[0].metadata.get("score", 0)
        if top_score < self.threshold:
            return self.low_confidence_response(query, docs)

        # Generate answer
        response = self.generate(query, docs)

        # Verify answer isn't a refusal or hallucination
        if self.is_uncertain(response):
            return self.uncertain_response(query, response, docs)

        return response

    def no_results_response(self, query):
        return {
            "answer": "I don't have information about this topic in my knowledge base. "
                     "Please contact support for further assistance.",
            "confidence": "low",
            "suggestion": "Try rephrasing your question or ask about a related topic."
        }

    def low_confidence_response(self, query, docs):
        return {
            "answer": "I found some potentially related information, but I'm not confident "
                     "it fully answers your question. Here's what I found: ...",
            "confidence": "low",
            "related_topics": self.suggest_related_topics(query, docs)
        }

    def is_uncertain(self, response):
        uncertainty_phrases = [
            "I don't know", "I'm not sure", "I cannot find",
            "the context doesn't", "no information available"
        ]
        return any(phrase.lower() in response.lower() for phrase in uncertainty_phrases)
```

### Q13: "How would you implement RAG for a codebase (code search and generation)?"

**Answer:**

```python
# Code RAG requires special considerations:

# 1. Code-aware chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,  # Larger chunks for code to keep functions intact
    chunk_overlap=200
)

# 2. Rich metadata
import ast

def extract_code_metadata(file_path, chunk):
    """Extract function/class names, imports, docstrings."""
    metadata = {
        "file_path": file_path,
        "language": "python",
    }

    try:
        tree = ast.parse(chunk)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]

        metadata["functions"] = functions
        metadata["classes"] = classes
        metadata["imports"] = imports
    except SyntaxError:
        pass

    return metadata

# 3. Code-specific embedding models
# Models like CodeBERT, StarCoder embeddings, or code-specific E5 variants
# perform better than general-purpose embeddings for code

# 4. Hybrid search is crucial for code
# Variable names, function names are best matched with keyword (BM25)
# Semantic intent is best matched with vector search

# 5. Include file structure context
def add_file_context(chunk, repo_structure):
    """Add file path and directory context to each chunk."""
    header = f"File: {chunk.metadata['file_path']}\n"
    header += f"Directory: {os.path.dirname(chunk.metadata['file_path'])}\n"
    if chunk.metadata.get('functions'):
        header += f"Functions: {', '.join(chunk.metadata['functions'])}\n"
    header += "---\n"
    chunk.page_content = header + chunk.page_content
    return chunk
```

### Q14: "What is the difference between RAG and Long-Context LLMs? Is RAG still needed with 1M+ token context windows?"

**Answer:**

Even with million-token context windows, RAG remains essential:

1. **Cost**: Stuffing 1M tokens per query is expensive. RAG retrieves only relevant chunks (5-10K tokens). At $2.50/1M input tokens for GPT-4o, a 1M token query costs $2.50 vs ~$0.025 for 10K tokens with RAG.

2. **Latency**: Processing 1M tokens takes 30-60 seconds. RAG with 10K tokens: 2-3 seconds.

3. **Accuracy**: "Needle in a haystack" tests show models still struggle to find specific information in very long contexts. RAG provides pre-filtered, relevant context.

4. **Scale**: You might have 100M+ tokens of documents. Even 1M token context can't fit everything.

5. **Freshness**: RAG can pull from continuously updated databases. Long context requires re-uploading documents.

6. **Attribution**: RAG naturally tracks which documents contributed to the answer.

**When long context IS better:**
- Analyzing a single long document end-to-end
- Tasks requiring understanding of the full document structure
- When the entire context is relevant (summarization)
- Simple questions over a small document set

**Best approach**: Use long context + RAG together. RAG retrieves relevant chunks, use a generous context window (32K-128K tokens) to fit many high-quality chunks.

---

### Q15: "Explain how you would implement citation/attribution in a RAG system."

**Answer:**

```python
# Approach 1: Post-hoc citation extraction
citation_prompt = """Based on the provided context, answer the question and cite your sources.

Context:
{context}

For each claim in your answer, add a citation in the format [Source N] where N corresponds to the context passage number.

Question: {question}

Answer with citations:"""

# Approach 2: Structured citation with JSON output
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Citation(BaseModel):
    source_id: str = Field(description="The source document identifier")
    page: int = Field(description="The page number")
    quote: str = Field(description="The exact quote from the source")

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The answer to the question")
    citations: List[Citation] = Field(description="Citations supporting the answer")

structured_llm = llm.with_structured_output(AnswerWithCitations)

# Approach 3: Inline highlighting
# Return which exact passages in the source were used
def extract_supporting_passages(answer, retrieved_docs, llm):
    """For each sentence in the answer, find the supporting passage."""
    sentences = answer.split(". ")
    attributed = []

    for sentence in sentences:
        prompt = f"""Which of these passages best supports this claim?
        Claim: {sentence}
        Passages:
        {chr(10).join(f'[{i}]: {doc.page_content[:200]}' for i, doc in enumerate(retrieved_docs))}

        Best matching passage number (or NONE):"""

        match = llm.invoke(prompt).content.strip()
        attributed.append({
            "sentence": sentence,
            "source": match,
            "source_doc": retrieved_docs[int(match)].metadata if match != "NONE" else None
        })

    return attributed
```
