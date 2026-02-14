
# RAG Interview Guide
# PART 3: CHUNKING STRATEGIES IN DETAIL

---

## Why Chunking Matters

Chunking is arguably the most impactful decision in RAG pipeline design. It directly affects:
- **Retrieval precision**: Too large = noise dilutes relevant info; too small = missing context
- **Embedding quality**: Embedding models have optimal input sizes; too long = information compressed/lost
- **LLM generation quality**: Context quality determines answer quality
- **Cost**: More chunks = more storage, more embedding API calls, more tokens in prompts
- **Latency**: More chunks = slower retrieval and longer prompts

**The Goldilocks Problem**: Chunks must be small enough for precise retrieval but large enough to contain sufficient context for the LLM to generate accurate answers.

---

## 1. Fixed-Size Chunking

The simplest approach: split text into chunks of a fixed number of characters or tokens.

```python
from langchain.text_splitter import CharacterTextSplitter

# Character-based fixed size
splitter = CharacterTextSplitter(
    separator="\n",      # Split on newlines first
    chunk_size=1000,      # Max 1000 characters per chunk
    chunk_overlap=200,    # 200 character overlap between chunks
    length_function=len   # Use character count
)

chunks = splitter.split_documents(documents)
```

**Pros:**
- Simple to implement and understand
- Predictable chunk sizes
- Fast processing

**Cons:**
- Splits mid-sentence, mid-paragraph, mid-thought
- No awareness of semantic boundaries
- Fixed overlap may be too much or too little

**When to use**: Quick prototyping, uniform documents (e.g., logs), when simplicity is priority.

---

## 2. Recursive Character Text Splitting

The most commonly used strategy. Splits hierarchically using a list of separators, trying to keep semantically related text together.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=[
        "\n\n",    # 1st: Split on double newlines (paragraphs)
        "\n",      # 2nd: Split on single newlines
        ". ",      # 3rd: Split on sentences
        ", ",      # 4th: Split on clauses
        " ",       # 5th: Split on words
        ""         # 6th: Split on characters (last resort)
    ]
)

chunks = splitter.split_documents(documents)
```

**How it works:**
1. Try to split on the first separator ("\n\n")
2. If any resulting chunk is still > chunk_size, recursively split that chunk using the next separator ("\n")
3. Continue down the separator hierarchy until all chunks are within chunk_size
4. Add overlap from the end of the previous chunk to the beginning of the next

**Language-specific separators:**

```python
# Python code splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
)
# Uses separators: ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]

# JavaScript
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=2000,
    chunk_overlap=200
)

# Markdown
md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=1000,
    chunk_overlap=100
)
# Uses: ["\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""]
```

**Pros:**
- Respects document structure (paragraphs, sentences)
- Good balance of simplicity and quality
- Language-aware variants available
- Industry standard for most use cases

**Cons:**
- Still rule-based, not truly semantic
- Separator hierarchy may not suit all document types
- Chunk size is in characters, not semantic units

---

## 3. Token-Based Chunking

Split based on token count rather than character count. Important because:
- Embedding models have token limits (not character limits)
- LLMs are billed per token
- 1 token ~ 4 characters (English), but varies by language and model tokenizer

```python
from langchain.text_splitter import TokenTextSplitter

# Using tiktoken (OpenAI tokenizer)
splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-4/3.5 tokenizer
    chunk_size=512,                # 512 tokens
    chunk_overlap=50               # 50 token overlap
)

chunks = splitter.split_documents(documents)

# Using RecursiveCharacterTextSplitter with token counting
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4o")

def token_length(text):
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=token_length,  # Count tokens, not characters
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Using SentenceTransformers tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

def hf_token_length(text):
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,  # BGE models work best with shorter inputs
    chunk_overlap=30,
    length_function=hf_token_length
)
```

**Pros:**
- Precisely controls token usage for embedding models and LLMs
- Prevents exceeding model context limits
- Accurate cost estimation

**Cons:**
- Tokenizer-specific (OpenAI vs HuggingFace tokenizers differ)
- Can still split mid-sentence if not combined with smart separators

---

## 4. Semantic Chunking

Groups text based on semantic similarity rather than arbitrary character/token counts. Adjacent sentences/paragraphs are kept together if they are semantically similar.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Percentile-based breakpoint
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # Options: percentile, standard_deviation, interquartile, gradient
    breakpoint_threshold_amount=95  # Split where similarity drops below 95th percentile
)

chunks = semantic_splitter.split_documents(documents)

# Standard deviation based
semantic_splitter_std = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5  # Split where similarity is 1.5 std devs below mean
)

# Gradient-based (detects sharp changes in similarity)
semantic_splitter_grad = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="gradient",
    breakpoint_threshold_amount=95
)
```

**How Semantic Chunking Works:**
1. Split text into sentences
2. Embed each sentence
3. Compare consecutive sentence embeddings using cosine similarity
4. When similarity drops below threshold, insert a chunk boundary
5. Group consecutive similar sentences into chunks

```python
# Manual implementation of semantic chunking
import numpy as np
from sentence_transformers import SentenceTransformer
import re

def semantic_chunk(text, model, threshold=0.75, min_chunk_size=100):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return [text]

    # Embed all sentences
    embeddings = model.encode(sentences)

    # Calculate cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        similarities.append(sim)

    # Find breakpoints where similarity drops below threshold
    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < threshold and len(' '.join(current_chunk)) >= min_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
chunks = semantic_chunk(long_text, model, threshold=0.65)
```

**Pros:**
- Semantically coherent chunks
- Adaptive chunk sizes based on content
- Preserves topic boundaries
- Better retrieval relevance

**Cons:**
- Expensive: requires embedding every sentence (API cost + latency)
- Unpredictable chunk sizes (may be very small or very large)
- Sensitive to threshold selection
- Sentence detection quality affects results

---

## 5. Document-Specific Chunking

Leverage document structure (headings, HTML tags, markdown) for natural boundaries.

### Markdown Header Splitting:

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # Keep headers in the chunk text
)

splits = md_splitter.split_text(markdown_text)
# Each split has metadata: {'Header 1': 'Title', 'Header 2': 'Section'}

# Combine with recursive splitting for long sections
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
final_chunks = recursive_splitter.split_documents(splits)
```

### HTML Splitting:

```python
from langchain.text_splitter import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = html_splitter.split_text(html_text)
```

### Code Splitting:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Python - splits on class/function boundaries
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
)

# Supported languages: PYTHON, JS, TS, JAVA, GO, RUST, CPP, C, SCALA, SWIFT,
# MARKDOWN, LATEX, HTML, SOL, PHP, RUBY, etc.
```

### Table-Aware Chunking:

```python
# For documents with tables, extract tables separately
import camelot  # For PDF tables

# Extract tables from PDF
tables = camelot.read_pdf("document.pdf", pages="all")

for table in tables:
    df = table.df  # pandas DataFrame
    # Convert table to text format
    table_text = df.to_markdown()  # or df.to_string()
    # Create a document chunk for this table with metadata
    table_chunk = Document(
        page_content=f"Table: {table_text}",
        metadata={"source": "document.pdf", "type": "table", "page": table.page}
    )
```

**Pros:**
- Respects natural document structure
- Rich metadata from document hierarchy
- Excellent for structured documents (docs, wikis, codebases)

**Cons:**
- Format-specific (different splitter per format)
- Requires well-structured source documents
- Sections may still be too large and need sub-splitting

---

## 6. Agentic Chunking (LLM-Driven)

Uses an LLM to decide chunk boundaries based on understanding of the content.

```python
class AgenticChunker:
    def __init__(self, llm):
        self.llm = llm
        self.chunks = []

    def add_propositions(self, text):
        """Break text into propositions (atomic facts) and group into chunks."""

        # Step 1: Extract propositions
        propositions = self.extract_propositions(text)

        # Step 2: For each proposition, decide which chunk it belongs to
        for prop in propositions:
            if not self.chunks:
                self.create_new_chunk(prop)
                continue

            # Ask LLM which existing chunk this proposition belongs to
            chunk_assignment = self.assign_to_chunk(prop)

            if chunk_assignment == "NEW":
                self.create_new_chunk(prop)
            else:
                self.chunks[chunk_assignment].append(prop)

    def extract_propositions(self, text):
        prompt = f"""Decompose the following text into clear, standalone propositions.
        Each proposition should:
        - Be a single, complete fact
        - Be understandable without the other propositions
        - Include necessary context (resolve pronouns, add implicit subjects)

        Text: {text}

        Propositions (one per line):"""

        result = self.llm.invoke(prompt).content
        return [p.strip() for p in result.split("\n") if p.strip()]

    def assign_to_chunk(self, proposition):
        chunk_summaries = "\n".join(
            f"Chunk {i}: {self.summarize_chunk(chunk)}"
            for i, chunk in enumerate(self.chunks)
        )

        prompt = f"""You have the following existing chunks:
        {chunk_summaries}

        Which chunk should this proposition be added to?
        Proposition: {proposition}

        Reply with the chunk number, or 'NEW' if it doesn't fit any existing chunk.
        Answer:"""

        result = self.llm.invoke(prompt).content.strip()
        if "NEW" in result.upper():
            return "NEW"
        try:
            return int(result)
        except ValueError:
            return "NEW"

    def create_new_chunk(self, proposition):
        self.chunks.append([proposition])

    def summarize_chunk(self, chunk):
        return self.llm.invoke(
            f"Summarize these facts in one sentence: {'; '.join(chunk)}"
        ).content
```

**Proposition-based chunking** (Chen et al., 2023 - "Dense X Retrieval"):
- Break text into atomic propositions (single facts)
- Each proposition is embedded independently
- Retrieval returns specific facts, not paragraphs
- More precise but more chunks and higher cost

```python
# Proposition extraction
propositions_prompt = """Decompose the following passage into independent propositions.
Each proposition should be:
1. A single atomic fact
2. Self-contained and understandable independently
3. Concise but complete

Passage: "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity while working at the Swiss Patent Office. His famous equation E=mc^2 revolutionized physics."

Propositions:
1. Albert Einstein was born in Ulm, Germany.
2. Albert Einstein was born in 1879.
3. Albert Einstein developed the theory of relativity.
4. Albert Einstein worked at the Swiss Patent Office.
5. Albert Einstein developed the theory of relativity while working at the Swiss Patent Office.
6. Albert Einstein's famous equation is E=mc^2.
7. The equation E=mc^2 revolutionized physics."""
```

**Pros:**
- Highest quality chunk boundaries
- Adaptive to content complexity
- Can create semantically perfect groupings

**Cons:**
- Very expensive (many LLM calls)
- Very slow (not suitable for large-scale ingestion)
- LLM may make inconsistent decisions
- Difficult to reproduce deterministically

---

## 7. Overlap Strategies

Overlap ensures that information at chunk boundaries is not lost.

```python
# Standard overlap: repeat N characters/tokens from end of chunk i at start of chunk i+1
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200  # 20% overlap
)

# Example:
# Chunk 1: "...the policy states that employees are entitled to 15 days of"
# Chunk 2: "employees are entitled to 15 days of annual leave, with additional..."
# The overlap ensures "15 days of annual leave" appears in both chunks
```

**Overlap guidelines:**
- **10-20% overlap** is typical (e.g., 200 chars for 1000-char chunks)
- **Too little overlap**: Risk losing context at boundaries
- **Too much overlap**: Redundancy, increased storage, retrieval may return near-duplicate chunks
- **Sentence-boundary overlap**: Instead of fixed character overlap, overlap at sentence boundaries

```python
# Sentence-boundary overlap (custom implementation)
import re

def chunk_with_sentence_overlap(text, target_size=1000, overlap_sentences=2):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        if current_size + len(sentence) > target_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Keep last N sentences for overlap
            current_chunk = current_chunk[-overlap_sentences:]
            current_size = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_size += len(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
```

---

## 8. Contextual Chunk Headers (Anthropic's Approach)

Add document-level context to each chunk so it can be understood independently.

**Anthropic's "Contextual Retrieval" technique (September 2024):**

```python
# Before embedding, prepend contextual header to each chunk
def add_contextual_header(chunk, full_document, llm):
    prompt = f"""<document>
    {full_document}
    </document>

    Here is the chunk we want to situate within the whole document:
    <chunk>
    {chunk.page_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document
    for the purposes of improving search retrieval of the chunk. Answer only with the
    succinct context and nothing else."""

    context = llm.invoke(prompt).content

    # Prepend context to chunk
    chunk.page_content = f"{context}\n\n{chunk.page_content}"
    return chunk

# Apply to all chunks
contextualized_chunks = [
    add_contextual_header(chunk, full_doc, llm)
    for chunk in chunks
]
```

**Example:**
- Original chunk: "The company's revenue grew 15% year over year."
- With context: "This chunk is from Apple Inc.'s Q3 2024 earnings report, specifically the financial highlights section. The company's revenue grew 15% year over year."

---

## 9. Chunking Strategy Comparison

| Strategy | Quality | Speed | Cost | Best For |
|----------|---------|-------|------|----------|
| Fixed-size | Low | Very Fast | Very Low | Prototyping, logs |
| Recursive Character | Medium-High | Fast | Low | General purpose (DEFAULT CHOICE) |
| Token-based | Medium | Fast | Low | When token limits matter |
| Semantic | High | Slow | Medium-High | Topic-diverse docs |
| Document-specific | High | Fast | Low | Structured docs (MD, HTML, code) |
| Agentic/Proposition | Very High | Very Slow | Very High | High-value, small document sets |

---

## 10. Interview Questions on Chunking

### Q: "How do you determine the optimal chunk size?"

**Answer**: There is no universal optimal chunk size. It depends on:
1. **Embedding model context**: Match chunk size to embedding model's sweet spot (e.g., 256-512 tokens for most models)
2. **Query type**: Short factual queries -> smaller chunks (256 tokens). Complex analytical queries -> larger chunks (1024 tokens)
3. **Document type**: Dense technical docs -> smaller chunks. Narrative text -> larger chunks
4. **Empirical testing**: The most reliable approach. Test multiple sizes (256, 512, 768, 1024 tokens) on a representative query set and measure retrieval metrics (precision, recall, MRR)
5. **Rule of thumb**: Start with 512 tokens, 10-20% overlap for most cases

### Q: "What happens when a key piece of information spans two chunks?"

**Answer**: This is the chunk boundary problem. Solutions:
1. **Overlap**: Ensures repeated text at boundaries (most common)
2. **Parent document retriever**: Retrieve small chunks, return larger parent
3. **Sliding window**: Chunks are windows that slide by a step size < chunk size
4. **Sentence-aware splitting**: Never split mid-sentence
5. **Contextual headers**: Add document context to each chunk
6. **Multi-vector retrieval**: Multiple small chunks map to same larger context
7. **Post-retrieval merging**: If retrieved chunks are adjacent, merge them before sending to LLM

### Q: "How do you chunk tables and structured data?"

**Answer**:
1. **Extract tables separately** using table extraction tools (Camelot, Tabula, Unstructured)
2. **Keep tables as single chunks** - never split a table across chunks
3. **Add table description/caption** as chunk header for better retrieval
4. **Convert to natural language**: Use LLM to convert table to text description for embedding
5. **Alternative**: Store tables in SQL, use text-to-SQL for structured queries
6. **Serialize format**: Markdown tables embed better than HTML tables or raw CSVs

### Q: "Should chunking strategy differ for different types of documents in the same RAG system?"

**Answer**: Yes, absolutely. A production RAG system should use different chunking strategies per document type:
- **API documentation**: Split by endpoints/methods, use code-aware splitting
- **Legal contracts**: Split by clauses/sections, preserve section numbering
- **Research papers**: Split by sections (abstract, methods, results), keep figures/tables intact
- **Chat logs**: Split by conversation turns, keep thread context
- **Code files**: Use language-aware splitting, keep functions/classes intact
- **FAQ pages**: Each Q&A pair as a chunk

Implement a chunking router that detects document type and applies the appropriate strategy.
