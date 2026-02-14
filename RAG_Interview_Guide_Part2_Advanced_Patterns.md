---
title: "Part 2 - Advanced Patterns"
layout: default
parent: "RAG Systems"
nav_order: 3
---


# RAG Interview Guide
# PART 2: ADVANCED RAG PATTERNS

---

## 1. RAG Evolution: Naive RAG vs Advanced RAG vs Modular RAG

### 1.1 Naive RAG (Basic RAG)

The simplest implementation following the basic Retrieve-Read pattern.

**Pipeline**: Query -> Retrieve top-k chunks -> Stuff into prompt -> Generate

**Limitations of Naive RAG:**
- Poor retrieval quality: semantic gap between query and relevant chunks
- Redundant/noisy retrieved chunks
- No query understanding or transformation
- Hallucination when context is insufficient
- Cannot handle complex multi-hop questions
- Lost in the middle problem with many retrieved chunks
- No verification of generated answer against sources

### 1.2 Advanced RAG

Adds pre-retrieval and post-retrieval optimizations around the Naive RAG core.

**Pre-retrieval optimizations:**
- Query rewriting, expansion, transformation
- Query routing to appropriate indices
- Query decomposition for complex questions

**Retrieval optimizations:**
- Hybrid search (dense + sparse)
- Metadata filtering
- Multi-index search

**Post-retrieval optimizations:**
- Re-ranking retrieved documents
- Contextual compression
- Diversity optimization (MMR)
- Filtering by relevance threshold

**Pipeline:**
```
Query -> Query Transformation -> Retrieval (Hybrid) -> Re-ranking -> Compression -> Generation -> Validation
```

### 1.3 Modular RAG

A flexible, composable architecture where RAG components are modules that can be mixed, matched, and orchestrated.

**Key Modules:**
- **Search Module**: Can include vector search, keyword search, SQL queries, knowledge graph queries, web search
- **Retrieval Module**: Multiple retrieval strategies selectable based on query type
- **Memory Module**: Conversational history, user preferences
- **Routing Module**: Directs queries to appropriate processing pipelines
- **Predict Module**: Query rewriting, HyDE
- **Fusion Module**: Combines results from multiple retrieval strategies
- **Evaluation Module**: Assesses retrieval quality, triggers retry if poor
- **Generation Module**: Multiple generation strategies (direct answer, summarization, comparison)

**Orchestration Patterns:**
1. **Sequential**: Module A -> Module B -> Module C (most common)
2. **Conditional**: Router decides which modules to activate based on query
3. **Branching**: Query processed by multiple modules in parallel, results merged
4. **Loop/Iterative**: Generate -> Evaluate -> If insufficient, re-retrieve with refined query

```python
# Conceptual Modular RAG Architecture
class ModularRAG:
    def __init__(self):
        self.router = QueryRouter()
        self.retrievers = {
            "vector": VectorRetriever(),
            "keyword": BM25Retriever(),
            "sql": SQLRetriever(),
            "kg": KnowledgeGraphRetriever(),
        }
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator()
        self.evaluator = ResponseEvaluator()

    def process(self, query: str):
        # Route query to appropriate retriever(s)
        route = self.router.route(query)

        # Retrieve from selected sources
        all_docs = []
        for retriever_name in route.retrievers:
            docs = self.retrievers[retriever_name].retrieve(query)
            all_docs.extend(docs)

        # Re-rank combined results
        ranked_docs = self.reranker.rerank(query, all_docs)

        # Generate response
        response = self.generator.generate(query, ranked_docs[:5])

        # Evaluate and potentially retry
        eval_result = self.evaluator.evaluate(query, response, ranked_docs)
        if eval_result.score < 0.7:
            # Retry with query transformation
            new_query = self.transform_query(query, response)
            return self.process(new_query)

        return response
```

---

## 2. Query Transformation Techniques

### 2.1 HyDE (Hypothetical Document Embeddings)

**Paper**: Gao et al., 2022 - "Precise Zero-Shot Dense Retrieval without Relevance Labels"

**Concept**: Instead of embedding the query directly, ask the LLM to generate a hypothetical answer, then embed THAT answer to search for similar real documents. The hypothesis is that a generated answer is more semantically similar to actual answer documents than the original short query.

**Why it works**: Short queries like "What causes rain?" have very different embedding representations than a passage explaining rain formation. A generated hypothetical answer about rain formation will be much closer in embedding space to the actual documents.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Step 1: Generate hypothetical document
hyde_prompt = ChatPromptTemplate.from_template(
    """Please write a detailed passage that would answer the following question.
    Do not say "I don't know". Generate a plausible answer even if you're not sure.

    Question: {question}

    Passage:"""
)

# Step 2: HyDE chain
hyde_chain = hyde_prompt | llm | StrOutputParser()

# Generate hypothetical answer
hypothetical_doc = hyde_chain.invoke({"question": "What is RLHF in AI?"})

# Step 3: Embed the hypothetical document and search
embeddings = OpenAIEmbeddings()
hyde_embedding = embeddings.embed_query(hypothetical_doc)

# Search vector store with hypothetical document embedding
results = vectorstore.similarity_search_by_vector(hyde_embedding, k=5)

# Full HyDE Retriever using LangChain
from langchain.retrievers import HyDERetriever  # Conceptual

class HyDERetriever:
    def __init__(self, llm, embeddings, vectorstore, k=5):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.k = k

    def retrieve(self, query: str):
        # Generate hypothetical document
        hyde_prompt = f"Write a passage that answers: {query}"
        hypothetical = self.llm.invoke(hyde_prompt).content

        # Embed hypothetical document
        hyde_vector = self.embeddings.embed_query(hypothetical)

        # Retrieve using hypothetical embedding
        return self.vectorstore.similarity_search_by_vector(hyde_vector, k=self.k)
```

**When to use HyDE:**
- Short or ambiguous queries
- When query-document semantic gap is large
- When you have a capable generation model

**When NOT to use HyDE:**
- Factual/specific queries (product IDs, names) - hallucinated answers may mislead
- When latency is critical (adds an LLM call)
- When the LLM might generate completely wrong hypothetical answers

---

### 2.2 Multi-Query Retrieval

Generate multiple variations of the query to capture different aspects and retrieve documents for each.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Automatic multi-query generation
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# This will:
# 1. Generate 3 alternative queries
# 2. Retrieve docs for each query
# 3. Deduplicate and return union of results
docs = multi_query_retriever.invoke("What are the effects of climate change on agriculture?")

# Custom multi-query prompt
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 3
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations of distance-based
    similarity search.

    Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
    prompt=custom_prompt
)
```

**Example**: Original query: "What is RAG?"
Generated alternatives:
1. "How does Retrieval Augmented Generation work?"
2. "What are the components of a RAG system?"
3. "Why is RAG used with large language models?"

---

### 2.3 RAG Fusion

Similar to multi-query but uses Reciprocal Rank Fusion (RRF) to combine results.

```python
from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    """Combine multiple ranked lists using RRF."""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [loads(doc) for doc, score in sorted_docs]

# Generate multiple queries
queries = generate_multi_queries(original_query)

# Retrieve for each query
all_results = []
for query in queries:
    results = vectorstore.similarity_search(query, k=10)
    all_results.append(results)

# Fuse results
fused_docs = reciprocal_rank_fusion(all_results)
```

**RRF Formula**: `RRF_score(d) = sum(1 / (k + rank_i(d)))` where k is typically 60.

---

### 2.4 Step-Back Prompting

Ask the LLM to generate a more abstract/general version of the query, retrieve for both the original and step-back query.

**Paper**: Zheng et al., 2023 - "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"

```python
step_back_prompt = ChatPromptTemplate.from_template(
    """You are an expert at world knowledge. Your task is to step back and
    paraphrase a question to a more generic step-back question, which is
    easier to answer.

    Original question: {question}
    Step-back question:"""
)

# Example:
# Original: "What happens to the pressure of an ideal gas if temperature increases by a factor of 2 and volume by a factor of 8?"
# Step-back: "What is the ideal gas law and how do pressure, temperature, and volume relate?"

# Retrieve for both queries and combine context
original_docs = retriever.invoke(original_question)
stepback_docs = retriever.invoke(stepback_question)
all_context = original_docs + stepback_docs
```

---

### 2.5 Query Decomposition

Break complex questions into sub-questions, answer each, then synthesize.

```python
decomposition_prompt = ChatPromptTemplate.from_template(
    """Break down the following complex question into 2-4 simpler sub-questions
    that, when answered together, would answer the original question.

    Question: {question}

    Sub-questions (one per line):"""
)

# Example:
# Complex: "Compare the economic impacts of renewable energy adoption in Germany vs China over the last decade"
# Sub-questions:
# 1. What have been the economic impacts of renewable energy in Germany in the last decade?
# 2. What have been the economic impacts of renewable energy in China in the last decade?
# 3. How do Germany and China's renewable energy policies differ?

class DecompositionRAG:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str):
        # Decompose
        sub_questions = self.decompose(question)

        # Answer each sub-question
        sub_answers = []
        for sub_q in sub_questions:
            docs = self.retriever.invoke(sub_q)
            context = format_docs(docs)
            answer = self.llm.invoke(
                f"Context: {context}\n\nQuestion: {sub_q}\n\nAnswer:"
            )
            sub_answers.append({"question": sub_q, "answer": answer.content})

        # Synthesize final answer
        synthesis_prompt = f"""Based on the following sub-questions and their answers,
        provide a comprehensive answer to: {question}

        {chr(10).join(f"Q: {sa['question']}\nA: {sa['answer']}" for sa in sub_answers)}

        Comprehensive answer:"""

        return self.llm.invoke(synthesis_prompt).content
```

---

## 3. Re-ranking Techniques

### 3.1 Why Re-ranking?

Initial retrieval (bi-encoder) is fast but approximate. Re-ranking with cross-encoders is more accurate but slower. This two-stage approach gives both speed and accuracy.

**Bi-encoder vs Cross-encoder:**
- **Bi-encoder**: Encodes query and document separately, compares with cosine similarity. Fast (precompute doc embeddings), but less accurate.
- **Cross-encoder**: Takes query-document PAIR as input, outputs relevance score. Much more accurate but slow (can't precompute).

### 3.2 Cohere Rerank

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Initial retrieval: get more candidates than needed
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Re-rank to top 5
compressor = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

results = reranking_retriever.invoke("What is RLHF?")
```

### 3.3 Cross-Encoder Re-ranking

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_with_cross_encoder(query, documents, top_k=5):
    # Create query-document pairs
    pairs = [[query, doc.page_content] for doc in documents]

    # Score all pairs
    scores = cross_encoder.predict(pairs)

    # Sort by score and return top_k
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]

# In the pipeline
initial_docs = vectorstore.similarity_search(query, k=20)
reranked_docs = rerank_with_cross_encoder(query, initial_docs, top_k=5)
```

### 3.4 ColBERT-style Re-ranking

Late interaction model - represents query and document as sets of token embeddings, computes MaxSim.

```python
# Using RAGatouille for ColBERT
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index documents
RAG.index(
    collection=[doc.page_content for doc in documents],
    index_name="my_index"
)

# Search (ColBERT handles both retrieval and ranking)
results = RAG.search(query="What is RAG?", k=5)
```

### 3.5 LLM-based Re-ranking

Use the LLM itself to score relevance.

```python
def llm_rerank(query, documents, llm, top_k=5):
    scored_docs = []
    for doc in documents:
        prompt = f"""On a scale of 1-10, rate how relevant the following passage is
        to answering the question. Return ONLY a number.

        Question: {query}
        Passage: {doc.page_content}

        Relevance score:"""

        score = float(llm.invoke(prompt).content.strip())
        scored_docs.append((doc, score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]
```

**Interview Question: "What is the latency impact of re-ranking and how do you optimize it?"**

**Answer**: Cross-encoder re-ranking adds 50-200ms for 20 documents. Optimization strategies:
1. Limit candidate pool (re-rank top 20-50, not all)
2. Use distilled/smaller cross-encoders (MiniLM-L-6 vs L-12)
3. Batch processing with GPU
4. Cache re-ranking scores for frequent queries
5. Use ColBERT (late interaction) as a middle ground - faster than cross-encoders, more accurate than bi-encoders
6. Async re-ranking: start generating with top results while re-ranking continues

---

## 4. Contextual Compression

Remove irrelevant portions of retrieved documents to reduce noise and context length.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# LLM-based extraction
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# This will:
# 1. Retrieve documents normally
# 2. For each document, ask LLM to extract only the relevant portions
# 3. Return compressed documents

# Cheaper alternative: EmbeddingsFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)

# Pipeline compressor (chain multiple compressors)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter

redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevant_filter]
)
```

---

## 5. Parent Document Retriever

Embed small chunks for precise retrieval, but return the larger parent chunk for more context to the LLM.

**Problem it solves**: Small chunks are better for precise retrieval (less noise in embeddings) but the LLM needs broader context to generate good answers.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Small chunks for retrieval (embedding)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# Larger chunks returned to LLM
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Storage for parent documents
store = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents
parent_retriever.add_documents(documents)

# Retrieval:
# 1. Query is matched against small child chunks
# 2. Parent chunks of matching children are returned
results = parent_retriever.invoke("What is RAG?")
# Returns the larger parent chunks that contain the matching small chunks
```

**Variants:**
- **Full document retriever**: Child chunks for search, return entire original document
- **Multi-level**: Three levels - sentence -> paragraph -> section

---

## 6. Self-RAG (Self-Reflective RAG)

**Paper**: Asai et al., 2023 - "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"

The LLM decides when to retrieve, evaluates retrieval quality, and critiques its own output.

**Key Reflection Tokens:**
- **[Retrieve]**: Should I retrieve? (yes/no/continue)
- **[IsRel]**: Is the retrieved passage relevant? (relevant/irrelevant)
- **[IsSup]**: Is the response supported by the passage? (fully/partially/no support)
- **[IsUse]**: Is the response useful to the user? (5/4/3/2/1)

```python
class SelfRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def process(self, query: str):
        # Step 1: Decide if retrieval is needed
        need_retrieval = self.should_retrieve(query)

        if not need_retrieval:
            # Generate without retrieval
            response = self.llm.invoke(f"Answer: {query}")
            return self.critique_response(query, response, None)

        # Step 2: Retrieve
        docs = self.retriever.invoke(query)

        # Step 3: Check relevance of each document
        relevant_docs = [doc for doc in docs if self.is_relevant(query, doc)]

        if not relevant_docs:
            # No relevant docs found - generate without context or re-query
            return self.handle_no_relevant_docs(query)

        # Step 4: Generate with relevant context
        context = format_docs(relevant_docs)
        response = self.llm.invoke(
            f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        )

        # Step 5: Check if response is supported by context
        support_level = self.check_support(response, relevant_docs)

        if support_level == "no_support":
            # Regenerate or flag as uncertain
            return self.regenerate(query, relevant_docs)

        # Step 6: Check usefulness
        usefulness = self.check_usefulness(query, response)

        return {
            "answer": response,
            "support_level": support_level,
            "usefulness": usefulness,
            "sources": relevant_docs
        }

    def should_retrieve(self, query: str) -> bool:
        prompt = f"""Determine if external information is needed to answer this question.
        Reply YES or NO.
        Question: {query}"""
        result = self.llm.invoke(prompt).content.strip().upper()
        return result == "YES"

    def is_relevant(self, query: str, doc) -> bool:
        prompt = f"""Is the following passage relevant to answering the question?
        Reply RELEVANT or IRRELEVANT.
        Question: {query}
        Passage: {doc.page_content}"""
        result = self.llm.invoke(prompt).content.strip().upper()
        return "RELEVANT" in result

    def check_support(self, response, docs) -> str:
        context = format_docs(docs)
        prompt = f"""Is the following response fully supported by the provided context?
        Reply: FULLY_SUPPORTED, PARTIALLY_SUPPORTED, or NO_SUPPORT
        Context: {context}
        Response: {response}"""
        return self.llm.invoke(prompt).content.strip()
```

**Interview Question: "What are the tradeoffs of Self-RAG vs standard RAG?"**

**Answer**:
- **Pros**: Reduces unnecessary retrieval (faster for simple queries), filters irrelevant docs, catches unsupported claims, adaptive behavior
- **Cons**: Multiple LLM calls per query (3-5x more expensive), higher latency, more complex to implement and debug, reflection quality depends on LLM capability

---

## 7. CRAG (Corrective RAG)

**Paper**: Yan et al., 2024 - "Corrective Retrieval Augmented Generation"

Evaluates retrieval quality and takes corrective action when retrieval is poor.

**Three Actions Based on Retrieval Quality:**
1. **Correct**: Retrieved docs are relevant -> use them as-is
2. **Incorrect**: Retrieved docs are irrelevant -> fall back to web search
3. **Ambiguous**: Partially relevant -> combine retrieved docs with web search

```python
from langchain_community.tools import TavilySearchResults

class CorrectiveRAG:
    def __init__(self, llm, retriever, web_search_tool):
        self.llm = llm
        self.retriever = retriever
        self.web_search = web_search_tool

    def process(self, query: str):
        # Step 1: Retrieve from vector store
        docs = self.retriever.invoke(query)

        # Step 2: Grade each document for relevance
        graded_docs = self.grade_documents(query, docs)
        relevant_docs = [d for d, grade in graded_docs if grade == "relevant"]
        irrelevant_docs = [d for d, grade in graded_docs if grade == "irrelevant"]

        # Step 3: Determine action based on relevance ratio
        relevance_ratio = len(relevant_docs) / len(docs) if docs else 0

        if relevance_ratio > 0.7:
            # CORRECT: Use retrieved documents
            action = "correct"
            final_context = relevant_docs
        elif relevance_ratio < 0.3:
            # INCORRECT: Fall back to web search
            action = "incorrect"
            web_results = self.web_search.invoke(query)
            final_context = self.convert_web_results(web_results)
        else:
            # AMBIGUOUS: Combine both
            action = "ambiguous"
            web_results = self.web_search.invoke(query)
            web_docs = self.convert_web_results(web_results)
            final_context = relevant_docs + web_docs

        # Step 4: Knowledge refinement - strip irrelevant sentences
        refined_context = self.refine_knowledge(query, final_context)

        # Step 5: Generate
        response = self.generate(query, refined_context)

        return {
            "answer": response,
            "action_taken": action,
            "sources": final_context
        }

    def grade_documents(self, query, docs):
        graded = []
        for doc in docs:
            prompt = f"""Grade whether this document is relevant to the question.
            Reply 'relevant' or 'irrelevant'.

            Question: {query}
            Document: {doc.page_content}"""

            grade = self.llm.invoke(prompt).content.strip().lower()
            graded.append((doc, grade))
        return graded

    def refine_knowledge(self, query, docs):
        """Extract only query-relevant sentences from documents."""
        refined = []
        for doc in docs:
            prompt = f"""Extract only the sentences from the following passage that are
            relevant to answering the question. Return the relevant sentences only.

            Question: {query}
            Passage: {doc.page_content}"""

            relevant_text = self.llm.invoke(prompt).content
            doc.page_content = relevant_text
            refined.append(doc)
        return refined
```

**CRAG with LangGraph** (state machine implementation):

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    documents: List
    generation: str
    web_search_needed: bool

def retrieve(state):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

def grade_documents(state):
    relevant_docs = []
    web_search_needed = False
    for doc in state["documents"]:
        grade = grade_single_doc(state["question"], doc)
        if grade == "relevant":
            relevant_docs.append(doc)

    if len(relevant_docs) < 2:
        web_search_needed = True

    return {"documents": relevant_docs, "web_search_needed": web_search_needed}

def decide_to_generate(state):
    if state["web_search_needed"]:
        return "web_search"
    return "generate"

def web_search(state):
    web_results = tavily_search.invoke(state["question"])
    web_docs = [Document(page_content=r["content"]) for r in web_results]
    return {"documents": state["documents"] + web_docs}

def generate(state):
    response = rag_chain.invoke({
        "context": format_docs(state["documents"]),
        "question": state["question"]
    })
    return {"generation": response}

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"web_search": "web_search", "generate": "generate"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
```

---

## 8. Agentic RAG

The LLM acts as an agent that decides WHEN and HOW to retrieve, which tools to use, and can perform multi-step reasoning.

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import TavilySearchResults

# Create tools from retrievers
vector_search_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    name="search_internal_docs",
    description="Search internal company documentation. Use this for questions about company policies, products, and procedures."
)

web_search_tool = TavilySearchResults(max_results=3)

sql_tool = create_sql_query_tool(db)  # For structured data queries

tools = [vector_search_tool, web_search_tool, sql_tool]

# Agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to various tools.
    For each question:
    1. Think about which tool(s) to use
    2. Retrieve relevant information
    3. If the information is insufficient, try a different tool or rephrase your search
    4. Synthesize a comprehensive answer with citations
    Always cite your sources."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create and run agent
agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

response = agent_executor.invoke({"input": "What is our company's vacation policy and how does it compare to industry standards?"})
# Agent will:
# 1. Search internal docs for vacation policy
# 2. Search web for industry standards
# 3. Compare and synthesize
```

**Agentic RAG with LlamaIndex:**

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Create query engines for different document sets
hr_engine = hr_index.as_query_engine(similarity_top_k=3)
engineering_engine = eng_index.as_query_engine(similarity_top_k=3)
finance_engine = fin_index.as_query_engine(similarity_top_k=3)

# Wrap as tools
tools = [
    QueryEngineTool(
        query_engine=hr_engine,
        metadata=ToolMetadata(
            name="hr_docs",
            description="HR policies, benefits, vacation, hiring procedures"
        )
    ),
    QueryEngineTool(
        query_engine=engineering_engine,
        metadata=ToolMetadata(
            name="engineering_docs",
            description="Technical documentation, architecture, APIs, coding standards"
        )
    ),
    QueryEngineTool(
        query_engine=finance_engine,
        metadata=ToolMetadata(
            name="finance_docs",
            description="Financial reports, budgets, expense policies"
        )
    ),
]

# Create ReAct agent
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
response = agent.chat("What is the engineering team's budget for cloud infrastructure?")
# Agent reasons: "This involves both engineering and finance docs"
# -> Queries engineering_docs for cloud infrastructure details
# -> Queries finance_docs for budget information
# -> Synthesizes answer
```

---

## 9. Graph RAG

**Paper**: Microsoft, 2024 - "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"

Uses knowledge graphs to capture relationships between entities, enabling multi-hop reasoning and global understanding that vector search alone cannot provide.

**Two modes:**
- **Local Search**: For specific questions about particular entities - uses entity neighborhoods in the graph
- **Global Search**: For broad/thematic questions - uses community summaries of the graph

```python
# Microsoft's GraphRAG
from graphrag.query.structured_search.local_search import LocalSearch
from graphrag.query.structured_search.global_search import GlobalSearch

# GraphRAG Indexing Pipeline:
# 1. Documents -> Text chunks
# 2. Text chunks -> Entity & Relationship extraction (LLM)
# 3. Entities & Relationships -> Knowledge Graph
# 4. Knowledge Graph -> Community detection (Leiden algorithm)
# 5. Communities -> Community summaries (LLM)

# Local Search Example
local_search = LocalSearch(
    llm=llm,
    context_builder=local_context_builder,
    response_type="multiple paragraphs",
)
result = await local_search.asearch("What is the relationship between Company A and Product X?")

# Global Search Example
global_search = GlobalSearch(
    llm=llm,
    context_builder=global_context_builder,
    response_type="multiple paragraphs",
    map_system_prompt=MAP_SYSTEM_PROMPT,
    reduce_system_prompt=REDUCE_SYSTEM_PROMPT,
)
result = await global_search.asearch("What are the major themes in the dataset?")
```

**Building a Knowledge Graph for RAG with Neo4j:**

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Extract entities and relationships from documents
graph_transformer = LLMGraphTransformer(llm=llm)
graph_documents = graph_transformer.convert_to_graph_documents(documents)

# Store in Neo4j
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
graph.add_graph_documents(graph_documents)

# Query using natural language -> Cypher
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)
result = chain.invoke({"query": "Who are the founders of companies that use RAG technology?"})
```

**Interview Question: "When would you use Graph RAG over standard vector-based RAG?"**

**Answer**:
- **Multi-hop questions**: "What companies were founded by people who studied at Stanford?" - requires traversing relationships
- **Global/thematic queries**: "What are the main themes across all documents?" - vector search returns individual chunks, not global patterns
- **Entity-centric queries**: When questions revolve around specific entities and their connections
- **Structured data**: When documents contain many interconnected entities (people, orgs, events, locations)
- **Reasoning over relationships**: "How is person A connected to organization B?" - graph traversal is natural
- **Trade-off**: Graph RAG is more expensive to build (entity extraction uses many LLM calls) but excels for relational queries

---

## 10. Query Routing

Direct queries to the most appropriate retrieval pipeline.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM-based routing
route_prompt = ChatPromptTemplate.from_template(
    """Given the user question, classify it into one of these categories:
    - 'vectorstore': General knowledge questions answerable from documents
    - 'sql_database': Questions requiring structured data/statistics
    - 'web_search': Questions about current events or real-time information
    - 'knowledge_graph': Questions about relationships between entities

    Question: {question}
    Category:"""
)

router_chain = route_prompt | llm | StrOutputParser()

def route_query(question):
    route = router_chain.invoke({"question": question}).strip().lower()

    if "vectorstore" in route:
        return vectorstore_retriever.invoke(question)
    elif "sql" in route:
        return sql_chain.invoke(question)
    elif "web" in route:
        return web_search.invoke(question)
    elif "knowledge_graph" in route:
        return kg_chain.invoke(question)
    else:
        return vectorstore_retriever.invoke(question)  # Default

# Semantic routing (embedding-based, no LLM call needed)
from langchain.utils.math import cosine_similarity
import numpy as np

# Define route descriptions
routes = {
    "vectorstore": "Questions about company documentation, policies, and procedures",
    "sql_database": "Questions about numbers, statistics, counts, revenue, metrics",
    "web_search": "Questions about current events, news, real-time information",
}

route_embeddings = {
    name: embeddings.embed_query(desc) for name, desc in routes.items()
}

def semantic_route(question):
    query_embedding = embeddings.embed_query(question)
    similarities = {
        name: cosine_similarity([query_embedding], [emb])[0][0]
        for name, emb in route_embeddings.items()
    }
    return max(similarities, key=similarities.get)
```
