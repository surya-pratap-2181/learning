---
title: "Common System Design Questions"
layout: default
parent: "System Design & Architecture"
nav_order: 2
---

SECTION 2: COMMON AI SYSTEM DESIGN INTERVIEW QUESTIONS WITH ANSWERS
AI System Design Interview Guide for AI/ML Engineers (2025-2026)
## 

2.1 DESIGN A CHATBOT SYSTEM (e.g., Customer Support Bot)

REQUIREMENTS GATHERING:
- Users: 10M DAU, 100M messages/day
- Latency: first token < 500ms, streaming responses
- Features: multi-turn conversation, context window management, tool use
- Safety: content filtering, PII handling
- Cost: $0.01 per conversation average

HIGH-LEVEL ARCHITECTURE:

```
[Web/Mobile Client]
        |
   [WebSocket / SSE Connection]
        |
[API Gateway (Kong/Nginx)]
   |-- Auth, Rate Limiting, Request Validation
        |
[Chat Orchestrator Service]
   |-- Session Management
   |-- Context Window Management
   |-- Tool/Function Calling Router
   |-- Guardrails (input/output filtering)
        |
   +----+----+----+
   |         |         |
[LLM Service]  [RAG Service]  [Tool Services]
   |              |              |-- Calendar API
[vLLM Cluster]   |              |-- CRM Lookup
               [Vector DB]      |-- Order Status
               [Embedding Svc]  |-- Escalate to Human
        |
[Conversation Store (DynamoDB/Redis)]
   |-- Chat History
   |-- Session State
   |-- User Preferences
        |
[Analytics Pipeline (Kafka -> Spark -> Data Warehouse)]
   |-- Conversation Logs
   |-- Quality Metrics
   |-- Cost Tracking
```

DETAILED COMPONENT DESIGN:

1. CONTEXT WINDOW MANAGEMENT:
   Problem: Models have fixed context windows (4K-128K tokens)
   Solution: Smart truncation strategy

   ```python
   class ContextManager:
       def __init__(self, max_tokens=8000):
           self.max_tokens = max_tokens
           self.system_prompt_tokens = 500  # reserved
           self.rag_context_tokens = 2000   # reserved
           self.available_for_history = max_tokens - 500 - 2000 - 1000  # 1000 for response

       def build_context(self, system_prompt, chat_history, rag_context, user_message):
           messages = [{"role": "system", "content": system_prompt}]

           # Add RAG context if available
           if rag_context:
               messages.append({"role": "system", "content": f"Context:\n{rag_context}"})

           # Add chat history (most recent first, within token budget)
           history_tokens = 0
           trimmed_history = []
           for msg in reversed(chat_history):
               msg_tokens = count_tokens(msg["content"])
               if history_tokens + msg_tokens > self.available_for_history:
                   break
               trimmed_history.insert(0, msg)
               history_tokens += msg_tokens

           messages.extend(trimmed_history)
           messages.append({"role": "user", "content": user_message})
           return messages
   ```

2. CONVERSATION MEMORY STRATEGIES:
   a) Full History (short conversations): Keep all messages
   b) Sliding Window: Keep last N messages
   c) Summary Memory: Periodically summarize old messages
   d) Entity Memory: Extract and track entities mentioned
   e) Hybrid: Summary of old + full recent messages

3. STREAMING ARCHITECTURE:
   ```
   Client <--SSE-- API Gateway <--SSE-- Orchestrator <--streaming-- vLLM

   Server-Sent Events (SSE):
   data: {"token": "Hello"}
   data: {"token": " how"}
   data: {"token": " can"}
   data: {"token": " I"}
   data: {"token": " help?"}
   data: [DONE]
   ```

4. SCALING DECISIONS:
   - Stateless orchestrator: horizontal scale with K8s
   - Session state in Redis (TTL = 30 min)
   - LLM cluster: auto-scale based on queue depth
   - Database: DynamoDB for chat history (partition key = user_id, sort key = timestamp)

INTERVIEW TALKING POINTS:
- Explain trade-offs: cost vs latency vs quality
- Discuss fallback: GPT-4 -> GPT-3.5 -> cached response
- Mention guardrails: Llama Guard, NeMo Guardrails, custom classifiers
- Address multi-tenancy: different system prompts per customer


## 2.2 DESIGN A RAG PIPELINE AT SCALE


REQUIREMENTS:
- Document corpus: 10M documents, 100GB text
- Query volume: 1M queries/day
- Latency: < 2 seconds end-to-end
- Accuracy: 90%+ relevance for top-5 retrieved docs
- Updates: new documents indexed within 5 minutes

HIGH-LEVEL ARCHITECTURE:

```
INGESTION PIPELINE:
[Document Sources] --> [Document Processor] --> [Chunking Service]
   |-- S3 uploads        |-- PDF parser         |-- Recursive text splitter
   |-- API feeds          |-- HTML cleaner       |-- Semantic chunking
   |-- Database CDC       |-- OCR (if needed)    |-- Overlapping windows
   |-- Web crawlers       |-- Metadata extract   |
                                                  v
                          [Embedding Service] --> [Vector Database]
                             |-- Batch embed      |-- Pinecone/Weaviate/Qdrant
                             |-- GPU cluster      |-- Metadata filtering
                                                  |-- HNSW index

QUERY PIPELINE:
[User Query]
      |
[Query Preprocessor]
   |-- Query expansion (generate related queries)
   |-- Query classification (simple vs complex)
   |-- Intent detection
      |
[Retrieval Layer]
   |-- Hybrid Search:
   |     |-- Dense retrieval (vector similarity)
   |     |-- Sparse retrieval (BM25/keyword)
   |     |-- Weighted combination (RRF - Reciprocal Rank Fusion)
   |-- Metadata filtering (date, source, category)
   |-- Multi-index search (different collections for different doc types)
      |
[Reranking Layer]
   |-- Cross-encoder reranker (Cohere Rerank, BGE Reranker)
   |-- Deduplication
   |-- Diversity enforcement
      |
[Context Builder]
   |-- Select top-K chunks
   |-- Add source metadata
   |-- Build prompt with context
      |
[LLM Generation]
   |-- Generate answer with citations
   |-- Streaming response
      |
[Post-Processing]
   |-- Citation verification
   |-- Hallucination detection
   |-- Response quality scoring
```

CHUNKING STRATEGIES (Critical Design Decision):

a) Fixed-size chunking:
   - 512 tokens per chunk, 50 token overlap
   - Simple, fast, but breaks semantic boundaries

b) Recursive character splitting:
   - Split by paragraph -> sentence -> word
   - Respects document structure
   - LangChain RecursiveCharacterTextSplitter

c) Semantic chunking:
   - Embed sentences, split where similarity drops
   - Preserves semantic coherence
   - Higher quality but more compute

d) Document-structure-aware chunking:
   - Use document headings, sections as boundaries
   - Maintain metadata (section title, page number)
   - Best for structured documents

HYBRID SEARCH WITH RECIPROCAL RANK FUSION:
```python
def reciprocal_rank_fusion(results_list, k=60):
    """Combine multiple ranked lists using RRF."""
    fused_scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

# Usage
dense_results = vector_db.search(query_embedding, top_k=20)
sparse_results = bm25_index.search(query_text, top_k=20)
fused = reciprocal_rank_fusion([dense_results, sparse_results])
```

ADVANCED RAG PATTERNS:

1. QUERY EXPANSION / MULTI-QUERY:
   - Generate 3-5 variations of user query using LLM
   - Retrieve for each, merge results
   - Catches different phrasings and aspects

2. HYDE (Hypothetical Document Embeddings):
   - Ask LLM to generate a hypothetical answer
   - Embed the hypothetical answer
   - Use that embedding for retrieval
   - Often retrieves better documents than query embedding

3. SELF-RAG / CORRECTIVE RAG:
   - After retrieval, LLM evaluates document relevance
   - If documents not relevant, reformulate query and retry
   - If answer not grounded in docs, regenerate
   - Reduces hallucinations significantly

4. AGENTIC RAG:
   - LLM decides when to retrieve, what to search
   - Can do multi-step retrieval
   - Can query different sources based on question type
   - More flexible but higher latency

SCALING CONSIDERATIONS:
- Vector DB: shard by document category, replicate for read throughput
- Embedding service: batch requests, GPU auto-scaling
- Cache frequent queries (semantic cache with threshold 0.95)
- Async ingestion pipeline with message queue (Kafka/SQS)
- Index updates: near-real-time with write-ahead log


## 2.3 DESIGN A RECOMMENDATION SYSTEM WITH LLMs


ARCHITECTURE:

```
[User Activity Stream]
        |
[Event Collector (Kafka)]
        |
   +----+----+
   |              |
[Real-time         [Batch Processing]
 Feature Store]     |-- User embeddings (daily)
   |-- Recent       |-- Item embeddings (on new items)
   |   interactions  |-- Collaborative filtering
   |-- Session       |-- Content-based features
   |   context       |
                     [Feature Store (Feast/Tecton)]
        |
[Candidate Generation]
   |-- ANN search (user embedding -> item embeddings)
   |-- Collaborative filtering candidates
   |-- Popularity-based candidates
   |-- LLM-based candidates (for cold start / exploration)
        |
[Ranking Service]
   |-- Traditional ML ranker (XGBoost/LightGBM)
   |-- LLM reranker for personalization
   |-- Business rules (diversity, freshness, fairness)
        |
[LLM Explanation Layer] (Optional)
   |-- Generate natural language explanations
   |-- "Recommended because you enjoyed X"
        |
[Response Assembly]
   |-- Final top-K recommendations
   |-- Explanations
   |-- Metadata
```

WHERE LLMs ADD VALUE:
1. Cold-start problem: LLM understands item content semantically
2. Cross-domain recommendations: LLM can reason across categories
3. Conversational recommendations: "Find me something like X but more Y"
4. Explanation generation: Natural language reasons for recommendations
5. Feature extraction: Extract structured features from unstructured content

LLM-ENHANCED RANKING:
```python
def llm_rerank(user_profile, candidates, top_k=10):
    prompt = f"""Given this user profile:
    - Interests: {user_profile['interests']}
    - Recent activity: {user_profile['recent_items']}
    - Preferences: {user_profile['preferences']}

    Rank these items by relevance (most relevant first):
    {[c['title'] + ': ' + c['description'] for c in candidates]}

    Return ranked list with brief reasoning for each."""

    response = llm.generate(prompt)
    return parse_ranked_list(response)
```

SCALING:
- Candidate generation: pre-compute, ANN index (FAISS/ScaNN)
- LLM calls only for top candidates (50-100 items, not millions)
- Cache LLM rankings for popular item sets
- Batch reranking for non-real-time recommendations


## 2.4 DESIGN A DOCUMENT PROCESSING PIPELINE


REQUIREMENTS:
- Input: PDFs, images, Word docs, emails, scanned documents
- Processing: OCR, classification, entity extraction, summarization
- Volume: 1M documents/day
- Output: structured data, searchable index, summaries

ARCHITECTURE:

```
[Document Ingestion]
   |-- S3 event trigger / API upload / Email ingestion
        |
[Document Classification]
   |-- ML classifier (document type: invoice, contract, report, etc.)
   |-- Route to appropriate processing pipeline
        |
[Pre-Processing]
   |-- PDF text extraction (PyPDF2, pdfplumber)
   |-- OCR for scanned docs (Tesseract, AWS Textract, Google Vision)
   |-- Table extraction (Camelot, Tabula)
   |-- Image extraction and captioning
        |
[Chunking & Structuring]
   |-- Document-aware chunking (respect sections, tables)
   |-- Metadata extraction (dates, parties, amounts)
        |
[AI Processing Layer] (Parallel processing with task queue)
   |-- Entity Extraction (NER: names, dates, amounts, addresses)
   |-- Summarization (hierarchical: section summaries -> doc summary)
   |-- Classification (topic, sentiment, urgency)
   |-- Key-value extraction (structured fields from unstructured text)
   |-- Relationship extraction (entities and their relationships)
        |
[Quality Assurance]
   |-- Confidence scoring
   |-- Human-in-the-loop for low-confidence extractions
   |-- Validation rules (dates make sense, amounts are valid)
        |
[Output Layer]
   |-- Structured data -> Database (PostgreSQL)
   |-- Embeddings -> Vector DB (for search)
   |-- Documents -> Document store (S3 + Elasticsearch)
   |-- Events -> Event bus (for downstream systems)

[Worker Architecture]
   |-- Celery/SQS workers for CPU tasks (OCR, parsing)
   |-- GPU workers for AI tasks (NER, summarization)
   |-- Auto-scaling based on queue depth
```

SCALING STRATEGY:
- Message queue (SQS/RabbitMQ) for work distribution
- Separate worker pools: CPU workers (OCR) vs GPU workers (LLM)
- Batch similar documents for efficient GPU utilization
- Priority lanes: urgent documents processed first
- Idempotent processing: safe to retry on failure


## 2.5 DESIGN A MULTI-AGENT WORKFLOW SYSTEM


REQUIREMENTS:
- Multiple specialized AI agents collaborating on complex tasks
- Agent types: researcher, writer, coder, reviewer, planner
- Coordination: sequential, parallel, hierarchical workflows
- Reliability: retry, fallback, human escalation

ARCHITECTURE:

```
[User Request]
      |
[Orchestrator Agent (Planner)]
   |-- Decomposes task into subtasks
   |-- Creates execution plan (DAG of tasks)
   |-- Assigns agents to tasks
      |
[Agent Registry]
   |-- Agent A: Research Agent (web search, RAG)
   |-- Agent B: Analysis Agent (data processing, reasoning)
   |-- Agent C: Code Agent (code generation, execution)
   |-- Agent D: Writing Agent (content generation)
   |-- Agent E: Review Agent (quality check, fact-check)
      |
[Execution Engine]
   |-- Task Queue (Redis/Celery)
   |-- Dependency resolution (run B after A completes)
   |-- Parallel execution where possible
   |-- State management (track progress of each agent)
      |
[Shared Memory / Context Store]
   |-- Working memory (current task context)
   |-- Long-term memory (past interactions, learned facts)
   |-- Artifact store (generated files, data, code)
      |
[Tool Registry]
   |-- Web search, Code execution (sandboxed)
   |-- Database queries, API calls
   |-- File operations, Image generation
      |
[Monitoring & Control]
   |-- Agent activity logs
   |-- Cost tracking per agent
   |-- Timeout / circuit breaker per agent
   |-- Human-in-the-loop checkpoints
```

AGENT COMMUNICATION PATTERNS:

a) SEQUENTIAL (Pipeline):
   [Research] --> [Analyze] --> [Write] --> [Review] --> [Output]

b) HIERARCHICAL:
   [Manager Agent]
      |-- [Worker Agent 1]
      |-- [Worker Agent 2]
      |-- [Worker Agent 3]
   Manager synthesizes results

c) COLLABORATIVE (Debate):
   [Agent A proposes] --> [Agent B critiques] --> [Agent A revises] --> ...

d) PARALLEL FAN-OUT / FAN-IN:
   [Planner] --> [Agent 1] (parallel)
             --> [Agent 2] (parallel)
             --> [Agent 3] (parallel)
                    |
             [Aggregator] --> [Output]

FRAMEWORKS: LangGraph, CrewAI, AutoGen, OpenAI Swarm

KEY DESIGN DECISIONS:
1. State management: How agents share context
2. Error handling: What happens when an agent fails
3. Cost control: Budget per agent, total budget per request
4. Determinism: How to make workflows reproducible
5. Observability: Tracing agent decisions and interactions


## 2.6 DESIGN AN AI-POWERED SEARCH ENGINE


ARCHITECTURE:

```
[Search Query]
      |
[Query Understanding]
   |-- Intent classification (navigational, informational, transactional)
   |-- Query expansion (synonyms, related terms)
   |-- Entity recognition (product names, people, places)
   |-- Spell correction
      |
[Multi-Stage Retrieval]
   |
   |-- Stage 1: Candidate Retrieval (fast, broad)
   |     |-- Inverted index (Elasticsearch/Lucene) - BM25
   |     |-- ANN vector search (embeddings)
   |     |-- Knowledge graph traversal
   |     |-- Return top 1000 candidates
   |
   |-- Stage 2: Reranking (accurate, narrow)
   |     |-- Cross-encoder model (query, document pairs)
   |     |-- Learning-to-rank features
   |     |-- Personalization signals
   |     |-- Return top 50
   |
   |-- Stage 3: LLM-Enhanced Results
   |     |-- Generate answer snippets
   |     |-- Summarize top results
   |     |-- Generate "People also ask" questions
   |     |-- Knowledge panel generation
      |
[Result Assembly]
   |-- Blended results (web, images, videos, news)
   |-- Featured snippet / direct answer
   |-- Related searches
   |-- Ads integration (if commercial)
      |
[Feedback Loop]
   |-- Click-through tracking
   |-- Dwell time measurement
   |-- Explicit feedback (thumbs up/down)
   |-- Feed back into ranking model training
```

EMBEDDING STRATEGY FOR SEARCH:
- Bi-encoder for candidate retrieval (fast, O(1) per document)
- Cross-encoder for reranking (accurate, O(n) per candidate)
- ColBERT-style late interaction for balance of speed and accuracy

SCALING:
- Elasticsearch cluster: sharded by document type, replicated
- Vector index: partitioned, in-memory for hot data
- LLM generation: only for top results, with caching
- Result caching: popular queries cached for 5-15 minutes


## 2.7 DESIGN A CONTENT MODERATION SYSTEM


ARCHITECTURE:

```
[User Generated Content]
   |-- Text, Images, Video, Audio
        |
[Pre-Filter (Fast, Rule-Based)]
   |-- Regex patterns (known bad words, URLs)
   |-- Blocklist matching
   |-- Rate limiting per user
   |-- File type validation
   |-- < 10ms latency
        |
[ML Classification Layer (Medium Speed)]
   |-- Text toxicity classifier (fine-tuned BERT/RoBERTa)
   |-- Image NSFW classifier (CNN-based)
   |-- Spam classifier
   |-- Each returns confidence score
   |-- < 100ms latency
        |
[LLM Analysis Layer (Slow, High Accuracy)]
   |-- Only for medium-confidence cases (0.3 < score < 0.7)
   |-- Nuanced understanding (sarcasm, context)
   |-- Policy-specific evaluation
   |-- Multi-label classification with reasoning
   |-- < 2s latency
        |
[Decision Engine]
   |-- Auto-approve (all classifiers say safe, high confidence)
   |-- Auto-reject (any classifier says harmful, high confidence)
   |-- Queue for human review (low confidence or borderline)
        |
[Human Review Queue]
   |-- Priority-based (severity of potential violation)
   |-- Time SLA (urgent content reviewed within 1 hour)
   |-- Reviewer tools: context view, user history, similar content
        |
[Actions]
   |-- Approve (content published)
   |-- Remove (content deleted)
   |-- Warn (user warned)
   |-- Restrict (user restricted)
   |-- Escalate (legal, law enforcement)
        |
[Feedback Loop]
   |-- Human decisions feed back to train ML models
   |-- False positive/negative tracking
   |-- Model retraining pipeline (weekly)
```

TIERED APPROACH RATIONALE:
- Tier 1 (Rules): Catches 60% of violations, < 10ms, near-zero cost
- Tier 2 (ML): Catches 30% more, < 100ms, low cost
- Tier 3 (LLM): Catches remaining 10%, < 2s, higher cost
- Only ~5% of content needs human review

COST OPTIMIZATION:
- Rule-based filter handles bulk (free)
- Lightweight ML models on CPU (cheap)
- LLM only for ambiguous cases (expensive but rare)
- Human review is most expensive, minimize volume

## 2.8 DESIGN AN AI AGENT SYSTEM (2025-2026)


Q: Design an autonomous AI agent system that can handle complex,
multi-step tasks with tool use.

ARCHITECTURE:
```
[User Request]
      |
[Agent Controller / Orchestrator]
      |
      ├── [Planning Module]
      │     └── Task decomposition, goal tracking
      ├── [Memory Module]
      │     ├── Short-term (conversation context)
      │     ├── Long-term (vector DB for past interactions)
      │     └── Working memory (current task state)
      ├── [Tool Execution Module]
      │     ├── MCP Servers (standardized tool access)
      │     ├── Function Calling (provider-native)
      │     └── Custom API integrations
      ├── [Safety / Guardrails]
      │     ├── Input validation
      │     ├── Output filtering
      │     └── Action approval (HITL for risky operations)
      └── [Observation / Evaluation]
            ├── Step-level logging
            ├── Token/cost tracking
            └── Quality metrics
```

KEY DESIGN DECISIONS:

1. Agent Pattern Selection:
   - ReAct (Reason + Act): Best for single-agent, tool-using tasks
   - Plan-and-Execute: Best for complex multi-step tasks
   - Multi-Agent Supervisor: Best for tasks requiring diverse expertise
   - Swarm: Best for dynamic, peer-to-peer agent collaboration

2. Tool Integration Strategy:
   - MCP (Model Context Protocol): Universal standard, reusable servers
   - Native Function Calling: Per-provider (OpenAI, Claude, Gemini)
   - A2A Protocol: For agent-to-agent communication
   - Composio: 800+ pre-built tool integrations

3. State Management:
   - LangGraph checkpointing for graph-based agents
   - Redis for ephemeral state across distributed agents
   - Vector DB for long-term memory (Pinecone, Qdrant, pgvector)

4. Reliability Patterns:
   - Max iteration limits to prevent infinite loops
   - Timeout per step and per task
   - Fallback models (if primary provider fails)
   - Human-in-the-loop for high-stakes actions

5. Observability:
   - LangSmith / Langfuse for tracing agent steps
   - Token usage and cost per agent run
   - Success/failure rates per tool
   - Latency per step breakdown

FRAMEWORK SELECTION (2026):
| Use Case | Best Framework |
|----------|---------------|
| Complex workflows, precise control | LangGraph |
| Multi-agent conversations | AG2 / AutoGen |
| Role-based teams, business flows | CrewAI |
| Simple OpenAI-only agents | OpenAI Agents SDK |
| Enterprise AWS deployments | Bedrock Agents (AgentCore) |
| Enterprise Azure deployments | Foundry Agent Service |

> YOUR EXPERIENCE: At RavianAI, you built exactly this -- a production
> agentic AI platform with WebSocket communication, tool integration, and
> multi-agent orchestration. Discuss your architectural decisions around
> state management, tool calling patterns, and reliability mechanisms.

## END OF SECTION 2 (Updated February 2026)

