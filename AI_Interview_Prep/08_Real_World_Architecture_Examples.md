---
title: "Real-World Architecture Examples"
layout: default
parent: "System Design & Architecture"
nav_order: 5
---

SECTION 8: REAL-WORLD ARCHITECTURE EXAMPLES
AI System Design Interview Guide for AI/ML Engineers (2025-2026)
## 

8.1 CHATGPT-LIKE CONVERSATIONAL AI PLATFORM

```
ARCHITECTURE DIAGRAM (Description):

Layer 1 - Client Layer:
+------------------+  +------------------+  +------------------+
|   Web Client     |  |   Mobile App     |  |   API Client     |
|   (React/Next)   |  |   (React Native) |  |   (REST/SDK)     |
+--------+---------+  +--------+---------+  +--------+---------+
         |                      |                      |
         +----------+-----------+----------+-----------+
                    |
Layer 2 - Edge & Gateway:
+-------------------------------------------------------------------+
|                     CDN (CloudFront/Cloudflare)                    |
+-------------------------------------------------------------------+
+-------------------------------------------------------------------+
|  API Gateway (Kong)                                                |
|  - Auth (JWT/API Key)                                              |
|  - Rate Limiting (token-based)                                     |
|  - WebSocket/SSE upgrade                                           |
|  - Request routing                                                 |
+-------------------------------------------------------------------+
                    |
Layer 3 - Application Services:
+------------------+  +------------------+  +------------------+
| Chat Orchestrator|  | User Service     |  | Billing Service  |
| - Session mgmt   |  | - Auth           |  | - Token counting |
| - Context build   |  | - Profiles       |  | - Usage tracking |
| - Tool routing    |  | - Preferences    |  | - Subscription   |
| - Guardrails      |  |                  |  | - Invoicing      |
+--------+---------+  +------------------+  +------------------+
         |
Layer 4 - AI Services:
+------------------+  +------------------+  +------------------+
| LLM Router       |  | RAG Service      |  | Tool Executor    |
| - Model selection |  | - Embedding      |  | - Code interpreter|
| - Fallback chain  |  | - Vector search  |  | - Web search     |
| - A/B routing     |  | - Reranking      |  | - File operations|
| - Load balancing  |  |                  |  | - API calls      |
+--------+---------+  +--------+---------+  +--------+---------+
         |                      |
Layer 5 - Model Serving:
+-------------------------------------------------------------------+
| Model Serving Cluster                                              |
| +-------------+ +-------------+ +-------------+ +-------------+   |
| | vLLM Pool   | | vLLM Pool   | | Embedding   | | Reranker    |   |
| | GPT-4 class | | GPT-3.5     | | Pool        | | Pool        |   |
| | 8xH100 each | | class       | | (A10G)      | | (A10G)      |   |
| | 10 replicas | | 4xA100 each | | 20 replicas | | 5 replicas  |   |
| |             | | 50 replicas | |             | |             |   |
| +-------------+ +-------------+ +-------------+ +-------------+   |
+-------------------------------------------------------------------+
                    |
Layer 6 - Data Layer:
+------------------+  +------------------+  +------------------+
| PostgreSQL       |  | Redis Cluster    |  | Vector DB        |
| - Users          |  | - Sessions       |  | (Pinecone)       |
| - Conversations  |  | - Cache          |  | - Document       |
| - Billing        |  | - Rate limits    |  |   embeddings     |
+------------------+  +------------------+  +------------------+
+------------------+  +------------------+
| S3/Blob Storage  |  | Kafka            |
| - File uploads   |  | - Event stream   |
| - Model artifacts|  | - Audit log      |
| - Generated files|  | - Analytics      |
+------------------+  +------------------+
```

SCALE NUMBERS:
- 100M+ users
- 10M+ DAU
- 1B+ messages/day
- 10K+ concurrent streaming connections per server
- P95 TTFT < 500ms
- Multi-region deployment (US, EU, Asia)

KEY DESIGN DECISIONS:
1. WebSocket for streaming (SSE as fallback)
2. Separate model pools for different tiers (paid vs free users)
3. Redis for session state (fast, TTL-based expiry)
4. Kafka for async event processing (analytics, logging)
5. Multi-model router for cost optimization


## 8.2 ENTERPRISE RAG SYSTEM (e.g., Internal Knowledge Base)


```
ARCHITECTURE DIAGRAM (Description):

INGESTION SIDE:
+------------------+  +------------------+  +------------------+
| Confluence/Docs  |  | Slack/Email      |  | Database/API     |
| Connector        |  | Connector        |  | Connector        |
+--------+---------+  +--------+---------+  +--------+---------+
         |                      |                      |
         +----------+-----------+----------+-----------+
                    |
+-------------------------------------------------------------------+
| Document Processing Pipeline (Airflow/Temporal)                    |
| 1. Extract text (Unstructured.io, Textract)                       |
| 2. Clean & normalize                                               |
| 3. Chunk (recursive, 512 tokens, 50 overlap)                      |
| 4. Extract metadata (source, date, author, permissions)            |
| 5. Generate embeddings (batch, GPU cluster)                        |
| 6. Index in vector DB + search engine                              |
+-------------------------------------------------------------------+
                    |
         +----------+-----------+
         |                      |
+--------+---------+  +--------+---------+
| Elasticsearch    |  | Qdrant / Weaviate|
| (BM25 keyword)   |  | (dense vectors)  |
| (metadata filter) |  | (HNSW index)     |
+------------------+  +------------------+

QUERY SIDE:
+------------------+
| User Query       |
+--------+---------+
         |
+-------------------------------------------------------------------+
| Query Pipeline                                                     |
| 1. Auth check (does user have access to requested docs?)           |
| 2. Query understanding (intent, entities)                          |
| 3. Query expansion (LLM generates 3 variations)                   |
| 4. Hybrid search (BM25 + vector, RRF fusion)                      |
| 5. Permission filtering (only docs user can access)                |
| 6. Cross-encoder reranking (top 50 -> top 5)                      |
| 7. Context building (top-5 chunks + metadata)                     |
| 8. LLM generation (answer + citations)                            |
| 9. Citation verification (check if answer is grounded)            |
| 10. Response with source links                                     |
+-------------------------------------------------------------------+
```

CRITICAL FEATURE - ACCESS CONTROL:
```python
class PermissionAwareRAG:
    def search(self, query, user):
        # Get user's accessible document IDs
        accessible_docs = self.acl_service.get_accessible_docs(user.id)

        # Search with permission filter
        results = self.vector_db.search(
            query_embedding=embed(query),
            filter={
                "document_id": {"$in": accessible_docs},
                "classification": {"$lte": user.clearance_level}
            },
            top_k=20
        )

        return results
```


## 8.3 AI-POWERED E-COMMERCE RECOMMENDATION ENGINE


```
ARCHITECTURE DIAGRAM (Description):

REAL-TIME DATA FLOW:
+------------------+
| User Activity    | (clicks, views, purchases, searches)
+--------+---------+
         |
+--------+---------+
| Event Stream     | (Kafka / Kinesis)
+--------+---------+
         |
    +----+----+
    |         |
+---+---+ +---+---+
|Real-  | |Batch  |
|time   | |Process|
|Feature| |(Spark)|
|Store  | |       |
|(Redis)| +---+---+
+---+---+     |
    |    +----+----+
    |    |Feature  |
    |    |Store    |
    |    |(Feast)  |
    |    +----+----+
    |         |
+---+---------+---+
| Candidate        |
| Generation       |
| - ANN (FAISS)    |
| - Collaborative  |
| - Content-based  |
| - LLM (cold start)|
+--------+---------+
         |
+--------+---------+
| Ranking Service  |
| - Two-tower model|
| - LLM reranker   |
| - Business rules |
+--------+---------+
         |
+--------+---------+
| Personalization  |
| - Diversity      |
| - Freshness      |
| - Price range    |
| - LLM explanation|
+--------+---------+
         |
+--------+---------+
| API Response     |
| Top-K items with |
| explanations     |
+------------------+

OFFLINE TRAINING PIPELINE:
+------------------+  +------------------+  +------------------+
| User Interaction |->| Feature          |->| Model Training   |
| Logs (S3)        |  | Engineering      |  | (GPU Cluster)    |
+------------------+  | (Spark)          |  | - Embedding model|
                      +------------------+  | - Ranking model  |
                                            | - LLM fine-tune  |
                                            +--------+---------+
                                                     |
                                            +--------+---------+
                                            | Model Registry   |
                                            | (MLflow)         |
                                            | - A/B test config|
                                            | - Canary deploy  |
                                            +------------------+
```

LATENCY BUDGET:
```
Total SLA: < 200ms
  Feature fetch (Redis):     10ms
  Candidate generation:      30ms
  ML Ranking:                50ms
  Business rules:            10ms
  LLM explanation (cached):  50ms (or async, don't block)
  Serialization + network:   20ms
  Buffer:                    30ms
```


## 8.4 MULTI-MODAL DOCUMENT INTELLIGENCE PLATFORM


```
ARCHITECTURE DIAGRAM (Description):

+------------------+
| Document Upload  | (PDF, images, scans, emails)
+--------+---------+
         |
+--------+---------+
| Classification   | (What type of document?)
| - CNN classifier |
| - LLM classifier|
+--------+---------+
         |
    +----+----+----+----+
    |         |         |
+---+---+ +---+---+ +---+---+
|Invoice| |Contract| |Report |
|Pipeline| |Pipeline| |Pipeline|
+---+---+ +---+---+ +---+---+
    |         |         |
    v         v         v
+-------------------------------------------------------------------+
| Shared Processing Layer                                            |
|                                                                    |
| +----------+  +----------+  +----------+  +----------+            |
| | OCR      |  | Table    |  | Layout   |  | Image    |            |
| | Service  |  | Extract  |  | Analysis |  | Caption  |            |
| | (Textract|  | (Camelot)|  | (LayoutLM|  | (GPT-4V/ |            |
| |  /Vision)|  |          |  |  v3)     |  |  LLaVA)  |            |
| +----------+  +----------+  +----------+  +----------+            |
|                                                                    |
| +----------+  +----------+  +----------+  +----------+            |
| | NER      |  | Relation |  | Summary  |  | KV       |            |
| | Extract  |  | Extract  |  | Generate |  | Extract  |            |
| | (SpaCy/  |  | (LLM)   |  | (LLM)   |  | (LLM)   |            |
| |  BERT)   |  |          |  |          |  |          |            |
| +----------+  +----------+  +----------+  +----------+            |
+-------------------------------------------------------------------+
         |
+--------+---------+
| Quality Assurance|
| - Confidence     |
|   scoring        |
| - Human review   |
|   queue          |
| - Validation     |
|   rules          |
+--------+---------+
         |
    +----+----+----+
    |         |         |
+---+---+ +---+---+ +---+---+
|Struct | |Vector | |Search |
|Data DB| |DB     | |Index  |
|(Postgres|(Qdrant)|(Elastic|
|       | |       | |search)|
+-------+ +-------+ +-------+
```


## 8.5 REAL-TIME AI CONTENT MODERATION (Social Media Scale)


```
ARCHITECTURE DIAGRAM (Description):

+-------------------------------------------------------------------+
| Content Stream (1M+ posts/minute)                                  |
| - Text, Images, Video, Audio, Links                                |
+-------------------------------------------------------------------+
         |
+--------+---------+
| Content Router   | (route by content type)
+--+----+----+--+--+
   |    |    |  |
   v    v    v  v
+----+ +----+ +----+ +----+
|Text| |Image| |Video| |Audio|
|    | |     | |     | |     |
+--+-+ +--+-+ +--+-+ +--+-+
   |       |       |       |
   v       v       v       v

TIER 1 - FAST FILTERS (< 10ms, CPU):
+-------------------------------------------------------------------+
| - Regex/blocklist matching                                         |
| - Known hash matching (PhotoDNA for CSAM)                          |
| - Spam signature matching                                          |
| - URL reputation check                                             |
| Result: 60% auto-approved, 5% auto-rejected                       |
+-------------------------------------------------------------------+
   |
TIER 2 - ML CLASSIFIERS (< 100ms, GPU):
+-------------------------------------------------------------------+
| - Text: Toxicity classifier (fine-tuned BERT)                      |
| - Image: NSFW/violence classifier (EfficientNet)                   |
| - Video: Frame sampling + image classifier                         |
| - Audio: Speech-to-text + text classifier                          |
| - Multi-modal: Combined signals                                    |
| Result: 30% auto-approved, 3% auto-rejected                       |
+-------------------------------------------------------------------+
   |
TIER 3 - LLM ANALYSIS (< 2s, GPU):
+-------------------------------------------------------------------+
| - Only for ambiguous cases (~5-10% of content)                     |
| - Context-aware analysis (sarcasm, cultural context)               |
| - Policy-specific evaluation                                       |
| - Multi-label with explanation                                     |
| Result: 4% approved, 1% rejected, 1% -> human review              |
+-------------------------------------------------------------------+
   |
TIER 4 - HUMAN REVIEW:
+-------------------------------------------------------------------+
| - ~1-2% of total content                                           |
| - Priority queue (severity-based)                                  |
| - Reviewer tools: context, user history, similar content           |
| - Decisions feed back to ML training                               |
+-------------------------------------------------------------------+

APPEALS FLOW:
[User appeals] -> [Different reviewer] -> [Senior reviewer if needed]
                                        -> [Policy update if pattern found]
```

SCALE NUMBERS:
- 1M+ content items/minute
- < 100ms for 95% of decisions
- < 2s for remaining 5%
- 99.9% automation rate (only 0.1% truly needs human)
- False positive rate < 0.1%
- False negative rate < 1%


## 8.6 AI CODING ASSISTANT (Copilot-Style)


```
ARCHITECTURE:

IDE PLUGIN SIDE:
+-------------------------------------------------------------------+
| IDE Plugin (VS Code / JetBrains)                                   |
| - Captures code context (current file, open files, cursor position)|
| - Debounce: wait 300ms after keystroke before sending              |
| - Local cache of recent completions                                |
| - Inline rendering of suggestions                                  |
| - Telemetry: accept/reject/ignore tracking                        |
+-------------------------------------------------------------------+
         |
SERVER SIDE:
+-------------------------------------------------------------------+
| API Gateway                                                        |
| - Auth (user license validation)                                   |
| - Rate limiting (requests per minute per user)                     |
| - Request deduplication (same context within 1s)                   |
+--------+---------+
         |
+--------+---------+
| Context Builder  |
| - Current file content (up to cursor)                              |
| - Surrounding files (imports, related files)                       |
| - Repository context (README, configs)                             |
| - Language/framework detection                                     |
| - Truncate to fit context window                                   |
+--------+---------+
         |
    +----+----+
    |         |
+---+---+ +---+--------+
|Code   | |Chat/       |
|Complet| |Explain     |
|ion    | |Refactor    |
+---+---+ +---+--------+
    |         |
+---+---------+---+
| Model Serving   |
| - Code model    |
|   (StarCoder/   |
|    CodeLlama/   |
|    GPT-4)       |
| - FIM (Fill-in- |
|   the-middle)   |
| - Speculative   |
|   decoding      |
+--------+--------+
         |
+--------+---------+
| Post-Processing  |
| - Syntax validation (parse generated code)                         |
| - Security scan (no hardcoded secrets)                             |
| - License check (no copyrighted code)                              |
| - Truncate to logical boundary                                     |
+------------------+

FEEDBACK LOOP:
[Accept/Reject signals] -> [Kafka] -> [Training pipeline]
                                   -> [Analytics dashboard]
```

LATENCY REQUIREMENTS:
- Inline completion: TTFT < 200ms (feels instant)
- Chat response: TTFT < 500ms
- Speculative decoding: 2-3x speedup for completions

KEY OPTIMIZATION:
- FIM (Fill-in-the-Middle): model trained to complete code given prefix AND suffix
- Prefix caching: system prompt + file context cached
- Speculative decoding: small model drafts, large model verifies
- Client-side debouncing: reduce unnecessary requests by 80%

## END OF SECTION 8

