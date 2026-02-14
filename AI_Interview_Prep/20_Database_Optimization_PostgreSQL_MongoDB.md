---
title: "Database Optimization"
layout: default
parent: "System Design & Architecture"
nav_order: 6
---

# Database Optimization: PostgreSQL & MongoDB Interview Guide
## For AI Engineers (2025-2026)

---

# TABLE OF CONTENTS
1. PostgreSQL Fundamentals for AI Engineers
2. Explaining to a Layman
3. PostgreSQL Index Types Deep Dive
4. Query Optimization & EXPLAIN
5. Schema Design Patterns
6. MongoDB for Semi-Structured Data
7. PostgreSQL vs MongoDB Decision Guide
8. pgvector: Vector Search in PostgreSQL
9. Database Design for AI Applications
10. Interview Questions (25+)

---

# SECTION 1: POSTGRESQL FUNDAMENTALS FOR AI ENGINEERS

PostgreSQL is the most advanced open-source relational database. For AI engineers, it's critical because:
- **pgvector**: Vector similarity search for RAG/embeddings
- **JSONB**: Semi-structured data for agent state, LLM responses
- **Full-text search**: Built-in text search for hybrid retrieval
- **Scalability**: Handles billions of rows with proper optimization

> 🔵 **YOUR EXPERIENCE**: At Fealty Technologies, you designed PostgreSQL database schemas with optimization strategies. At RavianAI, PostgreSQL (via AWS RDS) stores agent configurations, user data, and workflow state.

---

# SECTION 2: EXPLAINING TO A LAYMAN

> **Indexes** are like the index at the back of a textbook. Instead of reading every page to find "machine learning," you look in the index, find the page number, and go directly there. Database indexes work the same way -- they help the database find data without scanning every row.

> **EXPLAIN** is like asking the database "show me your plan before you do the work." It reveals whether the database will use an index (fast) or scan every row (slow).

---

# SECTION 3: POSTGRESQL INDEX TYPES DEEP DIVE

| Index Type | Best For | How It Works | Example Use Case |
|-----------|---------|-------------|-----------------|
| **B-Tree** (default) | Equality, range, sorting | Balanced tree structure | WHERE age > 25, ORDER BY name |
| **Hash** | Exact equality only | Hash table lookup | WHERE id = 'abc123' |
| **GIN** | Full-text search, JSONB, arrays | Inverted index | WHERE tags @> '{ai,ml}' |
| **GiST** | Geometric, range, nearest-neighbor | Generalized search tree | PostGIS geo queries |
| **BRIN** | Large tables with natural ordering | Block range summaries | Time-series data (created_at) |
| **SP-GiST** | Partitioned data (IP, phone) | Space-partitioned tree | IP range lookups |

## B-Tree Index (Default)

```sql
-- Single column index
CREATE INDEX idx_users_email ON users(email);

-- Multi-column (composite) index
-- IMPORTANT: Column order matters! Left-to-right prefix matching
CREATE INDEX idx_users_name_date ON users(last_name, created_at);

-- Partial index (only index active users)
CREATE INDEX idx_active_users ON users(email) WHERE is_active = true;

-- Unique index
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);

-- Index on expression
CREATE INDEX idx_users_lower_email ON users(LOWER(email));
```

## GIN Index (For JSONB and Full-Text)

```sql
-- JSONB index (for agent configurations, LLM response metadata)
CREATE INDEX idx_config_jsonb ON agent_configs USING gin(config);

-- Query: Find agents with specific tools
SELECT * FROM agent_configs WHERE config @> '{"tools": ["web_search"]}';

-- Full-text search index
CREATE INDEX idx_docs_fts ON documents USING gin(to_tsvector('english', content));

-- Query: Full-text search
SELECT * FROM documents
WHERE to_tsvector('english', content) @@ to_tsquery('english', 'machine & learning');
```

## BRIN Index (For Time-Series)

```sql
-- Perfect for large tables ordered by time (logs, events, metrics)
CREATE INDEX idx_logs_timestamp ON agent_logs USING brin(created_at);

-- Very small index size compared to B-Tree for time-series data
-- B-Tree on 1B rows: ~20GB index
-- BRIN on 1B rows: ~1MB index (but less precise)
```

---

# SECTION 4: QUERY OPTIMIZATION & EXPLAIN

## Using EXPLAIN ANALYZE

```sql
-- Always use EXPLAIN ANALYZE to see actual execution
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM agent_logs
WHERE agent_id = 'email-agent'
  AND created_at > '2025-01-01'
ORDER BY created_at DESC
LIMIT 100;

-- Output interpretation:
-- Seq Scan = Full table scan (BAD for large tables)
-- Index Scan = Using index (GOOD)
-- Bitmap Index Scan = Using index bitmap (OK for many results)
-- Sort = Sorting in memory or disk (check for disk sorts)
-- Hash Join = Hash-based join (good for large tables)
-- Nested Loop = Loop join (good for small result sets)
```

## Common Optimization Patterns

```sql
-- 1. Add missing index
-- Before: Seq Scan on agent_logs (cost=0.00..1500000.00)
CREATE INDEX idx_agent_logs_agent_id ON agent_logs(agent_id, created_at DESC);
-- After: Index Scan (cost=0.43..150.00) -- 10000x faster!

-- 2. Use covering index (INDEX ONLY SCAN)
CREATE INDEX idx_users_covering ON users(email) INCLUDE (name, created_at);
-- No need to access table at all -- everything in the index

-- 3. Avoid SELECT * -- only select needed columns
SELECT id, name FROM users WHERE email = 'x@y.com'; -- faster than SELECT *

-- 4. Use EXISTS instead of IN for subqueries
-- Bad:
SELECT * FROM users WHERE id IN (SELECT user_id FROM active_sessions);
-- Good:
SELECT * FROM users u WHERE EXISTS (
    SELECT 1 FROM active_sessions s WHERE s.user_id = u.id
);

-- 5. Batch operations instead of row-by-row
-- Bad: 1000 individual INSERTs
-- Good:
INSERT INTO embeddings (id, vector) VALUES
    ('id1', '[0.1, 0.2, ...]'),
    ('id2', '[0.3, 0.4, ...]'),
    ... -- Batch of 1000 in one statement

-- 6. Connection pooling (PgBouncer or built-in)
-- Without: New TCP connection per request (~50ms overhead)
-- With: Reuse persistent connections (~0ms overhead)

-- 7. VACUUM and ANALYZE regularly
ANALYZE agent_logs;  -- Update statistics for query planner
VACUUM agent_logs;   -- Reclaim dead row space
```

## Identifying Slow Queries

```sql
-- Enable slow query logging
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1s

-- Check pg_stat_statements for top slow queries
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;
```

---

# SECTION 5: SCHEMA DESIGN PATTERNS

## For AI Agent Applications

```sql
-- Agent configuration storage
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    system_prompt TEXT NOT NULL,
    config JSONB DEFAULT '{}',     -- Flexible config (tools, params)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversation history (for memory)
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    role VARCHAR(20) NOT NULL,     -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',   -- tokens_used, model, latency
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_conv_session ON conversations(session_id, created_at);

-- Tool execution logs
CREATE TABLE tool_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    input JSONB NOT NULL,
    output JSONB,
    status VARCHAR(20) DEFAULT 'pending', -- pending, success, error
    duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_tool_session ON tool_executions(session_id, created_at);
CREATE INDEX idx_tool_name ON tool_executions(tool_name, status);
```

---

# SECTION 6: MONGODB FOR SEMI-STRUCTURED DATA

## When to Use MongoDB over PostgreSQL

| Use Case | PostgreSQL | MongoDB |
|----------|-----------|---------|
| Structured, relational data | Best | Possible |
| Schema evolves frequently | Possible (JSONB) | Best |
| Document storage (varying shapes) | Possible | Best |
| Transactions (multi-table) | Best | Supported (4.0+) |
| Vector search | pgvector | Atlas Vector Search |
| Aggregation pipelines | SQL (powerful) | Aggregation framework |
| Horizontal scaling (sharding) | Limited (Citus) | Native |

## MongoDB Optimization

```javascript
// 1. Create compound indexes (order matters!)
db.conversations.createIndex({ session_id: 1, created_at: -1 });

// 2. Use projections (don't fetch unnecessary fields)
db.agents.find({ name: "email-agent" }, { config: 1, name: 1 });

// 3. Aggregation pipeline for analytics
db.tool_executions.aggregate([
    { $match: { status: "success" } },
    { $group: { _id: "$tool_name", avg_duration: { $avg: "$duration_ms" }, count: { $sum: 1 } } },
    { $sort: { count: -1 } }
]);

// 4. Use explain() to analyze queries
db.conversations.find({ session_id: "abc" }).explain("executionStats");
```

---

# SECTION 7: POSTGRESQL vs MONGODB DECISION GUIDE

```
Need ACID transactions?                    → PostgreSQL
Schema changes frequently?                 → MongoDB
Need vector search (RAG)?                  → PostgreSQL (pgvector)
Storing agent configs (varied shapes)?     → Either (Postgres JSONB or MongoDB)
Time-series agent logs?                    → PostgreSQL (BRIN index) or MongoDB
Need horizontal sharding at scale?         → MongoDB
Using Django ORM?                          → PostgreSQL
Need full-text search?                     → PostgreSQL (built-in) or MongoDB (Atlas)
Existing team expertise?                   → Use what you know
```

---

# SECTION 8: PGVECTOR - VECTOR SEARCH IN POSTGRESQL

```sql
-- Install extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
    metadata JSONB DEFAULT '{}'
);

-- Create HNSW index (recommended for most use cases)
CREATE INDEX idx_embeddings_hnsw ON document_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Similarity search
SELECT content, 1 - (embedding <=> query_embedding::vector) AS similarity
FROM document_embeddings
ORDER BY embedding <=> query_embedding::vector
LIMIT 5;

-- Hybrid search (vector + metadata filter)
SELECT content, 1 - (embedding <=> query_embedding::vector) AS similarity
FROM document_embeddings
WHERE metadata->>'category' = 'technical'
ORDER BY embedding <=> query_embedding::vector
LIMIT 5;
```

**pgvector vs Dedicated Vector DBs:**
| Feature | pgvector | Pinecone | Qdrant |
|---------|----------|----------|--------|
| Setup | Add extension | Managed service | Self-hosted/Cloud |
| Scale | Millions | Billions | Billions |
| Filtering | Full SQL | Metadata filters | Rich filters |
| Cost | Free | Pay per usage | Free (self-hosted) |
| Best for | Existing Postgres users | Zero-ops production | Complex filtering |

---

# SECTION 9: DATABASE DESIGN FOR AI APPLICATIONS

## RAG Application Schema

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    title TEXT,
    source TEXT,
    content TEXT,
    chunk_index INTEGER,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_docs_embedding ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_docs_metadata ON documents USING gin (metadata);

-- Query log (for analytics and improvement)
CREATE TABLE query_log (
    id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    query_embedding vector(1536),
    retrieved_doc_ids UUID[],
    response TEXT,
    feedback INTEGER,  -- 1 = thumbs up, -1 = thumbs down
    latency_ms INTEGER,
    tokens_used INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

# SECTION 10: INTERVIEW QUESTIONS (25+)

**Q1: What PostgreSQL index types do you know and when would you use each?**
B-Tree (default, equality/range), Hash (exact equality only), GIN (JSONB, full-text, arrays), GiST (geometric, range), BRIN (large time-series tables), SP-GiST (partitioned data like IP addresses). For AI apps, GIN for JSONB configs and pgvector's HNSW for embeddings are most relevant.

**Q2: How do you optimize a slow PostgreSQL query?**
1. Run EXPLAIN ANALYZE to see the execution plan
2. Look for Seq Scans on large tables (add missing indexes)
3. Check for Sort operations (add index with ORDER BY columns)
4. Ensure statistics are up to date (ANALYZE)
5. Consider covering indexes for INDEX ONLY SCAN
6. Check connection pooling configuration

**Q3: What is the difference between a B-Tree and a GIN index?**
B-Tree is a balanced tree for scalar comparisons (=, <, >, BETWEEN). GIN (Generalized Inverted Index) is an inverted index for composite values -- it indexes individual elements within arrays, JSONB documents, or text tokens. GIN is essential for full-text search and JSONB queries.

**Q4: How does VACUUM work and why is it important?**
PostgreSQL uses MVCC (Multi-Version Concurrency Control) -- UPDATE and DELETE don't remove old rows immediately, they mark them as dead. VACUUM reclaims space from dead rows. Without it, tables bloat and queries slow down. Autovacuum handles this automatically but may need tuning for write-heavy AI workloads.

**Q5: Explain connection pooling and why it matters.**
Creating a new PostgreSQL connection takes ~50ms (TCP handshake, authentication). Connection pooling (PgBouncer, built-in) maintains a pool of persistent connections that are reused. For FastAPI with async SQLAlchemy, this is critical -- asyncpg + connection pool handles thousands of concurrent requests.

**Q6: How do you design a schema for an AI agent application?**
Agents table (config as JSONB), Conversations table (session-based messages), Tool Executions log, Embeddings table (with pgvector), User Preferences (JSONB). Use JSONB for flexible schema parts, proper indexes on session_id and timestamps, and partitioning for large log tables.

**Q7: What is pgvector and how does it compare to dedicated vector databases?**
pgvector adds vector similarity search to PostgreSQL. It supports HNSW and IVFFlat indexes. Best for: existing Postgres users who need vector search for millions of vectors. Dedicated vector DBs (Pinecone, Qdrant) are better for billion-scale or when you need specialized features.

**Q8: What is a partial index and when would you use it?**
An index that only covers rows matching a WHERE condition. Example: CREATE INDEX idx_active ON users(email) WHERE is_active = true. Smaller index size, faster updates, faster queries for the common case.

**Q9: How do you handle schema migrations in production?**
Use Alembic (SQLAlchemy) or Django migrations. Always: test on staging first, use transactions, avoid locking operations on large tables (use CREATE INDEX CONCURRENTLY), have rollback plans.

**Q10: When would you choose MongoDB over PostgreSQL?**
When: schema changes frequently, documents have highly variable structure, you need native horizontal sharding, or your team has MongoDB expertise. PostgreSQL with JSONB covers many MongoDB use cases while providing ACID transactions.

**Q11: How do you implement full-text search in PostgreSQL?**
Use tsvector and tsquery with GIN index. Supports stemming, ranking, phrase search, and language-specific configurations. For AI apps, combine with pgvector for hybrid search (semantic + keyword).

**Q12: What are covering indexes?**
Indexes that INCLUDE non-key columns, enabling INDEX ONLY SCAN (no table access needed). Example: CREATE INDEX idx ON users(email) INCLUDE (name, created_at). Much faster for queries that only need the included columns.

**Q13: How do you handle database performance at scale?**
Read replicas for read-heavy workloads, connection pooling, table partitioning (by date for logs), proper indexing, query optimization, caching (Redis for hot data), and monitoring (pg_stat_statements).

**Q14: What is JSONB and how does it differ from JSON in PostgreSQL?**
JSONB is binary JSON -- stored in a decomposed format that's slower to write but much faster to query and index. JSON stores the exact text. Always use JSONB for queryable data. GIN indexes work with JSONB for fast lookups.

**Q15: How did you optimize database performance at Fealty Technologies?**
Implemented efficient query patterns and indexing strategies: identified slow queries via pg_stat_statements, added composite indexes for common query patterns, used partial indexes for filtered queries, implemented connection pooling, and designed schemas with denormalization where appropriate for read-heavy patterns.

---

## Sources
- [GeeksforGeeks: PostgreSQL Interview Questions 2025](https://www.geeksforgeeks.org/postgresql/postgresql-interview-questions/)
- [Turing: 100+ PostgreSQL Interview Questions](https://www.turing.com/interview-questions/postgresql)
- [Hello Interview: PostgreSQL Deep Dive for System Design](https://www.hellointerview.com/learn/system-design/deep-dives/postgres)
- [MentorCruise: 80 PostgreSQL Interview Questions 2026](https://mentorcruise.com/questions/postgresql/)
- [Second Talent: Advanced PostgreSQL Questions 2026](https://www.secondtalent.com/interview-guide/postgresql/)
