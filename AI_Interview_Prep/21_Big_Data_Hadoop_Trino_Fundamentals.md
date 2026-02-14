# Big Data Fundamentals: Hadoop, Trino & Distributed Computing
## Interview Guide for AI/Data Engineers (2025-2026)

---

# TABLE OF CONTENTS
1. Hadoop Ecosystem Overview
2. Explaining to a Layman
3. HDFS (Hadoop Distributed File System)
4. MapReduce Paradigm
5. YARN (Resource Management)
6. Trino (formerly Presto) - SQL Query Engine
7. Hadoop vs Modern Alternatives
8. Interview Questions (20+)

---

# SECTION 1: HADOOP ECOSYSTEM OVERVIEW

Hadoop is an open-source framework for distributed storage and processing of large datasets across clusters of computers.

```
┌─────────────────────────────────────────────────────────┐
│                 HADOOP ECOSYSTEM                         │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  Hive    │  │  Pig     │  │  HBase   │  │ Trino  │  │
│  │ (SQL on  │  │ (ETL     │  │ (NoSQL   │  │ (Fast  │  │
│  │  Hadoop) │  │  scripts)│  │  on HDFS)│  │  SQL)  │  │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  │
│        └────────────┼────────────┘              │       │
│                     │                           │       │
│  ┌──────────────────┴───────────────────────────┴────┐  │
│  │               YARN (Resource Manager)              │  │
│  └────────────────────────┬──────────────────────────┘  │
│                           │                              │
│  ┌────────────────────────┴──────────────────────────┐  │
│  │           HDFS (Distributed File System)           │  │
│  │   ┌────────┐  ┌────────┐  ┌────────┐             │  │
│  │   │DataNode│  │DataNode│  │DataNode│  ...         │  │
│  │   │(Block1)│  │(Block2)│  │(Block3)│             │  │
│  │   └────────┘  └────────┘  └────────┘             │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

> 🔵 **YOUR EXPERIENCE**: You have Hadoop listed as a data engineering skill. At MathCo, your data engineering work with PySpark, Databricks, and EMR involves the same distributed computing principles that Hadoop pioneered.

---

# SECTION 2: EXPLAINING TO A LAYMAN

> **Hadoop** is like a warehouse system. Instead of storing all your inventory in one giant warehouse (which could collapse or catch fire), you split it across many smaller warehouses across the city. Each item is stored in 3 different warehouses (replication) so if one burns down, you don't lose anything. When you need to count all items, instead of bringing everything to one place, each warehouse counts its own items and sends you the total (MapReduce).

> **Trino** is like having a universal translator that can ask questions to many different warehouses simultaneously -- it speaks SQL but can query data in HDFS, S3, PostgreSQL, MySQL, and more, all at once.

---

# SECTION 3: HDFS (HADOOP DISTRIBUTED FILE SYSTEM)

## Architecture

```
                    ┌─────────────────┐
                    │   NameNode      │
                    │   (Master)      │
                    │                 │
                    │  - File metadata│
                    │  - Block → Node │
                    │    mapping      │
                    │  - Namespace    │
                    └───────┬─────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ DataNode 1 │  │ DataNode 2 │  │ DataNode 3 │
     │            │  │            │  │            │
     │ Block A    │  │ Block A    │  │ Block B    │
     │ Block B    │  │ Block C    │  │ Block C    │
     │ Block D    │  │ Block D    │  │ Block A    │
     └────────────┘  └────────────┘  └────────────┘
     
     Default: 3 replicas of each block (configurable)
     Default block size: 128MB (configurable)
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Block size** | 128MB default (vs 4KB in regular filesystems). Large blocks reduce metadata overhead. |
| **Replication** | Default factor = 3. Each block stored on 3 different nodes for fault tolerance. |
| **NameNode** | Master that stores metadata (file → block mapping). Single point of failure → use HA with standby NameNode. |
| **DataNode** | Workers that store actual data blocks. Send heartbeats to NameNode. |
| **Write-once** | HDFS is optimized for append-only. Not good for random writes. |
| **Rack awareness** | Replicas placed across different racks for fault tolerance. |

---

# SECTION 4: MAPREDUCE PARADIGM

```
Input Data → [SPLIT] → [MAP] → [SHUFFLE & SORT] → [REDUCE] → Output

Example: Word count on a large document

Input: "hello world hello hadoop world hello"

SPLIT:
  Mapper 1: "hello world hello"
  Mapper 2: "hadoop world hello"

MAP (each mapper):
  Mapper 1: (hello, 1), (world, 1), (hello, 1)
  Mapper 2: (hadoop, 1), (world, 1), (hello, 1)

SHUFFLE & SORT (group by key):
  hadoop: [1]
  hello:  [1, 1, 1]
  world:  [1, 1]

REDUCE (aggregate):
  hadoop: 1
  hello:  3
  world:  2
```

**MapReduce in PySpark equivalent:**
```python
# MapReduce word count is essentially:
text_rdd = sc.textFile("hdfs://input/docs/")
word_counts = (
    text_rdd
    .flatMap(lambda line: line.split())  # MAP: split into words
    .map(lambda word: (word, 1))         # MAP: create (key, value)
    .reduceByKey(lambda a, b: a + b)     # REDUCE: sum counts
)
```

**Why MapReduce is being replaced:**
- Writes intermediate results to disk (slow)
- High latency for iterative algorithms (ML)
- Complex to program directly
- **Spark replaced it** with in-memory processing (10-100x faster)

---

# SECTION 5: YARN (YET ANOTHER RESOURCE NEGOTIATOR)

```
┌─────────────────────────────┐
│  ResourceManager (Master)   │
│  - Scheduler                │
│  - ApplicationsManager      │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌──────────┐  ┌──────────┐
│NodeManager│  │NodeManager│
│          │  │          │
│Container1│  │Container3│
│Container2│  │Container4│
└──────────┘  └──────────┘
```

| Component | Role |
|-----------|------|
| **ResourceManager** | Cluster-wide resource allocation. Decides which applications get CPU/memory. |
| **NodeManager** | Per-node agent. Manages containers on that node. |
| **ApplicationMaster** | Per-application. Negotiates resources, monitors tasks. |
| **Container** | Unit of resource allocation (CPU + memory). Runs actual tasks. |

---

# SECTION 6: TRINO (FORMERLY PRESTO) - SQL QUERY ENGINE

## What is Trino?

Trino is a distributed SQL query engine that can query data from multiple sources (HDFS, S3, PostgreSQL, MySQL, MongoDB, Kafka) using a single SQL query.

```
┌──────────────────┐
│  SQL Query        │
│  (Standard SQL)   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Trino           │
│  Coordinator     │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│  ← Distributed query execution
└────┬───┘ └────┬───┘
     │          │
┌────┴──────────┴────────────────────────┐
│          DATA SOURCES                   │
│  ┌─────┐ ┌─────┐ ┌──────┐ ┌────────┐  │
│  │ S3  │ │HDFS │ │Postgres│ │Snowflake│ │
│  └─────┘ └─────┘ └──────┘ └────────┘  │
└────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Federated query** | Join data across S3 + PostgreSQL + MySQL in one query |
| **In-memory** | No intermediate disk writes (unlike MapReduce) |
| **ANSI SQL** | Standard SQL syntax |
| **Connectors** | 30+ data source connectors |
| **Cost-based optimizer** | Optimizes query plans based on data statistics |
| **Scalable** | Handles petabytes of data |

## Trino vs Alternatives

| Engine | Best For | Speed | SQL Support |
|--------|---------|-------|-------------|
| **Trino** | Federated ad-hoc queries | Very fast | Full ANSI SQL |
| **Hive** | Batch ETL on HDFS | Slow (MapReduce) | HiveQL |
| **Spark SQL** | ETL + ML pipelines | Fast | SparkSQL |
| **Snowflake** | Cloud data warehouse | Fast | Full SQL |
| **Athena** | Serverless S3 queries | Fast (Trino-based!) | SQL |

**Fun fact:** AWS Athena is powered by Trino under the hood.

---

# SECTION 7: HADOOP vs MODERN ALTERNATIVES

```
2010s: Hadoop + MapReduce + Hive (batch, slow, on-premise)
   │
   ▼
2015s: Spark replaced MapReduce (10-100x faster, in-memory)
   │
   ▼
2020s: Cloud-native replaced HDFS (S3 + Databricks + Snowflake)
   │
   ▼
2025s: Lakehouse architecture (Delta Lake + Iceberg + Hudi)
```

| Aspect | Hadoop (Traditional) | Modern Stack |
|--------|---------------------|-------------|
| **Storage** | HDFS (on-premise) | S3/ADLS/GCS (cloud) |
| **Processing** | MapReduce (slow) | Spark/Databricks (fast) |
| **SQL** | Hive (batch) | Trino/Snowflake/Athena |
| **Management** | Manual cluster ops | Managed services |
| **Cost** | High (always-on clusters) | Pay-per-query |
| **Scale** | Horizontal (add nodes) | Elastic (auto-scale) |

**Why Hadoop still matters for interviews:**
1. Understanding distributed computing fundamentals
2. Many enterprises still run Hadoop clusters
3. Concepts (replication, partitioning, fault tolerance) apply everywhere
4. Spark evolved FROM the Hadoop ecosystem
5. YARN is still used as a resource manager

> 🔵 **YOUR EXPERIENCE**: At MathCo (AbbVie), you used AWS EMR which runs Spark on YARN -- directly building on Hadoop's resource management. Your PySpark + Databricks work uses the same distributed computing principles Hadoop pioneered, just with modern in-memory execution.

---

# SECTION 8: INTERVIEW QUESTIONS (20+)

**Q1: What is Hadoop and what are its core components?**
Hadoop is an open-source framework for distributed storage and processing. Core components: HDFS (distributed storage), MapReduce (distributed processing), YARN (resource management). The ecosystem includes Hive (SQL), HBase (NoSQL), Pig (ETL), ZooKeeper (coordination).

**Q2: Explain HDFS architecture.**
Master-slave: NameNode (master) stores metadata (file→block mapping), DataNodes (slaves) store actual data blocks. Default block size is 128MB, default replication factor is 3. NameNode is a single point of failure → use HA configuration with standby NameNode.

**Q3: How does MapReduce work?**
Three phases: Map (process input splits in parallel, emit key-value pairs), Shuffle & Sort (group by key, transfer to reducers), Reduce (aggregate values per key). Writes intermediate results to disk, which makes it slower than Spark's in-memory processing.

**Q4: Why is Spark faster than MapReduce?**
Spark keeps data in memory (RDDs/DataFrames) between operations, while MapReduce writes to disk after each Map and Reduce step. For iterative algorithms (ML, graph), Spark can be 10-100x faster. Spark also has a more efficient DAG execution engine.

**Q5: What is YARN and how does it work?**
YARN (Yet Another Resource Negotiator) manages cluster resources. ResourceManager allocates resources across applications. NodeManagers manage containers on each node. ApplicationMaster negotiates resources per application. Containers are units of CPU + memory allocation.

**Q6: What is Trino and how does it differ from Hive?**
Trino is a distributed SQL engine for interactive queries across multiple data sources. Unlike Hive (which compiles to MapReduce/Tez jobs), Trino executes queries entirely in memory with no intermediate disk writes. Trino is much faster for ad-hoc queries but Hive is better for very large batch ETL jobs.

**Q7: Explain federated queries in Trino.**
Trino can query multiple data sources (S3, PostgreSQL, MySQL, Kafka) in a single SQL query using connectors. Example: JOIN a table in PostgreSQL with data in S3 without moving the data. Each connector translates SQL to the native format of the data source.

**Q8: What is the difference between HDFS and S3?**
HDFS: on-premise, coupled with compute, always-on cost, high throughput. S3: cloud-native, decoupled storage/compute, pay-per-use, virtually unlimited scale. Modern architectures prefer S3 + Spark/Trino over HDFS + MapReduce.

**Q9: What is a data lake vs data warehouse vs lakehouse?**
Data Lake: Raw data in any format (S3/HDFS), schema-on-read, cheap storage. Data Warehouse: Structured, schema-on-write, optimized for queries (Snowflake, Redshift). Lakehouse: Best of both -- raw storage with warehouse features (ACID, schema enforcement) via Delta Lake/Iceberg.

**Q10: How does data replication work in HDFS?**
Each block is replicated to multiple DataNodes (default 3). Rack-aware placement: first replica on local node, second on different rack, third on same rack as second. NameNode detects node failures via heartbeats and triggers re-replication.

**Q11: What is block size in HDFS and why is it 128MB?**
128MB (vs 4KB in regular filesystems) to minimize NameNode metadata and reduce seek time. Fewer blocks = less metadata. Large blocks are efficient for sequential reads of large files. Can be configured per-file.

**Q12: How would you process a 1TB CSV file?**
Load into Spark DataFrame (or PySpark), which automatically partitions across cluster nodes. Apply transformations (filter, join, aggregate) using DataFrame API. Spark's lazy evaluation builds an optimized execution plan. Write results to S3/Snowflake in Parquet format.

**Q13: What is AWS EMR and how does it relate to Hadoop?**
EMR (Elastic MapReduce) is AWS's managed Hadoop/Spark service. It runs Spark, Hive, Trino, HBase on auto-scaling clusters using YARN. Data lives in S3 instead of HDFS (decoupled storage). You pay only when clusters are running.

**Q14: What is the CAP theorem and how does HDFS handle it?**
CAP: Consistency, Availability, Partition tolerance -- pick 2. HDFS prioritizes Consistency and Partition tolerance (CP). If NameNode is unavailable, writes fail (sacrifices availability). Data is always consistent across replicas.

**Q15: How does Trino handle large joins across data sources?**
Trino's cost-based optimizer decides join strategies: broadcast join (small table sent to all workers), hash join (partition both tables by join key), or merge join. For cross-source joins, data is pulled from connectors into Trino workers' memory.

**Q16: What are the limitations of Hadoop?**
(1) MapReduce is slow (disk-based). (2) Complex to operate and maintain. (3) Not good for real-time processing. (4) NameNode is a bottleneck. (5) High ops cost for on-premise clusters. (6) Being replaced by cloud-native solutions (Spark + S3 + Databricks).

**Q17: Explain the medallion architecture.**
Bronze (raw ingestion) → Silver (cleansed, enriched) → Gold (business-ready aggregates). Used in Databricks/Delta Lake. Each layer adds quality and structure. This is the modern replacement for Hadoop ETL pipelines.

**Q18: What is partition pruning and why does it matter?**
Only reading relevant partitions instead of scanning the entire dataset. Example: data partitioned by date, query filters date='2025-01-01' → only reads that partition. Massively reduces I/O and query time. Works in Spark, Hive, Trino, and Snowflake.

**Q19: How does your PySpark experience relate to Hadoop?**
PySpark is the Python API for Apache Spark, which runs on YARN (Hadoop's resource manager). Spark replaced MapReduce as the processing engine but still uses Hadoop's ecosystem (YARN, HDFS). My EMR experience at MathCo runs Spark on YARN with S3 as storage.

**Q20: What is Apache Iceberg/Delta Lake and how does it improve on HDFS?**
Table formats that add ACID transactions, schema evolution, time travel, and partition evolution to data lake storage (S3/HDFS). Delta Lake (Databricks) and Iceberg (open standard) are the lakehouse foundations. They solve Hadoop's "data swamp" problem.

---

## Sources
- [Edureka: Top 50 Hadoop Interview Questions 2025](https://www.edureka.co/blog/interview-questions/top-hadoop-interview-questions)
- [DataCamp: Top 24 Hadoop Interview Questions 2026](https://www.datacamp.com/blog/hadoop-interview-questions)
- [MentorCruise: 80 Hadoop Interview Questions 2026](https://mentorcruise.com/questions/hadoop/)
- [InterviewBit: Top Hadoop Interview Questions 2025](https://www.interviewbit.com/hadoop-interview-questions/)
