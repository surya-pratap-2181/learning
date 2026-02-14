---
title: "Data Engineering"
layout: default
parent: "Interview Practice & Career"
nav_order: 7
---

# Data Engineering: PySpark, Databricks & Snowflake Interview Guide (2025-2026)
## For AI Engineers with Production Data Pipeline Experience

---

# TABLE OF CONTENTS
1. Data Engineering for AI Engineers
2. Explaining to a Layman
3. PySpark Deep Dive
4. Databricks Platform
5. Delta Lake
6. Snowflake Architecture
7. ETL Pipeline Design
8. Data Quality
9. AWS Data Services (EMR, Glue)
10. Performance Optimization
11. Interview Questions (30+)
12. Code Examples

---

# SECTION 1: DATA ENGINEERING FOR AI ENGINEERS

Data engineering is the foundation of any AI system. Without clean, reliable, scalable data pipelines, ML models and RAG systems fail.

**Why it matters for AI:**
- RAG systems need clean, chunked, embedded documents
- ML models need feature-engineered training data
- Analytics platforms need aggregated, validated data
- LLM applications need curated, fresh context data

> 🔵 **YOUR EXPERIENCE**: At MathCo, you were Senior Data Engineer working on Stellantis and AbbVie projects. You built data pipelines with PySpark, Databricks, Snowflake, achieving 70% runtime improvement on pharmaceutical data workflows.

---

# SECTION 2: EXPLAINING TO A LAYMAN

> Data engineering is like building the plumbing for a building. The AI is like the beautiful kitchen faucet (what users see), but behind the walls, there are pipes, pumps, and filters that clean and deliver the water. Without good plumbing (data engineering), the faucet (AI) doesn't work. Data engineers build those pipes that move, clean, and transform raw data into something useful.

---

# SECTION 3: PYSPARK DEEP DIVE

## 3.1 Core Concepts

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("AI Data Pipeline") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Read data
df = spark.read.parquet("s3://bucket/data/")

# Transformations (lazy!)
result = (
    df
    .filter(F.col("status") == "active")
    .withColumn("year", F.year("created_at"))
    .groupBy("region", "year")
    .agg(
        F.count("*").alias("count"),
        F.avg("revenue").alias("avg_revenue"),
        F.sum("revenue").alias("total_revenue")
    )
    .orderBy(F.desc("total_revenue"))
)

# Action (triggers execution!)
result.show()
```

## 3.2 Key Concepts

| Concept | Description |
|---------|-------------|
| **RDD** | Low-level distributed dataset (rarely used directly now) |
| **DataFrame** | Structured, columnar (like pandas but distributed) |
| **Transformations** | Lazy operations (filter, map, join) - build execution plan |
| **Actions** | Trigger execution (show, collect, write, count) |
| **Partitioning** | How data is split across workers |
| **Caching** | Store intermediate results in memory |
| **Catalyst Optimizer** | Automatically optimizes execution plans |
| **AQE** | Adaptive Query Execution - runtime optimization |

## 3.3 Joins and Optimization

```python
# Broadcast join (small table < 10MB)
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_df), "key")

# Repartition for better parallelism
df = df.repartition(200, "partition_key")

# Coalesce to reduce partitions (no shuffle)
df = df.coalesce(10)

# Cache for repeated access
df.cache()  # or df.persist(StorageLevel.MEMORY_AND_DISK)
```

## 3.4 Window Functions

```python
# Rank within groups
window = Window.partitionBy("department").orderBy(F.desc("salary"))
df = df.withColumn("rank", F.rank().over(window))
df = df.withColumn("running_total", F.sum("revenue").over(
    Window.partitionBy("region").orderBy("date").rowsBetween(Window.unboundedPreceding, 0)
))
```

## 3.5 PySpark for AI/ML

```python
# Feature engineering
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer

# String to numeric
indexer = StringIndexer(inputCol="category", outputCol="category_idx")

# Assemble features
assembler = VectorAssembler(inputCols=["age", "income", "category_idx"], outputCol="features")

# Scale
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler])
model = pipeline.fit(train_df)
result = model.transform(test_df)
```

> 🔵 **YOUR EXPERIENCE**: At AbbVie (MathCo), you leveraged Dataiku and PySpark to engineer and optimize high-performance ETL workflows. At Stellantis, you designed ML-ready data pipelines.

---

# SECTION 4: DATABRICKS PLATFORM

## 4.1 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATABRICKS                          │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │            Control Plane                       │   │
│  │  (Workspace, Notebooks, Jobs, Unity Catalog)  │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │            Data Plane (your cloud)             │   │
│  │                                                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Clusters  │  │  Delta   │  │   MLflow  │   │   │
│  │  │ (Spark)   │  │  Lake    │  │           │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘   │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 4.2 Key Components

| Component | Description |
|-----------|-------------|
| **Workspace** | Notebooks, repos, experiments |
| **Clusters** | Managed Spark compute |
| **Unity Catalog** | Unified governance (tables, models, volumes) |
| **Delta Lake** | ACID lakehouse storage |
| **MLflow** | Experiment tracking, model registry |
| **Workflows** | Job orchestration |
| **SQL Warehouse** | Serverless SQL compute |

---

# SECTION 5: DELTA LAKE

Delta Lake provides ACID transactions on data lakes:

| Feature | Description |
|---------|-------------|
| **ACID Transactions** | Atomic writes, consistent reads |
| **Time Travel** | Query data as of any version/timestamp |
| **Schema Evolution** | Add/modify columns without rewrite |
| **Z-Ordering** | Optimize file layout for specific columns |
| **MERGE** | Upsert (insert + update) operations |
| **Vacuum** | Clean up old file versions |

```python
# MERGE (upsert) - very common in ETL
deltaTable.alias("target").merge(
    updates.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# Time travel
df = spark.read.format("delta").option("versionAsOf", 5).load("/data/table")
df = spark.read.format("delta").option("timestampAsOf", "2025-01-01").load("/data/table")

# Z-ordering for query optimization
spark.sql("OPTIMIZE my_table ZORDER BY (region, date)")
```

> 🔵 **YOUR EXPERIENCE**: At MathCo, you worked with Databricks for ML-ready data pipelines and Delta Lake for reliable data storage.

---

# SECTION 6: SNOWFLAKE ARCHITECTURE

## 6.1 Three-Layer Architecture

```
┌────────────────────────────────────┐
│        Cloud Services Layer         │  ← Authentication, metadata,
│   (Query optimization, security)    │     query optimization, RBAC
├────────────────────────────────────┤
│        Compute Layer                │  ← Virtual Warehouses
│   (Virtual Warehouses - XS to 6XL) │     (independent, scalable)
├────────────────────────────────────┤
│        Storage Layer                │  ← Micro-partitions
│   (Columnar, compressed,           │     (auto-clustered)
│    micro-partitioned)               │
└────────────────────────────────────┘
```

## 6.2 Key Features

| Feature | Description |
|---------|-------------|
| **Virtual Warehouses** | Independent compute clusters, scale up/out |
| **Micro-partitions** | 50-500MB compressed columnar files |
| **Time Travel** | Query historical data (up to 90 days) |
| **Streams** | Track changes (CDC) on tables |
| **Tasks** | Scheduled SQL/procedure execution |
| **Snowpark** | Python/Java/Scala DataFrame API |
| **Data Sharing** | Share data across accounts without copying |

```sql
-- Virtual warehouse management
CREATE WAREHOUSE analytics_wh SIZE = 'MEDIUM' AUTO_SUSPEND = 300 AUTO_RESUME = TRUE;

-- Time travel
SELECT * FROM customers AT(OFFSET => -3600);  -- 1 hour ago
SELECT * FROM customers BEFORE(STATEMENT => 'query_id');

-- Streams (CDC)
CREATE STREAM customer_changes ON TABLE customers;
SELECT * FROM customer_changes;  -- See inserts, updates, deletes

-- Tasks (scheduling)
CREATE TASK daily_etl WAREHOUSE = etl_wh SCHEDULE = 'USING CRON 0 2 * * * UTC'
AS INSERT INTO summary SELECT ... FROM raw_data;
```

> 🔵 **YOUR EXPERIENCE**: At Stellantis (MathCo), you designed ML-ready data pipelines in Snowflake and Databricks. At AbbVie, you extracted data from Snowflake and loaded into CDL infrastructure.

---

# SECTION 7: ETL PIPELINE DESIGN

## 7.1 ETL vs ELT

| Aspect | ETL | ELT |
|--------|-----|-----|
| **Transform where?** | Before loading (in pipeline) | After loading (in warehouse) |
| **Best for** | Complex transforms, data quality | Cloud warehouses (Snowflake, BQ) |
| **Tools** | PySpark, Dataiku, Airflow | dbt, Snowpark, SQL |

## 7.2 Medallion Architecture (Bronze-Silver-Gold)

```
Raw Data → Bronze (raw, as-is) → Silver (cleaned, validated) → Gold (business-ready)

Bronze: Raw ingestion, minimal processing, append-only
Silver: Deduplication, type casting, null handling, joins
Gold:   Business aggregations, features, ML-ready
```

```python
# Bronze: Raw ingestion
bronze_df = spark.read.json("s3://raw/events/")
bronze_df.write.format("delta").mode("append").save("/bronze/events/")

# Silver: Clean and transform
silver_df = (
    spark.read.format("delta").load("/bronze/events/")
    .dropDuplicates(["event_id"])
    .filter(F.col("timestamp").isNotNull())
    .withColumn("date", F.to_date("timestamp"))
)
silver_df.write.format("delta").mode("overwrite").save("/silver/events/")

# Gold: Business aggregations
gold_df = (
    spark.read.format("delta").load("/silver/events/")
    .groupBy("date", "region")
    .agg(F.count("*").alias("events"), F.countDistinct("user_id").alias("users"))
)
gold_df.write.format("delta").mode("overwrite").save("/gold/daily_metrics/")
```

> 🔵 **YOUR EXPERIENCE**: At AbbVie (MathCo), you designed scalable, platform-independent data pipelines transforming raw data into enterprise-ready insights. Your 70% runtime improvement on the orchestration system is a great example.

---

# SECTION 8: DATA QUALITY

```python
# Great Expectations style validation
from pyspark.sql import functions as F

def validate_data(df, table_name):
    checks = {
        "row_count": df.count() > 0,
        "no_null_ids": df.filter(F.col("id").isNull()).count() == 0,
        "valid_dates": df.filter(F.col("date") > "2030-01-01").count() == 0,
        "unique_ids": df.count() == df.dropDuplicates(["id"]).count(),
    }
    failures = {k: v for k, v in checks.items() if not v}
    if failures:
        raise DataQualityError(f"Failed checks for {table_name}: {failures}")
    return True
```

> 🔵 **YOUR EXPERIENCE**: At AbbVie, you implemented automated data quality frameworks to proactively validate data integrity at enterprise scale.

---

# SECTION 9: AWS DATA SERVICES

| Service | Purpose | When to Use |
|---------|---------|-------------|
| **EMR** | Managed Spark/Hadoop | PySpark jobs, large-scale processing |
| **Glue** | Serverless ETL | Simple transforms, catalog |
| **Athena** | Serverless SQL on S3 | Ad-hoc queries on data lake |
| **Redshift** | Data warehouse | BI, analytics workloads |
| **Lake Formation** | Data lake governance | Access control, catalog |
| **Kinesis** | Real-time streaming | Event processing |

> 🔵 **YOUR EXPERIENCE**: You have production experience with AWS EMR for PySpark jobs (AbbVie project) and extensive AWS experience (EC2, S3, Lambda, RDS, SES, EMR).

---

# SECTION 10: PERFORMANCE OPTIMIZATION

| Technique | Impact | When |
|-----------|--------|------|
| **Partitioning** | 10-100x speedup | Filter on partition column |
| **Z-ordering** | 5-20x speedup | Queries on specific columns |
| **Broadcast join** | Eliminate shuffle | Small table < 10MB |
| **Predicate pushdown** | Reduce I/O | Parquet/Delta filters |
| **Caching** | Avoid recomputation | Repeated reads |
| **AQE** | Auto-optimization | Always enable |
| **Coalesce** | Reduce small files | After filtering |

---

# SECTION 11: INTERVIEW QUESTIONS (30+)

## PySpark

**Q1: Explain transformations vs actions in Spark.**
Transformations are lazy (build execution plan): filter, map, join, groupBy. Actions trigger execution: show, collect, count, write. Lazy evaluation enables optimization.

**Q2: What is the difference between repartition and coalesce?**
Repartition: full shuffle, can increase or decrease partitions. Coalesce: no shuffle (only decreases), moves data to fewer partitions. Use coalesce to reduce partitions after filtering.

**Q3: Explain broadcast join.**
When one table is small (<10MB), broadcast it to all workers. Avoids expensive shuffle. Use: broadcast(small_df). Spark auto-broadcasts if spark.sql.autoBroadcastJoinThreshold is set.

**Q4: What is AQE (Adaptive Query Execution)?**
Runtime optimization: coalesces small shuffle partitions, converts sort-merge joins to broadcast if data is small, optimizes skew joins. Enable: spark.sql.adaptive.enabled=true.

**Q5: How do you handle data skew?**
Salting (add random prefix to skewed key), broadcast join for small tables, AQE skew join handling, repartition with more granular key.

**Q6: Explain window functions in PySpark.**
Compute values across a "window" of rows: rank, running totals, moving averages. Define with Window.partitionBy().orderBy(). Common: rank(), row_number(), lag(), lead(), sum().over().

**Q7: What is the Catalyst optimizer?**
Spark's query optimization engine. Phases: Analysis → Logical optimization → Physical planning → Code generation. Automatically: pushes filters down, optimizes joins, eliminates unnecessary columns.

**Q8: How do you handle null values?**
dropna(), fillna(), coalesce() function, isNull()/isNotNull() filters. Strategy depends on use case: drop for ML training, fill for reporting.

## Databricks

**Q9: What is Unity Catalog?**
Unified governance layer for all Databricks assets: tables, views, models, volumes, functions. Provides: RBAC, data lineage, audit logs, cross-workspace access control.

**Q10: Explain the medallion architecture.**
Bronze (raw, as-is) → Silver (cleaned, validated, deduplicated) → Gold (business aggregations, ML features). Each layer has higher data quality and is more purpose-specific.

**Q11: What is Delta Lake and why is it important?**
Open-source storage layer providing ACID transactions on data lakes. Key features: time travel, schema evolution, MERGE (upsert), Z-ordering. Enables reliable data lakes (lakehouse).

**Q12: How does MERGE work in Delta Lake?**
Upsert operation: MERGE INTO target USING source ON condition WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT. Atomic, supports multiple conditions.

**Q13: What is Z-ordering?**
Data file optimization that co-locates related data. If queries frequently filter on "region" and "date", Z-ordering those columns puts related data in the same files, reducing I/O.

## Snowflake

**Q14: Explain Snowflake's architecture.**
Three layers: Storage (micro-partitions, columnar), Compute (virtual warehouses, independent), Services (query optimization, metadata, security). Storage and compute scale independently.

**Q15: What are virtual warehouses?**
Independent compute clusters. Can have multiple (analytics, ETL, ML). Each auto-suspends/resumes. Scale: XS to 6XL. Don't share resources - no contention.

**Q16: Explain micro-partitions.**
50-500MB compressed columnar files. Snowflake auto-clusters data. Pruning skips irrelevant partitions. No manual partitioning needed (unlike Spark).

**Q17: What are Snowflake Streams and Tasks?**
Streams: CDC (Change Data Capture) on tables - track inserts/updates/deletes. Tasks: scheduled SQL/procedures. Together enable: incremental ETL pipelines.

**Q18: What is Snowpark?**
Python/Java/Scala DataFrame API for Snowflake. Write transformations in Python, execute on Snowflake compute. Enables data scientists to use familiar syntax.

## General Data Engineering

**Q19: ETL vs ELT - when to use which?**
ETL: transform before loading (PySpark, Dataiku). ELT: load first, transform in warehouse (dbt, Snowpark). ELT preferred for cloud warehouses. ETL for complex, multi-source pipelines.

**Q20: How do you handle slowly changing dimensions (SCD)?**
Type 1: Overwrite (no history). Type 2: Add new row with effective dates (most common). Type 3: Add new column for previous value. Delta Lake MERGE supports all types.

**Q21: How do you ensure data quality?**
Validation rules (null checks, range checks, uniqueness), automated testing (Great Expectations), data contracts, monitoring dashboards, alerting on anomalies.

**Q22: How do you design for idempotency?**
Same operation run multiple times produces same result. Use: MERGE instead of INSERT, overwrite partitions, deduplicate on load, use deterministic IDs.

**Q23: How do you handle late-arriving data?**
Reprocessing windows, watermarking in streaming, partition overwrite for batch, reconciliation jobs, separate late-data pipelines.

**Q24: Explain data lineage and why it matters.**
Track where data comes from, how it's transformed, where it goes. Critical for: debugging, compliance (GDPR), impact analysis. Tools: Unity Catalog, OpenLineage.

**Q25: How do you optimize PySpark job costs on cloud?**
Right-size clusters, use spot instances (EMR), auto-scaling, caching, broadcast joins, partition pruning, avoid collect() on large data.

**Q26: What is a data contract?**
Agreement between data producer and consumer on schema, quality, SLA. Defines: columns, types, constraints, freshness. Prevents breaking changes.

**Q27: How do you monitor data pipelines?**
Execution metrics (runtime, row counts), data quality metrics, freshness checks, alerting on failures/anomalies. Tools: Datadog, CloudWatch, custom dashboards.

**Q28: Explain incremental processing vs full refresh.**
Incremental: process only new/changed data (faster, cheaper). Full refresh: reprocess everything (simpler, safer). Use incremental for large tables, full refresh for small or when needed.

**Q29: How do you handle schema evolution?**
Delta Lake: mergeSchema option auto-adds new columns. Snowflake: ALTER TABLE ADD COLUMN. Always: backward compatible changes, version schemas, test downstream impacts.

**Q30: How do you build ML-ready feature pipelines?**
Transform raw data → feature engineering → feature store. PySpark for large-scale transforms. Ensure: point-in-time correctness, feature freshness, reproducibility.

---

## Sources
- [DataCamp: PySpark Interview Questions 2026](https://www.datacamp.com/blog/pyspark-interview-questions)
- [ProjectPro: PySpark Interview 2025](https://www.projectpro.io/article/pyspark-interview-questions-and-answers/520)
- [DataVidhya: Databricks Interview Questions](https://datavidhya.com/blog/databricks-data-engineering-interview-questions)
- [Data Engineer Academy: PySpark Questions 2025](https://dataengineeracademy.com/module/top-15-pyspark-questions-to-master-for-data-engineer-interviews-updated-2025/)
