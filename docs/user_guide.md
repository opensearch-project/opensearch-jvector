# OpenSearch JVector Plugin — User Guide

## Table of Contents

- [Overview](#overview)
  - [What is JVector?](#what-is-jvector)
  - [Why Use JVector?](#why-use-jvector)
  - [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Verify Installation](#verify-installation)
  - [[Optional] Download and install Jvector compatible neural search](#optional-download-and-install-jvector-compatible-neural-search)
  - [Running the Demo](#running-the-demo)
  - [Your First Index and Search](#your-first-index-and-search)
- [Index Management](#index-management)
  - [Creating Indices](#creating-indices)
  - [Index Configuration](#index-configuration)
  - [Indexing Data](#indexing-data)
  - [Force Merge Operations](#force-merge-operations)
- [Search Operations](#search-operations)
  - [Basic k-NN Search](#basic-k-nn-search)
  - [Search with Filters](#search-with-filters)
  - [Search Parameters](#search-parameters)
  - [Advanced Search](#advanced-search)
  - [Performance Monitoring](#performance-monitoring)
- [Performance Optimization](#performance-optimization)
  - [Index-time Optimization](#index-time-optimization)
  - [Query-time Optimization](#query-time-optimization)
  - [Memory Management](#memory-management)
- [Advanced Topics](#advanced-topics)
  - [Product Quantization (PQ)](#product-quantization-pq)
  - [DiskANN Mode](#diskann-mode)
  - [MMR Search](#mmr-search)
  - [Derived Source](#derived-source)
- [Reference](#reference)
  - [Index Settings Reference](#index-settings-reference)
  - [Mapping Parameters Reference](#mapping-parameters-reference)
  - [Space Types](#space-types)

---

## Overview

### What is JVector?

The OpenSearch JVector plugin is a pure Java implementation of vector similarity search that enables you to perform approximate nearest neighbor (ANN) search on billions of documents. It leverages the [JVector library](https://github.com/jbellis/jvector) to provide high-performance vector search capabilities directly within OpenSearch, replacing the default KNN engine with a DiskANN implementation.

### Why Use JVector?

**High-Level Benefits:**
- **Scalable**: Handle billions of documents across thousands of dimensions using DiskANN
- **Fast**: Pure Java implementation with minimal overhead and SIMD acceleration
- **Efficient Updates**: Incremental merges that extend existing graphs rather than rebuilding from scratch

### Key Features

1. **Higher Throughput**: Thread-safe concurrent inserts during indexing
2. **Lower Search Latency**: Quantized graph traversal for faster queries
3. **Incremental Merges**: Update large graphs without full rebuilds — dramatically faster merge times
4. **DiskANN Implementation**: RAM-efficient ANN search optimized for disk-based operations
5. **Product Quantization (PQ)**: Compress vectors to significantly reduce memory usage
6. **SIMD Support**: Hardware-accelerated vector operations

The plugin exposes the same `knn_vector` field type and `knn` query used by the standard KNN plugin, so switching is mostly a matter of specifying `"engine": "jvector"` in your mapping.

---

## Quick Start

### Prerequisites

- **OpenSearch**: Version 3.5 or later
- **Java**: JDK 21 or later (required by OpenSearch 3.5)
- **Memory**: Sufficient RAM for your vector dataset (see [Memory Management](#memory-management))
- **Disk Space**: Adequate storage for indices and vector data with buffer for merging segments

**Important**: The JVector plugin replaces `opensearch-knn`. Both cannot be active at the same time.

### Installation

#### Option 1: Install from Maven Repository

```bash
# Navigate to your OpenSearch directory
cd /path/to/opensearch

# Remove existing k-NN and neural-search plugins (if installed)
bin/opensearch-plugin remove opensearch-neural-search
bin/opensearch-plugin remove opensearch-knn

# Download and install JVector plugin
curl https://repo1.maven.org/maven2/org/opensearch/plugin/opensearch-jvector-plugin/3.5.0.0/opensearch-jvector-plugin-3.5.0.0.zip -o opensearch-jvector-plugin.zip
bin/opensearch-plugin install file://`pwd`/opensearch-jvector-plugin.zip

# Start OpenSearch
bin/opensearch
```

#### Option 2: Docker Installation

Create a `Dockerfile`:

```dockerfile
FROM opensearchproject/opensearch:3.5.0

# Remove default KNN plugin
RUN /usr/share/opensearch/bin/opensearch-plugin remove opensearch-neural-search && \
    /usr/share/opensearch/bin/opensearch-plugin remove opensearch-knn

# Install JVector plugin
RUN curl https://repo1.maven.org/maven2/org/opensearch/plugin/opensearch-jvector-plugin/3.5.0.0/opensearch-jvector-plugin-3.5.0.0.zip -o opensearch-jvector-plugin.zip && \
    /usr/share/opensearch/bin/opensearch-plugin install --batch file://`pwd`/opensearch-jvector-plugin.zip
```

Build and run:

```bash
docker build -t opensearch-jvector .
docker run -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=YourStrongPassword123!" \
  opensearch-jvector
```

#### Option 3: Development Build

If you're building from source or developing the plugin:

```bash
# Clone the repository
git clone https://github.com/opensearch-project/opensearch-jvector.git
cd opensearch-jvector

# Build and run (includes plugin installation)
./gradlew run
```

Wait until the log shows:
```
[INFO][o.e.h.AbstractHttpServerTransport] publish_address {127.0.0.1:9200}
[INFO][o.e.n.Node] started
```

### Verify Installation

Check that the plugin is installed:

```bash
curl -X GET "http://localhost:9200/_cat/plugins?v"
```

Expected output should include:

```
name       component           version
node-1     opensearch-jvector  3.5.0.0
```

**Security Notes:**
- OpenSearch 3.x comes with security enabled by default (HTTPS + authentication required)
- Default username is `admin`, password is set via `OPENSEARCH_INITIAL_ADMIN_PASSWORD`
- For development with self-signed certificates, add `--insecure` or `-k` flag to curl commands
- For production, configure proper SSL certificates

### [Optional] Download and install Jvector compatible neural search

If you want to use neural search capabilities with JVector, you can install the JVector-compatible neural search plugin from:
**https://github.com/IBM/neural-search-jvector/releases**

This plugin provides neural search functionality that works with the JVector engine.

---

### Running the Demo

The repository includes a comprehensive demo script that walks through all core operations.

```bash
./scripts/demo_full.sh
```

The script will:
1. Check cluster health and confirm the jVector plugin is installed
2. Create a `jvector-demo` index with an 8-dimensional `cosinesimil` vector field
3. Bulk-index 10 movie documents with genre, year, and embeddings
4. Run a basic KNN search (sci-fi/action query vector)
5. Run a filtered KNN search (restricted to `genre=action`)
6. Run a tuned KNN search with `overquery_factor=10`
7. Force merge to 1 segment and verify consistency
8. Print jVector node stats
9. Delete the demo index

**Demo Options:**

| Flag | Description |
|---|---|
| `-h HOST` | OpenSearch host:port (default: `localhost:9200`) |
| `-u USER` | Basic-auth username |
| `-p PASS` | Basic-auth password |
| `-s` | Use HTTPS instead of HTTP |
| `-x` | Keep the demo index after the run |

**Examples:**

```bash
# Default (localhost, no auth)
./scripts/demo_full.sh

# Remote cluster with auth over HTTPS
./scripts/demo_full.sh -h my-cluster:9200 -u admin -p secret -s

# Keep the index for exploration
./scripts/demo_full.sh -x
```

### Your First Index and Search

#### Step 1: Create an Index

Create an index with an 8-dimensional vector field for this simple example:

```bash
curl -X PUT "http://localhost:9200/my-vector-index" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true,
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 8,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "l2",
          "parameters": {
            "m": 16,
            "ef_construction": 100
          }
        }
      },
      "title": {
        "type": "text"
      }
    }
  }
}'
```

Key points:
- `index.knn: true` — required to activate the KNN plugin on this index
- `method.engine: "jvector"` — selects the jVector engine
- `method.name: "disk_ann"` — the only available algorithm for jVector
- `dimension` — must match the dimension of your embedding model exactly (8 in this example)

#### Step 2: Index Documents

```bash
# Index a single document
curl -X POST "http://localhost:9200/my-vector-index/_doc/1" \
  -H "Content-Type: application/json" -d '
{
  "my_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  "title": "First document"
}'

# Index another document
curl -X POST "http://localhost:9200/my-vector-index/_doc/2" \
  -H "Content-Type: application/json" -d '
{
  "my_vector": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
  "title": "Second document"
}'

# Refresh the index
curl -X POST "http://localhost:9200/my-vector-index/_refresh"
```

#### Step 3: Search for Similar Vectors

```bash
curl -X POST "http://localhost:9200/my-vector-index/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 5,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
        "k": 5
      }
    }
  }
}'
```

**Congratulations!** You've created your first JVector index and performed a similarity search.

---

## Index Management

### Creating Indices

#### Minimal Example

```bash
curl -X PUT "http://localhost:9200/my-vectors" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 128,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "l2"
        }
      }
    }
  }
}'
```

#### Index with Multiple Vector Fields

You can have multiple vector fields in a single index:

```bash
curl -X PUT "http://localhost:9200/multi-vector-index" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "image_vector": {
        "type": "knn_vector",
        "dimension": 512,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "l2"
        }
      },
      "text_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "cosinesimil"
        }
      },
      "title": {
        "type": "text"
      },
      "category": {
        "type": "keyword"
      }
    }
  }
}'
```

#### Full Configuration Example

```bash
curl -X PUT "http://localhost:9200/optimized-index" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true,
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "category": { "type": "keyword" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "cosinesimil",
          "parameters": {
            "m": 32,
            "ef_construction": 200,
            "advanced.alpha": 1.2,
            "advanced.neighbor_overflow": 1.2,
            "advanced.hierarchy_enabled": false
          }
        }
      }
    }
  }
}'
```

### Index Configuration

#### DiskANN Method Parameters

**Index Build Time Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m` | integer | 16 | Maximum number of bi-directional connections per node. Higher values improve recall but increase memory usage and slow indexing. |
| `ef_construction` | integer | 100 | Candidate list size during graph construction. Higher values improve index quality but slow indexing. |
| `advanced.alpha` | float | 1.2 | Diversity factor for neighbor selection |
| `advanced.neighbor_overflow` | float | 1.2 | Overflow factor for neighbor lists |
| `advanced.hierarchy_enabled` | boolean | false | Enable hierarchical graph structure |

**Choosing M Parameter:**

| M Value | Memory Usage | Recall | Indexing Speed | Use Case |
|---------|--------------|--------|----------------|----------|
| 8-12 | Low | Good | Fast | Small datasets, memory-constrained |
| 16-24 | Medium | Better | Medium | General purpose (recommended) |
| 32-48 | High | Best | Slow | High-recall requirements |

**Recommendation:** Start with `m=16`, increase if recall is insufficient.

**Choosing ef_construction:**

| ef_construction | Index Quality | Indexing Speed | Use Case |
|-----------------|---------------|----------------|----------|
| 50-100 | Good | Fast | Development, testing |
| 100-200 | Better | Medium | Production (recommended) |
| 200-500 | Best | Slow | High-quality indices |

**Recommendation:** Use `ef_construction=100` for most cases, increase to 200 for better recall.

### Indexing Data

#### Single Document Indexing

```bash
curl -X PUT "http://localhost:9200/my-vectors/_doc/1" \
  -H "Content-Type: application/json" -d '
{
  "title": "Introduction to vector search",
  "category": "technology",
  "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}'
```

#### Bulk Indexing

For large datasets, use the bulk API for better performance:

```bash
curl -X POST "http://localhost:9200/_bulk" \
  -H "Content-Type: application/x-ndjson" -d '
{"index": {"_index": "my-vectors", "_id": "1"}}
{"title": "Document one", "embedding": [0.1, 0.2, 0.3, 0.4]}
{"index": {"_index": "my-vectors", "_id": "2"}}
{"title": "Document two", "embedding": [0.9, 0.8, 0.7, 0.6]}
{"index": {"_index": "my-vectors", "_id": "3"}}
{"title": "Document three", "embedding": [0.5, 0.5, 0.5, 0.5]}
'
```

**Optimal Batch Sizes:**
- **Small vectors (< 256 dims)**: 1000-5000 documents per batch
- **Medium vectors (256-768 dims)**: 500-1000 documents per batch
- **Large vectors (> 768 dims)**: 100-500 documents per batch

**For Java applications**, refer to the [OpenSearch Java Client Bulk Indexing Guide](https://github.com/opensearch-project/opensearch-java/blob/main/guides/bulk.md).

#### Incremental Updates

JVector's unique advantage is efficient incremental updates:

```bash
# Add new documents to existing index
curl -X POST "http://localhost:9200/my-vectors/_doc/10001" \
  -H "Content-Type: application/json" -d '
{
  "title": "New document",
  "embedding": [0.5, 0.6, 0.7, 0.8]
}'

# Update existing document
curl -X POST "http://localhost:9200/my-vectors/_update/1" \
  -H "Content-Type: application/json" -d '
{
  "doc": {
    "title": "Updated title",
    "embedding": [0.15, 0.25, 0.35, 0.45]
  }
}'
```

**Why JVector is Better for Updates:**
- Traditional HNSW requires full graph rebuild on merge
- JVector performs incremental merges, adding new vectors to existing graph
- Result: Significantly faster merge times for large indices

### Force Merge Operations

#### When to Force Merge

Force merge consolidates index segments and optimizes the vector graph:

- After bulk indexing large batches
- Before running performance benchmarks
- Periodically for production indices (e.g., nightly)
- After significant updates or deletes

#### Performing Force Merge

```bash
# Force merge to 1 segment (optimal for search performance)
curl -X POST "http://localhost:9200/my-vectors/_forcemerge?max_num_segments=1"
```

Wait for the operation to complete before querying — it runs synchronously and returns when done.

#### Incremental Merge Advantage

JVector performs an *incremental merge* — nodes from smaller segments are inserted into the existing graph rather than re-indexing from scratch. This makes merges on large indexes dramatically faster.

**Key Benefits:**
- Merge time grows more linearly with index size
- Faster updates for large indices
- Reduced resource consumption during merges

To disable the leading segment merge feature:

```json
"parameters": {
  "advanced.leading_segment_merge_disabled": true
}
```

#### Monitoring Merge Performance

Check merge statistics:

```bash
curl -X GET "http://localhost:9200/my-vectors/_stats/merge?pretty"
```

Get JVector-specific statistics:

```bash
curl -X GET "http://localhost:9200/_nodes/stats/indices/knn?pretty"
```

**Key Metrics:**
- `knn_graph_merge_time`: Time spent merging vector graphs
- `knn_quantization_training_time`: Time spent on quantization
- `knn_graph_merge_count`: Number of graph merges performed

---

## Search Operations

### Basic k-NN Search

Find the k most similar vectors:

```bash
curl -X POST "http://localhost:9200/my-vectors/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 5,
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 5
      }
    }
  }
}'
```

- `vector` — the query vector; must have the same dimension as the field
- `k` — number of nearest neighbors to return (max 10,000)

#### Returning Specific Fields

```bash
curl -X POST "http://localhost:9200/my-vectors/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 10,
  "_source": ["title", "category"],
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 10
      }
    }
  }
}'
```

### Search with Filters

Combine KNN with standard OpenSearch filters to restrict the candidate set:

```bash
curl -X POST "http://localhost:9200/my-vectors/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 5,
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 5,
        "filter": {
          "term": { "category": "technology" }
        }
      }
    }
  }
}'
```

**Pre-filtering with bool query:**

```bash
curl -X POST "http://localhost:9200/my-vectors/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "knn": {
            "embedding": {
              "vector": [0.1, 0.2, 0.3, 0.4],
              "k": 10
            }
          }
        }
      ],
      "filter": [
        {
          "term": {
            "category": "electronics"
          }
        },
        {
          "range": {
            "price": {
              "gte": 100,
              "lte": 500
            }
          }
        }
      ]
    }
  }
}'
```

**Post-filtering:**

```bash
curl -X POST "http://localhost:9200/my-vectors/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 10,
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 50
      }
    }
  },
  "post_filter": {
    "term": {
      "category": "electronics"
    }
  }
}'
```

### Search Parameters

#### Tuning Search Parameters

Use `method_parameters` to trade off latency against recall at query time:

```bash
curl -X POST "http://localhost:9200/my-vectors/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 10,
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 10,
        "method_parameters": {
          "overquery_factor": 10,
          "advanced.rerank_floor": 0.1
        }
      }
    }
  }
}'
```

| Parameter | Default | Effect |
|---|---|---|
| `overquery_factor` | `5` | Internal candidate list = `k × overquery_factor`. Higher values improve recall at the cost of latency. |
| `advanced.threshold` | `0.0` | Minimum similarity score for a candidate to enter the result set. |
| `advanced.rerank_floor` | `0.0` | Minimum score for a candidate to be re-ranked with full-precision vectors. |
| `advanced.use_pruning` | `false` | Enable graph traversal pruning to reduce nodes visited. |

**Guidelines:**
- Start with `overquery_factor=5` (default)
- Increase by 5-10 if recall is low
- Monitor latency impact
- Find the sweet spot for your use case

### Advanced Search

#### Hybrid Search (Vector + Text)

To combine vector similarity search with traditional text search, OpenSearch provides a dedicated [Hybrid Search](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) plugin. It properly normalizes and combines scores from different query types.

**Note:** The `neural-search` plugin currently has tight integration with the official k-NN plugin and does not support `opensearch-jvector` at the moment. There are ongoing efforts to address this gap.

#### Nested Field Search

Search vectors in nested documents:

```bash
# Index mapping with nested field
curl -X PUT "http://localhost:9200/nested-index" \
  -H "Content-Type: application/json" -d '
{
  "mappings": {
    "properties": {
      "products": {
        "type": "nested",
        "properties": {
          "product_vector": {
            "type": "knn_vector",
            "dimension": 4,
            "method": {
              "name": "disk_ann",
              "engine": "jvector",
              "space_type": "l2"
            }
          },
          "name": {"type": "text"}
        }
      }
    }
  }
}'

# Search nested vectors
curl -X POST "http://localhost:9200/nested-index/_search" \
  -H "Content-Type: application/json" -d '
{
  "query": {
    "nested": {
      "path": "products",
      "query": {
        "knn": {
          "products.product_vector": {
            "vector": [0.1, 0.2, 0.3, 0.4],
            "k": 5
          }
        }
      }
    }
  }
}'
```

### Performance Monitoring

#### JVector Statistics

Get detailed JVector statistics:

```bash
curl -X GET "http://localhost:9200/_nodes/stats/indices/knn?pretty"
```

**Key Statistics:**
- `knn_query_visited_nodes`: Total nodes visited during searches
- `knn_query_expanded_nodes`: Nodes expanded during searches
- `knn_query_expanded_base_layer_nodes`: Base layer nodes expanded
- `knn_graph_merge_time`: Time spent on graph merges
- `knn_quantization_training_time`: Time spent on quantization

#### Query Profiling

Enable profiling to understand query performance:

```bash
curl -X POST "http://localhost:9200/my-vectors/_search" \
  -H "Content-Type: application/json" -d '
{
  "profile": true,
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 10
      }
    }
  }
}'
```

For detailed information, see the [OpenSearch Query Profiling documentation](https://opensearch.org/docs/latest/api-reference/profile/).

---

## Performance Optimization

### Index-time Optimization

See [Index Configuration](#index-configuration) for detailed parameter tuning guidance.

#### Merge Frequency Tuning

Control how often segments are merged:

```bash
curl -X PUT "http://localhost:9200/my-vectors/_settings" \
  -H "Content-Type: application/json" -d '
{
  "index": {
    "merge.policy.max_merged_segment": "5gb",
    "merge.scheduler.max_thread_count": 1
  }
}'
```

**Best Practices:**
- Perform force merge after bulk indexing
- Use incremental merges for continuous updates
- Monitor merge times with JVector statistics

### Query-time Optimization

#### Caching Strategies

Enable query result caching:

```bash
curl -X PUT "http://localhost:9200/my-vectors/_settings" \
  -H "Content-Type: application/json" -d '
{
  "index.queries.cache.enabled": true
}'
```

**Note:** Vector queries are typically unique, so caching may have limited benefit.

#### Retrieving vector fields using docvalue_fields

You can retrieve `knn_vector` fields using `docvalue_fields` instead of the `_source`. This is faster because OpenSearch reads the vector directly from `doc_values` rather than parsing the full `_source` document.

Retrieving `knn_vector` fields from `doc_values` supports all vector data types (`float`, `byte`, and `binary`), all compression levels, for Lucene engine, `float` for JVector engine. You can use it on existing indexes without reindexing.

| Format | Description |
| :--- | :--- |
| `binary` (Default) | Returns vectors as Base64-encoded little-endian byte strings. Provides approximately 2x throughput improvement over the `array` format for JSON transport and reduces response payload size by 30--40%. |
| `array` | Returns vectors as JSON numeric arrays. |

```bash
curl -X PUT "http://localhost:9200/my-index/_search" \
  -H "Content-Type: application/json" -d '
{
  "query": { "knn": { "my_vector": { "vector": [...], "k": 10 } } },
  "docvalue_fields.field": "my_vector",
  "docvalue_fields.format": "array",
  "_source": false
}'
```

### Memory Management

#### Memory Estimation

**Without quantization:**
```
Memory = num_vectors × dimension × 4 bytes × 1.5 (overhead)
```

**With 8-bit quantization:**
```
Memory = num_vectors × dimension × 1 byte × 1.5
```

**With PQ (e.g., 96 subvectors):**
```
Memory = num_vectors × 96 bytes × 1.5
```

**Examples:**
- 1M vectors, 768 dims, no quantization: ~4.6 GB
- 1M vectors, 768 dims, 8-bit quantization: ~1.2 GB
- 1M vectors, 768 dims, PQ (96 subvectors): ~144 MB

#### OS Cache Considerations

JVector leverages OS file system cache:

**Best Practices:**
- Leave 20-30% of RAM for OS cache
- Monitor cache hit rates
- Use SSDs for better performance
- Consider memory-mapped files for large indices

---

## Advanced Topics

### Product Quantization (PQ)

PQ compresses vector storage by dividing each vector into subspaces and quantizing each independently.

```bash
curl -X PUT "http://localhost:9200/pq-index" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "l2",
          "parameters": {
            "m": 16,
            "ef_construction": 100,
            "advanced.num_pq_subspaces": 192,
            "advanced.min_batch_size_for_quantization": 1024
          }
        }
      }
    }
  }
}'
```

**Rules:**
- `num_pq_subspaces` must be ≤ `dimension` and ideally a divisor of `dimension`
- Quantization training is triggered automatically once the segment has at least `min_batch_size_for_quantization` documents
- More subspaces = less compression but better recall

**Default Values by Dimension:**

| Dimension | Default Subspaces |
|-----------|-------------------|
| 384 | 96 |
| 768 | 192 |
| 1536 | 192 |
| 3072 | 384 |

Lower values provide more compression but may reduce recall. The defaults are tuned for good recall/compression balance.

**Trade-offs:**
- Memory: Significant reduction (configurable compression ratio)
- Recall: ~90-95% of original (depends on configuration)
- Speed: Faster for large datasets
- **Use when:** Large datasets with memory constraints

### DiskANN Mode

For indexes that exceed available RAM, combine `mode: on_disk` with PQ:

```bash
curl -X PUT "http://localhost:9200/my-vectors-disk" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "mode": "on_disk",
        "compression_level": "8x",
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "cosinesimil",
          "parameters": {
            "m": 32,
            "ef_construction": 200,
            "advanced.num_pq_subspaces": 192,
            "advanced.min_batch_size_for_quantization": 2048
          }
        }
      }
    }
  }
}'
```

During search, jVector uses the compressed vectors for graph traversal and re-ranks the top candidates using full-precision vectors read from disk. Increase `overquery_factor` at query time to improve recall at the cost of more re-ranking disk reads.

Higher compression levels require larger `advanced.num_pq_subspaces` to maintain recall quality.

### MMR Search

For diverse result sets, see [docs/mmr_search.md](mmr_search.md).

### Derived Source

For derived source configuration and version-specific behavior, see [docs/derived_source.md](derived_source.md).

---

## Reference

### Index Settings Reference

| Setting | Default | Description |
|---|---|---|
| `index.knn` | — | Set to `true` to enable the KNN plugin on the index (required) |
| `index.knn.advanced.approximate_threshold` | `15000` | Minimum document count per segment before an approximate graph is built |
| `index.knn.disk.vector.shard_level_rescoring_disabled` | `false` | Disable shard-level re-scoring for on-disk vectors |
| `index.knn.derived_source.enabled` | `true` | Store vectors in derived source fields |

### Mapping Parameters Reference

#### Field-level

| Parameter | Required | Default | Description |
|---|---|---|---|
| `type` | yes | — | Must be `"knn_vector"` |
| `dimension` | yes | — | Integer 1–16,000 |
| `data_type` | no | `"float"` | Only `"float"` is supported |
| `mode` | no | `"in_memory"` | `"in_memory"` or `"on_disk"` |
| `compression_level` | no | `"1x"` | `"1x"` through `"64x"` |

#### `method` object

| Parameter | Required | Default | Description |
|---|---|---|---|
| `name` | yes | — | Must be `"disk_ann"` |
| `engine` | yes | — | Must be `"jvector"` |
| `space_type` | no | `"l2"` | See [Space Types](#space-types) |

#### `method.parameters`

| Parameter | Default | Description |
|---|---|---|
| `m` | `16` | Bi-directional link count per node. Higher values improve recall, increase index size. |
| `ef_construction` | `100` | Candidate list size during graph construction. Higher values improve recall, slow ingestion. |
| `advanced.alpha` | `1.2` | Diversity factor for neighbor selection |
| `advanced.neighbor_overflow` | `1.2` | Overflow factor for neighbor lists |
| `advanced.hierarchy_enabled` | `false` | Enable hierarchical graph structure |
| `advanced.num_pq_subspaces` | — | Number of PQ subspaces. Must be ≤ dimension. |
| `advanced.min_batch_size_for_quantization` | `1024` | Documents needed before quantization is trained |
| `advanced.leading_segment_merge_disabled` | `false` | Prevent leading segment from being rebuilt during force merge |

### Space Types

The `space_type` field controls which distance metric is used.

| Value | Distance metric | Use case |
|---|---|---|
| `l2` | Euclidean distance (L2 norm) | General-purpose; raw coordinates |
| `cosinesimil` | Cosine similarity | Text embeddings; direction matters more than magnitude |
| `innerproduct` | Dot product (inner product) | Embeddings where magnitude carries meaning (e.g., biencoder models) |
| `l1` | Manhattan distance | Robust to outliers |
| `linf` | Chebyshev distance | Maximum per-dimension deviation |

The default is `l2` when `space_type` is omitted.

**Inner Product Note:** Unlike cosine similarity which normalizes vectors, inner product preserves magnitude information, making it suitable for scenarios where vector length is meaningful.

---

## Additional Resources

- **Benchmarks**: See [README.md](../README.md#incremental-merges) for performance comparisons
- **Developer Guide**: See [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) for contributing to the plugin
- **Testing Scripts**: See [scripts/jvector_index_and_search/README.md](../scripts/jvector_index_and_search/README.md) for benchmarking tools
- **MMR Search**: See [docs/mmr_search.md](mmr_search.md) for diverse result sets
- **Derived Source**: See [docs/derived_source.md](derived_source.md) for configuration details