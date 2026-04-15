# OpenSearch JVector Plugin - User Guide

- [Overview](#overview)
  - [What is JVector?](#what-is-jvector)
  - [Why Use JVector?](#why-use-jvector)
  - [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Verify Installation](#verify-installation)
  - [Your First Index and Search](#your-first-index-and-search)
- [Index Management](#index-management)
  - [Creating Indices](#creating-indices)
  - [Index Configuration](#index-configuration)
  - [Indexing Data](#indexing-data)
  - [Force Merge Operations](#force-merge-operations)
- [Search Operations](#search-operations)
  - [Basic k-NN Search](#basic-k-nn-search)
  - [Search Parameters](#search-parameters)
  - [Advanced Search](#advanced-search)
  - [Performance Monitoring](#performance-monitoring)
- [Performance Optimization](#performance-optimization)
  - [Index-time Optimization](#index-time-optimization)
  - [Query-time Optimization](#query-time-optimization)
  - [Memory Management](#memory-management)
  - [Benchmarking](#benchmarking)

---

## Overview

### What is JVector?

OpenSearch JVector Plugin is a pure Java implementation of vector similarity search that enables you to perform approximate nearest neighbor (ANN) search on billions of documents. It leverages the [JVector library](https://github.com/jbellis/jvector) (as referenced in the [source code](../src/main/java/org/opensearch/knn/index/codec/jvector/JVector.java#L21)) to provide high-performance vector search capabilities directly within OpenSearch.

### Why Use JVector?

**High-Level Benefits:**
- **Scalable**: Handle billions of documents across thousands of dimensions using DiskANN
- **Fast**: Pure Java implementation with minimal overhead
- **Lightweight**: Self-contained, no native dependencies, builds in seconds

**Key Features:**
1. **DiskANN Implementation**: Pure Java ANN search optimized for disk-based operations and RAM-constrained environments
2. **Thread-Safe Indexing**: Concurrent writes with excellent scalability
3. **Incremental Merges**: Update large graphs without full rebuilds for faster updates
4. **Product Quantization (PQ)**: Compress vectors to significantly reduce memory usage with quantization refinement during merge
5. **SIMD Support**: Hardware-accelerated vector operations
6. **Fused ADC**: Advanced distance computation optimizations
7. **Cassandra Compatibility**: Easy data transfer between Cassandra and OpenSearch

---

## Quick Start

### Prerequisites

- **Java**: JDK 21 or later (required by OpenSearch 3.x)
- **Memory**: Sufficient RAM for your vector dataset (see [Memory Management](#memory-management))
- **Disk Space**: Adequate storage for indices and vector data

### Installation

#### Installing OpenSearch

If you don't have OpenSearch installed yet, choose one of these options:

**Option A: Download and Install OpenSearch**

Download OpenSearch 3.3.2 or later from the [OpenSearch downloads page](https://opensearch.org/downloads.html). Choose the distribution appropriate for your platform and OS.

Example for macOS:

```bash
# Download OpenSearch 3.3.2 (example for macOS)
curl -O https://artifacts.opensearch.org/releases/bundle/opensearch/3.3.2/opensearch-3.3.2-darwin-x64.tar.gz

# Extract
tar -xzf opensearch-3.3.2-darwin-x64.tar.gz
cd opensearch-3.3.2

# Start OpenSearch
./bin/opensearch
```

**Option B: Use Docker**

```bash
docker pull opensearchproject/opensearch:3.3.2
docker run -d -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=YourStrongPassword123!" \
  opensearchproject/opensearch:3.3.2
```

**Verify OpenSearch is Running:**

```bash
curl http://localhost:9200
# Should return cluster information
```

#### Installing JVector Plugin

**Important:** JVector plugin replaces the official OpenSearch k-NN plugin. You must remove the following plugins before installing JVector:
- `opensearch-knn` (official k-NN plugin)
- `opensearch-neural-search` (depends on opensearch-knn)

#### Option 1: Install from Maven Repository

```bash
# Navigate to your OpenSearch directory
cd /path/to/opensearch

# Remove existing k-NN and neural-search plugins (if installed)
bin/opensearch-plugin remove opensearch-neural-search
bin/opensearch-plugin remove opensearch-knn

# Download and install JVector plugin
curl https://repo1.maven.org/maven2/org/opensearch/plugin/opensearch-jvector-plugin/3.3.2.0/opensearch-jvector-plugin-3.3.2.0.zip -o opensearch-jvector-plugin.zip
bin/opensearch-plugin install file://`pwd`/opensearch-jvector-plugin.zip

# Start OpenSearch
bin/opensearch
```

#### Option 2: Docker Installation with JVector Plugin

Create a `Dockerfile`:

```dockerfile
FROM opensearchproject/opensearch:3.3.2

# Remove default KNN plugin
RUN /usr/share/opensearch/bin/opensearch-plugin remove opensearch-neural-search && \
    /usr/share/opensearch/bin/opensearch-plugin remove opensearch-knn

# Install JVector plugin
RUN curl https://repo1.maven.org/maven2/org/opensearch/plugin/opensearch-jvector-plugin/3.3.2.0/opensearch-jvector-plugin-3.3.2.0.zip -o opensearch-jvector-plugin.zip && \
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

### Verify Installation

Check that the plugin is installed:

```bash
# For local development (with security disabled)
curl -X GET "http://localhost:9200/_cat/plugins?v"

# For production (with security enabled - default)
curl -X GET "https://localhost:9200/_cat/plugins?v" \
  -u admin:admin \
  --insecure
```

Expected output should include:

```
name       component           version
node-1     opensearch-jvector  3.3.2.0
```

**Note:** Replace `admin:admin` with your actual credentials. Use `--insecure` only for development with self-signed certificates.

### Your First Index and Search

#### Step 1: Create an Index

Create an index with a 128-dimensional vector field:

```bash
curl -X PUT "localhost:9200/my-vector-index" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "index": {
      "knn": true,
      "number_of_shards": 1,
      "number_of_replicas": 0
    }
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 128,
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
}
'
```

#### Step 2: Index Some Documents

```bash
# Index document 1
curl -X POST "localhost:9200/my-vector-index/_doc/1" -H 'Content-Type: application/json' -d'
{
  "my_vector": [0.1, 0.2, 0.3, ..., 0.128],
  "title": "First document"
}
'

# Index document 2
curl -X POST "localhost:9200/my-vector-index/_doc/2" -H 'Content-Type: application/json' -d'
{
  "my_vector": [0.2, 0.3, 0.4, ..., 0.129],
  "title": "Second document"
}
'

# Refresh the index
curl -X POST "localhost:9200/my-vector-index/_refresh"
```

#### Step 3: Search for Similar Vectors

```bash
curl -X POST "localhost:9200/my-vector-index/_search" -H 'Content-Type: application/json' -d'
{
  "size": 5,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [0.15, 0.25, 0.35, ..., 0.128],
        "k": 5
      }
    }
  }
}
'
```

**Congratulations!** You've created your first JVector index and performed a similarity search.

---

## Index Management

### Creating Indices

#### Basic Index Creation

The simplest way to create a JVector index:

```bash
curl -X PUT "localhost:9200/basic-index" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "l2"
        }
      }
    }
  }
}
'
```

#### Index with Multiple Vector Fields

You can have multiple vector fields in a single index:

```bash
curl -X PUT "localhost:9200/multi-vector-index" -H 'Content-Type: application/json' -d'
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
      }
    }
  }
}
'
```

### Index Configuration

#### DiskANN Method Parameters

The `disk_ann` method supports several parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m` | integer | 16 | Maximum number of connections per node in the graph. Higher values = better recall but more memory |
| `ef_construction` | integer | 100 | Size of the dynamic candidate list during index construction. Higher values = better quality but slower indexing |
| `ef_search` | integer | 100 | Size of the dynamic candidate list during search. Can be overridden at query time |

**Example with custom parameters:**

```bash
curl -X PUT "localhost:9200/optimized-index" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "l2",
          "parameters": {
            "m": 32,
            "ef_construction": 200,
            "ef_search": 150
          }
        }
      }
    }
  }
}
'
```

#### Space Types

JVector supports the following distance metrics:

| Space Type | Description | Use Case |
|------------|-------------|----------|
| `l2` | Euclidean distance (L2 norm) | General purpose, geometric similarity |
| `cosinesimil` | Cosine similarity | Text embeddings, normalized vectors |


**Example with cosine similarity:**

```bash
curl -X PUT "localhost:9200/text-embeddings" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "text_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "cosinesimil",
          "parameters": {
            "m": 16,
            "ef_construction": 100
          }
        }
      }
    }
  }
}
'
```

#### Quantization Options

JVector supports Product Quantization (PQ) for memory efficiency:

**Product Quantization (PQ):**

```bash
curl -X PUT "localhost:9200/pq-index" -H 'Content-Type: application/json' -d'
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "l2",
          "parameters": {
            "m": 16,
            "ef_construction": 100,
            "compression": {
              "type": "product_quantization",
              "subvectors": 96,
              "bits_per_subvector": 8
            }
          }
        }
      }
    }
  }
}
'
```

### Indexing Data

#### Single Document Indexing

Index one document at a time:

```bash
curl -X POST "localhost:9200/my-index/_doc/1" -H 'Content-Type: application/json' -d'
{
  "vector_field": [0.1, 0.2, 0.3, ..., 0.768],
  "title": "Document title",
  "description": "Document description"
}
'
```

#### Bulk Indexing

For large datasets, use the bulk API for better performance:

```bash
curl -X POST "localhost:9200/_bulk" -H 'Content-Type: application/x-ndjson' -d'
{"index": {"_index": "my-index", "_id": "1"}}
{"vector_field": [0.1, 0.2, ..., 0.768], "title": "Doc 1"}
{"index": {"_index": "my-index", "_id": "2"}}
{"vector_field": [0.2, 0.3, ..., 0.769], "title": "Doc 2"}
{"index": {"_index": "my-index", "_id": "3"}}
{"vector_field": [0.3, 0.4, ..., 0.770], "title": "Doc 3"}
'
```

**Optimal Batch Sizes:**
- **Small vectors (< 256 dims)**: 1000-5000 documents per batch
- **Medium vectors (256-768 dims)**: 500-1000 documents per batch
- **Large vectors (> 768 dims)**: 100-500 documents per batch

**Java Example for Bulk Indexing:**

```java
import org.opensearch.client.opensearch.OpenSearchClient;
import org.opensearch.client.opensearch.core.BulkRequest;
import org.opensearch.client.opensearch.core.BulkResponse;
import org.opensearch.client.opensearch.core.bulk.BulkOperation;
import org.opensearch.client.json.jackson.JacksonJsonpMapper;
import org.opensearch.client.transport.OpenSearchTransport;
import org.opensearch.client.transport.rest_client.RestClientTransport;
import org.apache.http.HttpHost;
import org.opensearch.client.RestClient;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class BulkIndexExample {
    public static void main(String[] args) throws Exception {
        // Create OpenSearch client
        RestClient restClient = RestClient.builder(
            new HttpHost("localhost", 9200, "http")
        ).build();
        
        OpenSearchTransport transport = new RestClientTransport(
            restClient, new JacksonJsonpMapper()
        );
        OpenSearchClient client = new OpenSearchClient(transport);
        
        // Bulk index documents
        int batchSize = 1000;
        int totalDocs = 10000;
        int dimension = 768;
        Random random = new Random();
        
        for (int batch = 0; batch < totalDocs / batchSize; batch++) {
            List<BulkOperation> operations = new ArrayList<>();
            
            for (int i = 0; i < batchSize; i++) {
                int docId = batch * batchSize + i;
                float[] vector = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    vector[j] = random.nextFloat();
                }
                
                operations.add(BulkOperation.of(op -> op
                    .index(idx -> idx
                        .index("my-index")
                        .id(String.valueOf(docId))
                        .document(new Document(vector, "Document " + docId))
                    )
                ));
            }
            
            BulkResponse response = client.bulk(BulkRequest.of(b -> b
                .operations(operations)
            ));
            
            System.out.println("Batch " + batch + ": " +
                (batchSize - response.errors()) + " succeeded, " +
                response.errors() + " failed");
        }
        
        transport.close();
    }
    
    static class Document {
        public float[] vector_field;
        public String title;
        
        public Document(float[] vector, String title) {
            this.vector_field = vector;
            this.title = title;
        }
    }
}
```

#### Incremental Updates

JVector's unique advantage is efficient incremental updates:

```bash
# Add new documents to existing index
curl -X POST "localhost:9200/my-index/_doc/10001" -H 'Content-Type: application/json' -d'
{
  "vector_field": [0.5, 0.6, ..., 0.771],
  "title": "New document"
}
'

# Update existing document
curl -X POST "localhost:9200/my-index/_update/1" -H 'Content-Type: application/json' -d'
{
  "doc": {
    "vector_field": [0.15, 0.25, ..., 0.768],
    "title": "Updated title"
  }
}
'
```

**Why JVector is Better for Updates:**
- Traditional HNSW requires full graph rebuild on merge
- JVector performs incremental merges, adding new vectors to existing graph
- Result: Significantly faster merge times for large indices (see [benchmarks](../README.md#incremental-merges))

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
curl -X POST "localhost:9200/my-index/_forcemerge?max_num_segments=1"
```

#### Monitoring Merge Performance

Check merge statistics:

```bash
curl -X GET "localhost:9200/my-index/_stats/merge?pretty"
```

Get JVector-specific statistics:

```bash
curl -X GET "localhost:9200/_nodes/stats/indices/knn?pretty"
```

**Key Metrics:**
- `knn_graph_merge_time`: Time spent merging vector graphs
- `knn_quantization_training_time`: Time spent on quantization
- `knn_graph_merge_count`: Number of graph merges performed

#### Incremental Merge Advantage

JVector's incremental merge capability provides significant performance improvements over traditional HNSW implementations that require full graph rebuilds on every merge.

**Key Benefits:**
- Merge time grows more linearly with index size
- Faster updates for large indices
- Reduced resource consumption during merges

See [benchmarks](../README.md#incremental-merges) for detailed performance comparisons with graphs showing merge time scaling.

---

## Search Operations

### Basic k-NN Search

#### Simple k-NN Query

Find the k most similar vectors:

```bash
curl -X POST "localhost:9200/my-index/_search" -H 'Content-Type: application/json' -d'
{
  "size": 10,
  "query": {
    "knn": {
      "vector_field": {
        "vector": [0.1, 0.2, 0.3, ..., 0.768],
        "k": 10
      }
    }
  }
}
'
```

**Response:**

```json
{
  "took": 15,
  "hits": {
    "total": {"value": 10, "relation": "eq"},
    "max_score": 0.95,
    "hits": [
      {
        "_index": "my-index",
        "_id": "1",
        "_score": 0.95,
        "_source": {
          "vector_field": [...],
          "title": "Most similar document"
        }
      },
      ...
    ]
  }
}
```

#### Returning Specific Fields

```bash
curl -X POST "localhost:9200/my-index/_search" -H 'Content-Type: application/json' -d'
{
  "size": 10,
  "_source": ["title", "description"],
  "query": {
    "knn": {
      "vector_field": {
        "vector": [0.1, 0.2, ..., 0.768],
        "k": 10
      }
    }
  }
}
'
```

### Search Parameters

#### Tuning ef_search

Override the default `ef_search` parameter at query time:

```bash
curl -X POST "localhost:9200/my-index/_search" -H 'Content-Type: application/json' -d'
{
  "size": 10,
  "query": {
    "knn": {
      "vector_field": {
        "vector": [0.1, 0.2, ..., 0.768],
        "k": 10,
        "ef_search": 200
      }
    }
  }
}
'
```

**ef_search Trade-offs:**
- **Lower values (50-100)**: Faster search, lower recall
- **Medium values (100-200)**: Balanced performance
- **Higher values (200-500)**: Better recall, slower search

#### Filtering Results

**Pre-filtering (recommended):**

```bash
curl -X POST "localhost:9200/my-index/_search" -H 'Content-Type: application/json' -d'
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "knn": {
            "vector_field": {
              "vector": [0.1, 0.2, ..., 0.768],
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
}
'
```

**Post-filtering:**

```bash
curl -X POST "localhost:9200/my-index/_search" -H 'Content-Type: application/json' -d'
{
  "size": 10,
  "query": {
    "knn": {
      "vector_field": {
        "vector": [0.1, 0.2, ..., 0.768],
        "k": 50
      }
    }
  },
  "post_filter": {
    "term": {
      "category": "electronics"
    }
  }
}
'
```

### Advanced Search

#### Hybrid Search (Vector + Text)

To combine vector similarity search with traditional text search, OpenSearch provides a dedicated [Hybrid Search](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) feature. This feature properly normalizes and combines scores from different query types (vector and lexical) to produce meaningful results.

**Key benefits of Hybrid Search:**
- Proper score normalization across different query types
- Combines semantic (vector) and lexical (keyword) search
- Configurable score combination strategies

Please refer to the [OpenSearch Hybrid Search documentation](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) for setup instructions and examples.

#### Nested Field Search

Search vectors in nested documents:

```bash
# Index mapping with nested field
curl -X PUT "localhost:9200/nested-index" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "products": {
        "type": "nested",
        "properties": {
          "product_vector": {
            "type": "knn_vector",
            "dimension": 128,
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
}
'

# Search nested vectors
curl -X POST "localhost:9200/nested-index/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "nested": {
      "path": "products",
      "query": {
        "knn": {
          "products.product_vector": {
            "vector": [0.1, 0.2, ..., 0.128],
            "k": 5
          }
        }
      }
    }
  }
}
'
```

### Performance Monitoring

#### JVector Statistics

Get detailed JVector statistics:

```bash
curl -X GET "localhost:9200/_nodes/stats/indices/knn?pretty"
```

**Key Statistics:**
- `knn_query_visited_nodes`: Total nodes visited during searches
- `knn_query_expanded_nodes`: Nodes expanded during searches
- `knn_query_expanded_base_layer_nodes`: Base layer nodes expanded
- `knn_graph_merge_time`: Time spent on graph merges
- `knn_quantization_training_time`: Time spent on quantization

#### Query Profiling

OpenSearch provides built-in query profiling capabilities. For detailed information on profiling queries, see the [OpenSearch Query Profiling documentation](https://opensearch.org/docs/latest/api-reference/profile/).

**Example:**

```bash
curl -X POST "localhost:9200/my-index/_search" -H 'Content-Type: application/json' -d'
{
  "profile": true,
  "query": {
    "knn": {
      "vector_field": {
        "vector": [0.1, 0.2, ..., 0.768],
        "k": 10
      }
    }
  }
}
'
```

The profile output will include timing information for the k-NN query execution.

---

## Performance Optimization

### Index-time Optimization

#### Choosing M Parameter

The `m` parameter controls graph connectivity:

| M Value | Memory Usage | Recall | Indexing Speed | Use Case |
|---------|--------------|--------|----------------|----------|
| 8-12 | Low | Good | Fast | Small datasets, memory-constrained |
| 16-24 | Medium | Better | Medium | General purpose (recommended) |
| 32-48 | High | Best | Slow | High-recall requirements |

**Recommendation:** Start with `m=16`, increase if recall is insufficient.

#### Choosing ef_construction

The `ef_construction` parameter affects index quality:

| ef_construction | Index Quality | Indexing Speed | Use Case |
|-----------------|---------------|----------------|----------|
| 50-100 | Good | Fast | Development, testing |
| 100-200 | Better | Medium | Production (recommended) |
| 200-500 | Best | Slow | High-quality indices |

**Recommendation:** Use `ef_construction=100` for most cases, increase to 200 for better recall.

#### Quantization Trade-offs

**Product Quantization:**
- Memory: Significant reduction (configurable compression ratio)
- Recall: ~90-95% of original (depends on configuration)
- Speed: Faster for large datasets
- **Use when:** Large datasets with memory constraints

**No Quantization:**
- Memory: Full precision
- Recall: 100% (baseline)
- Speed: Baseline
- **Use when:** Memory is not a constraint, maximum recall required

#### Merge Frequency Tuning

Control how often segments are merged:

```bash
curl -X PUT "localhost:9200/my-index/_settings" -H 'Content-Type: application/json' -d'
{
  "index": {
    "merge.policy.max_merged_segment": "5gb",
    "merge.scheduler.max_thread_count": 1
  }
}
'
```

**Best Practices:**
- Perform force merge after bulk indexing
- Use incremental merges for continuous updates
- Monitor merge times with JVector statistics

### Query-time Optimization

#### ef_search Parameter Tuning

Tune `ef_search` based on your recall requirements by testing different values:

```bash
# Test with ef_search=50
curl -X POST "localhost:9200/my-index/_search" -H 'Content-Type: application/json' -d'
{
  "size": 10,
  "query": {
    "knn": {
      "vector_field": {
        "vector": [0.1, 0.2, ..., 0.768],
        "k": 10,
        "ef_search": 50
      }
    }
  }
}
'

# Test with ef_search=100
# ... (repeat with different values: 150, 200, 300)
```

**Guidelines:**
- Start with `ef_search=100`
- Increase by 50-100 if recall is low
- Monitor latency impact
- Find the sweet spot for your use case

#### Caching Strategies

Enable query result caching:

```bash
curl -X PUT "localhost:9200/my-index/_settings" -H 'Content-Type: application/json' -d'
{
  "index.queries.cache.enabled": true
}
'
```

**Note:** Vector queries are typically unique, so caching may have limited benefit.

### Memory Management

#### DiskANN for Large-Scale Indices

JVector's DiskANN implementation is optimized for RAM-constrained environments:

**Memory Estimation:**
- **Without quantization**: `num_vectors × dimension × 4 bytes × 1.5` (overhead)
- **With 8-bit quantization**: `num_vectors × dimension × 1 byte × 1.5`
- **With PQ (96 subvectors)**: `num_vectors × 96 bytes × 1.5`

**Example:**
- 1M vectors, 768 dims, no quantization: ~4.6 GB
- 1M vectors, 768 dims, 8-bit quantization: ~1.2 GB
- 1M vectors, 768 dims, PQ (96 subvectors): ~144 MB

#### Quantization for Memory Reduction

**When to use Product Quantization:**
- Index size exceeds available RAM
- Cost optimization (smaller instances)
- Acceptable recall trade-off (90-95%)

**Product Quantization is automatically applied by JVector** when the vector count exceeds the minimum batch size threshold. You can configure PQ parameters in the index settings (see [Quantization Options](#quantization-options) section).

#### OS Cache Considerations

JVector leverages OS file system cache:

**Best Practices:**
- Leave 20-30% of RAM for OS cache
- Monitor cache hit rates
- Use SSDs for better performance
- Consider memory-mapped files for large indices

### Benchmarking

#### Using Provided Scripts

The plugin includes comprehensive benchmarking tools:

```bash
cd scripts/jvector_index_and_search

# Install dependencies
pip install -r ../requirements.txt

# Run benchmark with recall measurement
python create_and_test_large_index.py \
  --num-vectors 100000 \
  --dimension 768 \
  --measure-recall \
  --num-recall-queries 50 \
  --csv-output results.csv \
  --plot
```

#### Measuring Recall

Recall measures search quality (% of true neighbors found):

```python
from jvector_utils.recall_measurement import GroundTruthTracker
from jvector_utils.search_operations import test_search_with_stats

# Create tracker
tracker = GroundTruthTracker(
    num_queries=50,
    k=10,
    dimension=768
)

# Track ground truth during indexing
for vector in vectors:
    tracker.update_ground_truth(vector)

# Measure recall
recall = tracker.compute_recall(search_results)
print(f"Recall@10: {recall:.3f}")
```

#### Measuring Latency and Throughput

For comprehensive performance measurements including latency, throughput, and recall, see the [Benchmarking](#benchmarking) section which provides detailed instructions on using the provided benchmarking tools.

#### Performance Comparison

Compare JVector with Lucene HNSW:

```bash
# Create JVector index
curl -X PUT "localhost:9200/jvector-index" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {"name": "disk_ann", "engine": "jvector", "space_type": "l2"}
      }
    }
  }
}
'

# Create Lucene index
curl -X PUT "localhost:9200/lucene-index" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}
      }
    }
  }
}
'

# Index same data to both
# Run benchmarks and compare
```

**Expected Results:**
- JVector provides faster search for large datasets
- JVector provides significantly faster merges with incremental updates
- JVector performs well in RAM-constrained environments

See [benchmarks](../README.md#incremental-merges) for detailed performance comparisons.

---

## Next Steps

- **API Reference**: See [REFERENCE.md](REFERENCE.md) for complete API documentation
- **Migration Guide**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for migrating from other solutions
- **Developer Guide**: See [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) for contributing to the plugin
- **Testing Scripts**: See [scripts/jvector_index_and_search/README.md](../scripts/jvector_index_and_search/README.md) for benchmarking tools
