# OpenSearch jVector Plugin — User Guide

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Running the Demo](#running-the-demo)
- [Create an Index](#create-an-index)
  - [Minimal Example](#minimal-example)
  - [Full Configuration Reference](#full-configuration-reference)
  - [Quantized / On-Disk Index](#quantized--on-disk-index)
- [Index Documents](#index-documents)
- [Search](#search)
  - [Basic KNN Search](#basic-knn-search)
  - [Search with Filters](#search-with-filters)
  - [Search Tuning Parameters](#search-tuning-parameters)
- [Force Merge](#force-merge)
- [Space Types](#space-types)
- [Advanced Topics](#advanced-topics)
  - [Product Quantization (PQ)](#product-quantization-pq)
  - [DiskANN Mode](#diskann-mode)
  - [MMR Search](#mmr-search)
- [Index Settings Reference](#index-settings-reference)
- [Mapping Parameters Reference](#mapping-parameters-reference)

---

## Overview

The OpenSearch jVector plugin replaces the default KNN engine with [jVector](https://github.com/jbellis/jvector), a pure-Java DiskANN implementation. It provides:

- **Higher throughput** during indexing via thread-safe concurrent inserts
- **Lower search latency** due to quantized graph traversal
- **Incremental merges** — existing graphs are extended rather than rebuilt from scratch
- **DiskANN** — RAM-efficient ANN search that keeps quantized vectors on disk and re-ranks with full precision

The plugin exposes the same `knn_vector` field type and `knn` query used by the standard KNN plugin, so switching is mostly a matter of specifying `"engine": "jvector"` in your mapping.

---

## Prerequisites

- OpenSearch 3.x with the jVector plugin installed
- The plugin replaces `opensearch-knn`. Both cannot be active at the same time.

Verify the plugin is loaded:

```bash
curl -s http://localhost:9200/_cat/plugins?v | grep jvector
```

---

## Running the Demo

The repository ships a self-contained demo script that walks through every core operation end-to-end: create index → bulk index → search → filtered search → tuned search → force merge → post-merge search → node stats.

### 1. Start OpenSearch with the jVector plugin

If you don't have an existing cluster, start one locally using Gradle (from the repo root):

```bash
./gradlew run
```

Wait until the log shows:

```
[INFO][o.e.h.AbstractHttpServerTransport] publish_address {127.0.0.1:9200}
[INFO][o.e.n.Node] started
```

This builds the plugin zip, installs it into a local OpenSearch distribution, and boots a single-node cluster on `http://localhost:9200`.

### 2. Run the demo script

Open a second terminal and run:

```bash
./scripts/demo_full.sh
```

The script will:
1. Check cluster health and confirm the jVector plugin is installed (exits with an error if not)
2. Create a `jvector-demo` index with an 8-dimensional `cosinesimil` vector field
3. Bulk-index 10 movie documents, each with a genre, year, and embedding
4. Run a basic KNN search (sci-fi/action query vector)
5. Run a filtered KNN search (same vector, restricted to `genre=action`)
6. Run a tuned KNN search (`overquery_factor=10`) on an animation query
7. Force merge to 1 segment and repeat the search to verify consistency
8. Print jVector node stats (`visited_nodes`, `expanded_nodes`, `merge_time`)
9. Delete the demo index

### Options

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

# Keep the index so you can explore it afterwards
./scripts/demo_full.sh -x
```

---

## Create an Index

### Minimal Example

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

Key points:
- `index.knn: true` — required to activate the KNN plugin on this index
- `method.engine: "jvector"` — selects the jVector engine
- `method.name: "disk_ann"` — the only available algorithm for jVector
- `dimension` — must match the dimension of your embedding model exactly

### Full Configuration Reference

```bash
curl -X PUT "http://localhost:9200/my-vectors" \
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
        "dimension": 128,
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "cosinesimil",
          "parameters": {
            "m": 16,
            "ef_construction": 100,
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

### Quantized / On-Disk Index

For large datasets that exceed available RAM, use `on_disk` mode with a compression level:

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
        "dimension": 768,
        "mode": "on_disk",
        "compression_level": "8x",
        "method": {
          "name": "disk_ann",
          "engine": "jvector",
          "space_type": "l2",
          "parameters": {
            "advanced.num_pq_subspaces": 96,
            "advanced.min_batch_size_for_quantization": 1024
          }
        }
      }
    }
  }
}'
```

Higher compression levels require larger `advanced.num_pq_subspaces` to maintain recall quality.

---

## Index Documents

Index documents the same way as any OpenSearch index. The vector field accepts a flat array of floats whose length must equal `dimension`.

**Single document:**

```bash
curl -X PUT "http://localhost:9200/my-vectors/_doc/1" \
  -H "Content-Type: application/json" -d '
{
  "title": "Introduction to vector search",
  "category": "technology",
  "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}'
```

**Bulk indexing (recommended for large datasets):**

```bash
curl -X POST "http://localhost:9200/_bulk" \
  -H "Content-Type: application/json" -d '
{ "index": { "_index": "my-vectors", "_id": "1" } }
{ "title": "Document one", "embedding": [0.1, 0.2, 0.3, 0.4] }
{ "index": { "_index": "my-vectors", "_id": "2" } }
{ "title": "Document two", "embedding": [0.9, 0.8, 0.7, 0.6] }
{ "index": { "_index": "my-vectors", "_id": "3" } }
{ "title": "Document three", "embedding": [0.5, 0.5, 0.5, 0.5] }
'
```

---

## Search

### Basic KNN Search

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

### Search Tuning Parameters

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

---

## Force Merge

Force merging consolidates segments and triggers jVector's incremental graph merge, improving search performance significantly for write-heavy workloads.

```bash
# Merge down to 1 segment (recommended for read-heavy indexes)
curl -X POST "http://localhost:9200/my-vectors/_forcemerge?max_num_segments=1"
```

Wait for the operation to complete before querying — it runs synchronously and returns when done.

**Merge behaviour with jVector:**  
Unlike Lucene's full rebuild, jVector performs an *incremental merge* — nodes from smaller segments are inserted into the existing graph rather than re-indexing from scratch. This makes merges on large indexes dramatically faster (see the [README benchmarks](../README.md#incremental-merges)).

To prevent the leading (largest) segment from being touched during merge, set in your mapping:

```json
"parameters": {
  "advanced.leading_segment_merge_disabled": true
}
```

---

## Space Types

The `space_type` field controls which distance metric is used.

| Value | Distance metric | Use case |
|---|---|---|
| `l2` | Euclidean | General-purpose; raw coordinates |
| `cosinesimil` | Cosine similarity | Text embeddings; direction matters more than magnitude |
| `innerproduct` | Dot product | Embeddings where magnitude carries meaning (e.g. biencoder models) |
| `l1` | Manhattan | Robust to outliers |
| `linf` | Chebyshev | Maximum per-dimension deviation |

The default is `l2` when `space_type` is omitted.

---

## Advanced Topics

### Product Quantization (PQ)

PQ compresses vector storage by dividing each vector into subspaces and quantizing each independently. Set `advanced.num_pq_subspaces` in `method.parameters`:

```json
"parameters": {
  "advanced.num_pq_subspaces": 48,
  "advanced.min_batch_size_for_quantization": 1024
}
```

Rules:
- `num_pq_subspaces` must be ≤ `dimension` and ideally a divisor of it
- Quantization training is triggered automatically once the segment has at least `min_batch_size_for_quantization` documents
- More subspaces = less compression but better recall

### DiskANN Mode

For indexes that exceed available RAM, combine `mode: on_disk` with PQ:

```json
{
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
```

During search, jVector uses the compressed vectors for graph traversal and re-ranks the top candidates using full-precision vectors read from disk. Increase `overquery_factor` at query time to improve recall at the cost of more re-ranking disk reads.

### MMR Search

For diverse result sets, see [docs/mmr_search.md](mmr_search.md).

---

## Index Settings Reference

| Setting | Default | Description |
|---|---|---|
| `index.knn` | — | Set to `true` to enable the KNN plugin on the index (required) |
| `index.knn.advanced.approximate_threshold` | `15000` | Minimum document count per segment before an approximate graph is built |
| `index.knn.disk.vector.shard_level_rescoring_disabled` | `false` | Disable shard-level re-scoring for on-disk vectors |
| `index.knn.derived_source.enabled` | `true` | Store vectors in derived source fields |

---

## Mapping Parameters Reference

### Field-level

| Parameter | Required | Default | Description |
|---|---|---|---|
| `type` | yes | — | Must be `"knn_vector"` |
| `dimension` | yes | — | Integer 1–16,000 |
| `data_type` | no | `"float"` | Only `"float"` is supported |
| `mode` | no | `"in_memory"` | `"in_memory"` or `"on_disk"` |
| `compression_level` | no | `"1x"` | `"1x"` through `"64x"` |

### `method` object

| Parameter | Required | Default | Description |
|---|---|---|---|
| `name` | yes | — | Must be `"disk_ann"` |
| `engine` | yes | — | Must be `"jvector"` |
| `space_type` | no | `"l2"` | See [Space Types](#space-types) |

### `method.parameters`

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
