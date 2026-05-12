# Derived Source

## Overview

`opensearch-jvector` supports derived source for KNN vector fields through the plugin setting `index.knn.derived_source.enabled`.

Starting from OpenSearch 3.7, you can also use the OpenSearch core derived source setting `index.derived_source.enabled`. This adds support for derived source across all supported fields, including `knn_vector` fields when used with `opensearch-jvector`.

Derived source reduces stored `_source` overhead by reconstructing fields when documents are read back. This is useful when you want to reduce storage overhead while still retrieving documents that include vectors and regular fields from the same index.

---

## Configuration Options

There are two derived source mechanisms relevant to k-NN indexes:

- OpenSearch core derived source via `index.derived_source.enabled`
- `opensearch-jvector` plugin derived source via `index.knn.derived_source.enabled`

### `opensearch-jvector` plugin derived source

This mechanism is similar to k-nn plugin, and is controlled by the plugin setting:

```json
{
  "settings": {
    "index.knn": true,
    "index.knn.derived_source.enabled": true
  }
}
```

This option is specific to `opensearch-jvector` behavior and applies only when KNN index setting is enabled on the index.

### OpenSearch core derived source

Starting from OpenSearch 3.7, you can use the core setting:

```json
{
  "settings": {
    "index.knn": true,
    "index.derived_source.enabled": true
  }
}
```

When `index.derived_source.enabled` is enabled, the core implementation takes precedence over the `opensearch-jvector`-specific implementation, including for KNN fields.

---

## Version Behavior

### Before OpenSearch 3.7

Use the `opensearch-jvector` plugin setting:

- `index.knn.derived_source.enabled`

The core setting `index.derived_source.enabled` does not apply to this `opensearch-jvector` plugin use case before OpenSearch 3.7.

### OpenSearch 3.7 and later

Both options are available:

- `index.knn.derived_source.enabled`
- `index.derived_source.enabled`

If both are provided, `index.derived_source.enabled` takes precedence.

For KNN-only use cases, `index.knn.derived_source.enabled` remains valid. For indexes that should use derived source across all supported fields, prefer `index.derived_source.enabled`.

---

## Setting Behavior

General rules:

- Indexes created with `index.knn.derived_source.enabled` continue to work after upgrade
- The only supported migration path between implementations is to reindex data
- If both settings are present on OpenSearch 3.7+, core derived source takes precedence
- If the core setting is not enabled (default), `opensearch-jvector` can continue handling derived source for vector fields

This preserves compatibility for existing `opensearch-jvector` indexes while allowing newer clusters to adopt the core OpenSearch implementation. By default `index.derived_source.enabled` is disabled.

---

## Requirements and Limitations

`opensearch-jvector` plugin implementation

Requirements:

- `index.knn.derived_source.enabled` requires `index.knn: true`
- `_source` must remain enabled for the jVector plugin implementation

Supported Features:

- Nested vector fields
- Works independently of other field types

Limitations:
- copy_to not supported
- `opensearch-jvector`-derived source is disabled when segment replication with local node-to-node replication is enabled


OpenSearch Core Approach

Requirements:

- `_source` must remain enabled for the jVector plugin implementation

Supported Features:

- Eliminate redundant storage of the entire `_source` field by reconstructing it from indexed data structures (doc values/stored fields)
- Unified approach for all field types
- Translog-based real-time reads with consistent formatting
- Index-level enablement

Limitations:

- Nested objects not supported
- All-or-nothing: If ANY field doesn't support derived source, entire feature fails for the index (normalizer, geo_shape, date_range, copy_to not supported)
- Field-level enablement/disablement not supported (doc_values=false, store=false not supported)

For core-derived source limitations, see the OpenSearch documentation:
https://docs.opensearch.org/latest/mappings/metadata-fields/source/#limitations

---

## Example Index Creation

### Using `opensearch-jvector` plugin derived source

```bash
curl -X PUT "http://localhost:9200/my-derived-source-index" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true,
    "index.knn.derived_source.enabled": true
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
          "space_type": "cosinesimil"
        }
      }
    }
  }
}'
```

### Using OpenSearch core derived source

```bash
curl -X PUT "http://localhost:9200/my-derived-source-index" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true,
    "index.derived_source.enabled": true
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
          "space_type": "cosinesimil"
        }
      }
    }
  }
}'
```

---

## Summary

Use `index.knn.derived_source.enabled` for the `opensearch-jvector` plugin implementation or when support for nested fields is required. Starting from OpenSearch 3.7, you can also use `index.derived_source.enabled` for the core implementation, which takes precedence when both settings are present.