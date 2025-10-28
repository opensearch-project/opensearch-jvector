# JVector Utils

A modular Python package for testing and benchmarking JVector indices in OpenSearch.

## Overview

This package provides a clean, modular interface for working with JVector indices, including:
- Index creation and management
- Efficient vector indexing with batch operations
- Search testing with detailed statistics
- Memory-efficient recall measurement
- Performance visualization

## Package Structure

```
jvector_utils/
├── __init__.py                 # Package initialization
├── index_operations.py         # Index creation, indexing, force merge
├── search_operations.py        # Search testing and statistics
├── recall_measurement.py       # Ground truth tracking and recall calculation
├── stats_utils.py              # KNN statistics retrieval and display
└── visualization.py            # Performance plotting utilities
```

## Modules

### `index_operations.py`

Functions for creating and managing JVector indices:

- **`create_index(host, index_name, dimension, shards, min_batch_size_for_quantization)`**
  - Creates a new JVector index with specified configuration
  
- **`index_vectors(host, index_name, num_vectors, dimension, batch_size, ...)`**
  - Indexes random vectors in batches
  - Supports ground truth tracking for recall measurement
  - Optional intermediate force merges
  
- **`force_merge(host, index_name, max_segments)`**
  - Consolidates index segments
  - Tracks graph merge and quantization times
  
- **`get_index_stats(host, index_name)`**
  - Retrieves and displays index size statistics

### `search_operations.py`

Functions for testing search performance:

- **`test_search(host, index_name, dimension, k, query_vector)`**
  - Performs a single kNN search
  - Returns results, timing, and query vector
  
- **`test_search_with_stats(host, index_name, dimension, k, num_searches, ground_truth_tracker)`**
  - Runs multiple search iterations
  - Collects detailed JVector statistics
  - Calculates recall if ground truth tracker is provided
  - Reports per-iteration and aggregate metrics

### `recall_measurement.py`

Memory-efficient recall measurement:

- **`GroundTruthTracker` class**
  - Tracks exact k-nearest neighbors using min-heaps
  - Updates incrementally during indexing
  - Memory usage: O(num_queries × k) instead of O(num_vectors × dimension)
  - Supports L2 and cosine distance metrics
  
- **`calculate_recall(approximate_results, ground_truth)`**
  - Computes recall@k metric
  - Returns fraction of true neighbors found

### `stats_utils.py`

KNN statistics utilities:

- **`get_knn_stats(host)`**
  - Retrieves KNN plugin statistics from OpenSearch
  
- **`print_jvector_search_stats(stats)`**
  - Displays JVector-specific search metrics
  - Aggregates stats across all nodes
  
- **`get_knn_stat_value(stats, stat_name)`**
  - Extracts a specific metric value

### `visualization.py`

Performance visualization:

- **`plot_merge_times(csv_file)`**
  - Generates plots of graph merge time vs. document count
  - Plots quantization training time vs. document count
  - Saves high-resolution PNG output

## Usage Examples

### Basic Index Creation and Search

```python
from jvector_utils.index_operations import create_index, index_vectors
from jvector_utils.search_operations import test_search

# Create index
create_index("localhost:9200", "my-index", dimension=768, shards=1)

# Index vectors
index_vectors("localhost:9200", "my-index", num_vectors=10000, dimension=768)

# Test search
success, results, query, duration = test_search("localhost:9200", "my-index", dimension=768, k=10)
```

### Recall Measurement

```python
import numpy as np
from jvector_utils.recall_measurement import GroundTruthTracker
from jvector_utils.index_operations import index_vectors
from jvector_utils.search_operations import test_search_with_stats

# Pre-generate query vectors
query_vectors = [np.random.uniform(-1, 1, 768) for _ in range(10)]

# Initialize tracker
tracker = GroundTruthTracker(query_vectors, k=10, space_type='l2')

# Index with ground truth tracking
index_vectors("localhost:9200", "my-index", num_vectors=100000, dimension=768,
             ground_truth_tracker=tracker)

# Test search with recall measurement
test_search_with_stats("localhost:9200", "my-index", dimension=768, k=10,
                      num_searches=10, ground_truth_tracker=tracker)
```

### Performance Visualization

```python
from jvector_utils.visualization import plot_merge_times

# Generate plots from CSV data
plot_merge_times("merge_times.csv")
```

## Key Features

### Memory-Efficient Recall Measurement

The `GroundTruthTracker` uses a heap-based algorithm that:
- Pre-generates query vectors before indexing
- Maintains min-heaps during indexing (one per query)
- Only stores k-nearest neighbors per query
- Achieves ~1000x memory reduction compared to storing all vectors

**Memory Usage:**
```
Query vectors: num_queries × dimension × 8 bytes
Ground truth heaps: num_queries × k × 16 bytes
Total: ~6 MB for 1000 queries × 768 dimensions
```

Compare to storing all vectors:
```
All vectors: 1,000,000 × 768 × 8 bytes ≈ 5,859 MB (~6 GB)
```

### Detailed Statistics Tracking

JVector-specific metrics tracked:
- `knn_query_visited_nodes` - Total nodes visited during search
- `knn_query_expanded_nodes` - Nodes expanded during search
- `knn_query_expanded_base_layer_nodes` - Base layer nodes expanded

Statistics are reported:
- Per search iteration
- As aggregate totals
- As per-search averages

## Design Principles

1. **Modularity**: Each module has a single, clear responsibility
2. **Reusability**: Functions can be imported and used independently
3. **Testability**: Small, focused functions are easy to test
4. **Readability**: Clear naming and comprehensive documentation
5. **Efficiency**: Memory-efficient algorithms for large-scale testing

## Dependencies

- `requests` - HTTP client for OpenSearch API
- `numpy` - Numerical operations and vector generation
- `matplotlib` - Performance visualization
- Python 3.7+

## Integration

This package is designed to work seamlessly with the `create_and_test_large_index.py` script,
which provides a command-line interface to all functionality.

See `../README.md` for usage examples of the main script.

