# JVector Index and Search Testing

A comprehensive testing framework for JVector indices in OpenSearch, including indexing, search testing, and memory-efficient recall measurement.

## Overview

This directory contains tools for creating, testing, and benchmarking JVector indices with features including:
- Large-scale vector indexing with batch operations
- Force merge with detailed statistics tracking
- Search performance testing with JVector-specific metrics
- Memory-efficient recall measurement using heap-based ground truth tracking
- Performance visualization

## Directory Structure

```
jvector_index_and_search/
├── README.md                          # This file
├── create_and_test_large_index.py    # Main testing script
├── test_recall_measurement.py         # Unit tests for recall measurement
├── test_recall_integration.py         # Integration tests with OpenSearch
├── TESTING_RECALL.md                  # Comprehensive testing guide
├── jvector_utils/                     # Modular utilities package
│   ├── __init__.py
│   ├── index_operations.py           # Index creation and management
│   ├── search_operations.py          # Search testing and statistics
│   ├── recall_measurement.py         # Ground truth tracking
│   ├── stats_utils.py                # KNN statistics utilities
│   ├── visualization.py              # Performance plotting
│   └── README.md                     # Package documentation
├── merge_times.csv                    # Example output data
└── merge_times_plot.png               # Example visualization
```

## Quick Start

### Prerequisites

1. OpenSearch running with JVector plugin installed
2. Python 3.7+ with required dependencies

### Installation

```bash
cd scripts/jvector_index_and_search

# Install dependencies
pip install -r ../requirements.txt
```

### Basic Usage

```bash
# Create and test a large index
python create_and_test_large_index.py --num-vectors 100000

# With recall measurement
python create_and_test_large_index.py --measure-recall --num-vectors 100000

# With performance tracking
python create_and_test_large_index.py --csv-output merge_times.csv --plot
```

## Main Script: create_and_test_large_index.py

### Features

- **Large-scale indexing**: Index millions of vectors with configurable batch sizes
- **Force merge tracking**: Monitor graph merge and quantization times
- **Search testing**: Perform multiple searches with detailed JVector statistics
- **Recall measurement**: Memory-efficient ground truth tracking (uses ~1000x less memory)
- **Performance visualization**: Generate plots of merge times vs document count

### Common Options

```bash
# Basic options
--host localhost:9200              # OpenSearch host:port
--index large-jvector-index        # Index name
--dimension 768                    # Vector dimension
--num-vectors 3000000              # Number of vectors to index
--batch-size 1000                  # Batch size for indexing

# Search testing
--num-searches 5                   # Number of searches to perform
--skip-indexing                    # Skip indexing, only run searches

# Recall measurement
--measure-recall                   # Enable recall measurement
--num-recall-queries 10            # Number of query vectors for recall

# Performance tracking
--force-merge-frequency 100000     # Force merge every N documents
--csv-output merge_times.csv       # Save merge statistics to CSV
--plot                             # Generate plots from CSV data
```

### Examples

#### 1. Basic Large Index Test
```bash
python create_and_test_large_index.py \
  --num-vectors 1000000 \
  --dimension 768 \
  --batch-size 1000
```

#### 2. With Recall Measurement
```bash
python create_and_test_large_index.py \
  --measure-recall \
  --num-vectors 100000 \
  --num-recall-queries 20 \
  --num-searches 20
```

#### 3. Performance Analysis
```bash
python create_and_test_large_index.py \
  --num-vectors 500000 \
  --force-merge-frequency 100000 \
  --csv-output merge_times.csv \
  --plot
```

#### 4. Search Only (Existing Index)
```bash
python create_and_test_large_index.py \
  --skip-indexing \
  --index my-existing-index \
  --dimension 768 \
  --num-searches 10
```

## Testing

### Unit Tests (No OpenSearch Required)

Test the core recall measurement logic:

```bash
python test_recall_measurement.py
```

**Expected output:** All 6 tests pass ✅

### Integration Tests (Requires OpenSearch)

Test end-to-end with a real OpenSearch instance:

```bash
python test_recall_integration.py

# Customize parameters
python test_recall_integration.py --dimension 256 --num-vectors 5000
```

**Expected output:** Recall values >0.9 ✅

See [TESTING_RECALL.md](TESTING_RECALL.md) for comprehensive testing documentation.

## Memory-Efficient Recall Measurement

### How It Works

Traditional recall measurement requires storing all vectors in memory to compute ground truth:
- **Memory usage**: O(num_vectors × dimension)
- **Example**: 1M vectors × 768 dims ≈ 5,859 MB (~6 GB)

Our heap-based approach tracks ground truth incrementally:
- **Memory usage**: O(num_queries × k)
- **Example**: 100 queries × k=10 ≈ 6 MB

**Result**: ~1000x memory reduction! 🎉

### Usage

```bash
python create_and_test_large_index.py \
  --measure-recall \
  --num-recall-queries 50 \
  --num-vectors 1000000
```

The script will:
1. Pre-generate 50 query vectors
2. Track ground truth incrementally during indexing (using heaps)
3. Perform 50 searches and measure recall for each
4. Report average, min, max, and std dev of recall values

## JVector Statistics Tracked

The script collects detailed JVector-specific metrics:

- `knn_query_visited_nodes` - Total nodes visited during search
- `knn_query_expanded_nodes` - Nodes expanded during search
- `knn_query_expanded_base_layer_nodes` - Base layer nodes expanded
- `knn_graph_merge_time` - Time spent merging graphs
- `knn_quantization_training_time` - Time spent on quantization

Statistics are reported:
- Per search iteration
- As aggregate totals
- As per-search averages

## Performance Visualization

Generate plots showing how merge times scale with document count:

```bash
# During indexing
python create_and_test_large_index.py \
  --force-merge-frequency 100000 \
  --csv-output merge_times.csv

# Generate plots from existing CSV
python create_and_test_large_index.py \
  --plot \
  --csv-output merge_times.csv
```

Output: `merge_times_plot.png` with two subplots:
1. Graph merge time vs number of documents
2. Quantization training time vs number of documents

## Package: jvector_utils

The `jvector_utils` package provides reusable utilities that can be imported in other scripts:

```python
from jvector_utils.index_operations import create_index, index_vectors
from jvector_utils.search_operations import test_search_with_stats
from jvector_utils.recall_measurement import GroundTruthTracker
from jvector_utils.visualization import plot_merge_times

# Use in your own scripts
create_index("localhost:9200", "my-index", dimension=768)
index_vectors("localhost:9200", "my-index", num_vectors=10000, dimension=768)
```

See [jvector_utils/README.md](jvector_utils/README.md) for detailed package documentation.

## Sample Output

#### Sample Output
```bash
Index stats after final force merge:
Index size: 661120592 bytes (0.62 GB)

Testing search with JVector stats:

Initial JVector Stats:

JVector Search Statistics:
knn_query_visited_nodes: 0
knn_query_expanded_nodes: 0
knn_query_expanded_base_layer_nodes: 0

--- Search Iteration 1/1 ---
Search completed in 1.0714 seconds, found 10 results

JVector Stats for this iteration:
knn_query_visited_nodes: +961
knn_query_expanded_nodes: +66
knn_query_expanded_base_layer_nodes: +66

=== JVector Stats Summary ===

Initial Stats:
knn_query_visited_nodes: 0
knn_query_expanded_nodes: 0
knn_query_expanded_base_layer_nodes: 0

Final Stats:
knn_query_visited_nodes: 4769
knn_query_expanded_nodes: 327
knn_query_expanded_base_layer_nodes: 327

Total Differences (Final - Initial):
knn_query_visited_nodes: +4769
knn_query_expanded_nodes: +327
knn_query_expanded_base_layer_nodes: +327

Average per Search:
knn_query_visited_nodes: 953.80
knn_query_expanded_nodes: 65.40
knn_query_expanded_base_layer_nodes: 65.40

Test completed successfully!
```
## Troubleshooting

### Connection Errors

**Problem**: "Failed to connect to OpenSearch"

**Solution**:
```bash
# Verify OpenSearch is running
curl http://localhost:9200

# Check JVector plugin is installed
curl http://localhost:9200/_cat/plugins
```

### Low Recall Values

**Problem**: Recall consistently <0.8

**Solutions**:
1. Run unit tests to verify recall measurement: `python test_recall_measurement.py`
2. Increase index size (larger indices generally have better recall)
3. Check distance metric matches (L2 vs cosine)
4. Review JVector configuration parameters

### Memory Issues

**Problem**: Out of memory during indexing

**Solutions**:
1. Reduce batch size: `--batch-size 500`
2. Reduce number of recall queries: `--num-recall-queries 10`
3. Disable recall measurement if not needed
4. Use smaller vector dimension for testing

## Best Practices

1. **Start small**: Test with 10K-100K vectors before scaling to millions
2. **Use recall measurement**: Validate search quality with `--measure-recall`
3. **Monitor performance**: Use `--csv-output` to track merge times
4. **Run tests first**: Verify setup with `test_recall_measurement.py`
5. **Document results**: Save CSV outputs and plots for analysis

## Additional Resources

- **Testing Guide**: [TESTING_RECALL.md](TESTING_RECALL.md)
- **Package Documentation**: [jvector_utils/README.md](jvector_utils/README.md)
- **Main Scripts README**: [README.md](../README.md)

## Contributing

When making changes:
1. Run unit tests: `python test_recall_measurement.py`
2. Run integration tests: `python test_recall_integration.py`
3. Verify backward compatibility
4. Update documentation as needed

## License

See the main repository LICENSE file.

