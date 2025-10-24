# OpenSearch JVector Testing Scripts

This directory contains scripts for testing OpenSearch JVector functionality, particularly with large indices.

## Installation

### Prerequisites

- Python 3.6+
- OpenSearch instance with JVector plugin installed

### Setup

#### Using a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

1. Create a virtual environment:
   ```bash
   sudo apt install python3.11-venv
   # Using venv (Python 3.3+)
   python3 -m venv .venv
   
   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. Install the required Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. When you're done, you can deactivate the virtual environment:
   ```bash
   deactivate
   ```

#### Direct Installation

If you prefer not to use a virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

### Creating and Testing Large JVector Index

The `create_and_test_large_index.py` script creates a large JVector index that exceeds 2GB after force merge, which is useful for testing large index handling capabilities.

```bash
python create_and_test_large_index.py [options]
```

#### Options:

- `--host`: OpenSearch host:port (default: localhost:9200)
- `--index`: Index name (default: large-jvector-index)
- `--dimension`: Vector dimension (default: 768)
- `--num-vectors`: Number of vectors to index (default: 3,000,000)
- `--batch-size`: Batch size for indexing (default: 1,000)
- `--shards`: Number of shards (default: 1)

#### Example:

```bash
# Create a large index with default settings
python create_and_test_large_index.py

# Create a larger index with custom settings
python create_and_test_large_index.py --dimension 1024 --num-vectors 5000000 --batch-size 2000 --shards 2
```

#### What the script does:

1. Creates a knn_vector index with JVector engine
2. Indexes the specified number of vectors with the given dimension
3. Reports index stats before force merge
4. Performs a force merge to consolidate segments
5. Reports index stats after force merge
6. Tests search functionality on the large index

#### Notes:

- The default settings (3M vectors with 768 dimensions) should create an index exceeding 2GB after force merge
- Adjust the parameters based on your available system resources
- The script requires sufficient memory and disk space to handle large indices

#### JVector Statistics

The script collects and reports JVector-specific search and indexing statistics:

- `knn_query_visited_nodes`: Number of nodes visited during graph search
- `knn_query_expanded_nodes`: Number of nodes expanded during graph search
- `knn_query_expanded_base_layer_nodes`: Number of base layer nodes expanded
- `knn_query_graph_search_time`: Time spent on graph search (ms)
- `knn_quantization_training_time`: Time spent on quantization training (ms)
- `knn_graph_merge_time`: Time spent on graph merge (ms)

##### Search Testing
For each search iteration, the script:
1. Performs a kNN search
2. Collects the JVector stats
3. Reports the incremental changes for each metric

After all searches are complete, the script provides:
- Initial stats (before any searches)
- Final stats (after all searches)
- Total differences between initial and final stats
- Average values per search

This detailed reporting helps in understanding the search behavior and performance characteristics of the JVector engine on a per-query basis.

You can control the number of test searches with the `--num-searches` parameter:

```bash
python create_and_test_large_index.py --num-searches 10
```

#### Recall Measurement

The script now supports measuring recall@k to evaluate the quality of approximate nearest neighbor search results. Recall is calculated by comparing the approximate search results from the JVector index against ground truth computed using an efficient incremental approach.

**Key Feature**: The implementation uses a memory-efficient algorithm that computes ground truth incrementally during indexing using min-heaps, eliminating the need to store all vectors in memory.

##### Options:

- `--measure-recall`: Enable recall measurement (disabled by default)
- `--num-recall-queries`: Number of query vectors to pre-generate for recall measurement (default: same as `--num-searches`)

##### How it works:

1. When `--measure-recall` is enabled, the script pre-generates random query vectors
2. During indexing, for each indexed vector, it updates min-heaps to track the k-nearest neighbors for each query
3. After indexing completes, the ground truth is already computed and stored in the heaps
4. During search testing, it uses the pre-generated query vectors and compares results with the pre-computed ground truth
5. Reports recall@k for each query and provides summary statistics

This approach is **much more memory-efficient** than storing all vectors, as it only needs to store:
- The query vectors (num_queries × dimension)
- The k-nearest neighbors for each query (num_queries × k × small overhead)

##### Example:

```bash
# Enable recall measurement with default settings
python create_and_test_large_index.py --measure-recall --num-vectors 1000000

# Pre-generate 50 query vectors for recall testing
python create_and_test_large_index.py --measure-recall --num-recall-queries 50 --num-searches 50

# Combine with other options
python create_and_test_large_index.py --measure-recall --num-recall-queries 20 --num-searches 20 --dimension 384
```

##### Output:

For each search iteration, the script reports:
- Recall@k value (e.g., 0.9500 means 95% of results match ground truth)
- Number of correct results out of k (e.g., "9/10 correct")

After all searches, a summary is provided:
- Average Recall@k across all queries
- Minimum Recall@k
- Maximum Recall@k
- Standard deviation

##### Memory Considerations:

The memory usage for recall tracking is approximately:
```
Query vectors: num_queries × dimension × 8 bytes
Ground truth heaps: num_queries × k × 16 bytes
Total Memory (MB) ≈ (num_queries × dimension × 8 + num_queries × k × 16) / (1024 × 1024)
```

Examples (with k=10):
- 10 queries × 768 dimensions ≈ 0.06 MB (query vectors) + 0.002 MB (heaps) ≈ **0.06 MB total**
- 100 queries × 768 dimensions ≈ 0.59 MB (query vectors) + 0.015 MB (heaps) ≈ **0.60 MB total**
- 1000 queries × 768 dimensions ≈ 5.86 MB (query vectors) + 0.15 MB (heaps) ≈ **6.01 MB total**

Compare this to the old approach of storing all vectors:
- 1,000,000 vectors × 768 dimensions ≈ **5,859 MB** (nearly 6 GB!)

The new approach is **~1000x more memory efficient** for typical use cases!

**Note**: Recall measurement is not available when using `--skip-indexing` flag, as it requires tracking vectors during indexing.

#### CSV Output And Plotting

You can save the merge time data to a CSV file using the `--csv-output` option. The CSV file will contain the following columns:

- `num_documents`: Number of documents indexed
- `graph_merge_time_ms`: Time taken for graph merge (in milliseconds)
- `quantization_training_time_ms`: Time taken for quantization training (in milliseconds)
- `force_merge_duration_sec`: Duration of force merge (in seconds)
- `index_size_bytes`: Size of the index after force merge (in bytes)

```shell
# Run with CSV output
python create_and_test_large_index.py --batch-size 1000 --force-merge-frequency 1000 --num-vectors 100000 --csv-output merge_times.csv

# Generate plots from existing CSV
python create_and_test_large_index.py --csv-output merge_times.csv --plot
```

#### Important Note For Large Indices

When working with large indices, it's important to consider the point at which we will require quantization.
Quantization is becoming critical during index construction when we can't fit the full precision vectors in memory and are forced to use disk.
Therefore, we want to set the `minimum_batch_size_for_quantization` to a value high enough so we can avoid quantization during index construction.
Or alternatively, we can set it to a lower value and accept the additional compute cost of quantization during index construction, and thus avoid the disk access.

```shell
# Run with quantization disabled during index construction until we reach 10M documents
python create_and_test_large_index.py --batch-size 1000 --force-merge-frequency 1000 --num-vectors 100000 --min-batch-size-for-quantization 10000000
```

For long running tests you would want to move the script to run in the background and redirect the output to a file:
```shell
nohup python create_and_test_large_index.py --batch-size 5000 --force-merge-frequency 100000 --num-vectors 10000000 --min-batch-size-for-quantization 10000000 > output.log 2>&1 &
```

You can also profile the java process while running the script:
```shell
# Get the process id of the opensearch java process
PID=$(jps | grep OpenSearch | awk '{print $1}')
# Start profiling
jcmd $PID JFR.start name=OnDemand settings=profile duration=600s filename=/tmp/app_jfr_$(date +%s).jfr
```