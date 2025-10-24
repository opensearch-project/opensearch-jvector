# OpenSearch JVector Testing Scripts

This directory contains scripts for testing OpenSearch JVector functionality, particularly with large indices.

## Installation

### Prerequisites

- Python 3.6+
```bash
#Linux users
sudo apt install python3

# Mac users (install latest version of python3)
sudo brew install python3
```
- OpenSearch instance with JVector plugin installed. 

   See Install Jvector plugin section of [Developer Guide](DEVELOPER_GUIDE.md).

### Setup

#### Using a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

1. Create a virtual environment:
   ```bash
   # Linux users
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

Make sure you have pip installed. If you prefer not to use a virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

### Creating and Testing Large JVector Index

The `create_and_test_large_index.py` script creates a large JVector index that exceeds 2GB after force merge, which is useful for testing large index handling capabilities. The script with no parameters (i.e. default) can take up to 1hr to finish. 

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
4. Performs a force merge 1 segment to consolidate segments
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
The search performs a vector search, and defaults to 10 nearest neighbors (k). For each search iteration, the script:
1. Performs a kNN search
2. Collects the JVector stats
3. Reports the incremental changes for each metric

After all searches are complete, the script provides:
- Initial stats (before any searches)
- Final stats (after all searches)
- Total differences between initial and final stats
- Average values per search

This detailed reporting helps in understanding the search behavior and performance characteristics of the JVector engine on a per-query basis. Please refer to the last section of this readme for a sample report. 

You can control the number of test searches with the `--num-searches` parameter:

```bash
python create_and_test_large_index.py --num-searches 10
```

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

#### Sample Output 

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