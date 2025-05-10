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
   # Using venv (Python 3.3+)
   python -m venv .venv
   
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

#### JVector Search Statistics

The script collects and reports JVector-specific search statistics:

- `knn_query_visited_nodes`: Number of nodes visited during graph search
- `knn_query_expanded_nodes`: Number of nodes expanded during graph search
- `knn_query_expanded_base_layer_nodes`: Number of base layer nodes expanded

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