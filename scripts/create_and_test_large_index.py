
#!/usr/bin/env python3

import requests
import json
import time
import numpy as np
import argparse
import sys
import csv
import matplotlib.pyplot as plt
import os
import heapq

def create_index(host, index_name, dimension, shards=1, min_batch_size_for_quantization=1000000):
    """Create a knn index with jvector engine"""
    url = f"http://{host}/{index_name}"
    
    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.derived_source.enabled": True,
                "number_of_shards": shards,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "disk_ann",
                        "space_type": "l2",
                        "engine": "jvector",
                        "parameters": {
                            "advanced.min_batch_size_for_quantization": min_batch_size_for_quantization
                        }
                    }
                },
                "id": {"type": "keyword"}
            }
        }
    }
    
    response = requests.put(url, json=mapping)
    if response.status_code != 200:
        print(f"Failed to create index: {response.text}")
        sys.exit(1)
    print(f"Successfully created index {index_name}")
    return response.json()

def index_vectors(host, index_name, num_vectors, dimension, batch_size=1000, force_merge_frequency=0, csv_file=None, ground_truth_tracker=None):
    """Index vectors in batches

    Args:
        ground_truth_tracker: Optional GroundTruthTracker to update during indexing
    """
    url = f"http://{host}/{index_name}/_bulk"
    headers = {"Content-Type": "application/x-ndjson"}

    total_batches = (num_vectors + batch_size - 1) // batch_size

    # Initialize CSV file if provided
    csv_writer = None
    csv_file_handle = None
    if csv_file:
        csv_file_handle = open(csv_file, 'w', newline='')
        csv_writer = csv.writer(csv_file_handle)
        csv_writer.writerow(['num_documents', 'graph_merge_time_ms', 'quantization_training_time_ms', 'force_merge_duration_sec', 'index_size_bytes'])

    for batch in range(total_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_vectors)
        current_batch_size = end_idx - start_idx

        bulk_data = []
        for i in range(current_batch_size):
            doc_id = start_idx + i
            # Create action line
            action = {"index": {"_index": index_name, "_id": str(doc_id)}}
            bulk_data.append(json.dumps(action))

            # Create random vector with values between -1 and 1
            vector = np.random.uniform(-1, 1, dimension).tolist()
            document = {"vector_field": vector, "id": str(doc_id)}
            bulk_data.append(json.dumps(document))

            # Update ground truth tracker if provided
            if ground_truth_tracker:
                ground_truth_tracker.update(str(doc_id), vector)
        
        bulk_body = "\n".join(bulk_data) + "\n"
        
        response = requests.post(url, headers=headers, data=bulk_body)
        if response.status_code != 200:
            print(f"Failed to index batch {batch+1}/{total_batches}: {response.text}")
            sys.exit(1)
        
        print(f"Indexed batch {batch+1}/{total_batches} ({current_batch_size} vectors)")
        
        # Force merge if frequency is set and we've reached the threshold
        if force_merge_frequency > 0 and (end_idx % force_merge_frequency < batch_size):
            print(f"\nPerforming intermediate force merge after {end_idx} documents...")
            merge_result = force_merge(host, index_name)
            print(f"Index stats after intermediate force merge:")
            stats = get_index_stats(host, index_name)
            
            # Write to CSV if enabled
            if csv_writer and merge_result:
                index_size = stats["indices"][index_name]["total"]["store"]["size_in_bytes"] if stats else 0
                csv_writer.writerow([
                    end_idx,
                    merge_result['graph_merge_time'],
                    merge_result['quantization_time'],
                    merge_result['duration'],
                    index_size
                ])
                csv_file_handle.flush()
    
    if csv_file_handle:
        csv_file_handle.close()

    return True

def force_merge(host, index_name, max_segments=1):
    """Force merge the index to consolidate segments"""
    # Get initial KNN stats
    initial_stats = get_knn_stats(host)
    initial_graph_merge_time = get_knn_stat_value(initial_stats, "knn_graph_merge_time")
    initial_quantization_time = get_knn_stat_value(initial_stats, "knn_quantization_training_time")

    # Refresh first to ensure all documents are searchable
    refresh_url = f"http://{host}/{index_name}/_refresh"
    refresh_response = requests.post(refresh_url)
    if refresh_response.status_code != 200:
        print(f"Refresh failed: {refresh_response.text}")

    url = f"http://{host}/{index_name}/_forcemerge?max_num_segments={max_segments}&flush=true"
    
    print(f"Starting force merge to {max_segments} segments...")
    start_time = time.time()
    response = requests.post(url)
    
    if response.status_code != 200:
        print(f"Force merge failed: {response.text}")
        return False
    
    duration = time.time() - start_time
    print(f"Force merge completed in {duration:.2f} seconds")

    # Get final KNN stats
    final_stats = get_knn_stats(host)
    final_graph_merge_time = get_knn_stat_value(final_stats, "knn_graph_merge_time")
    final_quantization_time = get_knn_stat_value(final_stats, "knn_quantization_training_time")

    # Calculate and display the differences
    graph_merge_diff = final_graph_merge_time - initial_graph_merge_time
    quantization_diff = final_quantization_time - initial_quantization_time

    print(f"KNN Graph Merge Time: +{graph_merge_diff} ms")
    print(f"KNN Quantization Training Time: +{quantization_diff} ms")

    return {
        'graph_merge_time': graph_merge_diff,
        'quantization_time': quantization_diff,
        'duration': duration
    }

def get_index_stats(host, index_name):
    """Get index stats including size"""
    url = f"http://{host}/{index_name}/_stats"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to get index stats: {response.text}")
        return None
    
    stats = response.json()
    size_in_bytes = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]
    size_in_gb = size_in_bytes / (1024 * 1024 * 1024)
    
    print(f"Index size: {size_in_bytes} bytes ({size_in_gb:.2f} GB)")
    return stats

def get_knn_stats(host):
    """Get KNN plugin stats including JVector search metrics"""
    url = f"http://{host}/_plugins/_knn/stats"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to get KNN stats: {response.text}")
        return None
    
    stats = response.json()
    return stats

def print_jvector_search_stats(stats):
    """Extract and print JVector-specific search statistics"""
    if not stats or "nodes" not in stats:
        print("No KNN stats available")
        return
    
    # JVector-specific metrics we want to track
    jvector_metrics = [
        "knn_query_visited_nodes",
        "knn_query_expanded_nodes",
        "knn_query_expanded_base_layer_nodes"
    ]
    
    # Collect stats from all nodes
    all_nodes_stats = {}
    for node_id, node_stats in stats["nodes"].items():
        for metric in jvector_metrics:
            if metric in node_stats:
                if metric not in all_nodes_stats:
                    all_nodes_stats[metric] = 0
                all_nodes_stats[metric] += int(node_stats[metric])
    
    # Print the stats
    print("\nJVector Search Statistics:")
    for metric, value in all_nodes_stats.items():
        print(f"  {metric}: {value}")

class GroundTruthTracker:
    """Tracks ground truth for pre-generated query vectors using min-heaps

    This allows computing ground truth incrementally during indexing without
    storing all vectors in memory.
    """

    def __init__(self, query_vectors, k, space_type='l2'):
        """Initialize tracker with query vectors

        Args:
            query_vectors: List of query vectors to track ground truth for
            k: Number of nearest neighbors to track
            space_type: Distance metric ('l2' or 'cosine')
        """
        self.query_vectors = query_vectors
        self.k = k
        self.space_type = space_type

        # For each query, maintain a max-heap of size k
        # We use max-heap so we can efficiently remove the farthest neighbor
        # when we find a closer one
        # Heap stores tuples of (-distance, doc_id) - negative for max-heap behavior
        self.heaps = [[] for _ in query_vectors]

    def update(self, doc_id, vector):
        """Update ground truth with a new vector

        Args:
            doc_id: Document ID
            vector: Vector to add
        """
        vector_np = np.array(vector)

        for i, query_vector in enumerate(self.query_vectors):
            # Compute distance
            if self.space_type == 'l2':
                dist = np.linalg.norm(query_vector - vector_np)
            elif self.space_type == 'cosine':
                query_norm = np.linalg.norm(query_vector)
                vector_norm = np.linalg.norm(vector_np)
                if query_norm > 0 and vector_norm > 0:
                    cosine_sim = np.dot(query_vector, vector_np) / (query_norm * vector_norm)
                    dist = 1 - cosine_sim
                else:
                    dist = 1.0
            else:
                raise ValueError(f"Unsupported space_type: {self.space_type}")

            heap = self.heaps[i]

            # If heap is not full, add the item
            if len(heap) < self.k:
                heapq.heappush(heap, (-dist, doc_id))
            # If this distance is smaller than the largest in heap, replace it
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, doc_id))

    def get_ground_truth(self, query_index):
        """Get ground truth for a specific query

        Args:
            query_index: Index of the query vector

        Returns:
            List of doc_ids of the k nearest neighbors
        """
        heap = self.heaps[query_index]
        # Sort by distance (ascending) and return doc_ids
        sorted_results = sorted(heap, key=lambda x: -x[0])
        return [doc_id for _, doc_id in sorted_results]

    def get_query_vector(self, query_index):
        """Get the query vector at the given index"""
        return self.query_vectors[query_index]

def calculate_recall(approximate_results, ground_truth):
    """Calculate recall@k

    Args:
        approximate_results: List of doc_ids from approximate search
        ground_truth: List of doc_ids from exact search

    Returns:
        Recall value (0.0 to 1.0)
    """
    if not ground_truth:
        return 0.0

    approximate_set = set(approximate_results)
    ground_truth_set = set(ground_truth)

    intersection = approximate_set.intersection(ground_truth_set)
    recall = len(intersection) / len(ground_truth)

    return recall

def test_search(host, index_name, dimension, k=10, query_vector=None):
    """Test kNN search on the index

    Args:
        query_vector: Optional pre-generated query vector. If None, a random one is created.

    Returns:
        Tuple of (success, results_list, query_vector, duration)
    """
    url = f"http://{host}/{index_name}/_search"

    # Create random query vector if not provided
    if query_vector is None:
        query_vector = np.random.uniform(-1, 1, dimension).tolist()

    query = {
        "size": k,
        "query": {
            "knn": {
                "vector_field": {
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    }

    start_time = time.time()
    response = requests.post(url, json=query)
    duration = time.time() - start_time

    if response.status_code != 200:
        print(f"Search failed: {response.text}")
        return False, [], query_vector, duration

    results = response.json()
    hits = results["hits"]["hits"]
    result_ids = [hit["_id"] for hit in hits]

    print(f"Search completed in {duration:.4f} seconds, found {len(hits)} results")
    return True, result_ids, query_vector, duration

def test_search_with_stats(host, index_name, dimension, k=10, num_searches=5, ground_truth_tracker=None):
    """Test kNN search on the index and report JVector stats for each iteration

    Args:
        ground_truth_tracker: Optional GroundTruthTracker for recall calculation
    """
    # Get initial stats
    current_stats = get_knn_stats(host)
    print("\nInitial JVector Stats:")
    print_jvector_search_stats(current_stats)

    # Track recall if tracker is provided
    measure_recall = ground_truth_tracker is not None
    recall_values = []
    
    # JVector-specific metrics we want to track
    jvector_metrics = [
        "knn_query_visited_nodes",
        "knn_query_expanded_nodes",
        "knn_query_expanded_base_layer_nodes"
    ]
    
    # Store initial values
    prev_totals = {}
    if current_stats and "nodes" in current_stats:
        for node_id, node_stats in current_stats["nodes"].items():
            for metric in jvector_metrics:
                if metric in node_stats:
                    if metric not in prev_totals:
                        prev_totals[metric] = 0
                    prev_totals[metric] += int(node_stats[metric])
    
    # Perform multiple searches and check stats after each
    for i in range(num_searches):
        print(f"\n--- Search Iteration {i+1}/{num_searches} ---")

        # If using tracker, use the pre-generated query vector
        query_vector = None
        if measure_recall:
            query_vector = ground_truth_tracker.get_query_vector(i).tolist()

        # Perform search
        success, result_ids, query_vector_used, duration = test_search(host, index_name, dimension, k, query_vector)

        # Calculate recall if enabled
        if measure_recall and success:
            # Use pre-computed ground truth from tracker
            ground_truth = ground_truth_tracker.get_ground_truth(i)
            recall = calculate_recall(result_ids, ground_truth)
            recall_values.append(recall)
            print(f"Recall@{k}: {recall:.4f} ({len(set(result_ids).intersection(set(ground_truth)))}/{k} correct)")

        # Get stats after this search
        new_stats = get_knn_stats(host)
        
        # Calculate current totals
        current_totals = {}
        if new_stats and "nodes" in new_stats:
            for node_id, node_stats in new_stats["nodes"].items():
                for metric in jvector_metrics:
                    if metric in node_stats:
                        if metric not in current_totals:
                            current_totals[metric] = 0
                        current_totals[metric] += int(node_stats[metric])
        
        # Calculate and print differences for this iteration
        print("\nJVector Stats for this iteration:")
        for metric in jvector_metrics:
            prev_val = prev_totals.get(metric, 0)
            current_val = current_totals.get(metric, 0)
            diff = current_val - prev_val
            print(f"  {metric}: +{diff}")
        
        # Update previous totals for next iteration
        prev_totals = current_totals
    
    # Get final stats
    final_stats = get_knn_stats(host)
    
    # Calculate initial and final totals
    initial_totals = {}
    if current_stats and "nodes" in current_stats:
        for node_id, node_stats in current_stats["nodes"].items():
            for metric in jvector_metrics:
                if metric in node_stats:
                    if metric not in initial_totals:
                        initial_totals[metric] = 0
                    initial_totals[metric] += int(node_stats[metric])
    
    final_totals = {}
    if final_stats and "nodes" in final_stats:
        for node_id, node_stats in final_stats["nodes"].items():
            for metric in jvector_metrics:
                if metric in node_stats:
                    if metric not in final_totals:
                        final_totals[metric] = 0
                    final_totals[metric] += int(node_stats[metric])
    
    # Print summary
    print("\n=== JVector Stats Summary ===")
    print("\nInitial Stats:")
    for metric, value in initial_totals.items():
        print(f"  {metric}: {value}")
    
    print("\nFinal Stats:")
    for metric, value in final_totals.items():
        print(f"  {metric}: {value}")
    
    print("\nTotal Differences (Final - Initial):")
    for metric in jvector_metrics:
        initial_val = initial_totals.get(metric, 0)
        final_val = final_totals.get(metric, 0)
        diff = final_val - initial_val
        print(f"  {metric}: +{diff}")
    
    # Calculate per-search averages
    print("\nAverage per Search:")
    for metric in jvector_metrics:
        initial_val = initial_totals.get(metric, 0)
        final_val = final_totals.get(metric, 0)
        diff = final_val - initial_val
        avg = diff / num_searches if num_searches > 0 else 0
        print(f"  {metric}: {avg:.2f}")

    # Print recall summary if measured
    if measure_recall and recall_values:
        print("\n=== Recall Summary ===")
        print(f"Average Recall@{k}: {np.mean(recall_values):.4f}")
        print(f"Min Recall@{k}: {np.min(recall_values):.4f}")
        print(f"Max Recall@{k}: {np.max(recall_values):.4f}")
        print(f"Std Dev: {np.std(recall_values):.4f}")

    return True

def get_knn_stat_value(stats, stat_name):
    """Extract a specific KNN stat value from all nodes"""
    total = 0
    if stats and "nodes" in stats:
        for node_id, node_stats in stats["nodes"].items():
            if stat_name in node_stats:
                total += int(node_stats[stat_name])
    return total

def plot_merge_times(csv_file):
    """Plot graph merge time and quantization training time vs number of documents"""
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found")
        return
    
    # Read CSV data
    num_docs = []
    graph_merge_times = []
    quantization_times = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_docs.append(int(row['num_documents']))
            graph_merge_times.append(int(row['graph_merge_time_ms']))
            quantization_times.append(int(row['quantization_training_time_ms']))
    
    if not num_docs:
        print("No data found in CSV file")
        return
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graph merge time plot
    ax1.plot(num_docs, graph_merge_times, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Documents')
    ax1.set_ylabel('Graph Merge Time (ms)')
    ax1.set_title('JVector Graph Merge Time vs Number of Documents')
    ax1.grid(True, alpha=0.3)
    
    # Quantization training time plot
    ax2.plot(num_docs, quantization_times, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Documents')
    ax2.set_ylabel('Quantization Training Time (ms)')
    ax2.set_title('JVector Quantization Training Time vs Number of Documents')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    
    # Show plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Create and test a large JVector index in OpenSearch")
    parser.add_argument("--host", default="localhost:9200", help="OpenSearch host:port")
    parser.add_argument("--index", default="large-jvector-index", help="Index name")
    parser.add_argument("--dimension", type=int, default=768, help="Vector dimension")
    parser.add_argument("--num-vectors", type=int, default=3000000,
                        help="Number of vectors to index (3M vectors with dim=768 should exceed 2GB)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for indexing")
    parser.add_argument("--shards", type=int, default=1, help="Number of shards")
    parser.add_argument("--num-searches", type=int, default=5, help="Number of searches to perform for stats testing")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip index creation and indexing, only run searches")
    parser.add_argument("--force-merge-frequency", type=int, default=0, 
                        help="Force merge after every N documents (0 to disable intermediate merges)")
    parser.add_argument("--csv-output", type=str, help="CSV file to save merge time data")
    parser.add_argument("--plot", action="store_true", help="Generate plots from CSV data")
    parser.add_argument("--min-batch-size-for-quantization", type=int, default=1000000,
                        help="Minimum batch size for quantization (default: 1M)")
    parser.add_argument("--measure-recall", action="store_true",
                        help="Measure recall by computing ground truth incrementally (memory efficient)")
    parser.add_argument("--num-recall-queries", type=int, default=None,
                        help="Number of query vectors to pre-generate for recall measurement (default: same as --num-searches)")

    args = parser.parse_args()

    # Set default for num_recall_queries if not specified
    if args.num_recall_queries is None:
        args.num_recall_queries = args.num_searches
    
    if args.plot and args.csv_output:
        plot_merge_times(args.csv_output)
        return

    # Storage for ground truth tracker if recall measurement is enabled
    ground_truth_tracker = None

    if not args.skip_indexing:
        print(f"Creating large JVector index with {args.num_vectors} vectors of dimension {args.dimension}")
        print(f"Estimated size: ~{args.num_vectors * args.dimension * 4 / (1024*1024*1024):.2f} GB (raw vectors only)")

        if args.measure_recall:
            # Pre-generate query vectors for recall measurement
            print(f"\nRecall measurement enabled - pre-generating {args.num_recall_queries} query vectors")
            query_vectors = [np.random.uniform(-1, 1, args.dimension) for _ in range(args.num_recall_queries)]

            # Initialize ground truth tracker
            ground_truth_tracker = GroundTruthTracker(query_vectors, k=10, space_type='l2')

            # Calculate memory usage
            # Each query vector: dimension * 8 bytes
            # Each heap entry: ~(8 bytes for distance + 8 bytes for pointer) * k * num_queries
            query_mem = args.num_recall_queries * args.dimension * 8 / (1024 * 1024)
            heap_mem = args.num_recall_queries * 10 * 16 / (1024 * 1024)  # k=10, ~16 bytes per entry
            total_mem = query_mem + heap_mem

            print(f"Memory usage for recall tracking:")
            print(f"  Query vectors: ~{query_mem:.2f} MB")
            print(f"  Ground truth heaps: ~{heap_mem:.2f} MB")
            print(f"  Total: ~{total_mem:.2f} MB")

        # Create index
        create_index(args.host, args.index, args.dimension, args.shards, args.min_batch_size_for_quantization)

        # Index vectors with ground truth tracker
        index_vectors(args.host, args.index, args.num_vectors, args.dimension, args.batch_size,
                     args.force_merge_frequency, args.csv_output, ground_truth_tracker=ground_truth_tracker)

        if args.measure_recall:
            print(f"\nGround truth computed for {args.num_recall_queries} query vectors during indexing")
        
        # Get stats before final force merge
        print("\nIndex stats before final force merge:")
        get_index_stats(args.host, args.index)
        
        # Force merge
        force_merge(args.host, args.index)
        
        # Get stats after force merge
        print("\nIndex stats after final force merge:")
        get_index_stats(args.host, args.index)
    else:
        print(f"Skipping index creation and indexing. Using existing index: {args.index}")
        # Get current index stats
        print("\nCurrent index stats:")
        get_index_stats(args.host, args.index)

        if args.measure_recall:
            print("\nWarning: Recall measurement requires indexing to store vectors.")
            print("Cannot measure recall when using --skip-indexing flag.")

    # Test search with JVector stats
    print("\nTesting search with JVector stats:")

    # Limit num_searches to num_recall_queries if using ground truth tracker
    actual_num_searches = args.num_searches
    if ground_truth_tracker and args.num_searches > args.num_recall_queries:
        print(f"\nWarning: Limiting searches to {args.num_recall_queries} (number of pre-generated recall queries)")
        actual_num_searches = args.num_recall_queries

    test_search_with_stats(args.host, args.index, args.dimension, k=10, num_searches=actual_num_searches,
                          ground_truth_tracker=ground_truth_tracker)
    
    print("\nTest completed successfully!")
    
    # Generate plots if CSV output was specified
    if args.csv_output and os.path.exists(args.csv_output):
        print(f"\nCSV data saved to {args.csv_output}")
        plot_merge_times(args.csv_output)

if __name__ == "__main__":
    main()
