
#!/usr/bin/env python3

import requests
import json
import time
import numpy as np
import argparse
import sys

def create_index(host, index_name, dimension, shards=1):
    """Create a knn index with jvector engine"""
    url = f"http://{host}/{index_name}"
    
    mapping = {
        "settings": {
            "index": {
                "knn": True,
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
                        "parameters": {}
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

def index_vectors(host, index_name, num_vectors, dimension, batch_size=1000, force_merge_frequency=0):
    """Index vectors in batches"""
    url = f"http://{host}/{index_name}/_bulk"
    headers = {"Content-Type": "application/x-ndjson"}
    
    total_batches = (num_vectors + batch_size - 1) // batch_size
    
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
        
        bulk_body = "\n".join(bulk_data) + "\n"
        
        response = requests.post(url, headers=headers, data=bulk_body)
        if response.status_code != 200:
            print(f"Failed to index batch {batch+1}/{total_batches}: {response.text}")
            sys.exit(1)
        
        print(f"Indexed batch {batch+1}/{total_batches} ({current_batch_size} vectors)")
        
        # Force merge if frequency is set and we've reached the threshold
        if force_merge_frequency > 0 and (end_idx % force_merge_frequency < batch_size):
            print(f"\nPerforming intermediate force merge after {end_idx} documents...")
            force_merge(host, index_name)
            print(f"Index stats after intermediate force merge:")
            get_index_stats(host, index_name)
    
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

    return True

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

def test_search(host, index_name, dimension, k=10):
    """Test kNN search on the index"""
    url = f"http://{host}/{index_name}/_search"
    
    # Create random query vector
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
        return False
    
    results = response.json()
    hits = results["hits"]["hits"]
    
    print(f"Search completed in {duration:.4f} seconds, found {len(hits)} results")
    return True

def test_search_with_stats(host, index_name, dimension, k=10, num_searches=5):
    """Test kNN search on the index and report JVector stats for each iteration"""
    # Get initial stats
    current_stats = get_knn_stats(host)
    print("\nInitial JVector Stats:")
    print_jvector_search_stats(current_stats)
    
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
        
        # Perform search
        test_search(host, index_name, dimension, k)
        
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
    
    return True

def get_knn_stat_value(stats, stat_name):
    """Extract a specific KNN stat value from all nodes"""
    total = 0
    if stats and "nodes" in stats:
        for node_id, node_stats in stats["nodes"].items():
            if stat_name in node_stats:
                total += int(node_stats[stat_name])
    return total

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
    
    args = parser.parse_args()
    
    if not args.skip_indexing:
        print(f"Creating large JVector index with {args.num_vectors} vectors of dimension {args.dimension}")
        print(f"Estimated size: ~{args.num_vectors * args.dimension * 4 / (1024*1024*1024):.2f} GB (raw vectors only)")
        
        # Create index
        create_index(args.host, args.index, args.dimension, args.shards)
        
        # Index vectors
        index_vectors(args.host, args.index, args.num_vectors, args.dimension, args.batch_size, args.force_merge_frequency)
        
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
    
    # Test search with JVector stats
    print("\nTesting search with JVector stats:")
    test_search_with_stats(args.host, args.index, args.dimension, k=10, num_searches=args.num_searches)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()