"""
Search operations and testing utilities for JVector.

This module provides functions for testing kNN search performance,
collecting detailed statistics, and measuring recall.
"""

import requests
import time
import numpy as np

from .stats_utils import get_knn_stats, print_jvector_search_stats
from .recall_measurement import calculate_recall


def test_search(host, index_name, dimension, k=10, query_vector=None):
    """Test kNN search on the index

    Args:
        host: OpenSearch host:port
        index_name: Name of the index
        dimension: Vector dimension
        k: Number of nearest neighbors to retrieve
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
        host: OpenSearch host:port
        index_name: Name of the index
        dimension: Vector dimension
        k: Number of nearest neighbors to retrieve
        num_searches: Number of search iterations to perform
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

