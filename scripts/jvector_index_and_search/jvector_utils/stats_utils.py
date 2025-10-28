"""
Statistics utilities for JVector KNN operations.

This module provides functions for retrieving and displaying KNN statistics
from OpenSearch, including JVector-specific search metrics.
"""

import requests


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


def get_knn_stat_value(stats, stat_name):
    """Extract a specific KNN stat value from all nodes"""
    total = 0
    if stats and "nodes" in stats:
        for node_id, node_stats in stats["nodes"].items():
            if stat_name in node_stats:
                total += int(node_stats[stat_name])
    return total

