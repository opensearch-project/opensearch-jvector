"""
Index operations for JVector in OpenSearch.

This module provides functions for creating indices, indexing vectors,
and performing force merges with detailed statistics tracking.
"""

import requests
import json
import time
import numpy as np
import sys
import csv

from .stats_utils import get_knn_stats, get_knn_stat_value


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


def index_vectors(host, index_name, num_vectors, dimension, batch_size=1000, 
                  force_merge_frequency=0, csv_file=None, ground_truth_tracker=None):
    """Index vectors in batches

    Args:
        host: OpenSearch host:port
        index_name: Name of the index
        num_vectors: Total number of vectors to index
        dimension: Vector dimension
        batch_size: Number of vectors per batch
        force_merge_frequency: Force merge after every N documents (0 to disable)
        csv_file: Optional CSV file to save merge statistics
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
        csv_writer.writerow(['num_documents', 'graph_merge_time_ms', 'quantization_training_time_ms', 
                            'force_merge_duration_sec', 'index_size_bytes'])

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

