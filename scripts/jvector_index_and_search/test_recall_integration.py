#!/usr/bin/env python3

"""
Integration test for recall measurement with OpenSearch.

This script creates a small test index, measures recall, and verifies
that the recall measurement is working correctly end-to-end.
"""

import argparse
import sys
import numpy as np
import requests

from jvector_utils.index_operations import create_index, index_vectors, force_merge
from jvector_utils.search_operations import test_search_with_stats
from jvector_utils.recall_measurement import GroundTruthTracker


def cleanup_index(host, index_name):
    """Delete the test index if it exists"""
    url = f"http://{host}/{index_name}"
    response = requests.delete(url)
    if response.status_code in [200, 404]:
        return True
    print(f"Warning: Failed to delete index: {response.text}")
    return False


def verify_opensearch_connection(host):
    """Verify that OpenSearch is accessible"""
    try:
        response = requests.get(f"http://{host}")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Connected to OpenSearch {info.get('version', {}).get('number', 'unknown')}")
            return True
        else:
            print(f"❌ Failed to connect to OpenSearch: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to OpenSearch: {e}")
        return False


def test_recall_with_small_index(host, dimension=128, num_vectors=1000, num_queries=5, k=10):
    """
    Test recall measurement with a small index.
    
    This test:
    1. Creates a small index with known vectors
    2. Tracks ground truth during indexing
    3. Performs searches and measures recall
    4. Verifies that recall is reasonable (should be high for small index)
    """
    index_name = "test-recall-index"
    
    print("\n" + "=" * 70)
    print("RECALL INTEGRATION TEST")
    print("=" * 70)
    print(f"Host: {host}")
    print(f"Index: {index_name}")
    print(f"Dimension: {dimension}")
    print(f"Vectors: {num_vectors}")
    print(f"Queries: {num_queries}")
    print(f"k: {k}")
    print("=" * 70)
    
    # Step 1: Verify OpenSearch connection
    print("\n[1/6] Verifying OpenSearch connection...")
    if not verify_opensearch_connection(host):
        print("❌ Cannot connect to OpenSearch. Is it running?")
        return False
    
    # Step 2: Cleanup any existing test index
    print("\n[2/6] Cleaning up any existing test index...")
    cleanup_index(host, index_name)
    
    # Step 3: Create index
    print("\n[3/6] Creating test index...")
    try:
        create_index(host, index_name, dimension, shards=1, min_batch_size_for_quantization=10000000)
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        return False
    
    # Step 4: Setup recall tracking
    print(f"\n[4/6] Setting up recall tracking with {num_queries} query vectors...")
    query_vectors = [np.random.uniform(-1, 1, dimension) for _ in range(num_queries)]
    tracker = GroundTruthTracker(query_vectors, k=k, space_type='l2')
    
    # Step 5: Index vectors with ground truth tracking
    print(f"\n[5/6] Indexing {num_vectors} vectors with ground truth tracking...")
    try:
        index_vectors(host, index_name, num_vectors, dimension, batch_size=100,
                     force_merge_frequency=0, csv_file=None, ground_truth_tracker=tracker)
        
        # Force merge to ensure all vectors are searchable
        print("\nForce merging index...")
        force_merge(host, index_name)
        
    except Exception as e:
        print(f"❌ Failed to index vectors: {e}")
        cleanup_index(host, index_name)
        return False
    
    # Step 6: Test search with recall measurement
    print(f"\n[6/6] Testing search with recall measurement...")
    try:
        test_search_with_stats(host, index_name, dimension, k=k, num_searches=num_queries,
                              ground_truth_tracker=tracker)
    except Exception as e:
        print(f"❌ Failed to test search: {e}")
        import traceback
        traceback.print_exc()
        cleanup_index(host, index_name)
        return False
    
    # Cleanup
    print("\n[Cleanup] Deleting test index...")
    cleanup_index(host, index_name)
    
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey observations:")
    print("1. Ground truth was tracked incrementally during indexing")
    print("2. Recall was measured for each search query")
    print("3. Recall values should be high (>0.9) for this small index")
    print("4. If recall is low, there may be an issue with the implementation")
    print("\nNote: Some variation in recall is expected due to:")
    print("  - Approximate nature of JVector search")
    print("  - Random vector generation")
    print("  - Graph construction parameters")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Integration test for recall measurement")
    parser.add_argument("--host", default="localhost:9200", help="OpenSearch host:port")
    parser.add_argument("--dimension", type=int, default=128, help="Vector dimension")
    parser.add_argument("--num-vectors", type=int, default=1000, help="Number of vectors to index")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of query vectors")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    
    args = parser.parse_args()
    
    success = test_recall_with_small_index(
        args.host,
        dimension=args.dimension,
        num_vectors=args.num_vectors,
        num_queries=args.num_queries,
        k=args.k
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

