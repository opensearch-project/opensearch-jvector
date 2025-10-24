#!/usr/bin/env python3

"""
Create and test large JVector indices in OpenSearch.

This script provides a comprehensive testing framework for JVector indices,
including indexing, force merging, search testing, and recall measurement.
"""

import argparse
import numpy as np
import os

from jvector_utils.index_operations import create_index, index_vectors, force_merge, get_index_stats
from jvector_utils.search_operations import test_search_with_stats
from jvector_utils.recall_measurement import GroundTruthTracker
from jvector_utils.visualization import plot_merge_times


def parse_arguments():
    """Parse command line arguments"""
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

    return args


def setup_recall_tracking(args):
    """Set up recall tracking with ground truth tracker
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        GroundTruthTracker instance or None
    """
    if not args.measure_recall:
        return None
    
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
    
    return ground_truth_tracker


def run_indexing(args, ground_truth_tracker=None):
    """Run the indexing workflow
    
    Args:
        args: Parsed command line arguments
        ground_truth_tracker: Optional GroundTruthTracker for recall measurement
    """
    print(f"Creating large JVector index with {args.num_vectors} vectors of dimension {args.dimension}")
    print(f"Estimated size: ~{args.num_vectors * args.dimension * 4 / (1024*1024*1024):.2f} GB (raw vectors only)")

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


def run_search_tests(args, ground_truth_tracker=None):
    """Run search tests with statistics
    
    Args:
        args: Parsed command line arguments
        ground_truth_tracker: Optional GroundTruthTracker for recall measurement
    """
    print("\nTesting search with JVector stats:")

    # Limit num_searches to num_recall_queries if using ground truth tracker
    actual_num_searches = args.num_searches
    if ground_truth_tracker and args.num_searches > args.num_recall_queries:
        print(f"\nWarning: Limiting searches to {args.num_recall_queries} (number of pre-generated recall queries)")
        actual_num_searches = args.num_recall_queries

    test_search_with_stats(args.host, args.index, args.dimension, k=10, num_searches=actual_num_searches,
                          ground_truth_tracker=ground_truth_tracker)


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Handle plot-only mode
    if args.plot and args.csv_output:
        plot_merge_times(args.csv_output)
        return

    # Setup recall tracking if enabled
    ground_truth_tracker = None
    
    if not args.skip_indexing:
        # Setup recall tracking before indexing
        ground_truth_tracker = setup_recall_tracking(args)
        
        # Run indexing workflow
        run_indexing(args, ground_truth_tracker)
    else:
        print(f"Skipping index creation and indexing. Using existing index: {args.index}")
        # Get current index stats
        print("\nCurrent index stats:")
        get_index_stats(args.host, args.index)

        if args.measure_recall:
            print("\nWarning: Recall measurement requires indexing to track ground truth.")
            print("Cannot measure recall when using --skip-indexing flag.")
    
    # Run search tests
    run_search_tests(args, ground_truth_tracker)
    
    print("\nTest completed successfully!")
    
    # Generate plots if CSV output was specified
    if args.csv_output and os.path.exists(args.csv_output):
        print(f"\nCSV data saved to {args.csv_output}")
        plot_merge_times(args.csv_output)


if __name__ == "__main__":
    main()

