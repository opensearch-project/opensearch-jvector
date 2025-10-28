"""
Recall measurement utilities for evaluating approximate nearest neighbor search quality.

This module provides efficient recall calculation using incremental ground truth tracking
with min-heaps, avoiding the need to store all vectors in memory.
"""

import numpy as np
import heapq


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

