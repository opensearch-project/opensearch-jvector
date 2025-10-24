#!/usr/bin/env python3

"""
Test suite for recall measurement functionality.

This script tests the GroundTruthTracker and recall calculation to ensure
they work correctly before using them with real OpenSearch indices.
"""

import numpy as np
import sys
from jvector_utils.recall_measurement import GroundTruthTracker, calculate_recall


def test_ground_truth_tracker_basic():
    """Test basic functionality of GroundTruthTracker"""
    print("\n=== Test 1: Basic GroundTruthTracker Functionality ===")
    
    # Create a simple test case with known vectors
    dimension = 3
    k = 3
    
    # Pre-generate a single query vector
    query_vector = np.array([0.0, 0.0, 0.0])
    
    # Initialize tracker
    tracker = GroundTruthTracker([query_vector], k=k, space_type='l2')
    
    # Add vectors at known distances from the query
    # Vector at distance 1.0
    tracker.update("id_1", [1.0, 0.0, 0.0])
    
    # Vector at distance 2.0
    tracker.update("id_2", [0.0, 2.0, 0.0])
    
    # Vector at distance 3.0
    tracker.update("id_3", [0.0, 0.0, 3.0])
    
    # Vector at distance sqrt(2) ≈ 1.414
    tracker.update("id_4", [1.0, 1.0, 0.0])
    
    # Vector at distance sqrt(3) ≈ 1.732
    tracker.update("id_5", [1.0, 1.0, 1.0])
    
    # Get ground truth (should be the 3 closest)
    ground_truth = tracker.get_ground_truth(0)
    
    print(f"Query vector: {query_vector}")
    print(f"Ground truth (k={k}): {ground_truth}")
    
    # Expected: id_1 (dist=1.0), id_4 (dist≈1.414), id_5 (dist≈1.732)
    expected = ["id_1", "id_4", "id_5"]
    
    if set(ground_truth) == set(expected):
        print("✅ PASS: Ground truth matches expected nearest neighbors")
        return True
    else:
        print(f"❌ FAIL: Expected {expected}, got {ground_truth}")
        return False


def test_ground_truth_tracker_large():
    """Test GroundTruthTracker with more vectors than k"""
    print("\n=== Test 2: GroundTruthTracker with Many Vectors ===")
    
    dimension = 10
    k = 5
    num_vectors = 100
    
    # Create a random query vector
    query_vector = np.random.uniform(-1, 1, dimension)
    
    # Initialize tracker
    tracker = GroundTruthTracker([query_vector], k=k, space_type='l2')
    
    # Add many random vectors and compute ground truth manually
    vectors = {}
    for i in range(num_vectors):
        vec = np.random.uniform(-1, 1, dimension)
        doc_id = f"doc_{i}"
        vectors[doc_id] = vec
        tracker.update(doc_id, vec.tolist())
    
    # Get ground truth from tracker
    ground_truth = tracker.get_ground_truth(0)
    
    # Compute ground truth manually using brute force
    distances = []
    for doc_id, vec in vectors.items():
        dist = np.sum((query_vector - vec) ** 2)  # L2 distance squared
        distances.append((dist, doc_id))
    
    distances.sort()
    expected_ground_truth = [doc_id for _, doc_id in distances[:k]]
    
    print(f"Number of vectors: {num_vectors}")
    print(f"k: {k}")
    print(f"Ground truth from tracker: {ground_truth}")
    print(f"Expected ground truth: {expected_ground_truth}")
    
    if set(ground_truth) == set(expected_ground_truth):
        print("✅ PASS: Ground truth matches brute-force computation")
        return True
    else:
        print(f"❌ FAIL: Mismatch in ground truth")
        print(f"  Missing: {set(expected_ground_truth) - set(ground_truth)}")
        print(f"  Extra: {set(ground_truth) - set(expected_ground_truth)}")
        return False


def test_calculate_recall():
    """Test the calculate_recall function"""
    print("\n=== Test 3: Recall Calculation ===")
    
    # Test perfect recall
    ground_truth = ["1", "2", "3", "4", "5"]
    approximate = ["1", "2", "3", "4", "5"]
    recall = calculate_recall(approximate, ground_truth)
    print(f"Perfect recall: {recall} (expected 1.0)")
    
    if recall != 1.0:
        print("❌ FAIL: Perfect recall should be 1.0")
        return False
    
    # Test 80% recall
    approximate = ["1", "2", "3", "4", "6"]  # 4 out of 5 correct
    recall = calculate_recall(approximate, ground_truth)
    print(f"80% recall: {recall} (expected 0.8)")
    
    if recall != 0.8:
        print(f"❌ FAIL: Expected 0.8, got {recall}")
        return False
    
    # Test 0% recall
    approximate = ["6", "7", "8", "9", "10"]
    recall = calculate_recall(approximate, ground_truth)
    print(f"0% recall: {recall} (expected 0.0)")
    
    if recall != 0.0:
        print(f"❌ FAIL: Expected 0.0, got {recall}")
        return False
    
    print("✅ PASS: All recall calculations correct")
    return True


def test_multiple_queries():
    """Test GroundTruthTracker with multiple query vectors"""
    print("\n=== Test 4: Multiple Query Vectors ===")
    
    dimension = 5
    k = 3
    num_queries = 3
    num_vectors = 50
    
    # Create multiple query vectors
    query_vectors = [np.random.uniform(-1, 1, dimension) for _ in range(num_queries)]
    
    # Initialize tracker
    tracker = GroundTruthTracker(query_vectors, k=k, space_type='l2')
    
    # Add vectors
    vectors = {}
    for i in range(num_vectors):
        vec = np.random.uniform(-1, 1, dimension)
        doc_id = f"doc_{i}"
        vectors[doc_id] = vec
        tracker.update(doc_id, vec.tolist())
    
    # Verify ground truth for each query
    all_pass = True
    for query_idx in range(num_queries):
        ground_truth = tracker.get_ground_truth(query_idx)
        query_vec = query_vectors[query_idx]
        
        # Compute expected ground truth
        distances = []
        for doc_id, vec in vectors.items():
            dist = np.sum((query_vec - vec) ** 2)
            distances.append((dist, doc_id))
        
        distances.sort()
        expected = [doc_id for _, doc_id in distances[:k]]
        
        if set(ground_truth) != set(expected):
            print(f"❌ FAIL: Query {query_idx} ground truth mismatch")
            all_pass = False
        else:
            print(f"✅ Query {query_idx}: Ground truth correct")
    
    if all_pass:
        print("✅ PASS: All queries have correct ground truth")
        return True
    else:
        return False


def test_cosine_distance():
    """Test GroundTruthTracker with cosine distance"""
    print("\n=== Test 5: Cosine Distance Metric ===")
    
    dimension = 3
    k = 2
    
    # Query vector
    query_vector = np.array([1.0, 0.0, 0.0])
    
    # Initialize tracker with cosine distance
    tracker = GroundTruthTracker([query_vector], k=k, space_type='cosine')
    
    # Add vectors with known cosine similarities
    # Same direction (cosine distance = 0)
    tracker.update("id_1", [2.0, 0.0, 0.0])
    
    # Orthogonal (cosine distance = 1)
    tracker.update("id_2", [0.0, 1.0, 0.0])
    
    # 45 degrees (cosine distance ≈ 0.293)
    tracker.update("id_3", [1.0, 1.0, 0.0])
    
    # Opposite direction (cosine distance = 2)
    tracker.update("id_4", [-1.0, 0.0, 0.0])
    
    ground_truth = tracker.get_ground_truth(0)
    
    print(f"Query vector: {query_vector}")
    print(f"Ground truth (k={k}): {ground_truth}")
    
    # Expected: id_1 (same direction), id_3 (45 degrees)
    expected = ["id_1", "id_3"]
    
    if set(ground_truth) == set(expected):
        print("✅ PASS: Cosine distance ground truth correct")
        return True
    else:
        print(f"❌ FAIL: Expected {expected}, got {ground_truth}")
        return False


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Test 6: Edge Cases ===")
    
    all_pass = True
    
    # Test with k larger than number of vectors
    print("\nTest 6a: k larger than number of vectors")
    query_vector = np.array([0.0, 0.0])
    tracker = GroundTruthTracker([query_vector], k=10, space_type='l2')
    
    tracker.update("id_1", [1.0, 0.0])
    tracker.update("id_2", [0.0, 1.0])
    tracker.update("id_3", [1.0, 1.0])
    
    ground_truth = tracker.get_ground_truth(0)
    
    if len(ground_truth) == 3:  # Should return all 3 vectors
        print("✅ PASS: Returns all vectors when k > num_vectors")
    else:
        print(f"❌ FAIL: Expected 3 vectors, got {len(ground_truth)}")
        all_pass = False
    
    # Test with identical vectors
    print("\nTest 6b: Identical vectors (tie-breaking)")
    query_vector = np.array([0.0, 0.0])
    tracker = GroundTruthTracker([query_vector], k=2, space_type='l2')
    
    # All at same distance
    tracker.update("id_1", [1.0, 0.0])
    tracker.update("id_2", [0.0, 1.0])
    tracker.update("id_3", [-1.0, 0.0])
    tracker.update("id_4", [0.0, -1.0])
    
    ground_truth = tracker.get_ground_truth(0)
    
    if len(ground_truth) == 2:
        print(f"✅ PASS: Returns k={2} vectors even with ties")
    else:
        print(f"❌ FAIL: Expected 2 vectors, got {len(ground_truth)}")
        all_pass = False
    
    return all_pass


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 70)
    print("RECALL MEASUREMENT TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Basic Functionality", test_ground_truth_tracker_basic),
        ("Large Vector Set", test_ground_truth_tracker_large),
        ("Recall Calculation", test_calculate_recall),
        ("Multiple Queries", test_multiple_queries),
        ("Cosine Distance", test_cosine_distance),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Recall measurement is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

