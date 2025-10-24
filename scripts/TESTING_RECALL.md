# Testing Recall Measurement

This guide explains how to test and verify that the recall measurement functionality is working correctly.

## Quick Start

### 1. Unit Tests (No OpenSearch Required)

Run the comprehensive unit test suite to verify the core recall measurement logic:

```bash
cd scripts
python test_recall_measurement.py
```

**Expected Result:**
```
======================================================================
TEST SUMMARY
======================================================================
✅ PASS: Basic Functionality
✅ PASS: Large Vector Set
✅ PASS: Recall Calculation
✅ PASS: Multiple Queries
✅ PASS: Cosine Distance
✅ PASS: Edge Cases

Total: 6/6 tests passed

🎉 All tests passed! Recall measurement is working correctly.
```

### 2. Integration Test (Requires OpenSearch)

Test recall measurement end-to-end with a real OpenSearch instance:

```bash
cd scripts
python test_recall_integration.py
```

**Expected Result:**
- Index created successfully
- Vectors indexed with ground truth tracking
- Searches performed with recall measurement
- Recall values typically >0.9 for small indices
- Test index cleaned up automatically

## What Gets Tested

### Unit Tests (`test_recall_measurement.py`)

#### Test 1: Basic Functionality
- Creates a simple 3D test case with known vectors
- Verifies that the k-nearest neighbors are correctly identified
- Tests with vectors at known distances from the query

#### Test 2: Large Vector Set
- Tests with 100 random vectors
- Compares tracker results against brute-force computation
- Ensures heap-based approach matches exact computation

#### Test 3: Recall Calculation
- Tests perfect recall (100%)
- Tests partial recall (80%)
- Tests zero recall (0%)
- Verifies the `calculate_recall()` function

#### Test 4: Multiple Query Vectors
- Tests tracking ground truth for multiple queries simultaneously
- Verifies each query has correct ground truth
- Tests with 3 queries and 50 vectors

#### Test 5: Cosine Distance
- Tests with cosine distance metric (not just L2)
- Verifies correct distance calculations
- Tests with vectors at known angles

#### Test 6: Edge Cases
- k larger than number of vectors
- Identical vectors (tie-breaking)
- Boundary conditions

### Integration Test (`test_recall_integration.py`)

#### End-to-End Workflow
1. **Connection Verification**: Checks OpenSearch is accessible
2. **Index Creation**: Creates a test index with JVector configuration
3. **Ground Truth Setup**: Pre-generates query vectors
4. **Indexing with Tracking**: Indexes vectors while tracking ground truth
5. **Search Testing**: Performs searches and measures recall
6. **Cleanup**: Removes test index

#### Customization Options

```bash
# Test with different parameters
python test_recall_integration.py \
  --host localhost:9200 \
  --dimension 256 \
  --num-vectors 5000 \
  --num-queries 10 \
  --k 20
```

## Understanding Recall Values

### What is Recall@k?

Recall@k measures the fraction of true k-nearest neighbors found by approximate search:

```
Recall@k = (Number of true neighbors found) / k
```

### Expected Recall Values

| Index Size | Expected Recall | Notes |
|------------|----------------|-------|
| Small (<10K) | >0.95 | Very high accuracy expected |
| Medium (10K-100K) | >0.90 | Good accuracy |
| Large (>100K) | >0.85 | Acceptable for approximate search |

**Factors affecting recall:**
- Index size (larger = potentially lower recall)
- Vector dimension
- Graph construction parameters
- Query vector distribution
- Distance metric (L2 vs cosine)

### Interpreting Results

#### Good Results ✅
```
Average Recall@10: 0.9450
Min Recall@10: 0.9000
Max Recall@10: 1.0000
```
- High average recall (>0.9)
- Consistent across queries (low std dev)
- Minimum recall still acceptable

#### Concerning Results ⚠️
```
Average Recall@10: 0.6500
Min Recall@10: 0.3000
Max Recall@10: 0.9000
```
- Low average recall (<0.8)
- High variance between queries
- Some queries have very poor recall

## Troubleshooting

### Unit Tests Fail

**Problem:** `test_recall_measurement.py` fails

**Possible Causes:**
1. Bug in `GroundTruthTracker` implementation
2. Incorrect distance calculation
3. Heap logic error

**Solution:**
- Check the specific test that failed
- Review the implementation in `jvector_utils/recall_measurement.py`
- Verify distance calculations match expected metric (L2 or cosine)

### Integration Test Fails to Connect

**Problem:** "Cannot connect to OpenSearch"

**Solution:**
```bash
# Check if OpenSearch is running
curl http://localhost:9200

# Start OpenSearch if needed
# (depends on your installation method)
```

### Low Recall Values

**Problem:** Recall is consistently <0.8

**Possible Causes:**
1. Index too small (not enough vectors for good graph)
2. Incorrect distance metric
3. Graph construction parameters need tuning
4. Bug in ground truth tracking

**Debugging Steps:**

1. **Verify ground truth is correct:**
   ```bash
   # Run unit tests first
   python test_recall_measurement.py
   ```

2. **Test with larger index:**
   ```bash
   python test_recall_integration.py --num-vectors 10000
   ```

3. **Check distance metric matches:**
   - Index uses L2 distance
   - Tracker should use `space_type='l2'`

4. **Inspect individual queries:**
   - Look at per-query recall values
   - Check if some queries have much lower recall

### Memory Issues

**Problem:** Out of memory during testing

**Solution:**
```bash
# Reduce test size
python test_recall_integration.py --num-vectors 500 --dimension 64

# Or reduce number of queries
python test_recall_integration.py --num-queries 3
```

## Manual Verification

You can manually verify recall measurement with a small example:

```python
import numpy as np
from jvector_utils.recall_measurement import GroundTruthTracker, calculate_recall

# Create a simple test
query = np.array([0.0, 0.0, 0.0])
tracker = GroundTruthTracker([query], k=3, space_type='l2')

# Add vectors at known distances
tracker.update("close", [1.0, 0.0, 0.0])      # distance = 1.0
tracker.update("medium", [2.0, 0.0, 0.0])     # distance = 2.0
tracker.update("far", [3.0, 0.0, 0.0])        # distance = 3.0

# Get ground truth
ground_truth = tracker.get_ground_truth(0)
print(f"Ground truth: {ground_truth}")
# Expected: ['close', 'medium', 'far']

# Test recall calculation
approximate = ["close", "medium", "wrong"]
recall = calculate_recall(approximate, ground_truth)
print(f"Recall: {recall}")
# Expected: 0.6667 (2 out of 3 correct)
```

## Best Practices

### For Development

1. **Always run unit tests first** before integration tests
2. **Start with small indices** (1K vectors) for faster iteration
3. **Use consistent random seeds** for reproducible tests
4. **Monitor memory usage** when testing with large indices

### For Production Use

1. **Validate recall on representative data** before deploying
2. **Set appropriate k values** (typically 10-100)
3. **Pre-generate enough queries** for statistical significance (>20)
4. **Monitor recall over time** as index grows
5. **Document expected recall ranges** for your use case

## Additional Resources

- **Package Documentation**: See `jvector_utils/README.md`
- **Main Script Usage**: See `README.md`

## Summary

To verify recall measurement is working:

```bash
# Step 1: Run unit tests (fast, no dependencies)
python test_recall_measurement.py

# Step 2: Run integration test (requires OpenSearch)
python test_recall_integration.py

# Step 3: Use with real workload
python create_and_test_large_index.py --measure-recall --num-vectors 100000
```

If all tests pass, recall measurement is working correctly! 🎉

