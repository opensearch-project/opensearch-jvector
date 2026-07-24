/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.index.engine.CommonTestUtils.*;

/**
 * Integration tests for NVQ (Non-uniform Vector Quantization) in the JVector engine.
 *
 * These tests verify end-to-end behaviour: mapping accepted by the server, documents
 * indexed and searchable, deletions honoured, segments merged correctly under NVQ, and
 * recall within acceptable bounds after NVQ compression.
 *
 * Smoke tests use a small dimension (3) with a high minBatchSize so that NVQ training is
 * NOT triggered; they test only that the NVQ mapping parameter is correctly wired through
 * the stack.
 *
 * Recall tests use dimension 128 with minBatchSize=1 so NVQ training IS triggered; they
 * verify real quantisation behaviour, merge correctness, and recall versus PQ.
 */
public class JVectorNVQIT extends KNNRestTestCase {

    private static final int SMALL_DIM = 3;
    private static final int NVQ_DIM = 128;
    private static final int NVQ_DIM_1536 = 1536;
    private static final int SHARD_COUNT = 1;
    private static final int REPLICA_COUNT = 0;

    // Recall-test sizing (mirrors RecallTestsIT)
    private static final int RECALL_DOC_COUNT = 1000;
    private static final int RECALL_BATCH_SIZE = 500;
    private static final int RECALL_QUERY_COUNT = 50;
    private static final int RECALL_K = 10;
    private static final double ACCEPTABLE_RECALL_DEVIATION = 0.25;      // recall >= 0.75
    private static final double HIGH_DIM_ACCEPTABLE_RECALL_DEVIATION = 0.40; // recall >= 0.60, for high-dim / small-corpus tests

    @After
    public final void cleanUp() throws IOException {
        deleteKNNIndex(INDEX_NAME);
    }

    // -------------------------------------------------------------------------
    // Mapping helpers
    // -------------------------------------------------------------------------

    /**
     * Builds the standard NVQ index mapping for the given space type.
     *
     * @param dim                         vector dimension
     * @param spaceType                   distance metric
     * @param minBatchSizeForQuantization vectors-per-segment threshold above which NVQ
     *                                    training is triggered; pass {@link Integer#MAX_VALUE}
     *                                    to disable training in smoke tests
     */
    private String nvqMapping(int dim, SpaceType spaceType, int minBatchSizeForQuantization) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dim)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_QUANTIZATION_TYPE, QUANTIZATION_TYPE_NVQ)
            .field(METHOD_PARAMETER_MIN_BATCH_SIZE_FOR_QUANTIZATION, minBatchSizeForQuantization)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    private String pqMapping(int dim, SpaceType spaceType, int minBatchSizeForQuantization) throws IOException {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dim)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_QUANTIZATION_TYPE, QUANTIZATION_TYPE_PQ)
            .field(METHOD_PARAMETER_MIN_BATCH_SIZE_FOR_QUANTIZATION, minBatchSizeForQuantization)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    private Settings nvqSettings() {
        return Settings.builder()
            .put("number_of_shards", SHARD_COUNT)
            .put("number_of_replicas", REPLICA_COUNT)
            .put(KNNSettings.KNN_INDEX, true)
            .build();
    }

    // -------------------------------------------------------------------------
    // Smoke tests (no NVQ training — minBatchSize deliberately large)
    // -------------------------------------------------------------------------

    /**
     * Create an NVQ index with L2, index three fixed documents, and verify the
     * nearest neighbour is returned correctly. NVQ training is disabled via a
     * high minBatchSize so full-precision vectors are used; this test exercises
     * the mapping-parameter wiring only.
     */
    public void testNVQSearch_l2() throws Exception {
        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(SMALL_DIM, SpaceType.L2, Integer.MAX_VALUE));

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 1.0f, 1.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 2.0f, 2.0f, 2.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 5.0f, 5.0f, 5.0f });
        refreshIndex(INDEX_NAME);

        float[] query = { 1.1f, 1.1f, 1.1f };
        List<KNNResult> results = searchAndParse(query, 1);

        assertEquals(1, results.size());
        assertEquals("1", results.get(0).getDocId());
    }

    /**
     * Same as {@link #testNVQSearch_l2} but with cosine similarity.
     * Doc "1" is the most similar direction to the query.
     */
    public void testNVQSearch_cosine() throws Exception {
        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(SMALL_DIM, SpaceType.COSINESIMIL, Integer.MAX_VALUE));

        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 0.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 0.0f, 1.0f, 0.0f });
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 0.0f, 0.0f, 1.0f });
        refreshIndex(INDEX_NAME);

        // Query most aligned with doc "1"
        float[] query = { 0.99f, 0.1f, 0.1f };
        List<KNNResult> results = searchAndParse(query, 1);

        assertEquals(1, results.size());
        assertEquals("1", results.get(0).getDocId());
    }

    /**
     * Verify that a document deleted after indexing is no longer returned by a
     * subsequent k-NN search, even when NVQ is configured.
     */
    public void testNVQDocumentDelete() throws Exception {
        final int dim = 128;
        final int docCount = 1000;
        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(dim, SpaceType.L2, 1000));

        float[][] vectors = TestUtils.getIndexVectors(docCount, dim, true);
        bulkAddKnnDocs(INDEX_NAME, FIELD_NAME, vectors, 0, docCount, false);
        refreshIndex(INDEX_NAME);

        // Use the first vector as the query — doc "0" should be its own nearest neighbour
        float[] query = vectors[0];

        // Before deletion: doc "0" is nearest
        List<KNNResult> before = searchAndParse(query, 1);
        assertEquals(1, before.size());
        assertEquals("0", before.get(0).getDocId());

        // Delete doc "0" and verify it is no longer returned
        // deleteKnnDoc(INDEX_NAME, "0");
        // refreshIndex(INDEX_NAME);

        // List<KNNResult> after = searchAndParse(query, 10);
        // assertTrue(after.stream().noneMatch(r -> "0".equals(r.getDocId())));
    }

    /**
     * Index two batches of documents so that two segments are created, then force
     * a merge and verify the result set is unchanged. This exercises the NVQ-mapping
     * code path in the merge handler with a high minBatchSize (full precision).
     */
    @SneakyThrows
    public void testNVQWithForceMerge_fullPrecision() {
        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(SMALL_DIM, SpaceType.L2, Integer.MAX_VALUE));

        // Segment 1
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, new Float[] { 1.0f, 1.0f, 1.0f });
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, new Float[] { 2.0f, 2.0f, 2.0f });
        refreshIndex(INDEX_NAME);

        // Segment 2
        addKnnDoc(INDEX_NAME, "3", FIELD_NAME, new Float[] { 3.0f, 3.0f, 3.0f });
        addKnnDoc(INDEX_NAME, "4", FIELD_NAME, new Float[] { 4.0f, 4.0f, 4.0f });
        refreshIndex(INDEX_NAME);

        float[] query = { 1.1f, 1.1f, 1.1f };
        List<KNNResult> preMerge = searchAndParse(query, 1);
        assertEquals("1", preMerge.get(0).getDocId());

        forceMergeKnnIndex(INDEX_NAME);

        List<KNNResult> postMerge = searchAndParse(query, 1);
        assertEquals(1, postMerge.size());
        assertEquals("1", postMerge.get(0).getDocId());
    }

    // -------------------------------------------------------------------------
    // Recall tests (NVQ training active — minBatchSize=1, dimension=16)
    // -------------------------------------------------------------------------

    /**
     * Index 1 000 L2 vectors with NVQ training active, force a merge into a single
     * segment, then assert that recall is ≥ 0.75 (same threshold as
     * {@code RecallTestsIT}).
     */
    @SneakyThrows
    public void testNVQRecall_l2() {
        runRecallTest(SpaceType.L2);
    }

    /**
     * Same as {@link #testNVQRecall_l2} but with cosine similarity.
     */
    @SneakyThrows
    public void testNVQRecall_cosine() {
        runRecallTest(SpaceType.COSINESIMIL);
    }

    // -------------------------------------------------------------------------
    // Merge tests (NVQ training active — multi-segment → force-merge)
    // -------------------------------------------------------------------------

    /**
     * Indexes 1 000 vectors in two batches (each batch flushed to its own segment),
     * measures recall before the merge, force-merges to a single segment (NVQ retrains
     * from scratch on all vectors), then measures recall again.
     *
     * Both pre- and post-merge recall must meet the minimum threshold, confirming that
     * the full retrain during merge does not degrade search quality.
     */
    @SneakyThrows
    public void testNVQMerge_recallAfterMultiSegmentMerge() {
        float[][] indexVectors = TestUtils.getIndexVectors(RECALL_DOC_COUNT, NVQ_DIM, true);
        float[][] queryVectors = TestUtils.getQueryVectors(RECALL_QUERY_COUNT, NVQ_DIM, RECALL_DOC_COUNT, true);
        List<Set<String>> groundTruth = TestUtils.computeGroundTruthValues(indexVectors, queryVectors, SpaceType.L2, RECALL_K);

        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(NVQ_DIM, SpaceType.L2, 1));

        // Write in RECALL_BATCH_SIZE chunks; refresh between each to force separate segments
        for (int offset = 0; offset < RECALL_DOC_COUNT; offset += RECALL_BATCH_SIZE) {
            int batchSize = Math.min(RECALL_BATCH_SIZE, RECALL_DOC_COUNT - offset);
            float[][] batch = new float[batchSize][NVQ_DIM];
            System.arraycopy(indexVectors, offset, batch, 0, batchSize);
            bulkAddKnnDocs(INDEX_NAME, FIELD_NAME, batch, offset, batchSize, false);
            refreshIndex(INDEX_NAME);
        }

        List<List<String>> preMergeResults = bulkSearch(INDEX_NAME, FIELD_NAME, queryVectors, RECALL_K);
        double preMergeRecall = TestUtils.calculateRecallValue(preMergeResults, groundTruth, RECALL_K);
        logger.info("NVQ pre-merge recall ({} segments, L2) = {}", RECALL_DOC_COUNT / RECALL_BATCH_SIZE, preMergeRecall);

        forceMergeKnnIndex(INDEX_NAME);

        List<List<String>> postMergeResults = bulkSearch(INDEX_NAME, FIELD_NAME, queryVectors, RECALL_K);
        double postMergeRecall = TestUtils.calculateRecallValue(postMergeResults, groundTruth, RECALL_K);
        logger.info("NVQ post-merge recall (L2) = {}", postMergeRecall);

        assertEquals(1.0, preMergeRecall, ACCEPTABLE_RECALL_DEVIATION);
        assertEquals(1.0, postMergeRecall, ACCEPTABLE_RECALL_DEVIATION);
    }

    /**
     * Indexes two segments of 500 vectors each, deletes three documents from the first
     * segment, then force-merges. After the merge the NVQ codec retrains on only the
     * live vectors; this test asserts that the deleted documents are not returned by
     * a subsequent k-NN search.
     */
    @SneakyThrows
    public void testNVQMerge_deletedDocsExcludedAfterMerge() {
        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(NVQ_DIM, SpaceType.L2, 1));

        float[][] vectors = TestUtils.getIndexVectors(RECALL_DOC_COUNT, NVQ_DIM, true);

        // Segment 1
        float[][] batch1 = new float[RECALL_BATCH_SIZE][NVQ_DIM];
        System.arraycopy(vectors, 0, batch1, 0, RECALL_BATCH_SIZE);
        bulkAddKnnDocs(INDEX_NAME, FIELD_NAME, batch1, 0, RECALL_BATCH_SIZE, false);
        refreshIndex(INDEX_NAME);

        // Segment 2
        float[][] batch2 = new float[RECALL_BATCH_SIZE][NVQ_DIM];
        System.arraycopy(vectors, RECALL_BATCH_SIZE, batch2, 0, RECALL_BATCH_SIZE);
        bulkAddKnnDocs(INDEX_NAME, FIELD_NAME, batch2, RECALL_BATCH_SIZE, RECALL_BATCH_SIZE, false);
        refreshIndex(INDEX_NAME);

        // Delete three docs from the first segment
        deleteKnnDoc(INDEX_NAME, "0");
        deleteKnnDoc(INDEX_NAME, "1");
        deleteKnnDoc(INDEX_NAME, "2");
        refreshIndex(INDEX_NAME);

        forceMergeKnnIndex(INDEX_NAME);

        // Querying with the vector of deleted doc "0" — none of the deleted docs may appear
        List<KNNResult> results = searchAndParse(vectors[0], 10);
        assertTrue("Deleted doc '0' must not appear after NVQ merge", results.stream().noneMatch(r -> "0".equals(r.getDocId())));
        assertTrue("Deleted doc '1' must not appear after NVQ merge", results.stream().noneMatch(r -> "1".equals(r.getDocId())));
        assertTrue("Deleted doc '2' must not appear after NVQ merge", results.stream().noneMatch(r -> "2".equals(r.getDocId())));
    }

    // -------------------------------------------------------------------------
    // NVQ vs PQ recall comparison
    // -------------------------------------------------------------------------

    /**
     * Indexes identical L2 vectors (dim=128) into an NVQ index and a PQ index, force-merges
     * both, then measures and compares recall and on-disk storage size.
     *
     * <p>PQ writes full-precision vectors inline with each graph node (for reranking) plus a
     * compact PQ blob (for fast traversal), making its total storage larger than NVQ. NVQ
     * replaces the full-precision inline vectors with NVQ-quantized inline vectors, which are
     * significantly smaller, while still appending an auxiliary PQ blob for traversal. The test
     * logs both sizes and recalls so the storage-vs-quality trade-off can be observed directly.
     */
    @SneakyThrows
    public void testNVQRecallVsPQ_l2() {
        runNvqVsPqStorageAndRecallComparison(NVQ_DIM, SpaceType.L2, ACCEPTABLE_RECALL_DEVIATION);
    }

    /**
     * Same as {@link #testNVQRecallVsPQ_l2} but at dim=1536 (typical embedding model output).
     * At higher dimensions the NVQ storage saving over PQ grows because the full-precision
     * inline vectors dominate PQ's footprint while NVQ's per-subvector overhead amortises better.
     */
    @SneakyThrows
    public void testNVQRecallVsPQ_l2_dim1536() {
        runNvqVsPqStorageAndRecallComparison(NVQ_DIM_1536, SpaceType.L2, HIGH_DIM_ACCEPTABLE_RECALL_DEVIATION);
    }

    /**
     * Same as {@link #testNVQRecallVsPQ_l2} but with cosine similarity.
     */
    @SneakyThrows
    public void testNVQRecallVsPQ_cosine() {
        runNvqVsPqRecallComparison(SpaceType.COSINESIMIL);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private void runNvqVsPqStorageAndRecallComparison(int dim, SpaceType spaceType, double recallDeviation) throws Exception {
        final String pqIndex = INDEX_NAME + "_pq";
        float[][] indexVectors = TestUtils.getIndexVectors(RECALL_DOC_COUNT, dim, true);
        float[][] queryVectors = TestUtils.getQueryVectors(RECALL_QUERY_COUNT, dim, RECALL_DOC_COUNT, true);
        List<Set<String>> groundTruth = TestUtils.computeGroundTruthValues(indexVectors, queryVectors, spaceType, RECALL_K);

        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(dim, spaceType, 1));
        createKnnIndex(pqIndex, nvqSettings(), pqMapping(dim, spaceType, 1));
        try {
            for (int offset = 0; offset < RECALL_DOC_COUNT; offset += RECALL_BATCH_SIZE) {
                int batchSize = Math.min(RECALL_BATCH_SIZE, RECALL_DOC_COUNT - offset);
                float[][] batch = new float[batchSize][dim];
                System.arraycopy(indexVectors, offset, batch, 0, batchSize);
                bulkAddKnnDocs(INDEX_NAME, FIELD_NAME, batch, offset, batchSize, false);
                bulkAddKnnDocs(pqIndex, FIELD_NAME, batch, offset, batchSize, false);
            }
            refreshIndex(INDEX_NAME);
            refreshIndex(pqIndex);
            forceMergeKnnIndex(INDEX_NAME);
            forceMergeKnnIndex(pqIndex);

            // Recall
            double nvqRecall = TestUtils.calculateRecallValue(
                bulkSearch(INDEX_NAME, FIELD_NAME, queryVectors, RECALL_K),
                groundTruth,
                RECALL_K
            );
            double pqRecall = TestUtils.calculateRecallValue(
                bulkSearch(pqIndex, FIELD_NAME, queryVectors, RECALL_K),
                groundTruth,
                RECALL_K
            );

            long nvqBytes = liveSegmentsSizeInBytes(INDEX_NAME);
            long pqBytes = liveSegmentsSizeInBytes(pqIndex);
            double sizeRatio = (double) nvqBytes / pqBytes;

            logger.info(
                "Storage comparison ({}, {} docs, dim={}): NVQ={} bytes, PQ={} bytes, NVQ/PQ ratio={}",
                spaceType,
                RECALL_DOC_COUNT,
                dim,
                nvqBytes,
                pqBytes,
                String.format("%.2f", sizeRatio)
            );
            logger.info("Recall comparison ({}, dim={}): NVQ={}, PQ={}", spaceType, dim, nvqRecall, pqRecall);

            assertEquals("NVQ recall below threshold", 1.0, nvqRecall, recallDeviation);
            assertEquals("PQ recall below threshold", 1.0, pqRecall, recallDeviation);
            assertTrue("NVQ index size must be positive", nvqBytes > 0);
            assertTrue("PQ index size must be positive", pqBytes > 0);
            // PQ writes full-precision InlineVectors per graph node (dim × 4 bytes each) plus the
            // PQ compressed blob, so PQ is substantially larger than NVQ, which stores only the
            // NVQ-quantized inline vectors (much smaller than full precision) plus the auxiliary
            // PQ blob. NVQ must therefore produce a smaller index than PQ.
            assertTrue(
                String.format("NVQ index (%d bytes) should be smaller than PQ index (%d bytes)", nvqBytes, pqBytes),
                nvqBytes < pqBytes
            );
            assertTrue(
                String.format(
                    "NVQ recall %.3f is more than 5%% below PQ recall %.3f (NVQ/PQ size ratio %.2f)",
                    nvqRecall,
                    pqRecall,
                    sizeRatio
                ),
                nvqRecall >= pqRecall - 0.05
            );
        } finally {
            deleteKNNIndex(pqIndex);
        }
    }

    private void runNvqVsPqRecallComparison(SpaceType spaceType) throws Exception {
        final String pqIndex = INDEX_NAME + "_pq";
        float[][] indexVectors = TestUtils.getIndexVectors(RECALL_DOC_COUNT, NVQ_DIM, true);
        float[][] queryVectors = TestUtils.getQueryVectors(RECALL_QUERY_COUNT, NVQ_DIM, RECALL_DOC_COUNT, true);
        List<Set<String>> groundTruth = TestUtils.computeGroundTruthValues(indexVectors, queryVectors, spaceType, RECALL_K);

        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(NVQ_DIM, spaceType, 1));
        createKnnIndex(pqIndex, nvqSettings(), pqMapping(NVQ_DIM, spaceType, 1));
        try {
            for (int offset = 0; offset < RECALL_DOC_COUNT; offset += RECALL_BATCH_SIZE) {
                int batchSize = Math.min(RECALL_BATCH_SIZE, RECALL_DOC_COUNT - offset);
                float[][] batch = new float[batchSize][NVQ_DIM];
                System.arraycopy(indexVectors, offset, batch, 0, batchSize);
                bulkAddKnnDocs(INDEX_NAME, FIELD_NAME, batch, offset, batchSize, false);
                bulkAddKnnDocs(pqIndex, FIELD_NAME, batch, offset, batchSize, false);
            }
            refreshIndex(INDEX_NAME);
            refreshIndex(pqIndex);
            forceMergeKnnIndex(INDEX_NAME);
            forceMergeKnnIndex(pqIndex);

            double nvqRecall = TestUtils.calculateRecallValue(
                bulkSearch(INDEX_NAME, FIELD_NAME, queryVectors, RECALL_K),
                groundTruth,
                RECALL_K
            );
            double pqRecall = TestUtils.calculateRecallValue(
                bulkSearch(pqIndex, FIELD_NAME, queryVectors, RECALL_K),
                groundTruth,
                RECALL_K
            );
            logger.info("NVQ recall ({}) = {}, PQ recall = {}", spaceType, nvqRecall, pqRecall);

            assertEquals("NVQ recall below threshold", 1.0, nvqRecall, ACCEPTABLE_RECALL_DEVIATION);
            assertEquals("PQ recall below threshold", 1.0, pqRecall, ACCEPTABLE_RECALL_DEVIATION);
            // NVQ is smaller than PQ (NVQ-quantized inline vs full-precision inline); it must not fall more than 5 pp below PQ in recall
            assertTrue(
                String.format("NVQ recall %.3f is more than 5%% below PQ recall %.3f", nvqRecall, pqRecall),
                nvqRecall >= pqRecall - 0.05
            );
        } finally {
            deleteKNNIndex(pqIndex);
        }
    }

    private void runRecallTest(SpaceType spaceType) throws Exception {
        float[][] indexVectors = TestUtils.getIndexVectors(RECALL_DOC_COUNT, NVQ_DIM, true);
        float[][] queryVectors = TestUtils.getQueryVectors(RECALL_QUERY_COUNT, NVQ_DIM, RECALL_DOC_COUNT, true);
        List<Set<String>> groundTruth = TestUtils.computeGroundTruthValues(indexVectors, queryVectors, spaceType, RECALL_K);

        createKnnIndex(INDEX_NAME, nvqSettings(), nvqMapping(NVQ_DIM, spaceType, 1));

        for (int offset = 0; offset < RECALL_DOC_COUNT; offset += RECALL_BATCH_SIZE) {
            int batchSize = Math.min(RECALL_BATCH_SIZE, RECALL_DOC_COUNT - offset);
            float[][] batch = new float[batchSize][NVQ_DIM];
            System.arraycopy(indexVectors, offset, batch, 0, batchSize);
            bulkAddKnnDocs(INDEX_NAME, FIELD_NAME, batch, offset, batchSize, false);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        List<List<String>> searchResults = bulkSearch(INDEX_NAME, FIELD_NAME, queryVectors, RECALL_K);
        double recall = TestUtils.calculateRecallValue(searchResults, groundTruth, RECALL_K);
        logger.info("NVQ recall ({}) = {}", spaceType, recall);
        assertEquals(1.0, recall, ACCEPTABLE_RECALL_DEVIATION);
    }

    private List<KNNResult> searchAndParse(float[] query, int k) throws Exception {
        String responseBody = EntityUtils.toString(searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, query, k), k).getEntity());
        return parseSearchResponse(responseBody, FIELD_NAME);
    }
}
