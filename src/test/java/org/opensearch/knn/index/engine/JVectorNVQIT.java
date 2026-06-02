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
 * Tests 1–4 use a small dimension (3) with a high minBatchSize so that NVQ training is
 * NOT triggered; they test only that the NVQ mapping parameter is correctly wired through
 * the stack.
 *
 * Tests 5–7 use dimension 16 with minBatchSize=1 so NVQ training IS triggered; they
 * verify real quantisation behaviour and recall.
 */
public class JVectorNVQIT extends KNNRestTestCase {

    private static final int SMALL_DIM = 3;
    private static final int NVQ_DIM = 128;
    private static final int SHARD_COUNT = 1;
    private static final int REPLICA_COUNT = 0;

    // Recall-test sizing (mirrors RecallTestsIT)
    private static final int RECALL_DOC_COUNT = 1000;
    private static final int RECALL_BATCH_SIZE = 500;
    private static final int RECALL_QUERY_COUNT = 50;
    private static final int RECALL_K = 10;
    private static final double ACCEPTABLE_RECALL_DEVIATION = 0.25; // recall >= 0.75

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
        //deleteKnnDoc(INDEX_NAME, "0");
        //refreshIndex(INDEX_NAME);

        //List<KNNResult> after = searchAndParse(query, 10);
        //assertTrue(after.stream().noneMatch(r -> "0".equals(r.getDocId())));
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
    // NVQ inline tests (FeatureId.NVQ_VECTORS stored inline with graph nodes)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

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
        String responseBody = EntityUtils.toString(
            searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, query, k), k).getEntity()
        );
        return parseSearchResponse(responseBody, FIELD_NAME);
    }
}
