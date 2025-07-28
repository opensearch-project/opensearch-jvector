/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.junit.Test;
import org.opensearch.client.*;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.jvector.JVectorFormat;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.stats.StatNames;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import static org.opensearch.common.xcontent.XContentFactory.jsonBuilder;
import static org.opensearch.index.engine.EngineConfig.INDEX_USE_COMPOUND_FILE;
import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.index.engine.CommonTestUtils.*;

public class JVectorEngineIT extends KNNRestTestCase {

    @After
    public final void cleanUp() throws IOException {
        deleteKNNIndex(INDEX_NAME);
    }

    public void testQuery_l2() throws Exception {
        baseQueryTest(SpaceType.L2);
    }

    public void testQuery_cosine() throws Exception {
        baseQueryTest(SpaceType.COSINESIMIL);
    }

    public void testQuery_invalidVectorDimensionInQuery() throws Exception {

        createKnnIndexMappingWithJVectorEngine(CommonTestUtils.DIMENSION, SpaceType.L2, VectorDataType.FLOAT);
        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        float[] invalidQuery = new float[CommonTestUtils.DIMENSION - 1];
        int validK = 1;
        expectThrows(
            ResponseException.class,
            () -> searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, invalidQuery, validK), validK)
        );
    }

    public void testQuery_documentsMissingField() throws Exception {

        SpaceType spaceType = SpaceType.L2;

        createKnnIndexMappingWithJVectorEngine(CommonTestUtils.DIMENSION, spaceType, VectorDataType.FLOAT);
        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        // Add a doc without the lucene field set
        String secondField = "field-2";
        addDocWithNumericField(INDEX_NAME, Integer.toString(TEST_INDEX_VECTORS.length + 1), secondField, 0L);

        validateQueries(spaceType, FIELD_NAME);
    }

    public void testAddDoc() throws Exception {
        List<Integer> mValues = ImmutableList.of(16, 32, 64, 128);
        List<Integer> efConstructionValues = ImmutableList.of(16, 32, 64, 128);

        XContentBuilder builder = jsonBuilder().startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, CommonTestUtils.DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, DISK_ANN)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();
        Settings indexSettings = getDefaultIndexSettings();
        /* For now we will disable compound file to make it easier to read the index files */
        // TODO: Enable compound file once we have a way to read it with jVector files
        indexSettings = Settings.builder().put(indexSettings).put(INDEX_USE_COMPOUND_FILE.getKey(), false).build();

        createKnnIndex(INDEX_NAME, indexSettings, mapping);
        assertEquals(new TreeMap<>(mappingMap), new TreeMap<>(getIndexMappingAsMap(INDEX_NAME)));

        Float[] vector = new Float[] { 2.0f, 4.5f, 6.5f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        refreshIndex(INDEX_NAME);
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    public void testUpdateDoc() throws Exception {
        createKnnIndexMappingWithJVectorEngine(2, SpaceType.L2, VectorDataType.FLOAT);
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        Float[] updatedVector = { 8.0f, 8.0f };
        updateKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, updatedVector);

        refreshIndex(INDEX_NAME);
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    public void testDeleteDoc() throws Exception {
        createKnnIndexMappingWithJVectorEngine(2, SpaceType.L2, VectorDataType.FLOAT);
        Float[] vector = { 6.0f, 6.0f };
        addKnnDoc(INDEX_NAME, DOC_ID, FIELD_NAME, vector);

        deleteKnnDoc(INDEX_NAME, DOC_ID);

        refreshIndex(INDEX_NAME);
        assertEquals(0, getDocCount(INDEX_NAME));
    }

    public void testQueryWithFilterUsingFloatVectorDataType() throws Exception {
        createKnnIndexMappingWithJVectorEngine(CommonTestUtils.DIMENSION, SpaceType.L2, VectorDataType.FLOAT);

        addKnnDocWithAttributes(
            DOC_ID,
            new float[] { 6.0f, 7.9f, 3.1f },
            ImmutableMap.of(COLOR_FIELD_NAME, "red", TASTE_FIELD_NAME, "sweet")
        );
        addKnnDocWithAttributes(DOC_ID_2, new float[] { 3.2f, 2.1f, 4.8f }, ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes(DOC_ID_3, new float[] { 4.1f, 5.0f, 7.1f }, ImmutableMap.of(COLOR_FIELD_NAME, "red"));

        refreshIndex(INDEX_NAME);

        final float[] searchVector = { 6.0f, 6.0f, 4.1f };
        List<String> expectedDocIdsKGreaterThanFilterResult = Arrays.asList(DOC_ID, DOC_ID_3);
        List<String> expectedDocIdsKLimitsFilterResult = Arrays.asList(DOC_ID);
        validateQueryResultsWithFilters(searchVector, 5, 1, expectedDocIdsKGreaterThanFilterResult, expectedDocIdsKLimitsFilterResult);
    }

    @SneakyThrows
    public void testQueryWithFilterMultipleShards() {
        XContentBuilder builder = jsonBuilder().startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, CommonTestUtils.DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, DISK_ANN)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .endObject()
            .endObject()
            .startObject(INTEGER_FIELD_NAME)
            .field(TYPE_FIELD_NAME, FILED_TYPE_INTEGER)
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();

        createIndex(INDEX_NAME, Settings.builder().put("number_of_shards", 10).put("number_of_replicas", 0).put("index.knn", true).build());
        putMappingRequest(INDEX_NAME, mapping);

        addKnnDocWithAttributes("doc1", new float[] { 7.0f, 7.0f, 3.0f }, ImmutableMap.of("dateReceived", "2024-10-01"));

        refreshIndex(INDEX_NAME);

        final float[] searchVector = { 6.0f, 7.0f, 3.0f };
        final Response response = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(
                FIELD_NAME,
                searchVector,
                1,
                QueryBuilders.boolQuery().must(QueryBuilders.rangeQuery("dateReceived").gte("2023-11-01"))
            ),
            10
        );
        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(1, knnResults.size());
    }

    @SneakyThrows
    public void testQueryWithFilter_whenNonExistingFieldUsedInFilter_thenSuccessful() {
        XContentBuilder builder = jsonBuilder().startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, CommonTestUtils.DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, DISK_ANN)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .endObject()
            .endObject()
            .startObject(INTEGER_FIELD_NAME)
            .field(TYPE_FIELD_NAME, FILED_TYPE_INTEGER)
            .endObject()
            .endObject()
            .endObject();
        Map<String, Object> mappingMap = xContentBuilderToMap(builder);
        String mapping = builder.toString();

        createKnnIndex(INDEX_NAME, mapping);

        Float[] vector = new Float[] { 2.0f, 4.5f, 6.5f };

        String documentAsString = jsonBuilder().startObject().field(INTEGER_FIELD_NAME, 5).field(FIELD_NAME, vector).endObject().toString();

        addKnnDoc(INDEX_NAME, DOC_ID, documentAsString);

        refreshIndex(INDEX_NAME);
        assertEquals(1, getDocCount(INDEX_NAME));

        float[] searchVector = new float[] { 1.0f, 2.1f, 3.9f };
        int k = 10;

        // use filter where nonexistent field is must, we should have no results
        QueryBuilder filterWithRequiredNonExistentField = QueryBuilders.boolQuery()
            .must(QueryBuilders.rangeQuery(NON_EXISTENT_INTEGER_FIELD_NAME).gte(1));
        Response searchWithRequiredNonExistentFiledInFilterResponse = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, k, filterWithRequiredNonExistentField),
            k
        );
        List<KNNResult> resultsQuery1 = parseSearchResponse(
            EntityUtils.toString(searchWithRequiredNonExistentFiledInFilterResponse.getEntity()),
            FIELD_NAME
        );
        assertTrue(resultsQuery1.isEmpty());

        // use filter with non existent field as optional, we should have some results
        QueryBuilder filterWithOptionalNonExistentField = QueryBuilders.boolQuery()
            .should(QueryBuilders.rangeQuery(NON_EXISTENT_INTEGER_FIELD_NAME).gte(1))
            .must(QueryBuilders.rangeQuery(INTEGER_FIELD_NAME).gte(1));
        Response searchWithOptionalNonExistentFiledInFilterResponse = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, k, filterWithOptionalNonExistentField),
            k
        );
        List<KNNResult> resultsQuery2 = parseSearchResponse(
            EntityUtils.toString(searchWithOptionalNonExistentFiledInFilterResponse.getEntity()),
            FIELD_NAME
        );
        assertEquals(1, resultsQuery2.size());
    }

    public void testIndexReopening() throws Exception {
        createKnnIndexMappingWithJVectorEngine(CommonTestUtils.DIMENSION, SpaceType.L2, VectorDataType.FLOAT);

        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        final float[] searchVector = TEST_QUERY_VECTORS[0];
        final int k = 1 + RandomUtils.nextInt(TEST_INDEX_VECTORS.length);

        final List<float[]> knnResultsBeforeIndexClosure = queryResults(searchVector, k);

        closeIndex(INDEX_NAME);
        openIndex(INDEX_NAME);

        ensureGreen(INDEX_NAME);

        final List<float[]> knnResultsAfterIndexClosure = queryResults(searchVector, k);

        assertArrayEquals(knnResultsBeforeIndexClosure.toArray(), knnResultsAfterIndexClosure.toArray());
    }

    /**
     * Verify that the jVector-specific KNN stats counters
     * (visited, expanded, expanded-base-layer nodes) are present
     * and increase after a KNN search is executed.
     */
    public void testJVectorSearchStatsIncrement() throws Exception {
        /* ---------------------------------------------------
         * 1.  Create index and docs
         * --------------------------------------------------- */
        int dimension = 128;
        int vectorsCount = 2050; // we create 2050 docs to have PQ and re-rank kick in
        createKnnIndexMappingWithJVectorEngine(dimension, SpaceType.L2, VectorDataType.FLOAT);

        for (int j = 0; j < vectorsCount; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, randomFloatVector(dimension));
        }
        flushIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        /* ---------------------------------------------------
         * 2.  Read initial stats
         * --------------------------------------------------- */
        List<String> metrics = List.of(
            StatNames.KNN_QUERY_VISITED_NODES.getName(),
            StatNames.KNN_QUERY_RERANKED_COUNT.getName(),
            StatNames.KNN_QUERY_EXPANDED_NODES.getName(),
            StatNames.KNN_QUERY_EXPANDED_BASE_LAYER_NODES.getName(),
            StatNames.KNN_QUERY_GRAPH_SEARCH_TIME.getName(),
            StatNames.KNN_QUANTIZATION_TRAINING_TIME.getName(),
            StatNames.KNN_GRAPH_MERGE_TIME.getName()
        );

        var parsedBefore = parseNodeStatsResponse(EntityUtils.toString(getKnnStats(Collections.emptyList(), metrics).getEntity()));
        assertNotNull(parsedBefore);
        assertEquals(1, parsedBefore.size());
        Map<String, Object> before = parsedBefore.get(0);
        assertNotNull(before);

        /* ---------------------------------------------------
         * 3.  Execute a KNN query
         * --------------------------------------------------- */
        final float[] searchVector = randomFloatVector(dimension);
        int k = 5;
        var response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, searchVector, k), k);
        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);
        assertNotNull(knnResults);
        assertEquals(k, knnResults.size());

        /* ---------------------------------------------------
         * 4.  Read stats again and assert they have increased
         * --------------------------------------------------- */
        var parsedAfter = parseNodeStatsResponse(EntityUtils.toString(getKnnStats(Collections.emptyList(), metrics).getEntity()));
        assertNotNull(parsedAfter);
        assertEquals(1, parsedAfter.size());
        Map<String, Object> after = parsedAfter.get(0);

        assertNotNull(after);

        for (String metric : metrics) {
            assertNotNull(after.get(metric));
            // Check that our metrics increased
            assertTrue(String.format("Metric %s, didn't increase", metric), ((Number)after.get(metric)).longValue() > ((Number)before.get(metric)).longValue());
        }
    }

    /**
     * Verify that once a document has been deleted, it is no longer returned
     * by a k-NN search that previously matched it.
     */
    public void testSearchAfterDeleteDoc() throws Exception {
        int dimension = 2;
        createKnnIndexMappingWithJVectorEngine(dimension, SpaceType.L2, VectorDataType.FLOAT);

        // ----------------------------
        // 1. Index three simple docs
        // ----------------------------
        Map<String, float[]> docs = Map.of(
            "1",
            new float[] { 0.0f, 0.2f },
            "2",
            new float[] { 0.0f, 0.4f },
            "3",
            new float[] { 0.0f, 0.6f }
        );

        for (Map.Entry<String, float[]> entry : docs.entrySet()) {
            addKnnDoc(INDEX_NAME, entry.getKey(), FIELD_NAME, entry.getValue());
        }
        refreshAllIndices();

        float[] target = { 0.0f, 0.0f };

        // Ensure the nearest neighbor is doc "1"
        List<float[]> searchResults = queryResults(target, 1);
        assertEquals(1, searchResults.size());
        assertArrayEquals(docs.get("1"), searchResults.get(0), 0.0f);

        // -----------------
        // 2. Delete doc "1"
        // -----------------
        deleteKnnDoc(INDEX_NAME, "1");
        refreshAllIndices();

        // ------------------------------------
        // 3. Search again – doc "1" is gone
        // ------------------------------------
        searchResults = queryResults(target, 3);
        assertEquals(2, searchResults.size()); // only two live docs left
        // The first result should now be "2"
        assertArrayEquals(docs.get("2"), searchResults.get(0), 0.0f);
    }

    private List<float[]> queryResults(final float[] searchVector, final int k) throws Exception {
        final String responseBody = EntityUtils.toString(
            searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, searchVector, k), k).getEntity()
        );
        final List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);
        assertNotNull(knnResults);
        return knnResults.stream().map(KNNResult::getVector).collect(Collectors.toUnmodifiableList());
    }

    private void validateQueryResultsWithFilters(
        float[] searchVector,
        int kGreaterThanFilterResult,
        int kLimitsFilterResult,
        List<String> expectedDocIdsKGreaterThanFilterResult,
        List<String> expectedDocIdsKLimitsFilterResult
    ) throws IOException, ParseException {
        final Response response = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, kGreaterThanFilterResult, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
            kGreaterThanFilterResult
        );
        final String responseBody = EntityUtils.toString(response.getEntity());
        final List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);

        assertEquals(expectedDocIdsKGreaterThanFilterResult.size(), knnResults.size());
        assertTrue(
            knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toList()).containsAll(expectedDocIdsKGreaterThanFilterResult)
        );

        final Response responseKLimitsFilterResult = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, searchVector, kLimitsFilterResult, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
            kLimitsFilterResult
        );
        final String responseBodyKLimitsFilterResult = EntityUtils.toString(responseKLimitsFilterResult.getEntity());
        final List<KNNResult> knnResultsKLimitsFilterResult = parseSearchResponse(responseBodyKLimitsFilterResult, FIELD_NAME);

        assertEquals(expectedDocIdsKLimitsFilterResult.size(), knnResultsKLimitsFilterResult.size());
        assertTrue(
            knnResultsKLimitsFilterResult.stream()
                .map(KNNResult::getDocId)
                .collect(Collectors.toList())
                .containsAll(expectedDocIdsKLimitsFilterResult)
        );
    }

    private void createKnnIndexMappingWithJVectorEngine(int dimension, SpaceType spaceType, VectorDataType vectorDataType)
        throws Exception {
        String mapping = CommonTestUtils.createIndexMapping(dimension, spaceType, vectorDataType);
        Settings indexSettings = CommonTestUtils.getDefaultIndexSettings();
        // indexSettings = Settings.builder().put(indexSettings).put(INDEX_USE_COMPOUND_FILE.getKey(), false).build();
        createKnnIndex(INDEX_NAME, indexSettings, mapping);
    }

    private void baseQueryTest(SpaceType spaceType) throws Exception {

        createKnnIndexMappingWithJVectorEngine(CommonTestUtils.DIMENSION, spaceType, VectorDataType.FLOAT);
        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        validateQueries(spaceType, FIELD_NAME);
        validateQueries(spaceType, FIELD_NAME, Map.of("ef_search", 100));
    }

    private void validateQueries(SpaceType spaceType, String fieldName) throws Exception {
        validateQueries(spaceType, fieldName, null);
    }

    private void validateQueries(SpaceType spaceType, String fieldName, Map<String, ?> methodParameters) throws Exception {

        int k = CommonTestUtils.TEST_INDEX_VECTORS.length;
        for (float[] queryVector : TEST_QUERY_VECTORS) {
            Response response = searchKNNIndex(INDEX_NAME, buildSearchQuery(fieldName, k, queryVector, methodParameters), k);
            String responseBody = EntityUtils.toString(response.getEntity());
            List<KNNResult> knnResults = parseSearchResponse(responseBody, fieldName);
            assertEquals(k, knnResults.size());

            List<Float> actualScores = parseSearchResponseScore(responseBody, fieldName);
            for (int j = 0; j < k; j++) {
                float[] primitiveArray = knnResults.get(j).getVector();
                float distance = TestUtils.computeDistFromSpaceType(spaceType, primitiveArray, queryVector);
                float rawScore = VECTOR_SIMILARITY_TO_SCORE.get(spaceType.getKnnVectorSimilarityFunction()).apply(distance);
                assertEquals(KNNEngine.LUCENE.score(rawScore, spaceType), actualScores.get(j), 0.0001);
            }
        }
    }

    /**
     * Test that verifies JVector's behavior with mixed batch sizes for quantization.
     * Some batches will be smaller than DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION
     * and some will be larger.
     */
    public void testMixedBatchSizesForQuantization() throws Exception {
        int dimension = 16;
        final SpaceType spaceType = SpaceType.L2;
        createKnnIndexMappingWithJVectorEngine(dimension, spaceType, VectorDataType.FLOAT);

        // Define batch sizes - mix of small and large batches
        int smallBatchSize = JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION / 4;
        int largeBatchSize = JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION * 2;

        final float[][] vectors = new float[smallBatchSize + largeBatchSize + smallBatchSize][dimension];
        for (int i = 0; i < vectors.length; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = randomFloat();
            }
        }
        final int totalDocs = vectors.length;
        final int firstSmallBatchOffset = 0;
        final int largeBatchOffset = firstSmallBatchOffset + smallBatchSize;
        final int secondSmallBatchOffset = largeBatchOffset + largeBatchSize;

        // Add the first small batch (below quantization threshold)
        logger.info("Adding first small batch");
        for (int i = firstSmallBatchOffset; i < largeBatchOffset; i++) {
            addKnnDoc(INDEX_NAME, Integer.toString(i, 10), FIELD_NAME, vectors[i]);
        }
        flushIndex(INDEX_NAME);

        logger.info("Adding large batch");
        // Add the large batch (above quantization threshold)
        for (int i = largeBatchOffset; i < secondSmallBatchOffset; i++) {
            addKnnDoc(INDEX_NAME, Integer.toString(i, 10), FIELD_NAME, vectors[i]);
        }
        flushIndex(INDEX_NAME);

        logger.info("Adding second small batch");
        // Add another small batch
        for (int i = secondSmallBatchOffset; i < totalDocs; i++) {
            addKnnDoc(INDEX_NAME, Integer.toString(i, 10), FIELD_NAME, vectors[i]);
        }
        flushIndex(INDEX_NAME);

        // Force merge to trigger quantization for all segments
        logger.info("Force merging");
        forceMergeKnnIndex(INDEX_NAME);

        // Verify the total document count
        int expectedTotalDocs = vectors.length;
        assertEquals(expectedTotalDocs, getDocCount(INDEX_NAME));

        // Perform search and verify recall
        logger.info("Searching");
        float[] queryVector = randomFloatVector(dimension);
        int k = 10;

        // First search without forcing merge
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, k), k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        // Verify we got k results (or all docs if less than k)
        assertEquals(Math.min(k, expectedTotalDocs), results.size());

        // Calculate ground truth with brute force
        logger.info("Calculating ground truth");
        List<Set<String>> groundTruth = TestUtils.computeGroundTruthValues(vectors, new float[][] { queryVector }, spaceType, k);
        assertEquals(1, groundTruth.size());
        Set<String> expectedDocIds = groundTruth.getFirst();

        // calculate recall
        logger.info("Calculating recall");
        float recall = ((float) results.stream().filter(r -> expectedDocIds.contains(r.getDocId())).count()) / ((float) k);
        assertTrue("Expected recall to be at least 0.9 but got " + recall, recall >= 0.9);
    }

    /**
     * Test for recall with various overquery parameters.
     * We expect the recall to be higher the more we increase the overquery parameter
     * @throws Exception exception
     */
    @Test
    public void testQuantizationWithOverQueryParameter() throws Exception {
        int dimension = 512;
        final SpaceType spaceType = SpaceType.L2;
        createKnnIndexMappingWithJVectorEngine(dimension, spaceType, VectorDataType.FLOAT);

        // Choosing a batch size that will trigger quantization
        int batchSize = JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION * 2;

        final float[][] vectors = new float[batchSize][dimension];
        for (int i = 0; i < vectors.length; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = randomFloat();
            }
        }
        final int totalDocs = vectors.length;

        logger.info("Adding batch of vectors with size {} that is expected to trigger quantization", totalDocs);
        for (int i = 0; i < batchSize; i++) {
            addKnnDoc(INDEX_NAME, Integer.toString(i, 10), FIELD_NAME, vectors[i]);
        }
        flushIndex(INDEX_NAME);

        // Force merge to trigger quantization for all segments
        logger.info("Force merging just in case quantization didn't happen because of segment fragmentation");
        forceMergeKnnIndex(INDEX_NAME);

        // Verify the total document count
        int expectedTotalDocs = vectors.length;
        assertEquals(expectedTotalDocs, getDocCount(INDEX_NAME));

        // Perform search and verify recall
        float[] queryVector = randomFloatVector(dimension);
        int k = 10;

        // Calculate ground truth with brute force
        logger.info("Calculating ground truth");
        List<Set<String>> groundTruth = TestUtils.computeGroundTruthValues(vectors, new float[][] { queryVector }, spaceType, k);
        assertEquals(1, groundTruth.size());
        Set<String> expectedDocIds = groundTruth.getFirst();

        logger.info("Searching with low overquery factor");
        // 1. Search with a low-overquery factor
        final Map<String, Object> methodParametersWithLowOverQuery = new HashMap<>();
        methodParametersWithLowOverQuery.put(KNNConstants.METHOD_PARAMETER_OVERQUERY_FACTOR, 1);
        methodParametersWithLowOverQuery.put(KNNConstants.METHOD_PARAMETER_THRESHOLD, DEFAULT_QUERY_SIMILARITY_THRESHOLD);
        methodParametersWithLowOverQuery.put(METHOD_PARAMETER_RERANK_FLOOR, DEFAULT_QUERY_RERANK_FLOOR);

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .k(k)
            .vector(queryVector)
            .fieldName(FIELD_NAME)
            .methodParameters(methodParametersWithLowOverQuery)
            .build();
        Response response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        // Verify we got k results (or all docs if less than k)
        assertEquals(Math.min(k, expectedTotalDocs), results.size());

        // calculate recall
        logger.info("Calculating recall");
        float recall = ((float) results.stream().filter(r -> expectedDocIds.contains(r.getDocId())).count()) / ((float) k);
        assertTrue("Expected recall to be lower than 0.7 but got " + recall, recall < 0.7);

        // 2. Search with a high-overquery factor
        logger.info("Searching with high overquery factor");
        final Map<String, Object> methodParametersWithHighOverQuery = new HashMap<>();
        methodParametersWithHighOverQuery.put(KNNConstants.METHOD_PARAMETER_OVERQUERY_FACTOR, 100);
        methodParametersWithHighOverQuery.put(KNNConstants.METHOD_PARAMETER_THRESHOLD, DEFAULT_QUERY_SIMILARITY_THRESHOLD);
        methodParametersWithHighOverQuery.put(METHOD_PARAMETER_RERANK_FLOOR, DEFAULT_QUERY_RERANK_FLOOR);

        knnQueryBuilder = KNNQueryBuilder.builder()
            .k(k)
            .vector(queryVector)
            .fieldName(FIELD_NAME)
            .methodParameters(methodParametersWithHighOverQuery)
            .build();
        response = searchKNNIndex(INDEX_NAME, knnQueryBuilder, k);
        results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        // Verify we got k results (or all docs if less than k)
        assertEquals(Math.min(k, expectedTotalDocs), results.size());

        // calculate recall
        logger.info("Calculating recall");
        recall = ((float) results.stream().filter(r -> expectedDocIds.contains(r.getDocId())).count()) / ((float) k);
        assertTrue("Expected recall to be at least 0.9 but got " + recall, recall >= 0.8);
    }
}
