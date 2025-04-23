/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.action.bulk.BulkResponse;
import org.opensearch.client.*;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.http.ParseException;
import org.apache.http.util.EntityUtils;
import org.junit.After;
import org.opensearch.action.bulk.BulkRequest;
import org.opensearch.action.index.IndexRequest;
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
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import static org.opensearch.common.xcontent.XContentFactory.jsonBuilder;
import static org.opensearch.index.engine.EngineConfig.INDEX_USE_COMPOUND_FILE;
import static org.opensearch.knn.common.KNNConstants.DISK_ANN;
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

        createKnnIndexMappingWithJVectorEngine(DIMENSION, SpaceType.L2, VectorDataType.FLOAT);
        for (int j = 0; j < TEST_INDEX_VECTORS.length; j++) {
            addKnnDoc(INDEX_NAME, Integer.toString(j + 1), FIELD_NAME, TEST_INDEX_VECTORS[j]);
        }

        float[] invalidQuery = new float[DIMENSION - 1];
        int validK = 1;
        expectThrows(
            ResponseException.class,
            () -> searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, invalidQuery, validK), validK)
        );
    }

    public void testQuery_documentsMissingField() throws Exception {

        SpaceType spaceType = SpaceType.L2;

        createKnnIndexMappingWithJVectorEngine(DIMENSION, spaceType, VectorDataType.FLOAT);
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
            .field(DIMENSION_FIELD_NAME, DIMENSION)
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
        createKnnIndexMappingWithJVectorEngine(DIMENSION, SpaceType.L2, VectorDataType.FLOAT);

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
            .field(DIMENSION_FIELD_NAME, DIMENSION)
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
            .field(DIMENSION_FIELD_NAME, DIMENSION)
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
        createKnnIndexMappingWithJVectorEngine(DIMENSION, SpaceType.L2, VectorDataType.FLOAT);

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
     * Test indexing and batch updating of documents with testing for ordering issues
     * The test is testing two edge scenarios that may occur when rebuilding/merging the graph segments:
     * 1. No regressions in the end to end ordering of documents within search results
     * 2. Mapping of document IDs to vectors is not lost on multiple merges
     */
    public void testBatchUpdate_withDocOrderingTest() throws Exception {
        final int dimension = 3;
        logger.info("Starting testBatchUpdate with vector dimension: {}", dimension);

        createKnnIndexMappingWithJVectorEngine(dimension, SpaceType.L2, VectorDataType.FLOAT);
        logger.info("Created KNN index mapping with JVector engine");

        RestClientBuilder builder = RestClient.builder(client().getNodes().toArray(new Node[0]));
        try (final RestHighLevelClient highLevelClient = new RestHighLevelClient(builder)) {
            // Create 1000 documents
            final int totalDocuments = 1000;
            final int batchSize = 10;
            final int totalBatches = totalDocuments / batchSize;

            // Store vectors for later recall calculation
            final Map<String, float[]> docVectors = generateDocVectors(dimension, totalDocuments);

            logger.info("Starting to index {} documents in {} batches of {} documents each", totalDocuments, totalBatches, batchSize);

            long indexingStartTime = System.currentTimeMillis();
            addUpdateDocsInBatches(docVectors, batchSize, highLevelClient);
            long indexingEndTime = System.currentTimeMillis();
            long indexingDuration = indexingEndTime - indexingStartTime;

            // Final refresh and verify document count
            refreshIndex(INDEX_NAME);
            long actualDocCount = getDocCount(INDEX_NAME);
            logger.info(
                "Completed indexing {} documents in {} ms ({} docs/sec)",
                actualDocCount,
                indexingDuration,
                String.format("%.1f", actualDocCount * 1000.0 / indexingDuration)
            );
            assertEquals(totalDocuments, actualDocCount);

            // Create several query vectors for comprehensive testing
            final int queryCount = 5;
            List<float[]> queryVectors = new ArrayList<>(queryCount);
            for (int q = 0; q < queryCount; q++) {
                float[] queryVector = new float[dimension];
                for (int i = 0; i < dimension; i++) {
                    queryVector[i] = random().nextFloat();
                }
                queryVectors.add(queryVector);
            }

            // Search with each query vector before updates to establish baseline
            final int k = 20; // Higher k for better recall assessment
            Map<Integer, Set<String>> preUpdateResults = new HashMap<>();

            logger.info("Performing baseline searches before updates");
            for (int q = 0; q < queryCount; q++) {
                float[] queryVector = queryVectors.get(q);
                Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, k), k);
                String responseBody = EntityUtils.toString(response.getEntity());
                List<KNNResult> knnResults = parseSearchResponse(responseBody, FIELD_NAME);

                // Store the document IDs from the results
                preUpdateResults.put(q, knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet()));
            }

            logger.info("Starting to update {} documents in {} batches of {} documents each", totalDocuments, totalBatches, batchSize);

            long updateStartTime = System.currentTimeMillis();
            addUpdateDocsInBatches(docVectors, batchSize, highLevelClient);
            long updateEndTime = System.currentTimeMillis();
            long updateDuration = updateEndTime - updateStartTime;

            // Final refresh and verify document count remains the same
            refreshIndex(INDEX_NAME);
            actualDocCount = getDocCount(INDEX_NAME);
            logger.info(
                "Completed updating {} documents in {} ms ({} docs/sec)",
                actualDocCount,
                updateDuration,
                String.format("%.1f", actualDocCount * 1000.0 / updateDuration)
            );
            assertEquals(totalDocuments, actualDocCount);

            logger.info("Performing test searches after updates");

            // Search with each query vector after updates and check if the ordering is the same
            double orderingRecall = 0.0;

            for (int q = 0; q < queryCount; q++) {
                float[] queryVector = queryVectors.get(q);
                Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, k), k);

                String responseBody = EntityUtils.toString(response.getEntity());
                List<KNNResult> postUpdateResults = parseSearchResponse(responseBody, FIELD_NAME);

                // Calculate recall for this query
                Set<String> preUpdateDocsForQuery = preUpdateResults.get(q);
                Set<String> postUpdateDocsForQuery = postUpdateResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet());

                // Count how many docs from pre-update results are still in post-update results
                Set<String> commonDocs = new HashSet<>(preUpdateDocsForQuery);
                commonDocs.retainAll(postUpdateDocsForQuery);

                double queryRecall = (double) commonDocs.size() / preUpdateDocsForQuery.size();
                orderingRecall += queryRecall;

                logger.info("Query {} recall: {}", q, String.format("%.2f", queryRecall));
            }

            // Calculate average recall across all queries
            double avgRecall = orderingRecall / queryCount;
            logger.info("Average recall after updates: {}", String.format("%.2f", avgRecall));

            // Assert acceptable recall level (may need to adjust threshold)
            assertEquals("Recall after updates is too low: " + avgRecall, 1.0, avgRecall, 0.001);

            logger.info("testBatchUpdate completed successfully");
        }
    }

    private void addUpdateDocsInBatches(Map<String, float[]> docVectors, int batchSize, RestHighLevelClient highLevelClient)
        throws Exception {

        // Add documents in batches of batchSize
        int batch = 0;
        int remainingInBatch = 0;
        int totalDocuments = docVectors.size();
        final int totalBatches = totalDocuments / batchSize;
        BulkRequest bulkRequest = new BulkRequest(INDEX_NAME);
        for (Map.Entry<String, float[]> entry : docVectors.entrySet()) {
            String docId = entry.getKey();
            float[] vector = entry.getValue();
            // Add to bulk request with unique numeric identifier field
            bulkRequest.add(new IndexRequest().id(docId).source(jsonBuilder().startObject().field(FIELD_NAME, vector).endObject()));
            remainingInBatch++;
            // Trigger a batch
            if (remainingInBatch == batchSize) {
                // Execute bulk request
                final BulkResponse bulkResponse = highLevelClient.bulk(bulkRequest, RequestOptions.DEFAULT);

                // Refresh occasionally to avoid too much memory pressure
                if (batch % 10 == 0) {
                    refreshIndex(INDEX_NAME);
                    logger.info(
                        "Indexed batches: {}/{} ({}%)",
                        batch + 1,
                        totalBatches,
                        String.format("%.1f", (batch + 1) * 100.0 / totalBatches)
                    );
                }
                batch++;
                remainingInBatch = 0;
                bulkRequest = new BulkRequest(INDEX_NAME);
            }
        }
        if (remainingInBatch > 0) {
            logger.info("Indexing final batch of {} documents which is less than batchSize: {}", remainingInBatch, batchSize);
            final BulkResponse bulkResponse = highLevelClient.bulk(bulkRequest, RequestOptions.DEFAULT);
        }

    }

    private Map<String, float[]> generateDocVectors(int dimension, int totalDocuments) {
        Map<String, float[]> docVectors = new HashMap<>(totalDocuments);
        for (int i = 0; i < totalDocuments; i++) {
            String docId = "doc_" + i;

            // Create random vector
            float[] vector = generateRandomVector(dimension);

            // Store vector for recall comparison
            docVectors.put(docId, vector);
        }

        return docVectors;
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

        createKnnIndexMappingWithJVectorEngine(DIMENSION, spaceType, VectorDataType.FLOAT);
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
     * Generates a random float vector of the specified dimension.
     * Each element in the vector will be a random float value between 0.0 and 1.0.
     *
     * @param dimension The dimension of the vector to generate
     * @return A float array representing the random vector
     */
    public static float[] generateRandomVector(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be a positive integer");
        }

        final float[] vector = new float[dimension];

        for (int i = 0; i < dimension; i++) {
            vector[i] = ThreadLocalRandom.current().nextFloat();
        }

        return vector;
    }

}
