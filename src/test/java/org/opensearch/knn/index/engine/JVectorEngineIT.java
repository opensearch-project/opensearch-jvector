/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

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
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.KNNQueryBuilder;

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
            .field("doc_values", true)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, CommonTestUtils.DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, DISK_ANN)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .startObject(KNNConstants.PARAMETERS)
            // Adding all the parameters here (even though we could have just let it use whatever defaults) just to make sure everything is
            // correctly wired
            .field(KNNConstants.METHOD_PARAMETER_M, mValues.get(random().nextInt(mValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, efConstructionValues.get(random().nextInt(efConstructionValues.size())))
            .field(KNNConstants.METHOD_PARAMETER_ALPHA, 1.2)
            .field(KNNConstants.METHOD_PARAMETER_NEIGHBOR_OVERFLOW, 2.0)
            .field(KNNConstants.METHOD_PARAMETER_HIERARCHY_ENABLED, false)
            .field(KNNConstants.METHOD_PARAMETER_MIN_BATCH_SIZE_FOR_QUANTIZATION, 1000)
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
            searchKNNIndex(
                INDEX_NAME,
                new KNNQueryBuilder(FIELD_NAME, searchVector, k, new TermQueryBuilder(TERM_QUERY_FIELD_NAME, TERM_QUERY_FIELD_VALUE)),
                k
            ).getEntity()
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
            Response response = searchKNNIndex(
                INDEX_NAME,
                new KNNQueryBuilder(fieldName, queryVector, k, new TermQueryBuilder(TERM_QUERY_FIELD_NAME, TERM_QUERY_FIELD_VALUE)),
                k
            );
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
}
