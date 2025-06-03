/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.recall;

import lombok.SneakyThrows;
import org.junit.Before;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.DISK_ANN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY;
import static org.opensearch.knn.index.KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED;

/**
 * Tests confirm that for the different supported configurations, recall is sound. The recall thresholds are
 * conservatively and empirically determined to prevent flakiness.
 *
 */
public class RecallTestsIT extends KNNRestTestCase {

    private static final String PROPERTIES_FIELD = "properties";
    private final static String TEST_INDEX_PREFIX_NAME = "test_index";
    private final static String TEST_FIELD_NAME = "test_field";
    private final static String TRAIN_INDEX_NAME = "train_index";
    private final static String TRAIN_FIELD_NAME = "train_field";
    private final static int TEST_DIMENSION = 32;
    private final static int DOC_COUNT = 100000;
    private final static int QUERY_COUNT = 100;
    private final static int TEST_K = 100;
    private final static double PERFECT_RECALL = 1.0;
    private final static int SHARD_COUNT = 1;
    private final static int REPLICA_COUNT = 0;
    private final static int MAX_SEGMENT_COUNT = 1;

    // Standard algorithm parameters
    private final static int HNSW_M = 16;
    private final static int HNSW_EF_CONSTRUCTION = 100;

    // Setup ground truth for all tests once
    private final static float[][] INDEX_VECTORS = TestUtils.getIndexVectors(DOC_COUNT, TEST_DIMENSION, true);
    private final static float[][] QUERY_VECTORS = TestUtils.getQueryVectors(QUERY_COUNT, TEST_DIMENSION, DOC_COUNT, true);
    private final static Map<SpaceType, List<Set<String>>> GROUND_TRUTH = Map.of(
            SpaceType.L2,
            TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.L2, TEST_K),
            SpaceType.COSINESIMIL,
            TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.COSINESIMIL, TEST_K),
            SpaceType.INNER_PRODUCT,
            TestUtils.computeGroundTruthValues(INDEX_VECTORS, QUERY_VECTORS, SpaceType.INNER_PRODUCT, TEST_K)
    );

    @SneakyThrows
    @Before
    public void setupClusterSettings() {
        updateClusterSettings(KNN_ALGO_PARAM_INDEX_THREAD_QTY, 2);
        updateClusterSettings(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, true);
    }

    /**
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {DIMENSION},
     *      "method": {
     *          "name":"hnsw",
     *          "engine":"lucene",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION}
     *          }
     *       }
     *     }
     *   }
     * }
     */
    @SneakyThrows
    public void testRecall_whenLuceneHnswFP32_thenRecallAbove75percent() {
        List<SpaceType> spaceTypes = List.of(SpaceType.L2, SpaceType.COSINESIMIL);
        for (SpaceType spaceType : spaceTypes) {
            String indexName = createIndexName(KNNEngine.LUCENE, spaceType);
            XContentBuilder builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES_FIELD)
                    .startObject(TEST_FIELD_NAME)
                    .field(TYPE, TYPE_KNN_VECTOR)
                    .field(DIMENSION, TEST_DIMENSION)
                    .startObject(KNN_METHOD)
                    .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                    .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
                    .field(NAME, METHOD_HNSW)
                    .startObject(PARAMETERS)
                    .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
                    .field(METHOD_PARAMETER_M, HNSW_M)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject();
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), builder.toString());
            assertRecall(indexName, spaceType, 0.25f);
        }
    }

    /**
     * {
     * 	"properties": {
     *     {
     *      "type": "knn_vector",
     *      "dimension": {TEST_DIMENSION},
     *      "method": {
     *          "name":"diskann",
     *          "engine":"jvector",
     *          "space_type": "{SPACE_TYPE}",
     *          "parameters":{
     *              "m":{HNSW_M},
     *              "ef_construction": {HNSW_EF_CONSTRUCTION},
     *              "ef_search": {HNSW_EF_SEARCH},
     *          }
     *       }
     *     }
     *   }
     * }
     */
    @SneakyThrows
    public void testRecall_whenJVectorDiskANN_thenRecallAbove75percent() {
        List<SpaceType> spaceTypes = List.of(SpaceType.L2, SpaceType.COSINESIMIL);
        for (SpaceType spaceType : spaceTypes) {
            String indexName = createIndexName(KNNEngine.JVECTOR, spaceType);
            XContentBuilder builder = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES_FIELD)
                    .startObject(TEST_FIELD_NAME)
                    .field(TYPE, TYPE_KNN_VECTOR)
                    .field(DIMENSION, TEST_DIMENSION)
                    .startObject(KNN_METHOD)
                    .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
                    .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
                    .field(NAME, DISK_ANN)
                    .startObject(PARAMETERS)
                    .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
                    .field(METHOD_PARAMETER_M, HNSW_M)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject();
            createIndexAndIngestDocs(indexName, TEST_FIELD_NAME, getSettings(), builder.toString());
            assertRecall(indexName, spaceType, 0.25f);
        }
    }



    @SneakyThrows
    private void assertRecall(String testIndexName, SpaceType spaceType, float acceptableRecallFromPerfect) {
        List<List<String>> searchResults = bulkSearch(testIndexName, TEST_FIELD_NAME, QUERY_VECTORS, TEST_K);
        double recallValue = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH.get(spaceType), TEST_K);
        logger.info("Recall value = {}", recallValue);
        assertEquals(PERFECT_RECALL, recallValue, acceptableRecallFromPerfect);
    }

    private String createIndexName(KNNEngine knnEngine, SpaceType spaceType) {
        return String.format("%s_%s_%s", TEST_INDEX_PREFIX_NAME, knnEngine.getName(), spaceType.getValue());
    }

    @SneakyThrows
    private void createIndexAndIngestDocs(String indexName, String fieldName, Settings settings, String mapping) {
        createKnnIndex(indexName, settings, mapping);
        bulkAddKnnDocs(indexName, fieldName, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, MAX_SEGMENT_COUNT);
    }

    @SneakyThrows
    private void setupTrainingIndex() {
        XContentBuilder trainingIndexBuilder = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES_FIELD)
                .startObject(TRAIN_FIELD_NAME)
                .field(TYPE, TYPE_KNN_VECTOR)
                .field(DIMENSION, TEST_DIMENSION)
                .endObject()
                .endObject()
                .endObject();
        createIndexAndIngestDocs(
                TRAIN_INDEX_NAME,
                TRAIN_FIELD_NAME,
                Settings.builder().put("number_of_shards", SHARD_COUNT).put("number_of_replicas", REPLICA_COUNT).build(),
                trainingIndexBuilder.toString()
        );
    }

    private Settings getSettings() {
        return Settings.builder()
                .put("number_of_shards", SHARD_COUNT)
                .put("number_of_replicas", REPLICA_COUNT)
                .put("index.knn", true)
                .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
                .build();
    }
}