/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.engine;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import com.google.common.primitives.Floats;
import org.apache.http.util.EntityUtils;
import org.junit.Test;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.JVectorKNNPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchIntegTestCase;
import org.opensearch.transport.Netty4Plugin;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.opensearch.knn.KNNRestTestCase.FIELD_NAME;
import static org.opensearch.knn.KNNRestTestCase.INDEX_NAME;
import static org.opensearch.knn.index.engine.CommonTestUtils.DIMENSION;

@OpenSearchIntegTestCase.ClusterScope(scope = OpenSearchIntegTestCase.Scope.TEST, numDataNodes = 1)
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
public class JVectorConcurrentQueryTests extends OpenSearchIntegTestCase {

    private static final int NUM_VECTORS = 100;
    private static final int NUM_QUERIES = 10;
    private static final int NUM_CONCURRENT_QUERIES = 10; // Will be equivalent to the number of threads
    private static final int NUM_TOTAL_QUERIES_PER_THREAD = NUM_QUERIES * 50; // Iterate on the set of queries 50 times
    private static final int K = 5;
    private static final float[][] TEST_VECTORS = generateRandomVectors(NUM_VECTORS, DIMENSION);
    private static final float[][] QUERY_VECTORS = generateRandomVectors(NUM_QUERIES, DIMENSION);
    private static final SpaceType SPACE_TYPE = SpaceType.L2;

    /** ** Enable the http client *** */
    @Override
    protected boolean addMockHttpTransport() {
        return false;
    }

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        // Add the JVector plugin to the cluster
        return List.of(Netty4Plugin.class, JVectorKNNPlugin.class);
    }

    /**
     * Test that concurrently queries a vector field using jVector as the KNN engine
     * and verifies the accuracy of results by comparing against expected results
     * calculated using explicit distance measurements.
     */
    @Test
    public void testConcurrentJVectorQueries() throws Exception {
        createKnnIndexMappingWithJVectorEngine(DIMENSION, SPACE_TYPE, VectorDataType.FLOAT);
        indexTestVectors();

        final int numThreads = NUM_CONCURRENT_QUERIES;
        final ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        final CountDownLatch latch = new CountDownLatch(numThreads);
        final AtomicBoolean hasErrors = new AtomicBoolean(false);
        final AtomicInteger totalQueries = new AtomicInteger(0);

        // First compute expected results for each query
        List<List<KNNResult>> expectedResults = new ArrayList<>(NUM_QUERIES);
        for (int i = 0; i < NUM_QUERIES; i++) {
            expectedResults.add(computeExpectedResults(QUERY_VECTORS[i], K));
        }

        // Run concurrent queries
        for (int i = 0; i < numThreads; i++) {
            final int threadId = i;
            executorService.submit(() -> {
                try {
                    Thread.currentThread().setName("query-thread-" + threadId);
                    // Each thread will execute multiple queries round-robin style
                    for (int j = 0; j < NUM_TOTAL_QUERIES_PER_THREAD; j++) {
                        int queryIdx = j % NUM_QUERIES;
                        float[] queryVector = QUERY_VECTORS[queryIdx];

                        // Execute KNN search query
                        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, K), K);

                        // Parse response
                        String responseBody = EntityUtils.toString(response.getEntity());
                        List<KNNResult> results = parseSearchResponse(responseBody, FIELD_NAME);

                        // Verify results
                        verifyResults(results, expectedResults.get(queryIdx), queryVector);

                        totalQueries.incrementAndGet();
                    }
                } catch (Throwable e) {
                    logger.error("Query thread encountered an error", e);
                    hasErrors.set(true);
                } finally {
                    latch.countDown();
                }
            });
        }

        // Wait for all threads to complete or timeout after 30 seconds
        boolean completed = latch.await(60, TimeUnit.SECONDS);
        executorService.shutdown();

        logger.info("Completed {} concurrent queries across {} threads", totalQueries.get(), numThreads);

        assertTrue("Not all threads completed in time", completed);
        assertFalse("Errors were encountered during concurrent querying", hasErrors.get());
        assertEquals("Not all threads completed successfully", NUM_TOTAL_QUERIES_PER_THREAD * numThreads, totalQueries.get());
    }

    /**
     * Computes the expected KNN results by explicitly calculating distances
     * between the query vector and all indexed vectors.
     */
    private List<KNNResult> computeExpectedResults(float[] queryVector, int k) {
        // Compute distances for all vectors and sort
        List<KNNResult> results = new ArrayList<>(NUM_VECTORS);
        for (int i = 0; i < NUM_VECTORS; i++) {
            float[] vector = TEST_VECTORS[i];
            float distance = TestUtils.computeDistFromSpaceType(SPACE_TYPE, vector, queryVector);
            float score = CommonTestUtils.VECTOR_SIMILARITY_TO_SCORE.get(SPACE_TYPE.getKnnVectorSimilarityFunction()).apply(distance);
            results.add(new KNNResult(String.valueOf(i + 1), vector, score));
        }

        // Sort by score and take top k
        results.sort((r1, r2) -> Float.compare(r2.getScore(), r1.getScore()));
        if (results.size() > k) {
            results = results.subList(0, k);
        }

        return results;
    }

    /**
     * Verifies that the actual results match the expected results,
     * accounting for potential ties in scores.
     */
    private void verifyResults(List<KNNResult> actualResults, List<KNNResult> expectedResults, float[] queryVector) {
        // Verify result count
        assertEquals("Number of results doesn't match", expectedResults.size(), actualResults.size());

        // Verify each result's distance
        for (int i = 0; i < actualResults.size(); i++) {
            KNNResult actual = actualResults.get(i);

            // Calculate actual distance
            float[] actualVector = actual.getVector();
            float actualDistance = TestUtils.computeDistFromSpaceType(SPACE_TYPE, actualVector, queryVector);

            // Calculate expected distance
            float[] expectedVector = expectedResults.get(i).getVector();
            float expectedDistance = TestUtils.computeDistFromSpaceType(SPACE_TYPE, expectedVector, queryVector);

            // Allow for minor floating point differences
            assertEquals("Distance mismatch at position " + i, expectedDistance, actualDistance, 0.001);
        }
    }

    /**
     * Index the test vectors
     */
    private void indexTestVectors() throws IOException {
        for (int i = 0; i < TEST_VECTORS.length; i++) {
            client().prepareIndex(INDEX_NAME).setId("doc_" + i).setSource(FIELD_NAME, TEST_VECTORS[i]).get();
        }
        refresh(INDEX_NAME);
    }

    /**
     * Generate random vectors for testing
     */
    private static float[][] generateRandomVectors(int numVectors, int dimension) {
        float[][] vectors = new float[numVectors][dimension];
        for (int i = 0; i < numVectors; i++) {
            vectors[i] = JVectorEngineIT.generateRandomVector(dimension);
        }
        return vectors;
    }

    private void createKnnIndexMappingWithJVectorEngine(int dimension, SpaceType spaceType, VectorDataType vectorDataType)
        throws Exception {
        String mapping = CommonTestUtils.createIndexMapping(dimension, spaceType, vectorDataType);
        Settings indexSettings = CommonTestUtils.getDefaultIndexSettings();
        // indexSettings = Settings.builder().put(indexSettings).put(INDEX_USE_COMPOUND_FILE.getKey(), false).build();
        createKnnIndex(INDEX_NAME, indexSettings, mapping);
    }

    /**
     * Create KNN Index
     */
    protected void createKnnIndex(String index, Settings settings, String mapping) throws IOException {
        createIndex(index, settings);
        putMappingRequest(index, mapping);
    }

    /**
     * For a given index, make a mapping request
     */
    protected void putMappingRequest(String index, String mapping) throws IOException {
        // Put KNN mapping
        Request request = new Request("PUT", "/" + index + "/_mapping");

        request.setJsonEntity(mapping);
        Response response = getRestClient().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    /**
     * Deprecated
     * To better simulate user request, use {@link #searchKNNIndex(String, XContentBuilder, int)} instead
     */
    @Deprecated
    protected Response searchKNNIndex(String index, KNNQueryBuilder knnQueryBuilder, int resultSize) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnQueryBuilder.doXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject().endObject();
        return searchKNNIndex(index, builder, resultSize);
    }

    /**
     * Run KNN Search on Index with XContentBuilder query
     */
    protected Response searchKNNIndex(String index, XContentBuilder xContentBuilder, int resultSize) throws IOException {
        return searchKNNIndex(index, xContentBuilder.toString(), resultSize);
    }

    /**
     * Run KNN Search on Index with json string query
     */
    protected Response searchKNNIndex(String index, String query, int resultSize) throws IOException {
        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(query);

        request.addParameter("size", Integer.toString(resultSize));
        request.addParameter("search_type", "query_then_fetch");
        // Nested field does not support explain parameter and the request is rejected if we set explain parameter
        // request.addParameter("explain", Boolean.toString(true));

        Response response = getRestClient().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }

    /**
     * Parse the response of KNN search into a List of KNNResults
     */
    protected List<KNNResult> parseSearchResponse(String responseBody, String fieldName) throws IOException {
        @SuppressWarnings("unchecked")
        List<Object> hits = (List<Object>) ((Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody
        ).map().get("hits")).get("hits");

        @SuppressWarnings("unchecked")
        List<KNNResult> knnSearchResponses = hits.stream().map(hit -> {
            @SuppressWarnings("unchecked")
            final float[] vector = Floats.toArray(
                Arrays.stream(
                    ((ArrayList<Float>) ((Map<String, Object>) ((Map<String, Object>) hit).get("_source")).get(fieldName)).toArray()
                ).map(Object::toString).map(Float::valueOf).collect(Collectors.toList())
            );
            return new KNNResult(
                (String) ((Map<String, Object>) hit).get("_id"),
                vector,
                ((Double) ((Map<String, Object>) hit).get("_score")).floatValue()
            );
        }).collect(Collectors.toList());

        return knnSearchResponses;
    }
}
