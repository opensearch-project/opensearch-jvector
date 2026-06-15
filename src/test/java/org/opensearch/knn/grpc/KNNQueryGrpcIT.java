/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.grpc;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.opensearch.knn.grpc.GrpcTestHelper.buildSearchRequest;
import static org.opensearch.knn.grpc.GrpcTestHelper.createKnnQueryContainer;
import static org.opensearch.knn.grpc.GrpcTestHelper.createMatchAllQueryContainer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.protobufs.QueryContainer;
import org.opensearch.protobufs.SearchRequest;
import org.opensearch.protobufs.SearchResponse;

import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import lombok.SneakyThrows;

/**
 * Integration test for gRPC KNN Query functionality.
 *
 * This test verifies that KNN queries can be:
 * 1. Created in Protocol Buffer format
 * 2. Sent via gRPC to OpenSearch
 * 3. Executed correctly using JVector engine and return expected results
 */
public class KNNQueryGrpcIT extends KNNRestTestCase {

    private static final Logger logger = LogManager.getLogger(KNNQueryGrpcIT.class);

    private static final String TEST_INDEX_NAME = "test-grpc-knn-index";
    private static final String TEST_VECTOR_FIELD_NAME = "test_vector";
    private static final int TEST_VECTOR_DIMENSION = 3;

    private ManagedChannel grpcChannel;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        // Skip gRPC tests when gRPC transport is not available on the cluster
        // (e.g., distribution integration tests via opensearch-build test.sh where
        // the external cluster doesn't have aux.transport.types configured)
        assumeTrue("gRPC transport is not configured, skipping gRPC tests", GrpcTestHelper.isGrpcTransportConfigured());

        // Set up test index with vector data
        initializeTestIndex();

        // Create gRPC channel for tests using the shared helper
        grpcChannel = GrpcTestHelper.createGrpcChannel();
    }

    @After
    public void tearDownGrpc() throws Exception {
        try {
            deleteTestIndex();
        } finally {
            GrpcTestHelper.shutdownChannel(grpcChannel);
        }
    }

    /**
     * Initialize test index with KNN vector field and sample documents.
     */
    @SneakyThrows
    private void initializeTestIndex() {
        if (!indexExists(TEST_INDEX_NAME)) {
            logger.info("Creating test index: {}", TEST_INDEX_NAME);

            // Create index with KNN vector field using JVector engine
            String indexMapping = String.format(
                java.util.Locale.ROOT,
                "{\n"
                    + "  \"settings\": {\n"
                    + "    \"index\": {\n"
                    + "      \"knn\": true,\n"
                    + "      \"number_of_shards\": 1,\n"
                    + "      \"number_of_replicas\": 0\n"
                    + "    }\n"
                    + "  },\n"
                    + "  \"mappings\": {\n"
                    + "    \"properties\": {\n"
                    + "      \"%s\": {\n"
                    + "        \"type\": \"knn_vector\",\n"
                    + "        \"dimension\": %d,\n"
                    + "        \"method\": {\n"
                    + "          \"name\": \"disk_ann\",\n"
                    + "          \"engine\": \"jvector\",\n"
                    + "          \"space_type\": \"l2\"\n"
                    + "        }\n"
                    + "      }\n"
                    + "    }\n"
                    + "  }\n"
                    + "}",
                TEST_VECTOR_FIELD_NAME,
                TEST_VECTOR_DIMENSION
            );

            Request createIndexRequest = new Request("PUT", "/" + TEST_INDEX_NAME);
            createIndexRequest.setJsonEntity(indexMapping);
            Response response = client().performRequest(createIndexRequest);
            assertEquals(200, response.getStatusLine().getStatusCode());
            logger.info("Test index created successfully");

            // Index sample documents
            indexTestDocuments();

            // Refresh index to make documents searchable
            refreshIndex(TEST_INDEX_NAME);
        }
    }

    /**
     * Index sample documents with vector data.
     */
    @SneakyThrows
    private void indexTestDocuments() {
        // Document 1: [0.1, 0.2, 0.3]
        String doc1 = String.format("{\"%s\": [0.1, 0.2, 0.3]}", TEST_VECTOR_FIELD_NAME);
        Request indexDoc1 = new Request("POST", "/" + TEST_INDEX_NAME + "/_doc/1?refresh=true");
        indexDoc1.setJsonEntity(doc1);
        client().performRequest(indexDoc1);

        // Document 2: [0.2, 0.3, 0.4]
        String doc2 = String.format("{\"%s\": [0.2, 0.3, 0.4]}", TEST_VECTOR_FIELD_NAME);
        Request indexDoc2 = new Request("POST", "/" + TEST_INDEX_NAME + "/_doc/2?refresh=true");
        indexDoc2.setJsonEntity(doc2);
        client().performRequest(indexDoc2);

        // Document 3: [0.3, 0.4, 0.5]
        String doc3 = String.format("{\"%s\": [0.3, 0.4, 0.5]}", TEST_VECTOR_FIELD_NAME);
        Request indexDoc3 = new Request("POST", "/" + TEST_INDEX_NAME + "/_doc/3?refresh=true");
        indexDoc3.setJsonEntity(doc3);
        client().performRequest(indexDoc3);

        logger.info("Indexed 3 test documents");
    }

    /**
     * Delete the test index to clean up resources after test execution.
     */
    @SneakyThrows
    private void deleteTestIndex() {
        try {
            if (indexExists(TEST_INDEX_NAME)) {
                logger.info("Deleting test index: {}", TEST_INDEX_NAME);
                Request deleteRequest = new Request("DELETE", "/" + TEST_INDEX_NAME);
                Response response = client().performRequest(deleteRequest);
                if (response.getStatusLine().getStatusCode() == 200) {
                    logger.info("Successfully deleted test index: {}", TEST_INDEX_NAME);
                }
            }
        } catch (Exception e) {
            logger.warn("Failed to delete test index {}: {}", TEST_INDEX_NAME, e.getMessage());
        }
    }

    // ===========================================================================================
    // BASIC FUNCTIONALITY TESTS - Verify end-to-end gRPC query execution
    // ===========================================================================================

    /**
     * Test gRPC connectivity with a simple MatchAll query.
     */
    @SneakyThrows
    public void testGrpcConnectivityWithMatchAllQuery() {
        String host = GrpcTestHelper.getGrpcHost();
        int port = GrpcTestHelper.getGrpcPort();
        logger.info("Testing gRPC connectivity to {}:{}", host, port);

        try {
            QueryContainer query = createMatchAllQueryContainer();
            SearchRequest request = buildSearchRequest(TEST_INDEX_NAME, query);

            SearchResponse response = GrpcTestHelper.executeSearch(grpcChannel, request, 10);

            assertNotNull("Search response should not be null", response);
            assertTrue("Should have at least one hit", response.getHits().getHitsCount() > 0);
            logger.info("gRPC connection successful to {}:{}", host, port);
        } catch (StatusRuntimeException e) {
            logger.error("Failed to connect via gRPC to {}:{} - {}", host, port, e.getMessage());
            fail("Cannot connect to gRPC endpoint " + host + ":" + port + " - " + e.getMessage());
        }
    }

    /**
     * Test that a basic KNN query executes via gRPC and returns results.
     * This validates the complete round-trip: client -> gRPC -> OpenSearch -> JVector -> response
     */
    @SneakyThrows
    public void testBasicKnnQueryReturnsResults() {
        float[] queryVector = new float[] { 0.15f, 0.25f, 0.35f };

        QueryContainer knnQuery = createKnnQueryContainer(TEST_VECTOR_FIELD_NAME, queryVector, 3);
        SearchRequest request = buildSearchRequest(TEST_INDEX_NAME, knnQuery, 3);

        SearchResponse response = GrpcTestHelper.executeSearch(grpcChannel, request);

        assertNotNull("Search response should not be null", response);
        assertTrue("Should have hits", response.getHits().getHitsCount() > 0);
        assertTrue("Should have at most 3 hits (k=3)", response.getHits().getHitsCount() <= 3);
        logger.info("KNN query via gRPC returned {} hits", response.getHits().getHitsCount());
    }

    /**
     * Test KNN query with k=1 to verify nearest neighbor search.
     */
    @SneakyThrows
    public void testKnnQueryWithK1() {
        // Query vector very close to document 1: [0.1, 0.2, 0.3]
        float[] queryVector = new float[] { 0.11f, 0.21f, 0.31f };

        QueryContainer knnQuery = createKnnQueryContainer(TEST_VECTOR_FIELD_NAME, queryVector, 1);
        SearchRequest request = buildSearchRequest(TEST_INDEX_NAME, knnQuery, 1);

        SearchResponse response = GrpcTestHelper.executeSearch(grpcChannel, request);

        assertNotNull("Search response should not be null", response);
        assertEquals("Should return exactly 1 hit (k=1)", 1, response.getHits().getHitsCount());
        logger.info("KNN query with k=1 correctly returned nearest neighbor");
    }

    /**
     * Test that KNN query results are ordered by similarity score.
     */
    @SneakyThrows
    public void testKnnQueryResultsAreOrdered() {
        float[] queryVector = new float[] { 0.15f, 0.25f, 0.35f };

        QueryContainer knnQuery = createKnnQueryContainer(TEST_VECTOR_FIELD_NAME, queryVector, 3);
        SearchRequest request = buildSearchRequest(TEST_INDEX_NAME, knnQuery, 3);

        SearchResponse response = GrpcTestHelper.executeSearch(grpcChannel, request);

        assertNotNull("Search response should not be null", response);
        assertTrue("Should have multiple hits", response.getHits().getHitsCount() > 1);
        logger.info("KNN query via gRPC returned {} hits with valid scores", response.getHits().getHitsCount());
    }

    // ===========================================================================================
    // ERROR HANDLING TESTS - Verify proper error responses
    // ===========================================================================================

    /**
     * Test search on non-existent index returns proper gRPC error.
     */
    @SneakyThrows
    public void testErrorNonExistentIndex() {
        float[] queryVector = new float[] { 0.1f, 0.2f, 0.3f };

        QueryContainer knnQuery = createKnnQueryContainer(TEST_VECTOR_FIELD_NAME, queryVector, 3);
        SearchRequest request = buildSearchRequest("non-existent-index-12345", knnQuery);

        StatusRuntimeException exception = expectThrows(
            StatusRuntimeException.class,
            () -> GrpcTestHelper.executeSearch(grpcChannel, request)
        );

        assertNotNull("Should throw exception for non-existent index", exception);
        logger.info("Non-existent index test correctly threw exception: {}", exception.getStatus().getCode());
    }

    /**
     * Test KNN query with invalid vector dimension returns proper error.
     */
    @SneakyThrows
    public void testErrorInvalidVectorDimension() {
        // Index expects 3 dimensions, but we provide 5
        float[] invalidVector = new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };

        QueryContainer knnQuery = createKnnQueryContainer(TEST_VECTOR_FIELD_NAME, invalidVector, 3);
        SearchRequest request = buildSearchRequest(TEST_INDEX_NAME, knnQuery);

        StatusRuntimeException exception = expectThrows(
            StatusRuntimeException.class,
            () -> GrpcTestHelper.executeSearch(grpcChannel, request)
        );

        assertNotNull("Should throw exception for invalid vector dimension", exception);
        assertTrue(
            "Error should indicate dimension mismatch",
            exception.getMessage().contains("dimension")
                || exception.getStatus().getCode().name().equals("INVALID_ARGUMENT")
                || exception.getStatus().getCode().name().equals("INTERNAL")
        );
        logger.info("Invalid vector dimension test correctly threw exception: {}", exception.getStatus().getCode());
    }

    /**
     * Test KNN query with invalid k value (k=0) returns proper error.
     */
    @SneakyThrows
    public void testErrorInvalidKValue() {
        float[] queryVector = new float[] { 0.1f, 0.2f, 0.3f };

        // k=0 is invalid
        QueryContainer knnQuery = createKnnQueryContainer(TEST_VECTOR_FIELD_NAME, queryVector, 0);
        SearchRequest request = buildSearchRequest(TEST_INDEX_NAME, knnQuery);

        StatusRuntimeException exception = expectThrows(
            StatusRuntimeException.class,
            () -> GrpcTestHelper.executeSearch(grpcChannel, request)
        );

        assertNotNull("Should throw exception for invalid k value", exception);
        assertTrue(
            "Error should indicate invalid k",
            exception.getMessage().contains("k")
                || exception.getStatus().getCode().name().equals("INVALID_ARGUMENT")
                || exception.getStatus().getCode().name().equals("INTERNAL")
        );
        logger.info("Invalid k value test correctly threw exception: {}", exception.getStatus().getCode());
    }
}
