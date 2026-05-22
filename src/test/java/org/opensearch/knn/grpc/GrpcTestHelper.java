/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.grpc;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.concurrent.TimeUnit;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.protobufs.KnnQuery;
import org.opensearch.protobufs.MatchAllQuery;
import org.opensearch.protobufs.QueryContainer;
import org.opensearch.protobufs.SearchRequest;
import org.opensearch.protobufs.SearchRequestBody;
import org.opensearch.protobufs.SearchResponse;
import org.opensearch.protobufs.services.SearchServiceGrpc;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Metadata;
import io.grpc.stub.MetadataUtils;
import lombok.experimental.UtilityClass;

/**
 * Helper class providing common gRPC testing utilities for KNN integration tests.
 * This class provides utilities for creating gRPC channels, building search requests,
 * and executing searches via gRPC.
 */
@UtilityClass
public class GrpcTestHelper {

    private static final Logger logger = LogManager.getLogger(GrpcTestHelper.class);

    public static final String DEFAULT_GRPC_HOST = "127.0.0.1";
    public static final int DEFAULT_GRPC_PORT = 9400;
    public static final int DEFAULT_TIMEOUT_SECONDS = 30;

    /**
     * Check if gRPC transport is expected to be available.
     * Returns true when either:
     * - tests.grpc.port is explicitly set (Gradle-managed cluster discovered the port), or
     * - tests.rest.cluster is NOT set (local testClusters run where gRPC is configured in build.gradle)
     *
     * Returns false when tests.rest.cluster is set but tests.grpc.port is not,
     * indicating an external cluster that likely doesn't have gRPC transport configured.
     *
     * Use with {@code assumeTrue(GrpcTestHelper.isGrpcTransportConfigured())} in test setUp
     * to gracefully skip gRPC tests in distribution integration tests (opensearch-build test.sh).
     */
    public static boolean isGrpcTransportConfigured() {
        boolean grpcPortExplicitlySet = System.getProperty("tests.grpc.port") != null;
        boolean externalCluster = System.getProperty("tests.rest.cluster") != null;
        if (grpcPortExplicitlySet) {
            logger.info("gRPC port explicitly configured: {}", getGrpcPort());
            return true;
        }
        if (externalCluster) {
            logger.info(
                "External cluster detected (tests.rest.cluster={}) without tests.grpc.port — gRPC tests will be skipped",
                System.getProperty("tests.rest.cluster")
            );
            return false;
        }
        // Local testClusters run — gRPC is configured in build.gradle
        logger.info("Local test cluster run — gRPC transport expected to be available");
        return true;
    }

    // ===========================================================================================
    // CHANNEL MANAGEMENT
    // ===========================================================================================

    public static String getGrpcHost() {
        return System.getProperty("tests.grpc.host", DEFAULT_GRPC_HOST);
    }

    public static int getGrpcPort() {
        return Integer.parseInt(System.getProperty("tests.grpc.port", String.valueOf(DEFAULT_GRPC_PORT)));
    }

    public static ManagedChannel createGrpcChannel() {
        return createGrpcChannel(getGrpcHost(), getGrpcPort());
    }

    public static ManagedChannel createGrpcChannel(String host, int port) {
        String target = host + ":" + port;
        logger.info("Creating gRPC channel to target: {}", target);
        try {
            ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
            logger.info("gRPC channel created, state: {}", channel.getState(true));
            return channel;
        } catch (Exception e) {
            logger.error("Failed to create gRPC channel to target: {}. Error: {}", target, e.getMessage(), e);
            throw new RuntimeException("Failed to create gRPC channel to " + target, e);
        }
    }

    public static void shutdownChannel(ManagedChannel channel, int timeoutSeconds) throws InterruptedException {
        if (channel != null && !channel.isShutdown()) {
            channel.shutdown();
            channel.awaitTermination(timeoutSeconds, TimeUnit.SECONDS);
        }
    }

    public static void shutdownChannel(ManagedChannel channel) throws InterruptedException {
        shutdownChannel(channel, 5);
    }

    // ===========================================================================================
    // SEARCH REQUEST BUILDERS
    // ===========================================================================================

    public static SearchRequest buildSearchRequest(String index, QueryContainer query) {
        return buildSearchRequest(index, query, 10);
    }

    public static SearchRequest buildSearchRequest(String index, QueryContainer query, int size) {
        return SearchRequest.newBuilder()
            .addIndex(index)
            .setSearchRequestBody(SearchRequestBody.newBuilder().setQuery(query).setSize(size).build())
            .build();
    }

    // ===========================================================================================
    // SEARCH EXECUTION
    // ===========================================================================================

    public static SearchResponse executeSearch(ManagedChannel channel, SearchRequest request) {
        return executeSearch(channel, request, DEFAULT_TIMEOUT_SECONDS);
    }

    public static SearchResponse executeSearch(ManagedChannel channel, SearchRequest request, int timeoutSeconds) {
        SearchServiceGrpc.SearchServiceBlockingStub stub = SearchServiceGrpc.newBlockingStub(channel)
            .withDeadlineAfter(timeoutSeconds, TimeUnit.SECONDS);

        if (isSecurityEnabled()) {
            stub = addBasicAuthentication(stub);
        }

        return stub.search(request);
    }

    private static boolean isSecurityEnabled() {
        return "true".equals(System.getProperty("security.enabled"));
    }

    private static SearchServiceGrpc.SearchServiceBlockingStub addBasicAuthentication(SearchServiceGrpc.SearchServiceBlockingStub stub) {
        String username = System.getProperty("user", "admin");
        String password = System.getProperty("password", "admin");
        String credentials = username + ":" + password;
        String encodedCredentials = Base64.getEncoder().encodeToString(credentials.getBytes(StandardCharsets.UTF_8));

        Metadata headers = new Metadata();
        Metadata.Key<String> authKey = Metadata.Key.of("Authorization", Metadata.ASCII_STRING_MARSHALLER);
        headers.put(authKey, "Basic " + encodedCredentials);

        return stub.withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers));
    }

    // ===========================================================================================
    // QUERY BUILDERS
    // ===========================================================================================

    public static QueryContainer createMatchAllQueryContainer() {
        return QueryContainer.newBuilder().setMatchAll(MatchAllQuery.newBuilder().build()).build();
    }

    public static QueryContainer createKnnQueryContainer(String field, float[] vector, int k) {
        KnnQuery.Builder knnBuilder = KnnQuery.newBuilder().setField(field).setK(k);
        for (float v : vector) {
            knnBuilder.addVector(v);
        }
        return QueryContainer.newBuilder().setKnn(knnBuilder.build()).build();
    }

}
