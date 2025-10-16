/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Floats;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.junit.Assert;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.RestClient;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.*;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.codec.jvector.JVectorFormat;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.JVectorKNNPlugin;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.opensearch.common.xcontent.XContentFactory.jsonBuilder;
import static org.opensearch.knn.KNNRestTestCase.FIELD_NAME;
import static org.opensearch.knn.common.KNNConstants.DISK_ANN;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.common.KNNConstants.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;

public class CommonTestUtils {
    private static final NamedXContentRegistry DEFAULT_NAMED_X_CONTENT_REGISTRY = new NamedXContentRegistry(
        ClusterModule.getNamedXWriteables()
    );

    public static final int DIMENSION = 3;
    public static final String DOC_ID = "doc1";
    public static final String DOC_ID_2 = "doc2";
    public static final String DOC_ID_3 = "doc3";
    public static final int EF_CONSTRUCTION = 128;
    public static final String COLOR_FIELD_NAME = "color";
    public static final String TASTE_FIELD_NAME = "taste";
    public static final int M = 16;

    public static final Float[][] TEST_INDEX_VECTORS = { { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.0f, 2.0f }, { 3.0f, 3.0f, 3.0f } };
    public static final Float[][] TEST_COSINESIMIL_INDEX_VECTORS = { { 6.0f, 7.0f, 3.0f }, { 3.0f, 2.0f, 5.0f }, { 4.0f, 5.0f, 7.0f } };
    public static final Float[][] TEST_INNER_PRODUCT_INDEX_VECTORS = {
        { 1.0f, 1.0f, 1.0f },
        { 2.0f, 2.0f, 2.0f },
        { 3.0f, 3.0f, 3.0f },
        { -1.0f, -1.0f, -1.0f },
        { -2.0f, -2.0f, -2.0f },
        { -3.0f, -3.0f, -3.0f } };

    public static final float[][] TEST_QUERY_VECTORS = { { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.0f, 2.0f }, { 3.0f, 3.0f, 3.0f } };

    public static final Map<KNNVectorSimilarityFunction, Function<Float, Float>> VECTOR_SIMILARITY_TO_SCORE = ImmutableMap.of(
        KNNVectorSimilarityFunction.EUCLIDEAN,
        (similarity) -> 1 / (1 + similarity),
        KNNVectorSimilarityFunction.DOT_PRODUCT,
        (similarity) -> (1 + similarity) / 2,
        KNNVectorSimilarityFunction.COSINE,
        (similarity) -> (1 + similarity) / 2,
        KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
        (similarity) -> similarity <= 0 ? 1 / (1 - similarity) : similarity + 1
    );
    public static final String DIMENSION_FIELD_NAME = "dimension";
    public static final String KNN_VECTOR_TYPE = "knn_vector";
    public static final String PROPERTIES_FIELD_NAME = "properties";
    public static final String TYPE_FIELD_NAME = "type";
    public static final String INTEGER_FIELD_NAME = "int_field";
    public static final String FILED_TYPE_INTEGER = "integer";
    public static final String NON_EXISTENT_INTEGER_FIELD_NAME = "nonexistent_int_field";

    public static Settings getDefaultIndexSettings() {
        return Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put(KNN_INDEX, true).build();
    }

    public static String createIndexMapping(int dimension, SpaceType spaceType, VectorDataType vectorDataType) throws IOException {
        XContentBuilder builder = jsonBuilder().startObject()
            .startObject(PROPERTIES_FIELD_NAME)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, vectorDataType)
            .startObject(KNNConstants.KNN_METHOD)
            .field(KNNConstants.NAME, DISK_ANN)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .startObject(KNNConstants.PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        return builder.toString();
    }

    public static Codec getCodec() {
        return getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION);
    }

    public static Codec getCodec(int minBatchSizeForQuantization) {
        return new FilterCodec(KNNCodecVersion.V_10_01_0.getCodecName(), new Lucene101Codec()) {
            @Override
            public KnnVectorsFormat knnVectorsFormat() {
                return new PerFieldKnnVectorsFormat() {

                    @Override
                    public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                        return new JVectorFormat(minBatchSizeForQuantization);
                    }
                };
            }
        };
    }

    /**
     * Get Stats from KNN Plugin
     */
    public static Response getKnnStats(RestClient restClient, List<String> nodeIds, List<String> stats) throws IOException {
        return executeKnnStatRequest(restClient, nodeIds, stats, JVectorKNNPlugin.KNN_BASE_URI);
    }

    public static Response executeKnnStatRequest(RestClient restClient, List<String> nodeIds, List<String> stats, final String baseURI)
        throws IOException {
        String nodePrefix = "";
        if (!nodeIds.isEmpty()) {
            nodePrefix = "/" + String.join(",", nodeIds);
        }

        String statsSuffix = "";
        if (!stats.isEmpty()) {
            statsSuffix = "/" + String.join(",", stats);
        }

        Request request = new Request("GET", baseURI + nodePrefix + "/stats" + statsSuffix);

        Response response = restClient.performRequest(request);
        Assert.assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        return response;
    }

    /**
     * Parse KNN Node stats from response
     */
    public static List<Map<String, Object>> parseNodeStatsResponse(String responseBody) throws IOException {
        @SuppressWarnings("unchecked")
        Map<String, Object> responseMap = (Map<String, Object>) createParser(
            MediaTypeRegistry.getDefaultMediaType().xContent(),
            responseBody
        ).map().get("nodes");

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nodeResponses = responseMap.keySet()
            .stream()
            .map(key -> (Map<String, Object>) responseMap.get(key))
            .collect(Collectors.toList());

        return nodeResponses;
    }

    /**
     * Deprecated
     * To better simulate user request, use {@link #searchKNNIndex(RestClient, String, XContentBuilder, int)} instead
     */
    @Deprecated
    public static Response searchKNNIndex(RestClient restClient, String index, KNNQueryBuilder knnQueryBuilder, int resultSize)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        knnQueryBuilder.doXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject().endObject();
        return searchKNNIndex(restClient, index, builder, resultSize);
    }

    /**
     * Run KNN Search on Index with XContentBuilder query
     */
    public static Response searchKNNIndex(RestClient restClient, String index, XContentBuilder xContentBuilder, int resultSize)
        throws IOException {
        return searchKNNIndex(restClient, index, xContentBuilder.toString(), resultSize);
    }

    /**
     * Run KNN Search on Index with json string query
     */
    public static Response searchKNNIndex(RestClient restClient, String index, String query, int resultSize) throws IOException {
        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(query);

        request.addParameter("size", Integer.toString(resultSize));
        request.addParameter("search_type", "query_then_fetch");
        // Nested field does not support explain parameter and the request is rejected if we set explain parameter
        // request.addParameter("explain", Boolean.toString(true));

        Response response = restClient.performRequest(request);
        Assert.assertEquals(
            request.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode())
        );

        return response;
    }

    /**
     * Parse the response of KNN search into a List of KNNResults
     */
    public static List<KNNResult> parseSearchResponse(String responseBody, String fieldName) throws IOException {
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

    // Method that adds multiple documents into the index using Bulk API
    public static void bulkAddKnnDocs(
        RestClient restClient,
        String index,
        String fieldName,
        float[][] indexVectors,
        int docCount,
        boolean refresh
    ) throws IOException {
        bulkAddKnnDocs(restClient, index, fieldName, indexVectors, 0, docCount, refresh);
    }

    // Method that adds multiple documents into the index using Bulk API
    public static void bulkAddKnnDocs(
        RestClient restClient,
        String index,
        String fieldName,
        float[][] indexVectors,
        int baseDocId,
        int docCount,
        boolean refresh
    ) throws IOException {
        bulkAddKnnDocs(restClient, index, fieldName, indexVectors, 0, baseDocId, docCount, refresh);
    }

    // Method that adds multiple documents into the index using Bulk API
    public static void bulkAddKnnDocs(
        RestClient restClient,
        String index,
        String fieldName,
        float[][] sourceVectors,
        int sourceOffset,
        int baseDocId,
        int docCount,
        boolean refresh
    ) throws IOException {
        Request request = new Request("POST", "/_bulk");

        request.addParameter("refresh", Boolean.toString(refresh));
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < docCount; i++) {
            sb.append("{ \"index\" : { \"_index\" : \"")
                .append(index)
                .append("\", \"_id\" : \"")
                .append(baseDocId + i)
                .append("\" } }\n")
                .append("{ \"")
                .append(fieldName)
                .append("\" : ")
                .append(Arrays.toString(sourceVectors[sourceOffset + i]))
                .append(" }\n");
        }

        request.setJsonEntity(sb.toString());

        Response response = restClient.performRequest(request);
        Assert.assertEquals(200, response.getStatusLine().getStatusCode());
    }

    public static void flushIndex(RestClient restClient, final String index) throws IOException {
        Request request = new Request("POST", "/" + index + "/_flush");

        Response response = restClient.performRequest(request);
        Assert.assertEquals(
            request.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode())
        );
    }

    public static int getDocCount(RestClient restClient, String indexName) throws Exception {
        Request request = new Request("GET", "/" + indexName + "/_count");

        Response response = restClient.performRequest(request);

        Assert.assertEquals(
            request.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode())
        );

        String responseBody = EntityUtils.toString(response.getEntity());

        Map<String, Object> responseMap = createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map();
        return (Integer) responseMap.get("count");
    }

    /**
     * Force merge KNN index segments
     */
    public static void forceMergeKnnIndex(RestClient restClient, String index) throws Exception {
        forceMergeKnnIndex(restClient, index, 1);
    }

    /**
     * Force merge KNN index segments
     */
    public static void forceMergeKnnIndex(RestClient restClient, String index, int maxSegments) throws Exception {
        Request request = new Request("POST", "/" + index + "/_refresh");

        Response response = restClient.performRequest(request);
        Assert.assertEquals(
            request.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode())
        );

        request = new Request("POST", "/" + index + "/_forcemerge");

        request.addParameter("max_num_segments", String.valueOf(maxSegments));
        request.addParameter("flush", "true");
        response = restClient.performRequest(request);
        Assert.assertEquals(
            request.getEndpoint() + ": failed",
            RestStatus.OK,
            RestStatus.fromCode(response.getStatusLine().getStatusCode())
        );
        TimeUnit.SECONDS.sleep(5); // To make sure force merge is completed
    }

    private static NamedXContentRegistry xContentRegistry() {
        return DEFAULT_NAMED_X_CONTENT_REGISTRY;
    }

    private static XContentParser createParser(XContent xContent, String data) throws IOException {
        return xContent.createParser(xContentRegistry(), LoggingDeprecationHandler.INSTANCE, data);
    }
}
