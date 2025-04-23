/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.common.xcontent.XContentFactory.jsonBuilder;
import static org.opensearch.knn.KNNRestTestCase.FIELD_NAME;
import static org.opensearch.knn.common.KNNConstants.DISK_ANN;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

public class CommonTestUtils {
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
}
