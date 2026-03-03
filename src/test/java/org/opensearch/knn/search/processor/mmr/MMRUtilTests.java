/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.mockito.ArgumentCaptor;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.index.Index;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.search.extension.MMRSearchExtBuilder;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.pipeline.ProcessorGenerationContext;
import org.opensearch.transport.client.Client;

import java.util.*;

import static org.mockito.Mockito.*;
import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.search.processor.mmr.MMRUtil.getMMRFieldMappingByPath;

public class MMRUtilTests extends MMRTestCase {
    private Client mockClient;
    private ActionListener<MMRVectorFieldInfo> listener;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mockClient = mock(Client.class);
        listener = mock(ActionListener.class);
    }

    // ========== Tests for extractVectorFromHit ==========

    public void testExtractVectorFromHit_SimpleFloatVector() {
        Map<String, Object> source = Map.of("embedding", List.of(0.1, 0.2, 0.3, 0.4));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-1", true);

        assertNotNull(result);
        assertEquals(4, result.length);
        assertEquals(0.1f, result[0], DELTA);
        assertEquals(0.2f, result[1], DELTA);
        assertEquals(0.3f, result[2], DELTA);
        assertEquals(0.4f, result[3], DELTA);
    }

    public void testExtractVectorFromHit_SimpleByteVector() {
        Map<String, Object> source = Map.of("embedding", List.of(1.0, 2.0, 3.0, 127.0));

        byte[] result = (byte[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-1", false);

        assertNotNull(result);
        assertEquals(4, result.length);
        assertEquals((byte) 1, result[0]);
        assertEquals((byte) 2, result[1]);
        assertEquals((byte) 3, result[2]);
        assertEquals((byte) 127, result[3]);
    }

    public void testExtractVectorFromHit_NestedPath() {
        Map<String, Object> source = Map.of("user", Map.of("profile", Map.of("embedding", List.of(1.0, 2.0, 3.0))));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "user.profile.embedding", "doc-2", true);

        assertNotNull(result);
        assertEquals(3, result.length);
        assertEquals(1.0f, result[0], DELTA);
        assertEquals(2.0f, result[1], DELTA);
        assertEquals(3.0f, result[2], DELTA);
    }

    public void testExtractVectorFromHit_DeepNestedPath() {
        Map<String, Object> source = Map.of("level1", Map.of("level2", Map.of("level3", Map.of("vector", List.of(5.5, 6.6)))));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "level1.level2.level3.vector", "doc-3", true);

        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(5.5f, result[0], DELTA);
        assertEquals(6.6f, result[1], DELTA);
    }

    public void testExtractVectorFromHit_EmptyVector() {
        Map<String, Object> source = Map.of("embedding", List.of());

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-4", true);

        assertNotNull(result);
        assertEquals(0, result.length);
    }

    public void testExtractVectorFromHit_NullSource() {
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(null, "embedding", "doc-5", true)
        );
        assertNotNull(ex);
    }

    public void testExtractVectorFromHit_NullFieldPath() {
        Map<String, Object> source = Map.of("embedding", List.of(1.0, 2.0));
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, null, "doc-6", true)
        );
        assertNotNull(ex);
    }

    public void testExtractVectorFromHit_EmptyFieldPath() {
        Map<String, Object> source = Map.of("embedding", List.of(1.0, 2.0));
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, "", "doc-7", true)
        );
        assertNotNull(ex);
    }

    public void testExtractVectorFromHit_MissingField() {
        Map<String, Object> source = Map.of("other_field", "value");
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, "embedding", "doc-8", true)
        );
        assertTrue(ex.getMessage().contains("not found"));
    }

    public void testExtractVectorFromHit_MissingNestedField() {
        Map<String, Object> source = Map.of("user", Map.of("name", "John"));
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, "user.profile.embedding", "doc-9", true)
        );
        assertNotNull(ex);
    }

    public void testExtractVectorFromHit_NonMapInPath() {
        Map<String, Object> source = Map.of("user", "not_a_map");
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, "user.embedding", "doc-10", true)
        );
        assertNotNull(ex);
    }

    public void testExtractVectorFromHit_NonListAtEnd() {
        Map<String, Object> source = Map.of("embedding", "not_a_vector");
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, "embedding", "doc-11", true)
        );
        assertNotNull(ex);
    }

    public void testExtractVectorFromHit_InvalidVectorContent() {
        Map<String, Object> source = Map.of("embedding", List.of("not", "numbers"));
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.extractVectorFromHit(source, "embedding", "doc-12", true)
        );
        assertTrue(ex.getMessage().contains("unexpected value at the vector field"));
    }

    public void testExtractVectorFromHit_LargeVector() {
        List<Double> largeVector = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            largeVector.add((double) i);
        }

        Map<String, Object> source = Map.of("embedding", largeVector);

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-large", true);

        assertNotNull(result);
        assertEquals(1000, result.length);
        for (int i = 0; i < 1000; i++) {
            assertEquals((float) i, result[i], DELTA);
        }
    }

    public void testExtractVectorFromHit_NegativeValues() {
        Map<String, Object> source = Map.of("embedding", List.of(-1.5, -2.5, -3.5));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-negative", true);

        assertNotNull(result);
        assertEquals(3, result.length);
        assertEquals(-1.5f, result[0], DELTA);
        assertEquals(-2.5f, result[1], DELTA);
        assertEquals(-3.5f, result[2], DELTA);
    }

    public void testExtractVectorFromHit_MixedPositiveNegative() {
        Map<String, Object> source = Map.of("embedding", List.of(-1.0, 0.0, 1.0, -0.5, 0.5));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-mixed", true);

        assertNotNull(result);
        assertEquals(5, result.length);
        assertEquals(-1.0f, result[0], DELTA);
        assertEquals(0.0f, result[1], DELTA);
        assertEquals(1.0f, result[2], DELTA);
        assertEquals(-0.5f, result[3], DELTA);
        assertEquals(0.5f, result[4], DELTA);
    }

    // ========== Tests for resolveKnnVectorFieldInfo ==========

    public void testResolveKnnVectorFieldInfo_whenAllUnmappedField_thenDefaultFieldInfo() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(Collections.emptyMap())),
            mockClient,
            listener
        );

        verifyVectorFieldInfo(listener, new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.DEFAULT));
    }

    public void testResolveKnnVectorFieldInfo_whenAllUnmappedField_thenUserProvidedFieldInfo() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = SpaceType.COSINESIMIL;
        VectorDataType userProvidedVectorDataType = VectorDataType.FLOAT;

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(Collections.emptyMap())),
            mockClient,
            listener
        );

        verifyVectorFieldInfo(listener, new MMRVectorFieldInfo(SpaceType.COSINESIMIL, VectorDataType.FLOAT));
    }

    public void testResolveKnnVectorFieldInfo_whenNonKnnField_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of("properties", Map.of("field", Map.of(TYPE, "keyword")));

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError = "MMR query extension cannot support non knn_vector field [index:field].";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentSpaceTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue()))
        );
        Map<String, Object> mapping1 = Map.of(
            "properties",
            Map.of(
                "field",
                Map.of(
                    TYPE,
                    KNNVectorFieldMapper.CONTENT_TYPE,
                    KNN_METHOD,
                    Map.of(METHOD_PARAMETER_SPACE_TYPE, SpaceType.COSINESIMIL.getValue())
                )
            )
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping), createMockIndexMetadata(mapping1)),
            mockClient,
            listener
        );

        String expectedError =
            "MMR query extension cannot support different space type [l2, cosinesimil] for the knn_vector field at path field.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentVectorDataTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of(
                "field",
                Map.of(
                    TYPE,
                    KNNVectorFieldMapper.CONTENT_TYPE,
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.BINARY.getValue(),
                    TOP_LEVEL_PARAMETER_SPACE_TYPE,
                    SpaceType.L2.getValue()
                )
            )
        );
        Map<String, Object> mapping1 = Map.of(
            "properties",
            Map.of(
                "field",
                Map.of(
                    TYPE,
                    KNNVectorFieldMapper.CONTENT_TYPE,
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.FLOAT.getValue(),
                    KNN_METHOD,
                    Map.of(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                )
            )
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping), createMockIndexMetadata(mapping1)),
            mockClient,
            listener
        );

        String expectedError =
            "MMR query extension cannot support different vector data type [binary, float] for the knn_vector field at path field.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentUserProvidedSpaceTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = SpaceType.COSINESIMIL;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue()))
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError =
            "The space type [cosinesimil] provided in the MMR query extension does not match the space type [l2] in target indices.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenDifferentUserProvidedVectorDataTypes_thenException() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = VectorDataType.FLOAT;
        Map<String, Object> mapping = Map.of(
            "properties",
            Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, VECTOR_DATA_TYPE_FIELD, VectorDataType.BYTE.getValue()))
        );

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        String expectedError =
            "The vector data type [float] provided in the MMR query extension does not match the vector data type [byte] in target indices.";
        verifyException(listener, IllegalArgumentException.class, expectedError);
    }

    public void testResolveKnnVectorFieldInfo_whenMappedFieldNoInfo_thenDefaultFieldInfo() {
        String vectorFieldPath = "field";
        SpaceType userProvidedSpaceType = null;
        VectorDataType userProvidedVectorDataType = null;
        Map<String, Object> mapping = Map.of("properties", Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE)));

        MMRUtil.resolveKnnVectorFieldInfo(
            vectorFieldPath,
            userProvidedSpaceType,
            userProvidedVectorDataType,
            List.of(createMockIndexMetadata(mapping)),
            mockClient,
            listener
        );

        verifyVectorFieldInfo(listener, new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.DEFAULT));
    }

    // Model-based tests not relevant for JVector (native library feature)
    // public void testResolveKnnVectorFieldInfo_whenMappedFieldWithModelId_thenFieldInfoFromModel() {
    // String vectorFieldPath = "field";
    // SpaceType userProvidedSpaceType = null;
    // VectorDataType userProvidedVectorDataType = null;
    // String modelId = "modelId";
    // Map<String, Object> mapping = Map.of(
    // "properties",
    // Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId))
    // );
    // mockModelMetadata(mockClient, Map.of(modelId, new MMRVectorFieldInfo(SpaceType.HAMMING, VectorDataType.BINARY)));
    //
    // MMRUtil.resolveKnnVectorFieldInfo(
    // vectorFieldPath,
    // userProvidedSpaceType,
    // userProvidedVectorDataType,
    // List.of(createMockIndexMetadata(mapping)),
    // mockClient,
    // listener
    // );
    //
    // verifyVectorFieldInfo(listener, new MMRVectorFieldInfo(SpaceType.HAMMING, VectorDataType.BINARY));
    // }
    //
    // public void testResolveKnnVectorFieldInfo_whenDifferentModelSpaceTypes_thenException() {
    // String vectorFieldPath = "field";
    // SpaceType userProvidedSpaceType = null;
    // VectorDataType userProvidedVectorDataType = null;
    // String modelId1 = "model1";
    // String modelId2 = "model2";
    // Map<String, Object> mapping = Map.of(
    // "properties",
    // Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
    // );
    // Map<String, Object> mapping1 = Map.of(
    // "properties",
    // Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId2))
    // );
    // mockModelMetadata(
    // mockClient,
    // Map.of(
    // modelId1,
    // new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT),
    // modelId2,
    // new MMRVectorFieldInfo(SpaceType.COSINESIMIL, VectorDataType.FLOAT)
    // )
    // );
    //
    // MMRUtil.resolveKnnVectorFieldInfo(
    // vectorFieldPath,
    // userProvidedSpaceType,
    // userProvidedVectorDataType,
    // List.of(createMockIndexMetadata(mapping), createMockIndexMetadata(mapping1)),
    // mockClient,
    // listener
    // );
    //
    // String expectedError =
    // "MMR query extension cannot support different space type [l2, cosinesimil] for the knn_vector field at path field.";
    // verifyException(listener, IllegalArgumentException.class, expectedError);
    // }
    //
    // public void testResolveKnnVectorFieldInfo_whenDifferentModelVectorDataTypes_thenException() {
    // String vectorFieldPath = "field";
    // SpaceType userProvidedSpaceType = null;
    // VectorDataType userProvidedVectorDataType = null;
    // String modelId1 = "model1";
    // String modelId2 = "model2";
    // Map<String, Object> mapping = Map.of(
    // "properties",
    // Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
    // );
    // Map<String, Object> mapping1 = Map.of(
    // "properties",
    // Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId2))
    // );
    // mockModelMetadata(
    // mockClient,
    // Map.of(
    // modelId1,
    // new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT),
    // modelId2,
    // new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.BINARY)
    // )
    // );
    //
    // MMRUtil.resolveKnnVectorFieldInfo(
    // vectorFieldPath,
    // userProvidedSpaceType,
    // userProvidedVectorDataType,
    // List.of(createMockIndexMetadata(mapping), createMockIndexMetadata(mapping1)),
    // mockClient,
    // listener
    // );
    //
    // String expectedError =
    // "MMR query extension cannot support different vector data type [float, binary] for the knn_vector field at path field.";
    // verifyException(listener, IllegalArgumentException.class, expectedError);
    // }
    //
    // public void testResolveKnnVectorFieldInfo_whenDifferentSpaceTypeFromModelAndUser_thenException() {
    // String vectorFieldPath = "field";
    // SpaceType userProvidedSpaceType = SpaceType.COSINESIMIL;
    // VectorDataType userProvidedVectorDataType = null;
    // String modelId1 = "model1";
    // Map<String, Object> mapping = Map.of(
    // "properties",
    // Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
    // );
    // mockModelMetadata(mockClient, Map.of(modelId1, new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT)));
    //
    // MMRUtil.resolveKnnVectorFieldInfo(
    // vectorFieldPath,
    // userProvidedSpaceType,
    // userProvidedVectorDataType,
    // List.of(createMockIndexMetadata(mapping)),
    // mockClient,
    // listener
    // );
    //
    // String expectedError =
    // "The space type [cosinesimil] provided in the MMR query extension does not match the space type [l2] in target indices.";
    // verifyException(listener, IllegalArgumentException.class, expectedError);
    // }
    //
    // public void testResolveKnnVectorFieldInfo_whenDifferentVectorDataTypeFromModelAndUser_thenException() {
    // String vectorFieldPath = "field";
    // SpaceType userProvidedSpaceType = null;
    // VectorDataType userProvidedVectorDataType = VectorDataType.BINARY;
    // String modelId1 = "model1";
    // Map<String, Object> mapping = Map.of(
    // "properties",
    // Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
    // );
    // mockModelMetadata(mockClient, Map.of(modelId1, new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT)));
    //
    // MMRUtil.resolveKnnVectorFieldInfo(
    // vectorFieldPath,
    // userProvidedSpaceType,
    // userProvidedVectorDataType,
    // List.of(createMockIndexMetadata(mapping)),
    // mockClient,
    // listener
    // );
    //
    // String expectedError =
    // "The vector data type [binary] provided in the MMR query extension does not match the vector data type [float] in target indices.";
    // verifyException(listener, IllegalArgumentException.class, expectedError);
    // }
    //
    // public void testResolveKnnVectorFieldInfo_whenModelNotFount_thenException() {
    // String vectorFieldPath = "field";
    // SpaceType userProvidedSpaceType = null;
    // VectorDataType userProvidedVectorDataType = null;
    // String modelId1 = "model1";
    // Map<String, Object> mapping = Map.of(
    // "properties",
    // Map.of("field", Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, MODEL_ID, modelId1))
    // );
    // mockModelMetadata(mockClient, Collections.emptyMap());
    //
    // MMRUtil.resolveKnnVectorFieldInfo(
    // vectorFieldPath,
    // userProvidedSpaceType,
    // userProvidedVectorDataType,
    // List.of(createMockIndexMetadata(mapping)),
    // mockClient,
    // listener
    // );
    //
    // String expectedError =
    // "Failed to retrieve model(s) to resolve the space type and vector data type for the MMR query extension. Errors: Model ID model1 not
    // found.";
    // verifyException(listener, RuntimeException.class, expectedError);
    // }

    private IndexMetadata createMockIndexMetadata(Map<String, Object> mappings) {
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        when(indexMetadata.getIndex()).thenReturn(new Index("index", "uuid"));
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        when(mappingMetadata.sourceAsMap()).thenReturn(mappings);
        return indexMetadata;
    }

    private void verifyVectorFieldInfo(ActionListener<MMRVectorFieldInfo> listener, MMRVectorFieldInfo vectorFieldInfo) {
        ArgumentCaptor<MMRVectorFieldInfo> captor = ArgumentCaptor.forClass(MMRVectorFieldInfo.class);
        verify(listener).onResponse(captor.capture());
        SpaceType capturedSpaceType = captor.getValue().getSpaceType();
        VectorDataType capturedVectorDataType = captor.getValue().getVectorDataType();
        assertEquals(vectorFieldInfo.getSpaceType(), capturedSpaceType);
        assertEquals(vectorFieldInfo.getVectorDataType(), capturedVectorDataType);
    }

    public void testShouldGenerateMMRProcessor_whenExtContainsBuilder_thenReturnTrue() {
        SearchRequest searchRequest = new SearchRequest();
        SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
        searchSourceBuilder.ext(Collections.singletonList(new MMRSearchExtBuilder.Builder().build()));
        searchRequest.source(searchSourceBuilder);

        ProcessorGenerationContext ctx = new ProcessorGenerationContext(searchRequest);

        assertTrue(MMRUtil.shouldGenerateMMRProcessor(ctx));
    }

    public void testShouldGenerateMMRProcessor_whenNoExt_thenReturnFalse() {
        SearchRequest searchRequest = new SearchRequest();

        ProcessorGenerationContext ctx = new ProcessorGenerationContext(searchRequest);

        assertFalse(MMRUtil.shouldGenerateMMRProcessor(ctx));
    }

    // ========== Tests for getMMRFieldMappingByPath ==========

    public void testGetMMRFieldMappingByPath_SimpleField() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of("embedding", Map.of("type", "knn_vector", "dimension", 128, "space_type", "l2"))
        );

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "embedding");

        assertNotNull(result);
        assertEquals("knn_vector", result.get("type"));
        assertEquals(128, result.get("dimension"));
        assertEquals("l2", result.get("space_type"));
    }

    public void testGetMMRFieldMappingByPath_NestedField() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of(
                "user",
                Map.of(
                    "properties",
                    Map.of("profile", Map.of("properties", Map.of("embedding", Map.of("type", "knn_vector", "dimension", 256))))
                )
            )
        );

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "user.profile.embedding");

        assertNotNull(result);
        assertEquals("knn_vector", result.get("type"));
        assertEquals(256, result.get("dimension"));
    }

    public void testGetMMRFieldMappingByPath_NonKnnField() {
        Map<String, Object> mappings = Map.of("properties", Map.of("title", Map.of("type", "text")));

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "title");

        assertNotNull(result);
        assertEquals("text", result.get("type"));
    }

    public void testGetMMRFieldMappingByPath_MissingField() {
        Map<String, Object> mappings = Map.of("properties", Map.of("other_field", Map.of("type", "text")));

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "embedding");

        assertNull(result);
    }

    public void testGetMMRFieldMappingByPath_NullMappings() {
        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(null, "embedding");
        assertNull(result);
    }

    public void testGetMMRFieldMappingByPath_NoProperties() {
        Map<String, Object> mappings = Map.of("settings", Map.of("index", "value"));

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "embedding");

        assertNull(result);
    }

    public void testGetMMRFieldMappingByPath_PropertiesNotMap() {
        Map<String, Object> mappings = Map.of("properties", "not_a_map");

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "embedding");

        assertNull(result);
    }

    public void testGetMMRFieldMappingByPath_NestedFieldType() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of(
                "nested_field",
                Map.of(
                    "type",
                    ObjectMapper.NESTED_CONTENT_TYPE,
                    "properties",
                    Map.of("embedding", Map.of("type", "knn_vector", "dimension", 128))
                )
            )
        );

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.getMMRFieldMappingByPath(mappings, "nested_field.embedding")
        );
        assertNotNull(ex);
    }

    public void testGetMMRFieldMappingByPath_NestedInMiddleOfPath() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of(
                "level1",
                Map.of(
                    "properties",
                    Map.of(
                        "nested_level",
                        Map.of("type", ObjectMapper.NESTED_CONTENT_TYPE, "properties", Map.of("embedding", Map.of("type", "knn_vector")))
                    )
                )
            )
        );

        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> MMRUtil.getMMRFieldMappingByPath(mappings, "level1.nested_level.embedding")
        );
        assertNotNull(ex);
    }

    public void testGetMMRFieldMappingByPath_ObjectFieldType() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of("user", Map.of("type", "object", "properties", Map.of("embedding", Map.of("type", "knn_vector", "dimension", 64))))
        );

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "user.embedding");

        assertNotNull(result);
        assertEquals("knn_vector", result.get("type"));
        assertEquals(64, result.get("dimension"));
    }

    public void testGetMMRFieldMappingByPath_FieldWithoutType() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of("user", Map.of("properties", Map.of("embedding", Map.of("type", "knn_vector"))))
        );

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "user.embedding");

        assertNotNull(result);
        assertEquals("knn_vector", result.get("type"));
    }

    public void testGetMMRFieldMappingByPath_ComplexMapping() {
        Map<String, Object> mappings = new HashMap<>();
        Map<String, Object> properties = new HashMap<>();

        Map<String, Object> userField = new HashMap<>();
        Map<String, Object> userProperties = new HashMap<>();

        Map<String, Object> profileField = new HashMap<>();
        Map<String, Object> profileProperties = new HashMap<>();

        Map<String, Object> embeddingField = new HashMap<>();
        embeddingField.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        embeddingField.put("dimension", 512);
        embeddingField.put("space_type", "cosinesimil");

        profileProperties.put("embedding", embeddingField);
        profileField.put("properties", profileProperties);

        userProperties.put("profile", profileField);
        userField.put("properties", userProperties);

        properties.put("user", userField);
        mappings.put("properties", properties);

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "user.profile.embedding");

        assertNotNull(result);
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, result.get("type"));
        assertEquals(512, result.get("dimension"));
        assertEquals("cosinesimil", result.get("space_type"));
    }

    public void testGetMMRFieldMappingByPath_whenInNestedField_thenException() {
        Map<String, Object> mappings = new HashMap<>();
        Map<String, Object> userMapping = new HashMap<>();
        userMapping.put("type", "nested");

        Map<String, Object> properties = new HashMap<>();
        properties.put("user", userMapping);
        mappings.put("properties", properties);

        String fieldPath = "user.profile.age";

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> getMMRFieldMappingByPath(mappings, fieldPath));

        String expectedError = "MMR search extension cannot support the field user.profile.age because it is in the nested field user.";
        assertEquals(expectedError, ex.getMessage());
    }

    // ========== Tests for shouldGenerateMMRProcessor ==========
}
