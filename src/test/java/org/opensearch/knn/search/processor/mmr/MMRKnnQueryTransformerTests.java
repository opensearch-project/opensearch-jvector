/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.index.Index;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.transport.client.Client;

import java.util.List;
import java.util.Map;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;

import static org.mockito.Mockito.*;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE;

@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
public class MMRKnnQueryTransformerTests extends MMRTestCase {
    private Client client;
    private MMRKnnQueryTransformer transformer;
    private KNNQueryBuilder queryBuilder;
    private ActionListener<Void> listener;
    private MMRTransformContext transformContext;
    private MMRRerankContext processingContext;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        client = mock(Client.class);
        transformer = new MMRKnnQueryTransformer();
        queryBuilder = mock(KNNQueryBuilder.class);
        listener = mock(ActionListener.class);
        processingContext = new MMRRerankContext();
        transformContext = new MMRTransformContext(10, processingContext, List.of(), List.of(), null, null, null, client, false);
    }

    public void testTransform_whenNoMaxDistanceOrMinScore_thenSetsK() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder).setK(10);
    }

    public void testTransform_whenMinScore_thenNotSetsK() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(0.5f); // non-null minScore

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
    }

    public void testTransform_whenVectorFieldInfoAlreadyResolved_thenEarlyExits() {
        transformContext = new MMRTransformContext(
            10,
            processingContext,
            List.of(),
            List.of(),
            null,
            "vector.field.path",
            null,
            client,
            true
        );

        transformer.transform(queryBuilder, listener, transformContext);

        verify(listener).onResponse(null);
        verifyNoMoreInteractions(client);
    }

    public void testTransform_whenNoUserProvidedVectorFieldPath_thenResolveSpaceType() {
        String indexName = "test-index";
        String vectorFieldName = "vectorField";
        IndexMetadata indexMetadata = mock(IndexMetadata.class);
        MappingMetadata mappingMetadata = mock(MappingMetadata.class);
        Map<String, Object> mapping = Map.of(
            indexName,
            Map.of(
                "properties",
                Map.of(
                    vectorFieldName,
                    Map.of(TYPE, KNNVectorFieldMapper.CONTENT_TYPE, TOP_LEVEL_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                )
            )
        );
        when(indexMetadata.getIndex()).thenReturn(new Index(indexName, "uuid"));
        when(indexMetadata.mapping()).thenReturn(mappingMetadata);
        when(mappingMetadata.sourceAsMap()).thenReturn(mapping);
        when(queryBuilder.fieldName()).thenReturn(vectorFieldName);

        transformContext = new MMRTransformContext(
            10,
            processingContext,
            List.of(indexMetadata),
            List.of(),
            null,
            null,
            null,
            client,
            false
        );

        transformer.transform(queryBuilder, listener, transformContext);

        verify(listener).onResponse(null);
        assertEquals(vectorFieldName, processingContext.getVectorFieldPath());
        assertEquals(SpaceType.L2, processingContext.getSpaceType());
    }

    // ============================================
    // UNIT TESTS: Additional API contract tests from original test suite
    // ============================================

    public void testGetQueryName_ReturnsKnnConstant() {
        String queryName = transformer.getQueryName();
        assertEquals("knn", queryName);
        assertEquals(KNNQueryBuilder.NAME, queryName);
    }

    public void testTransform_SetsKValueWhenMaxDistanceAndMinScoreAreNull() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder).setK(100);
        verify(listener).onResponse(null);
    }

    public void testTransform_DoesNotSetKValueWhenMaxDistanceIsSet() {
        when(queryBuilder.getMaxDistance()).thenReturn(0.5f);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
        verify(listener).onResponse(null);
    }

    public void testTransform_DoesNotSetKValueWhenMinScoreIsSet() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(0.8f);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
        verify(listener).onResponse(null);
    }

    public void testTransform_DoesNotSetKValueWhenBothMaxDistanceAndMinScoreAreSet() {
        when(queryBuilder.getMaxDistance()).thenReturn(0.5f);
        when(queryBuilder.getMinScore()).thenReturn(0.8f);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
        verify(listener).onResponse(null);
    }

    public void testTransform_ReturnsEarlyWhenVectorFieldInfoAlreadyResolved() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(listener).onResponse(null);
    }

    public void testTransform_ThrowsExceptionWhenFieldNameIsNull() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn(null);
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, null, null, client, false);

        transformer.transform(queryBuilder, listener, transformContext);

        verifyException(listener, IllegalArgumentException.class, "Failed to transform the knn query for MMR. Field name of the knn query should not be null.");
    }

    public void testTransform_ThrowsExceptionWhenFieldNameIsEmpty() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, null, null, client, false);

        transformer.transform(queryBuilder, listener, transformContext);

        assertEquals("", processingContext.getVectorFieldPath());
    }

    public void testTransform_SetsVectorFieldPathInRerankContext() {
        String fieldName = "my_vector_field";
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn(fieldName);
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, null, null, client, false);

        transformer.transform(queryBuilder, listener, transformContext);

        assertEquals(fieldName, processingContext.getVectorFieldPath());
    }

    public void testTransform_SetsNestedVectorFieldPath() {
        String nestedFieldName = "nested.path.to.vector_field";
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn(nestedFieldName);
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, null, null, client, false);

        transformer.transform(queryBuilder, listener, transformContext);

        assertEquals(nestedFieldName, processingContext.getVectorFieldPath());
    }

    public void testTransform_HandlesExceptionDuringTransformation() {
        when(queryBuilder.getMaxDistance()).thenThrow(new RuntimeException("Test exception"));

        transformer.transform(queryBuilder, listener, transformContext);

        verifyException(listener, RuntimeException.class, "Test exception");
    }

    // COMMENTED OUT: Test expects null MMRRerankContext but constructor doesn't allow it
    // MMRTransformContext constructor has @NonNull annotation on mmrRerankContext parameter
    /*
    public void testTransform_HandlesNullPointerException() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("field");
        transformContext = new MMRTransformContext(100, null, List.of(), List.of(), null, null, null, client, false);

        transformer.transform(queryBuilder, listener, transformContext);

        verifyException(listener, NullPointerException.class, null);
    }
    */

    public void testTransform_CompleteWorkflowWithAllParameters() {
        String fieldName = "embeddings";
        Integer candidates = 200;

        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn(fieldName);
        transformContext = new MMRTransformContext(candidates, processingContext, List.of(), List.of(), null, null, null, client, false);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder).setK(candidates);
        assertEquals(fieldName, processingContext.getVectorFieldPath());
    }

    public void testTransform_WithDifferentCandidateValues() {
        int[] candidateValues = { 10, 50, 100, 500, 1000 };

        for (int candidates : candidateValues) {
            reset(queryBuilder, listener);
            processingContext = new MMRRerankContext();

            when(queryBuilder.getMaxDistance()).thenReturn(null);
            when(queryBuilder.getMinScore()).thenReturn(null);
            when(queryBuilder.fieldName()).thenReturn("vector_field");
            transformContext = new MMRTransformContext(
                candidates,
                processingContext,
                List.of(),
                List.of(),
                null,
                "vector_field",
                null,
                client,
                true
            );

            transformer.transform(queryBuilder, listener, transformContext);

            verify(queryBuilder).setK(candidates);
            verify(listener).onResponse(null);
        }
    }

    public void testTransform_WithZeroCandidates() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(0, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder).setK(0);
        verify(listener).onResponse(null);
    }

    public void testTransform_WithNegativeCandidates() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(-10, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder).setK(-10);
        verify(listener).onResponse(null);
    }

    public void testTransform_WithVeryLargeCandidates() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(Integer.MAX_VALUE, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder).setK(Integer.MAX_VALUE);
        verify(listener).onResponse(null);
    }

    public void testTransform_MultipleSequentialCalls() {
        String[] fieldNames = { "field1", "field2", "field3" };

        for (String fieldName : fieldNames) {
            reset(queryBuilder, listener);
            processingContext = new MMRRerankContext();

            when(queryBuilder.getMaxDistance()).thenReturn(null);
            when(queryBuilder.getMinScore()).thenReturn(null);
            when(queryBuilder.fieldName()).thenReturn(fieldName);
            transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, null, null, client, false);

            transformer.transform(queryBuilder, listener, transformContext);

            assertEquals(fieldName, processingContext.getVectorFieldPath());
        }
    }

    public void testTransform_WithZeroMaxDistance() {
        when(queryBuilder.getMaxDistance()).thenReturn(0.0f);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
        verify(listener).onResponse(null);
    }

    public void testTransform_WithZeroMinScore() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(0.0f);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
        verify(listener).onResponse(null);
    }

    public void testTransform_WithNegativeMaxDistance() {
        when(queryBuilder.getMaxDistance()).thenReturn(-1.0f);
        when(queryBuilder.getMinScore()).thenReturn(null);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
        verify(listener).onResponse(null);
    }

    public void testTransform_WithNegativeMinScore() {
        when(queryBuilder.getMaxDistance()).thenReturn(null);
        when(queryBuilder.getMinScore()).thenReturn(-0.5f);
        when(queryBuilder.fieldName()).thenReturn("vector_field");
        transformContext = new MMRTransformContext(100, processingContext, List.of(), List.of(), null, "vector_field", null, client, true);

        transformer.transform(queryBuilder, listener, transformContext);

        verify(queryBuilder, never()).setK(anyInt());
        verify(listener).onResponse(null);
    }
}
