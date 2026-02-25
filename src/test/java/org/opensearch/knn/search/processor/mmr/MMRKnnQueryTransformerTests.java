/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.transport.client.Client;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Unit tests for {@link MMRKnnQueryTransformer}
 */
public class MMRKnnQueryTransformerTests extends LuceneTestCase {

    private MMRKnnQueryTransformer transformer;
    private KNNQueryBuilder mockQueryBuilder;
    private ActionListener<Void> mockListener;
    private MMRTransformContext mockContext;
    private MMRRerankContext mockRerankContext;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        transformer = new MMRKnnQueryTransformer();
        mockQueryBuilder = mock(KNNQueryBuilder.class);
        mockListener = mock(ActionListener.class);
        mockRerankContext = new MMRRerankContext();
        
        // Setup default mock context
        mockContext = mock(MMRTransformContext.class);
        when(mockContext.getMmrRerankContext()).thenReturn(mockRerankContext);
        when(mockContext.getCandidates()).thenReturn(100);
        when(mockContext.getLocalIndexMetadataList()).thenReturn(new ArrayList<>());
        when(mockContext.getRemoteIndices()).thenReturn(new ArrayList<>());
        when(mockContext.getClient()).thenReturn(mock(Client.class));
    }

    // ============================================
    // Test: getQueryName()
    // ============================================

    @Test
    public void testGetQueryName_ReturnsKnnConstant() {
        String queryName = transformer.getQueryName();
        assertEquals("knn", queryName);
        assertEquals(KNNQueryBuilder.NAME, queryName);
    }

    // ============================================
    // Test: K value setting when maxDistance/minScore are null
    // ============================================

    @Test
    public void testTransform_SetsKValueWhenMaxDistanceAndMinScoreAreNull() {
        // Setup: maxDistance and minScore are null
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K should be set to candidates value
        verify(mockQueryBuilder).setK(100);
        verify(mockListener).onResponse(null);
    }

    @Test
    public void testTransform_DoesNotSetKValueWhenMaxDistanceIsSet() {
        // Setup: maxDistance is set
        when(mockQueryBuilder.getMaxDistance()).thenReturn(0.5f);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K should NOT be set
        verify(mockQueryBuilder, never()).setK(anyInt());
        verify(mockListener).onResponse(null);
    }

    @Test
    public void testTransform_DoesNotSetKValueWhenMinScoreIsSet() {
        // Setup: minScore is set
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(0.8f);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K should NOT be set
        verify(mockQueryBuilder, never()).setK(anyInt());
        verify(mockListener).onResponse(null);
    }

    @Test
    public void testTransform_DoesNotSetKValueWhenBothMaxDistanceAndMinScoreAreSet() {
        // Setup: both maxDistance and minScore are set
        when(mockQueryBuilder.getMaxDistance()).thenReturn(0.5f);
        when(mockQueryBuilder.getMinScore()).thenReturn(0.8f);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K should NOT be set
        verify(mockQueryBuilder, never()).setK(anyInt());
        verify(mockListener).onResponse(null);
    }

    // ============================================
    // Test: Early return when vector field info is already resolved
    // ============================================

    @Test
    public void testTransform_ReturnsEarlyWhenVectorFieldInfoAlreadyResolved() {
        // Setup: vector field info already resolved
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Should return early without calling resolveKnnVectorFieldInfo
        verify(mockListener).onResponse(null);
        verify(mockContext, never()).getLocalIndexMetadataList();
        verify(mockContext, never()).getClient();
    }

    // ============================================
    // Test: Field name validation
    // ============================================

    @Test
    public void testTransform_ThrowsExceptionWhenFieldNameIsNull() {
        // Setup: field name is null
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn(null);
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(false);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Should call onFailure with IllegalArgumentException
        ArgumentCaptor<Exception> exceptionCaptor = ArgumentCaptor.forClass(Exception.class);
        verify(mockListener).onFailure(exceptionCaptor.capture());
        
        Exception exception = exceptionCaptor.getValue();
        assertTrue(exception instanceof IllegalArgumentException);
        assertTrue(exception.getMessage().contains("Field name of the knn query should not be null"));
    }

    @Test
    public void testTransform_ThrowsExceptionWhenFieldNameIsEmpty() {
        // Setup: field name is empty string
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(false);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Should set empty field path in context (no exception for empty string)
        assertEquals("", mockRerankContext.getVectorFieldPath());
    }

    // ============================================
    // Test: Vector field path setting
    // ============================================

    @Test
    public void testTransform_SetsVectorFieldPathInRerankContext() {
        // Setup: vector field info NOT resolved so it sets the field path
        String fieldName = "my_vector_field";
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn(fieldName);
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(false);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Field path should be set in rerank context before attempting resolution
        assertEquals(fieldName, mockRerankContext.getVectorFieldPath());
    }

    @Test
    public void testTransform_SetsNestedVectorFieldPath() {
        // Setup: nested field path, vector field info NOT resolved
        String nestedFieldName = "nested.path.to.vector_field";
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn(nestedFieldName);
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(false);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Nested field path should be set correctly
        assertEquals(nestedFieldName, mockRerankContext.getVectorFieldPath());
    }

    // ============================================
    // Test: Exception handling
    // ============================================

    @Test
    public void testTransform_HandlesExceptionDuringTransformation() {
        // Setup: mock to throw exception
        when(mockQueryBuilder.getMaxDistance()).thenThrow(new RuntimeException("Test exception"));

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Should call onFailure
        ArgumentCaptor<Exception> exceptionCaptor = ArgumentCaptor.forClass(Exception.class);
        verify(mockListener).onFailure(exceptionCaptor.capture());
        
        Exception exception = exceptionCaptor.getValue();
        assertTrue(exception instanceof RuntimeException);
        assertEquals("Test exception", exception.getMessage());
    }

    @Test
    public void testTransform_HandlesNullPointerException() {
        // Setup: context returns null for rerank context
        when(mockContext.getMmrRerankContext()).thenReturn(null);
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(false);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Should call onFailure with NullPointerException
        ArgumentCaptor<Exception> exceptionCaptor = ArgumentCaptor.forClass(Exception.class);
        verify(mockListener).onFailure(exceptionCaptor.capture());
        
        Exception exception = exceptionCaptor.getValue();
        assertTrue("Expected NullPointerException but got: " + exception.getClass().getName(),
                   exception instanceof NullPointerException);
    }

    // ============================================
    // Test: Integration scenarios
    // ============================================

    @Test
    public void testTransform_CompleteWorkflowWithAllParameters() {
        // Setup: complete scenario with all parameters, field info NOT resolved
        String fieldName = "embeddings";
        Integer candidates = 200;
        
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn(fieldName);
        when(mockContext.getCandidates()).thenReturn(candidates);
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(false);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K set and field path set before resolution attempt
        verify(mockQueryBuilder).setK(candidates);
        assertEquals(fieldName, mockRerankContext.getVectorFieldPath());
    }

    @Test
    public void testTransform_WithDifferentCandidateValues() {
        // Test with various candidate values
        int[] candidateValues = {10, 50, 100, 500, 1000};
        
        for (int candidates : candidateValues) {
            // Reset mocks
            reset(mockQueryBuilder, mockListener);
            mockRerankContext = new MMRRerankContext();
            when(mockContext.getMmrRerankContext()).thenReturn(mockRerankContext);
            
            // Setup
            when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
            when(mockQueryBuilder.getMinScore()).thenReturn(null);
            when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
            when(mockContext.getCandidates()).thenReturn(candidates);
            when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

            // Execute
            transformer.transform(mockQueryBuilder, mockListener, mockContext);

            // Verify: K should be set to the specific candidate value
            verify(mockQueryBuilder).setK(candidates);
            verify(mockListener).onResponse(null);
        }
    }

    // ============================================
    // Test: Edge cases
    // ============================================

    @Test
    public void testTransform_WithZeroCandidates() {
        // Setup: zero candidates (edge case)
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.getCandidates()).thenReturn(0);
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Should set K to 0 (even though it's unusual)
        verify(mockQueryBuilder).setK(0);
        verify(mockListener).onResponse(null);
    }

    @Test
    public void testTransform_WithNegativeCandidates() {
        // Setup: negative candidates (edge case)
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.getCandidates()).thenReturn(-10);
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Should set K to negative value (validation happens elsewhere)
        verify(mockQueryBuilder).setK(-10);
        verify(mockListener).onResponse(null);
    }

    @Test
    public void testTransform_WithVeryLargeCandidates() {
        // Setup: very large candidates value
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.getCandidates()).thenReturn(Integer.MAX_VALUE);
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: Should handle large values
        verify(mockQueryBuilder).setK(Integer.MAX_VALUE);
        verify(mockListener).onResponse(null);
    }

    // ============================================
    // Test: Multiple sequential transforms
    // ============================================

    @Test
    public void testTransform_MultipleSequentialCalls() {
        // Test that transformer can be reused for multiple transforms
        String[] fieldNames = {"field1", "field2", "field3"};
        
        for (String fieldName : fieldNames) {
            // Reset mocks
            reset(mockQueryBuilder, mockListener);
            mockRerankContext = new MMRRerankContext();
            when(mockContext.getMmrRerankContext()).thenReturn(mockRerankContext);
            
            // Setup: field info NOT resolved so field path gets set
            when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
            when(mockQueryBuilder.getMinScore()).thenReturn(null);
            when(mockQueryBuilder.fieldName()).thenReturn(fieldName);
            when(mockContext.getCandidates()).thenReturn(100);
            when(mockContext.isVectorFieldInfoResolved()).thenReturn(false);

            // Execute
            transformer.transform(mockQueryBuilder, mockListener, mockContext);

            // Verify: Field path should be set
            assertEquals(fieldName, mockRerankContext.getVectorFieldPath());
        }
    }

    // ============================================
    // Test: Boundary conditions for maxDistance and minScore
    // ============================================

    @Test
    public void testTransform_WithZeroMaxDistance() {
        // Setup: maxDistance is 0.0
        when(mockQueryBuilder.getMaxDistance()).thenReturn(0.0f);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K should NOT be set (maxDistance is not null)
        verify(mockQueryBuilder, never()).setK(anyInt());
        verify(mockListener).onResponse(null);
    }

    @Test
    public void testTransform_WithZeroMinScore() {
        // Setup: minScore is 0.0
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(0.0f);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K should NOT be set (minScore is not null)
        verify(mockQueryBuilder, never()).setK(anyInt());
        verify(mockListener).onResponse(null);
    }

    @Test
    public void testTransform_WithNegativeMaxDistance() {
        // Setup: negative maxDistance
        when(mockQueryBuilder.getMaxDistance()).thenReturn(-1.0f);
        when(mockQueryBuilder.getMinScore()).thenReturn(null);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K should NOT be set (maxDistance is not null, validation happens elsewhere)
        verify(mockQueryBuilder, never()).setK(anyInt());
        verify(mockListener).onResponse(null);
    }

    @Test
    public void testTransform_WithNegativeMinScore() {
        // Setup: negative minScore
        when(mockQueryBuilder.getMaxDistance()).thenReturn(null);
        when(mockQueryBuilder.getMinScore()).thenReturn(-0.5f);
        when(mockQueryBuilder.fieldName()).thenReturn("vector_field");
        when(mockContext.isVectorFieldInfoResolved()).thenReturn(true);

        // Execute
        transformer.transform(mockQueryBuilder, mockListener, mockContext);

        // Verify: K should NOT be set (minScore is not null, validation happens elsewhere)
        verify(mockQueryBuilder, never()).setK(anyInt());
        verify(mockListener).onResponse(null);
    }
}
