/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.search.pipeline.PipelineProcessingContext;
import org.opensearch.transport.client.Client;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.mockito.Mockito.*;

/**
 * Unit tests for {@link MMROverSampleProcessor}
 */
public class MMROverSampleProcessorTests extends LuceneTestCase {

    private MMROverSampleProcessor processor;
    private Client mockClient;
    private Map<String, MMRQueryTransformer<? extends QueryBuilder>> mockTransformers;
    private String testTag = "test-tag";
    private boolean testIgnoreFailure = false;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        mockClient = mock(Client.class);
        mockTransformers = new HashMap<>();
        processor = new MMROverSampleProcessor(testTag, testIgnoreFailure, mockClient, mockTransformers);
    }

    // ============================================
    // Test: Public getter methods
    // ============================================

    @Test
    public void testGetType_ReturnsCorrectType() {
        assertEquals("mmr_over_sample", processor.getType());
        assertEquals(MMROverSampleProcessor.TYPE, processor.getType());
    }

    @Test
    public void testGetTag_ReturnsConfiguredTag() {
        assertEquals(testTag, processor.getTag());
        
        // Test with different tag
        MMROverSampleProcessor customProcessor = new MMROverSampleProcessor(
            "custom-tag",
            false,
            mockClient,
            mockTransformers
        );
        assertEquals("custom-tag", customProcessor.getTag());
    }

    @Test
    public void testGetDescription_ReturnsNonEmptyDescription() {
        String description = processor.getDescription();
        assertNotNull(description);
        assertFalse(description.isEmpty());
        assertTrue(description.contains("oversample"));
        assertTrue(description.contains("Maximal Marginal Relevance"));
        assertEquals(MMROverSampleProcessor.DESCRIPTION, description);
    }

    @Test
    public void testIsIgnoreFailure_ReturnsConfiguredValue() {
        assertEquals(testIgnoreFailure, processor.isIgnoreFailure());
        
        // Test with ignore failure enabled
        MMROverSampleProcessor processorWithIgnoreFailure = new MMROverSampleProcessor(
            "tag",
            true,
            mockClient,
            mockTransformers
        );
        assertTrue(processorWithIgnoreFailure.isIgnoreFailure());
    }

    @Test
    public void testGetExecutionStage_ReturnsPostUserDefined() {
        assertEquals(
            MMROverSampleProcessor.ExecutionStage.POST_USER_DEFINED,
            processor.getExecutionStage()
        );
    }

    // ============================================
    // Test: Synchronous processRequest methods throw UnsupportedOperationException
    // ============================================

    @Test
    public void testProcessRequest_ThrowsUnsupportedOperationException() {
        SearchRequest mockRequest = mock(SearchRequest.class);
        
        try {
            processor.processRequest(mockRequest);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("mmr_over_sample"));
            assertTrue(e.getMessage().contains("synchronously"));
        }
    }

    @Test
    public void testProcessRequestWithContext_ThrowsUnsupportedOperationException() {
        SearchRequest mockRequest = mock(SearchRequest.class);
        PipelineProcessingContext mockContext = mock(PipelineProcessingContext.class);
        
        try {
            processor.processRequest(mockRequest, mockContext);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("mmr_over_sample"));
            assertTrue(e.getMessage().contains("synchronously"));
        }
    }

    @Test
    public void testSynchronousMethodsErrorMessage_ContainsProcessorType() {
        SearchRequest mockRequest = mock(SearchRequest.class);
        
        try {
            processor.processRequest(mockRequest);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            String message = e.getMessage();
            assertTrue("Error message should contain processor type", 
                      message.contains("mmr_over_sample"));
            assertTrue("Error message should mention synchronous operation", 
                      message.contains("synchronously"));
        }
    }

    // ============================================
    // Test: Constructor variations
    // ============================================

    @Test
    public void testConstructor_WithAllParameters() {
        String tag = "custom-tag";
        boolean ignoreFailure = true;
        Client client = mock(Client.class);
        Map<String, MMRQueryTransformer<? extends QueryBuilder>> transformers = new HashMap<>();
        
        MMROverSampleProcessor proc = new MMROverSampleProcessor(tag, ignoreFailure, client, transformers);
        
        assertEquals(tag, proc.getTag());
        assertEquals(ignoreFailure, proc.isIgnoreFailure());
        assertEquals("mmr_over_sample", proc.getType());
        assertNotNull(proc.getDescription());
    }

    @Test
    public void testConstructor_WithEmptyTransformers() {
        Map<String, MMRQueryTransformer<? extends QueryBuilder>> emptyTransformers = Collections.emptyMap();
        
        MMROverSampleProcessor proc = new MMROverSampleProcessor(
            "tag",
            false,
            mockClient,
            emptyTransformers
        );
        
        assertNotNull(proc);
        assertEquals("tag", proc.getTag());
        assertEquals("mmr_over_sample", proc.getType());
    }

    @Test
    public void testConstructor_WithNullTag() {
        MMROverSampleProcessor proc = new MMROverSampleProcessor(
            null,
            false,
            mockClient,
            mockTransformers
        );
        
        assertNull(proc.getTag());
        assertEquals("mmr_over_sample", proc.getType());
    }

    @Test
    public void testConstructor_WithDifferentIgnoreFailureValues() {
        MMROverSampleProcessor procTrue = new MMROverSampleProcessor(
            "tag1",
            true,
            mockClient,
            mockTransformers
        );
        
        MMROverSampleProcessor procFalse = new MMROverSampleProcessor(
            "tag2",
            false,
            mockClient,
            mockTransformers
        );
        
        assertTrue(procTrue.isIgnoreFailure());
        assertFalse(procFalse.isIgnoreFailure());
    }

    // ============================================
    // Test: Multiple processor instances are independent
    // ============================================

    @Test
    public void testMultipleInstances_AreIndependent() {
        MMROverSampleProcessor proc1 = new MMROverSampleProcessor(
            "tag1",
            true,
            mockClient,
            mockTransformers
        );
        
        MMROverSampleProcessor proc2 = new MMROverSampleProcessor(
            "tag2",
            false,
            mockClient,
            mockTransformers
        );
        
        // Verify independence
        assertEquals("tag1", proc1.getTag());
        assertEquals("tag2", proc2.getTag());
        assertTrue(proc1.isIgnoreFailure());
        assertFalse(proc2.isIgnoreFailure());
        
        // Both should have same type and execution stage (class-level constants)
        assertEquals(proc1.getType(), proc2.getType());
        assertEquals(proc1.getExecutionStage(), proc2.getExecutionStage());
        assertEquals(proc1.getDescription(), proc2.getDescription());
    }

    // ============================================
    // Test: Factory class
    // ============================================

    @Test
    public void testFactory_TypeConstant() {
        assertEquals("mmr_over_sample_factory", MMROverSampleProcessor.MMROverSampleProcessorFactory.TYPE);
    }

    @Test
    public void testFactory_Constructor() {
        Client client = mock(Client.class);
        Map<String, MMRQueryTransformer<? extends QueryBuilder>> transformers = new HashMap<>();
        
        MMROverSampleProcessor.MMROverSampleProcessorFactory factory = 
            new MMROverSampleProcessor.MMROverSampleProcessorFactory(client, transformers);
        
        assertNotNull(factory);
    }

    @Test
    public void testFactory_ConstructorWithNullClient() {
        Map<String, MMRQueryTransformer<? extends QueryBuilder>> transformers = new HashMap<>();
        
        MMROverSampleProcessor.MMROverSampleProcessorFactory factory = 
            new MMROverSampleProcessor.MMROverSampleProcessorFactory(null, transformers);
        
        assertNotNull(factory);
    }

    @Test
    public void testFactory_ConstructorWithNullTransformers() {
        Client client = mock(Client.class);
        
        MMROverSampleProcessor.MMROverSampleProcessorFactory factory = 
            new MMROverSampleProcessor.MMROverSampleProcessorFactory(client, null);
        
        assertNotNull(factory);
    }

    // ============================================
    // Test: Constants and default values
    // ============================================

    @Test
    public void testConstants_HaveExpectedValues() {
        assertEquals("mmr_over_sample", MMROverSampleProcessor.TYPE);
        assertNotNull(MMROverSampleProcessor.DESCRIPTION);
        assertFalse(MMROverSampleProcessor.DESCRIPTION.isEmpty());
        assertTrue(MMROverSampleProcessor.DESCRIPTION.length() > 20);
    }

    @Test
    public void testTypeConstant_MatchesGetType() {
        assertEquals(MMROverSampleProcessor.TYPE, processor.getType());
    }

    @Test
    public void testDescriptionConstant_MatchesGetDescription() {
        assertEquals(MMROverSampleProcessor.DESCRIPTION, processor.getDescription());
    }

    // ============================================
    // Test: Execution stage behavior
    // ============================================

    @Test
    public void testExecutionStage_IsConsistentAcrossInstances() {
        MMROverSampleProcessor proc1 = new MMROverSampleProcessor(
            "tag1",
            true,
            mockClient,
            mockTransformers
        );
        
        MMROverSampleProcessor proc2 = new MMROverSampleProcessor(
            "tag2",
            false,
            mockClient,
            mockTransformers
        );
        
        assertEquals(proc1.getExecutionStage(), proc2.getExecutionStage());
        assertEquals(MMROverSampleProcessor.ExecutionStage.POST_USER_DEFINED, proc1.getExecutionStage());
    }

    // ============================================
    // Test: Edge cases
    // ============================================

    @Test
    public void testConstructor_WithEmptyTag() {
        MMROverSampleProcessor proc = new MMROverSampleProcessor(
            "",
            false,
            mockClient,
            mockTransformers
        );
        
        assertEquals("", proc.getTag());
        assertNotNull(proc.getType());
    }

    @Test
    public void testGetters_ReturnNonNullValues() {
        assertNotNull(processor.getType());
        assertNotNull(processor.getDescription());
        assertNotNull(processor.getExecutionStage());
        // getTag() can be null, so we don't assert it
    }

    @Test
    public void testProcessorBehavior_WithDifferentClients() {
        Client client1 = mock(Client.class);
        Client client2 = mock(Client.class);
        
        MMROverSampleProcessor proc1 = new MMROverSampleProcessor(
            "tag",
            false,
            client1,
            mockTransformers
        );
        
        MMROverSampleProcessor proc2 = new MMROverSampleProcessor(
            "tag",
            false,
            client2,
            mockTransformers
        );
        
        // Both should have same public behavior
        assertEquals(proc1.getType(), proc2.getType());
        assertEquals(proc1.getTag(), proc2.getTag());
        assertEquals(proc1.isIgnoreFailure(), proc2.isIgnoreFailure());
    }

    @Test
    public void testProcessorBehavior_WithDifferentTransformers() {
        Map<String, MMRQueryTransformer<? extends QueryBuilder>> transformers1 = new HashMap<>();
        Map<String, MMRQueryTransformer<? extends QueryBuilder>> transformers2 = new HashMap<>();
        transformers2.put("test", mock(MMRQueryTransformer.class));
        
        MMROverSampleProcessor proc1 = new MMROverSampleProcessor(
            "tag",
            false,
            mockClient,
            transformers1
        );
        
        MMROverSampleProcessor proc2 = new MMROverSampleProcessor(
            "tag",
            false,
            mockClient,
            transformers2
        );
        
        // Both should have same public behavior
        assertEquals(proc1.getType(), proc2.getType());
        assertEquals(proc1.getTag(), proc2.getTag());
    }

    // ============================================
    // Test: Error message formatting
    // ============================================

    @Test
    public void testErrorMessages_UseConsistentFormatting() {
        SearchRequest mockRequest = mock(SearchRequest.class);
        PipelineProcessingContext mockContext = mock(PipelineProcessingContext.class);
        
        String message1 = null;
        String message2 = null;
        
        try {
            processor.processRequest(mockRequest);
        } catch (UnsupportedOperationException e) {
            message1 = e.getMessage();
        }
        
        try {
            processor.processRequest(mockRequest, mockContext);
        } catch (UnsupportedOperationException e) {
            message2 = e.getMessage();
        }
        
        assertNotNull(message1);
        assertNotNull(message2);
        
        // Both messages should mention the processor type
        assertTrue(message1.contains("mmr_over_sample"));
        assertTrue(message2.contains("mmr_over_sample"));
        
        // Both should mention synchronous operation
        assertTrue(message1.contains("synchronously"));
        assertTrue(message2.contains("synchronously"));
    }
    
}
