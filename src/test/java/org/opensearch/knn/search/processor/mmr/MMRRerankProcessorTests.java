/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.search.pipeline.PipelineProcessingContext;

import static org.mockito.Mockito.*;

/**
 * Unit tests for {@link MMRRerankProcessor}
 */
public class MMRRerankProcessorTests extends LuceneTestCase {

    private MMRRerankProcessor processor;
    private String testTag = "test-tag";
    private boolean testIgnoreFailure = false;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        processor = new MMRRerankProcessor(testTag, testIgnoreFailure);
    }

    // ============================================
    // Test: Public getter methods
    // ============================================

    @Test
    public void testGetType_ReturnsCorrectType() {
        assertEquals("mmr_rerank", processor.getType());
        assertEquals(MMRRerankProcessor.TYPE, processor.getType());
    }

    @Test
    public void testGetTag_ReturnsConfiguredTag() {
        assertEquals(testTag, processor.getTag());
        
        // Test with different tag
        MMRRerankProcessor customProcessor = new MMRRerankProcessor("custom-tag", false);
        assertEquals("custom-tag", customProcessor.getTag());
    }

    @Test
    public void testGetDescription_ReturnsNonEmptyDescription() {
        String description = processor.getDescription();
        assertNotNull(description);
        assertFalse(description.isEmpty());
        assertTrue(description.contains("rerank"));
        assertTrue(description.contains("Maximal Marginal Relevance"));
        assertEquals(MMRRerankProcessor.DESCRIPTION, description);
    }

    @Test
    public void testIsIgnoreFailure_ReturnsConfiguredValue() {
        assertEquals(testIgnoreFailure, processor.isIgnoreFailure());
        
        // Test with ignore failure enabled
        MMRRerankProcessor processorWithIgnoreFailure = new MMRRerankProcessor("tag", true);
        assertTrue(processorWithIgnoreFailure.isIgnoreFailure());
    }

    @Test
    public void testGetExecutionStage_ReturnsPreUserDefined() {
        assertEquals(
            MMRRerankProcessor.ExecutionStage.PRE_USER_DEFINED,
            processor.getExecutionStage()
        );
    }

    // ============================================
    // Test: processResponse without context throws exception
    // ============================================

    @Test
    public void testProcessResponse_WithoutContext_ThrowsUnsupportedOperationException() {
        SearchRequest mockRequest = mock(SearchRequest.class);
        SearchResponse mockResponse = mock(SearchResponse.class);
        
        try {
            processor.processResponse(mockRequest, mockResponse);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("mmr_rerank"));
            assertTrue(e.getMessage().contains("PipelineProcessingContext"));
        }
    }

    @Test
    public void testProcessResponse_WithoutContext_ErrorMessageContainsProcessorType() {
        SearchRequest mockRequest = mock(SearchRequest.class);
        SearchResponse mockResponse = mock(SearchResponse.class);
        
        try {
            processor.processResponse(mockRequest, mockResponse);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            String message = e.getMessage();
            assertTrue("Error message should contain processor type", 
                      message.contains("mmr_rerank"));
            assertTrue("Error message should mention PipelineProcessingContext", 
                      message.contains("PipelineProcessingContext"));
        }
    }

    // ============================================
    // Test: Constructor variations
    // ============================================

    @Test
    public void testConstructor_WithAllParameters() {
        String tag = "custom-tag";
        boolean ignoreFailure = true;
        
        MMRRerankProcessor proc = new MMRRerankProcessor(tag, ignoreFailure);
        
        assertEquals(tag, proc.getTag());
        assertEquals(ignoreFailure, proc.isIgnoreFailure());
        assertEquals("mmr_rerank", proc.getType());
        assertNotNull(proc.getDescription());
    }

    @Test
    public void testConstructor_WithNullTag() {
        MMRRerankProcessor proc = new MMRRerankProcessor(null, false);
        
        assertNull(proc.getTag());
        assertEquals("mmr_rerank", proc.getType());
    }

    @Test
    public void testConstructor_WithEmptyTag() {
        MMRRerankProcessor proc = new MMRRerankProcessor("", false);
        
        assertEquals("", proc.getTag());
        assertNotNull(proc.getType());
    }

    @Test
    public void testConstructor_WithDifferentIgnoreFailureValues() {
        MMRRerankProcessor procTrue = new MMRRerankProcessor("tag1", true);
        MMRRerankProcessor procFalse = new MMRRerankProcessor("tag2", false);
        
        assertTrue(procTrue.isIgnoreFailure());
        assertFalse(procFalse.isIgnoreFailure());
    }

    // ============================================
    // Test: Multiple processor instances are independent
    // ============================================

    @Test
    public void testMultipleInstances_AreIndependent() {
        MMRRerankProcessor proc1 = new MMRRerankProcessor("tag1", true);
        MMRRerankProcessor proc2 = new MMRRerankProcessor("tag2", false);
        
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
        assertEquals("mmr_rerank_factory", MMRRerankProcessor.MMRRerankProcessorFactory.TYPE);
    }

    @Test
    public void testFactory_Constructor() {
        MMRRerankProcessor.MMRRerankProcessorFactory factory = 
            new MMRRerankProcessor.MMRRerankProcessorFactory();
        
        assertNotNull(factory);
    }

    // ============================================
    // Test: Constants and default values
    // ============================================

    @Test
    public void testConstants_HaveExpectedValues() {
        assertEquals("mmr_rerank", MMRRerankProcessor.TYPE);
        assertNotNull(MMRRerankProcessor.DESCRIPTION);
        assertFalse(MMRRerankProcessor.DESCRIPTION.isEmpty());
        assertTrue(MMRRerankProcessor.DESCRIPTION.length() > 20);
    }

    @Test
    public void testTypeConstant_MatchesGetType() {
        assertEquals(MMRRerankProcessor.TYPE, processor.getType());
    }

    @Test
    public void testDescriptionConstant_MatchesGetDescription() {
        assertEquals(MMRRerankProcessor.DESCRIPTION, processor.getDescription());
    }

    // ============================================
    // Test: Execution stage behavior
    // ============================================

    @Test
    public void testExecutionStage_IsConsistentAcrossInstances() {
        MMRRerankProcessor proc1 = new MMRRerankProcessor("tag1", true);
        MMRRerankProcessor proc2 = new MMRRerankProcessor("tag2", false);
        
        assertEquals(proc1.getExecutionStage(), proc2.getExecutionStage());
        assertEquals(MMRRerankProcessor.ExecutionStage.PRE_USER_DEFINED, proc1.getExecutionStage());
    }

    @Test
    public void testExecutionStage_IsPreUserDefined() {
        // MMR reranking should happen BEFORE user-defined response processors
        // This ensures the response is reranked and reduced to original size first
        assertEquals(
            MMRRerankProcessor.ExecutionStage.PRE_USER_DEFINED,
            processor.getExecutionStage()
        );
    }

    // ============================================
    // Test: Edge cases
    // ============================================

    @Test
    public void testGetters_ReturnNonNullValues() {
        assertNotNull(processor.getType());
        assertNotNull(processor.getDescription());
        assertNotNull(processor.getExecutionStage());
        // getTag() can be null, so we don't assert it
    }

    @Test
    public void testProcessorBehavior_WithDifferentTags() {
        String[] tags = {"tag1", "tag2", "tag3", null, ""};
        
        for (String tag : tags) {
            MMRRerankProcessor proc = new MMRRerankProcessor(tag, false);
            assertEquals(tag, proc.getTag());
            assertEquals("mmr_rerank", proc.getType());
        }
    }

    @Test
    public void testProcessorBehavior_WithDifferentIgnoreFailureFlags() {
        boolean[] flags = {true, false};
        
        for (boolean flag : flags) {
            MMRRerankProcessor proc = new MMRRerankProcessor("tag", flag);
            assertEquals(flag, proc.isIgnoreFailure());
            assertEquals("mmr_rerank", proc.getType());
        }
    }

    // ============================================
    // Test: Error message formatting
    // ============================================

    @Test
    public void testErrorMessage_UseConsistentFormatting() {
        SearchRequest mockRequest = mock(SearchRequest.class);
        SearchResponse mockResponse = mock(SearchResponse.class);
        
        String message = null;
        
        try {
            processor.processResponse(mockRequest, mockResponse);
        } catch (UnsupportedOperationException e) {
            message = e.getMessage();
        }
        
        assertNotNull(message);
        assertTrue(message.contains("mmr_rerank"));
        assertTrue(message.contains("PipelineProcessingContext"));
    }

}
