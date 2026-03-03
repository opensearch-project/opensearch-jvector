/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.search.TotalHits;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.search.SearchResponseSections;
import org.opensearch.action.search.ShardSearchFailure;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.search.SearchHit;
import org.opensearch.search.SearchHits;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.pipeline.PipelineProcessingContext;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import static org.mockito.Mockito.mock;
import static org.opensearch.knn.common.KNNConstants.MMR_RERANK_CONTEXT;

public class MMRRerankProcessorTests extends KNNTestCase {
    private MMRRerankProcessor processor;
    private SearchRequest searchRequest;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        processor = new MMRRerankProcessor("test-tag", false);
        searchRequest = new SearchRequest();
    }

    // ============================================
    // Test: Public getter methods
    // ============================================

    public void testGetType_ReturnsCorrectType() {
        assertEquals("mmr_rerank", processor.getType());
        assertEquals(MMRRerankProcessor.TYPE, processor.getType());
    }

    public void testGetTag_ReturnsConfiguredTag() {
        assertEquals("test-tag", processor.getTag());

        MMRRerankProcessor customProcessor = new MMRRerankProcessor("custom-tag", false);
        assertEquals("custom-tag", customProcessor.getTag());
    }

    public void testGetDescription_ReturnsNonEmptyDescription() {
        String description = processor.getDescription();
        assertNotNull(description);
        assertFalse(description.isEmpty());
        assertTrue(description.contains("rerank"));
        assertTrue(description.contains("Maximal Marginal Relevance"));
        assertEquals(MMRRerankProcessor.DESCRIPTION, description);
    }

    public void testIsIgnoreFailure_ReturnsConfiguredValue() {
        assertFalse(processor.isIgnoreFailure());

        MMRRerankProcessor processorWithIgnoreFailure = new MMRRerankProcessor("tag", true);
        assertTrue(processorWithIgnoreFailure.isIgnoreFailure());
    }

    public void testGetExecutionStage_ReturnsPreUserDefined() {
        assertEquals(MMRRerankProcessor.ExecutionStage.PRE_USER_DEFINED, processor.getExecutionStage());
    }

    // ============================================
    // Test: processResponse without context
    // ============================================

    public void testProcessResponse_WithoutContext_ThrowsUnsupportedOperationException() {
        UnsupportedOperationException ex = assertThrows(
            UnsupportedOperationException.class,
            () -> processor.processResponse(searchRequest, mock(SearchResponse.class))
        );

        assertTrue(ex.getMessage().contains("mmr_rerank"));
        assertTrue(ex.getMessage().contains("PipelineProcessingContext"));
    }

    // ============================================
    // Test: Integration tests with actual reranking
    // ============================================

    public void testProcessResponse_whenEmptyHits_thenReturnOriginalResponse() throws IOException {
        SearchResponse emptyResponse = createSearchResponse(new SearchHit[] {});

        PipelineProcessingContext ctx = mock(PipelineProcessingContext.class);

        SearchResponse result = processor.processResponse(searchRequest, emptyResponse, ctx);

        assertEquals("Processor should return the same response when there are no hits.", emptyResponse, result);
    }

    public void testProcessResponse_whenHappyCaseFloatWithL2_thenRerank() throws IOException {
        runProcessResponseRerankHappyCase(SpaceType.L2, VectorDataType.FLOAT);
    }

    public void testProcessResponse_whenHappyCaseBinaryWithHammingSpaceType_thenRerank() throws IOException {
        runProcessResponseRerankHappyCase(SpaceType.HAMMING, VectorDataType.BINARY);
    }

    private void runProcessResponseRerankHappyCase(SpaceType spaceType, VectorDataType vectorDataType) throws IOException {
        SearchResponse searchResponse = createSearchResponse();
        assertEquals(10, searchResponse.getInternalResponse().hits().getHits().length);
        assertEquals(0, searchResponse.getInternalResponse().hits().getHits()[0].docId());
        assertNotNull(
            "Should have the knn_vector in the source.",
            searchResponse.getInternalResponse().hits().getHits()[0].getSourceAsMap().get("knn_vector")
        );
        assertEquals(1, searchResponse.getInternalResponse().hits().getHits()[1].docId());
        assertEquals(2, searchResponse.getInternalResponse().hits().getHits()[2].docId());

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setDiversity(0.5f);
        mmrRerankContext.setOriginalQuerySize(3);
        mmrRerankContext.setSpaceType(spaceType);
        mmrRerankContext.setVectorDataType(vectorDataType);
        mmrRerankContext.setVectorFieldPath("knn_vector");
        mmrRerankContext.setOriginalFetchSourceContext(new FetchSourceContext(true, new String[] {}, new String[] { "knn_vector" }));
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        SearchResponse result = processor.processResponse(searchRequest, searchResponse, ctx);

        assertEquals("Should reduce the hits to the original query size.", 3, result.getInternalResponse().hits().getHits().length);
        assertEquals(0, result.getInternalResponse().hits().getHits()[0].docId());
        assertNull(
            "Should exclude the knn_vector from the source.",
            result.getInternalResponse().hits().getHits()[0].getSourceAsMap().get("knn_vector")
        );
        assertEquals("Should pick the hit with diversity.", 8, result.getInternalResponse().hits().getHits()[1].docId());
        assertEquals("Should pick the hit with diversity.", 9, result.getInternalResponse().hits().getHits()[2].docId());
    }

    public void testProcessResponse_whenMissingRerankContext_thenException() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        PipelineProcessingContext ctx = new PipelineProcessingContext();

        IllegalStateException exception = assertThrows(
            IllegalStateException.class,
            () -> processor.processResponse(searchRequest, searchResponse, ctx)
        );
        String expectedMessage = "MMR rerank context cannot be null";
        assertEquals(expectedMessage, exception.getMessage());
    }

    public void testProcessResponse_whenMissingSpaceType_thenException() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        IllegalStateException exception = assertThrows(
            IllegalStateException.class,
            () -> processor.processResponse(searchRequest, searchResponse, ctx)
        );
        String expectedMessage = "Space type in MMR rerank context cannot be null";
        assertEquals(expectedMessage, exception.getMessage());
    }

    public void testProcessResponse_whenMissingOriginalQuerySize_thenException() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setSpaceType(SpaceType.L2);
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        IllegalStateException exception = assertThrows(
            IllegalStateException.class,
            () -> processor.processResponse(searchRequest, searchResponse, ctx)
        );
        String expectedMessage = "Original query size in MMR rerank context cannot be null";
        assertEquals(expectedMessage, exception.getMessage());
    }

    public void testProcessResponse_whenMissingDiversity_thenException() throws IOException {
        SearchResponse searchResponse = createSearchResponse();

        MMRRerankContext mmrRerankContext = new MMRRerankContext();
        mmrRerankContext.setSpaceType(SpaceType.L2);
        mmrRerankContext.setOriginalQuerySize(3);
        PipelineProcessingContext ctx = new PipelineProcessingContext();
        ctx.setAttribute(MMR_RERANK_CONTEXT, mmrRerankContext);

        IllegalStateException exception = assertThrows(
            IllegalStateException.class,
            () -> processor.processResponse(searchRequest, searchResponse, ctx)
        );
        String expectedMessage = "Diversity in MMR rerank context cannot be null";
        assertEquals(expectedMessage, exception.getMessage());
    }

    // ============================================
    // Test: Constructor variations
    // ============================================

    public void testConstructor_WithAllParameters() {
        String tag = "custom-tag";
        boolean ignoreFailure = true;

        MMRRerankProcessor proc = new MMRRerankProcessor(tag, ignoreFailure);

        assertEquals(tag, proc.getTag());
        assertEquals(ignoreFailure, proc.isIgnoreFailure());
        assertEquals("mmr_rerank", proc.getType());
        assertNotNull(proc.getDescription());
    }

    public void testConstructor_WithNullTag() {
        MMRRerankProcessor proc = new MMRRerankProcessor(null, false);

        assertNull(proc.getTag());
        assertEquals("mmr_rerank", proc.getType());
    }

    public void testConstructor_WithEmptyTag() {
        MMRRerankProcessor proc = new MMRRerankProcessor("", false);

        assertEquals("", proc.getTag());
        assertNotNull(proc.getType());
    }

    public void testConstructor_WithDifferentIgnoreFailureValues() {
        MMRRerankProcessor procTrue = new MMRRerankProcessor("tag1", true);
        MMRRerankProcessor procFalse = new MMRRerankProcessor("tag2", false);

        assertTrue(procTrue.isIgnoreFailure());
        assertFalse(procFalse.isIgnoreFailure());
    }

    // ============================================
    // Test: Multiple processor instances
    // ============================================

    public void testMultipleInstances_AreIndependent() {
        MMRRerankProcessor proc1 = new MMRRerankProcessor("tag1", true);
        MMRRerankProcessor proc2 = new MMRRerankProcessor("tag2", false);

        assertEquals("tag1", proc1.getTag());
        assertEquals("tag2", proc2.getTag());
        assertTrue(proc1.isIgnoreFailure());
        assertFalse(proc2.isIgnoreFailure());

        assertEquals(proc1.getType(), proc2.getType());
        assertEquals(proc1.getExecutionStage(), proc2.getExecutionStage());
        assertEquals(proc1.getDescription(), proc2.getDescription());
    }

    // ============================================
    // Test: Factory class
    // ============================================

    public void testFactory_TypeConstant() {
        assertEquals("mmr_rerank_factory", MMRRerankProcessor.MMRRerankProcessorFactory.TYPE);
    }

    public void testFactory_Constructor() {
        MMRRerankProcessor.MMRRerankProcessorFactory factory = new MMRRerankProcessor.MMRRerankProcessorFactory();

        assertNotNull(factory);
    }

    // ============================================
    // Test: Constants and default values
    // ============================================

    public void testConstants_HaveExpectedValues() {
        assertEquals("mmr_rerank", MMRRerankProcessor.TYPE);
        assertNotNull(MMRRerankProcessor.DESCRIPTION);
        assertFalse(MMRRerankProcessor.DESCRIPTION.isEmpty());
        assertTrue(MMRRerankProcessor.DESCRIPTION.length() > 20);
    }

    public void testTypeConstant_MatchesGetType() {
        assertEquals(MMRRerankProcessor.TYPE, processor.getType());
    }

    public void testDescriptionConstant_MatchesGetDescription() {
        assertEquals(MMRRerankProcessor.DESCRIPTION, processor.getDescription());
    }

    // ============================================
    // Test: Execution stage behavior
    // ============================================

    public void testExecutionStage_IsConsistentAcrossInstances() {
        MMRRerankProcessor proc1 = new MMRRerankProcessor("tag1", true);
        MMRRerankProcessor proc2 = new MMRRerankProcessor("tag2", false);

        assertEquals(proc1.getExecutionStage(), proc2.getExecutionStage());
        assertEquals(MMRRerankProcessor.ExecutionStage.PRE_USER_DEFINED, proc1.getExecutionStage());
    }

    public void testExecutionStage_IsPreUserDefined() {
        assertEquals(MMRRerankProcessor.ExecutionStage.PRE_USER_DEFINED, processor.getExecutionStage());
    }

    // ============================================
    // Test: Edge cases
    // ============================================

    public void testGetters_ReturnNonNullValues() {
        assertNotNull(processor.getType());
        assertNotNull(processor.getDescription());
        assertNotNull(processor.getExecutionStage());
    }

    public void testProcessorBehavior_WithDifferentTags() {
        String[] tags = { "tag1", "tag2", "tag3", null, "" };

        for (String tag : tags) {
            MMRRerankProcessor proc = new MMRRerankProcessor(tag, false);
            assertEquals(tag, proc.getTag());
            assertEquals("mmr_rerank", proc.getType());
        }
    }

    public void testProcessorBehavior_WithDifferentIgnoreFailureFlags() {
        boolean[] flags = { true, false };

        for (boolean flag : flags) {
            MMRRerankProcessor proc = new MMRRerankProcessor("tag", flag);
            assertEquals(flag, proc.isIgnoreFailure());
            assertEquals("mmr_rerank", proc.getType());
        }
    }

    // ============================================
    // Test: Helper methods for creating test data
    // ============================================

    private SearchResponse createSearchResponse() throws IOException {
        SearchHit[] hits = new SearchHit[10];

        // 8 similar hits, high score
        float[] similarVector = new float[] { 1f, 1f };
        for (int i = 0; i < 8; i++) {
            XContentBuilder sourceBuilder = JsonXContent.contentBuilder().startObject().array("knn_vector", similarVector).endObject();

            SearchHit hit = new SearchHit(i, String.valueOf(i), Map.of(), Map.of());
            hit.sourceRef(BytesReference.bytes(sourceBuilder));
            hit.score(1f);
            hits[i] = hit;
        }

        // 2 diverse hits, slightly lower score
        float[][] diverseVectors = new float[][] { { 1f, 2f }, { 2f, 1f } };
        for (int i = 0; i < 2; i++) {
            int idx = i + 8;
            XContentBuilder sourceBuilder = JsonXContent.contentBuilder().startObject().array("knn_vector", diverseVectors[i]).endObject();

            SearchHit hit = new SearchHit(idx, String.valueOf(idx), Map.of(), Map.of());
            hit.sourceRef(BytesReference.bytes(sourceBuilder));
            hit.score(0.8f); // slightly lower than top similar hits
            hits[idx] = hit;
        }
        return createSearchResponse(hits);
    }

    private SearchResponse createSearchResponse(SearchHit... hits) {
        TotalHits totalHits = new TotalHits(hits.length, TotalHits.Relation.EQUAL_TO);

        float maxScore = Arrays.stream(hits).map(SearchHit::getScore).max(Float::compare).orElse(Float.NEGATIVE_INFINITY);

        SearchHits searchHits = new SearchHits(hits, totalHits, maxScore);

        SearchResponseSections sections = new SearchResponseSections(
            searchHits,
            null,   // aggregations
            null,   // suggest
            false,  // timedOut
            false,  // terminatedEarly
            null,   // profileShardResults
            0       // numReducePhases
        );

        return new SearchResponse(
            sections,
            null,   // scrollId
            1,      // totalShards
            1,      // successfulShards
            0,      // skippedShards
            1,      // tookInMillis
            new ShardSearchFailure[0],
            new SearchResponse.Clusters(1, 1, 0),
            null    // pitId
        );
    }
}
