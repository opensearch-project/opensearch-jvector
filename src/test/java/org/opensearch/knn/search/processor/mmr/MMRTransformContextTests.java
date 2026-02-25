/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.transport.client.Client;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.mockito.Mockito.mock;

/**
 * Unit tests for MMRTransformContext
 */
public class MMRTransformContextTests extends LuceneTestCase {

    // ========== Constructor Tests ==========

    @Test
    public void testConstructorWithAllRequiredFields() {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        MMRTransformContext context = new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );

        assertEquals(Integer.valueOf(10), context.getCandidates());
        assertNotNull(context.getMmrRerankContext());
        assertNotNull(context.getLocalIndexMetadataList());
        assertEquals(1, context.getLocalIndexMetadataList().size());
        assertNotNull(context.getRemoteIndices());
        assertTrue(context.getRemoteIndices().isEmpty());
        assertNull(context.getUserProvidedSpaceType());
        assertNull(context.getUserProvidedVectorFieldPath());
        assertNull(context.getUserProvidedVectorDataType());
        assertNotNull(context.getClient());
        assertFalse(context.isVectorFieldInfoResolved());
    }

    @Test
    public void testConstructorWithAllFields() {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Arrays.asList("remote:index1", "remote:index2");
        Client client = mock(Client.class);

        MMRTransformContext context = new MMRTransformContext(
            30,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            SpaceType.COSINESIMIL,
            "embedding.vector",
            VectorDataType.FLOAT,
            client,
            true
        );

        assertEquals(Integer.valueOf(30), context.getCandidates());
        assertNotNull(context.getMmrRerankContext());
        assertEquals(1, context.getLocalIndexMetadataList().size());
        assertEquals(2, context.getRemoteIndices().size());
        assertEquals(SpaceType.COSINESIMIL, context.getUserProvidedSpaceType());
        assertEquals("embedding.vector", context.getUserProvidedVectorFieldPath());
        assertEquals(VectorDataType.FLOAT, context.getUserProvidedVectorDataType());
        assertNotNull(context.getClient());
        assertTrue(context.isVectorFieldInfoResolved());
    }

    @Test(expected = NullPointerException.class)
    public void testConstructorWithNullCandidates() {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        new MMRTransformContext(
            null,  // Should throw NullPointerException
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );
    }

    @Test(expected = NullPointerException.class)
    public void testConstructorWithNullRerankContext() {
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        new MMRTransformContext(
            10,
            null,  // Should throw NullPointerException
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );
    }

    @Test(expected = NullPointerException.class)
    public void testConstructorWithNullIndexMetadataList() {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        new MMRTransformContext(
            10,
            rerankContext,
            null,  // Should throw NullPointerException
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );
    }

    @Test(expected = NullPointerException.class)
    public void testConstructorWithNullRemoteIndices() {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        Client client = mock(Client.class);

        new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            null,  // Should throw NullPointerException
            null,
            null,
            null,
            client,
            false
        );
    }

    // ========== Getter Tests ==========

    @Test
    public void testGetCandidates() {
        MMRTransformContext context = createBasicContext(50);
        assertEquals(Integer.valueOf(50), context.getCandidates());
    }

    @Test
    public void testGetMmrRerankContext() {
        MMRRerankContext rerankContext = new MMRRerankContext();
        rerankContext.setOriginalQuerySize(10);
        rerankContext.setDiversity(0.5f);

        MMRTransformContext context = createContextWithRerankContext(rerankContext);

        assertNotNull(context.getMmrRerankContext());
        assertEquals(Integer.valueOf(10), context.getMmrRerankContext().getOriginalQuerySize());
        assertEquals(Float.valueOf(0.5f), context.getMmrRerankContext().getDiversity());
    }

    @Test
    public void testGetLocalIndexMetadataList() {
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("index1", "index2", "index3");
        MMRTransformContext context = createContextWithIndexMetadata(indexMetadataList);

        assertEquals(3, context.getLocalIndexMetadataList().size());
    }

    @Test
    public void testGetRemoteIndices() {
        List<String> remoteIndices = Arrays.asList("cluster1:index1", "cluster2:index2");
        MMRTransformContext context = createContextWithRemoteIndices(remoteIndices);

        assertEquals(2, context.getRemoteIndices().size());
        assertEquals("cluster1:index1", context.getRemoteIndices().get(0));
        assertEquals("cluster2:index2", context.getRemoteIndices().get(1));
    }

    @Test
    public void testGetUserProvidedSpaceType() {
        MMRTransformContext context = createContextWithSpaceType(SpaceType.L2);
        assertEquals(SpaceType.L2, context.getUserProvidedSpaceType());

        context = createContextWithSpaceType(SpaceType.COSINESIMIL);
        assertEquals(SpaceType.COSINESIMIL, context.getUserProvidedSpaceType());

        context = createContextWithSpaceType(null);
        assertNull(context.getUserProvidedSpaceType());
    }

    @Test
    public void testGetUserProvidedVectorFieldPath() {
        MMRTransformContext context = createContextWithVectorFieldPath("embedding");
        assertEquals("embedding", context.getUserProvidedVectorFieldPath());

        context = createContextWithVectorFieldPath("user.profile.vector");
        assertEquals("user.profile.vector", context.getUserProvidedVectorFieldPath());

        context = createContextWithVectorFieldPath(null);
        assertNull(context.getUserProvidedVectorFieldPath());
    }

    @Test
    public void testGetUserProvidedVectorDataType() {
        MMRTransformContext context = createContextWithVectorDataType(VectorDataType.FLOAT);
        assertEquals(VectorDataType.FLOAT, context.getUserProvidedVectorDataType());

        context = createContextWithVectorDataType(VectorDataType.BYTE);
        assertEquals(VectorDataType.BYTE, context.getUserProvidedVectorDataType());

        context = createContextWithVectorDataType(null);
        assertNull(context.getUserProvidedVectorDataType());
    }

    @Test
    public void testGetClient() {
        Client mockClient = mock(Client.class);
        MMRTransformContext context = createContextWithClient(mockClient);
        assertNotNull(context.getClient());
        assertEquals(mockClient, context.getClient());
    }

    @Test
    public void testIsVectorFieldInfoResolved() {
        MMRTransformContext context = createContextWithResolvedFlag(false);
        assertFalse(context.isVectorFieldInfoResolved());

        context = createContextWithResolvedFlag(true);
        assertTrue(context.isVectorFieldInfoResolved());
    }

    // ========== Setter Tests ==========

    @Test
    public void testSetVectorFieldInfoResolved() {
        MMRTransformContext context = createBasicContext(10);

        assertFalse(context.isVectorFieldInfoResolved());

        context.setVectorFieldInfoResolved(true);
        assertTrue(context.isVectorFieldInfoResolved());

        context.setVectorFieldInfoResolved(false);
        assertFalse(context.isVectorFieldInfoResolved());
    }

    // ========== Integration Tests ==========

    @Test
    public void testCompleteContextWithAllOptionalFields() {
        MMRRerankContext rerankContext = new MMRRerankContext();
        rerankContext.setOriginalQuerySize(20);
        rerankContext.setDiversity(0.7f);
        rerankContext.setSpaceType(SpaceType.INNER_PRODUCT);

        List<IndexMetadata> indexMetadataList = createIndexMetadataList("products", "reviews");
        List<String> remoteIndices = Arrays.asList("remote:logs");
        Client client = mock(Client.class);

        MMRTransformContext context = new MMRTransformContext(
            60,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            SpaceType.INNER_PRODUCT,
            "product.embedding",
            VectorDataType.FLOAT,
            client,
            false
        );

        // Verify all fields
        assertEquals(Integer.valueOf(60), context.getCandidates());
        assertEquals(Integer.valueOf(20), context.getMmrRerankContext().getOriginalQuerySize());
        assertEquals(2, context.getLocalIndexMetadataList().size());
        assertEquals(1, context.getRemoteIndices().size());
        assertEquals(SpaceType.INNER_PRODUCT, context.getUserProvidedSpaceType());
        assertEquals("product.embedding", context.getUserProvidedVectorFieldPath());
        assertEquals(VectorDataType.FLOAT, context.getUserProvidedVectorDataType());
        assertNotNull(context.getClient());
        assertFalse(context.isVectorFieldInfoResolved());

        // Modify resolved flag
        context.setVectorFieldInfoResolved(true);
        assertTrue(context.isVectorFieldInfoResolved());
    }

    @Test
    public void testContextWithEmptyRemoteIndices() {
        MMRTransformContext context = createContextWithRemoteIndices(Collections.emptyList());

        assertNotNull(context.getRemoteIndices());
        assertTrue(context.getRemoteIndices().isEmpty());
        assertEquals(0, context.getRemoteIndices().size());
    }

    @Test
    public void testContextWithMultipleLocalIndices() {
        List<IndexMetadata> indexMetadataList = createIndexMetadataList(
            "index1", "index2", "index3", "index4", "index5"
        );
        MMRTransformContext context = createContextWithIndexMetadata(indexMetadataList);

        assertEquals(5, context.getLocalIndexMetadataList().size());
    }

    @Test
    public void testContextWithAllSpaceTypes() {
        SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT };

        for (SpaceType spaceType : spaceTypes) {
            MMRTransformContext context = createContextWithSpaceType(spaceType);
            assertEquals(spaceType, context.getUserProvidedSpaceType());
        }
    }

    @Test
    public void testContextWithAllVectorDataTypes() {
        VectorDataType[] dataTypes = { VectorDataType.FLOAT, VectorDataType.BYTE, VectorDataType.BINARY };

        for (VectorDataType dataType : dataTypes) {
            MMRTransformContext context = createContextWithVectorDataType(dataType);
            assertEquals(dataType, context.getUserProvidedVectorDataType());
        }
    }

    @Test
    public void testCandidatesVariations() {
        int[] candidateValues = { 1, 10, 50, 100, 500, 1000 };

        for (int candidates : candidateValues) {
            MMRTransformContext context = createBasicContext(candidates);
            assertEquals(Integer.valueOf(candidates), context.getCandidates());
        }
    }

    // ========== Helper Methods ==========

    private MMRTransformContext createBasicContext(int candidates) {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        return new MMRTransformContext(
            candidates,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );
    }

    private MMRTransformContext createContextWithRerankContext(MMRRerankContext rerankContext) {
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        return new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );
    }

    private MMRTransformContext createContextWithIndexMetadata(List<IndexMetadata> indexMetadataList) {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        return new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );
    }

    private MMRTransformContext createContextWithRemoteIndices(List<String> remoteIndices) {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        Client client = mock(Client.class);

        return new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );
    }

    private MMRTransformContext createContextWithSpaceType(SpaceType spaceType) {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        return new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            spaceType,
            null,
            null,
            client,
            false
        );
    }

    private MMRTransformContext createContextWithVectorFieldPath(String vectorFieldPath) {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        return new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            vectorFieldPath,
            null,
            client,
            false
        );
    }

    private MMRTransformContext createContextWithVectorDataType(VectorDataType vectorDataType) {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        return new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            vectorDataType,
            client,
            false
        );
    }

    private MMRTransformContext createContextWithClient(Client client) {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();

        return new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            false
        );
    }

    private MMRTransformContext createContextWithResolvedFlag(boolean isResolved) {
        MMRRerankContext rerankContext = new MMRRerankContext();
        List<IndexMetadata> indexMetadataList = createIndexMetadataList("test-index");
        List<String> remoteIndices = Collections.emptyList();
        Client client = mock(Client.class);

        return new MMRTransformContext(
            10,
            rerankContext,
            indexMetadataList,
            remoteIndices,
            null,
            null,
            null,
            client,
            isResolved
        );
    }

    private List<IndexMetadata> createIndexMetadataList(String... indexNames) {
        List<IndexMetadata> list = new ArrayList<>();
        for (String indexName : indexNames) {
            IndexMetadata metadata = IndexMetadata.builder(indexName)
                .settings(Settings.builder().put(IndexMetadata.SETTING_VERSION_CREATED, org.opensearch.Version.CURRENT))
                .numberOfShards(1)
                .numberOfReplicas(0)
                .build();
            list.add(metadata);
        }
        return list;
    }
}
